"""The PPO objective, the detached baseline and the update, exactly as specified.

docs/rl/pipeline-plan.md section 6. For game `g` and learner decision `t`, with
the learner's terminal return `R_g` in `[-1, 1]`:

    A_gt = R_g - V_old(x_gt),  r_gt = exp(logp_new - logp_old)

and the actor **maximizes the game-average sum**

    (1 / G) * sum_g sum_t min(r_gt * A_gt, clip(r_gt, 1 - eps, 1 + eps) * A_gt).

With uniformly shuffled decision minibatches, a minibatch's mean surrogate is
multiplied by `T / G` -- the complete rollout's decision and game counts -- which
preserves the full-batch objective's scale in expectation. Nothing divides a
game's contribution by its length, nothing normalizes by a minibatch's own game
count, and this is not a flat decision-mean loss in disguise. `weighted_gradient`
and `tests/rl/test_ppo.py` pin that equality exactly, including unequal game
lengths and a final partial minibatch.

The critic is **frozen before the batch is fitted**: advantages are computed once
from `V_old`, detached, and reused for every PPO epoch. They are not recomputed
with the newly fitted critic, and the critic's own loss is a separate
decision-mean MSE against the same undiscounted terminal returns -- no TD, no
GAE, no bootstrapping (contract section 3).

The KL number reported here is the sampled, nonnegative estimate
`mean((r - 1) - log r)` over the collection policy's own actions. It is a
diagnostic, not an exact KL guarantee, and its threshold stops **actor** epochs
only: critic fitting always completes.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from uno.rl.config import Optim
from uno.rl.features import Features
from uno.rl.model import Actor, Critic, entropy_from_log_probs, masked_log_probs, to_tensors
from uno.rl.rollout import Rollout
from uno.rl.streams import MINIBATCH, generator

FORWARD_CHUNK = 1024  # rows per no-grad forward pass; a memory setting, not an algorithmic one


def _synchronize_device(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))


@dataclass
class UpdateStats:
    """One update's diagnostics. `None` means *unavailable*, never a stand-in value."""

    actor_epochs: int = 0
    critic_epochs: int = 0
    actor_steps: int = 0
    critic_steps: int = 0
    surrogate: float = 0.0
    critic_mse: float = 0.0
    entropy: float = 0.0
    legal_actions: float = 0.0
    max_entropy: float = 0.0
    normalized_entropy: float = 0.0
    effective_actions: float = 0.0
    approx_kl: float = 0.0
    clip_fraction: float = 0.0
    max_abs_log_ratio: float = 0.0
    actor_grad_norm: float = 0.0
    actor_clipped_fraction: float = 0.0
    critic_grad_norm: float = 0.0
    critic_clipped_fraction: float = 0.0
    critic_pre_update_ev: float | None = None
    critic_post_update_same_rollout_ev: float | None = None
    advantage_mean: float = 0.0
    advantage_std: float = 0.0
    return_mean: float = 0.0
    return_std: float = 0.0
    early_stopped: bool = False
    kl_per_epoch: list[float] = field(default_factory=list)
    critic_value_seconds: float = 0.0
    actor_update_seconds: float = 0.0
    critic_update_seconds: float = 0.0

    def __str__(self) -> str:
        pre_ev = "n/a" if self.critic_pre_update_ev is None else f"{self.critic_pre_update_ev:.3f}"
        post_ev = "n/a" if self.critic_post_update_same_rollout_ev is None else f"{self.critic_post_update_same_rollout_ev:.3f}"
        return (
            f"surrogate {self.surrogate:+.5f} critic mse {self.critic_mse:.5f} (ev {pre_ev}->{post_ev}) entropy {self.entropy:.3f} "
            f"/ max {self.max_entropy:.3f} (norm {self.normalized_entropy:.2f}, eff {self.effective_actions:.2f}) "
            f"kl {self.approx_kl:.4f} clip {self.clip_fraction:.3f} |grad| {self.actor_grad_norm:.3f} "
            f"epochs {self.actor_epochs}/{self.critic_epochs}" + (" [early stop]" if self.early_stopped else "")
        )


# -- the frozen baseline --------------------------------------------------------


@torch.no_grad()
def critic_values(critic: Critic, features: Features, device: str = "cpu", chunk: int = FORWARD_CHUNK) -> np.ndarray:
    critic.eval()
    out = []
    for i in range(0, len(features), chunk):
        out.append(critic(*to_tensors(features[i : i + chunk], device)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def advantages(rollout: Rollout, values: np.ndarray) -> np.ndarray:
    """`A_gt = R_g - V_old(x_gt)`, detached by construction (a plain array)."""
    return (rollout.returns_per_decision - values).astype(np.float64)


# -- the objective ---------------------------------------------------------------


def surrogate(log_ratio: torch.Tensor, adv: torch.Tensor, clip: float, scale: float) -> torch.Tensor:
    """`scale * mean(min(r A, clip(r) A))`; `scale` is `T / G` for a minibatch."""
    r = torch.exp(log_ratio)
    return scale * torch.minimum(r * adv, torch.clamp(r, 1 - clip, 1 + clip) * adv).mean()


def new_log_probs(actor, features: Features, masks: np.ndarray, actions: np.ndarray, temperature: float, device: str, set_ids: np.ndarray | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Recomputed log probabilities of the taken actions, plus per-row entropy.

    The *same* masking and temperature convention as collection and evaluation
    (`uno.rl.model.masked_log_probs`), which is what makes the frozen-weight
    agreement check in stage 2 meaningful.
    """
    # A tabular actor (`uno.rl.tabular`) is parameterized by information set, not by
    # features; it declares that with `uses_set_ids` and everything else is shared.
    if getattr(actor, "uses_set_ids", False):
        logits = actor.logits(set_ids)
    else:
        logits = actor(*to_tensors(features, device))
    lp = masked_log_probs(logits, torch.from_numpy(masks).to(device), temperature)
    idx = torch.from_numpy(actions.astype(np.int64)).to(device)
    return lp.gather(1, idx.unsqueeze(1)).squeeze(1), entropy_from_log_probs(lp)


def actor_loss(
    actor: Actor, rollout: Rollout, index: np.ndarray, adv: np.ndarray, scale: float, optim: Optim, device: str = "cpu"
) -> tuple[torch.Tensor, dict]:
    """The negated surrogate for one minibatch, with its diagnostics."""
    feats = rollout.actor[index]
    set_ids = None if rollout.set_id is None else rollout.set_id[index]
    logp, entropy = new_log_probs(actor, feats, rollout.masks[index], rollout.actions[index], optim.temperature, device, set_ids)
    old = torch.from_numpy(rollout.old_logp[index].astype(np.float32)).to(device)
    a = torch.from_numpy(adv[index].astype(np.float32)).to(device)
    if optim.normalize_advantages:
        a = (a - a.mean()) / (a.std(unbiased=False) + 1e-8)
    log_ratio = logp - old
    obj = surrogate(log_ratio, a, optim.clip, scale)
    loss = -obj
    if optim.entropy_coef:
        loss = loss - optim.entropy_coef * entropy.mean()
    with torch.no_grad():
        r = torch.exp(log_ratio)
        diag = {
            "surrogate": float(obj),
            "entropy": float(entropy.mean()),
            "clip_fraction": float(((r - 1.0).abs() > optim.clip).float().mean()),
            "max_abs_log_ratio": float(log_ratio.abs().max()),
        }
    return loss, diag


def weighted_gradient(actor: Actor, rollout: Rollout, adv: np.ndarray, batches, optim: Optim, device: str = "cpu") -> list[torch.Tensor]:
    """`sum_k (|mb_k| / T) * grad_k`, which equals the full-batch gradient exactly.

    Exposed because that identity is the acceptance check for the game-summed
    objective, not because the training loop uses it: training takes one optimizer
    step per minibatch.
    """
    t, g = rollout.n_decisions, rollout.n_games
    total = [torch.zeros_like(p) for p in actor.parameters()]
    for index in batches:
        loss, _ = actor_loss(actor, rollout, index, adv, t / g, optim, device)
        actor.zero_grad(set_to_none=True)
        loss.backward()
        for acc, p in zip(total, actor.parameters()):
            if p.grad is not None:
                acc += p.grad * (len(index) / t)
    actor.zero_grad(set_to_none=True)
    return total


# -- the update -------------------------------------------------------------------


@torch.no_grad()
def rollout_actor_diagnostics(
    actor: Actor,
    rollout: Rollout,
    optim: Optim,
    adv: np.ndarray | None = None,
    device: str = "cpu",
    chunk: int = FORWARD_CHUNK,
) -> dict[str, float]:
    """Final-policy diagnostics over every decision in the rollout.

    These values describe one coherent policy, unlike an average over optimizer
    steps whose minibatches were evaluated under changing parameters. The
    surrogate uses the frozen rollout advantages and the same game-average-sum
    scale as the objective. With advantage normalization enabled, its diagnostic
    normalization is rollout-wide; the shipped configuration leaves normalization
    disabled.
    """
    if adv is not None and len(adv) != rollout.n_decisions:
        raise ValueError("rollout diagnostics need one advantage per decision")
    diagnostic_adv = None
    if adv is not None:
        diagnostic_adv = adv.astype(np.float32, copy=False)
        if optim.normalize_advantages:
            diagnostic_adv = (
                diagnostic_adv - diagnostic_adv.mean()
            ) / (diagnostic_adv.std() + 1e-8)
    surrogate_total = entropy_total = kl_total = clip_total = 0.0
    legal_actions_total = max_entropy_total = normalized_entropy_total = effective_actions_total = 0.0
    max_abs_log_ratio = 0.0
    n = 0
    for i in range(0, rollout.n_decisions, chunk):
        sl = slice(i, i + chunk)
        logp, entropy = new_log_probs(actor, rollout.actor[sl], rollout.masks[sl], rollout.actions[sl], optim.temperature, device, None if rollout.set_id is None else rollout.set_id[sl])
        log_ratio = logp - torch.from_numpy(rollout.old_logp[sl].astype(np.float32)).to(device)
        r = torch.exp(log_ratio)
        kl_total += float(((r - 1.0) - log_ratio).sum())
        entropy_total += float(entropy.sum())
        legal_count = rollout.masks[sl].sum(axis=1).astype(np.float64)
        max_entropy = np.log(np.maximum(legal_count, 1.0))
        entropy_np = entropy.detach().cpu().numpy()
        legal_actions_total += float(legal_count.sum())
        max_entropy_total += float(max_entropy.sum())
        normalized_entropy_total += float(np.divide(entropy_np, max_entropy, out=np.zeros_like(entropy_np), where=max_entropy > 0).sum())
        effective_actions_total += float(np.exp(entropy_np).sum())
        clip_total += float(((r - 1.0).abs() > optim.clip).sum())
        max_abs_log_ratio = max(max_abs_log_ratio, float(log_ratio.abs().max()))
        if diagnostic_adv is not None:
            a = torch.from_numpy(diagnostic_adv[sl]).to(device)
            surrogate_total += float(
                torch.minimum(
                    r * a,
                    torch.clamp(r, 1 - optim.clip, 1 + optim.clip) * a,
                ).sum()
            )
        n += len(logp)
    scale = rollout.n_decisions / rollout.n_games if rollout.n_games else 0.0
    return {
        "surrogate": scale * surrogate_total / n if n and diagnostic_adv is not None else 0.0,
        "entropy": entropy_total / n if n else 0.0,
        "legal_actions": legal_actions_total / n if n else 0.0,
        "max_entropy": max_entropy_total / n if n else 0.0,
        "normalized_entropy": normalized_entropy_total / n if n else 0.0,
        "effective_actions": effective_actions_total / n if n else 0.0,
        "approx_kl": kl_total / n if n else 0.0,
        "clip_fraction": clip_total / n if n else 0.0,
        "max_abs_log_ratio": max_abs_log_ratio,
    }


def rollout_kl(actor: Actor, rollout: Rollout, optim: Optim, device: str = "cpu", chunk: int = FORWARD_CHUNK) -> float:
    """`mean((r - 1) - log r)` over the whole rollout: nonnegative, and sampled."""
    return rollout_actor_diagnostics(actor, rollout, optim, device=device, chunk=chunk)["approx_kl"]


def _explained_variance(returns: np.ndarray, values: np.ndarray) -> float | None:
    variance = float(np.var(returns))
    return None if variance == 0.0 else 1.0 - float(np.var(returns - values)) / variance


def _minibatches(rng, n: int, size: int) -> list[np.ndarray]:
    perm = rng.permutation(n)
    return [perm[i : i + size] for i in range(0, n, size)]


def update(
    actor: Actor,
    critic: Critic,
    actor_opt: torch.optim.Optimizer,
    critic_opt: torch.optim.Optimizer,
    rollout: Rollout,
    optim: Optim,
    training_seed: int,
    update_index: int,
    device: str = "cpu",
) -> UpdateStats:
    """One synchronous optimization pass over a complete rollout.

    Step 5 of pipeline plan section 3: evaluate critic `k` on the pre-action critic
    inputs, freeze the advantages for every PPO epoch, then optimize. The rollout's
    training data is discarded by the caller afterwards.
    """
    stats = UpdateStats()
    t, g = rollout.n_decisions, rollout.n_games
    if t == 0:
        return stats
    started = time.perf_counter()
    values = critic_values(critic, rollout.critic, device)
    stats.critic_value_seconds = time.perf_counter() - started
    adv = advantages(rollout, values)
    rets = rollout.returns_per_decision.astype(np.float64)
    stats.advantage_mean, stats.advantage_std = float(adv.mean()), float(adv.std())
    stats.return_mean, stats.return_std = float(rets.mean()), float(rets.std())
    # An explained variance with zero return variance is undefined; log it as
    # unavailable rather than inventing a value (pipeline plan section 10).
    stats.critic_pre_update_ev = _explained_variance(rets, values)

    size = min(optim.minibatch, t)
    scale = t / g

    started = time.perf_counter()
    actor.train()
    for epoch in range(optim.actor_epochs):
        rng = generator(training_seed, MINIBATCH, "actor", update_index, epoch)
        for index in _minibatches(rng, t, size):
            loss, _ = actor_loss(actor, rollout, index, adv, scale, optim, device)
            actor_opt.zero_grad(set_to_none=True)
            loss.backward()
            norm = float(torch.nn.utils.clip_grad_norm_(actor.parameters(), optim.actor_grad_norm))
            actor_opt.step()
            stats.actor_steps += 1
            stats.actor_grad_norm += norm
            stats.actor_clipped_fraction += float(norm > optim.actor_grad_norm)
        stats.actor_epochs = epoch + 1
        diagnostic = rollout_actor_diagnostics(actor, rollout, optim, adv, device)
        stats.surrogate = diagnostic["surrogate"]
        stats.entropy = diagnostic["entropy"]
        stats.legal_actions = diagnostic["legal_actions"]
        stats.max_entropy = diagnostic["max_entropy"]
        stats.normalized_entropy = diagnostic["normalized_entropy"]
        stats.effective_actions = diagnostic["effective_actions"]
        stats.approx_kl = diagnostic["approx_kl"]
        stats.clip_fraction = diagnostic["clip_fraction"]
        stats.max_abs_log_ratio = diagnostic["max_abs_log_ratio"]
        stats.kl_per_epoch.append(stats.approx_kl)
        if stats.approx_kl > optim.target_kl:
            stats.early_stopped = True
            break
    if stats.actor_steps:
        stats.actor_grad_norm /= stats.actor_steps
        stats.actor_clipped_fraction /= stats.actor_steps
    _synchronize_device(device)
    stats.actor_update_seconds = time.perf_counter() - started

    # The critic is fitted to completion even when the actor stopped early.
    started = time.perf_counter()
    critic.train()
    targets = torch.from_numpy(rets.astype(np.float32)).to(device)
    for epoch in range(optim.critic_epochs):
        rng = generator(training_seed, MINIBATCH, "critic", update_index, epoch)
        total, seen = 0.0, 0
        for index in _minibatches(rng, t, size):
            pred = critic(*to_tensors(rollout.critic[index], device))
            loss = ((pred - targets[torch.from_numpy(index.astype(np.int64)).to(device)]) ** 2).mean()
            critic_opt.zero_grad(set_to_none=True)
            loss.backward()
            norm = float(torch.nn.utils.clip_grad_norm_(critic.parameters(), optim.critic_grad_norm))
            stats.critic_grad_norm += norm
            stats.critic_clipped_fraction += float(norm > optim.critic_grad_norm)
            critic_opt.step()
            stats.critic_steps += 1
            total += float(loss.detach()) * len(index)
            seen += len(index)
        stats.critic_epochs = epoch + 1
        stats.critic_mse = total / seen if seen else math.nan
    if stats.critic_steps:
        stats.critic_grad_norm /= stats.critic_steps
        stats.critic_clipped_fraction /= stats.critic_steps
    post_values = critic_values(critic, rollout.critic, device)
    stats.critic_post_update_same_rollout_ev = _explained_variance(rets, post_values)
    _synchronize_device(device)
    stats.critic_update_seconds = time.perf_counter() - started
    actor.eval()
    critic.eval()
    return stats


__all__ = [
    "UpdateStats", "actor_loss", "advantages", "critic_values", "new_log_probs",
    "rollout_actor_diagnostics", "rollout_kl", "surrogate", "update", "weighted_gradient",
]  # fmt: skip
