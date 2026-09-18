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
from dataclasses import dataclass, field

import numpy as np
import torch

from uno.rl.config import Optim
from uno.rl.features import Features
from uno.rl.model import Actor, Critic, entropy_from_log_probs, masked_log_probs, to_tensors
from uno.rl.rollout import Rollout
from uno.rl.streams import MINIBATCH, generator

FORWARD_CHUNK = 1024  # rows per no-grad forward pass; a memory setting, not an algorithmic one


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
    approx_kl: float = 0.0
    clip_fraction: float = 0.0
    max_abs_log_ratio: float = 0.0
    actor_grad_norm: float = 0.0
    actor_clipped_fraction: float = 0.0
    critic_grad_norm: float = 0.0
    explained_variance: float | None = None
    advantage_mean: float = 0.0
    advantage_std: float = 0.0
    return_mean: float = 0.0
    return_std: float = 0.0
    early_stopped: bool = False
    kl_per_epoch: list[float] = field(default_factory=list)

    def __str__(self) -> str:
        ev = "n/a" if self.explained_variance is None else f"{self.explained_variance:.3f}"
        return (
            f"surrogate {self.surrogate:+.5f} critic mse {self.critic_mse:.5f} (ev {ev}) entropy {self.entropy:.3f} "
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
def rollout_kl(actor: Actor, rollout: Rollout, optim: Optim, device: str = "cpu", chunk: int = FORWARD_CHUNK) -> float:
    """`mean((r - 1) - log r)` over the whole rollout: nonnegative, and sampled."""
    total, n = 0.0, 0
    for i in range(0, rollout.n_decisions, chunk):
        sl = slice(i, i + chunk)
        logp, _ = new_log_probs(actor, rollout.actor[sl], rollout.masks[sl], rollout.actions[sl], optim.temperature, device, None if rollout.set_id is None else rollout.set_id[sl])
        log_ratio = logp - torch.from_numpy(rollout.old_logp[sl].astype(np.float32)).to(device)
        r = torch.exp(log_ratio)
        total += float(((r - 1.0) - log_ratio).sum())
        n += len(logp)
    return total / n if n else 0.0


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
    values = critic_values(critic, rollout.critic, device)
    adv = advantages(rollout, values)
    rets = rollout.returns_per_decision.astype(np.float64)
    stats.advantage_mean, stats.advantage_std = float(adv.mean()), float(adv.std())
    stats.return_mean, stats.return_std = float(rets.mean()), float(rets.std())
    var = float(np.var(rets))
    # An explained variance with zero return variance is undefined; log it as
    # unavailable rather than inventing a value (pipeline plan section 10).
    stats.explained_variance = None if var == 0.0 else 1.0 - float(np.var(rets - values)) / var

    size = min(optim.minibatch, t)
    scale = t / g

    actor.train()
    for epoch in range(optim.actor_epochs):
        rng = generator(training_seed, MINIBATCH, "actor", update_index, epoch)
        for index in _minibatches(rng, t, size):
            loss, diag = actor_loss(actor, rollout, index, adv, scale, optim, device)
            actor_opt.zero_grad(set_to_none=True)
            loss.backward()
            norm = float(torch.nn.utils.clip_grad_norm_(actor.parameters(), optim.actor_grad_norm))
            actor_opt.step()
            stats.actor_steps += 1
            stats.actor_grad_norm += norm
            stats.actor_clipped_fraction += float(norm > optim.actor_grad_norm)
            stats.surrogate, stats.entropy = diag["surrogate"], diag["entropy"]
            stats.clip_fraction, stats.max_abs_log_ratio = diag["clip_fraction"], max(stats.max_abs_log_ratio, diag["max_abs_log_ratio"])
        stats.actor_epochs = epoch + 1
        kl = rollout_kl(actor, rollout, optim, device)
        stats.kl_per_epoch.append(kl)
        stats.approx_kl = kl
        if kl > optim.target_kl:
            stats.early_stopped = True
            break
    if stats.actor_steps:
        stats.actor_grad_norm /= stats.actor_steps
        stats.actor_clipped_fraction /= stats.actor_steps

    # The critic is fitted to completion even when the actor stopped early.
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
            stats.critic_grad_norm += float(torch.nn.utils.clip_grad_norm_(critic.parameters(), optim.critic_grad_norm))
            critic_opt.step()
            stats.critic_steps += 1
            total += float(loss.detach()) * len(index)
            seen += len(index)
        stats.critic_epochs = epoch + 1
        stats.critic_mse = total / seen if seen else math.nan
    if stats.critic_steps:
        stats.critic_grad_norm /= stats.critic_steps
    actor.eval()
    critic.eval()
    return stats


__all__ = [
    "UpdateStats", "actor_loss", "advantages", "critic_values", "new_log_probs",
    "rollout_kl", "surrogate", "update", "weighted_gradient",
]  # fmt: skip
