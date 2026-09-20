"""The PPO objective, the detached baseline and the update, exactly as specified.

docs/rl/pipeline-plan.md section 6. For game `g` and learner decision `t`, with
the learner's terminal return `R_g` in `[-1, 1]`:

    A_gt = R_g - V_old(x_gt),  r_gt = exp(logp_new - logp_old)

and the actor maximizes the decision-mean PPO surrogate

    mean_t min(r_gt * A_gt, clip(r_gt, 1 - eps, 1 + eps) * A_gt).

With uniformly shuffled decision minibatches, each minibatch uses its ordinary
decision mean. `weighted_gradient` and `tests/rl/test_ppo.py` pin equality with
the full decision-mean gradient, including unequal game lengths and a final
partial minibatch.

The critic is **frozen before the batch is fitted**: advantages are computed once
from `V_old`, detached, and reused for every PPO epoch. They are not recomputed
with the newly fitted critic. By default (`gamma = 1`) the critic's loss is a
separate decision-mean MSE against the same undiscounted terminal returns -- no
TD, no GAE, no bootstrapping (contract section 3). A non-unit gamma applies
decision-step discounting to the terminal return, and `value_clip` optionally
uses PPO's clipped value loss around the frozen pre-update values.

The KL diagnostic is sampled on the collection policy's own actions, not an
exact sum over the current policy's full legal-action distribution. The actor
computes the nonnegative log-space K2 approximation `0.5 * mean(log_ratio^2)`
for every minibatch, aggregates it over each completed epoch, and optionally
early-stops between epochs when `target_kl` is not None. Critic fitting always
completes.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from uno.rl.config import Optim
from uno.rl.features import Features
from uno.rl.model import (
    Actor,
    Critic,
    entropy_from_log_probs,
    masked_log_probs,
    to_tensors,
)
from uno.rl.rollout import Rollout
from uno.rl.streams import MINIBATCH, generator

FORWARD_CHUNK = (
    1024  # rows per no-grad forward pass; a memory setting, not an algorithmic one
)


from uno.rl.device import synchronize_device as _synchronize_device


@dataclass
class UpdateStats:
    """One update's diagnostics. `None` means *unavailable*, never a stand-in value."""

    actor_epochs: int = 0
    critic_epochs: int = 0
    actor_steps: int = 0
    critic_steps: int = 0
    actor_presentations: int = 0
    critic_presentations: int = 0
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
        pre_ev = (
            "n/a"
            if self.critic_pre_update_ev is None
            else f"{self.critic_pre_update_ev:.3f}"
        )
        post_ev = (
            "n/a"
            if self.critic_post_update_same_rollout_ev is None
            else f"{self.critic_post_update_same_rollout_ev:.3f}"
        )
        return (
            f"surrogate {self.surrogate:+.5f} critic mse {self.critic_mse:.5f} (ev {pre_ev}->{post_ev}) entropy {self.entropy:.3f} "
            f"/ max {self.max_entropy:.3f} (norm {self.normalized_entropy:.2f}, eff {self.effective_actions:.2f}) "
            f"kl {self.approx_kl:.4f} clip {self.clip_fraction:.3f} |grad| {self.actor_grad_norm:.3f} "
            f"epochs {self.actor_epochs}/{self.critic_epochs}"
            + (" [early stop]" if self.early_stopped else "")
        )


# -- the frozen baseline --------------------------------------------------------


@torch.no_grad()
def critic_values(
    critic: Critic, features: Features, device: str = "cpu", chunk: int = FORWARD_CHUNK
) -> np.ndarray:
    critic.eval()
    out = []
    for i in range(0, len(features), chunk):
        out.append(critic(*to_tensors(features[i : i + chunk], device)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


@torch.no_grad()
def _critic_values_tensors(
    critic: Critic,
    tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    chunk: int = FORWARD_CHUNK,
) -> np.ndarray:
    critic.eval()
    tok, col, glob = tensors
    out = []
    for i in range(0, len(tok), chunk):
        out.append(
            critic(tok[i : i + chunk], col[i : i + chunk], glob[i : i + chunk])
            .cpu()
            .numpy()
        )
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def advantages(rollout: Rollout, values: np.ndarray) -> np.ndarray:
    """`A_gt = R_g - V_old(x_gt)`, detached by construction (a plain array)."""
    return (rollout.returns_per_decision - values).astype(np.float64)


def normalize_advantages(adv: np.ndarray) -> np.ndarray:
    """Normalize one complete rollout's advantages before minibatch shuffling."""
    return (adv - adv.mean()) / (adv.std() + 1e-8)


def discounted_returns(rollout: Rollout, gamma: float) -> np.ndarray:
    """Return one (possibly discounted) terminal target for every decision.

    The exponent counts remaining learner decisions after the current action.
    Thus the existing `gamma=1` configuration is exactly the undiscounted
    terminal-return objective, while `gamma<1` values earlier decisions less.
    """
    if not np.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be finite and in [0, 1]")
    terminal = rollout.returns_per_decision.astype(np.float64)
    if gamma == 1.0 or rollout.n_decisions == 0:
        return terminal
    remaining = rollout.learner_decisions[rollout.episode] - rollout.decision_index - 1
    return terminal * np.power(gamma, remaining)


# -- the objective ---------------------------------------------------------------


def surrogate(log_ratio: torch.Tensor, adv: torch.Tensor, clip: float) -> torch.Tensor:
    """`mean(min(r A, clip(r) A))` over learner decisions."""
    r = torch.exp(log_ratio)
    return torch.minimum(r * adv, torch.clamp(r, 1 - clip, 1 + clip) * adv).mean()


def _approx_kl_k2(log_ratio: torch.Tensor) -> torch.Tensor:
    """The sampled, nonnegative log-space K2 approximation to KL."""
    return 0.5 * log_ratio.square().mean()


def new_log_probs(
    actor,
    features: Features,
    masks: np.ndarray,
    actions: np.ndarray,
    temperature: float,
    device: str,
    set_ids: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recomputed log probabilities of the taken actions, plus per-row entropy.

    The *same* masking and temperature convention as collection and evaluation
    (`uno.rl.model.masked_log_probs`), which is what makes the frozen-weight
    agreement check in stage 2 meaningful.
    """
    tok, col, glob = to_tensors(features, device)
    return _new_log_probs_tensors(
        actor,
        tok,
        col,
        glob,
        torch.from_numpy(masks).to(device),
        torch.from_numpy(actions.astype(np.int64)).to(device),
        temperature,
        set_ids,
    )


def _new_log_probs_tensors(
    actor,
    tok: torch.Tensor,
    col: torch.Tensor,
    glob: torch.Tensor,
    masks: torch.Tensor,
    actions: torch.Tensor,
    temperature: float,
    set_ids: np.ndarray | None = None,
    *,
    need_entropy: bool = True,
    validate_mask: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Taken-action log-probs from already staged tensors."""
    # A tabular actor (`uno.rl.tabular`) is parameterized by information set, not by
    # features; it declares that with `uses_set_ids` and everything else is shared.
    if getattr(actor, "uses_set_ids", False):
        logits = actor.logits(set_ids)
    else:
        logits = actor(tok, col, glob)
    lp = masked_log_probs(logits, masks, temperature, validate=validate_mask)
    entropy = entropy_from_log_probs(lp) if need_entropy else None
    return lp.gather(1, actions.unsqueeze(1)).squeeze(1), entropy


def actor_loss(
    actor: Actor,
    rollout: Rollout,
    index: np.ndarray,
    adv: np.ndarray,
    optim: Optim,
    device: str = "cpu",
) -> tuple[torch.Tensor, dict]:
    """The negated surrogate for one minibatch, with its diagnostics."""
    feats = rollout.actor[index]
    set_ids = None if rollout.set_id is None else rollout.set_id[index]
    logp, entropy = new_log_probs(
        actor,
        feats,
        rollout.masks[index],
        rollout.actions[index],
        optim.temperature,
        device,
        set_ids,
    )
    # `old_logp` was recorded during collection and remains fixed throughout
    # all actor epochs and minibatches. Only `logp` follows the updated actor.
    old = torch.from_numpy(rollout.old_logp[index].astype(np.float32)).to(device)
    a = torch.from_numpy(adv[index].astype(np.float32)).to(device)
    log_ratio = logp - old
    obj = surrogate(log_ratio, a, optim.clip)
    loss = -obj
    if optim.entropy_coef:
        assert entropy is not None
        loss = loss - optim.entropy_coef * entropy.mean()
    with torch.no_grad():
        r = torch.exp(log_ratio)
        diag = {
            "surrogate": float(obj),
            "entropy": float(entropy.mean()),
            "approx_kl": float(_approx_kl_k2(log_ratio)),
            "clip_fraction": float(((r - 1.0).abs() > optim.clip).float().mean()),
            "max_abs_log_ratio": float(log_ratio.abs().max()),
        }
    return loss, diag


@dataclass(frozen=True)
class _PreparedRollout:
    actor_tok: torch.Tensor
    actor_col: torch.Tensor
    actor_glob: torch.Tensor
    actor_masks: torch.Tensor
    actor_actions: torch.Tensor
    old_logp: torch.Tensor
    actor_set_ids: np.ndarray | None
    critic_tok: torch.Tensor
    critic_col: torch.Tensor
    critic_glob: torch.Tensor
    returns: torch.Tensor
    values: torch.Tensor
    advantages: torch.Tensor


def _prepare_rollout(
    rollout: Rollout,
    adv: torch.Tensor,
    rets: torch.Tensor,
    values: torch.Tensor,
    device: str,
    actor_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    critic_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> _PreparedRollout:
    """Stage one update once and keep derived training tensors on the accelerator."""
    # Validate the invariant once while masks are still CPU/NumPy data. Doing the
    # same check inside every CUDA actor forward would force a host synchronization
    # for every minibatch.
    if not np.asarray(rollout.masks).any(axis=1).all():
        raise ValueError("a decision row has no legal action")

    def tensors(features: Features) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return to_tensors(features, device)

    actor_tok, actor_col, actor_glob = actor_data or tensors(rollout.actor)
    critic_tok, critic_col, critic_glob = critic_data or tensors(rollout.critic)
    return _PreparedRollout(
        actor_tok,
        actor_col,
        actor_glob,
        torch.from_numpy(rollout.masks).to(device),
        torch.from_numpy(rollout.actions.astype(np.int64)).to(device),
        torch.from_numpy(rollout.old_logp.astype(np.float32)).to(device),
        None if rollout.set_id is None else rollout.set_id,
        critic_tok,
        critic_col,
        critic_glob,
        rets,
        values,
        adv,
    )


def _prepared_actor_loss(
    actor,
    prepared: _PreparedRollout,
    index: np.ndarray,
    rows: torch.Tensor,
    optim: Optim,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    set_ids = None if prepared.actor_set_ids is None else prepared.actor_set_ids[index]
    logp, entropy = _new_log_probs_tensors(
        actor,
        prepared.actor_tok[rows],
        prepared.actor_col[rows],
        prepared.actor_glob[rows],
        prepared.actor_masks[rows],
        prepared.actor_actions[rows],
        optim.temperature,
        set_ids,
        need_entropy=bool(optim.entropy_coef),
        validate_mask=False,
    )
    log_ratio = logp - prepared.old_logp[rows]
    obj = surrogate(log_ratio, prepared.advantages[rows], optim.clip)
    loss = -obj
    if optim.entropy_coef:
        assert entropy is not None
        loss = loss - optim.entropy_coef * entropy.mean()
    with torch.no_grad():
        diag = {
            "approx_kl": _approx_kl_k2(log_ratio),
        }
    return loss, diag


@torch.no_grad()
def _prepared_actor_diagnostics(
    actor: Actor,
    prepared: _PreparedRollout,
    optim: Optim,
    chunk: int = FORWARD_CHUNK,
) -> dict[str, float]:
    """Final-policy diagnostics using already-staged tensors.

    Accumulators stay on-device across chunks and are copied to the host once at
    the end, avoiding repeated CPU<->GPU staging and per-chunk synchronizations.
    """
    n_rows = len(prepared.actor_tok)
    if n_rows == 0:
        return {
            "surrogate": 0.0,
            "entropy": 0.0,
            "legal_actions": 0.0,
            "max_entropy": 0.0,
            "normalized_entropy": 0.0,
            "effective_actions": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "max_abs_log_ratio": 0.0,
        }

    device = prepared.actor_tok.device
    totals = torch.zeros(8, device=device, dtype=torch.float64)
    max_abs = torch.zeros((), device=device)

    for i in range(0, n_rows, chunk):
        sl = slice(i, i + chunk)
        set_ids = None if prepared.actor_set_ids is None else prepared.actor_set_ids[sl]
        logp, entropy = _new_log_probs_tensors(
            actor,
            prepared.actor_tok[sl],
            prepared.actor_col[sl],
            prepared.actor_glob[sl],
            prepared.actor_masks[sl],
            prepared.actor_actions[sl],
            optim.temperature,
            set_ids,
            need_entropy=True,
            validate_mask=False,
        )
        assert entropy is not None
        log_ratio = logp - prepared.old_logp[sl]
        ratio = torch.exp(log_ratio)
        adv = prepared.advantages[sl]
        clipped = torch.clamp(ratio, 1 - optim.clip, 1 + optim.clip)
        surrogate_sum = torch.minimum(ratio * adv, clipped * adv).sum()

        legal_count = prepared.actor_masks[sl].sum(dim=1).to(torch.float32)
        max_entropy = torch.log(torch.clamp_min(legal_count, 1.0))
        normalized_entropy = torch.where(
            max_entropy > 0, entropy / max_entropy, torch.zeros_like(entropy)
        )

        totals = totals + torch.stack([
            surrogate_sum.to(torch.float64),
            entropy.sum().to(torch.float64),
            legal_count.sum().to(torch.float64),
            max_entropy.sum().to(torch.float64),
            normalized_entropy.sum().to(torch.float64),
            torch.exp(entropy).sum().to(torch.float64),
            (0.5 * log_ratio.square()).sum().to(torch.float64),
            ((ratio - 1.0).abs() > optim.clip).sum().to(torch.float64),
        ])
        max_abs = torch.maximum(max_abs, log_ratio.abs().max())

    values = torch.cat([totals, max_abs.reshape(1).to(torch.float64)]).cpu().tolist()
    (
        surrogate_total,
        entropy_total,
        legal_total,
        max_entropy_total,
        norm_entropy_total,
        effective_total,
        kl_total,
        clip_total,
        max_abs_value,
    ) = values
    denom = float(n_rows)
    return {
        "surrogate": surrogate_total / denom,
        "entropy": entropy_total / denom,
        "legal_actions": legal_total / denom,
        "max_entropy": max_entropy_total / denom,
        "normalized_entropy": norm_entropy_total / denom,
        "effective_actions": effective_total / denom,
        "approx_kl": kl_total / denom,
        "clip_fraction": clip_total / denom,
        "max_abs_log_ratio": max_abs_value,
    }


def weighted_gradient(
    actor: Actor,
    rollout: Rollout,
    adv: np.ndarray,
    batches,
    optim: Optim,
    device: str = "cpu",
) -> list[torch.Tensor]:
    """`sum_k (|mb_k| / T) * grad_k`, which equals the full-batch gradient exactly.

    Exposed because that identity is the acceptance check for the decision-mean
    objective, not because the training loop uses it: training takes one optimizer
    step per minibatch.
    """
    t = rollout.n_decisions
    total = [torch.zeros_like(p) for p in actor.parameters()]
    for index in batches:
        loss, _ = actor_loss(actor, rollout, index, adv, optim, device)
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
    advantages_normalized: bool = False,
) -> dict[str, float]:
    """Final-policy diagnostics over every decision in the rollout.

    These values describe one coherent policy, unlike an average over optimizer
    steps whose minibatches were evaluated under changing parameters. The
    surrogate uses the frozen rollout advantages and the decision-mean objective.
    With advantage normalization enabled, its diagnostic normalization is
    rollout-wide; the shipped configuration leaves normalization disabled.
    """
    if adv is not None and len(adv) != rollout.n_decisions:
        raise ValueError("rollout diagnostics need one advantage per decision")
    diagnostic_adv = None
    if adv is not None:
        diagnostic_adv = adv.astype(np.float32, copy=False)
        if optim.normalize_advantages and not advantages_normalized:
            diagnostic_adv = normalize_advantages(diagnostic_adv)
    surrogate_total = entropy_total = kl_total = clip_total = 0.0
    legal_actions_total = max_entropy_total = normalized_entropy_total = (
        effective_actions_total
    ) = 0.0
    max_abs_log_ratio = 0.0
    n = 0
    for i in range(0, rollout.n_decisions, chunk):
        sl = slice(i, i + chunk)
        logp, entropy = new_log_probs(
            actor,
            rollout.actor[sl],
            rollout.masks[sl],
            rollout.actions[sl],
            optim.temperature,
            device,
            None if rollout.set_id is None else rollout.set_id[sl],
        )
        # Compare the current actor to the frozen collection-policy reference;
        # never recompute the reference from the actor being trained.
        log_ratio = logp - torch.from_numpy(rollout.old_logp[sl].astype(np.float32)).to(
            device
        )
        r = torch.exp(log_ratio)
        kl_total += float((0.5 * log_ratio.square()).sum())
        entropy_total += float(entropy.sum())
        legal_count = rollout.masks[sl].sum(axis=1).astype(np.float64)
        max_entropy = np.log(np.maximum(legal_count, 1.0))
        entropy_np = entropy.detach().cpu().numpy()
        legal_actions_total += float(legal_count.sum())
        max_entropy_total += float(max_entropy.sum())
        normalized_entropy_total += float(
            np.divide(
                entropy_np,
                max_entropy,
                out=np.zeros_like(entropy_np),
                where=max_entropy > 0,
            ).sum()
        )
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
    return {
        "surrogate": surrogate_total / n if n and diagnostic_adv is not None else 0.0,
        "entropy": entropy_total / n if n else 0.0,
        "legal_actions": legal_actions_total / n if n else 0.0,
        "max_entropy": max_entropy_total / n if n else 0.0,
        "normalized_entropy": normalized_entropy_total / n if n else 0.0,
        "effective_actions": effective_actions_total / n if n else 0.0,
        "approx_kl": kl_total / n if n else 0.0,
        "clip_fraction": clip_total / n if n else 0.0,
        "max_abs_log_ratio": max_abs_log_ratio,
    }


def rollout_kl(
    actor: Actor,
    rollout: Rollout,
    optim: Optim,
    device: str = "cpu",
    chunk: int = FORWARD_CHUNK,
) -> float:
    """`0.5 * mean(log_ratio^2)` over the whole rollout: sampled and approximate."""
    return rollout_actor_diagnostics(actor, rollout, optim, device=device, chunk=chunk)[
        "approx_kl"
    ]


def _explained_variance(returns: np.ndarray, values: np.ndarray) -> float | None:
    variance = float(np.var(returns))
    return None if variance == 0.0 else 1.0 - float(np.var(returns - values)) / variance


def _explained_variance_tensors(
    returns: torch.Tensor, values: torch.Tensor
) -> float | None:
    """Explained variance without moving full prediction vectors back to the CPU."""
    variance = torch.var(returns, correction=0)
    if float(variance) == 0.0:
        return None
    residual_variance = torch.var(returns - values, correction=0)
    return 1.0 - float(residual_variance / variance)


def _minibatches(rng, n: int, n_batches: int) -> list[np.ndarray]:
    """Return ``n_batches`` balanced shuffled partitions.

    The old helper accepted an approximate minibatch *size*.  Count semantics are
    more stable when rollout size is fixed and avoid tiny tail batches entirely.
    """
    if n <= 0:
        return []
    n_batches = max(1, min(int(n_batches), n))
    perm = rng.permutation(n)
    return np.array_split(perm, n_batches)


def _configured_n_minibatches(optim: Optim, n: int) -> int:
    """Resolve the configured number of balanced minibatches for this rollout."""
    configured = int(optim.n_minibatches)
    if configured <= 0:
        raise ValueError("n_minibatches must be positive")
    return min(configured, n)


def critic_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    old_prediction: torch.Tensor,
    value_clip: float | None,
) -> torch.Tensor:
    """PPO value loss, optionally clipped around the pre-update prediction."""
    if value_clip is None:
        return ((prediction - target) ** 2).mean()
    if not np.isfinite(value_clip) or value_clip < 0.0:
        raise ValueError("value_clip must be None or a finite nonnegative number")
    clipped = old_prediction + torch.clamp(
        prediction - old_prediction, -value_clip, value_clip
    )
    return torch.maximum(
        (prediction - target) ** 2,
        (clipped - target) ** 2,
    ).mean()


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

    Derived tensors stay on-device for the whole update.  KL early stopping is
    computed for every minibatch and checked once per completed actor epoch.
    With ``target_kl=None`` the exact same KL work and epoch diagnostics remain;
    only the early-stop control decision is disabled.
    """
    stats = UpdateStats()
    t = rollout.n_decisions
    if t == 0:
        return stats

    actor_tensor_data = to_tensors(rollout.actor, device)
    critic_tensor_data = to_tensors(rollout.critic, device)

    # Freeze the critic baseline on-device.  Only scalar diagnostics cross back
    # to the host; the values/advantages themselves never make a GPU->CPU->GPU trip.
    started = time.perf_counter()
    critic.eval()
    with torch.no_grad():
        value_chunks = []
        tok, col, glob = critic_tensor_data
        for i in range(0, len(tok), FORWARD_CHUNK):
            value_chunks.append(
                critic(
                    tok[i : i + FORWARD_CHUNK],
                    col[i : i + FORWARD_CHUNK],
                    glob[i : i + FORWARD_CHUNK],
                )
            )
        values = (
            torch.cat(value_chunks) if value_chunks else torch.zeros(0, device=device)
        )
    _synchronize_device(device)
    stats.critic_value_seconds = time.perf_counter() - started

    raw_returns = torch.from_numpy(rollout.returns_per_decision.astype(np.float32)).to(
        device
    )
    rets = torch.from_numpy(
        discounted_returns(rollout, optim.gamma).astype(np.float32)
    ).to(device)
    adv = raw_returns - values

    # Pull only scalar summaries to Python.
    adv_mean, adv_std, ret_mean, ret_std = (
        torch
        .stack([adv.mean(), adv.std(correction=0), rets.mean(), rets.std(correction=0)])
        .detach()
        .cpu()
        .tolist()
    )
    stats.advantage_mean, stats.advantage_std = adv_mean, adv_std
    stats.return_mean, stats.return_std = ret_mean, ret_std
    stats.critic_pre_update_ev = _explained_variance_tensors(rets, values)

    if optim.normalize_advantages:
        adv = (adv - adv.mean()) / (adv.std(correction=0) + 1e-8)

    prepared = _prepare_rollout(
        rollout, adv, rets, values, device, actor_tensor_data, critic_tensor_data
    )
    n_minibatches = _configured_n_minibatches(optim, t)

    # Actor -----------------------------------------------------------------
    started = time.perf_counter()
    actor.train()
    actor_grad_total = torch.zeros((), device=device)
    actor_clipped_total = torch.zeros((), device=device)

    target_kl = getattr(optim, "target_kl", None)
    for epoch in range(optim.actor_epochs):
        rng = generator(training_seed, MINIBATCH, "actor", update_index, epoch)
        epoch_kl_total = torch.zeros((), device=device)
        epoch_rows = 0

        for index in _minibatches(rng, t, n_minibatches):
            rows = torch.from_numpy(index.astype(np.int64)).to(device)
            loss, minibatch_diag = _prepared_actor_loss(
                actor, prepared, index, rows, optim
            )
            epoch_kl_total = epoch_kl_total + minibatch_diag[
                "approx_kl"
            ].detach() * len(index)
            epoch_rows += len(index)

            actor_opt.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                actor.parameters(), optim.actor_grad_norm
            )
            actor_opt.step()
            stats.actor_steps += 1
            stats.actor_presentations += len(index)
            actor_grad_total = actor_grad_total + norm.detach()
            actor_clipped_total = actor_clipped_total + (
                norm > optim.actor_grad_norm
            ).to(actor_clipped_total.dtype)

        stats.actor_epochs = epoch + 1
        if epoch_rows:
            epoch_kl = epoch_kl_total / epoch_rows
            epoch_kl_value = float(
                epoch_kl
            )  # one host sync per epoch, not per minibatch
            stats.kl_per_epoch.append(epoch_kl_value)
            if (
                target_kl is not None
                and math.isfinite(float(target_kl))
                and epoch_kl_value > float(target_kl)
            ):
                stats.early_stopped = True
                break

    # Coherent final-policy diagnostics, using the already-staged rollout.
    diagnostic = _prepared_actor_diagnostics(actor, prepared, optim)
    stats.surrogate = diagnostic["surrogate"]
    stats.entropy = diagnostic["entropy"]
    stats.legal_actions = diagnostic["legal_actions"]
    stats.max_entropy = diagnostic["max_entropy"]
    stats.normalized_entropy = diagnostic["normalized_entropy"]
    stats.effective_actions = diagnostic["effective_actions"]
    stats.approx_kl = diagnostic["approx_kl"]
    stats.clip_fraction = diagnostic["clip_fraction"]
    stats.max_abs_log_ratio = diagnostic["max_abs_log_ratio"]
    if stats.actor_steps:
        grad_mean, clipped_frac = (
            torch
            .stack([
                actor_grad_total / stats.actor_steps,
                actor_clipped_total / stats.actor_steps,
            ])
            .detach()
            .cpu()
            .tolist()
        )
        stats.actor_grad_norm = grad_mean
        stats.actor_clipped_fraction = clipped_frac
    _synchronize_device(device)
    stats.actor_update_seconds = time.perf_counter() - started

    # Critic ----------------------------------------------------------------
    started = time.perf_counter()
    critic.train()
    critic_grad_total = torch.zeros((), device=device)
    critic_clipped_total = torch.zeros((), device=device)
    last_total = torch.zeros((), device=device)
    last_seen = 0

    for epoch in range(optim.critic_epochs):
        rng = generator(training_seed, MINIBATCH, "critic", update_index, epoch)
        total = torch.zeros((), device=device)
        seen = 0
        for index in _minibatches(rng, t, n_minibatches):
            rows = torch.from_numpy(index.astype(np.int64)).to(device)
            pred = critic(
                prepared.critic_tok[rows],
                prepared.critic_col[rows],
                prepared.critic_glob[rows],
            )
            loss = critic_loss(
                pred, prepared.returns[rows], prepared.values[rows], optim.value_clip
            )
            critic_opt.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                critic.parameters(), optim.critic_grad_norm
            )
            critic_grad_total = critic_grad_total + norm.detach()
            critic_clipped_total = critic_clipped_total + (
                norm > optim.critic_grad_norm
            ).to(critic_clipped_total.dtype)
            critic_opt.step()
            stats.critic_steps += 1
            stats.critic_presentations += len(index)
            total = total + loss.detach() * len(index)
            seen += len(index)
        stats.critic_epochs = epoch + 1
        last_total, last_seen = total, seen

    if last_seen:
        stats.critic_mse = float(last_total / last_seen)
    if stats.critic_steps:
        grad_mean, clipped_frac = (
            torch
            .stack([
                critic_grad_total / stats.critic_steps,
                critic_clipped_total / stats.critic_steps,
            ])
            .detach()
            .cpu()
            .tolist()
        )
        stats.critic_grad_norm = grad_mean
        stats.critic_clipped_fraction = clipped_frac

    critic.eval()
    with torch.no_grad():
        post_chunks = []
        tok, col, glob = critic_tensor_data
        for i in range(0, len(tok), FORWARD_CHUNK):
            post_chunks.append(
                critic(
                    tok[i : i + FORWARD_CHUNK],
                    col[i : i + FORWARD_CHUNK],
                    glob[i : i + FORWARD_CHUNK],
                )
            )
        post_values = (
            torch.cat(post_chunks) if post_chunks else torch.zeros(0, device=device)
        )
    stats.critic_post_update_same_rollout_ev = _explained_variance_tensors(
        rets, post_values
    )
    _synchronize_device(device)
    stats.critic_update_seconds = time.perf_counter() - started

    actor.eval()
    critic.eval()
    return stats


__all__ = [
    "UpdateStats", "actor_loss", "advantages", "critic_loss", "critic_values", "normalize_advantages",
    "discounted_returns", "new_log_probs",
    "rollout_actor_diagnostics", "rollout_kl", "surrogate", "update", "weighted_gradient",
]  # fmt: skip
