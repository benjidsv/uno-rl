"""The resolved, immutable learner configuration.

Every number here is one of docs/rl/pipeline-plan.md section 6's **proposed first
diagnostic** values, not a measured optimum and not a hyperparameter Benji
endorsed individually. They are bounded starting proposals: change them during
the diagnostic pilot with a recorded reason, and freeze them before multi-seed
gates and comparisons (pipeline plan section 1).

A `Config` has two identities. `hash` covers the complete resolved run for
provenance. `algorithm_hash` excludes budgets and execution/evaluation settings,
so a checkpoint can continue for longer or on another device without silently
changing the model, optimizer, batch construction, seed or objective. Changing
the observation schema, league, optimizer hyperparameters or objective still
makes an explicit experiment branch, never an invisible resume (section 11).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace

from uno.game.rules import NAMED, Ruleset
from uno.rl.features import ARM_B, SCHEMA_VERSION

CONFIG_VERSION = 2


@dataclass(frozen=True)
class Optim:
    """PPO and optimizer settings (pipeline plan section 6's table)."""

    gamma: float = 1.0  # terminal Monte Carlo returns; no bootstrapping
    gae_lambda: float | None = None # none is plain mc returns, should match it
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    clip: float = 0.2
    actor_epochs: int = 4
    critic_epochs: int = 4
    n_minibatches: int = 32  # balanced shuffled partitions per PPO epoch
    precision: str = "fp32"  # PPO CUDA compute: "fp32" or BF16 autocast via "bf16"
    normalize_advantages: bool = False
    entropy_coef: float = 0.0  # 0.001 is a conditional exploration diagnostic
    actor_grad_norm: float = 0.5
    critic_grad_norm: float = 0.5
    target_kl: float | None = (
        0.02  # None disables early stopping; KL diagnostics are still computed
    )
    temperature: float = 1.0
    value_clip: float | None = None  # no value clipping and no LR schedule initially


@dataclass(frozen=True)
class Collection:
    """What one synchronous iteration collects (pipeline plan section 3)."""

    games_per_cell: int = 64  # the throughput smoke's sizing: 8 cells -> 512 games
    live_games_per_worker: int = 64
    workers: int = 1  # 1 is the reference collector; more is `uno.rl.parallel`
    target_decisions: int | None = None  # after the pilot: quota chosen from this


@dataclass(frozen=True)
class Evaluation:
    """Scheduled DEV evaluation (pipeline plan section 9)."""

    deals_per_cell: int = 256  # pilot sizing
    every_fraction: float = 0.1  # of the declared learner-data cap
    panel_start: int = 800_000  # the fixed scheduled selection panel partition


@dataclass(frozen=True)
class Config:
    """One resolved training configuration."""

    ruleset: str = "FULL_UNO"
    arm: str = ARM_B
    width: int = 128
    blocks: int = 2
    training_seed: int = 0
    device: str = "cpu"  # optimization device; collection is always CPU workers
    data_cap: int = 200_000  # learner decisions; a soft cap at iteration boundaries
    time_cap_s: float | None = None  # wall-time cap, also soft at boundaries
    optim: Optim = field(default_factory=Optim)
    collection: Collection = field(default_factory=Collection)
    evaluation: Evaluation = field(default_factory=Evaluation)
    schema_version: int = SCHEMA_VERSION
    version: int = CONFIG_VERSION

    @property
    def rs(self) -> Ruleset:
        return NAMED[self.ruleset]

    def resolved(self) -> dict:
        return asdict(self)

    @property
    def hash(self) -> str:
        from uno.eval.play import stable_hash

        return f"{stable_hash(_canonical(self.resolved())):016x}"

    def algorithm_resolved(self) -> dict:
        """Configuration that must remain fixed across a true continuation."""
        return algorithm_resolved(self.resolved())

    @property
    def algorithm_hash(self) -> str:
        return algorithm_hash(self.resolved())

    def with_(self, **kw) -> Config:
        return replace(self, **kw)


def _canonical(d: dict):
    """Sorted, primitive-only form of a config, so its hash is stable across runs."""
    if isinstance(d, dict):
        return tuple(sorted((k, _canonical(v)) for k, v in d.items()))
    if isinstance(d, (list, tuple)):
        return tuple(_canonical(x) for x in d)
    return d


def algorithm_resolved(resolved: dict) -> dict:
    """Project a serialized full configuration onto continuation invariants."""
    algorithm = {
        key: value
        for key, value in resolved.items()
        if key not in {"device", "data_cap", "time_cap_s", "evaluation"}
    }
    algorithm["optim"] = {
        key: value
        for key, value in resolved["optim"].items()
        if key != "precision"
    }
    algorithm["collection"] = {
        key: value
        for key, value in resolved["collection"].items()
        if key not in {"workers", "live_games_per_worker"}
    }
    return algorithm


def algorithm_hash(resolved: dict) -> str:
    from uno.eval.play import stable_hash

    return f"{stable_hash(_canonical(algorithm_resolved(resolved))):016x}"


__all__ = [
    "CONFIG_VERSION",
    "Collection",
    "Config",
    "Evaluation",
    "Optim",
    "algorithm_hash",
    "algorithm_resolved",
]
