from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from uno.rl.core.config import Config
from uno.rl.core.device import resolve_device
from uno.rl.algo.parallel import ParallelCollector, WorkerConfig
from uno.rl.algo.train import Run, train, start_tracker
from uno.rl.eval.opponents import league


@dataclass(frozen=True)
class Execution:
    collection_threads: int = 0
    update_threads: int = 0


@dataclass(frozen=True)
class Tracking:
    study_id: str = "experimental"
    kind: str = "train"
    name: str = ""


@dataclass(frozen=True)
class TrainSpec:
    config: Config
    league: str = "rulebot"
    execution: Execution = field(default_factory=Execution)
    tracking: Tracking = field(default_factory=Tracking)


def validate(spec: TrainSpec) -> None:
    cfg = spec.config

    if cfg.data_cap <= 0:
        raise ValueError("data_cap must be positive")

    if cfg.time_cap_s is not None and cfg.time_cap_s < 0:
        raise ValueError("time_cap_s must be nonnegative")

    if cfg.collection.workers < 1:
        raise ValueError("workers must be positive")

    if cfg.collection.live_games_per_worker < 1:
        raise ValueError("live games must be positive")

    if cfg.collection.games_per_cell < 1:
        raise ValueError("games per cell must be positive")

    if not 0 < cfg.evaluation.every_fraction <= 1:
        raise ValueError("evaluation fraction must be in (0, 1]")

    if cfg.optim.n_minibatches < 1:
        raise ValueError("n_minibatches must be positive")

    if cfg.optim.precision not in {"fp32", "bf16"}:
        raise ValueError("unsupported precision")

    if cfg.optim.precision == "bf16" and not cfg.device.startswith("cuda"):
        raise ValueError("BF16 PPO requires CUDA")

    if spec.execution.collection_threads < 0:
        raise ValueError("collection threads must be nonnegative")

    if spec.execution.update_threads < 0:
        raise ValueError("update threads must be nonnegative")
        
    league(cfg.rs, spec.league)


def make_run_directory(spec: TrainSpec) -> Path:
    import time
    cfg = spec.config
    out = Path("logs/rl") / f"{cfg.ruleset}_{cfg.arm}_seed{cfg.training_seed}_nmb{cfg.optim.n_minibatches}_{cfg.optim.precision}_{cfg.device}_{int(time.time())}"
    return out


def run(spec: TrainSpec) -> dict:
    validate(spec)
    
    cfg = spec.config
    device = resolve_device(cfg.device)
    cfg = cfg.with_(device=device)

    directory = make_run_directory(spec)

    training = Run(
        cfg,
        directory,
        league_name=spec.league,
    )

    training.execution = {
        "collection_torch_threads": spec.execution.collection_threads,
        "update_torch_threads": spec.execution.update_threads,
    }

    training.execution = {
        k: v for k, v in training.execution.items()
        if v != 0
    }

    training.tracker = start_tracker(
        training,
        study_id=spec.tracking.study_id,
        kind=spec.tracking.kind,
        mlflow_prefix=spec.tracking.name,
    )

    if cfg.collection.workers == 1:
        return train(training, training.tracker)

    worker_config = WorkerConfig(
        ruleset=cfg.ruleset,
        arm=cfg.arm,
        model_config=__import__("dataclasses").asdict(cfg.model),
        league=spec.league,
        training_seed=cfg.training_seed,
        live_games=cfg.collection.live_games_per_worker,
        temperature=cfg.optim.temperature,
    )

    with ParallelCollector(
        worker_config,
        cfg.collection.workers,
    ) as collector:
        return train(training, training.tracker, collector)
