"""The coordinator: one synchronous iteration at a time, with caps at boundaries.

docs/rl/pipeline-plan.md section 3, step by step:

1. freeze actor `k`, critic `k` and the preprocessing, and build the complete
   episode manifest before any work is dispatched;
2. equal quotas on the opponent x seat cells;
3. collect every assigned game to a true terminal state under actor `k`;
4. reassemble in logical ID order and validate;
5. evaluate critic `k` on the pre-action critic inputs and freeze the advantages;
6. optimize actor and critic on this batch, then discard its training data;
7. advance counters, evaluate if due, checkpoint if due, publish actor `k+1`.

The coordinator owns episode assignment, policy versions, rollout assembly, the
optimizers, evaluation scheduling, checkpoints and the MLflow writes. Workers own
none of those.

**Caps are soft at iteration boundaries.** Another iteration is not started when
the estimated remaining time cannot cover collection, the update, a required
evaluation and saving; an overrun by the final complete iteration is logged rather
than hidden. The first Ctrl-C requests a graceful stop: finish the current
iteration, evaluate, save and close tracking. A second Ctrl-C is an immediate hard
interrupt. This loop never records a truncation as a terminal result.
"""

from __future__ import annotations

import json
import shutil
import signal
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from uno.progress import bar
from uno.rl import checkpoint as ckpt
from uno.rl import evaluate as ev
from uno.rl import manifest as mf
from uno.rl import ppo
from uno.rl.collect import collect
from uno.rl.config import Config
from uno.rl.features import spec
from uno.rl.model import Actor, Critic, shapes
from uno.rl.opponents import league, league_hash
from uno.rl.policy import ActorPolicy
from uno.rl.streams import INIT, STREAM_VERSION, torch_seed
from uno.rl.track import Tracker, provenance

CHECKPOINT_INTERVAL_S = 120.0  # at most two minutes between boundary saves


_TRAIN_MLFLOW_PATHS = {
    # Progress counters and clocks.
    "update": "progress/update",
    "games": "progress/games",
    "engine_decisions": "progress/engine_decisions",
    "optimizer_steps": "progress/optimizer_steps",
    "active_seconds": "progress/active_seconds",
    "run_wall_seconds": "progress/wall_seconds",
    "actor_presentations": "progress/actor_presentations",
    "critic_presentations": "progress/critic_presentations",
    # PPO health and optimization statistics.
    "surrogate": "actor/surrogate",
    "policy_loss": "actor/policy_loss",
    "entropy": "actor/entropy",
    "legal_actions": "actor/legal_actions",
    "max_entropy": "actor/max_entropy",
    "normalized_entropy": "actor/normalized_entropy",
    "effective_actions": "actor/effective_actions",
    "approx_kl": "actor/approx_kl",
    "clip_fraction": "actor/clip_fraction",
    "max_abs_log_ratio": "actor/max_abs_log_ratio",
    "actor_grad_norm": "actor/grad_norm",
    "actor_clipped_fraction": "actor/grad_clipped_fraction",
    "actor_epochs": "actor/epochs",
    "actor_steps": "actor/steps",
    "early_stopped": "actor/early_stopped",
    "critic_mse": "critic/mse",
    "value_loss": "critic/value_loss",
    "critic_pre_update_ev": "critic/pre_update_explained_variance",
    "critic_post_update_same_rollout_ev": "critic/post_update_same_rollout_explained_variance",
    "critic_grad_norm": "critic/grad_norm",
    "critic_clipped_fraction": "critic/grad_clipped_fraction",
    "critic_epochs": "critic/epochs",
    "critic_steps": "critic/steps",
    "advantage_mean": "batch/advantage_mean",
    "advantage_std": "batch/advantage_std",
    "return_mean": "batch/return_mean",
    "return_std": "batch/return_std",
    # Current on-policy rollout summaries.
    "rollout_games": "rollout/games",
    "rollout_learner_decisions": "rollout/learner_decisions",
    "rollout_engine_decisions": "rollout/engine_decisions",
    "rollout_reward_mean": "rollout/reward_mean",
    "rollout_reward_std": "rollout/reward_std",
    "rollout_score_mean": "rollout/score_mean",
    "rollout_score_std": "rollout/score_std",
    "rollout_win_rate": "rollout/win_rate",
    "rollout_draw_rate": "rollout/draw_rate",
    "rollout_loss_rate": "rollout/loss_rate",
    "rollout_cap_rate": "rollout/cap_rate",
    "rollout_engine_length_mean": "rollout/engine_length_mean",
    "rollout_engine_length_std": "rollout/engine_length_std",
    "rollout_engine_length_min": "rollout/engine_length_min",
    "rollout_engine_length_p50": "rollout/engine_length_p50",
    "rollout_engine_length_p90": "rollout/engine_length_p90",
    "rollout_engine_length_max": "rollout/engine_length_max",
    "rollout_learner_length_mean": "rollout/learner_length_mean",
    "rollout_learner_length_std": "rollout/learner_length_std",
    "rollout_learner_length_min": "rollout/learner_length_min",
    "rollout_learner_length_p50": "rollout/learner_length_p50",
    "rollout_learner_length_p90": "rollout/learner_length_p90",
    "rollout_learner_length_max": "rollout/learner_length_max",
    # End-to-end and stage performance.
    "collect_s": "timing/collection_total_seconds",
    "update_s": "timing/update_total_seconds",
    "iteration_s": "timing/iteration_seconds",
    "worker_s": "timing/collection_worker_sum_seconds",
    "simulation_s": "timing/collection_simulation_sum_seconds",
    "feature_s": "timing/collection_features_sum_seconds",
    "actor_inference_s": "timing/collection_actor_inference_sum_seconds",
    "worker_assembly_s": "timing/collection_worker_assembly_sum_seconds",
    "serialization_s": "timing/collection_serialization_sum_seconds",
    "result_queue_put_s": "timing/collection_result_queue_put_sum_seconds",
    "result_queue_wait_s": "timing/collection_coordinator_queue_wait_seconds",
    "deserialization_s": "timing/collection_deserialization_seconds",
    "coordinator_assembly_s": "timing/collection_coordinator_assembly_seconds",
    "drain_s": "timing/collection_final_game_drain_seconds",
    "critic_values_s": "timing/update_critic_values_seconds",
    "actor_update_s": "timing/update_actor_seconds",
    "critic_update_s": "timing/update_critic_seconds",
    "iteration_fresh_decisions_per_s": "throughput/iteration_fresh_decisions_per_second",
    "collect_decisions_per_s": "throughput/collection_decisions_per_second",
    "update_fresh_decisions_per_s": "throughput/update_fresh_decisions_per_second",
    "actor_presentations_per_s": "throughput/actor_presentations_per_second",
    "critic_presentations_per_s": "throughput/critic_presentations_per_second",
    # Collection shape, transfer and runtime resources.
    "mean_inference_batch": "collection/inference_batch_mean",
    "inference_batch_calls": "collection/inference_batch_calls",
    "inference_batch_min": "collection/inference_batch_min",
    "inference_batch_p50": "collection/inference_batch_p50",
    "inference_batch_p90": "collection/inference_batch_p90",
    "inference_batch_max": "collection/inference_batch_max",
    "retried_bundles": "collection/retried_bundles",
    "rollout_bytes": "resources/rollout_bytes",
    "transfer_bytes": "resources/transfer_bytes",
    "collection_torch_threads": "resources/collection_torch_threads",
    "update_torch_threads": "resources/update_torch_threads",
}

_PRIMARY_TRAIN_METRICS = {
    "train_score": "primary/train_score",
    "iteration_fresh_decisions_per_s": "primary/iteration_decisions_per_second",
    "entropy": "primary/entropy",
    "normalized_entropy": "primary/normalized_entropy",
    "effective_actions": "primary/effective_actions",
    "approx_kl": "primary/approx_kl",
    "critic_pre_update_ev": "primary/critic_pre_update_explained_variance",
}


def _training_mlflow_metrics(metrics: dict, run: "Run") -> dict:
    """Namespaced MLflow view of one update; the CSV keeps the flat keys."""
    cell_paths = {
        f"train_score_{opponent}_s{seat}": f"training/score_{opponent}_seat_{seat}"
        for opponent in run.opponents
        for seat in (0, 1)
    }
    grouped = {}
    for name, value in metrics.items():
        if name == "train_score":
            path = "training/score_league_mean"
        elif name in cell_paths:
            path = cell_paths[name]
        elif name.startswith("end_"):
            path = f"collection/end_count_{name.removeprefix('end_')}"
        else:
            path = _TRAIN_MLFLOW_PATHS.get(name, f"diagnostic/{name}")
        grouped[path] = value
    grouped.update({
        target: metrics[source]
        for source, target in _PRIMARY_TRAIN_METRICS.items()
        if source in metrics
    })
    return grouped


def _evaluation_mlflow_metrics(metrics: dict, result: ev.PanelResult) -> dict:
    """Namespaced MLflow view of a fixed-panel evaluation."""
    paths = {
        "dev_league_mean": "evaluation/score_league_mean",
        "dev_se": "evaluation/score_standard_error",
        "dev_deals": "evaluation/deals_per_cell",
        "active_seconds": "progress/active_seconds",
        "run_wall_seconds": "progress/wall_seconds",
        **{
            f"dev_{opponent}_s{seat}": f"evaluation/score_{opponent}_seat_{seat}"
            for opponent, seat in result.cells
        },
        **{
            f"dev_{name}": f"evalbehavior/{name}"
            for name in result.behavior.as_dict()
        },
    }
    grouped = {
        paths.get(name, f"diagnostic/{name}"): value for name, value in metrics.items()
    }
    grouped["primary/dev_score"] = metrics["dev_league_mean"]
    return grouped


@dataclass
class Run:
    """One training run's live objects and its output directory."""

    config: Config
    directory: Path
    league_name: str = "rulebot"
    actor: Actor = None
    critic: Critic = None
    actor_opt: torch.optim.Optimizer = None
    critic_opt: torch.optim.Optimizer = None
    progress: ckpt.Progress = field(default_factory=ckpt.Progress)
    tracker: Tracker | None = None
    schedule: mf.DealSchedule = None
    eval_history: list[tuple[int, float]] = field(default_factory=list)
    lineage: dict = field(default_factory=dict)
    execution: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        cfg = self.config
        rs = cfg.rs
        self.directory = Path(self.directory)
        torch.manual_seed(torch_seed(cfg.training_seed, INIT))
        self.actor = self.actor or Actor(spec(rs, cfg.arm), cfg.width, cfg.blocks).to(
            cfg.device
        )
        self.critic = self.critic or Critic(
            spec(rs, "CRITIC"), cfg.width, cfg.blocks
        ).to(cfg.device)
        o = cfg.optim
        self.actor_opt = self.actor_opt or torch.optim.Adam(
            self.actor.parameters(),
            lr=o.actor_lr,
            betas=o.betas,
            eps=o.eps,
            weight_decay=o.weight_decay,
        )
        self.critic_opt = self.critic_opt or torch.optim.Adam(
            self.critic.parameters(),
            lr=o.critic_lr,
            betas=o.betas,
            eps=o.eps,
            weight_decay=o.weight_decay,
        )
        self.schedule = self.schedule or mf.DealSchedule(cfg.training_seed)
        self.opponents = league(rs, self.league_name)
        self.panel = mf.panel(
            list(self.opponents),
            cfg.evaluation.deals_per_cell,
            cfg.evaluation.panel_start,
        )

    @property
    def policy(self) -> ActorPolicy:
        return ActorPolicy(
            spec(self.config.rs, self.config.arm),
            self.actor,
            self.config.optim.temperature,
            self.config.device,
        )


def start_tracker(
    run: Run,
    study_id: str = "pilot",
    phase: str = "diagnostic",
    kind: str = "train",
    store_root=None,
    mlflow_prefix: str = "",
) -> Tracker:
    cfg = run.config
    tags = {
        "study_id": study_id, "arm": cfg.arm, "phase": phase, "seed": cfg.training_seed,
        "machine": provenance(cfg.hash)["machine"], "kind": kind, "league": run.league_name,
    }  # fmt: skip
    params = {
        **cfg.resolved(), "config_hash": cfg.hash, "algorithm_hash": cfg.algorithm_hash,
        "stream_version": STREAM_VERSION,
        "league_hash": league_hash(cfg.rs, run.league_name), "actor_schema": spec(cfg.rs, cfg.arm).hash,
        "critic_schema": spec(cfg.rs, "CRITIC").hash, "shapes": str(shapes(spec(cfg.rs, cfg.arm), spec(cfg.rs, "CRITIC"), cfg.width, cfg.blocks)),
        **provenance(cfg.hash),
        **{f"execution_{k}": v for k, v in run.execution.items()},
    }  # fmt: skip
    if run.lineage:  # a resumed lineage records where it came from and what it replays
        params.update({f"lineage_{k}": v for k, v in run.lineage.items()})
    tracker = Tracker(
        run.directory / "tracking",
        run_name=f"{mlflow_prefix}_{cfg.arm}-seed{cfg.training_seed}",
        tags=tags,
        params=params,
        store_root=store_root,
    )
    _write_run_manifest(run, tracker)
    return tracker


def _write_run_manifest(run: Run, tracker: Tracker) -> None:
    """The seed/panel manifest artifact: which deals this run may touch, and which it evaluates on."""
    cfg = run.config
    panel_deals = sorted({e.deal_seed for e in run.panel.episodes})
    payload = {
        "config": cfg.resolved(),
        "config_hash": cfg.hash,
        "algorithm_hash": cfg.algorithm_hash,
        "stream_version": STREAM_VERSION,
        "league": run.league_name,
        "league_members": list(run.opponents),
        "league_hash": league_hash(cfg.rs, run.league_name),
        "partitions": {
            "train": [mf.TRAIN_DEALS.start, mf.TRAIN_DEALS.stop],
            "panel": [mf.PANEL_DEALS.start, mf.PANEL_DEALS.stop],
            "diagnostic": [mf.DIAGNOSTIC_DEALS.start, mf.DIAGNOSTIC_DEALS.stop],
        },
        "panel": {
            "cells": [list(c) for c in run.panel.cells],
            "deals_per_cell": run.panel.games_per_cell,
            "first_deal": panel_deals[0],
            "last_deal": panel_deals[-1],
        },
        "deal_schedule": run.schedule.state,
        "lineage": run.lineage,
        "execution": run.execution,
        "provenance": provenance(cfg.hash),
    }
    path = run.directory / "run_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))
    tracker.artifact(path)


def iteration(
    run: Run,
    collector=None,
    collection_threads: int | None = None,
    update_threads: int | None = None,
) -> tuple[ppo.UpdateStats, dict]:
    """One iteration with optional runtime-only CPU thread budgets."""
    if collection_threads is None:
        collection_threads = run.execution.get("collection_torch_threads")
    if update_threads is None:
        update_threads = run.execution.get("update_torch_threads")
    previous_threads = torch.get_num_threads()
    try:
        if collection_threads is not None:
            torch.set_num_threads(collection_threads)
        return _iteration(run, collector, update_threads)
    finally:
        if torch.get_num_threads() != previous_threads:
            torch.set_num_threads(previous_threads)


from uno.rl.device import synchronize_device as _synchronize_device


def _iteration(
    run: Run, collector, update_threads: int | None
) -> tuple[ppo.UpdateStats, dict]:
    """One complete synchronous iteration. Returns its update stats and timings."""
    cfg = run.config
    rs = cfg.rs
    critic_spec = spec(rs, "CRITIC")
    t0 = time.perf_counter()

    manifest, next_id = mf.build(
        run.progress.update,
        list(run.opponents),
        run.schedule,
        cfg.collection.games_per_cell,
        run.progress.next_episode_id,
    )
    if collector is None:
        # Collection is deliberately CPU-local even when the coordinator updates
        # on MPS/CUDA.  Do not let ActorPolicy move the optimizer's model back to
        # CPU: that makes the subsequent device update invalid.  The parallel
        # collector already receives this equivalent CPU replica via state_dict.
        if cfg.device == "cpu":
            policy = run.policy
        else:
            actor_cpu = Actor(spec(rs, cfg.arm), cfg.width, cfg.blocks)
            actor_cpu.load_state_dict({
                k: v.detach().cpu() for k, v in run.actor.state_dict().items()
            })
            policy = ActorPolicy(
                spec(rs, cfg.arm), actor_cpu, cfg.optim.temperature, "cpu"
            )
        rollout, cstats = collect(
            manifest,
            policy,
            critic_spec,
            run.opponents,
            cfg.training_seed,
            run.progress.update,
            cfg.collection.live_games_per_worker,
        )
    else:
        rollout, cstats = collector.collect(
            manifest, run.actor, critic_spec, run.progress.update
        )
    _synchronize_device(cfg.device)
    collected = time.perf_counter()

    collection_threads_used = torch.get_num_threads()
    if update_threads is not None:
        torch.set_num_threads(update_threads)
    update_threads_used = torch.get_num_threads()
    stats = ppo.update(
        run.actor,
        run.critic,
        run.actor_opt,
        run.critic_opt,
        rollout,
        cfg.optim,
        cfg.training_seed,
        run.progress.update,
        cfg.device,
    )
    _synchronize_device(cfg.device)
    updated = time.perf_counter()

    if cstats.batch_sizes:
        batch_sizes = np.asarray(cstats.batch_sizes)
        batch_distribution = {
            "inference_batch_calls": len(batch_sizes),
            "inference_batch_min": int(batch_sizes.min()),
            "inference_batch_p50": float(np.quantile(batch_sizes, 0.50)),
            "inference_batch_p90": float(np.quantile(batch_sizes, 0.90)),
            "inference_batch_max": int(batch_sizes.max()),
        }
    else:
        batch_distribution = {
            "inference_batch_calls": 0,
            "inference_batch_min": 0,
            "inference_batch_p50": 0.0,
            "inference_batch_p90": 0.0,
            "inference_batch_max": 0,
        }

    run.progress.update += 1
    run.progress.next_episode_id = next_id
    run.progress.games += rollout.n_games
    run.progress.learner_decisions += rollout.n_decisions
    run.progress.engine_decisions += int(rollout.engine_decisions.sum())
    run.progress.optimizer_steps += stats.actor_steps + stats.critic_steps
    run.progress.deal_schedule = run.schedule.state
    update_seconds = updated - collected
    iteration_seconds = updated - t0
    actor_presentations = rollout.n_decisions * stats.actor_epochs
    critic_presentations = rollout.n_decisions * stats.critic_epochs
    timings = {
        "collect_s": collected - t0, "update_s": update_seconds, "iteration_s": iteration_seconds,
        "iteration_fresh_decisions_per_s": rollout.n_decisions / iteration_seconds if iteration_seconds else 0.0,
        "update_fresh_decisions_per_s": rollout.n_decisions / update_seconds if update_seconds else 0.0,
        "actor_presentations": actor_presentations, "critic_presentations": critic_presentations,
        "actor_presentations_per_s": actor_presentations / stats.actor_update_seconds if stats.actor_update_seconds else 0.0,
        "critic_presentations_per_s": critic_presentations / stats.critic_update_seconds if stats.critic_update_seconds else 0.0,
        "actor_steps": stats.actor_steps, "critic_steps": stats.critic_steps,
        "collect_decisions_per_s": cstats.decisions_per_second, "mean_inference_batch": cstats.mean_batch,
        "drain_s": cstats.drain_seconds, "rollout_bytes": rollout.nbytes, "retried_bundles": cstats.retried_bundles,
        "worker_s": cstats.worker_seconds, "simulation_s": cstats.simulation_seconds,
        "feature_s": cstats.feature_seconds, "actor_inference_s": cstats.inference_seconds,
        "worker_assembly_s": cstats.assembly_seconds, "serialization_s": cstats.serialization_seconds,
        "result_queue_put_s": cstats.result_queue_put_seconds,
        "result_queue_wait_s": cstats.result_queue_wait_seconds,
        "deserialization_s": cstats.deserialization_seconds, "coordinator_assembly_s": cstats.merge_seconds,
        "transfer_bytes": cstats.transfer_bytes,
        "critic_values_s": stats.critic_value_seconds, "actor_update_s": stats.actor_update_seconds,
        "critic_update_s": stats.critic_update_seconds,
        "collection_torch_threads": collection_threads_used,
        "update_torch_threads": update_threads_used,
        **batch_distribution,
    }  # fmt: skip
    per_cell = _cell_scores(rollout)
    return stats, {
        **timings,
        **_rollout_metrics(rollout),
        **per_cell,
        **{f"end_{k}": v for k, v in cstats.ends.items()},
    }


def _cell_scores(rollout) -> dict:
    """Training score per opponent x seat cell: a diagnostic, not the selection metric."""
    out = {}
    scores = (rollout.terminal_return + 1.0) / 2.0
    for name in sorted(set(rollout.opponent)):
        for seat in (0, 1):
            m = np.asarray([o == name for o in rollout.opponent]) & (
                rollout.seat == seat
            )
            if m.any():
                out[f"train_score_{name}_s{seat}"] = float(scores[m].mean())
    out["train_score"] = (
        float(np.mean([v for k, v in out.items() if k.startswith("train_score_")]))
        if out
        else 0.0
    )
    return out


def _rollout_metrics(rollout) -> dict:
    """Per-update summaries of the completed on-policy episodes."""
    rewards = np.asarray(rollout.terminal_return, dtype=np.float64)
    scores = (rewards + 1.0) / 2.0
    engine = np.asarray(rollout.engine_decisions, dtype=np.float64)
    learner = np.asarray(rollout.learner_decisions, dtype=np.float64)

    def summary(values: np.ndarray, prefix: str) -> dict:
        if not len(values):
            return {
                f"{prefix}_mean": 0.0, f"{prefix}_std": 0.0,
                f"{prefix}_min": 0.0, f"{prefix}_p50": 0.0,
                f"{prefix}_p90": 0.0, f"{prefix}_max": 0.0,
            }
        return {
            f"{prefix}_mean": float(values.mean()),
            f"{prefix}_std": float(values.std()),
            f"{prefix}_min": float(values.min()),
            f"{prefix}_p50": float(np.quantile(values, 0.50)),
            f"{prefix}_p90": float(np.quantile(values, 0.90)),
            f"{prefix}_max": float(values.max()),
        }

    n = len(rewards)
    return {
        "rollout_games": rollout.n_games,
        "rollout_learner_decisions": rollout.n_decisions,
        "rollout_engine_decisions": int(engine.sum()),
        "rollout_reward_mean": float(rewards.mean()) if n else 0.0,
        "rollout_reward_std": float(rewards.std()) if n else 0.0,
        "rollout_score_mean": float(scores.mean()) if n else 0.0,
        "rollout_score_std": float(scores.std()) if n else 0.0,
        "rollout_win_rate": float(np.mean(rewards > 0.0)) if n else 0.0,
        "rollout_draw_rate": float(np.mean(rewards == 0.0)) if n else 0.0,
        "rollout_loss_rate": float(np.mean(rewards < 0.0)) if n else 0.0,
        "rollout_cap_rate": float(np.mean(np.asarray(rollout.end_reason) == "cap")) if n else 0.0,
        **summary(engine, "rollout_engine_length"),
        **summary(learner, "rollout_learner_length"),
    }


def evaluate_now(run: Run) -> ev.PanelResult:
    """Synchronous scheduled DEV evaluation of the current actor on the fixed panel."""
    previous_threads = torch.get_num_threads()
    evaluation_threads = run.execution.get("collection_torch_threads")
    try:
        if evaluation_threads is not None:
            torch.set_num_threads(evaluation_threads)
        with bar(total=len(run.panel), desc="DEV evaluation", unit="games") as progress:
            return ev.play_panel(
                run.panel,
                run.policy,
                run.opponents,
                run.config.training_seed,
                on_games=progress.update,
            )
    finally:
        if torch.get_num_threads() != previous_threads:
            torch.set_num_threads(previous_threads)


@dataclass
class _BoundaryStop:
    """Turn the first SIGINT into a request to stop at a valid run boundary."""

    requested: bool = False
    _previous: object = None
    _installed: bool = False

    def __enter__(self):
        if threading.current_thread() is threading.main_thread():
            self._previous = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, self._handle)
            self._installed = True
        return self

    def _handle(self, signum, frame) -> None:
        if self.requested:
            signal.default_int_handler(signum, frame)
        self.requested = True
        print(
            "\n   Ctrl-C received; stopping after the current complete boundary. "
            "Press Ctrl-C again to abort immediately.",
            flush=True,
        )

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._installed:
            signal.signal(signal.SIGINT, self._previous)


def train(
    run: Run, tracker: Tracker | None = None, collector=None, verbose: bool = True
) -> dict:
    """Run to a cap; the first Ctrl-C saves and exits at a complete boundary."""
    with _BoundaryStop() as stop:
        return _train(run, tracker, collector, verbose, stop)


def _train(
    run: Run,
    tracker: Tracker | None,
    collector,
    verbose: bool,
    stop: _BoundaryStop,
) -> dict:
    cfg = run.config
    tracker = tracker or run.tracker
    started = time.perf_counter()
    last_save = started
    eval_every = max(1, int(cfg.data_cap * cfg.evaluation.every_fraction))
    last_evaluation = None
    last_evaluation_step = None
    last_dev_score = None
    if run.progress.next_eval_at == 0:
        run.progress.next_eval_at = eval_every
        first = evaluate_now(run)  # the initial policy is a scheduled evaluation point
        _record_evaluation(run, tracker, first, 0, time.perf_counter() - started)
        last_evaluation, last_evaluation_step = first, 0
        last_dev_score = first.mean
    overrun = 0.0
    iterations = []
    stop_reason = "data_cap"
    training_progress = (
        bar(
            total=max(0, cfg.data_cap - run.progress.learner_decisions),
            desc="PPO training",
            unit="decisions",
        )
        if verbose
        else None
    )

    while run.progress.learner_decisions < cfg.data_cap:
        if stop.requested:
            stop_reason = "interrupt"
            break
        if cfg.time_cap_s is not None:
            elapsed = time.perf_counter() - started
            budget = float(np.mean(iterations[-3:])) if iterations else 0.0
            if iterations and elapsed + budget > cfg.time_cap_s:
                stop_reason = "time_cap"
                break
        t0 = time.perf_counter()
        decisions_before = run.progress.learner_decisions
        stats, extra = iteration(run, collector)
        iteration_decisions = run.progress.learner_decisions - decisions_before
        iterations.append(time.perf_counter() - t0)
        run.progress.active_seconds += iterations[-1]
        metrics = {
            "update": run.progress.update, "games": run.progress.games, "engine_decisions": run.progress.engine_decisions,
            "optimizer_steps": run.progress.optimizer_steps, "active_seconds": run.progress.active_seconds,
            "run_wall_seconds": time.perf_counter() - started,
            "surrogate": stats.surrogate, "policy_loss": -stats.surrogate,
            "critic_mse": stats.critic_mse, "value_loss": stats.critic_mse,
            "critic_pre_update_ev": stats.critic_pre_update_ev,
            "critic_post_update_same_rollout_ev": stats.critic_post_update_same_rollout_ev,
            "entropy": stats.entropy, "legal_actions": stats.legal_actions, "max_entropy": stats.max_entropy,
            "normalized_entropy": stats.normalized_entropy, "effective_actions": stats.effective_actions,
            "approx_kl": stats.approx_kl, "clip_fraction": stats.clip_fraction,
            "max_abs_log_ratio": stats.max_abs_log_ratio, "actor_grad_norm": stats.actor_grad_norm,
            "actor_clipped_fraction": stats.actor_clipped_fraction, "critic_grad_norm": stats.critic_grad_norm,
            "critic_clipped_fraction": stats.critic_clipped_fraction,
            "advantage_mean": stats.advantage_mean, "advantage_std": stats.advantage_std,
            "return_mean": stats.return_mean, "return_std": stats.return_std,
            "actor_epochs": stats.actor_epochs, "critic_epochs": stats.critic_epochs, "early_stopped": int(stats.early_stopped),
            **extra,
        }  # fmt: skip
        if tracker:
            tracker.log(
                metrics,
                step=run.progress.learner_decisions,
                mlflow_metrics=_training_mlflow_metrics(metrics, run),
            )
        if verbose and training_progress.disable:
            print(
                f"   update {run.progress.update:4d}  {run.progress.learner_decisions:>9,} dec  {stats}  train {extra.get('train_score', float('nan')):.3f}"
            )

        if run.progress.learner_decisions >= run.progress.next_eval_at:
            result = evaluate_now(run)
            last_evaluation = result
            last_evaluation_step = run.progress.learner_decisions
            last_dev_score = result.mean
            while run.progress.next_eval_at <= run.progress.learner_decisions:
                run.progress.next_eval_at += eval_every
            _record_evaluation(
                run,
                tracker,
                result,
                run.progress.learner_decisions,
                time.perf_counter() - started,
            )
            _save(run, "latest")
            last_save = time.perf_counter()
        elif time.perf_counter() - last_save > CHECKPOINT_INTERVAL_S:
            _save(run, "latest")
            last_save = time.perf_counter()

        if training_progress is not None:
            training_progress.set_postfix(
                update=run.progress.update,
                collect=f"{extra['collect_s']:.1f}s",
                ppo=f"{extra['update_s']:.1f}s",
                train=f"{extra.get('train_score', float('nan')):.3f}",
                dev=("--" if last_dev_score is None else f"{last_dev_score:.3f}"),
                refresh=False,
            )
            training_progress.update(iteration_decisions)

    if training_progress is not None:
        training_progress.close()

    final_evaluation_reused = (
        last_evaluation is not None
        and last_evaluation_step == run.progress.learner_decisions
    )
    if final_evaluation_reused:
        final = last_evaluation
    else:
        final = evaluate_now(run)
        _record_evaluation(
            run,
            tracker,
            final,
            run.progress.learner_decisions,
            time.perf_counter() - started,
        )
    _save(run, "final")
    wall_seconds = time.perf_counter() - started
    if cfg.time_cap_s is not None:
        overrun = max(0.0, wall_seconds - cfg.time_cap_s)
    summary = {
        "stop_reason": stop_reason,
        "overrun_s": overrun,
        "updates": run.progress.update,
        "learner_decisions": run.progress.learner_decisions,
        "games": run.progress.games,
        "engine_decisions": run.progress.engine_decisions,
        "active_seconds": run.progress.active_seconds,
        "wall_seconds": wall_seconds,
        "evaluation_points": len(run.eval_history),
        "final_evaluation_reused": final_evaluation_reused,
        "selected_update": run.progress.best_update,
        "selected_decisions": run.progress.best_decisions,
        "selected_checkpoint": run.progress.best_checkpoint,
        "selected_checkpoint_available": bool(
            run.progress.best_checkpoint
            and (run.directory / run.progress.best_checkpoint).exists()
        ),
        "selected_score": run.progress.best_score,
        "final_score": final.mean,
        "final_ci95": final.ci95,
        "behavior": final.behavior.as_dict(),
        "execution": run.execution,
        "cells": {f"{n}_s{s}": v for (n, s), v in final.cell_means.items()},
    }
    (run.directory / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    if tracker:
        tracker.artifact(run.directory / "summary.json")
        tracker.finish()
    return summary


def _log_eval(
    run: Run,
    tracker: Tracker | None,
    result: ev.PanelResult,
    step: int,
    run_wall_seconds: float,
) -> None:
    if tracker is None:
        return
    metrics = {
        "dev_league_mean": result.mean,
        "dev_se": result.standard_error,
        "dev_deals": len(result.deals),
        "active_seconds": run.progress.active_seconds,
        "run_wall_seconds": run_wall_seconds,
    }
    metrics.update({f"dev_{n}_s{s}": v for (n, s), v in result.cell_means.items()})
    metrics.update({f"dev_{k}": v for k, v in result.behavior.as_dict().items()})
    tracker.log(
        metrics,
        step=step,
        mlflow_metrics=_evaluation_mlflow_metrics(metrics, result),
    )


def _record_evaluation(
    run: Run,
    tracker: Tracker | None,
    result: ev.PanelResult,
    step: int,
    run_wall_seconds: float,
) -> bool:
    """Log one evaluation and persist a coherent global best checkpoint.

    Progress owns the selection state so it survives resume. ``eval_history`` is
    retained as the current process segment's diagnostic history; it is not used
    to reconstruct a lineage-wide winner.
    """
    run.eval_history.append((run.progress.update, result.mean))
    _log_eval(run, tracker, result, step, run_wall_seconds)
    improved = (
        run.progress.best_score is None or result.mean > run.progress.best_score + 1e-6
    )
    if improved:
        run.progress.best_score = result.mean
        run.progress.best_checkpoint = "checkpoints/best.pt"
        run.progress.best_update = run.progress.update
        run.progress.best_decisions = run.progress.learner_decisions
        _save(run, "best")
    return improved


def _save(run: Run, label: str) -> dict:
    path = run.directory / "checkpoints" / f"{label}.pt"
    lineage = {
        **run.lineage,
        "league": run.league_name,
        "label": label,
        "run_id": getattr(run.tracker, "run_id", None),
    }
    manifest = ckpt.save(
        path,
        run.config,
        run.actor,
        run.critic,
        run.actor_opt,
        run.critic_opt,
        run.progress,
        lineage,
    )
    ckpt.export_actor(
        run.directory / "checkpoints" / f"{label}_actor.pt",
        run.config,
        run.actor,
        f"rl_{run.config.arm}_seed{run.config.training_seed}_{label}",
        run.progress,
    )
    return manifest


def resume(
    path: str | Path,
    config: Config,
    directory: str | Path,
    league_name: str = "rulebot",
    device: str | None = None,
) -> Run:
    """Continue a lineage from a checkpoint, possibly on another machine.

    Device, worker count and artifact paths may change; anything that changes the
    algorithm is refused by `uno.rl.checkpoint.restore`. The continuation is a new
    local run linked by parent run ID and checkpoint hash, with logical counters
    carried over and segment time recorded separately from cumulative time.
    """
    payload = ckpt.load(path)
    cfg = config if device is None else config.with_(device=device)
    saved_league_hash = payload.get("league_hash")
    expected_league_hash = league_hash(cfg.rs, league_name)
    if saved_league_hash is not None and saved_league_hash != expected_league_hash:
        raise ValueError(
            "checkpoint's league differs: that is an experiment branch, not a resume"
        )
    run = Run(cfg, Path(directory), league_name)
    progress = ckpt.restore(
        payload, cfg, run.actor, run.critic, run.actor_opt, run.critic_opt, cfg.device
    )
    run.progress = progress
    if progress.best_update is None and progress.best_checkpoint is not None:
        try:
            progress.best_update = int(progress.best_checkpoint.removeprefix("update"))
        except ValueError:
            pass
    _carry_best_artifacts(Path(path), run.directory, progress)
    run.schedule = (
        mf.DealSchedule.from_state(cfg.training_seed, progress.deal_schedule)
        if progress.deal_schedule
        else mf.DealSchedule(cfg.training_seed)
    )
    run.lineage = {
        "parent_run_id": payload.get("lineage", {}).get("run_id"),
        "parent_checkpoint": str(path),
        "resumed_at": time.time(),
    }
    return run


def _carry_best_artifacts(
    resumed_checkpoint: Path, destination: Path, progress: ckpt.Progress
) -> None:
    """Keep the lineage's selected artifact available in a resumed run directory."""
    if progress.best_score is None:
        return
    source_dir = resumed_checkpoint.parent
    source_checkpoint = source_dir / "best.pt"
    source_actor = source_dir / "best_actor.pt"
    if not source_checkpoint.exists() and progress.best_update == progress.update:
        source_checkpoint = resumed_checkpoint
        candidate = resumed_checkpoint.with_name(resumed_checkpoint.stem + "_actor.pt")
        source_actor = candidate
    if not source_checkpoint.exists():
        # The score remains valid metadata, but the caller copied no selected
        # artifact. The final summary exposes availability instead of claiming one.
        progress.best_checkpoint = None
        return

    target_dir = destination / "checkpoints"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_checkpoint = target_dir / "best.pt"
    if source_checkpoint.resolve() != target_checkpoint.resolve():
        shutil.copy2(source_checkpoint, target_checkpoint)
    source_manifest = source_checkpoint.with_suffix(
        source_checkpoint.suffix + ".manifest.json"
    )
    if source_manifest.exists():
        target_manifest = target_dir / "best.pt.manifest.json"
        if source_manifest.resolve() != target_manifest.resolve():
            shutil.copy2(source_manifest, target_manifest)
    if source_actor.exists():
        target_actor = target_dir / "best_actor.pt"
        if source_actor.resolve() != target_actor.resolve():
            shutil.copy2(source_actor, target_actor)
    progress.best_checkpoint = "checkpoints/best.pt"


__all__ = [
    "CHECKPOINT_INTERVAL_S",
    "Run",
    "evaluate_now",
    "iteration",
    "resume",
    "start_tracker",
    "train",
]
