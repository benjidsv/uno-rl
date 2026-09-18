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
than hidden. A hard interrupt rolls back to the last valid saved boundary -- this
loop never records a truncation as a terminal result.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

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

    def __post_init__(self) -> None:
        cfg = self.config
        rs = cfg.rs
        self.directory = Path(self.directory)
        torch.manual_seed(torch_seed(cfg.training_seed, INIT))
        self.actor = self.actor or Actor(spec(rs, cfg.arm), cfg.width, cfg.blocks).to(cfg.device)
        self.critic = self.critic or Critic(spec(rs, "CRITIC"), cfg.width, cfg.blocks).to(cfg.device)
        o = cfg.optim
        self.actor_opt = self.actor_opt or torch.optim.Adam(self.actor.parameters(), lr=o.actor_lr, betas=o.betas, eps=o.eps, weight_decay=o.weight_decay)
        self.critic_opt = self.critic_opt or torch.optim.Adam(self.critic.parameters(), lr=o.critic_lr, betas=o.betas, eps=o.eps, weight_decay=o.weight_decay)
        self.schedule = self.schedule or mf.DealSchedule(cfg.training_seed)
        self.opponents = league(rs, self.league_name)
        self.panel = mf.panel(list(self.opponents), cfg.evaluation.deals_per_cell, cfg.evaluation.panel_start)

    @property
    def policy(self) -> ActorPolicy:
        return ActorPolicy(spec(self.config.rs, self.config.arm), self.actor, self.config.optim.temperature)


def start_tracker(run: Run, study_id: str = "pilot", phase: str = "diagnostic", kind: str = "train", store_root=None) -> Tracker:
    cfg = run.config
    tags = {
        "study_id": study_id, "arm": cfg.arm, "phase": phase, "seed": cfg.training_seed,
        "machine": provenance(cfg.hash)["machine"], "kind": kind, "league": run.league_name,
    }  # fmt: skip
    params = {
        **cfg.resolved(), "config_hash": cfg.hash, "stream_version": STREAM_VERSION,
        "league_hash": league_hash(cfg.rs, run.league_name), "actor_schema": spec(cfg.rs, cfg.arm).hash,
        "critic_schema": spec(cfg.rs, "CRITIC").hash, "shapes": str(shapes(spec(cfg.rs, cfg.arm), spec(cfg.rs, "CRITIC"), cfg.width, cfg.blocks)),
        **provenance(cfg.hash),
    }  # fmt: skip
    if run.lineage:  # a resumed lineage records where it came from and what it replays
        params.update({f"lineage_{k}": v for k, v in run.lineage.items()})
    tracker = Tracker(run.directory / "tracking", run_name=f"{cfg.arm}-seed{cfg.training_seed}", tags=tags, params=params, store_root=store_root)
    _write_run_manifest(run, tracker)
    return tracker


def _write_run_manifest(run: Run, tracker: Tracker) -> None:
    """The seed/panel manifest artifact: which deals this run may touch, and which it evaluates on."""
    cfg = run.config
    panel_deals = sorted({e.deal_seed for e in run.panel.episodes})
    payload = {
        "config": cfg.resolved(),
        "config_hash": cfg.hash,
        "stream_version": STREAM_VERSION,
        "league": run.league_name,
        "league_members": list(run.opponents),
        "league_hash": league_hash(cfg.rs, run.league_name),
        "partitions": {"train": [mf.TRAIN_DEALS.start, mf.TRAIN_DEALS.stop], "panel": [mf.PANEL_DEALS.start, mf.PANEL_DEALS.stop], "diagnostic": [mf.DIAGNOSTIC_DEALS.start, mf.DIAGNOSTIC_DEALS.stop]},
        "panel": {"cells": [list(c) for c in run.panel.cells], "deals_per_cell": run.panel.games_per_cell, "first_deal": panel_deals[0], "last_deal": panel_deals[-1]},
        "deal_schedule": run.schedule.state,
        "lineage": run.lineage,
        "provenance": provenance(cfg.hash),
    }
    path = run.directory / "run_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))
    tracker.artifact(path)


def iteration(run: Run, collector=None) -> tuple[ppo.UpdateStats, dict]:
    """One complete synchronous iteration. Returns its update stats and timings."""
    cfg = run.config
    rs = cfg.rs
    critic_spec = spec(rs, "CRITIC")
    t0 = time.perf_counter()

    manifest, next_id = mf.build(run.progress.update, list(run.opponents), run.schedule, cfg.collection.games_per_cell, run.progress.next_episode_id)
    if collector is None:
        rollout, cstats = collect(manifest, run.policy, critic_spec, run.opponents, cfg.training_seed, run.progress.update, cfg.collection.live_games_per_worker)
    else:
        rollout, cstats = collector.collect(manifest, run.actor, critic_spec, run.progress.update)
    collected = time.perf_counter()

    stats = ppo.update(run.actor, run.critic, run.actor_opt, run.critic_opt, rollout, cfg.optim, cfg.training_seed, run.progress.update, cfg.device)
    updated = time.perf_counter()

    run.progress.update += 1
    run.progress.next_episode_id = next_id
    run.progress.games += rollout.n_games
    run.progress.learner_decisions += rollout.n_decisions
    run.progress.engine_decisions += int(rollout.engine_decisions.sum())
    run.progress.optimizer_steps += stats.actor_steps + stats.critic_steps
    run.progress.deal_schedule = run.schedule.state
    timings = {
        "collect_s": collected - t0, "update_s": updated - collected, "iteration_s": updated - t0,
        "collect_decisions_per_s": cstats.decisions_per_second, "mean_inference_batch": cstats.mean_batch,
        "drain_s": cstats.drain_seconds, "rollout_bytes": rollout.nbytes, "retried_bundles": cstats.retried_bundles,
    }  # fmt: skip
    per_cell = _cell_scores(rollout)
    return stats, {**timings, **per_cell, **{f"end_{k}": v for k, v in cstats.ends.items()}}


def _cell_scores(rollout) -> dict:
    """Training score per opponent x seat cell: a diagnostic, not the selection metric."""
    out = {}
    scores = (rollout.terminal_return + 1.0) / 2.0
    for name in sorted(set(rollout.opponent)):
        for seat in (0, 1):
            m = np.asarray([o == name for o in rollout.opponent]) & (rollout.seat == seat)
            if m.any():
                out[f"train_score_{name}_s{seat}"] = float(scores[m].mean())
    out["train_score"] = float(np.mean([v for k, v in out.items() if k.startswith("train_score_")])) if out else 0.0
    return out


def evaluate_now(run: Run) -> ev.PanelResult:
    """Synchronous scheduled DEV evaluation of the current actor on the fixed panel."""
    return ev.play_panel(run.panel, run.policy, run.opponents, run.config.training_seed)


def train(run: Run, tracker: Tracker | None = None, collector=None, verbose: bool = True) -> dict:
    """Run to the declared data or time cap, evaluating and checkpointing on schedule."""
    cfg = run.config
    tracker = tracker or run.tracker
    started = time.perf_counter()
    last_save = started
    eval_every = max(1, int(cfg.data_cap * cfg.evaluation.every_fraction))
    if run.progress.next_eval_at == 0:
        run.progress.next_eval_at = eval_every
        first = evaluate_now(run)  # the initial policy is a scheduled evaluation point
        run.eval_history.append((0, first.mean))
        _log_eval(run, tracker, first, 0)
    overrun = 0.0
    iterations = []
    stop_reason = "data_cap"

    while run.progress.learner_decisions < cfg.data_cap:
        if cfg.time_cap_s is not None:
            elapsed = time.perf_counter() - started
            budget = float(np.mean(iterations[-3:])) if iterations else 0.0
            if iterations and elapsed + budget > cfg.time_cap_s:
                stop_reason = "time_cap"
                break
        t0 = time.perf_counter()
        stats, extra = iteration(run, collector)
        iterations.append(time.perf_counter() - t0)
        run.progress.active_seconds += iterations[-1]
        metrics = {
            "update": run.progress.update, "games": run.progress.games, "engine_decisions": run.progress.engine_decisions,
            "optimizer_steps": run.progress.optimizer_steps, "active_seconds": run.progress.active_seconds,
            "surrogate": stats.surrogate, "critic_mse": stats.critic_mse, "explained_variance": stats.explained_variance,
            "entropy": stats.entropy, "approx_kl": stats.approx_kl, "clip_fraction": stats.clip_fraction,
            "max_abs_log_ratio": stats.max_abs_log_ratio, "actor_grad_norm": stats.actor_grad_norm,
            "actor_clipped_fraction": stats.actor_clipped_fraction, "critic_grad_norm": stats.critic_grad_norm,
            "advantage_mean": stats.advantage_mean, "advantage_std": stats.advantage_std,
            "return_mean": stats.return_mean, "return_std": stats.return_std,
            "actor_epochs": stats.actor_epochs, "critic_epochs": stats.critic_epochs, "early_stopped": int(stats.early_stopped),
            **extra,
        }  # fmt: skip
        if tracker:
            tracker.log(metrics, step=run.progress.learner_decisions)
        if verbose:
            print(f"   update {run.progress.update:4d}  {run.progress.learner_decisions:>9,} dec  {stats}  train {extra.get('train_score', float('nan')):.3f}")

        if run.progress.learner_decisions >= run.progress.next_eval_at:
            result = evaluate_now(run)
            run.eval_history.append((run.progress.update, result.mean))
            _log_eval(run, tracker, result, run.progress.learner_decisions)
            if run.progress.best_score is None or result.mean > run.progress.best_score + 1e-6:
                run.progress.best_score = result.mean
                run.progress.best_checkpoint = f"update{run.progress.update}"
                _save(run, "best")
            while run.progress.next_eval_at <= run.progress.learner_decisions:
                run.progress.next_eval_at += eval_every
            _save(run, "latest")
            last_save = time.perf_counter()
        elif time.perf_counter() - last_save > CHECKPOINT_INTERVAL_S:
            _save(run, "latest")
            last_save = time.perf_counter()

    if cfg.time_cap_s is not None and time.perf_counter() - started > cfg.time_cap_s:
        overrun = time.perf_counter() - started - cfg.time_cap_s
    final = evaluate_now(run)
    run.eval_history.append((run.progress.update, final.mean))
    _log_eval(run, tracker, final, run.progress.learner_decisions)
    _save(run, "final")
    best_update, best_score = ev.select(run.eval_history)
    summary = {
        "stop_reason": stop_reason,
        "overrun_s": overrun,
        "updates": run.progress.update,
        "learner_decisions": run.progress.learner_decisions,
        "games": run.progress.games,
        "engine_decisions": run.progress.engine_decisions,
        "active_seconds": run.progress.active_seconds,
        "selected_update": best_update,
        "selected_score": best_score,
        "final_score": final.mean,
        "final_ci95": final.ci95,
        "behavior": final.behavior.as_dict(),
        "cells": {f"{n}_s{s}": v for (n, s), v in final.cell_means.items()},
    }
    (run.directory / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    if tracker:
        tracker.artifact(run.directory / "summary.json")
        tracker.finish()
    return summary


def _log_eval(run: Run, tracker: Tracker | None, result: ev.PanelResult, step: int) -> None:
    if tracker is None:
        return
    metrics = {"dev_league_mean": result.mean, "dev_se": result.standard_error, "dev_deals": len(result.deals)}
    metrics.update({f"dev_{n}_s{s}": v for (n, s), v in result.cell_means.items()})
    metrics.update({f"dev_{k}": v for k, v in result.behavior.as_dict().items()})
    tracker.log(metrics, step=step)


def _save(run: Run, label: str) -> dict:
    path = run.directory / "checkpoints" / f"{label}.pt"
    lineage = {**run.lineage, "league": run.league_name, "label": label, "run_id": getattr(run.tracker, "run_id", None)}
    manifest = ckpt.save(path, run.config, run.actor, run.critic, run.actor_opt, run.critic_opt, run.progress, lineage)
    ckpt.export_actor(run.directory / "checkpoints" / f"{label}_actor.pt", run.config, run.actor, f"rl_{run.config.arm}_seed{run.config.training_seed}_{label}", run.progress)
    return manifest


def resume(path: str | Path, config: Config, directory: str | Path, league_name: str = "rulebot", device: str | None = None) -> Run:
    """Continue a lineage from a checkpoint, possibly on another machine.

    Device, worker count and artifact paths may change; anything that changes the
    algorithm is refused by `uno.rl.checkpoint.restore`. The continuation is a new
    local run linked by parent run ID and checkpoint hash, with logical counters
    carried over and segment time recorded separately from cumulative time.
    """
    payload = ckpt.load(path)
    cfg = config if device is None else config.with_(device=device)
    run = Run(cfg, Path(directory), league_name)
    progress = ckpt.restore(payload, cfg, run.actor, run.critic, run.actor_opt, run.critic_opt, cfg.device)
    run.progress = progress
    run.schedule = mf.DealSchedule.from_state(cfg.training_seed, progress.deal_schedule) if progress.deal_schedule else mf.DealSchedule(cfg.training_seed)
    run.lineage = {"parent_run_id": payload.get("lineage", {}).get("run_id"), "parent_checkpoint": str(path), "resumed_at": time.time()}
    return run


__all__ = ["CHECKPOINT_INTERVAL_S", "Run", "evaluate_now", "iteration", "resume", "start_tracker", "train"]
