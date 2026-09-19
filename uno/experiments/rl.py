"""Run, resume or inspect a PPO training run (docs/rl/pipeline-plan.md stages 1-4).

`uv run uno rl shapes [RULESET] [ARM]`            recorded tensor shapes and parameter counts
`uv run uno rl smoke [RULESET] [ARM] [GAMES/CELL]` one bounded iteration end to end
`uv run uno rl train [RULESET] [ARM] [SEED] [DECISIONS] [WORKERS] [GAMES/CELL] [DEV DEALS/CELL] [DEVICE] [COLLECTION_THREADS] [UPDATE_THREADS] [MINIBATCH] [TIME_CAP_S] [EVALUATION_FRACTION]`
`uv run uno rl resume CHECKPOINT [RULESET] [ARM] [SEED] [DECISIONS] [DEVICE] [WORKERS] [LIVE] [COLLECTION_THREADS] [UPDATE_THREADS] [TIME_CAP_S] [EVALUATION_FRACTION]`
`uv run uno rl systems [RULESET] [ARM] [WORKERS] [LIVE] [DEVICE] [GAMES/CELL] [SAMPLES] [MINIBATCH] [COLLECTION_THREADS] [UPDATE_THREADS]`
`uv run uno rl devsize [RULESET] [ARM] [SEED] [MAX_DEALS/CELL] [TARGET_95_HALFWIDTH] [TORCH_THREADS] [ACTOR_EXPORT]`
`uv run uno rl pilot [SEED]`                     one-hour full-UNO B pilot preset

`train` is a **diagnostic** command. It runs the synchronous loop of pipeline plan
section 3 to a declared learner-decision cap, evaluating on the fixed DEV panel at
every 10% of that cap, checkpointing at update boundaries and logging to a local
MLflow run (SQLite plus a local artifact directory under `logs/rl/`, both
gitignored) and to a local CSV. A tracking failure never stops training.

No result this command produces is a frozen claim: the DEV panel is DEV, the
configuration is the plan's *proposed first diagnostic*, and the systems settings
(workers, live games, device) are unmeasured until the stage 6 pilot runs. Choose
the run's budget explicitly; nothing here picks it from a proxy rate.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import re
import resource
import subprocess
import threading
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from uno.agents.baseline import Heuristic
from uno.game.rules import NAMED
from uno.progress import bar
from uno.rl import checkpoint as ckpt
from uno.rl import evaluate as ev
from uno.rl import manifest as mf
from uno.rl.config import Collection, Config, Evaluation, Optim
from uno.rl.features import ARM_A, ARM_B, ARM_CRITIC, spec
from uno.rl.model import shapes
from uno.rl.opponents import league, league_hash
from uno.rl.parallel import ParallelCollector, WorkerConfig
from uno.rl.rollout import estimate_bytes
from uno.rl.train import Run, iteration, resume, start_tracker, train
from uno.rl.track import provenance

TOY_LEAGUES = {
    "TOY": "toy",
    "TOY_FULL": "toy",
    "TOY2": "toy",
    "TOY2_SKIP": "toy",
    "TOY2_FULL": "toy",
    "TOY_CANDIDATE": "toy",
}


def _league_for(ruleset: str) -> str:
    return TOY_LEAGUES.get(ruleset, "rulebot")


def _config(
    ruleset: str,
    arm: str,
    seed: int,
    cap: int,
    workers: int,
    games_per_cell: int,
    deals_per_cell: int,
    device: str,
    live_games: int = 64,
    minibatch: int = 512,
    evaluation_fraction: float = 0.1,
) -> Config:
    return Config(
        ruleset=ruleset, arm=arm, training_seed=seed, data_cap=cap, device=device,
        collection=Collection(games_per_cell=games_per_cell, live_games_per_worker=live_games, workers=workers),
        evaluation=Evaluation(deals_per_cell=deals_per_cell, every_fraction=evaluation_fraction),
        optim=replace(Optim(), minibatch=minibatch),
    )  # fmt: skip


def cmd_shapes(ruleset: str = "FULL_UNO", arm: str = ARM_B) -> None:
    rs = NAMED[ruleset]
    sp, spc = spec(rs, arm), spec(rs, ARM_CRITIC)
    sh = shapes(sp, spc)
    print(f"\n== {ruleset} arm {arm}")
    print(f"   {sh}")
    print(
        f"   actor schema {sp.hash}, critic schema {spc.hash}, league {league_hash(rs, _league_for(ruleset))}"
    )
    other = ARM_A if arm == ARM_B else ARM_B
    alt = shapes(spec(rs, other), spc)
    print(
        f"   {other}: {alt.actor_params:,} actor params ({sh.actor_params - alt.actor_params:+,} against {arm}; the arms share the position encoder's shape, not its width)"
    )
    for decisions in (8192, 65536):
        mb = estimate_bytes(sp, spc, decisions, decisions // 4) / 2**20
        print(
            f"   a rollout of {decisions:,} decisions is about {mb:,.0f} MiB before allocation"
        )


def cmd_smoke(ruleset: str = "TOY", arm: str = ARM_B, games_per_cell: int = 8) -> None:
    """One iteration end to end: collection, update, counters. Not a learning claim."""
    cfg = _config(ruleset, arm, 0, 10**9, 1, games_per_cell, 16, "cpu")
    out = Path("logs/rl") / f"smoke_{ruleset}_{int(time.time())}"
    run = Run(cfg, out, _league_for(ruleset))
    t0 = time.perf_counter()
    stats, extra = iteration(run)
    print(
        f"\n== smoke {ruleset} arm {arm}: {run.progress.games} games, {run.progress.learner_decisions:,} learner decisions in {time.perf_counter() - t0:.1f}s"
    )
    print(f"   {stats}")
    print(
        f"   collect {extra['collect_s']:.2f}s ({extra['collect_decisions_per_s']:,.0f} learner dec/s, mean batch {extra['mean_inference_batch']:.1f}), "
        f"update {extra['update_s']:.2f}s, rollout {extra['rollout_bytes'] / 2**20:.1f} MiB"
    )
    print(
        f"   training score per cell: "
        + " ".join(
            f"{k.replace('train_score_', '')}={v:.3f}"
            for k, v in extra.items()
            if k.startswith("train_score_")
        )
    )


def cmd_train(
    ruleset: str = "TOY",
    arm: str = ARM_B,
    seed: int = 0,
    cap: int = 100_000,
    workers: int = 1,
    games_per_cell: int = 64,
    deals_per_cell: int = 256,
    device: str = "cpu",
    collection_threads: int = 0,
    update_threads: int = 0,
    minibatch: int = 512,
    time_cap_s: float = 0.0,
    evaluation_fraction: float = 0.1,
    *,
    tracking_study_id: str = "pilot",
    tracking_kind: str = "train",
) -> None:
    if collection_threads < 0 or update_threads < 0:
        raise SystemExit(
            "collection and update threads may be zero for the current default, or positive"
        )
    if minibatch < 1 or time_cap_s < 0 or not 0 < evaluation_fraction <= 1:
        raise SystemExit(
            "minibatch must be positive, time cap nonnegative, and evaluation fraction in (0, 1]"
        )
    if device == "mps":
        import torch

        if not torch.backends.mps.is_available():
            raise SystemExit("MPS was requested but is unavailable")
    cfg = _config(
        ruleset,
        arm,
        seed,
        cap,
        workers,
        games_per_cell,
        deals_per_cell,
        device,
        minibatch=minibatch,
        evaluation_fraction=evaluation_fraction,
    )
    if time_cap_s:
        cfg = cfg.with_(time_cap_s=time_cap_s)
    out = (
        Path("logs/rl")
        / f"{ruleset}_{arm}_seed{seed}_mb{minibatch}_{device}_{int(time.time())}"
    )
    run = Run(cfg, out, _league_for(ruleset))
    run.execution = {
        **(
            {"collection_torch_threads": collection_threads}
            if collection_threads
            else {}
        ),
        **({"update_torch_threads": update_threads} if update_threads else {}),
    }
    run.tracker = start_tracker(run, study_id=tracking_study_id, kind=tracking_kind)
    print(
        f"\n== train {ruleset} arm {arm} seed {seed}: cap {cap:,} learner decisions, {len(run.opponents)} opponents, "
        f"{cfg.collection.games_per_cell} games/cell, minibatch {minibatch}, device {device}, "
        f"time cap {time_cap_s or 'disabled'}s, DEV every {100 * evaluation_fraction:g}% of the decision cap"
    )
    print(
        f"   {shapes(spec(cfg.rs, arm), spec(cfg.rs, ARM_CRITIC), cfg.width, cfg.blocks)}"
    )
    print(f"   config {cfg.hash}; artifacts in {out}")
    if not run.tracker.available:
        print(f"   note: {run.tracker.unavailable_reason}")
    else:
        print(f"   MLflow run {run.tracker.run_id} in experiment uno-v5-rl")
    collector = None
    if workers > 1:
        wc = WorkerConfig(
            ruleset=ruleset,
            arm=arm,
            width=cfg.width,
            blocks=cfg.blocks,
            league=run.league_name,
            training_seed=seed,
            live_games=cfg.collection.live_games_per_worker,
        )
        collector = ParallelCollector(wc, workers)
    try:
        summary = train(run, run.tracker, collector)
    finally:
        if collector is not None:
            collector.close()
    print(
        f"\n   stop: {summary['stop_reason']}, {summary['updates']} updates, {summary['learner_decisions']:,} learner decisions, "
        f"{summary['active_seconds']:.0f}s active, {summary['wall_seconds']:.0f}s wall"
    )
    print(
        f"   DEV-selected update {summary['selected_update']} at {summary['selected_score']:.4f}; final {summary['final_score']:.4f} "
        f"[{summary['final_ci95'][0]:.4f}, {summary['final_ci95'][1]:.4f}] (95% CI, DEV panel: descriptive, not confirmatory)"
    )
    print(f"   {json.dumps(summary['cells'])}")


def cmd_pilot(seed: int = 0) -> None:
    """The frozen first one-hour full-UNO B learning pilot."""
    cmd_train(
        ruleset="FULL_UNO",
        arm=ARM_B,
        seed=seed,
        cap=500_000,
        workers=1,
        games_per_cell=64,
        deals_per_cell=256,
        device="cpu",
        collection_threads=1,
        update_threads=4,
        minibatch=512,
        time_cap_s=3_600,
        evaluation_fraction=0.25,
        tracking_study_id="stage7-full-uno-b-pilot",
        tracking_kind="pilot",
    )


def _rss_mib(pids: list[int]) -> dict[str, float | None]:
    """Resident memory at an iteration boundary; macOS has no /proc dependency."""
    out = {"coordinator": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20}
    for pid in pids:
        try:
            # `ps` reports KiB on macOS. This is not a physical-memory total:
            # shared pages may be counted for each worker on unified memory.
            kib = int(
                subprocess.check_output(
                    ["ps", "-o", "rss=", "-p", str(pid)], text=True
                ).strip()
            )
            out[str(pid)] = kib / 1024
        except ValueError, subprocess.CalledProcessError:
            out[str(pid)] = None
    return out


def _current_rss_mib(pids: list[int]) -> dict[int, float]:
    """Current per-process RSS from ps; missing/exited processes are omitted."""
    if not pids:
        return {}
    try:
        output = subprocess.check_output(
            ["ps", "-o", "pid=,rss=", "-p", ",".join(map(str, pids))],
            text=True,
        )
    except OSError, subprocess.CalledProcessError:
        return {}
    out = {}
    for line in output.splitlines():
        fields = line.split()
        if len(fields) == 2:
            out[int(fields[0])] = int(fields[1]) / 1024
    return out


def _macos_memory_pressure() -> tuple[float | None, float | None]:
    """System free percentage and swap used MiB, when macOS exposes them."""
    if platform.system() != "Darwin":
        return None, None
    free, swap = None, None
    try:
        output = subprocess.check_output(["memory_pressure", "-Q"], text=True)
        match = re.search(r"free percentage:\s*([0-9.]+)%", output)
        if match:
            free = float(match.group(1))
    except OSError, subprocess.CalledProcessError:
        pass
    try:
        output = subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True)
        match = re.search(r"used = ([0-9.]+)([MG])", output)
        if match:
            swap = float(match.group(1)) * (1024 if match.group(2) == "G" else 1)
    except OSError, subprocess.CalledProcessError:
        pass
    return free, swap


class _MemoryMonitor:
    """Low-rate whole-process RSS and macOS pressure sampling during an iteration."""

    def __init__(self, pids: list[int], interval_s: float = 1.0) -> None:
        self.pids = pids
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.peak_by_pid: dict[int, float] = {}
        self.peak_aggregate_mib = 0.0
        self.minimum_free_percent: float | None = None
        self.minimum_swap_used_mib: float | None = None
        self.maximum_swap_used_mib: float | None = None

    def _sample(self) -> None:
        rss = _current_rss_mib(self.pids)
        for pid, mib in rss.items():
            self.peak_by_pid[pid] = max(self.peak_by_pid.get(pid, 0.0), mib)
        self.peak_aggregate_mib = max(self.peak_aggregate_mib, sum(rss.values()))
        free, swap = _macos_memory_pressure()
        if free is not None:
            self.minimum_free_percent = (
                free
                if self.minimum_free_percent is None
                else min(self.minimum_free_percent, free)
            )
        if swap is not None:
            self.minimum_swap_used_mib = (
                swap
                if self.minimum_swap_used_mib is None
                else min(self.minimum_swap_used_mib, swap)
            )
            self.maximum_swap_used_mib = (
                swap
                if self.maximum_swap_used_mib is None
                else max(self.maximum_swap_used_mib, swap)
            )

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        self._thread.start()

    def finish(self) -> dict:
        self._stop.set()
        self._thread.join()
        self._sample()
        return {
            "peak_rss_mib_by_pid": {str(k): v for k, v in self.peak_by_pid.items()},
            "peak_aggregate_rss_mib": self.peak_aggregate_mib,
            "minimum_system_free_percent": self.minimum_free_percent,
            "minimum_swap_used_mib": self.minimum_swap_used_mib,
            "maximum_swap_used_mib": self.maximum_swap_used_mib,
            "swap_growth_mib": None
            if self.minimum_swap_used_mib is None
            else self.maximum_swap_used_mib - self.minimum_swap_used_mib,
        }


def cmd_systems(
    ruleset: str = "FULL_UNO",
    arm: str = ARM_B,
    workers: int = 1,
    live_games: int = 64,
    device: str = "cpu",
    games_per_cell: int = 16,
    samples: int = 1,
    minibatch: int = 512,
    collection_threads: int = 1,
    update_threads: int = 0,
) -> None:
    """One bounded Stage-6 systems point; no DEV evaluation or learning claim.

    The warm-up and each retained sample use the same initial policy and logical
    manifest. PPO still runs because collection-only throughput cannot set an
    end-to-end run budget. A parallel pool remains alive across samples, so its
    process startup cost is deliberately excluded.
    """
    if device == "mps":
        import torch

        if not torch.backends.mps.is_available():
            raise SystemExit("MPS was requested but is unavailable")
    if (
        min(workers, live_games, games_per_cell, samples, minibatch, collection_threads)
        < 1
        or update_threads < 0
    ):
        raise SystemExit(
            "workers, live games, games/cell, samples, minibatch and collection threads must be positive; update threads may be zero for the current default"
        )
    resolved_update_threads = None if update_threads == 0 else update_threads
    cfg = _config(
        ruleset,
        arm,
        0,
        10**9,
        workers,
        games_per_cell,
        1,
        device,
        live_games,
        minibatch,
    )
    out = (
        Path("logs/rl")
        / f"systems_{ruleset}_{arm}_w{workers}_l{live_games}_{device}_mb{minibatch}_ct{collection_threads}_ut{update_threads or 'default'}_{int(time.time())}"
    )
    collector = None
    if workers > 1:
        wc = WorkerConfig(
            ruleset=ruleset,
            arm=arm,
            width=cfg.width,
            blocks=cfg.blocks,
            league=_league_for(ruleset),
            training_seed=0,
            live_games=live_games,
        )
        collector = ParallelCollector(wc, workers)
    try:
        iteration(
            Run(cfg, out / "warmup", _league_for(ruleset)),
            collector,
            collection_threads=collection_threads,
            update_threads=resolved_update_threads,
        )
        rows = []
        for n in range(samples):
            run = Run(cfg, out / f"sample{n}", _league_for(ruleset))
            pids = (
                []
                if collector is None
                else [p.pid for p in collector._procs if p.pid is not None]
            )
            memory = _MemoryMonitor([os.getpid(), *pids])
            memory.start()
            try:
                stats, extra = iteration(
                    run,
                    collector,
                    collection_threads=collection_threads,
                    update_threads=resolved_update_threads,
                )
            finally:
                memory_summary = memory.finish()
            elapsed = extra["iteration_s"]
            rows.append({
                "sample": n,
                "learner_decisions": run.progress.learner_decisions,
                "games": run.progress.games,
                "engine_decisions": run.progress.engine_decisions,
                "iteration_s": elapsed,
                "collection_s": extra["collect_s"],
                "update_s": extra["update_s"],
                "learner_decisions_per_s": run.progress.learner_decisions / elapsed,
                "update_fresh_decisions_per_s": extra["update_fresh_decisions_per_s"],
                "actor_presentations": extra["actor_presentations"],
                "critic_presentations": extra["critic_presentations"],
                "actor_presentations_per_s": extra["actor_presentations_per_s"],
                "critic_presentations_per_s": extra["critic_presentations_per_s"],
                "actor_steps": extra["actor_steps"],
                "critic_steps": extra["critic_steps"],
                "games_per_s": run.progress.games / elapsed,
                "engine_decisions_per_s": run.progress.engine_decisions / elapsed,
                "collection_learner_decisions_per_s": extra["collect_decisions_per_s"],
                "mean_inference_batch": extra["mean_inference_batch"],
                "inference_batch_distribution": {
                    "calls": extra["inference_batch_calls"],
                    "min": extra["inference_batch_min"],
                    "p50": extra["inference_batch_p50"],
                    "p90": extra["inference_batch_p90"],
                    "max": extra["inference_batch_max"],
                },
                "drain_s": extra["drain_s"],
                "rollout_mib": extra["rollout_bytes"] / 2**20,
                "actor_epochs": stats.actor_epochs,
                "critic_epochs": stats.critic_epochs,
                "rss_mib": _rss_mib(pids),
                "memory": memory_summary,
                "worker_stage_s": {
                    "simulation": extra["simulation_s"],
                    "features": extra["feature_s"],
                    "actor_inference": extra["actor_inference_s"],
                    "assembly": extra["worker_assembly_s"],
                },
                "transfer": {
                    "serialization_s": extra["serialization_s"],
                    "result_queue_put_s": extra["result_queue_put_s"],
                    "result_queue_wait_s": extra["result_queue_wait_s"],
                    "deserialization_s": extra["deserialization_s"],
                    "coordinator_assembly_s": extra["coordinator_assembly_s"],
                    "bytes": extra["transfer_bytes"],
                },
                "update_stage_s": {
                    "critic_values": extra["critic_values_s"],
                    "actor": extra["actor_update_s"],
                    "critic": extra["critic_update_s"],
                },
                "torch_threads": {
                    "collection": extra["collection_torch_threads"],
                    "update": extra["update_torch_threads"],
                },
            })
    finally:
        if collector is not None:
            collector.close()
    keys = (
        "iteration_s",
        "collection_s",
        "update_s",
        "learner_decisions_per_s",
        "update_fresh_decisions_per_s",
        "actor_presentations_per_s",
        "critic_presentations_per_s",
        "actor_steps",
        "critic_steps",
        "games_per_s",
        "engine_decisions_per_s",
        "collection_learner_decisions_per_s",
        "mean_inference_batch",
        "drain_s",
        "rollout_mib",
    )
    payload = {
        "purpose": "Bounded systems calibration; throughput only, not a learning result",
        "machine": platform.platform(),
        "architecture": platform.machine(),
        "pid": os.getpid(),
        "config": cfg.resolved(),
        "config_hash": cfg.hash,
        "warmup_excluded": True,
        "execution": {
            "collection_torch_threads": collection_threads,
            "update_torch_threads": update_threads or "default",
            "device_synchronized_for_timing": device in {"mps", "cuda"}
            or device.startswith("cuda:"),
        },
        "collector_execution": {
            "task_bundle_games": None if collector is None else collector.bundle,
            "maximum_live_games_per_worker_task": cfg.collection.live_games_per_worker
            if collector is None
            else min(cfg.collection.live_games_per_worker, collector.bundle),
            "note": "The parallel collector still collects each task independently; use the measured inference-batch distribution, not the requested live_games value, as the realized width.",
        },
        "samples": rows,
        "mean": {k: sum(r[k] for r in rows) / len(rows) for k in keys},
        "memory_note": "Aggregate RSS sums process RSS and can double-count shared pages; on unified memory it is not physical usage. System pressure and swap provide the whole-machine context.",
        "timing_note": "Worker stage times sum overlapping workers; result_queue_wait_s is coordinator blocked time and includes useful worker computation.",
        "presentation_note": "Actor/critic presentations count fresh rollout rows times completed epochs. They exclude the extra no-gradient KL pass and critic baseline evaluation, so they are a stable workload proxy rather than all model forwards.",
    }
    out.mkdir(parents=True, exist_ok=True)
    path = out / "systems.json"
    path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print(f"\n   saved {path}")


def _panel_sizes(max_deals: int) -> tuple[int, ...]:
    """Nested prefixes concentrated near an explicitly requested maximum.

    Once a target panel size has been selected, nearby 80% and 90% prefixes are
    more useful for checking its precision boundary than the old quarter/half
    prefixes. Small panels may collapse duplicate rounded sizes, which is fine.
    """
    return tuple(
        sorted({
            min(max_deals, max(2, round(max_deals * fraction)))
            for fraction in (0.7, 0.85, 1.0)
        })
    )


def _prefix(result: ev.PanelResult, deals: int) -> ev.PanelResult:
    """The first `deals` from one already evaluated nested panel."""
    if deals < 2 or deals > len(result.deals):
        raise ValueError(
            "a panel prefix needs between 2 and the evaluated number of deals"
        )
    return ev.PanelResult(
        result.cells,
        {cell: scores[:deals] for cell, scores in result.scores.items()},
        result.deals[:deals],
    )


def _estimate(values: np.ndarray) -> dict[str, float | int | list[float]]:
    mean = float(values.mean())
    se = float(values.std(ddof=1) / math.sqrt(len(values)))
    half = 1.96 * se
    return {
        "n": len(values),
        "mean": mean,
        "standard_error": se,
        "ci95_halfwidth": half,
        "ci95": [mean - half, mean + half],
    }


def _sizing_row(
    actor: ev.PanelResult, reference: ev.PanelResult, deals: int, target: float
) -> dict:
    a, b = _prefix(actor, deals), _prefix(reference, deals)
    difference = ev.paired(a, b)
    half = 1.96 * difference.standard_error
    required = max(2, math.ceil(deals * (half / target) ** 2))
    cells = {}
    for cell in a.cells:
        delta = a.scores[cell] - b.scores[cell]
        cells[f"{cell[0]}_s{cell[1]}"] = {
            "actor_mean": float(a.scores[cell].mean()),
            "reference_mean": float(b.scores[cell].mean()),
            "difference": _estimate(delta),
        }
    return {
        "deals_per_cell": deals,
        "actor": _estimate(a.per_deal_mean),
        "reference": _estimate(b.per_deal_mean),
        "paired_difference": {
            "n": difference.n,
            "mean": difference.difference,
            "standard_error": difference.standard_error,
            "ci95_halfwidth": half,
            "ci95": list(difference.ci95),
        },
        "estimated_deals_for_target": required,
        "meets_target": half <= target,
        "cells": cells,
    }


def _write_panel_scores(
    path: Path, actor: ev.PanelResult, reference: ev.PanelResult
) -> None:
    fields = ["deal_seed"]
    for label in ("actor", "heuristic"):
        fields.extend(f"{label}_{name}_s{seat}" for name, seat in actor.cells)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, deal in enumerate(actor.deals):
            row = {"deal_seed": int(deal)}
            for cell in actor.cells:
                suffix = f"{cell[0]}_s{cell[1]}"
                row[f"actor_{suffix}"] = float(actor.scores[cell][i])
                row[f"heuristic_{suffix}"] = float(reference.scores[cell][i])
            writer.writerow(row)


def _file_checksum(path: Path) -> str:
    digest = hashlib.blake2b(digest_size=16)
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cmd_devsize(
    ruleset: str = "FULL_UNO",
    arm: str = ARM_B,
    seed: int = 0,
    max_deals: int = 256,
    target_halfwidth: float = 0.005,
    torch_threads: int = 1,
    actor_export: str = "",
) -> None:
    """Size the scheduled DEV panel from a separate nested diagnostic panel."""
    if max_deals < 2 or target_halfwidth <= 0 or torch_threads < 1:
        raise SystemExit(
            "max deals must be at least 2, target half-width positive, and Torch threads positive"
        )
    rs = NAMED[ruleset]
    league_name = _league_for(ruleset)
    opponents = league(rs, league_name)
    manifest = mf.diagnostic_panel(list(opponents), max_deals)
    out = Path("logs/rl") / f"devsize_{ruleset}_{arm}_seed{seed}_{int(time.time())}"
    out.mkdir(parents=True, exist_ok=True)

    if actor_export:
        actor_path = Path(actor_export)
        agent, meta = ckpt.load_actor(actor_path, rs)
        if meta["arm"] != arm:
            raise SystemExit(
                f"actor export uses arm {meta['arm']}, not requested arm {arm}"
            )
        policy = agent.actor
        source = {
            "kind": "actor_export",
            "path": str(actor_path.resolve()),
            "checksum": _file_checksum(actor_path),
            "name": meta["name"],
            "update": meta["update"],
            "trained_decisions": meta["trained_decisions"],
            "actor_schema": meta["actor_schema"],
        }
        config_hash = "actor-export"
    else:
        cfg = _config(ruleset, arm, seed, 1, 1, 1, max_deals, "cpu")
        run = Run(cfg, out, league_name)
        policy = run.policy
        source = {
            "kind": "initialized",
            "training_seed": seed,
            "config_hash": cfg.hash,
            "actor_schema": spec(rs, arm).hash,
        }
        config_hash = cfg.hash

    print(
        f"\n== DEV panel sizing: {ruleset} arm {arm}, diagnostic deals {mf.DIAGNOSTIC_DEALS.start:,}..{mf.DIAGNOSTIC_DEALS.start + max_deals - 1:,}"
    )
    print(
        f"   {len(manifest.cells)} opponent/seat cells x {max_deals} deals; target paired 95% half-width {target_halfwidth:.4f}"
    )
    started = time.perf_counter()
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(torch_threads)
        t0 = time.perf_counter()
        with bar(
            total=len(manifest), desc="actor DEV sizing", unit="games"
        ) as progress:
            actor_result = ev.play_panel(
                manifest, policy, opponents, seed, on_games=progress.update
            )
        actor_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        with bar(
            total=len(manifest), desc="heuristic DEV sizing", unit="games"
        ) as progress:
            reference = ev.baseline_panel(
                manifest, Heuristic(rs), opponents, rs, seed, on_games=progress.update
            )
        reference_seconds = time.perf_counter() - t0
    finally:
        if torch.get_num_threads() != previous_threads:
            torch.set_num_threads(previous_threads)

    sizes = _panel_sizes(max_deals)
    rows = [_sizing_row(actor_result, reference, n, target_halfwidth) for n in sizes]
    observed = next((r["deals_per_cell"] for r in rows if r["meets_target"]), None)
    payload = {
        "purpose": "DEV panel sizing; diagnostic data only, not a learner-strength claim",
        "protocol": {
            "ruleset": ruleset,
            "arm": arm,
            "training_seed": seed,
            "league": league_name,
            "league_hash": league_hash(rs, league_name),
            "reference": "heuristic",
            "cells": [list(c) for c in manifest.cells],
            "diagnostic_partition": [
                mf.DIAGNOSTIC_DEALS.start,
                mf.DIAGNOSTIC_DEALS.stop,
            ],
            "first_deal": int(actor_result.deals[0]),
            "last_deal": int(actor_result.deals[-1]),
            "nested_sizes": list(sizes),
            "target_ci95_halfwidth": target_halfwidth,
            "torch_threads": torch_threads,
        },
        "policy_source": source,
        "timing": {
            "actor_s": actor_seconds,
            "reference_s": reference_seconds,
            "total_s": time.perf_counter() - started,
            "games": 2 * len(manifest),
        },
        "rows": rows,
        "decision": {
            "smallest_observed_size_meeting_target": observed,
            "estimated_deals_from_max_panel": rows[-1]["estimated_deals_for_target"],
            "note": f"Validate the chosen size on the later trained pilot checkpoint; this {source['kind'].replace('_', ' ')} diagnostic does not guarantee its variance.",
        },
        "behavior_at_max_panel": {
            "actor": actor_result.behavior.as_dict(),
            "reference": reference.behavior.as_dict(),
        },
        "provenance": provenance(config_hash),
    }
    scores_path = out / "per_deal_scores.csv"
    result_path = out / "dev_panel_sizing.json"
    _write_panel_scores(scores_path, actor_result, reference)
    result_path.write_text(json.dumps(payload, indent=2, default=str))

    print("\n   deals   paired difference   95% half-width   estimated deals   target")
    for row in rows:
        paired = row["paired_difference"]
        print(
            f"   {row['deals_per_cell']:5d}   {paired['mean']:+.4f}              "
            f"{paired['ci95_halfwidth']:.4f}          {row['estimated_deals_for_target']:7d}   "
            f"{'met' if row['meets_target'] else 'not met'}"
        )
    print(f"\n   actor {actor_seconds:.1f}s; heuristic {reference_seconds:.1f}s")
    if observed is None:
        print(
            f"   target not observed; max-panel estimate is {rows[-1]['estimated_deals_for_target']} deals/cell"
        )
    else:
        print(f"   smallest tested size meeting the target: {observed} deals/cell")
    print(f"   saved {result_path}")
    print(f"   saved {scores_path}")


def cmd_resume(
    checkpoint: str,
    ruleset: str = "TOY",
    arm: str = ARM_B,
    seed: int = 0,
    cap: int = 100_000,
    device: str = "cpu",
    workers: int = 1,
    live_games: int = 64,
    collection_threads: int = 0,
    update_threads: int = 0,
    time_cap_s: float = 0.0,
    evaluation_fraction: float = 0.1,
) -> None:
    if workers < 1 or live_games < 1:
        raise SystemExit("workers and live games must be positive")
    if collection_threads < 0 or update_threads < 0:
        raise SystemExit("collection and update threads may be zero or positive")
    if time_cap_s < 0 or not 0 < evaluation_fraction <= 1:
        raise SystemExit(
            "time cap must be nonnegative and evaluation fraction in (0, 1]"
        )
    cfg = _config(
        ruleset,
        arm,
        seed,
        cap,
        workers,
        64,
        256,
        device,
        live_games=live_games,
        evaluation_fraction=evaluation_fraction,
    )
    if time_cap_s:
        cfg = cfg.with_(time_cap_s=time_cap_s)
    out = Path("logs/rl") / f"resume_{ruleset}_{arm}_seed{seed}_{int(time.time())}"
    run = resume(checkpoint, cfg, out, _league_for(ruleset), device)
    run.execution = {
        **(
            {"collection_torch_threads": collection_threads}
            if collection_threads
            else {}
        ),
        **({"update_torch_threads": update_threads} if update_threads else {}),
    }
    run.tracker = start_tracker(run, kind="resume")
    print(
        f"\n== resume from {checkpoint}: update {run.progress.update}, {run.progress.learner_decisions:,} learner decisions carried over"
    )
    print(
        f"   a continuation preserves progress and configuration, not bitwise numerical trajectories"
    )
    collector = None
    if workers > 1:
        collector = ParallelCollector(
            WorkerConfig(
                ruleset=ruleset,
                arm=arm,
                width=cfg.width,
                blocks=cfg.blocks,
                league=run.league_name,
                training_seed=seed,
                live_games=live_games,
            ),
            workers,
        )
    try:
        summary = train(run, run.tracker, collector)
    finally:
        if collector is not None:
            collector.close()
    print(
        f"   final {summary['final_score']:.4f}, selected update {summary['selected_update']}"
    )


def main(argv: list[str]) -> None:
    what = argv[0] if argv else "shapes"
    rest = argv[1:]
    match what:
        case "shapes":
            cmd_shapes(*rest)
        case "smoke":
            cmd_smoke(*(f(x) for f, x in zip((str, str, int), rest)))
        case "train":
            cmd_train(
                *(
                    f(x)
                    for f, x in zip(
                        (
                            str,
                            str,
                            int,
                            int,
                            int,
                            int,
                            int,
                            str,
                            int,
                            int,
                            int,
                            float,
                            float,
                        ),
                        rest,
                    )
                )
            )
        case "pilot":
            cmd_pilot(*(f(x) for f, x in zip((int,), rest)))
        case "resume":
            cmd_resume(
                rest[0],
                *(
                    f(x)
                    for f, x in zip(
                        (str, str, int, int, str, int, int, int, int, float, float),
                        rest[1:],
                    )
                ),
            )
        case "systems":
            cmd_systems(
                *(
                    f(x)
                    for f, x in zip(
                        (str, str, int, int, str, int, int, int, int, int), rest
                    )
                )
            )
        case "devsize":
            cmd_devsize(
                *(f(x) for f, x in zip((str, str, int, int, float, int, str), rest))
            )
        case _:
            raise SystemExit(
                f"unknown subcommand {what!r}; expected shapes, smoke, pilot, train, resume, systems or devsize"
            )
