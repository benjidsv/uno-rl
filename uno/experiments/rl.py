"""Run, resume or inspect a PPO training run (docs/rl/pipeline-plan.md stages 1-4).

`uv run uno rl shapes [RULESET] [ARM]`            recorded tensor shapes and parameter counts
`uv run uno rl smoke [RULESET] [ARM] [GAMES/CELL]` one bounded iteration end to end
`uv run uno rl train [RULESET] [ARM] [SEED] [DECISIONS] [WORKERS] [GAMES/CELL] [DEV DEALS/CELL] [DEVICE] [COLLECTION_THREADS] [UPDATE_THREADS] [MINIBATCH] [TIME_CAP_S]`
`uv run uno rl resume CHECKPOINT [RULESET] [ARM] [SEED] [DECISIONS]`
`uv run uno rl systems [RULESET] [ARM] [WORKERS] [LIVE] [DEVICE] [GAMES/CELL] [SAMPLES] [MINIBATCH] [COLLECTION_THREADS] [UPDATE_THREADS]`

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

import json
import os
import platform
import re
import resource
import subprocess
import threading
import time
from dataclasses import replace
from pathlib import Path

from uno.game.rules import NAMED
from uno.rl.config import Collection, Config, Evaluation, Optim
from uno.rl.features import ARM_A, ARM_B, ARM_CRITIC, spec
from uno.rl.model import shapes
from uno.rl.opponents import league, league_hash
from uno.rl.parallel import ParallelCollector, WorkerConfig
from uno.rl.rollout import estimate_bytes
from uno.rl.train import Run, iteration, resume, start_tracker, train

TOY_LEAGUES = {"TOY": "toy", "TOY_FULL": "toy", "TOY2": "toy", "TOY2_SKIP": "toy", "TOY2_FULL": "toy", "TOY_CANDIDATE": "toy"}


def _league_for(ruleset: str) -> str:
    return TOY_LEAGUES.get(ruleset, "rulebot")


def _config(ruleset: str, arm: str, seed: int, cap: int, workers: int, games_per_cell: int, deals_per_cell: int, device: str, live_games: int = 64, minibatch: int = 512) -> Config:
    return Config(
        ruleset=ruleset, arm=arm, training_seed=seed, data_cap=cap, device=device,
        collection=Collection(games_per_cell=games_per_cell, live_games_per_worker=live_games, workers=workers),
        evaluation=Evaluation(deals_per_cell=deals_per_cell),
        optim=replace(Optim(), minibatch=minibatch),
    )  # fmt: skip


def cmd_shapes(ruleset: str = "FULL_UNO", arm: str = ARM_B) -> None:
    rs = NAMED[ruleset]
    sp, spc = spec(rs, arm), spec(rs, ARM_CRITIC)
    sh = shapes(sp, spc)
    print(f"\n== {ruleset} arm {arm}")
    print(f"   {sh}")
    print(f"   actor schema {sp.hash}, critic schema {spc.hash}, league {league_hash(rs, _league_for(ruleset))}")
    other = ARM_A if arm == ARM_B else ARM_B
    alt = shapes(spec(rs, other), spc)
    print(f"   {other}: {alt.actor_params:,} actor params ({sh.actor_params - alt.actor_params:+,} against {arm}; the arms share the position encoder's shape, not its width)")
    for decisions in (8192, 65536):
        mb = estimate_bytes(sp, spc, decisions, decisions // 4) / 2**20
        print(f"   a rollout of {decisions:,} decisions is about {mb:,.0f} MiB before allocation")


def cmd_smoke(ruleset: str = "TOY", arm: str = ARM_B, games_per_cell: int = 8) -> None:
    """One iteration end to end: collection, update, counters. Not a learning claim."""
    cfg = _config(ruleset, arm, 0, 10**9, 1, games_per_cell, 16, "cpu")
    out = Path("logs/rl") / f"smoke_{ruleset}_{int(time.time())}"
    run = Run(cfg, out, _league_for(ruleset))
    t0 = time.perf_counter()
    stats, extra = iteration(run)
    print(f"\n== smoke {ruleset} arm {arm}: {run.progress.games} games, {run.progress.learner_decisions:,} learner decisions in {time.perf_counter() - t0:.1f}s")
    print(f"   {stats}")
    print(f"   collect {extra['collect_s']:.2f}s ({extra['collect_decisions_per_s']:,.0f} learner dec/s, mean batch {extra['mean_inference_batch']:.1f}), "
          f"update {extra['update_s']:.2f}s, rollout {extra['rollout_bytes'] / 2**20:.1f} MiB")
    print(f"   training score per cell: " + " ".join(f"{k.replace('train_score_', '')}={v:.3f}" for k, v in extra.items() if k.startswith("train_score_")))


def cmd_train(
    ruleset: str = "TOY", arm: str = ARM_B, seed: int = 0, cap: int = 100_000,
    workers: int = 1, games_per_cell: int = 64, deals_per_cell: int = 256,
    device: str = "cpu", collection_threads: int = 0, update_threads: int = 0,
    minibatch: int = 512, time_cap_s: float = 0.0,
) -> None:
    if collection_threads < 0 or update_threads < 0:
        raise SystemExit("collection and update threads may be zero for the current default, or positive")
    if minibatch < 1 or time_cap_s < 0:
        raise SystemExit("minibatch must be positive and time cap may be zero (disabled) or positive")
    if device == "mps":
        import torch
        if not torch.backends.mps.is_available():
            raise SystemExit("MPS was requested but is unavailable")
    cfg = _config(ruleset, arm, seed, cap, workers, games_per_cell, deals_per_cell, device, minibatch=minibatch)
    if time_cap_s:
        cfg = cfg.with_(time_cap_s=time_cap_s)
    out = Path("logs/rl") / f"{ruleset}_{arm}_seed{seed}_mb{minibatch}_{device}_{int(time.time())}"
    run = Run(cfg, out, _league_for(ruleset))
    run.execution = {
        **({"collection_torch_threads": collection_threads} if collection_threads else {}),
        **({"update_torch_threads": update_threads} if update_threads else {}),
    }
    run.tracker = start_tracker(run)
    print(f"\n== train {ruleset} arm {arm} seed {seed}: cap {cap:,} learner decisions, {len(run.opponents)} opponents, "
          f"{cfg.collection.games_per_cell} games/cell, minibatch {minibatch}, device {device}, "
          f"time cap {time_cap_s or 'disabled'}s")
    print(f"   {shapes(spec(cfg.rs, arm), spec(cfg.rs, ARM_CRITIC), cfg.width, cfg.blocks)}")
    print(f"   config {cfg.hash}; artifacts in {out}")
    if not run.tracker.available:
        print(f"   note: {run.tracker.unavailable_reason}")
    collector = None
    if workers > 1:
        wc = WorkerConfig(ruleset=ruleset, arm=arm, width=cfg.width, blocks=cfg.blocks, league=run.league_name, training_seed=seed, live_games=cfg.collection.live_games_per_worker)
        collector = ParallelCollector(wc, workers)
    try:
        summary = train(run, run.tracker, collector)
    finally:
        if collector is not None:
            collector.close()
    print(f"\n   stop: {summary['stop_reason']}, {summary['updates']} updates, {summary['learner_decisions']:,} learner decisions, "
          f"{summary['active_seconds']:.0f}s active, {summary['wall_seconds']:.0f}s wall")
    print(f"   DEV-selected update {summary['selected_update']} at {summary['selected_score']:.4f}; final {summary['final_score']:.4f} "
          f"[{summary['final_ci95'][0]:.4f}, {summary['final_ci95'][1]:.4f}] (95% CI, DEV panel: descriptive, not confirmatory)")
    print(f"   {json.dumps(summary['cells'])}")


def _rss_mib(pids: list[int]) -> dict[str, float | None]:
    """Resident memory at an iteration boundary; macOS has no /proc dependency."""
    out = {"coordinator": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20}
    for pid in pids:
        try:
            # `ps` reports KiB on macOS. This is not a physical-memory total:
            # shared pages may be counted for each worker on unified memory.
            kib = int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)], text=True).strip())
            out[str(pid)] = kib / 1024
        except (ValueError, subprocess.CalledProcessError):
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
    except (OSError, subprocess.CalledProcessError):
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
    except (OSError, subprocess.CalledProcessError):
        pass
    try:
        output = subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True)
        match = re.search(r"used = ([0-9.]+)([MG])", output)
        if match:
            swap = float(match.group(1)) * (1024 if match.group(2) == "G" else 1)
    except (OSError, subprocess.CalledProcessError):
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
            self.minimum_free_percent = free if self.minimum_free_percent is None else min(self.minimum_free_percent, free)
        if swap is not None:
            self.minimum_swap_used_mib = swap if self.minimum_swap_used_mib is None else min(self.minimum_swap_used_mib, swap)
            self.maximum_swap_used_mib = swap if self.maximum_swap_used_mib is None else max(self.maximum_swap_used_mib, swap)

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
            "swap_growth_mib": None if self.minimum_swap_used_mib is None else self.maximum_swap_used_mib - self.minimum_swap_used_mib,
        }


def cmd_systems(
    ruleset: str = "FULL_UNO", arm: str = ARM_B, workers: int = 1, live_games: int = 64,
    device: str = "cpu", games_per_cell: int = 16, samples: int = 1, minibatch: int = 512,
    collection_threads: int = 1, update_threads: int = 0,
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
    if min(workers, live_games, games_per_cell, samples, minibatch, collection_threads) < 1 or update_threads < 0:
        raise SystemExit("workers, live games, games/cell, samples, minibatch and collection threads must be positive; update threads may be zero for the current default")
    resolved_update_threads = None if update_threads == 0 else update_threads
    cfg = _config(ruleset, arm, 0, 10**9, workers, games_per_cell, 1, device, live_games, minibatch)
    out = Path("logs/rl") / f"systems_{ruleset}_{arm}_w{workers}_l{live_games}_{device}_mb{minibatch}_ct{collection_threads}_ut{update_threads or 'default'}_{int(time.time())}"
    collector = None
    if workers > 1:
        wc = WorkerConfig(ruleset=ruleset, arm=arm, width=cfg.width, blocks=cfg.blocks, league=_league_for(ruleset), training_seed=0, live_games=live_games)
        collector = ParallelCollector(wc, workers)
    try:
        iteration(
            Run(cfg, out / "warmup", _league_for(ruleset)), collector,
            collection_threads=collection_threads,
            update_threads=resolved_update_threads,
        )
        rows = []
        for n in range(samples):
            run = Run(cfg, out / f"sample{n}", _league_for(ruleset))
            pids = [] if collector is None else [p.pid for p in collector._procs if p.pid is not None]
            memory = _MemoryMonitor([os.getpid(), *pids])
            memory.start()
            try:
                stats, extra = iteration(
                    run, collector,
                    collection_threads=collection_threads,
                    update_threads=resolved_update_threads,
                )
            finally:
                memory_summary = memory.finish()
            elapsed = extra["iteration_s"]
            rows.append({
                "sample": n, "learner_decisions": run.progress.learner_decisions,
                "games": run.progress.games, "engine_decisions": run.progress.engine_decisions,
                "iteration_s": elapsed, "collection_s": extra["collect_s"], "update_s": extra["update_s"],
                "learner_decisions_per_s": run.progress.learner_decisions / elapsed,
                "update_fresh_decisions_per_s": extra["update_fresh_decisions_per_s"],
                "actor_presentations": extra["actor_presentations"],
                "critic_presentations": extra["critic_presentations"],
                "actor_presentations_per_s": extra["actor_presentations_per_s"],
                "critic_presentations_per_s": extra["critic_presentations_per_s"],
                "actor_steps": extra["actor_steps"], "critic_steps": extra["critic_steps"],
                "games_per_s": run.progress.games / elapsed,
                "engine_decisions_per_s": run.progress.engine_decisions / elapsed,
                "collection_learner_decisions_per_s": extra["collect_decisions_per_s"],
                "mean_inference_batch": extra["mean_inference_batch"],
                "inference_batch_distribution": {
                    "calls": extra["inference_batch_calls"], "min": extra["inference_batch_min"],
                    "p50": extra["inference_batch_p50"], "p90": extra["inference_batch_p90"],
                    "max": extra["inference_batch_max"],
                },
                "drain_s": extra["drain_s"],
                "rollout_mib": extra["rollout_bytes"] / 2**20, "actor_epochs": stats.actor_epochs,
                "critic_epochs": stats.critic_epochs, "rss_mib": _rss_mib(pids),
                "memory": memory_summary,
                "worker_stage_s": {
                    "simulation": extra["simulation_s"], "features": extra["feature_s"],
                    "actor_inference": extra["actor_inference_s"], "assembly": extra["worker_assembly_s"],
                },
                "transfer": {
                    "serialization_s": extra["serialization_s"], "result_queue_put_s": extra["result_queue_put_s"],
                    "result_queue_wait_s": extra["result_queue_wait_s"], "deserialization_s": extra["deserialization_s"],
                    "coordinator_assembly_s": extra["coordinator_assembly_s"], "bytes": extra["transfer_bytes"],
                },
                "update_stage_s": {
                    "critic_values": extra["critic_values_s"], "actor": extra["actor_update_s"],
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
        "iteration_s", "collection_s", "update_s", "learner_decisions_per_s",
        "update_fresh_decisions_per_s", "actor_presentations_per_s",
        "critic_presentations_per_s", "actor_steps", "critic_steps", "games_per_s",
        "engine_decisions_per_s", "collection_learner_decisions_per_s",
        "mean_inference_batch", "drain_s", "rollout_mib",
    )
    payload = {
        "purpose": "Bounded systems calibration; throughput only, not a learning result",
        "machine": platform.platform(), "architecture": platform.machine(), "pid": os.getpid(),
        "config": cfg.resolved(), "config_hash": cfg.hash, "warmup_excluded": True,
        "execution": {
            "collection_torch_threads": collection_threads,
            "update_torch_threads": update_threads or "default",
            "device_synchronized_for_timing": device in {"mps", "cuda"} or device.startswith("cuda:"),
        },
        "collector_execution": {
            "task_bundle_games": None if collector is None else collector.bundle,
            "maximum_live_games_per_worker_task": cfg.collection.live_games_per_worker if collector is None else min(cfg.collection.live_games_per_worker, collector.bundle),
            "note": "The parallel collector still collects each task independently; use the measured inference-batch distribution, not the requested live_games value, as the realized width.",
        },
        "samples": rows, "mean": {k: sum(r[k] for r in rows) / len(rows) for k in keys},
        "memory_note": "Aggregate RSS sums process RSS and can double-count shared pages; on unified memory it is not physical usage. System pressure and swap provide the whole-machine context.",
        "timing_note": "Worker stage times sum overlapping workers; result_queue_wait_s is coordinator blocked time and includes useful worker computation.",
        "presentation_note": "Actor/critic presentations count fresh rollout rows times completed epochs. They exclude the extra no-gradient KL pass and critic baseline evaluation, so they are a stable workload proxy rather than all model forwards.",
    }
    out.mkdir(parents=True, exist_ok=True)
    path = out / "systems.json"
    path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print(f"\n   saved {path}")


def cmd_resume(checkpoint: str, ruleset: str = "TOY", arm: str = ARM_B, seed: int = 0, cap: int = 100_000, device: str = "cpu") -> None:
    cfg = _config(ruleset, arm, seed, cap, 1, 64, 256, device)
    out = Path("logs/rl") / f"resume_{ruleset}_{arm}_seed{seed}_{int(time.time())}"
    run = resume(checkpoint, cfg, out, _league_for(ruleset), device)
    run.tracker = start_tracker(run, kind="resume")
    print(f"\n== resume from {checkpoint}: update {run.progress.update}, {run.progress.learner_decisions:,} learner decisions carried over")
    print(f"   a continuation preserves progress and configuration, not bitwise numerical trajectories")
    summary = train(run, run.tracker)
    print(f"   final {summary['final_score']:.4f}, selected update {summary['selected_update']}")


def main(argv: list[str]) -> None:
    what = argv[0] if argv else "shapes"
    rest = argv[1:]
    match what:
        case "shapes":
            cmd_shapes(*rest)
        case "smoke":
            cmd_smoke(*(f(x) for f, x in zip((str, str, int), rest)))
        case "train":
            cmd_train(*(f(x) for f, x in zip((str, str, int, int, int, int, int, str, int, int, int, float), rest)))
        case "resume":
            cmd_resume(rest[0], *(f(x) for f, x in zip((str, str, int, int, str), rest[1:])))
        case "systems":
            cmd_systems(*(f(x) for f, x in zip((str, str, int, int, str, int, int, int, int, int), rest)))
        case _:
            raise SystemExit(f"unknown subcommand {what!r}; expected shapes, smoke, train, resume or systems")
