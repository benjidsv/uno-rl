"""Run, resume or inspect a PPO training run (docs/rl/pipeline-plan.md stages 1-4).

`uv run uno rl shapes [RULESET] [ARM]`            recorded tensor shapes and parameter counts
`uv run uno rl smoke [RULESET] [ARM] [GAMES/CELL]` one bounded iteration end to end
`uv run uno rl train [RULESET] [ARM] [SEED] [DECISIONS] [WORKERS]`
`uv run uno rl resume CHECKPOINT [RULESET] [ARM] [SEED] [DECISIONS]`
`uv run uno rl systems [RULESET] [ARM] [WORKERS] [LIVE] [DEVICE] [GAMES/CELL] [SAMPLES]`

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
import resource
import subprocess
import time
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


def _config(ruleset: str, arm: str, seed: int, cap: int, workers: int, games_per_cell: int, deals_per_cell: int, device: str, live_games: int = 64) -> Config:
    return Config(
        ruleset=ruleset, arm=arm, training_seed=seed, data_cap=cap, device=device,
        collection=Collection(games_per_cell=games_per_cell, live_games_per_worker=live_games, workers=workers),
        evaluation=Evaluation(deals_per_cell=deals_per_cell),
        optim=Optim(),
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


def cmd_train(ruleset: str = "TOY", arm: str = ARM_B, seed: int = 0, cap: int = 100_000, workers: int = 1, games_per_cell: int = 64, deals_per_cell: int = 256, device: str = "cpu") -> None:
    cfg = _config(ruleset, arm, seed, cap, workers, games_per_cell, deals_per_cell, device)
    out = Path("logs/rl") / f"{ruleset}_{arm}_seed{seed}_{int(time.time())}"
    run = Run(cfg, out, _league_for(ruleset))
    run.tracker = start_tracker(run)
    print(f"\n== train {ruleset} arm {arm} seed {seed}: cap {cap:,} learner decisions, {len(run.opponents)} opponents, "
          f"{cfg.collection.games_per_cell} games/cell, device {device}")
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
          f"{summary['active_seconds']:.0f}s active")
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


def cmd_systems(
    ruleset: str = "FULL_UNO", arm: str = ARM_B, workers: int = 1, live_games: int = 64,
    device: str = "cpu", games_per_cell: int = 16, samples: int = 1,
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
    if workers < 1 or live_games < 1 or games_per_cell < 1 or samples < 1:
        raise SystemExit("workers, live games, games/cell and samples must be positive")
    cfg = _config(ruleset, arm, 0, 10**9, workers, games_per_cell, 1, device, live_games)
    out = Path("logs/rl") / f"systems_{ruleset}_{arm}_w{workers}_l{live_games}_{device}_{int(time.time())}"
    collector = None
    if workers > 1:
        wc = WorkerConfig(ruleset=ruleset, arm=arm, width=cfg.width, blocks=cfg.blocks, league=_league_for(ruleset), training_seed=0, live_games=live_games)
        collector = ParallelCollector(wc, workers)
    try:
        iteration(Run(cfg, out / "warmup", _league_for(ruleset)), collector)
        rows = []
        for n in range(samples):
            run = Run(cfg, out / f"sample{n}", _league_for(ruleset))
            stats, extra = iteration(run, collector)
            pids = [] if collector is None else [p.pid for p in collector._procs if p.pid is not None]
            elapsed = extra["iteration_s"]
            rows.append({
                "sample": n, "learner_decisions": run.progress.learner_decisions,
                "games": run.progress.games, "engine_decisions": run.progress.engine_decisions,
                "iteration_s": elapsed, "collection_s": extra["collect_s"], "update_s": extra["update_s"],
                "learner_decisions_per_s": run.progress.learner_decisions / elapsed,
                "games_per_s": run.progress.games / elapsed,
                "engine_decisions_per_s": run.progress.engine_decisions / elapsed,
                "collection_learner_decisions_per_s": extra["collect_decisions_per_s"],
                "mean_inference_batch": extra["mean_inference_batch"], "drain_s": extra["drain_s"],
                "rollout_mib": extra["rollout_bytes"] / 2**20, "actor_epochs": stats.actor_epochs,
                "critic_epochs": stats.critic_epochs, "rss_mib": _rss_mib(pids),
            })
    finally:
        if collector is not None:
            collector.close()
    keys = ("iteration_s", "collection_s", "update_s", "learner_decisions_per_s", "games_per_s", "engine_decisions_per_s", "collection_learner_decisions_per_s", "mean_inference_batch", "drain_s", "rollout_mib")
    payload = {
        "purpose": "Stage 6 systems pilot; throughput only, not a learning result",
        "machine": platform.platform(), "architecture": platform.machine(), "pid": os.getpid(),
        "config": cfg.resolved(), "config_hash": cfg.hash, "warmup_excluded": True,
        "samples": rows, "mean": {k: sum(r[k] for r in rows) / len(rows) for k in keys},
        "memory_note": "RSS boundary samples count shared pages per process; do not sum them as physical unified memory.",
        "missing": ["queue wait and transfer bytes are not instrumented by the current collector"],
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
            cmd_train(*(f(x) for f, x in zip((str, str, int, int, int, int, int, str), rest)))
        case "resume":
            cmd_resume(rest[0], *(f(x) for f, x in zip((str, str, int, int, str), rest[1:])))
        case "systems":
            cmd_systems(*(f(x) for f, x in zip((str, str, int, int, str, int, int), rest)))
        case _:
            raise SystemExit(f"unknown subcommand {what!r}; expected shapes, smoke, train, resume or systems")
