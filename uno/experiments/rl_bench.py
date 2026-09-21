"""Benchmark PPO collection/update loops with isolated benchmark state.

Key properties:
- Every n_minibatches value gets its own Config/Run, so _iteration sees the requested partition count.
- Microbenchmark, complete-update benchmark, and full-cycle benchmark use separate Runs.
- All update benchmarks start from the same actor/critic/optimizer snapshot that generated
  the frozen rollout, avoiding cross-contamination between sweep points.
- CUDA timings synchronize before and after macro measurements.
"""

import argparse
import copy
import json
import statistics
import time
from dataclasses import replace
from pathlib import Path

import torch
from tqdm import tqdm

from uno.rl import ppo
from uno.rl.config import Collection, Config, Evaluation, Optim
from uno.rl.features import ARM_B, Features
from uno.rl.train import Run, _iteration
from uno.rl.parallel import ParallelCollector, WorkerConfig
from uno.rl.rollout import Rollout
from uno.rl.device import resolve_device, synchronize_device


def slice_rollout(r: Rollout, size: int) -> Rollout:
    """Slice a rollout to at most ``size`` decisions."""
    if size >= r.n_decisions:
        return r

    actor_features = Features(
        r.actor.tok[:size], r.actor.col[:size], r.actor.glob[:size]
    )
    critic_features = Features(
        r.critic.tok[:size], r.critic.col[:size], r.critic.glob[:size]
    )
    return replace(
        r,
        actor=actor_features,
        critic=critic_features,
        masks=r.masks[:size],
        actions=r.actions[:size],
        old_logp=r.old_logp[:size],
        episode=r.episode[:size],
        decision_index=r.decision_index[:size],
        set_id=r.set_id[:size] if r.set_id is not None else None,
    )


def make_config(
    *,
    workers: int,
    n_minibatches: int,
    device,
    ruleset: str,
    arm,
    seed: int,
    cap: int,
    games_per_cell: int,
    optim_overrides: dict,
) -> Config:
    """Build a fresh immutable config for one sweep point."""
    return Config(
        ruleset=ruleset,
        arm=arm,
        training_seed=seed,
        data_cap=cap,
        device=device,
        collection=Collection(
            games_per_cell=games_per_cell,
            live_games_per_worker=64,
            workers=workers,
        ),
        evaluation=Evaluation(deals_per_cell=256, every_fraction=0.016),
        optim=Optim(n_minibatches=n_minibatches, **optim_overrides),
    )


def make_run(
    cfg: Config,
    *,
    workers: int,
    n_minibatches: int,
    tag: str,
    root: Path,
    collection_threads: int,
    update_threads: int,
) -> Run:
    """Create a fresh Run whose internal config already has the requested minibatch count."""
    out = root / f"w{workers}_nmb{n_minibatches}_{tag}_{time.time_ns()}"
    run = Run(cfg, out, "rulebot")
    run.execution = {
        "collection_torch_threads": collection_threads,
        "update_torch_threads": update_threads,
    }
    return run


def snapshot_training_state(run: Run) -> dict:
    """Deep-copy the trainable state so every benchmark starts identically."""
    return {
        "actor": copy.deepcopy(run.actor.state_dict()),
        "critic": copy.deepcopy(run.critic.state_dict()),
        "actor_opt": copy.deepcopy(run.actor_opt.state_dict()),
        "critic_opt": copy.deepcopy(run.critic_opt.state_dict()),
    }


def restore_training_state(run: Run, state: dict) -> None:
    run.actor.load_state_dict(state["actor"])
    run.critic.load_state_dict(state["critic"])
    run.actor_opt.load_state_dict(state["actor_opt"])
    run.critic_opt.load_state_dict(state["critic_opt"])


def make_collector(
    cfg: Config,
    workers: int,
    *,
    ruleset: str,
    arm,
    seed: int,
) -> ParallelCollector | None:
    if workers <= 1:
        return None

    wc = WorkerConfig(
        ruleset=ruleset,
        arm=arm,
        width=cfg.width,
        blocks=cfg.blocks,
        league="rulebot",
        training_seed=seed,
        live_games=cfg.collection.live_games_per_worker,
        temperature=cfg.optim.temperature,
    )
    return ParallelCollector(wc, workers)


def collect_frozen_rollout(run: Run, collector, cfg: Config) -> Rollout:
    """Collect one rollout from the exact policy state that will be benchmarked."""
    from uno.rl.collect import collect
    from uno.rl import manifest as mf
    from uno.rl.features import spec

    manifest, _next_id = mf.build(
        run.progress.update,
        list(run.opponents),
        run.schedule,
        cfg.collection.games_per_cell,
        run.progress.next_episode_id,
    )
    critic_spec = spec(cfg.rs, "CRITIC")

    if collector is not None:
        rollout, _ = collector.collect(
            manifest, run.actor, critic_spec, run.progress.update
        )
        return rollout

    if cfg.device == "cpu":
        policy = run.policy
    else:
        from uno.rl.model import Actor
        from uno.rl.policy import ActorPolicy

        actor_cpu = Actor(spec(cfg.rs, cfg.arm), cfg.width, cfg.blocks)
        actor_cpu.load_state_dict({
            k: v.detach().cpu() for k, v in run.actor.state_dict().items()
        })
        policy = ActorPolicy(
            spec(cfg.rs, cfg.arm), actor_cpu, cfg.optim.temperature, "cpu"
        )

    rollout, _ = collect(
        manifest,
        policy,
        critic_spec,
        run.opponents,
        cfg.training_seed,
        run.progress.update,
        cfg.collection.live_games_per_worker,
    )
    return rollout


def timed_update(
    run: Run,
    rollout: Rollout,
    optim,
    *,
    seed: int,
    device,
) -> float:
    """Time exactly one PPO update with correct accelerator synchronization."""
    synchronize_device(device)
    t0 = time.perf_counter()
    ppo.update(
        run.actor,
        run.critic,
        run.actor_opt,
        run.critic_opt,
        rollout,
        optim,
        seed,
        0,
        device,
    )
    synchronize_device(device)
    return time.perf_counter() - t0


def main():
    parser = argparse.ArgumentParser(description="Benchmark PPO loops.")
    parser.add_argument(
        "--workers",
        type=int,
        nargs="+",
        default=[1],
        help="List of collection worker counts to sweep.",
    )
    parser.add_argument(
        "--n-minibatches",
        type=int,
        nargs="+",
        default=[32],
        dest="n_minibatches",
        help="List of balanced PPO minibatch counts per epoch to sweep.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use for benchmarking (e.g. mps, cuda, cpu, auto).",
    )
    parser.add_argument(
        "--micro-repeats",
        type=int,
        default=3,
        help="Independent repeats for the one-minibatch microbenchmark.",
    )
    args = parser.parse_args()

    # Exact settings from current run settings.
    ruleset = "FULL_UNO"
    arm = ARM_B
    seed = 0
    cap = 108_000
    games_per_cell = 64
    collection_threads = 1
    update_threads = 1
    optim_overrides = {
        "normalize_advantages": True,
        "entropy_coef": 0.01,
        "actor_grad_norm": 0.5,
        "actor_lr": 1e-4,
        "actor_epochs": 4,
        "critic_grad_norm": 0.5,
        "critic_lr": 1e-4,
        "target_kl": None,  # disable stopping only; PPO still computes KL diagnostics
    }

    device = resolve_device(args.device)
    is_cuda = str(device).startswith("cuda")

    root = Path("logs/rl") / f"benchmark_fixed_{int(time.time())}"
    root.mkdir(parents=True, exist_ok=True)
    results_path = root / "results.json"
    results = []

    pbar = tqdm(
        total=len(args.workers) * len(args.n_minibatches),
        desc="Sweeping Benchmarks",
    )

    def safe_print(msg=""):
        pbar.write(str(msg))

    print = safe_print
    print(f"Resolved device: {device}")
    print(f"Results: {results_path}")

    for workers in args.workers:
        print(f"\n{'=' * 50}")
        print(f" Sweeping workers={workers}")
        print(f"{'=' * 50}")

        # Use the first requested minibatch count only for collector/run construction while
        # warming infrastructure. This warmup Run is discarded afterward.
        warm_cfg = make_config(
            workers=workers,
            n_minibatches=args.n_minibatches[0],
            device=device,
            ruleset=ruleset,
            arm=arm,
            seed=seed,
            cap=cap,
            games_per_cell=games_per_cell,
            optim_overrides=optim_overrides,
        )
        collector = make_collector(
            warm_cfg, workers, ruleset=ruleset, arm=arm, seed=seed
        )

        try:
            print("Warming up infrastructure (discarded run)...")
            warm_run = make_run(
                warm_cfg,
                workers=workers,
                n_minibatches=args.n_minibatches[0],
                tag="warmup",
                root=root,
                collection_threads=collection_threads,
                update_threads=update_threads,
            )
            _iteration(warm_run, collector, update_threads)
            synchronize_device(device)
            del warm_run

            # Build a clean baseline Run after warmup. Snapshot it BEFORE collection,
            # then collect the frozen rollout from exactly this policy state.
            base_cfg = make_config(
                workers=workers,
                n_minibatches=args.n_minibatches[0],
                device=device,
                ruleset=ruleset,
                arm=arm,
                seed=seed,
                cap=cap,
                games_per_cell=games_per_cell,
                optim_overrides=optim_overrides,
            )
            base_run = make_run(
                base_cfg,
                workers=workers,
                n_minibatches=args.n_minibatches[0],
                tag="baseline",
                root=root,
                collection_threads=collection_threads,
                update_threads=update_threads,
            )
            baseline_state = snapshot_training_state(base_run)

            print("Collecting frozen rollout for update benchmarks...")
            rollout = collect_frozen_rollout(base_run, collector, base_cfg)
            rollout_size = rollout.n_decisions
            print(f"Collected rollout of size {rollout_size} decisions.")
            del base_run

            for nmb in args.n_minibatches:
                print(f"\n--- Benchmarking n_minibatches={nmb} (workers={workers}) ---")

                cfg = make_config(
                    workers=workers,
                    n_minibatches=nmb,
                    device=device,
                    ruleset=ruleset,
                    arm=arm,
                    seed=seed,
                    cap=cap,
                    games_per_cell=games_per_cell,
                    optim_overrides=optim_overrides,
                )

                # -------------------------------------------------------------
                # Benchmark 1: one minibatch.
                # Each repeat gets a fresh Run restored to the same baseline, so
                # timeit-style repeated optimizer mutations cannot contaminate it.
                # -------------------------------------------------------------
                effective_nmb = max(1, min(nmb, rollout_size))
                micro_rows = (rollout_size + effective_nmb - 1) // effective_nmb
                mb_rollout = slice_rollout(rollout, micro_rows)
                mb_optim = replace(
                    cfg.optim,
                    actor_epochs=1,
                    critic_epochs=1,
                    n_minibatches=1,
                )
                micro_times = []
                for rep in range(args.micro_repeats):
                    micro_run = make_run(
                        cfg,
                        workers=workers,
                        n_minibatches=nmb,
                        tag=f"micro{rep}",
                        root=root,
                        collection_threads=collection_threads,
                        update_threads=update_threads,
                    )
                    restore_training_state(micro_run, baseline_state)
                    micro_times.append(
                        timed_update(
                            micro_run,
                            mb_rollout,
                            mb_optim,
                            seed=seed,
                            device=device,
                        )
                    )
                    del micro_run
                mb_time = statistics.median(micro_times)

                # -------------------------------------------------------------
                # Benchmark 2: one complete update over the frozen rollout.
                # Fresh model/optimizer state; no prior microbenchmark mutations.
                # -------------------------------------------------------------
                update_run = make_run(
                    cfg,
                    workers=workers,
                    n_minibatches=nmb,
                    tag="update",
                    root=root,
                    collection_threads=collection_threads,
                    update_threads=update_threads,
                )
                restore_training_state(update_run, baseline_state)

                if is_cuda:
                    torch.cuda.reset_peak_memory_stats()

                update_time = timed_update(
                    update_run,
                    rollout,
                    cfg.optim,
                    seed=seed,
                    device=device,
                )

                peak_alloc_gib = None
                peak_reserved_gib = None
                if is_cuda:
                    peak_alloc_gib = torch.cuda.max_memory_allocated() / (1024**3)
                    peak_reserved_gib = torch.cuda.max_memory_reserved() / (1024**3)

                del update_run

                # -------------------------------------------------------------
                # Benchmark 3: full collect + update cycle.
                # Crucially, this Run is CONSTRUCTED WITH cfg.optim.n_minibatches=nmb,
                # so _iteration cannot silently keep using the first sweep value.
                # -------------------------------------------------------------
                cycle_run = make_run(
                    cfg,
                    workers=workers,
                    n_minibatches=nmb,
                    tag="cycle",
                    root=root,
                    collection_threads=collection_threads,
                    update_threads=update_threads,
                )
                restore_training_state(cycle_run, baseline_state)

                synchronize_device(device)
                t0 = time.perf_counter()
                stats, extra = _iteration(cycle_run, collector, update_threads)
                synchronize_device(device)
                wall_cycle_time = time.perf_counter() - t0
                del cycle_run

                rollout_t = extra["collect_s"]
                actual_update_t = extra["update_s"]
                actor_presentations = extra["actor_presentations"]

                # Recover the number of unique decisions from the actual number
                # of actor presentations and the epochs ACTUALLY reported by this
                # run. This no longer mixes local cfg values with another Run.
                cycle_rollout_size = actor_presentations / stats.actor_epochs

                # The training-cycle denominator is exactly the two components we
                # are trying to optimize. Keep wall time too as a sanity check.
                train_cycle_time = rollout_t + actual_update_t

                unique_env_throughput = (
                    cycle_rollout_size / train_cycle_time
                    if train_cycle_time > 0
                    else float("inf")
                )
                rollout_throughput = (
                    cycle_rollout_size / rollout_t if rollout_t > 0 else float("inf")
                )
                update_throughput = (
                    actor_presentations / actual_update_t
                    if actual_update_t > 0
                    else float("inf")
                )

                print(
                    f"1 Minibatch (~{mb_rollout.n_decisions} decisions; {nmb} per epoch): {mb_time:.4f}s"
                )
                print(
                    f"1 Complete Update ({rollout_size} decisions, "
                    f"{cfg.optim.actor_epochs} actor / {cfg.optim.critic_epochs} critic epochs): "
                    f"{update_time:.4f}s"
                )
                print(
                    f"Full Cycle: train={train_cycle_time:.4f}s "
                    f"(collect={rollout_t:.4f}s + update={actual_update_t:.4f}s), "
                    f"wall={wall_cycle_time:.4f}s"
                )
                print(
                    f"Unique env throughput: {unique_env_throughput:,.0f} decisions/s"
                )
                print(f"Rollout throughput: {rollout_throughput:,.0f} decisions/s")
                print(f"Update throughput: {update_throughput:,.0f} presentations/s")
                if is_cuda:
                    print(
                        f"Peak CUDA memory (complete update): "
                        f"allocated={peak_alloc_gib:.2f} GiB, "
                        f"reserved={peak_reserved_gib:.2f} GiB"
                    )

                row = {
                    "workers": workers,
                    "n_minibatches": nmb,
                    "frozen_rollout_decisions": rollout_size,
                    "micro_minibatch_decisions": mb_rollout.n_decisions,
                    "approx_decisions_per_minibatch": rollout_size / effective_nmb,
                    "micro_median_s": mb_time,
                    "complete_update_s": update_time,
                    "cycle_unique_decisions": cycle_rollout_size,
                    "cycle_actor_presentations": actor_presentations,
                    "cycle_actor_epochs": stats.actor_epochs,
                    "cycle_collect_s": rollout_t,
                    "cycle_update_s": actual_update_t,
                    "cycle_train_s": train_cycle_time,
                    "cycle_wall_s": wall_cycle_time,
                    "unique_env_decisions_s": unique_env_throughput,
                    "rollout_decisions_s": rollout_throughput,
                    "update_presentations_s": update_throughput,
                    "peak_cuda_allocated_gib": peak_alloc_gib,
                    "peak_cuda_reserved_gib": peak_reserved_gib,
                }
                results.append(row)

                # Persist after every point so a long sweep survives interruption/OOM.
                results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
                pbar.update(1)

        finally:
            if collector is not None:
                collector.close()

    pbar.close()
    print(f"\nSaved {len(results)} benchmark rows to {results_path}")


if __name__ == "__main__":
    main()
