"""Benchmark PPO collection/update loops with isolated benchmark state.

Key properties:
- Every n_minibatches value gets its own Config/Run, so _iteration sees the requested partition count.
- Microbenchmark, complete-update benchmark, and full-cycle benchmark use separate Runs.
- All update benchmarks start from the same actor/critic/optimizer snapshot that generated
  the frozen rollout, avoiding cross-contamination between sweep points.
- CUDA timings synchronize before and after macro measurements.
- torch.compile modes are warmed on a discarded PPO update, then model/optimizer
  state is restored before timing so compilation cost never contaminates results.
"""

import argparse
import copy
import json
import math
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
    precision: str,
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
        optim=Optim(
            n_minibatches=n_minibatches, precision=precision, **optim_overrides
        ),
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


COMPILE_MODES = ("eager", "default", "reduce-overhead", "max-autotune")


def configure_compile(run: Run, compile_mode: str) -> None:
    """Compile only model forwards while keeping the Run/module identity intact.

    Replacing ``run.actor`` with ``torch.compile(run.actor)`` wraps the module and
    changes its state_dict key namespace.  The parallel collector consumes the
    normal actor module/state_dict, so compile the bound forward methods instead.
    Parameters and optimizer references remain the exact same objects.
    """
    if compile_mode == "eager":
        run.execution["compile_mode"] = compile_mode
        return
    if compile_mode not in COMPILE_MODES:
        raise ValueError(f"unknown compile mode: {compile_mode}")

    kwargs = {} if compile_mode == "default" else {"mode": compile_mode}
    run.actor.forward = torch.compile(run.actor.forward, **kwargs)
    run.critic.forward = torch.compile(run.critic.forward, **kwargs)
    run.execution["compile_mode"] = compile_mode


def warm_compiled_update(
    run: Run,
    rollout: Rollout,
    optim: Optim,
    baseline_state: dict,
    *,
    compile_mode: str,
    seed: int,
    device,
) -> None:
    """Pay compile/recompile cost outside timing, then restore pristine state.

    A complete update is intentional: it exercises actor + critic forward/backward
    and every balanced minibatch shape that the measured update will encounter.
    The discarded optimizer/model mutations are restored before measurement.
    """
    if compile_mode == "eager":
        return

    synchronize_device(device)
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

    restore_training_state(run, baseline_state)
    # Match a fresh Run: the discarded backward pass must not leave .grad tensors.
    run.actor_opt.zero_grad(set_to_none=True)
    run.critic_opt.zero_grad(set_to_none=True)


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
) -> tuple[float, ppo.UpdateStats]:
    """Time exactly one PPO update with correct accelerator synchronization."""
    synchronize_device(device)
    t0 = time.perf_counter()
    stats = ppo.update(
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
    return time.perf_counter() - t0, stats


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
        "--precision",
        type=str,
        nargs="+",
        choices=("fp32", "bf16"),
        default=["fp32"],
        help="PPO compute modes to sweep. BF16 uses CUDA autocast only.",
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        nargs="+",
        choices=COMPILE_MODES,
        default=["eager"],
        help=(
            "Model execution modes to sweep. 'eager' disables torch.compile; "
            "'default' is torch.compile with default settings. Other values map "
            "to torch.compile(mode=...). Compilation is warmed before timing."
        ),
    )
    parser.add_argument(
        "--micro-repeats",
        type=int,
        default=3,
        help="Independent repeats for the one-minibatch microbenchmark.",
    )
    parser.add_argument(
        "--update-repeats",
        type=int,
        default=3,
        help="Independent complete-update repeats; the median is reported.",
    )
    parser.add_argument(
        "--cycle-repeats",
        type=int,
        default=3,
        help="Independent full-cycle repeats; the median train time is reported.",
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
    if min(args.micro_repeats, args.update_repeats, args.cycle_repeats) <= 0:
        parser.error("all repeat counts must be positive")
    if "bf16" in args.precision:
        if not is_cuda:
            parser.error("--precision bf16 requires a CUDA device")
        if not torch.cuda.is_bf16_supported():
            parser.error("the selected CUDA device does not report BF16 support")
    if any(mode != "eager" for mode in args.compile_mode) and not hasattr(
        torch, "compile"
    ):
        parser.error("this PyTorch build does not provide torch.compile")

    root = Path("logs/rl") / f"benchmark_fixed_{int(time.time())}"
    root.mkdir(parents=True, exist_ok=True)
    results_path = root / "results.json"
    results = []

    pbar = tqdm(
        total=(
            len(args.workers)
            * len(args.n_minibatches)
            * len(args.precision)
            * len(args.compile_mode)
        ),
        desc="Sweeping Benchmarks",
    )

    def safe_print(msg=""):
        pbar.write(str(msg))

    print = safe_print
    print(f"Resolved device: {device}")
    print(f"PyTorch: {torch.__version__}; CUDA runtime: {torch.version.cuda}")
    if is_cuda:
        print(
            f"GPU: {torch.cuda.get_device_name(device)}; "
            f"BF16 supported: {torch.cuda.is_bf16_supported()}"
        )
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
            precision=args.precision[0],
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
                precision=args.precision[0],
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
                for precision in args.precision:
                    for compile_mode in args.compile_mode:
                        print(
                            f"\n--- Benchmarking precision={precision}, "
                            f"compile={compile_mode}, n_minibatches={nmb} "
                            f"(workers={workers}) ---"
                        )

                        cfg = make_config(
                            workers=workers,
                            n_minibatches=nmb,
                            precision=precision,
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
                                tag=f"{precision}_{compile_mode}_micro{rep}",
                                root=root,
                                collection_threads=collection_threads,
                                update_threads=update_threads,
                            )
                            restore_training_state(micro_run, baseline_state)
                            configure_compile(micro_run, compile_mode)
                            warm_compiled_update(
                                micro_run,
                                mb_rollout,
                                mb_optim,
                                baseline_state,
                                compile_mode=compile_mode,
                                seed=seed,
                                device=device,
                            )
                            micro_time, _ = timed_update(
                                micro_run,
                                mb_rollout,
                                mb_optim,
                                seed=seed,
                                device=device,
                            )
                            micro_times.append(micro_time)
                            del micro_run
                        mb_time = statistics.median(micro_times)

                        # -------------------------------------------------------------
                        # Benchmark 2: complete update over the same frozen rollout.
                        # Every repeat starts from the identical model/optimizer snapshot;
                        # report the median wall time and diagnostics from the closest run.
                        # -------------------------------------------------------------
                        update_records = []
                        for rep in range(args.update_repeats):
                            update_run = make_run(
                                cfg,
                                workers=workers,
                                n_minibatches=nmb,
                                tag=f"{precision}_{compile_mode}_update{rep}",
                                root=root,
                                collection_threads=collection_threads,
                                update_threads=update_threads,
                            )
                            restore_training_state(update_run, baseline_state)
                            configure_compile(update_run, compile_mode)
                            warm_compiled_update(
                                update_run,
                                rollout,
                                cfg.optim,
                                baseline_state,
                                compile_mode=compile_mode,
                                seed=seed,
                                device=device,
                            )

                            if is_cuda:
                                torch.cuda.reset_peak_memory_stats()

                            update_time_rep, update_stats_rep = timed_update(
                                update_run,
                                rollout,
                                cfg.optim,
                                seed=seed,
                                device=device,
                            )
                            peak_alloc_rep = None
                            peak_reserved_rep = None
                            if is_cuda:
                                peak_alloc_rep = torch.cuda.max_memory_allocated() / (
                                    1024**3
                                )
                                peak_reserved_rep = torch.cuda.max_memory_reserved() / (
                                    1024**3
                                )
                            update_records.append((
                                update_time_rep,
                                update_stats_rep,
                                peak_alloc_rep,
                                peak_reserved_rep,
                            ))
                            del update_run

                        update_time = statistics.median(r[0] for r in update_records)
                        update_record = min(
                            update_records, key=lambda r: abs(r[0] - update_time)
                        )
                        update_stats = update_record[1]
                        peak_alloc_gib = (
                            statistics.median(r[2] for r in update_records)
                            if is_cuda
                            else None
                        )
                        peak_reserved_gib = (
                            statistics.median(r[3] for r in update_records)
                            if is_cuda
                            else None
                        )

                        # -------------------------------------------------------------
                        # Benchmark 3: full collect + update cycle. Each repeat is a fresh
                        # Run restored to the same baseline. Choose the run closest to the
                        # median collect+update time so all reported components stay coherent.
                        # -------------------------------------------------------------
                        cycle_records = []
                        for rep in range(args.cycle_repeats):
                            cycle_run = make_run(
                                cfg,
                                workers=workers,
                                n_minibatches=nmb,
                                tag=f"{precision}_{compile_mode}_cycle{rep}",
                                root=root,
                                collection_threads=collection_threads,
                                update_threads=update_threads,
                            )
                            restore_training_state(cycle_run, baseline_state)
                            configure_compile(cycle_run, compile_mode)
                            # Warm on the frozen rollout rather than an extra full cycle:
                            # this excludes compile startup without additionally warming
                            # the CPU collector for compiled measurements.
                            warm_compiled_update(
                                cycle_run,
                                rollout,
                                cfg.optim,
                                baseline_state,
                                compile_mode=compile_mode,
                                seed=seed,
                                device=device,
                            )

                            synchronize_device(device)
                            t0 = time.perf_counter()
                            stats_rep, extra_rep = _iteration(
                                cycle_run, collector, update_threads
                            )
                            synchronize_device(device)
                            wall_rep = time.perf_counter() - t0
                            train_rep = extra_rep["collect_s"] + extra_rep["update_s"]
                            cycle_records.append((
                                train_rep,
                                wall_rep,
                                stats_rep,
                                extra_rep,
                            ))
                            del cycle_run

                        train_cycle_time = statistics.median(
                            r[0] for r in cycle_records
                        )
                        cycle_record = min(
                            cycle_records, key=lambda r: abs(r[0] - train_cycle_time)
                        )
                        _, wall_cycle_time, stats, extra = cycle_record
                        rollout_t = extra["collect_s"]
                        actual_update_t = extra["update_s"]
                        actor_presentations = extra["actor_presentations"]

                        # Actor early stopping occurs only between complete epochs, so the
                        # exact presentation count divided by completed epochs recovers the
                        # fresh rollout size without assuming configured epoch count.
                        cycle_rollout_size = (
                            actor_presentations / stats.actor_epochs
                            if stats.actor_epochs
                            else 0.0
                        )
                        cycle_matches_frozen_shape = (
                            int(cycle_rollout_size) == rollout_size
                        )

                        unique_env_throughput = (
                            cycle_rollout_size / train_cycle_time
                            if train_cycle_time > 0
                            else float("inf")
                        )
                        rollout_throughput = (
                            cycle_rollout_size / rollout_t
                            if rollout_t > 0
                            else float("inf")
                        )
                        update_throughput = (
                            actor_presentations / actual_update_t
                            if actual_update_t > 0
                            else float("inf")
                        )

                        sanity_values = [
                            update_stats.surrogate,
                            update_stats.critic_mse,
                            update_stats.entropy,
                            update_stats.approx_kl,
                            update_stats.clip_fraction,
                            update_stats.max_abs_log_ratio,
                            update_stats.actor_grad_norm,
                            update_stats.critic_grad_norm,
                            *update_stats.kl_per_epoch,
                        ]
                        numerically_finite = all(
                            math.isfinite(float(x)) for x in sanity_values
                        )
                        if compile_mode != "eager" and not cycle_matches_frozen_shape:
                            print(
                                "NOTE: timed full-cycle rollout size differed from the frozen "
                                "compile-warmup rollout; any shape recompile cost is therefore "
                                "part of the measured cycle (representative of changing rollout shapes)."
                            )
                        print(
                            f"1 Minibatch (~{mb_rollout.n_decisions} decisions; {nmb} per epoch): {mb_time:.4f}s"
                        )
                        print(
                            f"1 Complete Update ({rollout_size} decisions, "
                            f"{cfg.optim.actor_epochs} actor / {cfg.optim.critic_epochs} critic epochs): "
                            f"{update_time:.4f}s median ({args.update_repeats} repeats)"
                        )
                        print(
                            f"Update diagnostics: kl={update_stats.approx_kl:.6f}, "
                            f"clip={update_stats.clip_fraction:.4f}, "
                            f"actor|g|={update_stats.actor_grad_norm:.4f}, "
                            f"critic_mse={update_stats.critic_mse:.6f}, "
                            f"finite={numerically_finite}"
                        )
                        print(
                            f"Full Cycle: train={train_cycle_time:.4f}s "
                            f"(collect={rollout_t:.4f}s + update={actual_update_t:.4f}s), "
                            f"wall={wall_cycle_time:.4f}s "
                            f"({args.cycle_repeats} repeats; median train time)"
                        )
                        print(
                            f"Unique env throughput: {unique_env_throughput:,.0f} decisions/s"
                        )
                        print(
                            f"Rollout throughput: {rollout_throughput:,.0f} decisions/s"
                        )
                        print(
                            f"Update throughput: {update_throughput:,.0f} presentations/s"
                        )
                        if is_cuda:
                            print(
                                f"Peak CUDA memory (complete update): "
                                f"allocated={peak_alloc_gib:.2f} GiB, "
                                f"reserved={peak_reserved_gib:.2f} GiB"
                            )

                        row = {
                            "workers": workers,
                            "precision": precision,
                            "compile_mode": compile_mode,
                            "torch_version": torch.__version__,
                            "cuda_runtime": torch.version.cuda,
                            "device": str(device),
                            "gpu_name": torch.cuda.get_device_name(device)
                            if is_cuda
                            else None,
                            "n_minibatches": nmb,
                            "frozen_rollout_decisions": rollout_size,
                            "micro_minibatch_decisions": mb_rollout.n_decisions,
                            "approx_decisions_per_minibatch": rollout_size
                            / effective_nmb,
                            "micro_median_s": mb_time,
                            "micro_repeats": args.micro_repeats,
                            "complete_update_s": update_time,
                            "complete_update_repeats": args.update_repeats,
                            "update_surrogate": update_stats.surrogate,
                            "update_critic_mse": update_stats.critic_mse,
                            "update_entropy": update_stats.entropy,
                            "update_approx_kl": update_stats.approx_kl,
                            "update_clip_fraction": update_stats.clip_fraction,
                            "update_max_abs_log_ratio": update_stats.max_abs_log_ratio,
                            "update_actor_grad_norm": update_stats.actor_grad_norm,
                            "update_critic_grad_norm": update_stats.critic_grad_norm,
                            "update_kl_per_epoch": update_stats.kl_per_epoch,
                            "update_numerically_finite": numerically_finite,
                            "cycle_repeats": args.cycle_repeats,
                            "cycle_unique_decisions": cycle_rollout_size,
                            "cycle_matches_frozen_compile_shape": cycle_matches_frozen_shape,
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
                        results_path.write_text(
                            json.dumps(results, indent=2), encoding="utf-8"
                        )
                        pbar.update(1)
        finally:
            if collector is not None:
                collector.close()

    pbar.close()
    print(f"\nSaved {len(results)} benchmark rows to {results_path}")


if __name__ == "__main__":
    main()
