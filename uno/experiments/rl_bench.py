"""Benchmark PPO loops using torch.utils.benchmark and standard timers."""

import time
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.utils.benchmark as benchmark

from uno.rl import ppo
from uno.rl.config import Collection, Config, Evaluation, Optim
from uno.rl.features import ARM_B, Features
from uno.rl.train import Run, _iteration
from uno.rl.parallel import ParallelCollector, WorkerConfig
from uno.rl.rollout import Rollout
from uno.rl.device import resolve_device, synchronize_device


def slice_rollout(r: Rollout, size: int) -> Rollout:
    """Slice the rollout to the given number of decisions."""
    if size >= r.n_decisions:
        return r
    actor_features = Features(r.actor.tok[:size], r.actor.col[:size], r.actor.glob[:size])
    critic_features = Features(r.critic.tok[:size], r.critic.col[:size], r.critic.glob[:size])
    return replace(r,
        actor=actor_features,
        critic=critic_features,
        masks=r.masks[:size],
        actions=r.actions[:size],
        old_logp=r.old_logp[:size],
        episode=r.episode[:size],
        decision_index=r.decision_index[:size],
        set_id=r.set_id[:size] if r.set_id is not None else None,
    )


def main():
    # Exact settings from current run settings
    ruleset = "FULL_UNO"
    arm = ARM_B
    seed = 0
    cap = 108_000
    workers = 1
    games_per_cell = 64
    device = "auto"
    collection_threads = 1
    update_threads = 6
    minibatch = 512
    optim_overrides = {
        "normalize_advantages": True,
        "entropy_coef": 0.01,
        "actor_grad_norm": 0.5,
        "actor_lr": 1e-4,
        "actor_epochs": 4,
        "critic_grad_norm": 0.5,
        "critic_lr": 1e-4,
    }

    device = resolve_device(device)
    print(f"Resolved device: {device}")

    optim = Optim(minibatch=minibatch, **optim_overrides)
    cfg = Config(
        ruleset=ruleset,
        arm=arm,
        training_seed=seed,
        data_cap=cap,
        device=device,
        collection=Collection(games_per_cell=games_per_cell, live_games_per_worker=64, workers=workers),
        evaluation=Evaluation(deals_per_cell=256, every_fraction=0.016),
        optim=optim,
    )

    out = Path("logs/rl") / f"benchmark_{ruleset}_{arm}_{device}_{int(time.time())}"
    run = Run(cfg, out, "rulebot")
    run.execution = {
        "collection_torch_threads": collection_threads,
        "update_torch_threads": update_threads,
    }

    collector = None
    if workers > 1:
        wc = WorkerConfig(
            ruleset=ruleset,
            arm=arm,
            width=cfg.width,
            blocks=cfg.blocks,
            league="rulebot",
            training_seed=seed,
            live_games=cfg.collection.live_games_per_worker,
        )
        collector = ParallelCollector(wc, workers)

    try:
        print("Warming up (1 full cycle)...")
        _iteration(run, collector, update_threads)

        print("Collecting rollout for benchmarks...")
        from uno.rl.collect import collect
        from uno.rl import manifest as mf
        from uno.rl.features import spec
        
        manifest, next_id = mf.build(
            run.progress.update,
            list(run.opponents),
            run.schedule,
            cfg.collection.games_per_cell,
            run.progress.next_episode_id,
        )
        critic_spec = spec(cfg.rs, "CRITIC")
        if collector is None:
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
        else:
            rollout, _ = collector.collect(
                manifest, run.actor, critic_spec, run.progress.update
            )

        rollout_size = rollout.n_decisions
        print(f"Collected rollout of size {rollout_size} decisions.")

        # Benchmark 1: 1 Minibatch (fast microbenchmark, safe to use timeit)
        print("Benchmarking 1 minibatch...")
        mb_rollout = slice_rollout(rollout, minibatch)
        mb_optim = replace(cfg.optim, actor_epochs=1, critic_epochs=1, minibatch=minibatch)
        t_mb = benchmark.Timer(
            stmt="update(actor, critic, actor_opt, critic_opt, r, o, seed, 0, device)",
            globals={
                "update": ppo.update, "actor": run.actor, "critic": run.critic,
                "actor_opt": run.actor_opt, "critic_opt": run.critic_opt,
                "r": mb_rollout, "o": mb_optim, "seed": seed, "device": device,
            },
            num_threads=update_threads,
        ).timeit(10)
        mb_time = t_mb.mean
        
        # Benchmark 2: 1 Complete Update (macro benchmark, use single pass to avoid 20m calibrations)
        print("Benchmarking 1 complete update...")
        t0 = time.perf_counter()
        ppo.update(run.actor, run.critic, run.actor_opt, run.critic_opt, rollout, cfg.optim, seed, 0, device)
        synchronize_device(device)
        update_time = time.perf_counter() - t0
        
        # Benchmark 3: Full Cycle (macro benchmark, use single pass)
        print("Benchmarking full cycle (collect + update)...")
        t0 = time.perf_counter()
        stats, extra = _iteration(run, collector, update_threads)
        cycle_time = time.perf_counter() - t0

        print("\n--- Benchmark Results ---")
        print(f"1 Minibatch ({minibatch} decisions): {mb_time:.4f}s")
        print(f"1 Complete Update ({rollout_size} decisions, {cfg.optim.actor_epochs} actor / {cfg.optim.critic_epochs} critic epochs): {update_time:.4f}s")
        print(f"Full Cycle (Collect + Update): {cycle_time:.4f}s")

        # Throughputs
        # Use the actual rollout size and timings from the full cycle execution we just performed,
        # otherwise a smaller cycle collection can falsely inflate throughputs if we use the fixed `rollout_size`
        cycle_rollout_size = extra["actor_presentations"] / stats.actor_epochs
        rollout_t = extra["collect_s"]
        actual_update_t = extra["update_s"]

        unique_env_throughput = cycle_rollout_size / cycle_time
        rollout_throughput = cycle_rollout_size / rollout_t if rollout_t > 0 else float('inf')
        update_throughput = cycle_rollout_size * cfg.optim.actor_epochs / actual_update_t

        print("\n--- Derived Throughputs (from Full Cycle) ---")
        print(f"Unique env throughput: {unique_env_throughput:,.0f} decisions/s (rollout_size / total_cycle_t)")
        print(f"Rollout throughput: {rollout_throughput:,.0f} decisions/s (rollout_size / rollout_t)")
        print(f"Update throughput: {update_throughput:,.0f} presentations/s (rollout_size * epochs / update_t)")

    finally:
        if collector is not None:
            collector.close()

if __name__ == "__main__":
    main()
