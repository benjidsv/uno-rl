import argparse
import json
import subprocess
import sys
from itertools import product
from pathlib import Path

from uno.rl.core import checkpoint as ckpt
from uno.rl.core.config import Config, algorithm_resolved
from uno.rl.core.config_io import (
    apply_overrides,
    from_dict,
    load,
    load_preset,
    override_seed,
)
from uno.rl.core.runner import run


def combinations(grid: dict):
    names = list(grid.keys())
    values = [grid[name] for name in names]

    for combination in product(*values):
        yield dict(zip(names, combination))


def run_sweep(path: Path) -> None:
    import tomllib
    with path.open("rb") as f:
        sweep = tomllib.load(f)

    base = sweep["base"]
    grid = sweep["sweep"]

    for overrides in combinations(grid):
        command = [
            sys.executable,
            "-m",
            "uno.rl.cli",
            "train",
            base,
        ]

        for key, value in overrides.items():
            command += [
                "--set",
                f"{key}={json.dumps(value)}",
            ]

        subprocess.run(command, check=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="uno rl")
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train")
    train_parser.add_argument("preset", nargs="?")
    train_parser.add_argument("--config", type=Path)
    train_parser.add_argument("--set", action="append", default=[])
    train_parser.add_argument("--seed", type=int)

    resume_parser = commands.add_parser("resume")
    resume_parser.add_argument("checkpoint", type=Path)
    resume_parser.add_argument("--set", action="append", default=[])
    
    sweep_parser = commands.add_parser("sweep")
    sweep_parser.add_argument("config", type=Path)

    args = parser.parse_args(argv)

    if args.command == "train":
        if args.config and args.preset:
            parser.error("use either PRESET or --config, not both")

        if args.config:
            spec = load(args.config, args.set)
        else:
            spec = load_preset(
                args.preset or "experiment",
                args.set,
            )

        if args.seed is not None:
            spec = override_seed(spec, args.seed)

        run(spec)
        
    elif args.command == "resume":
        payload = ckpt.load(args.checkpoint)
        
        # We need to construct a TrainSpec from the checkpoint config
        # and then allow safe overrides.
        
        # The payload["config"] is just the resolved Config dictionary.
        # But we need execution, tracking, and league for TrainSpec.
        # We will use defaults for those, but you'd typically want to carry them over
        # if they were saved in the manifest. The tracker saves run_manifest.json,
        # but the checkpoint only has payload["config"]. We'll reconstruct a spec dictionary.
        
        # Let's extract the old algorithm definition
        old_config_dict = payload["config"]
        old_algorithm = algorithm_resolved(old_config_dict)
        
        # Let's apply overrides to the config
        new_config_dict = old_config_dict.copy()
        if args.set:
            # We want to apply overrides on the root dict as if it was a spec config
            # But the user overrides might apply to 'execution', 'tracking' too.
            # So let's wrap it in a mock spec dictionary
            mock_spec_dict = {
                "league": payload.get("league", "rulebot"),
                **new_config_dict,
                "execution": {},
                "tracking": {"kind": "resume"},
            }
            apply_overrides(mock_spec_dict, args.set)
            
            # Extract the config back
            new_config_dict = mock_spec_dict.copy()
            new_config_dict.pop("league", None)
            new_config_dict.pop("execution", None)
            new_config_dict.pop("tracking", None)
            
            new_algorithm = algorithm_resolved(new_config_dict)
            if old_algorithm != new_algorithm:
                # Let's find out what changed to print a nice error
                diff = {
                    k: (old_algorithm.get(k), new_algorithm.get(k))
                    for k in set(old_algorithm) | set(new_algorithm)
                    if old_algorithm.get(k) != new_algorithm.get(k)
                }
                parser.error(f"unsafe continuation overrides changed algorithm invariants: {diff}")
                
            spec = from_dict(mock_spec_dict)
        else:
            # No overrides
            spec = from_dict({**old_config_dict, "tracking": {"kind": "resume"}})

        # The runner handles resuming if we call run()? No, `runner.run()` uses Run() which trains from scratch.
        # Wait, the user specifically noted: "run = resume(checkpoint, cfg, out, _league_for(ruleset), device)"
        # So we need to update runner.run() or have runner.resume()
        
        import time
        from uno.rl.core.device import resolve_device
        from uno.rl.algo.train import resume as resume_run, start_tracker, train
        from uno.rl.algo.parallel import ParallelCollector, WorkerConfig
        
        cfg = spec.config
        device = resolve_device(cfg.device)
        cfg = cfg.with_(device=device)

        out = Path("logs/rl") / f"resume_{cfg.ruleset}_{cfg.arm}_seed{cfg.training_seed}_{int(time.time())}"
        
        training = resume_run(str(args.checkpoint), cfg, out, spec.league, device)
        
        training.execution = {
            "collection_torch_threads": spec.execution.collection_threads,
            "update_torch_threads": spec.execution.update_threads,
        }
        training.execution = {k: v for k, v in training.execution.items() if v != 0}
        
        training.tracker = start_tracker(
            training, 
            study_id=spec.tracking.study_id,
            kind=spec.tracking.kind,
            mlflow_prefix=spec.tracking.name,
        )
        
        collector = None
        if cfg.collection.workers > 1:
            collector = ParallelCollector(
                WorkerConfig(
                    ruleset=cfg.ruleset,
                    arm=cfg.arm,
                    model_config=__import__("dataclasses").asdict(cfg.model),
                    league=spec.league,
                    training_seed=cfg.training_seed,
                    live_games=cfg.collection.live_games_per_worker,
                    temperature=cfg.optim.temperature,
                ),
                cfg.collection.workers,
            )
        try:
            train(training, training.tracker, collector)
        finally:
            if collector is not None:
                collector.close()
                
    elif args.command == "sweep":
        run_sweep(args.config)


if __name__ == "__main__":
    main()
