"""Portable training checkpoints and the actor-only export.

docs/rl/pipeline-plan.md section 11. A checkpoint is written only at a completed
**update boundary**, to a temporary file that is then atomically replaced, with a
checksum manifest beside it. It carries:

- actor and critic parameters, optimizer states and any scheduler state;
- the resolved full configuration and separate immutable algorithm hash, the
  feature/action/rules/league schemas and their hashes, the model architecture and
  a format version;
- the next update and episode IDs, the deal schedule position, the RNG states and
  the stream derivation scheme;
- cumulative counters and time, the next evaluation threshold, the selected-best
  score and checkpoint ID, the panel manifest and the lineage/source run ID.

No worker pool, live game, open queue or unfinished rollout is serialized. Tensors
are stored on CPU and optimizer state is remapped on load, so a checkpoint moves
between the Mac and the desktop.

**What portability is and is not.** Run caps, evaluation settings, device, worker
count and live-game width may change while the algorithm hash keeps batch
construction, seed, model, observation schema and optimizer fixed. Continuing
elsewhere preserves progress and algorithm configuration, not bitwise numerical
trajectories. Saving RNG states is not a proof of exact continuation: that is a
tested property only where deterministic kernels and orderings support it.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from uno.rl.config import Config, algorithm_hash
from uno.rl.features import spec
from uno.rl.model import Actor, Critic
from uno.rl.opponents import league_hash
from uno.rl.streams import STREAM_VERSION

FORMAT_VERSION = 1


@dataclass
class Progress:
    """The cumulative counters a resume must continue rather than restart."""

    update: int = 0
    next_episode_id: int = 0
    learner_decisions: int = 0
    engine_decisions: int = 0
    games: int = 0
    optimizer_steps: int = 0
    active_seconds: float = 0.0
    next_eval_at: int = 0
    best_score: float | None = None
    best_checkpoint: str | None = None
    best_update: int | None = None
    best_decisions: int | None = None
    deal_schedule: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def _checksum(path: Path) -> str:
    h = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _cpu_state(module) -> dict:
    return {k: v.detach().to("cpu").clone() for k, v in module.state_dict().items()}


def save(
    path: str | Path,
    config: Config,
    actor: Actor,
    critic: Critic,
    actor_opt: torch.optim.Optimizer,
    critic_opt: torch.optim.Optimizer,
    progress: Progress,
    lineage: dict | None = None,
) -> dict:
    """Write a checkpoint atomically and return its manifest.

    Failure to write a recoverable checkpoint is a reason to stop gracefully, so
    this raises rather than warning: the caller decides to stop.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rs = config.rs
    payload = {
        "format_version": FORMAT_VERSION,
        "saved_at": time.time(),
        "config": config.resolved(),
        "config_hash": config.hash,
        "algorithm_hash": config.algorithm_hash,
        "ruleset_repr": repr(rs),
        "actor_schema": spec(rs, config.arm).hash,
        "critic_schema": spec(rs, "CRITIC").hash,
        "league_hash": league_hash(rs, lineage.get("league", "rulebot")) if lineage else None,
        "stream_version": STREAM_VERSION,
        "actor": _cpu_state(actor),
        "critic": _cpu_state(critic),
        "actor_opt": actor_opt.state_dict(),
        "critic_opt": critic_opt.state_dict(),
        "progress": progress.as_dict(),
        "rng": {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        "lineage": lineage or {},
    }
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    manifest = {
        "file": path.name,
        "bytes": path.stat().st_size,
        "checksum": _checksum(path),
        "format_version": FORMAT_VERSION,
        "config_hash": config.hash,
        "algorithm_hash": config.algorithm_hash,
        "progress": progress.as_dict(),
        "lineage": lineage or {},
    }
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    tmp_manifest = manifest_path.with_suffix(f".tmp{os.getpid()}")
    tmp_manifest.write_text(json.dumps(manifest, indent=2, default=str))
    os.replace(tmp_manifest, manifest_path)
    return manifest


def load(path: str | Path, map_location: str = "cpu") -> dict:
    payload = torch.load(Path(path), weights_only=False, map_location=map_location)
    if payload["format_version"] != FORMAT_VERSION:
        raise ValueError(f"checkpoint format {payload['format_version']} != {FORMAT_VERSION}")
    return payload


def verify(path: str | Path) -> bool:
    """Whether the checksum manifest still matches the file."""
    path = Path(path)
    manifest = json.loads(path.with_suffix(path.suffix + ".manifest.json").read_text())
    return manifest["checksum"] == _checksum(path)


def restore(payload: dict, config: Config, actor: Actor, critic: Critic, actor_opt=None, critic_opt=None, device: str = "cpu") -> Progress:
    """Load into live objects, refusing an incompatible branch.

    Device, worker count and artifact paths may change freely; the observation
    schema, the algorithm configuration and the rules may not.
    """
    if payload["ruleset_repr"] != repr(config.rs):
        raise ValueError(f"checkpoint is for another ruleset: {payload['ruleset_repr']}")
    if payload["actor_schema"] != spec(config.rs, config.arm).hash:
        raise ValueError("checkpoint's actor feature schema differs: that is an experiment branch, not a resume")
    if payload["critic_schema"] != spec(config.rs, "CRITIC").hash:
        raise ValueError("checkpoint's critic feature schema differs: that is an experiment branch, not a resume")
    saved_algorithm_hash = payload.get("algorithm_hash") or algorithm_hash(payload["config"])
    if saved_algorithm_hash != config.algorithm_hash:
        raise ValueError("checkpoint's algorithm configuration differs: branch the experiment explicitly")
    actor.load_state_dict(payload["actor"])
    critic.load_state_dict(payload["critic"])
    actor.to(device)
    critic.to(device)
    if actor_opt is not None:
        actor_opt.load_state_dict(payload["actor_opt"])
        _remap(actor_opt, device)
    if critic_opt is not None:
        critic_opt.load_state_dict(payload["critic_opt"])
        _remap(critic_opt, device)
    rng = payload.get("rng") or {}
    if rng.get("torch") is not None:
        torch.set_rng_state(rng["torch"].to("cpu", torch.uint8) if hasattr(rng["torch"], "to") else rng["torch"])
    if rng.get("numpy") is not None:
        np.random.set_state(rng["numpy"])
    return Progress(**payload["progress"])


def _remap(opt: torch.optim.Optimizer, device: str) -> None:
    for state in opt.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


def export_actor(path: str | Path, config: Config, actor: Actor, name: str, progress: Progress) -> dict:
    """A small actor-only export for evaluation and deployment: no critic, no
    optimizer state, no training metadata that could become actor input."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rs = config.rs
    payload = {
        "format_version": FORMAT_VERSION,
        "name": name,
        "ruleset_repr": repr(rs),
        "arm": config.arm,
        "width": config.width,
        "blocks": config.blocks,
        "temperature": config.optim.temperature,
        "actor_schema": spec(rs, config.arm).hash,
        "actor": _cpu_state(actor),
        "trained_decisions": progress.learner_decisions,
        "update": progress.update,
    }
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return {"file": path.name, "checksum": _checksum(path), "name": name}


def load_actor(path: str | Path, rs=None):
    """Rebuild an actor export only under the exact saved feature schema."""
    from uno.game.rules import NAMED
    from uno.rl.policy import ActorPolicy, NeuralAgent

    payload = torch.load(Path(path), weights_only=False, map_location="cpu")
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"actor export format {payload.get('format_version')} != {FORMAT_VERSION}"
        )
    if rs is None:
        rs = next(r for r in NAMED.values() if repr(r) == payload["ruleset_repr"])
    elif repr(rs) != payload["ruleset_repr"]:
        raise ValueError(f"export is for another ruleset: {payload['ruleset_repr']}")
    candidates = [spec(rs, payload["arm"])]
    # Schema 2 is retained as an explicit, read-only evaluator for the diagnostic
    # launched under it. Unknown and schema-1 exports remain rejected.
    candidates.append(spec(rs, payload["arm"], version=2))
    sp = next(
        (candidate for candidate in candidates if payload.get("actor_schema") == candidate.hash),
        None,
    )
    if sp is None:
        raise ValueError(
            "actor export's feature schema differs; evaluate it with its original "
            "versioned extractor rather than silently changing the policy"
        )
    actor = Actor(sp, payload["width"], payload["blocks"])
    actor.load_state_dict(payload["actor"])
    return NeuralAgent(ActorPolicy(sp, actor, payload["temperature"]), payload["name"]), payload


__all__ = ["FORMAT_VERSION", "Progress", "export_actor", "load", "load_actor", "restore", "save", "verify"]
