"""Randomness keyed by logical work, not by process or completion order.

docs/rl/pipeline-plan.md section 4: model initialization, episode/deal
scheduling, learner action sampling, opponent action sampling and minibatch
permutations are separate streams, and the derivation scheme is versioned. A
change of worker count must preserve the assigned episodes and their logical
streams; floating-point batch or device changes may still move sampled
trajectories, and that limitation is not papered over here.

**This differs from the evaluator's sampler.** `uno.eval.play` keys an agent's
action rng on `(game seed, agent name, information state)`. The collector keys on
`(training seed, role, episode id, decision index)` instead, because a collector
must reproduce an *episode* regardless of which worker ran it and of what else
was in its batch, and because two decisions at the same information state within
one episode are different work. Consequently the collector does **not** reproduce
historical evaluator traces, and nothing here claims it does: what is tested is
the distribution and the game semantics (`tests/rl/test_streams.py`). Coupled
evaluation keeps using the existing explicit shared-stream contract.
"""

from __future__ import annotations

import hashlib

import numpy as np

STREAM_VERSION = 1

INIT = "init"
SCHEDULE = "schedule"
ACTOR = "actor"
OPPONENT = "opponent"
MINIBATCH = "minibatch"
EVAL = "eval"

_SCALE = 1.0 / float(1 << 64)


def digest(*parts) -> int:
    """A process-independent 64-bit value for a tuple of logical coordinates."""
    payload = repr((STREAM_VERSION, *parts)).encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def uniform(*parts) -> float:
    """One uniform in [0, 1) for a piece of logical work."""
    return digest(*parts) * _SCALE


def uniforms(role: str, training_seed: int, episode_ids, decision_index: int) -> np.ndarray:
    """One uniform per episode at the same decision index, for a batched decision."""
    return np.fromiter(
        (uniform(training_seed, role, int(e), decision_index) for e in episode_ids),
        dtype=np.float64,
        count=len(episode_ids),
    )


def generator(*parts) -> np.random.Generator:
    """A numpy generator for a named stream (scheduling, minibatch permutations)."""
    return np.random.default_rng(digest(*parts))


def torch_seed(*parts) -> int:
    """A torch seed for model initialization; torch wants a 63-bit value."""
    return digest(*parts) >> 1


__all__ = [
    "ACTOR", "EVAL", "INIT", "MINIBATCH", "OPPONENT", "SCHEDULE", "STREAM_VERSION",
    "digest", "generator", "torch_seed", "uniform", "uniforms",
]  # fmt: skip
