"""Strategies over a keying of a compiled tree, from agents and from
best-response tables. See docs/spec.md sections 16 and 18.
"""

from __future__ import annotations

import numpy as np

from uno.agents.baseline import Agent
from uno.compiled.tree import CompiledTree, Key
from uno.game.keys import decode_key
from uno.game.observe import Level, observation_from_key


def tabulate(tree: CompiledTree, level: Level, agent: Agent) -> np.ndarray:
    """The agent's policy over every information set of the keying (both seats),
    from the observation rebuilt out of each set's key."""
    k = tree.keyings[level]
    sigma = np.zeros(k.n_ia)
    for kb, i in k.keys.items():
        start = k.info_start[i]
        legal = k.actions(i)
        probs = agent.policy(observation_from_key(tree.rs, level, decode_key(level, kb)), legal)
        for a, p in probs.items():
            sigma[start + legal.index(a)] = p
    if tree.symmetric and not k.is_symmetric(sigma, 1e-9):
        raise ValueError(f"{agent.name} is not symmetric: it is not uniform within some orbit of a canonical set (docs/spec.md section 18)")
    return sigma


def tabulate_table(tree: CompiledTree, level: Level, table: dict[Key, int]) -> np.ndarray:
    """A pure strategy from a key -> action table (a `uno.reference.solve.BestResponse.table`,
    keyed by raw keys; on a symmetric tree the canonical key of each set is looked
    up, which is a raw key of the full game); sets missing from the table take
    their first action."""
    k = tree.keyings[level]
    sigma = np.zeros(k.n_ia)
    for kb, i in k.keys.items():
        a = table.get(decode_key(level, kb))
        legal = k.actions(i)
        sigma[k.info_start[i] + (legal.index(a) if a in legal else 0)] = 1.0
    return k.orbit_mean(sigma)  # on a symmetric tree: the choice spread over its orbit, same value against a symmetric opponent
