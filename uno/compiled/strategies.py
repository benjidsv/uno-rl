"""Strategies over a keying of a compiled tree, from agents and from
best-response tables. See docs/spec.md sections 16 and 18.
"""

from __future__ import annotations

import numpy as np

from uno.agents.baseline import Agent
from uno.compiled.solvers import BestResponse, best_response
from uno.compiled.tree import CompiledTree, Key, Strategy
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


def br_row(tree: CompiledTree, opponent: Strategy, completion: np.ndarray | None = None, level: Level = Level.L1) -> tuple[Strategy, tuple[BestResponse, BestResponse]]:
    """The ladder's "BR Lx(opponent)" row agent: both seats' best responses to
    `opponent` as one strategy over the keying (each seat's sets from its own best
    response), and the two best responses. `completion` is played at the sets the
    opponent never lets the responder reach; None keeps the solver's first orbit
    there (docs/spec.md section 16)."""
    brs = (best_response(tree, level, opponent, 0, completion), best_response(tree, level, opponent, 1, completion))
    k = tree.keyings[level]
    return (level, np.where(k.info_player[k.ia_info] == 0, brs[0].sigma, brs[1].sigma)), brs
