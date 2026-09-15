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
from uno.progress import bar


def tabulate(tree: CompiledTree, level: Level, agent: Agent) -> np.ndarray:
    """The agent's policy over every information set of the keying (both seats),
    from the observation rebuilt out of each set's key."""
    k = tree.keyings[level]
    sigma = np.zeros(k.n_ia)
    for kb, i in bar(k.keys.items(), total=len(k.keys), desc=f"tabulate {getattr(agent, 'name', '?')}", unit="sets"):
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


def transfer(src: CompiledTree, dst: CompiledTree, level: Level, sigma: np.ndarray) -> np.ndarray:
    """A strategy over `src`'s keying at `level`, rewritten over `dst`'s keying of the
    same level by matching the sets' keys (step 9). The lazy tree and the world tree of
    a ruleset have the same L1 sets numbered differently, so a strategy tabulated or
    solved on one moves to the other without asking any agent again. Raises unless
    every set of `dst` is in `src` with the same actions."""
    ks, kd = src.keyings[level], dst.keyings[level]
    if len(sigma) != ks.n_ia:
        raise ValueError("sigma is not a strategy over the source keying")
    n = kd.n_info
    ids = np.fromiter(kd.keys.values(), dtype=np.int64, count=n)
    try:
        src_ids = np.fromiter((ks.keys[kb] for kb in kd.keys), dtype=np.int64, count=n)
    except KeyError:
        raise ValueError("a set of the destination keying is missing from the source keying") from None
    d0 = kd.info_start[ids]
    cnt = kd.info_start[ids + 1] - d0
    s0 = ks.info_start[src_ids]
    if not np.array_equal(ks.info_start[src_ids + 1] - s0, cnt):
        raise ValueError("matched sets have different numbers of actions")
    within = np.arange(int(cnt.sum())) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    dst_ia = np.repeat(d0, cnt) + within
    src_ia = np.repeat(s0, cnt) + within
    if not np.array_equal(ks.ia_action[src_ia], kd.ia_action[dst_ia]):
        raise ValueError("matched sets list different actions")
    out = np.zeros(kd.n_ia)
    out[dst_ia] = sigma[src_ia]
    return out
