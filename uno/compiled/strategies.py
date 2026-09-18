"""Strategies over a keying of a compiled tree, from agents and from
best-response tables. See docs/foundations/spec.md sections 16 and 18.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from uno.agents.baseline import Agent
from uno.compiled.solvers import BestResponse, best_response
from uno.compiled.tree import CompiledTree, Key, Strategy
from uno.game.keys import decode_key
from uno.game.observe import Level, batch_l1_observations, observation_from_key
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
        raise ValueError(f"{agent.name} is not symmetric: it is not uniform within some orbit of a canonical set (docs/foundations/spec.md section 18)")
    return sigma


class BatchPolicy(Protocol):
    """An agent that answers a whole batch of information sets in one call.

    The seam `tabulate_batched` is written against, and the reason it exists: a network
    asked one set at a time is dominated by its framework's per-call dispatch, not by
    its arithmetic. Measured in step 10a on this project's own value-network shape,
    float64 CPU, one thread: **69-92 us a row at batch 1 against 2.4-3.3 us at batch
    256+**, a 25-30x gap at every ruleset width (`TOY`, `TOY2`, `FULL_UNO`). That is
    the same phenomenon `uno.eval.value_net.NetLeaf` was built for in step 5.21, one
    level up: there it was a search's leaves, here it is a report card's information
    sets.

    `policy_batch` receives the sets' observations as arrays (`L1Batch`) and their
    actions in the keying's own `ia` layout, and returns one probability per `ia`.
    Working in `ia` order rather than returning per-set dicts is what keeps the whole
    path vectorised: the result scatters into `sigma` with no Python loop, and
    `Keying.orbit_mean` symmetrises it in one `bincount`.
    """

    name: str

    def policy_batch(self, batch, ia_action: np.ndarray, info_start: np.ndarray) -> np.ndarray:
        """Probabilities over `ia_action`, summing to 1 within each set's slice of it."""
        ...


def tabulate_batched(tree: CompiledTree, level: Level, policy: BatchPolicy, chunk: int = 65_536) -> np.ndarray:
    """`tabulate` for an agent that answers a batch at a time (item A2 of step 10a).

    Same result as `tabulate`, and required to be **array-equal** to it for an agent
    that implements both paths (`tests/compiled/test_strategies.py`) -- a report card
    that reproduced its pinned values only to a tolerance would not be the step 9.5f
    "reproduces to the digit" device.

    Two costs are removed, because the measurement said both had to be. Batching the
    forward pass alone is not enough: with the policy call down from ~78 us to ~3,
    `observation_from_key` becomes the dominant term (it is already 57-62 % of
    `tabulate` with a *rulebot*, whose policy costs 1.7 us), so the observations are
    rebuilt in batch too (`uno.game.observe.batch_l1_observations`, 3.0-3.7x).

    Symmetry is handled exactly as in `tabulate`, and is nearly free here: on a
    quotient tree every set's key is already canonical, so there is no `canonical`
    call to make -- only the orbit averaging, which `Keying.orbit_mean` does over the
    whole vector in one `bincount`. An agent that is equivariant by construction is
    already a fixed point of it; one that is not is caught by the same
    `is_symmetric` check the row-at-a-time path applies.
    """
    if level is not Level.L1:
        raise ValueError("tabulate_batched is L1 only (uno.game.observe.batch_l1_observations)")
    k = tree.keyings[level]
    sigma = np.zeros(k.n_ia)
    ids = np.fromiter(k.keys.values(), dtype=np.int64, count=len(k.keys))
    encoded = list(k.keys.keys())
    order = np.argsort(ids)  # sets in id order, so a chunk's ia range is contiguous
    with bar(total=k.n_info, desc=f"tabulate {getattr(policy, 'name', '?')}", unit="sets") as progress:
        for start in range(0, k.n_info, chunk):
            sel = order[start : start + chunk]
            keys = [decode_key(level, encoded[j]) for j in sel]
            batch = batch_l1_observations(tree.rs, keys)
            lo, hi = int(k.info_start[ids[sel[0]]]), int(k.info_start[ids[sel[-1]] + 1])
            probs = policy.policy_batch(batch, k.ia_action[lo:hi], k.info_start[ids[sel]] - lo)
            if len(probs) != hi - lo:
                raise ValueError(f"policy_batch returned {len(probs)} probabilities for {hi - lo} set-actions")
            sigma[lo:hi] = probs
            progress.update(len(sel))
    if tree.symmetric and not k.is_symmetric(sigma, 1e-9):
        raise ValueError(f"{policy.name} is not symmetric: it is not uniform within some orbit of a canonical set (docs/foundations/spec.md section 18)")
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
    there (docs/foundations/spec.md section 16)."""
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
