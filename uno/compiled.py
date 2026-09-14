"""Compiled game tree: the lazy-chance tree (or the world tree) as flat numpy
arrays, and the exact computations that become vectorized passes over them.

See RULES.md section 16. Why: the pure-Python solvers of `uno.solve` walk
`State` objects through `step` at ~150k histories/s, so one pass over `TOY`
(1.25M histories) costs 7-15 s. CFR needs hundreds of passes and the RL rows
of the ladder need a best response per checkpoint, so the tree is walked
*once* here and everything else is numpy.

Layout. Nodes are histories (plus a virtual chance root, node 0, whose edges
are the deals or the worlds). A decision node owns one **arc** per legal
action; an arc owns one **edge** per chance outcome (child, probability): a
non-drawing action has one edge, a draw has one per distinct card type left
in the pile. Nodes are sorted by *level*, arcs by node, edges by arc, so every
level is a contiguous slice of all three tables and a pass over the tree is a
loop over ~25 levels of a few numpy operations each.

The level of a decision node is `2 * len(public) + [phase == TURN_START]`. It
grows strictly along every edge (a play or a draw adds a public event; a pass
adds none but moves from AFTER_DRAW to TURN_START) and it is the same for all
histories of an information set at any level, because the phase at a player's
own decision is a function of what they know. Depth by action count would not
do: the opponent's "drew a playable card and passed" is one action longer than
"drew an unplayable card" and looks identical. Terminals get their parent's
level plus one.

A **keying** partitions the decision nodes into information sets under one of
`uno.solve.KEYS` (L1, L2, L3). Information sets are numbered in order of first
appearance so they are contiguous per level, and their actions (**ia**,
information-set x action) are contiguous per set. A strategy for one player,
or a whole profile, is a vector over ia; `weights` turns it into per-arc
weights, which is what the passes consume. Several keyings can share a tree
(L1 and L2 on the lazy tree; L1 on the world tree to pit an L1 policy against
an L3 responder).

Two kernels:

- `reach(w0, w1)`: forward, per node the product of each player's own action
  probabilities and of chance along the path, `(n, 3)`.
- `values(w, util)`: backward, per node the expectation of `util` (any number
  of columns) under per-arc weights `w`; also returns the per-arc values.

Everything else (evaluation, outcome distributions, best response,
exploitability, exact mirrored moments, CFR in `uno.cfr`) is a few lines over
those two. The reference implementations in `uno.solve` / `uno.estimate` are
kept as the independent cross-check (`tests/test_compiled.py`).
"""

from __future__ import annotations

import time
from array import array
from dataclasses import dataclass, field
from typing import Hashable

import numpy as np

from uno.agents import Agent, Policy
from uno.engine import Phase, State, returns
from uno.observe import Level, Observation, observation_from_key
from uno.rules import Ruleset
from uno.solve import KEYS, engine_children, obs_key, worlds
from uno.tree import children, initial_deals

Key = Hashable
CHANCE, TERMINAL = -2, -1


def node_level(s: State) -> int:
    return 2 * len(s.public) + (1 if s.phase is Phase.TURN_START else 0)


# -- keyings ------------------------------------------------------------------


@dataclass
class Keying:
    """Information sets of one level over a compiled tree, and vectors over them."""

    level: Level
    node_info: np.ndarray  # (n,) int32, information set of each decision node, -1 elsewhere
    arc_ia: np.ndarray  # (m,) int32, information-set action of each player arc, -1 for chance
    ia_info: np.ndarray  # (K,) int32
    ia_action: np.ndarray  # (K,) int32 action id
    info_player: np.ndarray  # (k,) int8
    info_level: np.ndarray  # (k,) int32
    info_start: np.ndarray  # (k + 1,) int32, ia range of each set
    keys: dict[Key, int]  # information-set key -> id

    @property
    def n_info(self) -> int:
        return len(self.info_player)

    @property
    def n_ia(self) -> int:
        return len(self.ia_info)

    @property
    def info_n(self) -> np.ndarray:
        return self.info_start[1:] - self.info_start[:-1]

    def info_sum(self, x: np.ndarray) -> np.ndarray:
        """Sum of an ia vector over each information set."""
        return np.add.reduceat(x, self.info_start[:-1])

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """An ia vector of non-negative masses into a strategy: proportional within
        each set, uniform where the set's mass is zero."""
        s = self.info_sum(x)[self.ia_info]
        n = self.info_n[self.ia_info]
        return np.where(s > 0, x / np.where(s > 0, s, 1.0), 1.0 / n)

    def uniform(self) -> np.ndarray:
        return 1.0 / self.info_n[self.ia_info].astype(np.float64)

    def ia_of(self, player: int) -> np.ndarray:
        return np.nonzero(self.info_player[self.ia_info] == player)[0]

    def actions(self, info: int) -> tuple[int, ...]:
        return tuple(int(a) for a in self.ia_action[self.info_start[info] : self.info_start[info + 1]])


# -- the tree -----------------------------------------------------------------


@dataclass(frozen=True)
class LevelSlice:
    level: int
    n0: int  # nodes [n0, nd) are decisions (chance root included), [nd, n1) terminals
    nd: int
    n1: int
    a0: int  # arcs of the level's decision nodes
    a1: int
    e0: int  # edges of those arcs
    e1: int


@dataclass
class CompiledTree:
    rs: Ruleset
    roots_level: Level  # L3: roots are worlds and the tree below is the bare engine; else deals + lazy chance
    player: np.ndarray  # (n,) int8: 0 / 1, TERMINAL, CHANCE (node 0)
    level: np.ndarray  # (n,) int32
    util: np.ndarray  # (n,) float64, seat-0 return at terminals
    arc_node: np.ndarray  # (m,) int32
    arc_action: np.ndarray  # (m,) int32, -1 for the root's arc
    arc_player: np.ndarray  # (m,) int8
    node_arc_start: np.ndarray  # (n + 1,) int32
    edge_arc: np.ndarray  # (e,) int32
    edge_child: np.ndarray  # (e,) int32
    edge_prob: np.ndarray  # (e,) float64
    arc_edge_start: np.ndarray  # (m + 1,) int32
    levels: list[LevelSlice]
    keyings: dict[Level, Keying]
    seconds: float
    arcs_of: tuple[np.ndarray, np.ndarray] = field(init=False)
    terminals: np.ndarray = field(init=False)
    roots: np.ndarray = field(init=False)  # node ids of the deals / worlds, in edge order
    root_prob: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.arcs_of = (np.nonzero(self.arc_player == 0)[0], np.nonzero(self.arc_player == 1)[0])
        self.terminals = np.nonzero(self.player == TERMINAL)[0]
        e0, e1 = self.arc_edge_start[0], self.arc_edge_start[1]
        self.roots = self.edge_child[e0:e1]
        self.root_prob = self.edge_prob[e0:e1]

    @property
    def n(self) -> int:
        return len(self.player)

    @property
    def m(self) -> int:
        return len(self.arc_node)

    @property
    def e(self) -> int:
        return len(self.edge_arc)

    @property
    def n_terminals(self) -> int:
        return len(self.terminals)

    @property
    def n_decisions(self) -> int:
        return self.n - self.n_terminals - 1

    def __str__(self) -> str:
        ks = ", ".join(f"{k.name}: {v.n_info} sets / {v.n_ia} ia" for k, v in self.keyings.items())
        return (
            f"CompiledTree({self.roots_level.name} roots={len(self.roots)} nodes={self.n} decisions={self.n_decisions} "
            f"terminals={self.n_terminals} arcs={self.m} edges={self.e} levels={len(self.levels)} {ks}; {self.seconds:.1f}s)"
        )

    # -- strategies to per-arc weights

    def weights(self, level: Level, sigma: np.ndarray, player: int | None = None) -> np.ndarray:
        """Per-arc weights of a strategy over the `level` keying: the acting player's
        probability on player arcs (only `player`'s if given), 1 elsewhere."""
        k = self.keyings[level]
        w = np.ones(self.m)
        arcs = np.nonzero(k.arc_ia >= 0)[0] if player is None else self.arcs_of[player]
        w[arcs] = sigma[k.arc_ia[arcs]]
        return w

    def profile_weights(self, seat0: tuple[Level, np.ndarray], seat1: tuple[Level, np.ndarray]) -> np.ndarray:
        """Per-arc weights of a profile: each seat's strategy on its own arcs."""
        return self.weights(seat0[0], seat0[1], 0) * self.weights(seat1[0], seat1[1], 1)

    # -- kernels

    def reach(self, w0: np.ndarray, w1: np.ndarray) -> np.ndarray:
        """Forward pass. Column p in {0, 1} is the product of player p's own
        weights along the path to each node, column 2 the product of chance."""
        r = np.zeros((self.n, 3))
        r[0] = 1.0
        for L in self.levels:
            if L.e1 == L.e0:
                continue
            arcs = self.edge_arc[L.e0 : L.e1]
            f = np.stack((w0[arcs], w1[arcs], self.edge_prob[L.e0 : L.e1]), axis=1)
            r[self.edge_child[L.e0 : L.e1]] = r[self.arc_node[arcs]] * f
        return r

    def values(self, w: np.ndarray, util: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Backward pass. `util` is (n_terminals, c) in `terminals` order; returns the
        node values (n, c) and the arc values (m, c) under per-arc weights `w`."""
        util = np.asarray(util, dtype=np.float64)
        if util.ndim == 1:
            util = util[:, None]
        c = util.shape[1]
        v = np.zeros((self.n, c))
        av = np.zeros((self.m, c))
        v[self.terminals] = util
        for L in reversed(self.levels):
            if L.a1 == L.a0:
                continue
            x = self.edge_prob[L.e0 : L.e1, None] * v[self.edge_child[L.e0 : L.e1]]
            av[L.a0 : L.a1] = np.add.reduceat(x, self.arc_edge_start[L.a0 : L.a1] - L.e0, axis=0)
            y = w[L.a0 : L.a1, None] * av[L.a0 : L.a1]
            v[L.n0 : L.nd] = np.add.reduceat(y, self.node_arc_start[L.n0 : L.nd] - L.a0, axis=0)
        return v, av

    # -- terminal utilities

    def outcome_columns(self) -> np.ndarray:
        """(n_terminals, 3) indicators of seat-0 return +1, 0, -1."""
        u = self.util[self.terminals]
        return np.stack((u > 0, u == 0, u < 0), axis=1).astype(np.float64)

    def seat_util(self, seat: int) -> np.ndarray:
        return self.util[self.terminals] * (1.0 if seat == 0 else -1.0)


# -- build --------------------------------------------------------------------


def compile_tree(rs: Ruleset, levels: tuple[Level, ...] = (Level.L1,)) -> CompiledTree:
    """Walk the tree once and compile it, with one keying per requested level.
    Roots are the worlds when L3 is requested (its sets are the histories, which
    the lazy tree merges), the deals otherwise."""
    t0 = time.perf_counter()
    if Level.L3 in levels:
        roots, expand, roots_level = worlds(rs), engine_children, Level.L3
    else:
        roots, expand, roots_level = initial_deals(rs), children, Level.L1
    keyfns = {lv: KEYS[lv] for lv in levels}

    n_player, n_level, n_util = array("b"), array("i"), array("d")
    a_node, a_action = array("i"), array("i")
    e_arc, e_child, e_prob = array("i"), array("i"), array("d")
    n_info = {lv: array("i") for lv in levels}
    infos: dict[Level, dict[Key, tuple[int, tuple[int, ...], int]]] = {lv: {} for lv in levels}  # key -> (id, actions, level)
    info_player = {lv: array("b") for lv in levels}

    # the virtual root and its one arc
    n_player.append(CHANCE), n_level.append(0), n_util.append(0.0)
    for lv in levels:
        n_info[lv].append(-1)
    a_node.append(0), a_action.append(-1)
    stack: list[tuple[State, int]] = []
    for s, p in roots:
        e_arc.append(0), e_child.append(-1), e_prob.append(p)
        stack.append((s, len(e_arc) - 1))

    while stack:
        s, e = stack.pop()
        nid = len(n_player)
        e_child[e] = nid
        if s.terminal:
            n_player.append(TERMINAL)
            n_level.append(n_level[a_node[e_arc[e]]] + 1)
            n_util.append(returns(s)[0])
            for lv in levels:
                n_info[lv].append(-1)
            continue
        p = s.player
        lvl = node_level(s)
        ch = expand(s)
        acts = tuple(ch)
        n_player.append(p), n_level.append(lvl), n_util.append(0.0)
        for lv in levels:
            key = keyfns[lv](s, p)
            rec = infos[lv].get(key)
            if rec is None:
                rec = (len(infos[lv]), acts, lvl)
                infos[lv][key] = rec
                info_player[lv].append(p)
            elif rec[1] != acts:
                raise RuntimeError(f"{lv.name} information set with differing legal actions: {key!r}")
            elif rec[2] != lvl:
                raise RuntimeError(f"{lv.name} information set spanning two levels: {key!r}")
            n_info[lv].append(rec[0])
        for a in acts:
            aid = len(a_node)
            a_node.append(nid), a_action.append(a)
            for c, pr in ch[a]:
                e_arc.append(aid), e_child.append(-1), e_prob.append(pr)
                stack.append((c, len(e_arc) - 1))

    # -- to numpy, sorted: nodes by (level, terminal, dfs), arcs by node, edges by arc
    player = np.frombuffer(n_player, dtype=np.int8).copy()
    level = np.frombuffer(n_level, dtype=np.int32).copy()
    util = np.frombuffer(n_util, dtype=np.float64).copy()
    arc_node = np.frombuffer(a_node, dtype=np.int32).copy()
    arc_action = np.frombuffer(a_action, dtype=np.int32).copy()
    edge_arc = np.frombuffer(e_arc, dtype=np.int32).copy()
    edge_child = np.frombuffer(e_child, dtype=np.int32).copy()
    edge_prob = np.frombuffer(e_prob, dtype=np.float64).copy()
    n, m, e = len(player), len(arc_node), len(edge_arc)

    node_order = np.lexsort((np.arange(n), player == TERMINAL, level))
    new_node = np.empty(n, dtype=np.int32)
    new_node[node_order] = np.arange(n, dtype=np.int32)
    arc_order = np.lexsort((np.arange(m), new_node[arc_node]))
    new_arc = np.empty(m, dtype=np.int32)
    new_arc[arc_order] = np.arange(m, dtype=np.int32)
    edge_order = np.lexsort((np.arange(e), new_arc[edge_arc]))

    player, level, util = player[node_order], level[node_order], util[node_order]
    arc_node, arc_action = new_node[arc_node][arc_order], arc_action[arc_order]
    arc_player = player[arc_node]
    edge_arc, edge_child, edge_prob = new_arc[edge_arc][edge_order], new_node[edge_child][edge_order], edge_prob[edge_order]
    node_arc_start = np.zeros(n + 1, dtype=np.int32)
    np.cumsum(np.bincount(arc_node, minlength=n), out=node_arc_start[1:])
    arc_edge_start = np.zeros(m + 1, dtype=np.int32)
    np.cumsum(np.bincount(edge_arc, minlength=m), out=arc_edge_start[1:])

    slices = []
    for L in np.unique(level):
        n0, n1 = np.searchsorted(level, L, "left"), np.searchsorted(level, L, "right")
        nd = n0 + int(np.count_nonzero(player[n0:n1] != TERMINAL))
        a0, a1 = node_arc_start[n0], node_arc_start[nd]
        slices.append(LevelSlice(int(L), int(n0), int(nd), int(n1), int(a0), int(a1), int(arc_edge_start[a0]), int(arc_edge_start[a1])))

    keyings = {}
    for lv in levels:
        old = np.frombuffer(n_info[lv], dtype=np.int32)[node_order].copy()
        k = len(infos[lv])
        first = np.full(k, n, dtype=np.int64)
        valid = old >= 0
        np.minimum.at(first, old[valid], np.nonzero(valid)[0])
        info_order = np.argsort(first, kind="stable")
        new_info = np.empty(k, dtype=np.int32)
        new_info[info_order] = np.arange(k, dtype=np.int32)
        node_info = np.where(valid, new_info[np.maximum(old, 0)], -1).astype(np.int32)
        n_actions = np.zeros(k, dtype=np.int32)
        keys: dict[Key, int] = {}
        for key, (oid, acts, _) in infos[lv].items():
            i = int(new_info[oid])
            n_actions[i] = len(acts)
            keys[key] = i
        info_start = np.zeros(k + 1, dtype=np.int32)
        np.cumsum(n_actions, out=info_start[1:])
        pos = np.arange(m, dtype=np.int32) - node_arc_start[arc_node]
        arc_ia = np.where(arc_player >= 0, info_start[np.maximum(node_info[arc_node], 0)] + pos, -1).astype(np.int32)
        ia_info = np.repeat(np.arange(k, dtype=np.int32), n_actions)
        ia_action = np.empty(len(ia_info), dtype=np.int32)
        ia_action[arc_ia[arc_ia >= 0]] = arc_action[arc_ia >= 0]
        keyings[lv] = Keying(
            level=lv,
            node_info=node_info,
            arc_ia=arc_ia,
            ia_info=ia_info,
            ia_action=ia_action,
            info_player=np.frombuffer(info_player[lv], dtype=np.int8)[info_order].copy(),
            info_level=level[first[info_order]],
            info_start=info_start,
            keys=keys,
        )

    return CompiledTree(
        rs=rs,
        roots_level=roots_level,
        player=player,
        level=level,
        util=util,
        arc_node=arc_node,
        arc_action=arc_action,
        arc_player=arc_player,
        node_arc_start=node_arc_start,
        edge_arc=edge_arc,
        edge_child=edge_child,
        edge_prob=edge_prob,
        arc_edge_start=arc_edge_start,
        levels=slices,
        keyings=keyings,
        seconds=time.perf_counter() - t0,
    )


# -- strategies from agents ---------------------------------------------------

Strategy = tuple[Level, np.ndarray]  # (keying level, sigma over that keying's ia)


def tabulate(tree: CompiledTree, level: Level, agent: Agent) -> np.ndarray:
    """The agent's policy over every information set of the keying (both seats),
    from the observation rebuilt out of each set's key."""
    k = tree.keyings[level]
    sigma = np.zeros(k.n_ia)
    for key, i in k.keys.items():
        start = k.info_start[i]
        legal = k.actions(i)
        probs = agent.policy(observation_from_key(tree.rs, level, key), legal)
        for a, p in probs.items():
            sigma[start + legal.index(a)] = p
    return sigma


def tabulate_table(tree: CompiledTree, level: Level, table: dict[Key, int]) -> np.ndarray:
    """A pure strategy from a key -> action table (a `uno.solve.BestResponse.table`);
    sets missing from the table take their first action."""
    k = tree.keyings[level]
    sigma = np.zeros(k.n_ia)
    for key, i in k.keys.items():
        a = table.get(key)
        legal = k.actions(i)
        sigma[k.info_start[i] + (legal.index(a) if a in legal else 0)] = 1.0
    return sigma


class CompiledAgent:
    """Plays a strategy over a keying through `uno.play`. Must observe at `level`.
    Sets missing from the keying (never reached in the compiled tree, e.g. under
    a different ruleset) fall back to the first legal action and are counted."""

    def __init__(self, tree: CompiledTree, level: Level, sigma: np.ndarray, name: str = "compiled") -> None:
        self.rs = tree.rs
        self.keying = tree.keyings[level]
        self.level = level
        self.sigma = sigma
        self.name = name
        self.misses = 0

    def policy(self, obs: Observation, legal: tuple[int, ...]) -> Policy:
        if obs.level is not self.level:
            raise ValueError(f"{self.name} needs {self.level.name} observations, got {obs.level.name}")
        k = self.keying
        i = k.keys.get(obs_key(self.rs, obs))
        if i is None:
            self.misses += 1
            return {legal[0]: 1.0}
        s, t = k.info_start[i], k.info_start[i + 1]
        out = {int(a): float(p) for a, p in zip(k.ia_action[s:t], self.sigma[s:t]) if p > 0}
        assert all(a in legal for a in out), "keying and game disagree on legal actions"
        return out


# -- exact computations --------------------------------------------------------


def outcome_distribution(tree: CompiledTree, seat0: Strategy, seat1: Strategy) -> dict[float, float]:
    """`uno.solve.outcome_distribution` on the compiled tree: {+1, 0, -1} -> P."""
    v, _ = tree.values(tree.profile_weights(seat0, seat1), tree.outcome_columns())
    return {1.0: float(v[0, 0]), 0.0: float(v[0, 1]), -1.0: float(v[0, 2])}


def evaluate(tree: CompiledTree, seat0: Strategy, seat1: Strategy) -> tuple[float, float]:
    """Exact expected returns (seat 0, seat 1) of a profile."""
    v, _ = tree.values(tree.profile_weights(seat0, seat1), tree.seat_util(0))
    return (float(v[0, 0]), -float(v[0, 0]))


@dataclass
class BestResponse:
    """A best response computed on the compiled tree. `sigma` is the pure strategy
    over the responder's keying (both seats' sets; the opponent's rows and the
    responder's unreached sets hold their first action)."""

    seat: int
    level: Level
    value: float  # responder's expected return in [-1, 1]
    sigma: np.ndarray
    seconds: float

    @property
    def score(self) -> float:
        return (self.value + 1.0) / 2.0


def best_response(tree: CompiledTree, level: Level, opponent: Strategy, seat: int) -> BestResponse:
    """Best response of `seat` at information level `level` (a keying of the tree)
    to a fixed opponent strategy. One forward pass for the counterfactual reach
    (chance x opponent), one backward pass choosing per information set, from
    the deepest level up, the action maximising the reach-weighted sum of its
    histories' continuation values. Ties: the lowest action id, as `uno.solve`."""
    t0 = time.perf_counter()
    k = tree.keyings[level]
    opp = seat ^ 1
    w_opp = tree.weights(opponent[0], opponent[1], opp)
    ones = np.ones(tree.m)
    r = tree.reach(w_opp if opp == 0 else ones, w_opp if opp == 1 else ones)
    cf = r[:, 2] * r[:, opp]
    v = np.zeros(tree.n)
    av = np.zeros(tree.m)
    v[tree.terminals] = tree.seat_util(seat)
    choice = np.zeros(k.n_ia)
    for L in reversed(tree.levels):
        if L.a1 == L.a0:
            continue
        x = tree.edge_prob[L.e0 : L.e1] * v[tree.edge_child[L.e0 : L.e1]]
        av[L.a0 : L.a1] = np.add.reduceat(x, tree.arc_edge_start[L.a0 : L.a1] - L.e0)
        ia = k.arc_ia[L.a0 : L.a1]
        mine = tree.arc_player[L.a0 : L.a1] == seat
        w = w_opp[L.a0 : L.a1].copy()
        if mine.any():
            k0, k1 = ia[mine].min(), ia[mine].max() + 1
            i0, i1 = k.ia_info[k0], k.ia_info[k1 - 1] + 1
            k0, k1 = k.info_start[i0], k.info_start[i1]  # whole sets of the level's responder arcs
            q = np.bincount(ia[mine] - k0, weights=cf[tree.arc_node[L.a0 : L.a1]][mine] * av[L.a0 : L.a1][mine], minlength=k1 - k0)
            local_info = k.ia_info[k0:k1] - i0
            mx = np.maximum.reduceat(q, k.info_start[i0:i1] - k0)
            is_max = q == mx[local_info]
            first = np.full(i1 - i0, k1 - k0, dtype=np.int64)
            np.minimum.at(first, local_info[is_max], np.nonzero(is_max)[0])
            c = np.zeros(k1 - k0)
            c[first] = 1.0
            choice[k0:k1] = c
            w[mine] = c[ia[mine] - k0]
        v[L.n0 : L.nd] = np.add.reduceat(w * av[L.a0 : L.a1], tree.node_arc_start[L.n0 : L.nd] - L.a0)
    # sets never touched above (no responder arc at their level, i.e. the opponent's) take their first action
    untouched = k.info_sum(choice) == 0
    choice[k.info_start[:-1][untouched]] = 1.0
    return BestResponse(seat=seat, level=level, value=float(v[0]), sigma=choice, seconds=time.perf_counter() - t0)


@dataclass
class Exploitability:
    br: tuple[BestResponse, BestResponse]  # best response to the profile, per seat
    value: float  # the profile's own seat-0 return

    @property
    def nashconv(self) -> float:
        """Sum over seats of (best-response return - own return), in return units."""
        return (self.br[0].value - self.value) + (self.br[1].value + self.value)

    @property
    def score_points(self) -> float:
        """NashConv in score points: score(BR seat 0) + score(BR seat 1) - 1."""
        return self.nashconv / 2.0

    @property
    def per_seat(self) -> tuple[float, float]:
        """How much each seat gains, in score points, by best-responding instead of following the profile."""
        return ((self.br[0].value - self.value) / 2.0, (self.br[1].value + self.value) / 2.0)

    @property
    def bracket(self) -> tuple[float, float]:
        """The Nash seat-0 score lies in [1 - score(BR seat 1), score(BR seat 0)]."""
        return (1.0 - self.br[1].score, self.br[0].score)


def exploitability(tree: CompiledTree, level: Level, sigma: np.ndarray) -> Exploitability:
    """Of a profile given as one strategy over a keying (both seats' sets),
    against best responses at the same level."""
    prof = (level, sigma)
    v = evaluate(tree, prof, prof)[0]
    return Exploitability((best_response(tree, level, prof, 0), best_response(tree, level, prof, 1)), v)


def exact_mirrored(tree: CompiledTree, a: Strategy, b: Strategy):
    """`uno.estimate.exact_mirrored` on a compiled world tree: the moments of `a`'s
    score over worlds when it plays seat 0 against `b`, and seat 1 against `b`.
    The strategies cover both seats' sets, so swapping is a different profile,
    not a different vector. Needs L3 roots (one root per world)."""
    from uno.estimate import ExactMirrored

    if tree.roots_level is not Level.L3:
        raise ValueError("exact_mirrored needs a tree compiled with L3 (world roots)")
    cols = tree.outcome_columns()
    v0, _ = tree.values(tree.profile_weights(a, b), cols)
    v1, _ = tree.values(tree.profile_weights(b, a), cols)
    d0 = v0[tree.roots]  # P(a wins, draw, loses) at seat 0, per world
    d1 = v1[tree.roots][:, ::-1]  # at seat 1: seat-0 return -1 is a's win
    scores = np.array([1.0, 0.5, 0.0])
    e0, e1 = d0 @ scores, d1 @ scores
    q0, q1 = d0 @ scores**2, d1 @ scores**2
    p = tree.root_prob
    m0, m1 = float(p @ e0), float(p @ e1)
    split = float(p @ (d0 * d1[:, ::-1]).sum(axis=1))  # s0 + s1 == 1: (win, loss), (draw, draw), (loss, win)
    return ExactMirrored(
        mean=(m0, m1),
        var=(float(p @ q0) - m0 * m0, float(p @ q1) - m1 * m1),
        cov=float(p @ (e0 * e1)) - m0 * m1,
        splits=split,
        worlds=len(tree.roots),
    )
