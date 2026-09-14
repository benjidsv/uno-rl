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
from pathlib import Path
from typing import Hashable

import numpy as np

from uno.agents import Agent, Policy
from uno.engine import Phase, State, returns
from uno.observe import Level, Observation, observation_from_key
from uno.rules import Ruleset
from uno.solve import KEYS, decode_key, encode_key, engine_children, obs_key, worlds
from uno.symmetry import Symmetry
from uno.tree import children, initial_deals

Key = Hashable
CHANCE, TERMINAL = -2, -1
END_KINDS = (None, "win", "exhaustion", "cap")  # index = `CompiledTree.end` code


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
    keys: dict[bytes, int]  # encoded (canonical, on a symmetric tree) information-set key -> id
    # orbits of a set's actions under the stabilizer of its canonical key (singletons on a
    # non-symmetric tree). A strategy over ia is a symmetric strategy iff it is uniform within
    # every orbit, see RULES.md section 18. Orbits are numbered in set order, so a set's orbits
    # are a contiguous range.
    ia_orbit: np.ndarray  # (K,) int32
    orbit_info: np.ndarray  # (O,) int32
    orbit_size: np.ndarray  # (O,) int32
    info_orbit_start: np.ndarray  # (k + 1,) int32

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

    @property
    def n_orbits(self) -> int:
        return len(self.orbit_info)

    @property
    def has_orbits(self) -> bool:
        """Whether any orbit holds more than one action (only on a symmetric tree)."""
        return bool(np.any(self.orbit_size > 1))

    def orbit_mean(self, x: np.ndarray) -> np.ndarray:
        """An ia vector averaged within every orbit (the identity on singleton orbits)."""
        if not self.has_orbits:
            return x
        return (np.bincount(self.ia_orbit, weights=x, minlength=self.n_orbits) / self.orbit_size)[self.ia_orbit]

    def is_symmetric(self, sigma: np.ndarray, tol: float = 1e-12) -> bool:
        return bool(np.all(np.abs(sigma - self.orbit_mean(sigma)) <= tol))

    def first_orbit_uniform(self) -> np.ndarray:
        """Uniform over the first orbit of every set (the lowest action id and its images)."""
        first = self.ia_orbit[self.info_start[:-1]][self.ia_info]
        return np.where(self.ia_orbit == first, 1.0 / self.orbit_size[self.ia_orbit], 0.0)

    def ia_of(self, player: int) -> np.ndarray:
        return np.nonzero(self.info_player[self.ia_info] == player)[0]

    def actions(self, info: int) -> tuple[int, ...]:
        return tuple(int(a) for a in self.ia_action[self.info_start[info] : self.info_start[info + 1]])

    def key(self, info: int) -> Key:
        """The (canonical) key of a set, decoded. Builds the reverse index on first use."""
        rev = getattr(self, "_rev", None)
        if rev is None:
            rev = self._rev = {i: b for b, i in self.keys.items()}
        return decode_key(self.level, rev[info])


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
    end: np.ndarray  # (n,) int8, index into END_KINDS (0 = not terminal)
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
    symmetry: Symmetry | None = None  # set when the roots are orbit representatives and the keys canonical
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

    @property
    def symmetric(self) -> bool:
        return self.symmetry is not None

    def __str__(self) -> str:
        ks = ", ".join(f"{k.name}: {v.n_info} sets / {v.n_ia} ia" for k, v in self.keyings.items())
        sym = f" symmetric |G|={self.symmetry.order}" if self.symmetric else ""
        return (
            f"CompiledTree({self.roots_level.name} roots={len(self.roots)}{sym} nodes={self.n} decisions={self.n_decisions} "
            f"terminals={self.n_terminals} arcs={self.m} edges={self.e} levels={len(self.levels)} {ks}; {self.seconds:.1f}s)"
        )

    @property
    def nbytes(self) -> int:
        """Bytes held in numpy arrays plus the encoded keys."""
        n = sum(getattr(self, f).nbytes for f in ("player", "level", "util", "end", "arc_node", "arc_action", "arc_player", "node_arc_start", "edge_arc", "edge_child", "edge_prob", "arc_edge_start"))
        for k in self.keyings.values():
            n += sum(getattr(k, f).nbytes for f in ("node_info", "arc_ia", "ia_info", "ia_action", "info_player", "info_level", "info_start", "ia_orbit", "orbit_info", "orbit_size", "info_orbit_start"))
            n += sum(len(b) for b in k.keys)
        return n

    # -- persistence

    def save(self, path: str | Path) -> None:
        """Everything except the Symmetry (rebuilt on load). `np.savez` (uncompressed) for load speed."""
        arrays = {f: getattr(self, f) for f in ("player", "level", "util", "end", "arc_node", "arc_action", "arc_player", "node_arc_start", "edge_arc", "edge_child", "edge_prob", "arc_edge_start")}
        arrays["levels"] = np.array([(L.level, L.n0, L.nd, L.n1, L.a0, L.a1, L.e0, L.e1) for L in self.levels], dtype=np.int64)
        arrays["meta"] = np.array([repr(self.rs), self.roots_level.name, str(self.symmetric), repr(self.seconds)])
        arrays["keying_levels"] = np.array([lv.value for lv in self.keyings], dtype=np.int64)
        for lv, k in self.keyings.items():
            for f in ("node_info", "arc_ia", "ia_info", "ia_action", "info_player", "info_level", "info_start", "ia_orbit", "orbit_info", "orbit_size", "info_orbit_start"):
                arrays[f"{lv.name}_{f}"] = getattr(k, f)
            ids = np.fromiter(k.keys.values(), dtype=np.int32, count=len(k.keys))
            lens = np.fromiter((len(b) for b in k.keys), dtype=np.int32, count=len(k.keys))
            arrays[f"{lv.name}_key_ids"] = ids
            arrays[f"{lv.name}_key_lens"] = lens
            arrays[f"{lv.name}_key_bytes"] = np.frombuffer(b"".join(k.keys), dtype=np.uint8)
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str | Path, rs: Ruleset) -> CompiledTree:
        z = np.load(path)
        rs_repr, roots_level, symmetric, seconds = z["meta"].tolist()
        if rs_repr != repr(rs):
            raise ValueError(f"{path} was compiled for another ruleset: {rs_repr}")
        keyings = {}
        for v in z["keying_levels"].tolist():
            lv = Level(v)
            ids, lens, buf = z[f"{lv.name}_key_ids"], z[f"{lv.name}_key_lens"], z[f"{lv.name}_key_bytes"].tobytes()
            keys, pos = {}, 0
            for i, n in zip(ids.tolist(), lens.tolist()):
                keys[buf[pos : pos + n]] = i
                pos += n
            keyings[lv] = Keying(level=lv, keys=keys, **{f: z[f"{lv.name}_{f}"] for f in ("node_info", "arc_ia", "ia_info", "ia_action", "info_player", "info_level", "info_start", "ia_orbit", "orbit_info", "orbit_size", "info_orbit_start")})
        return cls(
            rs=rs,
            roots_level=Level[roots_level],
            **{f: z[f] for f in ("player", "level", "util", "end", "arc_node", "arc_action", "arc_player", "node_arc_start", "edge_arc", "edge_child", "edge_prob", "arc_edge_start")},
            levels=[LevelSlice(*(int(x) for x in row)) for row in z["levels"]],
            keyings=keyings,
            seconds=float(seconds),
            symmetry=Symmetry(rs) if symmetric == "True" else None,
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

    def end_columns(self) -> np.ndarray:
        """(n_terminals, len(END_KINDS) - 1) indicators of how the game ended: win, exhaustion, cap."""
        e = self.end[self.terminals]
        return np.stack([e == i for i in range(1, len(END_KINDS))], axis=1).astype(np.float64)

    def seat_util(self, seat: int) -> np.ndarray:
        return self.util[self.terminals] * (1.0 if seat == 0 else -1.0)


# -- build --------------------------------------------------------------------


def compile_tree(rs: Ruleset, levels: tuple[Level, ...] = (Level.L1,), symmetric: bool = True, cache: str | Path | None = None) -> CompiledTree:
    """Walk the tree once and compile it, with one keying per requested level.
    Roots are the worlds when L3 is requested (its sets are the histories, which
    the lazy tree merges), the deals otherwise. `symmetric`: one root per color /
    rank orbit and canonical keys (RULES.md section 18); exact for symmetric
    policies only. `cache`: a directory; the tree is loaded from it if a file for
    this (ruleset, levels, symmetric) exists there, saved to it otherwise."""
    if cache is not None:
        cache = Path(cache)
        path = cache / f"tree_{cache_name(rs, levels, symmetric)}.npz"
        if path.exists():
            return CompiledTree.load(path, rs)
        tree = compile_tree(rs, levels, symmetric)
        cache.mkdir(parents=True, exist_ok=True)
        tree.save(path)
        return tree
    t0 = time.perf_counter()
    sym = Symmetry(rs) if symmetric else None
    if Level.L3 in levels:
        roots, expand, roots_level = worlds(rs, symmetric), engine_children, Level.L3
    else:
        roots, expand, roots_level = initial_deals(rs, symmetric), children, Level.L1
    keyfns = {lv: KEYS[lv] for lv in levels}

    n_player, n_level, n_util, n_end = array("b"), array("i"), array("d"), array("b")
    a_node, a_action = array("i"), array("i")
    e_arc, e_child, e_prob = array("i"), array("i"), array("d")
    n_info = {lv: array("i") for lv in levels}
    a_cpos = {lv: array("b") for lv in levels}  # position of each arc's action in its set's (canonical) action list
    infos: dict[Level, dict[bytes, tuple[int, tuple[int, ...], int, tuple[int, ...]]]] = {lv: {} for lv in levels}  # key -> (id, actions, level, action orbits)
    info_player = {lv: array("b") for lv in levels}

    # the virtual root and its one arc
    n_player.append(CHANCE), n_level.append(0), n_util.append(0.0), n_end.append(0)
    for lv in levels:
        n_info[lv].append(-1)
    a_node.append(0), a_action.append(-1)
    for lv in levels:
        a_cpos[lv].append(0)
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
            n_end.append(END_KINDS.index(s.end))
            for lv in levels:
                n_info[lv].append(-1)
            continue
        p = s.player
        lvl = node_level(s)
        ch = expand(s)
        acts = tuple(ch)
        n_player.append(p), n_level.append(lvl), n_util.append(0.0), n_end.append(0)
        for lv in levels:
            key = keyfns[lv](s, p)
            if sym is None:
                cacts = acts
            else:
                key, g = sym.canonical(lv, key)
                am = sym.actions[g]
                cacts = tuple(sorted(int(am[a]) for a in acts))  # legal actions of the canonical representative
            kb = encode_key(lv, key)
            rec = infos[lv].get(kb)
            if rec is None:
                orbits = tuple(range(len(cacts))) if sym is None else sym.action_orbits(sym.stabilizer(lv, key), cacts)
                rec = (len(infos[lv]), cacts, lvl, orbits)
                infos[lv][kb] = rec
                info_player[lv].append(p)
            elif rec[1] != cacts:
                raise RuntimeError(f"{lv.name} information set with differing legal actions: {key!r}")
            elif rec[2] != lvl:
                raise RuntimeError(f"{lv.name} information set spanning two levels: {key!r}")
            n_info[lv].append(rec[0])
            if sym is None:
                a_cpos[lv].extend(range(len(acts)))
            else:
                a_cpos[lv].extend(cacts.index(int(am[a])) for a in acts)
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
    end = np.frombuffer(n_end, dtype=np.int8).copy()
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

    player, level, util, end = player[node_order], level[node_order], util[node_order], end[node_order]
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
        n_orbits = np.zeros(k, dtype=np.int32)
        keys: dict[bytes, int] = {}
        ia_action_list: list[tuple[int, ...]] = [()] * k
        orbit_list: list[tuple[int, ...]] = [()] * k
        for key, (oid, acts, _, orbits) in infos[lv].items():
            i = int(new_info[oid])
            n_actions[i] = len(acts)
            n_orbits[i] = max(orbits) + 1
            ia_action_list[i] = acts
            orbit_list[i] = orbits
            keys[key] = i
        info_start = np.zeros(k + 1, dtype=np.int32)
        np.cumsum(n_actions, out=info_start[1:])
        info_orbit_start = np.zeros(k + 1, dtype=np.int32)
        np.cumsum(n_orbits, out=info_orbit_start[1:])
        ia_orbit = np.fromiter((o for orbits in orbit_list for o in orbits), dtype=np.int32, count=int(info_start[-1])) + np.repeat(info_orbit_start[:-1], n_actions)
        orbit_info = np.repeat(np.arange(k, dtype=np.int32), n_orbits)
        orbit_size = np.bincount(ia_orbit, minlength=int(info_orbit_start[-1])).astype(np.int32)
        pos = np.frombuffer(a_cpos[lv], dtype=np.int8)[arc_order].astype(np.int32)
        arc_ia = np.where(arc_player >= 0, info_start[np.maximum(node_info[arc_node], 0)] + pos, -1).astype(np.int32)
        ia_info = np.repeat(np.arange(k, dtype=np.int32), n_actions)
        ia_action = np.fromiter((a for acts in ia_action_list for a in acts), dtype=np.int32, count=len(ia_info))
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
            ia_orbit=ia_orbit,
            orbit_info=orbit_info,
            orbit_size=orbit_size,
            info_orbit_start=info_orbit_start,
        )

    return CompiledTree(
        rs=rs,
        roots_level=roots_level,
        player=player,
        level=level,
        util=util,
        end=end,
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
        symmetry=sym,
    )


def cache_name(rs: Ruleset, levels: tuple[Level, ...], symmetric: bool) -> str:
    """Short stable name for a compiled tree: named rulesets by name, others by a hash of their repr."""
    import hashlib

    from uno import rules

    name = next((n for n in ("TOY", "TOY_FULL", "TOY2", "TOY2_FULL", "TOY2_SKIP", "TOY_CANDIDATE", "FULL_UNO") if getattr(rules, n) == rs), None)
    if name is None:
        name = hashlib.blake2b(repr(rs).encode(), digest_size=6).hexdigest()
    return f"{name}_{''.join(lv.name for lv in levels)}_{'sym' if symmetric else 'full'}"


# -- strategies from agents ---------------------------------------------------

Strategy = tuple[Level, np.ndarray]  # (keying level, sigma over that keying's ia)


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
        raise ValueError(f"{agent.name} is not symmetric: it is not uniform within some orbit of a canonical set (RULES.md section 18)")
    return sigma


def tabulate_table(tree: CompiledTree, level: Level, table: dict[Key, int]) -> np.ndarray:
    """A pure strategy from a key -> action table (a `uno.solve.BestResponse.table`,
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


class CompiledAgent:
    """Plays a strategy over a keying through `uno.play`. Must observe at `level`.
    Sets missing from the keying (never reached in the compiled tree, e.g. under
    a different ruleset) fall back to the first legal action and are counted."""

    def __init__(self, tree: CompiledTree, level: Level, sigma: np.ndarray, name: str = "compiled") -> None:
        self.rs = tree.rs
        self.keying = tree.keyings[level]
        self.symmetry = tree.symmetry
        self.level = level
        self.sigma = sigma
        self.name = name
        self.misses = 0

    def policy(self, obs: Observation, legal: tuple[int, ...]) -> Policy:
        if obs.level is not self.level:
            raise ValueError(f"{self.name} needs {self.level.name} observations, got {obs.level.name}")
        k = self.keying
        key = obs_key(self.rs, obs)
        g = 0
        if self.symmetry is not None:
            key, g = self.symmetry.canonical(self.level, key)
        i = k.keys.get(encode_key(self.level, key))
        if i is None:
            self.misses += 1
            return {legal[0]: 1.0}
        s, t = k.info_start[i], k.info_start[i + 1]
        out = {int(a): float(p) for a, p in zip(k.ia_action[s:t], self.sigma[s:t]) if p > 0}
        if g:
            out = self.symmetry.map_policy(int(self.symmetry.inverse[g]), out)  # back from canonical to the real labels
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
    """A best response computed on the compiled tree. `sigma` is the strategy over
    the responder's keying (both seats' sets; the opponent's rows and the
    responder's unreached sets hold their first action): pure on a full tree,
    uniform within the chosen action orbit on a symmetric one."""

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
    histories' continuation values. Ties: the lowest action id, as `uno.solve`.
    On a symmetric tree the choice is an *orbit* of actions under the set's
    stabilizer, played uniformly (RULES.md section 18): the value equals the
    full game's best-response value against a symmetric opponent, and the
    strategy is symmetric."""
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
            o0, o1 = k.info_orbit_start[i0], k.info_orbit_start[i1]  # and their orbits
            q = np.bincount(ia[mine] - k0, weights=cf[tree.arc_node[L.a0 : L.a1]][mine] * av[L.a0 : L.a1][mine], minlength=k1 - k0)
            orb = k.ia_orbit[k0:k1] - o0
            size = k.orbit_size[o0:o1]
            qo = np.bincount(orb, weights=q, minlength=o1 - o0) / size  # value of playing an orbit uniformly
            local_info = k.orbit_info[o0:o1] - i0
            mx = np.maximum.reduceat(qo, k.info_orbit_start[i0:i1] - o0)
            is_max = qo == mx[local_info]
            first = np.full(i1 - i0, o1 - o0, dtype=np.int64)
            np.minimum.at(first, local_info[is_max], np.nonzero(is_max)[0])
            chosen = np.zeros(o1 - o0)
            chosen[first] = 1.0
            c = chosen[orb] / size[orb]
            choice[k0:k1] = c
            w[mine] = c[ia[mine] - k0]
        v[L.n0 : L.nd] = np.add.reduceat(w * av[L.a0 : L.a1], tree.node_arc_start[L.n0 : L.nd] - L.a0)
    # sets never touched above (no responder arc at their level, i.e. the opponent's) take their first orbit
    untouched = (k.info_sum(choice) == 0)[k.ia_info]
    choice[untouched] = k.first_orbit_uniform()[untouched]
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
