"""Compiled game tree: the lazy-chance tree (or the world tree) as flat numpy
arrays, and the exact computations that become vectorized passes over them.

See docs/spec.md section 16. Why: the pure-Python solvers of `uno.reference.solve` walk
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
`uno.game.keys.KEYS` (L1, L2, L3). Information sets are numbered in order of first
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
exploitability, exact mirrored moments, CFR in `uno.compiled.cfr`) is a few lines over
those two. The reference implementations in `uno.reference` are
kept as the independent cross-check (`tests/compiled/test_compiled.py`).
"""

from __future__ import annotations

import time
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Hashable

import numpy as np

from uno.game.chance import children, engine_children, initial_deals, worlds
from uno.game.engine import Phase, State, returns
from uno.game.keys import KEYS, decode_key, encode_key
from uno.game.observe import Level
from uno.game.rules import Ruleset
from uno.game.symmetry import Symmetry

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
    # every orbit, see docs/spec.md section 18. Orbits are numbered in set order, so a set's orbits
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


def compile_tree(rs: Ruleset, levels: tuple[Level, ...] = (Level.L1,), symmetric: bool = True, cache: str | Path | None = None, world: bool | None = None) -> CompiledTree:
    """Walk the tree once and compile it, with one keying per requested level.
    Roots are the worlds when L3 is requested (its sets are the histories, which
    the lazy tree merges), the deals otherwise; `world=True` forces world roots
    without an L3 keying (for `best_response_perfect` on trees too big for one).
    `symmetric`: one root per color / rank orbit and canonical keys (docs/spec.md
    section 18); exact for symmetric policies only. `cache`: a directory; the
    tree is loaded from it if a file for this (ruleset, levels, symmetric,
    world) exists there, saved to it otherwise.

    Two compiles of the same (ruleset, symmetric, world) walk the same nodes in
    the same order whatever the keyings, so a keying from one can be attached
    to the other (`tree.keyings[lv] = other.keyings[lv]`); `tests/compiled/test_compiled.py`
    checks the node order."""
    if world is None:
        world = Level.L3 in levels
    if cache is not None:
        from uno.store import cache_name

        cache = Path(cache)
        path = cache / f"tree_{cache_name(rs, levels, symmetric, world)}.npz"
        if path.exists():
            return CompiledTree.load(path, rs)
        tree = compile_tree(rs, levels, symmetric, world=world)
        cache.mkdir(parents=True, exist_ok=True)
        tree.save(path)
        return tree
    t0 = time.perf_counter()
    sym = Symmetry(rs) if symmetric else None
    if world:
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


# -- strategies ---------------------------------------------------------------

Strategy = tuple[Level, np.ndarray]  # (keying level, sigma over that keying's ia)
