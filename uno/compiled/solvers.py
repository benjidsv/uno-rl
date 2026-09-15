"""Exact computations as passes over a compiled tree: evaluation, outcome
distributions, best responses (keyed, and perfect-information on a world tree),
exploitability and the exact mirrored moments. See RULES.md section 16.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from uno.compiled.tree import CompiledTree, Strategy
from uno.game.observe import Level

# -- exact computations --------------------------------------------------------


def outcome_distribution(tree: CompiledTree, seat0: Strategy, seat1: Strategy) -> dict[float, float]:
    """`uno.reference.solve.outcome_distribution` on the compiled tree: {+1, 0, -1} -> P."""
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
    histories' continuation values. Ties: the lowest action id, as `uno.reference.solve`.
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


def best_response_perfect(tree: CompiledTree, opponent: Strategy, seat: int) -> BestResponse:
    """Best response with perfect information on a *world* tree: every history is
    its own information set, so no keying is needed and the responder simply
    takes, at each of its nodes, the arc of maximal value (ties: lowest action
    id). Equals `best_response(tree, Level.L3, ...)`: on a symmetric tree the L3
    sets are singletons up to a stabilizer that fixes the whole state, and
    actions in one orbit of such a stabilizer have isomorphic subtrees, hence
    equal values, so a single best action and the uniform orbit score the same.
    The result's `sigma` is the whole profile per *arc*: 1 on the chosen
    responder arcs, 0 on its other arcs, the opponent's weights on the
    opponent's arcs and 1 on chance; `tree.values(sigma, util)` replays it."""
    if tree.roots_level is not Level.L3:
        raise ValueError("best_response_perfect needs a world tree (compile_tree(..., world=True))")
    t0 = time.perf_counter()
    opp = seat ^ 1
    w = tree.weights(opponent[0], opponent[1], opp)
    v = np.zeros(tree.n)
    av = np.zeros(tree.m)
    v[tree.terminals] = tree.seat_util(seat)
    for L in reversed(tree.levels):
        if L.a1 == L.a0:
            continue
        x = tree.edge_prob[L.e0 : L.e1] * v[tree.edge_child[L.e0 : L.e1]]
        av[L.a0 : L.a1] = np.add.reduceat(x, tree.arc_edge_start[L.a0 : L.a1] - L.e0)
        mine = tree.arc_player[L.a0 : L.a1] == seat
        if mine.any():
            nodes = tree.arc_node[L.a0 : L.a1]
            starts = tree.node_arc_start[L.n0 : L.nd] - L.a0
            a = av[L.a0 : L.a1].copy()
            a[~mine] = -np.inf
            mx = np.maximum.reduceat(a, starts)
            local = nodes - L.n0
            is_max = mine & (a == mx[local])
            first = np.full(L.nd - L.n0, L.a1 - L.a0, dtype=np.int64)
            np.minimum.at(first, local[is_max], np.nonzero(is_max)[0])
            chosen = np.zeros(L.a1 - L.a0)
            chosen[first[first < L.a1 - L.a0]] = 1.0
            w[L.a0 : L.a1][mine] = chosen[mine]
        v[L.n0 : L.nd] = np.add.reduceat(w[L.a0 : L.a1] * av[L.a0 : L.a1], tree.node_arc_start[L.n0 : L.nd] - L.a0)
    return BestResponse(seat=seat, level=Level.L3, value=float(v[0]), sigma=w, seconds=time.perf_counter() - t0)


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
    """`uno.reference.mirrored.exact_mirrored` on a compiled world tree: the moments of `a`'s
    score over worlds when it plays seat 0 against `b`, and seat 1 against `b`.
    The strategies cover both seats' sets, so swapping is a different profile,
    not a different vector. Needs L3 roots (one root per world)."""
    from uno.eval.estimate import ExactMirrored

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
