"""AIVAT on a compiled tree: the exact moments of the estimator of `uno.eval.aivat`, the
value functions it is validated with, and `TreeValues`, the value model that reads them
for played games. See docs/spec.md section 23 (step 9.5b).

Values are seat-0 returns. A value function is a node vector `nu` (terminals included:
it must be a function of the terminal's return, which every family below satisfies);
the pre-chance value of an arc is `arc_expectation(tree, nu)`. On a symmetric (quotient)
tree `nu` must be the same on every node of an L2 set, so that it is a function of the
state and the quotient's moments are the full game's: `exact_values` of a symmetric
profile is, `noisy` adds its noise per L2 set, `l1_set_values` averages over canonical
L1 sets.

`aivat_moments` is one backward pass carrying, per node, the mean and second moment of
the estimator from that node on, `F(h) = u(z) - (corrections at h and below)`:

    along a chance edge   F = F(child) - c,  c = nu(child) - alpha(arc)     (the root arc: the deal)
    at a decision         F = F(arc)   - c,  c = alpha(arc) - sum_arcs w alpha

with the profile's weights w. Its mean is the profile's value for any `nu` and any subset
of corrections; its variance is what the sampled estimator's variance must match.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uno.compiled.lookup import NodeIndex
from uno.compiled.tree import CompiledTree, Strategy
from uno.eval.aivat import Corrections, StepRecord, rescored_return
from uno.eval.play import Trace
from uno.game.observe import Level


def exact_values(tree: CompiledTree, seat0: Strategy, seat1: Strategy) -> np.ndarray:
    """(n,) seat-0 value of every node under the profile."""
    v, _ = tree.values(tree.profile_weights(seat0, seat1), tree.seat_util(0))
    return v[:, 0]


def arc_expectation(tree: CompiledTree, nu: np.ndarray) -> np.ndarray:
    """(m,) the expectation of `nu` over each arc's chance outcomes (the root arc: over deals)."""
    return np.add.reduceat(tree.edge_prob * nu[tree.edge_child], tree.arc_edge_start[:-1])


@dataclass(frozen=True)
class Moments:
    mean: float  # seat-0 return
    var: float

    def score(self, seat: int) -> tuple[float, float]:
        """(mean, variance) of the score of the player at `seat`."""
        m = self.mean if seat == 0 else -self.mean
        return (m + 1.0) / 2.0, self.var / 4.0


def aivat_moments(tree: CompiledTree, seat0: Strategy, seat1: Strategy, nu: np.ndarray, c: Corrections, seat: int) -> Moments:
    """Exact mean and variance of the AIVAT estimate of one game under the profile, with the
    row agent at `seat` (which says whose decisions `c.row` / `c.column` correct)."""
    w = tree.profile_weights(seat0, seat1)
    alpha = arc_expectation(tree, nu)
    players = c.players(seat)
    act_on = ((tree.arc_player == 0) & players[0]) | ((tree.arc_player == 1) & players[1])
    chance_on = np.where(tree.arc_player >= 0, c.chance, c.deal)
    m1 = np.zeros(tree.n)
    m2 = np.zeros(tree.n)
    u = tree.seat_util(0)
    m1[tree.terminals] = u
    m2[tree.terminals] = u * u
    for L in reversed(tree.levels):
        if L.a1 == L.a0:
            continue
        child, arcs = tree.edge_child[L.e0 : L.e1], tree.edge_arc[L.e0 : L.e1]
        cc = np.where(chance_on[arcs], nu[child] - alpha[arcs], 0.0)
        p = tree.edge_prob[L.e0 : L.e1]
        e_starts = tree.arc_edge_start[L.a0 : L.a1] - L.e0
        a1 = np.add.reduceat(p * (m1[child] - cc), e_starts)
        a2 = np.add.reduceat(p * (m2[child] - 2.0 * cc * m1[child] + cc * cc), e_starts)
        ww, al = w[L.a0 : L.a1], alpha[L.a0 : L.a1]
        n_starts = tree.node_arc_start[L.n0 : L.nd] - L.a0
        base = np.add.reduceat(ww * al, n_starts)
        ca = np.where(act_on[L.a0 : L.a1], al - base[tree.arc_node[L.a0 : L.a1] - L.n0], 0.0)
        m1[L.n0 : L.nd] = np.add.reduceat(ww * (a1 - ca), n_starts)
        m2[L.n0 : L.nd] = np.add.reduceat(ww * (a2 - 2.0 * ca * a1 + ca * ca), n_starts)
    return Moments(float(m1[0]), max(float(m2[0] - m1[0] ** 2), 0.0))


# -- value functions for the imperfect-V curve -----------------------------------------


def noisy(tree: CompiledTree, nu: np.ndarray, sigma: float, seed: int = 0) -> np.ndarray:
    """`nu` plus independent N(0, sigma^2) noise per L2 set (one draw per set, so still a
    function of the state on the quotient); terminals and the root keep `nu`."""
    k = tree.keyings[Level.L2]
    noise = np.random.default_rng(seed).normal(0.0, sigma, k.n_info)
    return nu + np.where(k.node_info >= 0, noise[np.maximum(k.node_info, 0)], 0.0)


def full_reach(tree: CompiledTree, seat0: Strategy, seat1: Strategy) -> np.ndarray:
    """(n,) probability of reaching each node under the profile (chance x both players)."""
    r = tree.reach(tree.weights(seat0[0], seat0[1], 0), tree.weights(seat1[0], seat1[1], 1))
    return r[:, 0] * r[:, 1] * r[:, 2]


def l1_set_values(tree: CompiledTree, seat0: Strategy, seat1: Strategy, nu: np.ndarray) -> np.ndarray:
    """At every decision node, the reach-weighted mean of `nu` over the mover's L1 set (what
    the mover can know of the value), unweighted where the profile never reaches the set."""
    k = tree.keyings[Level.L1]
    rho = full_reach(tree, seat0, seat1)
    dec = np.nonzero(k.node_info >= 0)[0]
    sets = k.node_info[dec]
    mass = np.bincount(sets, weights=rho[dec], minlength=k.n_info)
    num = np.bincount(sets, weights=rho[dec] * nu[dec], minlength=k.n_info)
    cnt = np.bincount(sets, minlength=k.n_info)
    plain = np.bincount(sets, weights=nu[dec], minlength=k.n_info) / np.maximum(cnt, 1)
    per_set = np.where(mass > 0, num / np.where(mass > 0, mass, 1.0), plain)
    out = nu.copy()
    out[dec] = per_set[sets]
    return out


def outcome_r2(tree: CompiledTree, seat0: Strategy, seat1: Strategy, nu: np.ndarray) -> float:
    """`1 - E[(u - nu(h))^2] / E[(u - mean u)^2]` over the decisions a game visits (each visit
    counted once), u the game's final seat-0 return: the same quantity a sampled critic's R^2
    over visited states estimates, so exact and sampled value functions sit on one axis."""
    u = tree.seat_util(0)
    v, _ = tree.values(tree.profile_weights(seat0, seat1), np.stack((u, u * u), axis=1))
    rho = full_reach(tree, seat0, seat1)
    dec = tree.player >= 0
    r, e1, e2, x = rho[dec], v[dec, 0], v[dec, 1], nu[dec]
    ubar = float(v[0, 0])
    num = float(r @ (e2 - 2.0 * x * e1 + x * x))
    den = float(r @ (e2 - 2.0 * ubar * e1 + ubar * ubar))
    return 1.0 - num / den


def critic_values(tree: CompiledTree, critic, exhaustion: str = "hand") -> np.ndarray:
    """A fitted critic (`uno.eval.critic`) as a node vector: its value at every decision node,
    from the node's L2 key (a function of the state, so the same on every node of a set), the
    rescored return at the terminals."""
    from uno.eval.critic import observation_features
    from uno.game.keys import decode_key
    from uno.game.observe import observation_from_key

    k = tree.keyings[Level.L2]
    rows = [None] * k.n_info
    for b, i in k.keys.items():
        rows[i] = observation_features(tree.rs, observation_from_key(tree.rs, Level.L2, decode_key(Level.L2, b)))
    per_set = critic.values(rows)
    rt = tree.rescored(exhaustion)
    nu = np.zeros(tree.n)
    dec = k.node_info >= 0
    nu[dec] = per_set[k.node_info[dec]]
    nu[rt.terminals] = rt.util[rt.terminals]
    return nu


# -- the value model for played games ---------------------------------------------------


class TreeValues:
    """`uno.eval.aivat.ValueModel` reading a tree value function `nu` for played games through
    `uno.compiled.lookup.NodeIndex` (an L2 keying on the tree). `exhaustion`: the scoring the
    values use; `nu` must come from the tree rescored the same way (`CompiledTree.rescored`).
    Works for ticket-draw games too: an L2 key never holds the pile order."""

    def __init__(self, idx: NodeIndex, nu: np.ndarray, exhaustion: str = "hand") -> None:
        tree = idx.tree.rescored(exhaustion)
        self.idx, self.nu, self.exhaustion = idx, nu, exhaustion
        self.alpha = arc_expectation(tree, nu)
        self.deal_mean: float | None = float(self.alpha[0])
        u, t = tree.util[tree.terminals], nu[tree.terminals]
        self.terminal: dict[float, float] = {}
        for val in np.unique(u):
            vals = t[u == val]
            if np.ptp(vals) > 1e-12:
                raise ValueError("nu at the terminals must be a function of their return")
            self.terminal[float(val)] = float(vals[0])

    def records(self, trace: Trace) -> list[StepRecord]:
        steps = trace.steps
        out = []
        for i, st in enumerate(steps):
            node, arcs = self.idx.arcs(st.state, tuple(st.probs))
            nxt = steps[i + 1].state if i + 1 < len(steps) else trace.final
            child = self.terminal[rescored_return(trace, self.exhaustion)] if nxt.terminal else float(self.nu[self.idx.node(nxt)])
            out.append(StepRecord(st.state.player, st.probs, st.action, float(self.nu[node]), {a: float(self.alpha[arc]) for a, arc in arcs.items()}, child))
        return out
