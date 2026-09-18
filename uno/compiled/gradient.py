"""The exact policy gradient on a compiled tree, and the exact mean of the *sampled*
gradient of a luck-corrected return (step 10a.2, item A6). See docs/foundations/spec.md section 28.

Step 11's objective is a policy gradient, and step 10's question is which of AIVAT's
corrections may appear in the return it is taken of. The argument
(docs/foundations/results/10-readiness/10-brainstorm.md 5.2) is that the agent's own (row) correction must not,
because with an exact V the row term *is* `Q(s, a) - V(s)`, so subtracting it removes
exactly the part of the return that depends on the action. This module turns that
argument into two exact computations, so the claim is measured and not asserted.

**The parameterisation matters, and it is the one PPO uses.** A tabulated policy over a
keying is a point of the product of simplices; the gradient of the game value with
respect to the *probabilities* is

    dJ/dsigma(I, a) = sum over the histories h of I of rho(h) * Q(h, a)

with `rho` the full reach (chance x both players) and `Q` the arc value in the learner's
own return units. The score-function (REINFORCE) estimator of that quantity is
`1{a taken} / sigma(a) * G`, and it is the *logit* gradient

    dJ/dtheta(I, a) = sigma(a) * [ dJ/dsigma(I, a) - sum_a' sigma(a') dJ/dsigma(I, a') ]

whose estimator is `(1{a taken} - sigma(a)) * G`, that every statement below is about.
The difference is not cosmetic: the two differ by a per-set constant, and a corrected
return contributes to that constant everything the corrections did *before* the decision
(the deal term, the earlier chance and opponent terms). Those cancel in the centred form
and do not in the raw one, which is why the logit gradient is unbiased under corrections
that the raw one is biased under. `estimator_mean` computes both, so the cancellation is
visible rather than argued.

**Orbits are the parameters on a quotient tree.** A symmetric strategy is uniform within
every action orbit of a set's stabiliser (docs/foundations/spec.md section 18), so an equivariant
learner has one logit per orbit, not per action, and
`d J / d theta(orbit) = sum over the orbit's actions of dJ/dtheta(I, a)` -- the orbit
*sum* of the vector these functions return (`orbit_gradient`). The exact gradient is
already constant within an orbit, since actions of one orbit have isomorphic subtrees; a
sampled one is not, and reducing it the same way is what compares like with like.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uno.compiled.aivat import arc_expectation, full_reach
from uno.compiled.tree import CompiledTree, Keying, Strategy
from uno.eval.aivat import Corrections


def _profile(seat: int, learner: Strategy, opponent: Strategy) -> tuple[Strategy, Strategy]:
    return (learner, opponent) if seat == 0 else (opponent, learner)


def logit_form(k: Keying, sigma: np.ndarray, per_action: np.ndarray) -> np.ndarray:
    """`sigma(a) * [x(a) - sum_a' sigma(a') x(a')]` per set-action: the gradient with respect
    to a softmax logit, given the gradient `x` with respect to the probability.

    Everything constant within a set drops out here, which is the whole reason the
    score-function estimator is unbiased under corrections taken before the decision."""
    base = k.info_sum(sigma * per_action)[k.ia_info]
    return sigma * (per_action - base)


def orbit_gradient(k: Keying, g: np.ndarray) -> np.ndarray:
    """(n_orbits,) the per-orbit logit gradient: the orbit sum of a per-set-action one.

    For a symmetric learner an orbit is one parameter (docs/foundations/spec.md section 18), so this
    is the vector a learner on the quotient actually descends."""
    return np.bincount(k.ia_orbit, weights=g, minlength=k.n_orbits)


@dataclass(frozen=True)
class Gradient:
    """The exact policy gradient of the learner's expected return at `seat`."""

    seat: int
    value: float  # the learner's expected return under the profile, in [-1, 1]
    sigma: np.ndarray  # the learner's strategy over the keying (both seats' sets)
    prob: np.ndarray  # (n_ia,) dJ/dsigma, zero outside the learner's own sets
    logit: np.ndarray  # (n_ia,) dJ/dtheta for a softmax over the set's actions
    reach: np.ndarray  # (n_info,) the set's full reach: how often the profile visits it

    def orbits(self, k: Keying) -> np.ndarray:
        return orbit_gradient(k, self.logit)


def exact_gradient(tree: CompiledTree, learner: Strategy, opponent: Strategy, seat: int) -> Gradient:
    """The exact policy gradient, in the learner's own return units, from two passes.

    `dJ/dsigma(I, a)` is the reach-weighted arc value, which is CFR's instantaneous
    regret term with the learner's *own* reach left in (CFR drops it: a counterfactual
    regret asks what the set would be worth if the learner had steered to it, a gradient
    asks what its value is worth given that it does not). The logit form is the centred
    one, and sums to zero over every set by construction -- checked, since it is the
    identity that makes a policy gradient a direction within the simplex."""
    k = tree.keying_of(learner[0])
    sigma = learner[1]
    s0, s1 = _profile(seat, learner, opponent)
    w = tree.profile_weights(s0, s1)
    v, av = tree.values(w, tree.seat_util(seat))
    rho = full_reach(tree, s0, s1)
    mine = tree.arcs_of[seat]
    nodes = tree.arc_node[mine]
    prob = np.bincount(k.arc_ia[mine], weights=rho[nodes] * av[mine, 0], minlength=k.n_ia)
    dec = np.nonzero(tree.player == seat)[0]
    reach = np.bincount(k.node_info[dec], weights=rho[dec], minlength=k.n_info)
    return Gradient(
        seat=seat,
        value=float(v[0, 0]),
        sigma=sigma,
        prob=prob,
        logit=logit_form(k, sigma, prob),
        reach=reach,
    )


@dataclass(frozen=True)
class EstimatorMean:
    """The exact mean of the *sampled* gradient of a corrected return, per set-action.

    `prob` is the mean of the raw score-function estimator `1{a taken} / sigma(a) * G`,
    `logit` that of `(1{a taken} - sigma(a)) * G`. With no corrections both equal the
    corresponding field of `exact_gradient`. The gap between them is `prefix`, the
    corrections the game had already collected before the decision, which is constant
    within a set and therefore invisible to `logit`."""

    corrections: Corrections
    prob: np.ndarray  # (n_ia,)
    logit: np.ndarray  # (n_ia,)
    prefix: np.ndarray  # (n_ia,) the pre-decision corrections' contribution (per set-action, constant within a set)
    own_corrected: bool  # whether the learner's own decisions carry a correction

    def orbits(self, k: Keying) -> np.ndarray:
        return orbit_gradient(k, self.logit)


def estimator_mean(
    tree: CompiledTree,
    learner: Strategy,
    opponent: Strategy,
    seat: int,
    nu: np.ndarray,
    c: Corrections,
    deal_mean: float | None = None,
) -> EstimatorMean:
    """The exact mean of the sampled policy gradient when the return is corrected by `c`
    with value function `nu` (a node vector in seat-0 units, as `uno.compiled.aivat`).

    One backward pass is enough for the post-decision part, because every correction at or
    below the decision has conditional mean zero given the decision -- except the
    learner's own action correction, which is a function of the action taken and is
    exactly what the row correction removes. One forward pass carries the pre-decision
    part (`prefix`), which no algebra is needed to see cancel: it is computed and the
    centred form is compared against `exact_gradient`.

    `c` is read relative to the row agent as everywhere in `uno.eval.aivat`, so whether
    the learner's own decisions are corrected is `c.players(seat)[seat]` -- `c.row` when
    the learner is the row agent, which is the case the training target is about.
    `deal_mean`: the reference the deal correction subtracts, `arc_expectation(tree,
    nu)[0]` by default, which is what `uno.compiled.aivat.TreeValues` uses."""
    k = tree.keying_of(learner[0])
    sigma = learner[1]
    s0, s1 = _profile(seat, learner, opponent)
    w = tree.profile_weights(s0, s1)
    sign = 1.0 if seat == 0 else -1.0
    v, av = tree.values(w, tree.seat_util(seat))
    nu_s = sign * nu
    alpha = sign * arc_expectation(tree, nu)
    if deal_mean is not None:
        alpha = alpha.copy()
        alpha[0] = sign * deal_mean
    rho = full_reach(tree, s0, s1)

    # the action correction of every player arc, and which arcs carry one
    # sum_a sigma(a|h) alpha(h, a) at each decision node (`bincount`, not `np.add.at`: the
    # unbuffered scatter is an order of magnitude slower at tree sizes)
    base = np.bincount(tree.arc_node, weights=w * alpha, minlength=tree.n)
    action_term = alpha - base[tree.arc_node]
    players = c.players(seat)
    act_on = ((tree.arc_player == 0) & players[0]) | ((tree.arc_player == 1) & players[1])
    c_action = np.where(act_on, action_term, 0.0)
    chance_on = np.where(tree.arc_player >= 0, c.chance, c.deal)
    c_chance = np.where(chance_on[tree.edge_arc], nu_s[tree.edge_child] - alpha[tree.edge_arc], 0.0)

    # forward: S(h) = rho(h) * (the corrections collected on the path to h)
    S = np.zeros(tree.n)
    for L in tree.levels:
        if L.e1 == L.e0:
            continue
        e = slice(L.e0, L.e1)
        arcs = tree.edge_arc[e]
        nodes = tree.arc_node[arcs]
        S[tree.edge_child[e]] = w[arcs] * tree.edge_prob[e] * (S[nodes] + rho[nodes] * (c_action[arcs] + c_chance[e]))

    mine = tree.arcs_of[seat]
    nodes = tree.arc_node[mine]
    own_corrected = bool(players[seat])
    q = av[mine, 0] - (action_term[mine] if own_corrected else 0.0)
    prob = np.bincount(k.arc_ia[mine], weights=rho[nodes] * q, minlength=k.n_ia)
    prefix = np.bincount(k.arc_ia[mine], weights=S[nodes], minlength=k.n_ia)
    return EstimatorMean(
        corrections=c,
        prob=prob - prefix,
        logit=logit_form(k, sigma, prob - prefix),
        prefix=prefix,
        own_corrected=own_corrected,
    )


def finite_difference(tree: CompiledTree, learner: Strategy, opponent: Strategy, seat: int, orbits, h: float = 1e-5) -> np.ndarray:
    """The learner's expected return differentiated numerically with respect to the softmax
    logit of each orbit in `orbits`, by a central difference through `evaluate`.

    The independent check of `exact_gradient`: it shares no code with it beyond the tree's
    own backward pass, and it is the definition rather than a rearrangement of it. A
    symmetric strategy is uniform within an orbit, so perturbing one orbit's logit means
    re-normalising its whole set -- which is why the parameter here is the orbit and not
    the action (docs/foundations/spec.md section 18).

    **The softmax is over the set's support, not all of its actions.** An action at
    probability zero is a logit at minus infinity; it stays there under any perturbation of
    the others, and `logit_form` already gives it a gradient of zero. Requiring full
    support instead would make this check inapplicable to most real policies -- every
    rulebot here puts zero on something (`PlayFirstRandom` never draws with a play in hand),
    and on a free-play toy that left *no* set with both a real choice and full support, so
    the check could only be run on orbits whose gradient was identically zero. Only the
    perturbed orbit needs positive probability."""
    from uno.compiled.solvers import evaluate

    k = tree.keying_of(learner[0])
    sigma = learner[1]
    out = []
    for o in orbits:
        i = int(k.orbit_info[o])
        lo, hi = int(k.info_start[i]), int(k.info_start[i + 1])
        cell = sigma[lo:hi]
        on = cell > 0
        mine = (k.ia_orbit[lo:hi] == o) & on
        if not mine.any():
            raise ValueError("the perturbed orbit has probability zero: its logit is -inf and the derivative is not defined there")
        theta = np.zeros(hi - lo)
        theta[on] = np.log(cell[on])
        vals = []
        for step in (+h, -h):
            t = theta.copy()
            t[mine] += step
            p = np.where(on, np.exp(t - t[on].max()), 0.0)
            s = sigma.copy()
            s[lo:hi] = p / p.sum()
            prof = _profile(seat, (learner[0], s), opponent)
            vals.append(evaluate(tree, *prof)[seat])
        out.append((vals[0] - vals[1]) / (2 * h))
    return np.array(out)
