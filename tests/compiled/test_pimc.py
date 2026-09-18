"""Exact PIMC-infinity (`uno.compiled.pimc`, step 9.5d): the policy it builds is a real L1
policy (never worth more than BR L1), the perfect-information values it acts on are the ones
`best_response_perfect` / `two_sided` produce, the prior belief never leaves a set without
mass, every policy is orbit-uniform, and on a small ruleset the quotient's value is the value
of that same policy lifted onto the full tree -- which is what makes the quotient number the
full game's (docs/foundations/spec.md section 18).
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest

from tests.strategies import SMALL
from uno import store
from uno.agents.baseline import Heuristic, PlayFirstRandom
from uno.agents.tabular import CompiledAgent
from uno.compiled.pimc import CONTINUATIONS, WEIGHTINGS, pimc_n, pimc_response, pimc_row
from uno.compiled.solvers import best_response, evaluate, two_sided
from uno.compiled.strategies import tabulate, transfer
from uno.compiled.tree import compile_tree
from uno.game import rules
from uno.game.observe import Level

L1, L3 = Level.L1, Level.L3
VARIANTS = [(w, c) for w in WEIGHTINGS for c in CONTINUATIONS]


def _profile(seat: int, mine, opp):
    return (mine, opp) if seat == 0 else (opp, mine)


@cache
def _toy():
    """(world tree, {opponent name: strategy over the world tree's L1 keying})."""
    rs = rules.TOY
    world = compile_tree(rs, (L1, L3), True, cache=store.CACHE)
    lazy = compile_tree(rs, (L1,), True, cache=store.CACHE)
    opps = {a.name: (L1, transfer(lazy, world, L1, store.load_or_tabulate(lazy, L1, a))) for a in (PlayFirstRandom(rs), Heuristic(rs))}
    return world, opps


@pytest.mark.parametrize("weighting,continuation", VARIANTS)
def test_pimc_is_an_l1_policy_and_never_beats_the_best_response(weighting, continuation):
    """PIMC-infinity acts on perfect-information values, but what it *plays* is a function of
    the L1 set alone, so BR L1 -- the best L1 policy against that opponent -- must dominate it.
    A PIMC value above BR L1 would mean the thing is not a policy at all (the step 8.6 lesson,
    "a best response above the maximum is not a strategy")."""
    world, opps = _toy()
    for opp in opps.values():
        for seat in (0, 1):
            p = pimc_response(world, opp, seat, weighting, continuation)
            br = best_response(world, L1, opp, seat)
            assert p.value <= br.value + 1e-9, f"PIMC {p.value} above BR L1 {br.value}"


@pytest.mark.parametrize("weighting,continuation", VARIANTS)
def test_believed_value_is_the_solver_it_claims_to_use(weighting, continuation):
    """`believed` is the root of the same perfect-information solve the per-arc values come
    from: BR L3's value under "response", the two-sided minimax value under "minimax"."""
    world, opps = _toy()
    ts = two_sided(world).value
    for opp in opps.values():
        for seat in (0, 1):
            p = pimc_response(world, opp, seat, weighting, continuation)
            expected = best_response(world, L3, opp, seat).value if continuation == "response" else (ts if seat == 0 else -ts)
            assert p.believed == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("continuation", CONTINUATIONS)
def test_prior_belief_leaves_no_set_without_mass(continuation):
    """Every information set of the tree holds at least one world, so weighting by the worlds'
    prior alone always gives positive mass: unlike a best response (and unlike the posterior
    belief), prior PIMC has no arbitrary fill-in anywhere."""
    world, opps = _toy()
    for opp in opps.values():
        for seat in (0, 1):
            assert pimc_response(world, opp, seat, "prior", continuation).unreached == 0
            assert pimc_response(world, opp, seat, "posterior", continuation).unreached > 0  # the opponent does block sets


@pytest.mark.parametrize("weighting,continuation", VARIANTS)
def test_policies_are_uniform_within_action_orbits(weighting, continuation):
    """A symmetric strategy is uniform within every orbit of its set's stabilizer; a pure choice
    there would condition on hidden information and break the quotient (docs/foundations/spec.md section 18)."""
    world, opps = _toy()
    k = world.keyings[L1]
    for opp in opps.values():
        for seat in (0, 1):
            assert k.is_symmetric(pimc_response(world, opp, seat, weighting, continuation).sigma, 1e-12)
        assert k.is_symmetric(pimc_row(world, opp, weighting, continuation)[0][1], 1e-12)


def test_a_worlds_arcs_are_not_listed_in_its_information_sets_order():
    """Why the next test matters. The members of an L1 set are different worlds, each
    canonicalised by its own group element on a quotient tree, so arc `j` of a member's node is
    *not* in general set-action `j`. Anything that averages per-action values across the members
    of a set must map arcs through `k.arc_ia`, never through their position under their node
    (the trap `uno.compiled.lookup` documents for `tree.arc_action`).

    This is pinned because taking the positional shortcut in `pimc_n` cost 21 score points at
    `TOY` seat 1 and, being a systematic misassignment, got *worse* as more worlds were sampled.
    """
    world, _ = _toy()
    k = world.keyings[L1]
    for seat, floor in ((0, 0.05), (1, 0.15)):
        mine = np.nonzero(world.arc_player == seat)[0]
        node = world.arc_node[mine]
        positional = k.info_start[k.node_info[node]] + (mine - world.node_arc_start[node])
        assert (positional != k.arc_ia[mine]).mean() > floor, "TOY no longer exercises the arc-order trap"


@pytest.mark.parametrize("seat", (0, 1))
def test_pimc_n_closes_the_gap_to_pimc_infinity_as_n_grows(seat):
    """Sampling n worlds instead of the whole belief must approach `pimc_response` as n grows,
    since PIMC-infinity is the n -> infinity limit of the same argmax. A systematic misassignment
    of values to actions shows up here as a gap that *grows* with n instead of shrinking.

    The gap is asserted to shrink and to end near zero, not to stay positive: PIMC-infinity
    maximises the belief-weighted perfect-information score rather than the true value, so a
    randomised `pi_n` may end up slightly the better policy (docs/foundations/spec.md section 25)."""
    world, opps = _toy()
    limit = pimc_response(world, opps["random"], seat, "posterior", "response")
    gaps = [limit.value - pimc_n(world, opps["random"], seat, n, realizations=32, weighting="posterior", continuation="response", seed=1).value for n in (1, 64)]
    assert gaps[1] < gaps[0], f"more sampled worlds did not help: gaps {gaps}"
    assert abs(gaps[1]) < 0.02, f"n=64 is {50 * gaps[1]:.2f} score points from the limit"


@cache
def _small():
    """SMALL's world trees, quotient and full, with the opponent tabulated on each."""
    out = {}
    for symmetric in (True, False):
        tree = compile_tree(SMALL, (L1, L3), symmetric)
        out[symmetric] = (tree, (L1, tabulate(tree, L1, PlayFirstRandom(SMALL))))
    return out


@pytest.mark.parametrize("weighting,continuation", VARIANTS)
def test_quotient_value_equals_the_same_policy_lifted_onto_the_full_tree(weighting, continuation):
    """The quotient's number is only the full game's if the policy really is symmetric. Lift it
    with `CompiledAgent` (canonical key in, action mapped back) and re-evaluate on the full,
    un-quotiented tree: the two must agree exactly.

    This is the check that separates a bug from a property, because the quotient and the full
    tree do *not* in general produce the same PIMC policy: unlike a best response, where tied
    actions have equal continuation value, a PIMC tie is a tie in the belief-weighted
    perfect-information score, and two tied actions can be worth different amounts in the real
    game. The quotient plays the whole tied orbit uniformly, a full tree picks one pure
    representative per set; both are genuine PIMC policies, and only the first is symmetric.
    """
    small = _small()
    qw, q_opp = small[True]
    fw, f_opp = small[False]
    for seat in (0, 1):
        p = pimc_response(qw, q_opp, seat, weighting, continuation)
        lifted = tabulate(fw, L1, CompiledAgent(qw, L1, p.sigma, name="pimc_quotient"))
        v = evaluate(fw, *_profile(seat, (L1, lifted), f_opp))[seat]
        assert v == pytest.approx(p.value, abs=1e-12)
