"""Coarse (history-free) keys and the memoryless response (`uno.compiled.coarse`,
step 9.5a, D3): identity coarsening against ordinary best response, brute force
over pure K0 policies, and full/quotient agreement."""

from __future__ import annotations

import itertools

import numpy as np
from hypothesis import HealthCheck, assume, given, settings

from tests.strategies import SMALL, small, small_lazy_rulesets
from uno.agents.baseline import Heuristic, PlayFirstRandom
from uno.agents.tabular import CompiledAgent
from uno.compiled.coarse import (
    CoarseAgent,
    coarsen,
    identity_key,
    k0,
    k1,
    k2,
    lift_to_l1,
    memoryless_response,
    project_onto_coarse,
)
from uno.compiled.solvers import best_response, evaluate
from uno.compiled.strategies import tabulate
from uno.compiled.tree import compile_tree
from uno.game import rules
from uno.game.observe import Level, observation_from_key
from uno.game.symmetry import Symmetry


def test_identity_coarsening_matches_best_response_on_small():
    for symmetric in (False, True):
        tree = compile_tree(SMALL, (Level.L1,), symmetric=symmetric)
        opp = (Level.L1, tabulate(tree, Level.L1, PlayFirstRandom(SMALL)))
        c = coarsen(tree, identity_key)
        assert c.keying.n_info == tree.keyings[Level.L1].n_info
        for seat in (0, 1):
            br1 = best_response(tree, Level.L1, opp, seat)
            brc = best_response(tree, c.keying, opp, seat)
            assert abs(br1.value - brc.value) < 1e-9


def test_identity_coarsening_matches_the_pinned_toy_values():
    tree = compile_tree(rules.TOY, (Level.L1,), symmetric=True)
    opp = (Level.L1, tabulate(tree, Level.L1, PlayFirstRandom(rules.TOY)))
    c = coarsen(tree, identity_key)
    br0 = best_response(tree, c.keying, opp, 0)
    br1 = best_response(tree, c.keying, opp, 1)
    assert abs(br0.score - 0.7934) < 1e-4
    assert abs(br1.score - 0.5513) < 1e-4


def test_random_and_heuristic_are_constant_within_every_k0_set():
    tree = compile_tree(rules.TOY, (Level.L1,), symmetric=True)
    for agent in (PlayFirstRandom(rules.TOY), Heuristic(rules.TOY)):
        sigma = tabulate(tree, Level.L1, agent)
        c = coarsen(tree, k0)
        projected = project_onto_coarse(tree, c, sigma)
        # The coarse and L1 sets can use different canonical coordinate frames.
        # Lifting applies that map back before comparing the deployed policies.
        assert np.allclose(lift_to_l1(tree, c, projected), sigma, atol=1e-15)


def test_review_k1_collision_is_measurable_from_the_raw_input():
    """Regression for docs/results/10-review.md's positive-reach counterexample.

    Full-L1 canonicalisation puts these symmetry images in different coordinate
    frames even though their raw K1 fields are identical. A K1 policy must not use
    the discarded initial-hand/draw split to choose between actions 1 and 3.
    """
    tree = compile_tree(rules.TOY, (Level.L1,), symmetric=True)
    k = tree.keyings[Level.L1]
    rnd = tabulate(tree, Level.L1, PlayFirstRandom(rules.TOY))
    heu = tabulate(tree, Level.L1, Heuristic(rules.TOY))
    c = coarsen(tree, k1)
    res = memoryless_response(tree, c, (Level.L1, rnd), 1, heu, iterations=20)
    lifted = lift_to_l1(tree, c, res.sigma)

    obs = observation_from_key(rules.TOY, Level.L1, k.key(423))
    image = observation_from_key(
        rules.TOY, Level.L1, Symmetry(rules.TOY).key(Level.L1, 12, k.key(423))
    )
    assert k1(rules.TOY, obs) == k1(rules.TOY, image)
    assert obs.initial_hand == (0, 1, 2) and obs.own_draws == (3,)
    assert image.initial_hand == (0, 2, 3) and image.own_draws == (1,)

    # The collision remains on-policy at exactly the reach reproduced by the
    # review, so this is not an unreachable-coordinate formality.
    w0 = tree.weights(Level.L1, rnd, 0)
    w1 = tree.weights(Level.L1, lifted, 1)
    reach = tree.reach(w0, w1).prod(axis=1)
    nodes = np.nonzero(tree.player >= 0)[0]
    set_reach = np.bincount(k.node_info[nodes], weights=reach[nodes], minlength=k.n_info)
    assert abs(set_reach[423] - 0.0017460317460317458) < 1e-15

    expected = {1: 0.5, 3: 0.5}
    coarse_agent = CoarseAgent(tree, c, res.sigma, "k1_response")
    lifted_agent = CompiledAgent(tree, Level.L1, lifted, "lifted_k1_response")
    for agent in (coarse_agent, lifted_agent):
        assert agent.policy(obs, (1, 3)) == expected
        assert agent.policy(image, (1, 3)) == expected


def test_same_coarse_policy_has_the_same_partition_and_value_on_full_and_quotient_trees():
    """Independent SMALL oracle: compare one policy, not two local optima."""
    full = compile_tree(SMALL, (Level.L1,), symmetric=False)
    quotient = compile_tree(SMALL, (Level.L1,), symmetric=True)
    random = PlayFirstRandom(SMALL)
    opponents = {
        False: (Level.L1, tabulate(full, Level.L1, random)),
        True: (Level.L1, tabulate(quotient, Level.L1, random)),
    }

    for keyfn in (k0, k1, k2):
        cf = coarsen(full, keyfn)
        cq = coarsen(quotient, keyfn)
        # The quotient groups are exactly the symmetry orbits of the full tree's
        # raw coarse groups.
        assert {cq.canonical(key)[0] for key in cf.keys} == set(cq.keys)

        coarse_sigma = cq.keying.first_orbit_uniform()
        agent = CoarseAgent(quotient, cq, coarse_sigma, f"{keyfn.__name__}_first")
        q_l1 = tabulate(quotient, Level.L1, agent)
        f_l1 = tabulate(full, Level.L1, agent)
        assert np.array_equal(q_l1, lift_to_l1(quotient, cq, coarse_sigma))
        assert agent.misses == 0

        for seat in (0, 1):
            qp = ((Level.L1, q_l1), opponents[True]) if seat == 0 else (opponents[True], (Level.L1, q_l1))
            fp = ((Level.L1, f_l1), opponents[False]) if seat == 0 else (opponents[False], (Level.L1, f_l1))
            assert abs(evaluate(quotient, *qp)[seat] - evaluate(full, *fp)[seat]) < 1e-12


def test_coarse_response_replays_through_evaluate_and_never_exceeds_br_l1():
    tree = compile_tree(rules.TOY, (Level.L1,), symmetric=True)
    rnd, heu = PlayFirstRandom(rules.TOY), Heuristic(rules.TOY)
    opp = (Level.L1, tabulate(tree, Level.L1, rnd))
    heu_l1 = tabulate(tree, Level.L1, heu)
    expected = {
        ("K0", 0): 0.5348525132275134,
        ("K0", 1): 0.0375548941798942,
        ("K1", 0): 0.585388888888889,
        ("K1", 1): 0.09797222222222222,
    }
    for name, fn in (("K0", k0), ("K1", k1)):
        c = coarsen(tree, fn)
        for seat in (0, 1):
            br = best_response(tree, Level.L1, opp, seat)
            res = memoryless_response(tree, c, opp, seat, heu_l1, iterations=20)
            prof = ((c.keying, res.sigma), opp) if seat == 0 else (opp, (c.keying, res.sigma))
            replayed = evaluate(tree, *prof)[seat]
            assert abs(replayed - res.value) < 1e-9, (name, seat)
            assert abs(res.value - expected[(name, seat)]) < 1e-15, (name, seat)
            assert res.value <= br.value + 1e-9, (name, seat, res.value, br.value)
            # every iterate is non-decreasing on the value actually kept
            assert res.value >= res.log[0] - 1e-9


@given(rs=small_lazy_rulesets())
@settings(deadline=None, max_examples=25, suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow])
def test_k0_response_bounded_by_brute_force_over_pure_policies(rs):
    small(rs)
    tree = compile_tree(rs, (Level.L1,), symmetric=False)
    opp_agent = PlayFirstRandom(rs)
    opp = (Level.L1, tabulate(tree, Level.L1, opp_agent))
    c = coarsen(tree, k0)
    kc = c.keying
    for seat in (0, 1):
        seat_sets = np.nonzero(kc.info_player == seat)[0]
        if len(seat_sets) == 0:
            continue
        counts = [int(kc.info_start[s + 1] - kc.info_start[s]) for s in seat_sets]
        total = 1
        for cnt in counts:
            total *= cnt
            if total > 4096:
                break
        assume(total <= 4096)
        best = float("-inf")
        for choice in itertools.product(*(range(cnt) for cnt in counts)):
            sigma = np.zeros(kc.n_ia)
            for s, ch in zip(seat_sets, choice):
                sigma[kc.info_start[s] + ch] = 1.0
            prof = ((kc, sigma), opp) if seat == 0 else (opp, (kc, sigma))
            v = evaluate(tree, *prof)[seat]
            best = max(best, v)
        start = tabulate(tree, Level.L1, opp_agent)
        res = memoryless_response(tree, c, opp, seat, start, iterations=20)
        assert res.value <= best + 1e-9, (seat, res.value, best)


def test_own_weighted_best_response_replays_and_matches_own_none_value_on_each_tree():
    """`own` is a constant multiplier within any *perfect-recall* set (every history
    of an L1 set shares the same public history, hence the same own-reach), so it
    cannot change which action is optimal there and `best_response`'s *value* with
    `own` set must equal its value with `own=None`, on either tree, checked
    independently (full and quotient can pick a different action among a true tie,
    since `own` rescales the tie-break comparison's magnitudes -- not tested for
    cross-tree agreement here, only that each tree's own value is unaffected and
    the returned strategy still replays to it)."""
    rnd = PlayFirstRandom(SMALL)
    for symmetric in (False, True):
        tree = compile_tree(SMALL, (Level.L1,), symmetric=symmetric)
        opp = (Level.L1, tabulate(tree, Level.L1, rnd))
        c = coarsen(tree, identity_key)
        start = tabulate(tree, Level.L1, rnd)
        for seat in (0, 1):
            br_none = best_response(tree, c.keying, opp, seat)
            sigma0 = project_onto_coarse(tree, c, start)
            own = tree.weights(c.keying, sigma0, seat)
            br_own = best_response(tree, c.keying, opp, seat, own=own)
            assert abs(br_none.value - br_own.value) < 1e-9, (symmetric, seat)
            prof = ((c.keying, br_own.sigma), opp) if seat == 0 else (opp, (c.keying, br_own.sigma))
            assert abs(evaluate(tree, *prof)[seat] - br_own.value) < 1e-9, (symmetric, seat)


def test_k0_is_a_valid_lower_bound_on_both_full_and_quotient_trees():
    """The solver remains bounded independently on both representations."""
    rnd, heu = PlayFirstRandom(SMALL), Heuristic(SMALL)
    for symmetric in (False, True):
        tree = compile_tree(SMALL, (Level.L1,), symmetric=symmetric)
        opp = (Level.L1, tabulate(tree, Level.L1, rnd))
        heu_l1 = tabulate(tree, Level.L1, heu)
        c = coarsen(tree, k0)
        for seat in (0, 1):
            br = best_response(tree, Level.L1, opp, seat)
            heu_val = evaluate(tree, *(((Level.L1, heu_l1), opp) if seat == 0 else (opp, (Level.L1, heu_l1))))[seat]
            res = memoryless_response(tree, c, opp, seat, heu_l1, iterations=20)
            assert heu_val - 1e-9 <= res.value <= br.value + 1e-9, (symmetric, seat, heu_val, res.value, br.value)
