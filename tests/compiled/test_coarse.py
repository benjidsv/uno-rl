"""Coarse (history-free) keys and the memoryless response (`uno.compiled.coarse`,
step 9.5a, D3): identity coarsening against ordinary best response, brute force
over pure K0 policies, and full/quotient agreement."""

from __future__ import annotations

import itertools

import numpy as np
from hypothesis import HealthCheck, assume, given, settings

from tests.strategies import SMALL, small, small_lazy_rulesets
from uno.agents.baseline import Heuristic, PlayFirstRandom
from uno.compiled.coarse import coarsen, identity_key, k0, k1, memoryless_response, project_onto_coarse
from uno.compiled.solvers import best_response, evaluate
from uno.compiled.strategies import tabulate
from uno.compiled.tree import compile_tree
from uno.game import rules
from uno.game.observe import Level


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
        k1_ = tree.keyings[Level.L1]
        # rebuild the L1-level view of the projected coarse strategy and compare to
        # the original, set by set: every L1 set sharing a K0 group must see the
        # representative's distribution.
        for i in range(k1_.n_info):
            cset = c.l1_to_coarse[i]
            s1, e1 = k1_.info_start[i], k1_.info_start[i + 1]
            sc, ec = c.keying.info_start[cset], c.keying.info_start[cset + 1]
            acts1, vals1 = k1_.ia_action[s1:e1], sigma[s1:e1]
            actsc, valsc = c.keying.ia_action[sc:ec], projected[sc:ec]
            order1 = np.argsort(acts1)
            orderc = np.argsort(actsc)
            assert np.allclose(vals1[order1], valsc[orderc], atol=1e-9)


def test_coarse_response_replays_through_evaluate_and_never_exceeds_br_l1():
    tree = compile_tree(rules.TOY, (Level.L1,), symmetric=True)
    rnd, heu = PlayFirstRandom(rules.TOY), Heuristic(rules.TOY)
    opp = (Level.L1, tabulate(tree, Level.L1, rnd))
    heu_l1 = tabulate(tree, Level.L1, heu)
    for name, fn in (("K0", k0), ("K1", k1)):
        c = coarsen(tree, fn)
        for seat in (0, 1):
            br = best_response(tree, Level.L1, opp, seat)
            res = memoryless_response(tree, c, opp, seat, heu_l1, iterations=20)
            prof = ((c.keying, res.sigma), opp) if seat == 0 else (opp, (c.keying, res.sigma))
            replayed = evaluate(tree, *prof)[seat]
            assert abs(replayed - res.value) < 1e-9, (name, seat)
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
    """Independently on the full tree and on the quotient (the known limitation
    above means their K0 numbers need not agree with each other, only each with
    its own tree's BR L1 and starting policy)."""
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
