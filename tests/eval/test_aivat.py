"""AIVAT (uno/eval/aivat.py, uno/compiled/aivat.py, step 9.5b): the identities that pin the
estimator to the exact tree, on the 7-card SMALL ruleset, full and quotient trees."""

from __future__ import annotations

import math
from dataclasses import replace
from functools import cache

import numpy as np

from tests.strategies import SMALL
from uno.agents.baseline import Heuristic, PlayFirstRandom
from uno.compiled.aivat import TreeValues, aivat_moments, exact_values, l1_set_values, noisy, outcome_r2
from uno.compiled.edge import sampled_mirror
from uno.compiled.lookup import build
from uno.compiled.solvers import evaluate, outcome_distribution
from uno.compiled.strategies import tabulate
from uno.compiled.tree import compile_tree
from uno.eval.aivat import ALL, NONE, Corrections, estimate_aivat, mirror_regret, pair_terms, rescored_return, seat_scores
from uno.eval.estimate import play_pairs
from uno.eval.play import play_game
from uno.game.observe import Level

SUBSETS = (NONE, ALL, Corrections(True, False, False, False), Corrections(True, True, False, False), Corrections(False, True, True, False), Corrections(True, True, True, False))


@cache
def setup(symmetric: bool):
    tree = compile_tree(SMALL, (Level.L1, Level.L2), symmetric=symmetric)
    idx = build(tree, Level.L2)
    heu, rnd = Heuristic(SMALL), PlayFirstRandom(SMALL)
    sh, sr = (Level.L1, tabulate(tree, Level.L1, heu)), (Level.L1, tabulate(tree, Level.L1, rnd))
    return tree, idx, heu, rnd, sh, sr


def models(idx, nus, exhaustion="hand"):
    return tuple(TreeValues(idx, nu, exhaustion) for nu in nus)


def test_constant_value_reproduces_raw():
    tree, idx, heu, rnd, *_ = setup(True)
    pairs = play_pairs(SMALL, (heu, rnd), range(150))
    nu = np.full(tree.n, 0.3)
    terms = pair_terms(pairs, models(idx, (nu, nu)))
    for c in SUBSETS:
        for seat in (0, 1):
            assert np.array_equal(seat_scores(terms[seat], c, seat), np.array(pairs.seat_scores(seat)))


def test_exact_values_with_every_correction_give_the_root_value_on_every_game():
    for symmetric in (False, True):
        tree, idx, heu, rnd, sh, sr = setup(symmetric)
        nus = (exact_values(tree, sh, sr), exact_values(tree, sr, sh))
        for draws in ("pile", "tickets"):
            pairs = play_pairs(replace(SMALL, draws=draws), (heu, rnd), range(120))
            terms = pair_terms(pairs, models(idx, nus))
            for seat in (0, 1):
                got = np.array([t.estimate(ALL, seat) for t in terms[seat]])
                assert np.allclose(got, nus[seat][0], atol=1e-9), (symmetric, draws, seat)
                assert all(t.skipped_chance == t.skipped_action == 0 for t in terms[seat])


def test_moments_mean_is_the_value_and_raw_variance_is_the_outcome_variance():
    for symmetric in (False, True):
        tree, idx, heu, rnd, sh, sr = setup(symmetric)
        value = evaluate(tree, sh, sr)[0]
        d = outcome_distribution(tree, sh, sr)
        var_u = d[1.0] + d[-1.0] - (d[1.0] - d[-1.0]) ** 2
        exact = exact_values(tree, sh, sr)
        for nu in (exact, noisy(tree, exact, 0.3, seed=1), l1_set_values(tree, sh, sr, exact), np.zeros(tree.n)):
            for c in SUBSETS:
                m = aivat_moments(tree, sh, sr, nu, c, 0)
                assert math.isclose(m.mean, value, abs_tol=1e-12)
        assert math.isclose(aivat_moments(tree, sh, sr, exact, NONE, 0).var, var_u, abs_tol=1e-12)
        assert aivat_moments(tree, sh, sr, exact, ALL, 0).var < 1e-12
        assert 0.0 < aivat_moments(tree, sh, sr, exact, Corrections(True, True, False, True), 0).var  # Random's own noise stays in


def test_quotient_and_full_tree_agree_for_a_symmetric_value_function():
    out = []
    for symmetric in (False, True):
        tree, idx, heu, rnd, sh, sr = setup(symmetric)
        nu = l1_set_values(tree, sh, sr, exact_values(tree, sh, sr))
        out.append([aivat_moments(tree, sh, sr, nu, c, 0) for c in SUBSETS] + [outcome_r2(tree, sh, sr, nu)])
    for a, b in zip(*out):
        if isinstance(a, float):
            assert math.isclose(a, b, abs_tol=1e-9)
        else:
            assert math.isclose(a.mean, b.mean, abs_tol=1e-9) and math.isclose(a.var, b.var, abs_tol=1e-9)


def test_sampled_moments_match_exact_with_a_stochastic_uncorrected_column():
    tree, idx, heu, rnd, sh, sr = setup(True)
    nus = tuple(l1_set_values(tree, *prof, exact_values(tree, *prof)) for prof in ((sh, sr), (sr, sh)))
    pairs = play_pairs(SMALL, (heu, rnd), range(3000))
    c = Corrections(chance=True, row=True, column=False, deal=True)
    est = estimate_aivat(pair_terms(pairs, models(idx, nus)), c)
    for seat, prof in ((0, (sh, sr)), (1, (sr, sh))):
        mean, var = aivat_moments(tree, *prof, nus[seat], c, seat).score(seat)
        e = est.seat[seat]
        assert abs(e.mean - mean) / e.se <= 3.0, (seat, e.mean, mean)
        assert 0.85 <= e.var / var <= 1.15, (seat, e.var, var)
        raw_mean, raw_var = aivat_moments(tree, *prof, nus[seat], NONE, seat).score(seat)
        assert var < raw_var and abs(est.raw[seat].mean - raw_mean) / est.raw[seat].se <= 3.0


def test_rescored_return_matches_the_rescored_tree():
    tree, idx, heu, rnd, sh, sr = setup(False)
    checked = 0
    for exhaustion in ("hand", "draw", "stuck"):
        rt = tree.rescored(exhaustion)
        for seed in range(300):
            t = play_game(SMALL, seed, (heu, rnd))
            if t.final.end != "exhaustion":
                continue
            last = t.steps[-1]
            arc = idx.arc(last.state, last.action)
            e0, e1 = rt.arc_edge_start[arc], rt.arc_edge_start[arc + 1]
            if e1 - e0 != 1:
                continue
            assert rt.util[rt.edge_child[e0]] == rescored_return(t, exhaustion)
            checked += 1
    assert checked > 30


def test_mirror_regret_reproduces_sampled_mirror():
    tree, idx, heu, rnd, sh, sr = setup(True)
    ref = sr  # any strategy: the identity does not need a best response
    seeds = range(200)
    for seat in (0, 1):
        sm = sampled_mirror(tree, idx, heu, rnd, ref, seat, seeds)
        nu = exact_values(tree, *((ref, sr) if seat == 0 else (sr, ref)))
        model = TreeValues(idx, nu)
        sign = 1.0 if seat == 0 else -1.0
        agents = (heu, rnd) if seat == 0 else (rnd, heu)
        totals = [sign * mirror_regret(model.records(play_game(SMALL, s, agents)), seat) for s in seeds]
        assert math.isclose(float(np.mean(totals)), sm.total.mean, abs_tol=1e-12)
        assert math.isclose(float(np.var(totals, ddof=1)), sm.total.var, abs_tol=1e-12)
