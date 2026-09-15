"""Exact solvers (uno/reference/solve.py): brute-force cross-checks on tiny rulesets, consistency on small ones."""

from __future__ import annotations

import itertools
import math
from dataclasses import replace

from hypothesis import HealthCheck, assume, given, settings

from tests.strategies import SMALL, pair, responder_infosets, small_lazy_rulesets
from uno import TOY, Level, initial_state
from uno.agents.baseline import Heuristic, PlayFirstRandom, UniformLegal
from uno.agents.tabular import PureStrategy, TabularAgent
from uno.eval.play import match
from uno.game.chance import worlds
from uno.game.keys import KEYS
from uno.reference.census import census
from uno.reference.solve import best_response, evaluate, evaluate_on_seeds, scores_on_seeds

LEVELS = (Level.L1, Level.L2, Level.L3)


# -- brute force: enumerate every pure strategy ---------------------------------


@given(rs=small_lazy_rulesets())
@settings(deadline=None, max_examples=30, suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow])
def test_best_response_equals_max_over_pure_strategies(rs):
    assume(rs.deck_size <= 7)
    assume(sum(rs.deck_counts[t] for t, ct in enumerate(rs.card_types) if ct.rank is not None) > 0)
    opp = PlayFirstRandom(rs)
    for seat in (0, 1):
        for level in LEVELS:
            sets = responder_infosets(rs, opp, seat, level)
            keys = list(sets)
            n = math.prod(len(sets[k]) for k in keys)
            assume(n <= 1500)
            br = best_response(rs, opp, seat, level)
            assert set(br.table) == set(keys)
            assert all(br.table[k] in sets[k] for k in keys)
            best = -math.inf
            for choice in itertools.product(*(sets[k] for k in keys)):
                agents, levels = pair(rs, PureStrategy(rs, level, dict(zip(keys, choice))), opp, seat, level)
                best = max(best, evaluate(rs, agents, levels)[seat])
            assert math.isclose(best, br.value, abs_tol=1e-9), (level, seat, best, br.value)


@given(rs=small_lazy_rulesets())
@settings(deadline=None, max_examples=30, suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow])
def test_worlds_match_permutations(rs):
    """L3 roots: the set of (state without seed) over every deck permutation, with
    probability = share of permutations, conditioned on valid deals."""
    assume(rs.deck_size <= 7)
    assume(sum(rs.deck_counts[t] for t, ct in enumerate(rs.card_types) if ct.rank is not None) > 0)
    key = KEYS[Level.L3]
    counts: dict = {}
    total = 0
    for perm in set(itertools.permutations(rs.deck)):
        try:
            s = initial_state(rs, perm)
        except ValueError:
            continue
        total += 1
        counts[key(s, 0)] = counts.get(key(s, 0), 0) + 1
    # distinct permutations of a multiset are equiprobable under a uniform shuffle
    ws = worlds(rs)
    assert math.isclose(sum(p for _, p in ws), 1.0)
    got = {key(s, 0): p for s, p in ws}
    assert set(got) == set(counts)
    for k, c in counts.items():
        assert math.isclose(got[k], c / total, abs_tol=1e-9), (k, got[k], c / total)


def test_table_replays_to_value_and_dominates_baselines():
    rs = SMALL
    for opp in (PlayFirstRandom(rs), Heuristic(rs)):
        for seat in (0, 1):
            prev = -math.inf
            for level in LEVELS:
                br = best_response(rs, opp, seat, level)
                agent = TabularAgent(rs, br)
                agents, levels = pair(rs, agent, opp, seat, level)
                assert math.isclose(evaluate(rs, agents, levels)[seat], br.value, abs_tol=1e-9)
                assert agent.misses == 0
                assert br.value >= prev - 1e-12, "more information must not hurt"
                prev = br.value
                for base in (PlayFirstRandom(rs), Heuristic(rs), UniformLegal(rs)):
                    agents, levels = pair(rs, base, opp, seat, Level.L1)
                    assert evaluate(rs, agents, levels)[seat] <= br.value + 1e-12


def test_pile_zero_levels_agree():
    """With no draw pile the opponent's hand is deducible from public information,
    so the three levels coincide exactly."""
    rs = replace(TOY, hand_size=4)  # 9 cards, hand 4, top card: pile 0 (or the skipped wild, public)
    opp = PlayFirstRandom(rs)
    for seat in (0, 1):
        vals = [best_response(rs, opp, seat, level).value for level in LEVELS]
        assert all(math.isclose(v, vals[0], abs_tol=1e-9) for v in vals), vals


def test_best_response_to_uniform_beats_census_value():
    rs = SMALL
    c = census(rs)
    br = best_response(rs, UniformLegal(rs), 0, Level.L1)
    assert br.score >= c.uniform_legal_p0_score - 1e-12


def test_simulated_score_matches_exact_on_the_same_deals():
    """A fixed seed range carries its own deal luck (seeds 0-1500 on SMALL favour
    seat 0 by 1.6 points), so compare the simulation with the exact value on
    those very deals; the residual is action-sampling noise only, scored against
    its own exact variance."""
    rs = SMALL
    opp = PlayFirstRandom(rs)
    seeds = range(1500)
    for seat in (0, 1):
        br = best_response(rs, opp, seat, Level.L1)
        agents, levels = pair(rs, TabularAgent(rs, br), opp, seat, Level.L1)
        r = match(rs, (agents[seat], opp), seeds, swap=(seat == 1), levels=(Level.L1, Level.L1))
        on = scores_on_seeds(rs, agents, seeds, levels)
        assert on.ok(r.score, seat), (seat, r.score, on.mean[seat], on.z(r.score, seat))


def test_on_seeds_moments():
    """`scores_on_seeds`: the means are the on-seeds returns, the conditional variance
    is zero for two deterministic agents (their simulation then matches to the
    digit) and positive but below the total variance of the simulated scores otherwise."""
    rs = SMALL
    seeds = range(400)
    first = PureStrategy(rs, Level.L1, {})  # the first legal action everywhere
    on = scores_on_seeds(rs, (first, first), seeds)
    assert on.cond_var == (0.0, 0.0)
    r = match(rs, (first, first), seeds)
    assert math.isclose(r.score, on.mean[0], abs_tol=1e-12) and on.z(r.score, 0) == 0.0
    assert on.z(r.score + 0.01, 0) == math.inf
    rnd = PlayFirstRandom(rs)
    on = scores_on_seeds(rs, (rnd, rnd), seeds)
    v = evaluate_on_seeds(rs, (rnd, rnd), seeds)
    assert math.isclose(2 * on.mean[0] - 1, v[0], abs_tol=1e-12) and math.isclose(on.mean[0] + on.mean[1], 1.0, abs_tol=1e-12)
    r = match(rs, (rnd, rnd), seeds)
    total_var = r.n * ((r.score_ci[1] - r.score_ci[0]) / (2 * 1.96)) ** 2
    assert 0 < on.cond_var[0] < total_var
    assert on.ok(r.score, 0)
