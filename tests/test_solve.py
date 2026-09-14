"""Exact solvers (uno/solve.py): brute-force cross-checks on tiny rulesets, consistency on small ones."""

from __future__ import annotations

import itertools
import math
from dataclasses import replace

from hypothesis import HealthCheck, assume, given, settings

from uno import TOY, Level, Ruleset, initial_state, observe
from uno.agents import Heuristic, PlayFirstRandom, UniformLegal
from uno.play import match
from uno.solve import KEYS, PureStrategy, TabularAgent, best_response, evaluate, evaluate_on_seeds, roots_and_expand, worlds
from uno.tree import census

from .test_tree import small_lazy_rulesets

LEVELS = (Level.L1, Level.L2, Level.L3)


def responder_infosets(rs, opp, seat, level) -> dict:
    """Every information set of the responder reachable under the opponent's
    support, with its legal actions. Independent of the solver's bookkeeping."""
    key = KEYS[level]
    roots, expand = roots_and_expand(rs, (level,))
    sets: dict = {}
    stack = [s for s, _ in roots]
    while stack:
        s = stack.pop()
        if s.terminal:
            continue
        ch = expand(s)
        if s.player == seat:
            sets.setdefault(key(s, seat), tuple(ch))
            for outs in ch.values():
                stack.extend(c for c, _ in outs)
        else:
            for a in opp.policy(observe(s, seat ^ 1, Level.L1), tuple(ch)):
                stack.extend(c for c, _ in ch[a])
    return sets


def pair(rs, responder, opp, seat, level):
    agents = (responder, opp) if seat == 0 else (opp, responder)
    levels = (level, Level.L1) if seat == 0 else (Level.L1, level)
    return agents, levels


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


# -- consistency on small rulesets ----------------------------------------------


SMALL = Ruleset(
    colors=2, rank_copies=(1, 1, 1), skip_per_color=0, reverse_per_color=0, draw_two_per_color=0,
    wild_count=1, wild_draw_four_count=0, hand_size=2, reshuffle="none", max_turns=None,
)  # fmt: skip  # candidate D: 7 cards, pile 2, 365k histories


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
    those very deals; the residual is action-sampling noise only."""
    rs = SMALL
    opp = PlayFirstRandom(rs)
    seeds = range(1500)
    for seat in (0, 1):
        br = best_response(rs, opp, seat, Level.L1)
        agents, levels = pair(rs, TabularAgent(rs, br), opp, seat, Level.L1)
        r = match(rs, (agents[seat], opp), seeds, swap=(seat == 1), levels=(Level.L1, Level.L1))
        exact = (evaluate_on_seeds(rs, agents, seeds, levels)[seat] + 1) / 2
        half = (r.score_ci[1] - r.score_ci[0]) / 2
        assert abs(r.score - exact) <= half, (seat, r.score, exact, br.score)
