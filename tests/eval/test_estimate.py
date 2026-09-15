"""Estimators (uno/eval/estimate.py): replay, symmetry, exact moments, variance reduction."""

from __future__ import annotations

import math

from hypothesis import assume, given, settings

from tests.strategies import SMALL, rulesets, seeds
from uno import TOY, Level, legal_actions, new_game, step
from uno.agents.baseline import BySeat, Heuristic, PlayFirstRandom, UniformLegal
from uno.agents.tabular import TabularAgent
from uno.eval.estimate import Pairs, bootstrap_moments, compare, estimate_mirrored, games_needed, play_pairs
from uno.eval.play import DEV_SEEDS
from uno.reference.mirrored import exact_mirrored
from uno.reference.solve import best_response, evaluate, evaluate_on_seeds, outcome_distribution, scores_on_seeds


class Deterministic:
    """The lowest action in the wrapped agent's support: a deterministic (and not
    symmetric) agent for the exact-means checks, now that the Heuristic splits ties."""

    def __init__(self, agent) -> None:
        self.agent = agent
        self.name = agent.name + "_det"

    def policy(self, obs, legal):
        return {min(self.agent.policy(obs, legal)): 1.0}


def replays(rs, trace) -> None:
    s = trace.steps[0].state
    for st_ in trace.steps:
        assert st_.state == s and st_.action in legal_actions(s)
        s = step(s, st_.action)
    assert s == trace.final and s.terminal


# -- records ---------------------------------------------------------------------


@given(rs=rulesets(), seed=seeds)
@settings(deadline=None, max_examples=60)
def test_pairs_replay_and_symmetries(rs, seed):
    try:
        new_game(rs, seed)
    except ValueError:
        assume(False)
    a, b = UniformLegal(rs), PlayFirstRandom(rs)
    ab = play_pairs(rs, (a, b), [seed])
    ba = play_pairs(rs, (b, a), [seed])
    assert ab.n == ba.n == 1 and ab.seeds == (seed,)
    for g0, g1 in ab.games:
        replays(rs, g0.trace)
        replays(rs, g1.trace)
        assert g0.seat == 0 and g1.seat == 1
        assert g0.trace.steps[0].state == g1.trace.steps[0].state == new_game(rs, seed)
        assert g0.score in (0.0, 0.5, 1.0) and g1.score in (0.0, 0.5, 1.0)
    # naming the agents the other way round plays the same two games, scored from the other side
    assert ab.games[0][0].trace == ba.games[0][1].trace
    assert ab.games[0][1].trace == ba.games[0][0].trace
    assert ab.games[0][0].score == 1.0 - ba.games[0][1].score
    # self-play under common random numbers: the two games of a pair are the same game
    aa = play_pairs(rs, (a, a), [seed])
    g0, g1 = aa.games[0]
    assert g0.trace == g1.trace
    assert g0.score + g1.score == 1.0


def test_self_play_pairs_are_degenerate():
    rs = TOY
    rnd = PlayFirstRandom(rs)
    m = estimate_mirrored(play_pairs(rs, (rnd, rnd), range(300)))
    assert m.mirrored.mean == 0.5 and m.mirrored.var == 0.0 and m.splits == 1.0
    assert m.corr == -1.0
    assert math.isclose(m.seat[0].mean, 1.0 - m.seat[1].mean, abs_tol=1e-12)


def test_games_needed():
    assert math.isclose(games_needed(0.03, 0.25, z=3.0), 2500.0)


# -- exact side --------------------------------------------------------------------


def test_outcome_distribution_is_a_distribution_with_the_evaluate_mean():
    rs = SMALL
    for agents in ((Heuristic(rs), PlayFirstRandom(rs)), (UniformLegal(rs), Heuristic(rs))):
        d = outcome_distribution(rs, agents)
        assert set(d) <= {-1.0, 0.0, 1.0} and all(p > 0 for p in d.values())
        assert math.isclose(sum(d.values()), 1.0, abs_tol=1e-12)
        assert math.isclose(sum(r * p for r, p in d.items()), evaluate(rs, agents)[0], abs_tol=1e-12)


def test_exact_mirrored_means_equal_evaluate():
    """Worlds + bare engine (exact_mirrored) against the lazy chance tree (evaluate)."""
    rs = SMALL
    heu, rnd = Heuristic(rs), PlayFirstRandom(rs)
    ex = exact_mirrored(rs, (heu, rnd))
    assert math.isclose(ex.mean[0], (evaluate(rs, (heu, rnd))[0] + 1) / 2, abs_tol=1e-12)
    assert math.isclose(ex.mean[1], (evaluate(rs, (rnd, heu))[1] + 1) / 2, abs_tol=1e-12)
    assert 0 < ex.splits < 1 and ex.var_raw > 0
    assert math.isclose(ex.var_mirrored, (ex.var[0] + ex.var[1] + 2 * ex.cov) / 4)


# -- simulation against exact ----------------------------------------------------------


def check_against_exact(rs, agents, levels, pairs: Pairs, exact_means: bool) -> None:
    m = estimate_mirrored(pairs)
    ex = exact_mirrored(rs, agents, levels)
    # means: on the played seeds, against the exact action noise of those seeds (none for deterministic agents)
    ons = (scores_on_seeds(rs, agents, pairs.seeds, levels), scores_on_seeds(rs, (agents[1], agents[0]), pairs.seeds, (levels[1], levels[0])))
    for seat in (0, 1):
        got = m.seat[seat].mean
        assert ons[seat].ok(got, seat), (seat, got, ons[seat].mean[seat], ons[seat].z(got, seat))
        if exact_means:
            assert ons[seat].cond_var[seat] == 0.0 and math.isclose(got, ons[seat].mean[seat], abs_tol=1e-12)
    assert math.isclose(m.mirrored.mean, (m.seat[0].mean + m.seat[1].mean) / 2, abs_tol=1e-12)
    # second moments: every exact one inside the bootstrap CI of the sampled one (99.8 % each, family-wise 99 %)
    covered = bootstrap_moments(pairs, reps=2000).covers(ex)
    assert all(covered.values()), covered
    # the point of the exercise, exact and observed
    assert ex.cov < 0 and ex.var_ratio < 1
    assert m.var_ratio < 1


def test_mirrored_matches_exact_stochastic_row():
    rs = TOY
    heu, rnd = Heuristic(rs), PlayFirstRandom(rs)
    check_against_exact(rs, (heu, rnd), (Level.L1, Level.L1), play_pairs(rs, (heu, rnd), DEV_SEEDS), exact_means=False)


def test_mirrored_matches_exact_deterministic_row():
    rs = SMALL
    heu = Deterministic(Heuristic(rs))
    br = BySeat(*(TabularAgent(rs, best_response(rs, heu, seat, Level.L1)) for seat in (0, 1)))
    check_against_exact(rs, (br, heu), (Level.L1, Level.L1), play_pairs(rs, (br, heu), range(3000)), exact_means=True)


def test_compare_is_exact_for_deterministic_agents_and_pairs_help():
    rs = SMALL
    heu = Deterministic(Heuristic(rs))
    br = BySeat(*(TabularAgent(rs, best_response(rs, heu, seat, Level.L1)) for seat in (0, 1)))
    seeds_ = range(1500)
    a, b = play_pairs(rs, (br, heu), seeds_), play_pairs(rs, (heu, heu), seeds_)
    c = compare(a, b)
    for seat in (0, 1):
        rows = ((br, heu), (heu, heu)) if seat == 0 else ((heu, br), (heu, heu))
        want = (evaluate_on_seeds(rs, rows[0], seeds_)[seat] - evaluate_on_seeds(rs, rows[1], seeds_)[seat]) / 2
        assert math.isclose(c.seat[seat].mean, want, abs_tol=1e-12)
        assert c.seat[seat].var <= c.unpaired[seat].var
    # stochastic column: the Heuristic's edge over Random against Random, positively correlated across deals
    rs = TOY
    heu, rnd = Heuristic(rs), PlayFirstRandom(rs)
    c = compare(play_pairs(rs, (heu, rnd), range(1000)), play_pairs(rs, (rnd, rnd), range(1000)))
    for seat in (0, 1):
        assert c.corr[seat] > 0.3 and c.var_ratio(seat) < 0.7
        assert c.seat[seat].mean > 0


def test_bootstrap_moments():
    """The bootstrap CIs contain the sample moments, are seeded, and narrow with more pairs."""
    rs = TOY
    heu, rnd = Heuristic(rs), PlayFirstRandom(rs)
    pairs = play_pairs(rs, (heu, rnd), range(400))
    m = estimate_mirrored(pairs)
    b = bootstrap_moments(pairs, reps=200)
    for ci, x in ((b.corr, m.corr), (b.splits, m.splits), (b.var_ratio, m.var_ratio), (b.var[0], m.seat[0].var), (b.var[1], m.seat[1].var)):
        assert ci[0] <= x <= ci[1], (ci, x)
    assert b == bootstrap_moments(pairs, reps=200)
    wide = bootstrap_moments(play_pairs(rs, (heu, rnd), range(100)), reps=200)
    assert wide.corr[1] - wide.corr[0] > b.corr[1] - b.corr[0]
