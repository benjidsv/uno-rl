"""Runner and statistics (uno/eval/play.py)."""

from __future__ import annotations

import math

from hypothesis import assume, given, settings

from tests.strategies import rulesets, seeds
from uno import legal_actions, new_game, step
from uno.agents.baseline import Heuristic, PlayFirstRandom, UniformLegal
from uno.eval.play import match, play_game, stable_hash, summarize, wilson


@given(rs=rulesets(), seed=seeds)
@settings(deadline=None, max_examples=100)
def test_trace_is_deterministic_and_replays(rs, seed):
    try:
        new_game(rs, seed)
    except ValueError:
        assume(False)
    agents = (UniformLegal(rs), PlayFirstRandom(rs))
    t1 = play_game(rs, seed, agents)
    t2 = play_game(rs, seed, agents)
    assert t1 == t2
    s = new_game(rs, seed)
    for st_ in t1.steps:
        assert st_.state == s
        assert st_.action in legal_actions(s)
        assert st_.action in st_.probs
        assert st_.obs.seat == s.player
        s = step(s, st_.action)
    assert s == t1.final and s.terminal


STABLE_HASH_PIN = 0x6289483E54D14D59  # blake2b of repr((1, ('play', 0, 3, 1))); must never change


def test_stable_hash_is_stable():
    assert stable_hash((1, ("play", 0, 3, 1))) == STABLE_HASH_PIN
    assert stable_hash(("a", 1)) != stable_hash(("a", 2))


def test_wilson():
    lo, hi = wilson(0, 10)
    assert lo == 0.0 and 0 < hi < 0.35
    lo, hi = wilson(10, 10)
    assert 0.65 < lo < 1 and hi == 1.0
    lo, hi = wilson(500, 1000)
    assert lo < 0.5 < hi and hi - lo < 0.07


def test_summarize():
    r = summarize([1.0, 0.5, 0.0, 1.0], [10, 12, 8, 10])
    assert (r.wins, r.draws, r.losses) == (2, 1, 1)
    assert math.isclose(r.score, 0.625)
    assert r.score_ci[0] < r.score < r.score_ci[1]
    assert r.win_rate == 0.5 and r.mean_length == 10


def test_match_swap_is_symmetric():
    from uno import TOY_CANDIDATE as rs

    a, b = Heuristic(rs), PlayFirstRandom(rs)
    seeds_ = range(200)
    r_ab = match(rs, (a, b), seeds_)
    r_ba = match(rs, (b, a), seeds_, swap=True)
    assert r_ab.n == r_ba.n == 200
    assert (r_ab.wins, r_ab.draws, r_ab.losses) == (r_ba.losses, r_ba.draws, r_ba.wins)
