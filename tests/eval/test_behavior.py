"""Behavioral fingerprints use opportunity rates and common-state policy distance."""

from __future__ import annotations

import math

import numpy as np

from uno import FULL_UNO
from uno.agents.baseline import Heuristic, KeepsDrawn, PlayFirstRandom
from uno.agents.rulebot import DistilledRuleBot, UnpredictableRuleBot
from uno.eval.behavior import BehaviorStats, collect_probes, observe_trace, policy_distances
from uno.eval.play import play_game


def _profile(agent, games: int = 80) -> BehaviorStats:
    rs, rnd = FULL_UNO, PlayFirstRandom(FULL_UNO)
    out = BehaviorStats()
    for seed in range(games):
        observe_trace(rs, play_game(rs, seed, (agent, rnd)), 0, out)
        observe_trace(rs, play_game(rs, seed, (rnd, agent)), 1, out)
    return out


def test_profiler_separates_honest_and_assumption_breaking_policies():
    honest = _profile(Heuristic(FULL_UNO))
    noisy = _profile(UnpredictableRuleBot(FULL_UNO))
    passer = _profile(KeepsDrawn(FULL_UNO, 0.5))
    assert honest.denominators["voluntary_draw"] > 100 and honest.rate("voluntary_draw") == 0.0
    assert honest.denominators["keep_drawn"] > 20 and honest.rate("keep_drawn") == 0.0
    assert math.isclose(noisy.rate("voluntary_draw"), UnpredictableRuleBot(FULL_UNO).draw_rate, abs_tol=1e-12)
    assert math.isclose(passer.rate("keep_drawn"), 0.5, abs_tol=1e-12)
    assert noisy.rate("normalised_entropy") > honest.rate("normalised_entropy")


def test_draw_two_and_block_switch_are_distinct_supported_metrics():
    stats = _profile(DistilledRuleBot(FULL_UNO), games=120)
    assert stats.denominators["draw_two_use"] > 20
    assert stats.denominators["block_switch_use"] > 20
    assert np.isfinite(stats.rate("draw_two_use"))
    assert np.isfinite(stats.rate("block_switch_use"))


def test_common_state_policy_distance_is_symmetric_with_a_zero_diagonal():
    rs = FULL_UNO
    buckets = {}
    rnd, heu = PlayFirstRandom(rs), Heuristic(rs)
    for seed in range(20):
        collect_probes(rs, play_game(rs, seed, (heu, rnd)), buckets, cap=20)
    probes = [x for rows in buckets.values() for x in rows]
    names, distance = policy_distances(
        {"heuristic": heu, "distilled": DistilledRuleBot(rs), "unpredictable": UnpredictableRuleBot(rs)},
        probes,
    )
    assert names == ["heuristic", "distilled", "unpredictable"]
    assert np.allclose(distance, distance.T)
    assert np.allclose(np.diag(distance), 0.0)
    assert distance[0, 1] > 0 and distance[1, 2] > 0
