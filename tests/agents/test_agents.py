"""Baseline agents (uno/agents/baseline.py, docs/foundations/spec.md section 13)."""

from __future__ import annotations

import math
import random

from hypothesis import assume, given, settings

from tests.helpers import act, position
from tests.strategies import rulesets, seeds
from uno import Kind, Play, legal_actions, new_game, observe, step
from uno.agents.baseline import Heuristic, PlayFirstRandom, UniformLegal, sample


@given(rs=rulesets(), seed=seeds, policy_seed=seeds)
@settings(deadline=None)
def test_policies_are_distributions_over_legal(rs, seed, policy_seed):
    try:
        s = new_game(rs, seed)
    except ValueError:
        assume(False)
    agents = [UniformLegal(rs), PlayFirstRandom(rs), Heuristic(rs)]
    rng = random.Random(policy_seed)
    while not s.terminal:
        legal = legal_actions(s)
        obs = observe(s, s.player)
        plays = [a for a in legal if isinstance(rs.actions[a], Play)]
        for ag in agents:
            probs = ag.policy(obs, legal)
            assert set(probs) <= set(legal)
            assert all(p > 0 for p in probs.values())
            assert math.isclose(sum(probs.values()), 1.0)
            assert sample(probs, rng) in probs
        assert set(agents[0].policy(obs, legal)) == set(legal)
        pf = agents[1].policy(obs, legal)
        if plays:
            assert set(pf) == set(plays)
            assert rs.draw_action not in pf and rs.pass_action not in pf
        h = agents[2].policy(obs, legal)
        assert len(set(h.values())) == 1, "the Heuristic is uniform over its ties"
        if plays:
            assert set(h) <= set(plays)
        assert agents[2].policy(obs, legal) == h, "deterministic"
        s = step(s, rng.choice(legal))


def toy3():
    from dataclasses import replace

    from uno import TOY_CANDIDATE

    return replace(TOY_CANDIDATE, hand_size=3)


def test_heuristic_prefers_majority_color():
    rs = toy3()
    # top c1r2: c1r0 plays on color, c0r2 plays on rank; we hold two c0 cards
    s = position(rs, ["c0r0", "c0r2", "c1r0"], ["c1r1", "c1r1", "c0r1"], ["c1r2"])
    legal = legal_actions(s)
    assert set(legal) == {act(rs, "c0r2"), act(rs, "c1r0"), rs.draw_action}
    assert Heuristic(rs).policy(observe(s, 0), legal) == {act(rs, "c0r2"): 1.0}


def test_heuristic_holds_wild_and_picks_majority_color():
    rs = toy3()
    # top c1r2, hand W c0r0 c0r1: only the wild plays, colour 0 is held most
    s = position(rs, ["W", "c0r0", "c0r1"], ["c1r1", "c1r1", "c0r1"], ["c1r2"])
    assert Heuristic(rs).policy(observe(s, 0), legal_actions(s)) == {act(rs, "W", 0): 1.0}
    # with a colored play available the wild is held
    s = position(rs, ["W", "c0r0", "c1r1"], ["c1r0", "c1r0", "c0r1"], ["c1r2"])
    assert Heuristic(rs).policy(observe(s, 0), legal_actions(s)) == {act(rs, "c1r1"): 1.0}


def test_heuristic_draws_then_plays_drawn():
    rs = toy3()
    s = position(rs, ["c0r0", "c0r0", "c0r1"], ["c1r1", "c1r1", "c0r1"], ["c1r2", "c1r0"])
    assert Heuristic(rs).policy(observe(s, 0), legal_actions(s)) == {rs.draw_action: 1.0}
    s = step(s, rs.draw_action)
    assert s.phase.name == "AFTER_DRAW"
    assert Heuristic(rs).policy(observe(s, 0), legal_actions(s)) == {act(rs, "c1r0"): 1.0}
    assert PlayFirstRandom(rs).policy(observe(s, 0), legal_actions(s)) == {act(rs, "c1r0"): 1.0}


def test_heuristic_action_card_when_opponent_low():
    from dataclasses import replace

    from uno import TOY_CANDIDATE

    rs = replace(TOY_CANDIDATE, skip_per_color=1, hand_size=3)
    kinds = {rs.card_types[a.card].kind for a in rs.actions if isinstance(a, Play)}
    assert Kind.SKIP in kinds
    # opponent holds 3 (> threshold 2): the number card is preferred (same color count); with a 2-card opponent the skip
    s = position(rs, ["c1S", "c1r0", "c0r0"], ["c0r1", "c0r1", "c1r1"], ["c1r2"])
    assert Heuristic(rs).policy(observe(s, 0), legal_actions(s)) == {act(rs, "c1r0"): 1.0}
    rs2 = replace(rs, hand_size=2)
    s = position(rs2, ["c1S", "c1r0"], ["c0r1", "c0r1"], ["c1r2"])
    assert Heuristic(rs2).policy(observe(s, 0), legal_actions(s)) == {act(rs2, "c1S"): 1.0}
