"""Step 10C's cheap memory/tactical rulebots."""

from __future__ import annotations

import math
import random
from dataclasses import replace

from tests.agents.test_agents import toy3
from tests.helpers import act, position
from uno import FULL_UNO, TOY, legal_actions, new_game, observe, step
from uno.agents.baseline import Heuristic
from uno.agents.rulebot import FROZEN_RULEBOT_LEAGUE, FROZEN_RULEBOT_WEIGHTS, DistilledRuleBot, MemoryHeuristic, TransferRuleBot, UnpredictableRuleBot, rulebot_league
from uno.game.engine import Phase
from uno.game.symmetry import Symmetry


def test_memory_ablation_is_the_heuristic_without_evidence_and_uses_a_live_void():
    rs = toy3()
    s = position(rs, ["c0r1", "c1r0", "W"], ["c0r0", "c0r0", "c1r2"], ["c1r1"])
    obs, legal = observe(s, 0), legal_actions(s)
    assert MemoryHeuristic(rs).policy(obs, legal) == Heuristic(rs).policy(obs, legal) == {act(rs, "c1r0"): 1.0}

    # Earlier the opponent drew on c0r0, after which c1r1 became the board.  The two
    # candidate colours have equal support; memory now breaks that tie toward c0.
    c0r0 = next(i for i, x in enumerate(rs.card_types) if str(x) == "c0r0")
    c1r1 = next(i for i, x in enumerate(rs.card_types) if str(x) == "c1r1")
    with_void = replace(obs, public=(("setup", c0r0, ()), ("draw", 1, 1), ("play", 0, c1r1, 1)))
    assert MemoryHeuristic(rs).policy(with_void, legal) == {act(rs, "c0r1"): 1.0}


def test_unpredictable_bot_reports_its_draw_and_keep_probabilities_exactly():
    rs = FULL_UNO
    bot = UnpredictableRuleBot(rs)
    # Find a real opening where a play and voluntary draw are both legal.
    for seed in range(100):
        s = new_game(rs, seed)
        legal = legal_actions(s)
        if rs.draw_action in legal and len(legal) > 1:
            p = bot.policy(observe(s, s.player), legal)
            assert math.isclose(p[rs.draw_action], bot.draw_rate)
            assert math.isclose(sum(p.values()), 1.0)
            zero = UnpredictableRuleBot(rs, draw_rate=0.0).policy(observe(s, s.player), legal)
            assert rs.draw_action not in zero and all(x > 0 for x in zero.values())
            break
    else:
        raise AssertionError("no voluntary-draw opening found")

    # A playable drawn wild is always retained with the separately declared wild rate.
    rs = toy3()
    s = position(rs, ["c0r0", "c0r0", "c0r1"], ["c1r0", "c1r0", "c1r1"], ["c1r2", "W"])
    s = step(s, rs.draw_action)
    assert s.phase is Phase.AFTER_DRAW and rs.card_types[s.drawn].is_wild
    bot = UnpredictableRuleBot(rs)
    p = bot.policy(observe(s, 0), legal_actions(s))
    assert math.isclose(p[rs.pass_action], bot.keep_wild_rate)
    assert math.isclose(sum(p.values()), 1.0)
    zero = UnpredictableRuleBot(rs, keep_rate=0.0, keep_wild_rate=0.0).policy(observe(s, 0), legal_actions(s))
    assert rs.pass_action not in zero and all(x > 0 for x in zero.values())


def test_rulebot_names_identify_four_distinct_policies():
    league = rulebot_league(FULL_UNO)
    assert len(league) == 4
    assert len(set(league)) == 4
    assert tuple(league) == FROZEN_RULEBOT_LEAGUE
    assert len(FROZEN_RULEBOT_WEIGHTS) == len(league)
    assert math.isclose(sum(FROZEN_RULEBOT_WEIGHTS), 1.0)


def test_rulebots_are_symmetric_and_return_legal_distributions():
    rs = TOY
    sym = Symmetry(rs)
    agents = (MemoryHeuristic(rs), TransferRuleBot(rs), DistilledRuleBot(rs), UnpredictableRuleBot(rs))
    rng = random.Random(7)
    for seed in range(10):
        s = new_game(rs, seed)
        while not s.terminal:
            legal = legal_actions(s)
            g = rng.randrange(sym.order)
            gs = sym.state(g, s)
            glegal = legal_actions(gs)
            for agent in agents:
                got = agent.policy(observe(s, s.player), legal)
                transformed = sym.map_policy(g, got)
                want = agent.policy(observe(gs, gs.player), glegal)
                assert set(got) <= set(legal) and all(x > 0 for x in got.values()) and math.isclose(sum(got.values()), 1.0)
                assert set(want) == set(transformed)
                assert all(math.isclose(want[a], transformed[a]) for a in want), agent.name
            s = step(s, rng.choice(legal))


def test_rulebots_are_position_only_targets_for_the_response_attack():
    from uno.eval.attack import supports

    for agent in rulebot_league(TOY).values():
        support = supports(TOY, agent, games=15)
        assert support.conclusive and support.position_only, (agent.name, support)
