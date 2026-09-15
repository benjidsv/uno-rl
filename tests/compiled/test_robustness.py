"""Step 9 tools on the compiled tree: rescoring exhaustion, two-sided values, CFR with fixed sets, tied and unreached sets."""

from __future__ import annotations

from functools import cache

import numpy as np

from tests.strategies import SMALL
from uno.agents.baseline import PlayFirstRandom
from uno.compiled.cfr import cfr
from uno.compiled.solvers import (
    best_response,
    best_response_perfect,
    evaluate,
    two_sided,
)
from uno.compiled.strategies import tabulate
from uno.compiled.tree import END_KINDS, compile_tree
from uno.game.chance import children, engine_children, initial_deals, worlds
from uno.game.engine import returns
from uno.game.observe import Level
from uno.game.rules import Ruleset

TINY = Ruleset(
    colors=2, rank_copies=(1, 1), skip_per_color=1, reverse_per_color=0, draw_two_per_color=0,
    wild_count=1, wild_draw_four_count=0, hand_size=2, reshuffle="none", max_turns=None,
)  # fmt: skip  # 7 cards with skips and the voluntary draw, pile 2


def _reference_two_sided(rs: Ruleset, exhaustion: str):
    def util(s):
        if s.end != "exhaustion" or exhaustion == "hand":
            return returns(s)[0]
        if exhaustion == "draw":
            return 0.0
        return 1.0 if s.player == 1 else -1.0  # stuck: no +2 / +4 here, so the player to move is the one who drew

    @cache
    def l3(s, mode):
        if s.terminal:
            return util(s)
        vals = [l3(c, mode) for outs in engine_children(s).values() for c, _ in outs]
        if mode == "minimax":
            return max(vals) if s.player == 0 else min(vals)
        return max(vals) if mode == "max" else min(vals)

    @cache
    def l2(s):
        if s.terminal:
            return util(s)
        vals = [sum(p * l2(c) for c, p in outs) for outs in children(s).values()]
        return max(vals) if s.player == 0 else min(vals)

    ws = worlds(rs)
    value3 = sum(p * l3(s, "minimax") for s, p in ws)
    decided = sum(p for s, p in ws if l3(s, "max") == l3(s, "min"))
    value2 = sum(p * l2(s) for s, p in initial_deals(rs))
    return value2, value3, decided


def test_two_sided_and_rescored_match_brute_force():
    lazy = compile_tree(TINY, (Level.L1,))
    world = compile_tree(TINY, (Level.L1,), world=True)
    for exhaustion in ("hand", "draw", "stuck"):
        v2, v3, decided = _reference_two_sided(TINY, exhaustion)
        ts2, ts3 = two_sided(lazy.rescored(exhaustion)), two_sided(world.rescored(exhaustion))
        assert abs(ts2.value - v2) < 1e-9 and abs(ts3.value - v3) < 1e-9 and abs(ts3.decided - decided) < 1e-9
        assert abs(sum(ts3.outcome) - 1) < 1e-9
        # a fixed opponent does no better than a minimising one
        rnd_l, rnd_w = (Level.L1, tabulate(lazy, Level.L1, PlayFirstRandom(TINY))), (Level.L1, tabulate(world, Level.L1, PlayFirstRandom(TINY)))
        assert -best_response_perfect(world.rescored(exhaustion), rnd_w, 1).value - 1e-9 <= ts3.value <= best_response_perfect(world.rescored(exhaustion), rnd_w, 0).value + 1e-9
        assert evaluate(lazy.rescored(exhaustion), rnd_l, rnd_l)[0] <= best_response(lazy.rescored(exhaustion), Level.L1, rnd_l, 0).value + 1e-9
    drawn = lazy.rescored("draw")
    exh = lazy.end == END_KINDS.index("exhaustion")
    assert lazy.rescored("hand") is lazy
    assert np.all(drawn.util[exh] == 0) and np.array_equal(drawn.util[~exh], lazy.util[~exh])
    assert exh.any() and np.any(lazy.util[exh] != 0)


def test_cfr_with_fixed_sets_best_responds():
    tree = compile_tree(SMALL, (Level.L1,))
    k = tree.keyings[Level.L1]
    uniform = k.uniform()
    mask = k.info_player[k.ia_info] == 0
    res = cfr(tree, iterations=300, fixed=(mask, uniform), log_at=(300,))
    assert np.allclose(res.avg[mask], uniform[mask])
    v1 = evaluate(tree, (Level.L1, res.avg), (Level.L1, res.avg))[1]
    br = best_response(tree, Level.L1, (Level.L1, uniform), 1).value
    assert v1 <= br + 1e-9 and br - v1 < 2e-3  # within 0.1 score point of the best response


def test_unreached_sets_and_ties():
    tree = compile_tree(SMALL, (Level.L1,))
    k = tree.keyings[Level.L1]
    rnd = (Level.L1, tabulate(tree, Level.L1, PlayFirstRandom(SMALL)))
    for seat in (0, 1):
        br = best_response(tree, Level.L1, rnd, seat)
        assert br.unreached_sets.sum() == br.unreached > 0
        assert np.all(k.info_player[br.unreached_sets] == seat)
        n_seat = int((k.info_player == seat).sum())
        assert 0 <= br.ties <= n_seat - br.unreached
