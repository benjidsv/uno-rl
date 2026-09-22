"""Stage 1 acceptance for arm B's evidence: pinned histories and full/incremental agreement.

docs/rl/pipeline-plan.md section 13 asks for "full/incremental evidence agreement"
and pinned cases for "board rank/color, penalty draws through a reshuffle, actual
card counts, evidence ages, recent plays, and consumed setup tails", each able to
detect an intentionally incorrect extractor.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.helpers import tid
from tests.rl.helpers import decision_games, decision_states
from uno.game.observe import Level, observe
from uno.game.rules import FULL_UNO, MID_25, TOY, TOY2_SKIP
from uno.rl.env import features as F
from uno.rl.env.evidence import (
    HIST_FOLLOW,
    HIST_SWITCH,
    HIST_WILD,
    HISTORY_DECAYS,
    MAX_AGE,
    Evidence,
    Item,
)

RS = FULL_UNO
OPP = 1  # evidence is always about the other seat


def ev(public) -> Evidence:
    return Evidence.from_public(RS, tuple(public), OPP)


def setup(top: str = "c0r1", skipped=()) -> tuple:
    return ("setup", tid(RS, top), tuple(tid(RS, s) for s in skipped))


def play(p: int, card: str, color: int) -> tuple:
    return ("play", p, tid(RS, card), color)


# -- what a voluntary draw is indexed by ---------------------------------------


def test_evidence_records_the_full_board_not_only_the_color():
    """Contract section 2.2: the board at the draw is active color *and* top identity."""
    a = ev([setup("c0r1"), ("draw", OPP, 1)])
    b = ev([setup("c0r2"), ("draw", OPP, 1)])
    assert a.ages() == [(tid(RS, "c0r1"), 0, 0)]
    assert b.ages() == [(tid(RS, "c0r2"), 0, 0)]
    assert a.ages() != b.ages()  # same color, different top rank: different evidence


def test_wild_top_keeps_the_chosen_color_as_the_active_color():
    e = ev([setup("c0r1"), play(0, "W", 2), ("draw", OPP, 1)])
    assert e.ages() == [(tid(RS, "W"), 2, 0)]


def test_a_play_does_not_clear_earlier_evidence():
    """B extends the old K2 hint, which cleared on a play; this one must not."""
    e = ev([setup("c0r1"), ("draw", OPP, 1), play(OPP, "c0r1", 0), play(0, "c0r3", 0)])
    assert len(e.ages()) == 1


# -- ages -----------------------------------------------------------------------


def test_age_excludes_the_card_drawn_at_the_event_and_counts_cards_after():
    e = ev([setup("c0r1"), ("draw", OPP, 1)])
    assert e.ages()[0][2] == 0  # the card drawn at the event does not age it
    e2 = ev([setup("c0r1"), ("draw", OPP, 1), play(0, "c1r1", 1), ("draw", OPP, 1)])
    ages = sorted(a for _, _, a in e2.ages())
    assert ages == [0, 1]  # the second draw aged the first item by one card


def test_age_counts_cards_not_draw_events():
    e = ev([setup("c0r1"), ("draw", OPP, 1), play(0, "c0+2", 0), ("draw", OPP, 2)])
    items = e.ages()
    assert len(items) == 1 and items[0][2] == 2  # a +2 ages it by two cards, not by one event


def test_old_evidence_is_retained_in_the_overflow_bin():
    """The newest 5+ board survives and the global channel counts every old item."""
    public = [setup("c0r1"), ("draw", OPP, 1)]
    for _ in range(10):
        public += [play(0, "c1r1", 1), ("draw", OPP, 1)]
    e = ev(public)
    ages = [a for _, _, a in e.ages()]
    assert max(ages) == MAX_AGE
    assert ages.count(MAX_AGE) == 1
    assert len(e.items) == 11
    assert e.age_counts()[MAX_AGE] == 6

    sp = F.spec(RS, F.ARM_B)
    state = decision_states(RS, range(3))[0]
    row = F.extract(sp, [observe(state, state.player, Level.L1)], [e])
    assert row.glob[0, F.GLOB_EV_TOTAL + MAX_AGE] == pytest.approx(6 / 32)


def test_overflow_bin_deduplication_prevents_pair_collision():
    """One joint overflow pair distinguishes histories with identical marginals."""
    sp = F.spec(RS, F.ARM_B)
    o = observe(decision_states(RS, range(3))[0], 0, Level.L1)
    top0, top2 = tid(RS, "c0r1"), tid(RS, "c2r1")
    # The old all-item projection saw the same top cards, active colors and top
    # colors in both rows; only the pairing differs.
    a = Evidence(RS, OPP, [Item(top0, 1, 1), Item(top2, 3, 2)], cum_opp_drawn=10)
    b = Evidence(RS, OPP, [Item(top0, 3, 1), Item(top2, 1, 2)], cum_opp_drawn=10)
    old_a = sorted((it.top for it in a.items)), sorted((it.color for it in a.items))
    old_b = sorted((it.top for it in b.items)), sorted((it.color for it in b.items))
    assert old_a == old_b
    assert a.ages() == [(top2, 3, MAX_AGE)]
    assert b.ages() == [(top2, 1, MAX_AGE)]
    assert a.age_counts() == b.age_counts()

    fa, fb = F.extract(sp, [o], [a]), F.extract(sp, [o], [b])
    assert not np.array_equal(fa.col, fb.col)
    assert fa.glob[0, F.GLOB_EV_TOTAL + MAX_AGE] == pytest.approx(2 / 32)

    # The explicit schema-2 evaluator reproduces the launched diagnostic: oldest
    # representative and a one-item total scaled by both 8 and 32.
    assert F._identity_ages(a, 2) == [(top0, 1, MAX_AGE)]
    legacy = F.extract(F.spec(RS, F.ARM_B, version=2), [o], [a])
    assert legacy.glob[0, F.GLOB_EV_TOTAL + MAX_AGE] == pytest.approx(1 / 256)




# -- penalties ------------------------------------------------------------------


def test_penalty_draw_is_not_voluntary_evidence():
    e = ev([setup("c0r1"), play(0, "c0+2", 0), ("draw", OPP, 2)])
    assert e.ages() == []
    assert (e.penalty_events, e.penalty_cards) == (1, 2)


def test_penalty_draw_through_a_reshuffle_is_still_a_penalty():
    """The engine appends ("reshuffle",) between the play and the penalty draw when
    the pile runs out mid-penalty, so the scan must look past it."""
    e = ev([setup("c0r1"), play(0, "c0+2", 0), ("reshuffle",), ("draw", OPP, 1)])
    assert e.ages() == [], "a penalty draw after a reshuffle was read as voluntary evidence"
    assert (e.penalty_events, e.penalty_cards) == (1, 1)


def test_a_truncated_penalty_of_one_card_is_still_a_penalty():
    e = ev([setup("c0r1"), play(0, "W+4", 0), ("draw", OPP, 1)])
    assert e.ages() == [] and e.penalty_cards == 1


def test_own_penalty_draw_is_not_the_opponents_evidence():
    e = ev([setup("c0r1"), play(OPP, "c0+2", 0), ("draw", 0, 2)])
    assert e.ages() == [] and e.penalty_events == 0


# -- counters and recent play ----------------------------------------------------


def test_draws_since_last_opponent_play_counts_cards_and_resets_on_a_play():
    e = ev([setup("c0r1"), ("draw", OPP, 1), ("draw", OPP, 1)])
    assert e.drawn_since_opp_play == 2
    e2 = ev([setup("c0r1"), ("draw", OPP, 1), play(OPP, "c0r1", 0)])
    assert e2.drawn_since_opp_play == 0


def test_last_opponent_play_keeps_identity_and_chosen_color():
    e = ev([setup("c0r1"), play(OPP, "W", 3)])
    assert (e.last_play_card, e.last_play_color) == (tid(RS, "W"), 3)
    e2 = ev([setup("c0r1"), play(0, "c1r1", 1)])
    assert e2.last_play_card == -1  # our own play is not the opponent's


# -- arm C's play history -----------------------------------------------------------


def test_play_history_classifies_how_the_opponent_set_the_color():
    e = ev([setup("c0r1"), play(OPP, "c0r2", 0), play(OPP, "c1r2", 1), play(OPP, "W", 3)])
    fresh = e.hist_color[:, 0]  # (kinds, colors) at the fast decay; nothing drawn, so exact counts
    assert fresh[HIST_FOLLOW, 0] == 1.0  # red on red
    assert fresh[HIST_SWITCH, 1] == 1.0  # a matching rank moved it to color 1
    assert fresh[HIST_WILD, 3] == 1.0  # a wild named color 3
    assert fresh.sum() == 3.0
    assert e.hist_played[0, tid(RS, "W")] == 1.0 and e.hist_played[0].sum() == 3.0


def test_play_history_ignores_our_own_plays():
    e = ev([setup("c0r1"), play(0, "W", 2), play(0, "c2r5", 2)])
    assert not e.hist_color.any() and not e.hist_played.any()


def test_play_history_decays_per_card_the_opponent_draws_penalties_included():
    e = ev([setup("c0r1"), play(OPP, "W", 2), play(0, "c2+2", 2), ("draw", OPP, 2), ("draw", OPP, 1)])
    want = HISTORY_DECAYS**3  # a two-card penalty and one voluntary card
    assert np.allclose(e.hist_color[HIST_WILD, :, 2], want)
    assert np.allclose(e.hist_played[:, tid(RS, "W")], want)
    # our own draws do not age the opponent's history
    same = ev([setup("c0r1"), play(OPP, "W", 2), ("draw", 0, 1)])
    assert np.allclose(same.hist_color[HIST_WILD, :, 2], 1.0)


def test_repeated_plays_reinforce_rather_than_collide():
    e = ev([setup("c0r1"), play(OPP, "W", 2), play(0, "c2r1", 2), play(OPP, "W", 2)])
    assert np.allclose(e.hist_color[HIST_WILD, :, 2], 2.0)


def test_copy_does_not_share_history_arrays():
    e = ev([setup("c0r1"), play(OPP, "W", 2)])
    c = e.copy()
    c.hist_color[:] = 0.0
    c.hist_played[:] = 0.0
    assert e.hist_color.any() and e.hist_played.any()


# -- the setup tail ---------------------------------------------------------------


def test_setup_tail_is_bounded_by_the_pile_and_cleared_by_a_reshuffle():
    e = ev([setup("c0r1", ("c0S", "c1S", "W"))])
    assert e.setup_tail(pile_size=10) == 3
    assert e.setup_tail(pile_size=2) == 2  # the tail is being consumed
    after = ev([setup("c0r1", ("c0S", "c1S", "W")), ("reshuffle",)])
    assert after.setup_tail(pile_size=30) == 0


# -- full history vs incremental ---------------------------------------------------


@pytest.mark.parametrize("rs", (TOY, TOY2_SKIP, MID_25, FULL_UNO))
def test_incremental_update_equals_full_reconstruction(rs):
    """Pipeline plan section 5: incremental bin updates must agree with a full replay."""
    compared = 0
    for game in decision_games(rs, range(20)):
        incremental = {0: Evidence(rs, 1), 1: Evidence(rs, 0)}
        seen = {0: 0, 1: 0}
        for s in game:
            p = s.player
            seen[p] = incremental[p].update(s.public, seen[p])
            full = Evidence.from_public(rs, s.public, p ^ 1)
            assert sorted(incremental[p].ages()) == sorted(full.ages())
            assert incremental[p].penalty_events == full.penalty_events
            assert incremental[p].penalty_cards == full.penalty_cards
            assert incremental[p].drawn_since_opp_play == full.drawn_since_opp_play
            assert (incremental[p].last_play_card, incremental[p].last_play_color) == (full.last_play_card, full.last_play_color)
            assert incremental[p].reshuffles == full.reshuffles
            assert np.array_equal(incremental[p].hist_color, full.hist_color)
            assert np.array_equal(incremental[p].hist_played, full.hist_played)
            compared += 1
    assert compared > 100


@pytest.mark.parametrize("rs", (TOY2_SKIP, FULL_UNO))
@pytest.mark.parametrize("arm", (F.ARM_B, F.ARM_C))
def test_incremental_and_full_evidence_give_identical_features(rs, arm):
    sp = F.spec(rs, arm)
    obs, inc, full = [], [], []
    for game in decision_games(rs, range(10)):
        incremental = {0: Evidence(rs, 1), 1: Evidence(rs, 0)}
        seen = {0: 0, 1: 0}
        for s in game:
            p = s.player
            seen[p] = incremental[p].update(s.public, seen[p])
            obs.append(observe(s, p, Level.L1))
            inc.append(incremental[p].copy())
            full.append(Evidence.from_public(rs, s.public, p ^ 1))
    a, b = F.extract(sp, obs, inc), F.extract(sp, obs, full)
    assert np.array_equal(a.tok, b.tok) and np.array_equal(a.col, b.col) and np.array_equal(a.glob, b.glob)


def test_evidence_features_can_detect_a_broken_extractor():
    """An extractor that dropped the top identity would collapse these two rows."""
    sp = F.spec(RS, F.ARM_B)
    states = decision_states(RS, range(3))
    o = observe(states[0], states[0].player, Level.L1)
    a = F.extract(sp, [o], [ev([setup("c0r1"), ("draw", OPP, 1)])])
    b = F.extract(sp, [o], [ev([setup("c0r2"), ("draw", OPP, 1)])])
    assert not np.array_equal(a.tok, b.tok)
    # ... and a color-only summary would collapse these
    c = F.extract(sp, [o], [ev([setup("c0r1"), play(0, "c1r1", 1), ("draw", OPP, 1)])])
    assert not np.array_equal(a.tok, c.tok)
