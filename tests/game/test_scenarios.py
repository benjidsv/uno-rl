"""One hand-built position per rule in docs/spec.md."""

from dataclasses import replace

import pytest

from tests.helpers import act, names, position, tid
from uno import (
    FULL_UNO,
    TOY_CANDIDATE,
    IllegalAction,
    Level,
    Phase,
    Ruleset,
    deal,
    initial_state,
    legal_actions,
    new_game,
    observe,
    reconstruct,
    returns,
    step,
    win_prob,
)

# 2 colors, ranks 0-1 with two copies, one of each action card per color, one W, one W+4.
SCEN = Ruleset(
    colors=2,
    rank_copies=(2, 2),
    skip_per_color=1,
    reverse_per_color=1,
    draw_two_per_color=1,
    wild_count=1,
    wild_draw_four_count=1,
    hand_size=2,
    reshuffle="none",
    max_turns=None,
)
# 1 color, 5 number cards, no action cards: tiny decks for end-of-game rules.
LINE = Ruleset(
    colors=1,
    rank_copies=(1, 1, 1, 1, 1),
    skip_per_color=0,
    reverse_per_color=0,
    draw_two_per_color=0,
    wild_count=0,
    wild_draw_four_count=0,
    hand_size=1,
    reshuffle="none",
    max_turns=None,
)


def play_seq(s, *actions):
    for a in actions:
        s = step(s, a)
    return s


# -- rules.py -----------------------------------------------------------------


def test_deck_sizes_and_action_spaces():
    assert TOY_CANDIDATE.deck_size == 13
    assert FULL_UNO.deck_size == 108
    # toy: 6 number types + 1 wild -> 6 plays + 2 wild plays + draw + pass
    assert TOY_CANDIDATE.num_actions == 6 + 2 + 2
    # full: 40 numbers + 12 action types = 52 plays, 2 wild types x 4 colors, draw, pass
    assert FULL_UNO.num_actions == 52 + 8 + 2
    assert FULL_UNO.actions[FULL_UNO.draw_action].__class__.__name__ == "Draw"
    assert FULL_UNO.actions[FULL_UNO.pass_action].__class__.__name__ == "Pass"


def test_invalid_rulesets():
    with pytest.raises(ValueError):
        Ruleset(colors=1, rank_copies=(1,), skip_per_color=0, reverse_per_color=0,
                draw_two_per_color=0, wild_count=0, wild_draw_four_count=0, hand_size=1)  # fmt: skip
    with pytest.raises(ValueError):
        replace(FULL_UNO, reshuffle="always")


def test_deal_is_a_seeded_permutation_of_the_deck():
    p = deal(FULL_UNO, 42)
    assert tuple(sorted(p)) == FULL_UNO.deck
    assert p == deal(FULL_UNO, 42)
    assert p != deal(FULL_UNO, 43)
    assert new_game(FULL_UNO, 42) == new_game(FULL_UNO, 42)


# -- setup --------------------------------------------------------------------


def test_setup_block_deal_and_first_discard_skips_action_cards():
    s = position(SCEN, ["c0r0", "c1r1"], ["c0S", "W+4"], ["c1S", "W", "c1r0", "c0r1"])
    assert names(SCEN, s.hands[0]) == ["c0r0", "c1r1"]
    assert names(SCEN, s.hands[1]) == ["c0S", "W+4"]
    assert names(SCEN, s.discard) == ["c1r0"]
    assert s.current_color == 1
    assert s.player == 0 and s.phase is Phase.TURN_START
    assert names(SCEN, s.draw_pile)[0] == "c0r1"
    assert names(SCEN, s.draw_pile)[-2:] == ["c1S", "W"]  # skipped cards go to the bottom, in order
    assert s.public == (("setup", tid(SCEN, "c1r0"), (tid(SCEN, "c1S"), tid(SCEN, "W"))),)


def test_setup_without_number_card_in_pile_is_invalid():
    rs = replace(LINE, rank_copies=(1, 1), skip_per_color=1)  # 2 numbers + 1 skip, hand 1
    perm = (tid(rs, "c0r0"), tid(rs, "c0r1"), tid(rs, "c0S"))
    with pytest.raises(ValueError):
        initial_state(rs, perm)


def test_seed_required_for_seeded_reshuffle():
    with pytest.raises(ValueError):
        initial_state(FULL_UNO, deal(FULL_UNO, 0))
    with pytest.raises(ValueError):
        initial_state(FULL_UNO, FULL_UNO.deck[::-1] + (0,), seed=0)  # not a permutation


# -- legality -----------------------------------------------------------------


def test_playable_matches_color_rank_kind_or_wild():
    # top c0r1, color 0: c0r0 by color, c1r1 by rank
    s = position(SCEN, ["c0r0", "c1r1"], ["c1S", "c0S"], ["c0r1"])
    assert legal_actions(s) == (act(SCEN, "c0r0"), act(SCEN, "c1r1"), SCEN.draw_action)
    # skip on skip regardless of color; a number of the other color is not playable
    rs = replace(SCEN, hand_size=3)
    s = position(rs, ["c0S", "c1S", "c1r0"], ["c0r0", "c0r0", "c1r1"], ["c0r1"])
    s = step(s, act(rs, "c0S"))
    assert legal_actions(s) == (act(rs, "c1S"), rs.draw_action)


def test_wild_plays_need_a_color_and_are_always_legal():
    s = position(SCEN, ["W", "c1r1"], ["c0r0", "c0r0"], ["c0r1"])
    la = legal_actions(s)
    assert act(SCEN, "W", 0) in la and act(SCEN, "W", 1) in la
    assert act(SCEN, "c1r1") in la  # rank match
    s = step(s, act(SCEN, "W", 1))
    assert s.current_color == 1 and s.player == 1
    assert names(SCEN, s.discard) == ["c0r1", "W"]
    assert s.public[-1] == ("play", 0, tid(SCEN, "W"), 1)


def test_illegal_actions_raise():
    s = position(SCEN, ["c0r0", "c1r1"], ["c1S", "c0S"], ["c0r1"])
    with pytest.raises(IllegalAction):
        step(s, SCEN.pass_action)  # pass only after a draw
    with pytest.raises(IllegalAction):
        step(s, act(SCEN, "c1S"))  # not in hand
    with pytest.raises(IllegalAction):
        step(s, act(SCEN, "c1r0"))  # p1's card
    with pytest.raises(IllegalAction):
        step(s, -1)


def test_forced_play_removes_draw_when_a_play_exists():
    rs = replace(SCEN, forced_play=True)
    s = position(rs, ["c0r0", "c1r1"], ["c1S", "c0S"], ["c0r1"])
    assert rs.draw_action not in legal_actions(s)
    s = position(rs, ["c1S", "c1r0"], ["c0r0", "c0r0"], ["c0r1"])
    assert legal_actions(s) == (rs.draw_action,)


# -- effects ------------------------------------------------------------------


def test_skip_and_reverse_give_the_same_player_another_turn():
    s = position(SCEN, ["c0S", "c0R"], ["c1r0", "c1r1"], ["c0r1"])
    s = step(s, act(SCEN, "c0S"))
    assert s.player == 0 and names(SCEN, s.hands[0]) == ["c0R"]
    s = step(s, act(SCEN, "c0R"))  # last card: win, effect irrelevant
    assert s.end == "win" and s.winner == 0
    assert returns(s) == (1.0, -1.0) and win_prob(s, 1) == 0.0


def test_draw_two_makes_opponent_draw_two_and_gives_another_turn():
    s = position(SCEN, ["c0+2", "c1r0"], ["c1r0", "c1r1"], ["c0r1", "c0r0", "W+4"])
    s = step(s, act(SCEN, "c0+2"))
    assert s.player == 0 and s.phase is Phase.TURN_START
    assert names(SCEN, s.hands[1]) == sorted(["c1r0", "c1r1", "c0r0", "W+4"], key=lambda n: tid(SCEN, n))
    assert s.own_draws[1] == (tid(SCEN, "c0r0"), tid(SCEN, "W+4"))
    assert s.public[-1] == ("draw", 1, 2)
    assert s.turn == 1


def test_wild_draw_four_makes_opponent_draw_four_then_same_player():
    s = position(SCEN, ["W+4", "c1r0"], ["c1r0", "c1r1"], ["c0r1"])
    s = step(s, act(SCEN, "W+4", 1))
    assert s.player == 0 and s.current_color == 1
    assert len(s.hands[1]) == 6 and s.public[-1] == ("draw", 1, 4)
    assert legal_actions(s) == (act(SCEN, "c1r0"), SCEN.draw_action)


def test_last_card_wins_even_when_it_is_a_penalty_card():
    s = position(SCEN, ["c0S", "W+4"], ["c1r0", "c1r1"], ["c0r1"])
    s = play_seq(s, act(SCEN, "c0S"), act(SCEN, "W+4", 0))
    assert s.end == "win" and s.winner == 0
    assert len(s.hands[1]) == 2  # no penalty draw after the win
    assert legal_actions(s) == ()
    with pytest.raises(IllegalAction):
        step(s, SCEN.draw_action)


# -- drawing ------------------------------------------------------------------


def test_draw_then_may_play_only_the_drawn_card_and_pass_is_silent():
    # hand has playable c0r0 too, but after drawing c0r1 only c0r1 / pass are legal
    s = position(SCEN, ["c0r0", "c1r0"], ["c1S", "c1R"], ["c0r1", "c0r1"])
    s = step(s, SCEN.draw_action)
    assert s.phase is Phase.AFTER_DRAW and s.player == 0
    assert s.drawn == tid(SCEN, "c0r1")
    assert legal_actions(s) == (act(SCEN, "c0r1"), SCEN.pass_action)
    assert s.public[-1] == ("draw", 0, 1) and s.turn == 1
    passed = step(s, SCEN.pass_action)
    assert passed.player == 1 and passed.phase is Phase.TURN_START and passed.drawn is None
    assert passed.public == s.public  # pass leaves no trace
    assert passed.turn == 1
    played = step(s, act(SCEN, "c0r1"))
    assert played.player == 1 and names(SCEN, played.discard) == ["c0r1", "c0r1"]


def test_unplayable_draw_ends_the_turn():
    s = position(SCEN, ["c0r0", "c1r0"], ["c1S", "c1R"], ["c0r1", "c1+2"])
    s = step(s, SCEN.draw_action)
    assert s.player == 1 and s.phase is Phase.TURN_START
    assert tid(SCEN, "c1+2") in s.hands[0]


def test_without_draw_then_may_play_the_turn_always_ends():
    rs = replace(SCEN, draw_then_may_play=False)
    s = position(rs, ["c0r0", "c1r0"], ["c1S", "c1R"], ["c0r1", "c0r1"])
    s = step(s, rs.draw_action)
    assert s.player == 1 and s.phase is Phase.TURN_START
    assert rs.pass_action not in legal_actions(s)


# -- end of game --------------------------------------------------------------


def test_exhaustion_tie_is_a_draw():
    rs = replace(LINE, rank_copies=(1, 1, 1))  # deck 3: hand, hand, top, empty pile
    s = position(rs, ["c0r0"], ["c0r1"], ["c0r2"])
    assert s.draw_pile == ()
    s = step(s, rs.draw_action)
    assert s.end == "exhaustion" and s.winner is None
    assert returns(s) == (0.0, 0.0) and win_prob(s, 0) == 0.5
    assert s.public[-1] == ("setup", tid(rs, "c0r2"), ())  # nothing was drawn, no draw event


def test_exhaustion_fewest_cards_wins_and_partial_penalty_draw_counts():
    rs = replace(LINE, draw_two_per_color=1, hand_size=2)  # 5 numbers + 1 draw two = 6 cards
    s = position(rs, ["c0+2", "c0r0"], ["c0r1", "c0r2"], ["c0r3", "c0r4"])
    assert len(s.draw_pile) == 1
    s = step(s, act(rs, "c0+2"))
    assert s.end == "exhaustion" and s.winner == 0
    assert len(s.hands[0]) == 1 and len(s.hands[1]) == 3
    assert s.public[-1] == ("draw", 1, 1)
    assert returns(s) == (1.0, -1.0)


def test_exhaustion_during_voluntary_draw_after_a_play():
    rs = replace(LINE, hand_size=2)  # 5 cards: 2 + 2 + top, empty pile
    s = position(rs, ["c0r0", "c0r1"], ["c0r2", "c0r3"], ["c0r4"])
    s = play_seq(s, act(rs, "c0r0"), rs.draw_action)
    assert s.end == "exhaustion" and s.winner == 0  # 1 card vs 2


def test_seeded_reshuffle_keeps_top_discard_and_is_reproducible():
    rs = replace(LINE, reshuffle="seeded", max_turns=100)

    def run(seed):
        s = position(rs, ["c0r0"], ["c0r1"], ["c0r2", "c0r3", "c0r4"], seed=seed)
        # p0 draws c0r3 (playable), passes; p1 draws c0r4, passes; p0 plays r0; p1 plays r1
        s = play_seq(s, rs.draw_action, rs.pass_action, rs.draw_action, rs.pass_action,
                     act(rs, "c0r0"), act(rs, "c0r1"))  # fmt: skip
        assert s.draw_pile == () and names(rs, s.discard) == ["c0r2", "c0r0", "c0r1"]
        return step(s, rs.draw_action)  # p0 needs a card: reshuffle

    a, b = run(7), run(7)
    assert a == b
    assert a.n_reshuffles == 1
    assert names(rs, a.discard) == ["c0r1"]
    assert sorted(names(rs, a.draw_pile + tuple(a.hands[0]))) == sorted(["c0r2", "c0r0", "c0r3"])
    assert ("reshuffle",) in a.public
    assert a.public.index(("reshuffle",)) == len(a.public) - 2 and a.public[-1] == ("draw", 0, 1)
    assert sorted(a.hands[0] + a.hands[1] + a.draw_pile + a.discard) == list(rs.deck)


def test_reshuffle_with_nothing_to_reshuffle_is_exhaustion():
    rs = replace(LINE, rank_copies=(1, 1, 1), reshuffle="seeded", max_turns=100)
    s = position(rs, ["c0r0"], ["c0r1"], ["c0r2"], seed=1)
    s = step(s, rs.draw_action)
    assert s.end == "exhaustion"


def test_turn_cap_counts_plays_and_draws_not_passes():
    rs = replace(LINE, reshuffle="seeded", max_turns=2)
    s = position(rs, ["c0r0"], ["c0r1"], ["c0r2", "c0r3", "c0r4"], seed=0)
    s = step(s, rs.draw_action)  # turn 1
    s = step(s, rs.pass_action)  # still 1
    assert s.turn == 1 and not s.terminal
    s = step(s, rs.draw_action)  # turn 2 -> cap
    assert s.end == "cap" and s.winner is None and returns(s) == (0.0, 0.0)


def test_win_beats_cap():
    rs = replace(LINE, reshuffle="seeded", max_turns=1)
    s = position(rs, ["c0r0"], ["c0r1"], ["c0r2", "c0r3", "c0r4"], seed=0)
    s = step(s, act(rs, "c0r0"))
    assert s.end == "win" and s.winner == 0


# -- observations -------------------------------------------------------------


def test_observation_levels():
    s = position(SCEN, ["c0r0", "c1r0"], ["c1S", "c1R"], ["c0r1", "c0r1"])
    s = step(s, SCEN.draw_action)  # p0 in AFTER_DRAW holding drawn c0r1
    o0, o1 = observe(s, 0), observe(s, 1)
    assert o0.hand == s.hands[0] and o0.opp_hand is None and o0.draw_pile is None
    assert o0.drawn == tid(SCEN, "c0r1") and o1.drawn is None  # p1 cannot see the drawn card
    assert o1.opp_hand_size == 3 and o1.public == s.public
    assert o0.info_state == (0, s.initial_hands[0], s.public, (tid(SCEN, "c0r1"),))
    o2 = observe(s, 1, Level.L2)
    assert o2.opp_hand == s.hands[0] and o2.draw_pile is None
    o3 = observe(s, 1, Level.L3)
    assert o3.draw_pile == s.draw_pile
    assert reconstruct(SCEN, o3) == s
    assert reconstruct(SCEN, observe(s, 0, Level.L3)) == s
    hash(o0), hash(o2), hash(o3)
    with pytest.raises(ValueError):
        reconstruct(SCEN, o2)


def test_full_uno_random_game_runs_to_an_end():
    import random

    rng = random.Random(0)
    s = new_game(FULL_UNO, 0)
    while not s.terminal:
        s = step(s, rng.choice(legal_actions(s)))
    assert s.end in ("win", "exhaustion", "cap")
    assert sorted(s.hands[0] + s.hands[1] + s.draw_pile + s.discard) == list(FULL_UNO.deck)
