"""RULES.md section 11: invariants checked on random games over random small rulesets."""

from __future__ import annotations

import copy
import random
from collections import Counter
from dataclasses import replace

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from uno import (
    IllegalAction,
    Kind,
    Level,
    Phase,
    Ruleset,
    State,
    deal,
    initial_state,
    legal_actions,
    observe,
    reconstruct,
    step,
)


@st.composite
def rulesets(draw) -> Ruleset:
    n_ranks = draw(st.integers(1, 4))
    reshuffle = draw(st.sampled_from(["none", "seeded"]))
    try:
        return Ruleset(
            colors=draw(st.integers(1, 4)),
            rank_copies=tuple(draw(st.lists(st.integers(1, 2), min_size=n_ranks, max_size=n_ranks))),
            skip_per_color=draw(st.integers(0, 2)),
            reverse_per_color=draw(st.integers(0, 2)),
            draw_two_per_color=draw(st.integers(0, 2)),
            wild_count=draw(st.integers(0, 2)),
            wild_draw_four_count=draw(st.integers(0, 2)),
            hand_size=draw(st.integers(1, 5)),
            draw_then_may_play=draw(st.booleans()),
            forced_play=draw(st.booleans()),
            reshuffle=reshuffle,
            max_turns=None if reshuffle == "none" else draw(st.integers(10, 150)),
        )
    except ValueError:  # deck too small for the hand size
        assume(False)


seeds = st.integers(0, 2**31 - 1)


def start(rs: Ruleset, seed: int) -> State:
    try:
        return initial_state(rs, deal(rs, seed), seed)
    except ValueError:  # no number card left for the first discard
        assume(False)


def walk(rs: Ruleset, seed: int, policy_seed: int) -> tuple[list[State], list[int]]:
    """Random legal play until terminal. Returns the states (initial first) and actions."""
    rng = random.Random(policy_seed)
    s = start(rs, seed)
    states, actions = [s], []
    while not s.terminal:
        a = rng.choice(legal_actions(s))
        actions.append(a)
        s = step(s, a)
        states.append(s)
    return states, actions


def multiset(s: State) -> Counter:
    return Counter(s.hands[0]) + Counter(s.hands[1]) + Counter(s.draw_pile) + Counter(s.discard)


game = given(rs=rulesets(), seed=seeds, policy_seed=seeds)
walks = settings(deadline=None)  # max_examples comes from the profile in conftest.py


@game
@walks
def test_1_conservation(rs, seed, policy_seed):
    deck = Counter(rs.deck)
    for s in walk(rs, seed, policy_seed)[0]:
        assert multiset(s) == deck
        assert s.hands[0] == tuple(sorted(s.hands[0])) and s.hands[1] == tuple(sorted(s.hands[1]))


@game
@walks
def test_2_legal_actions_apply_and_illegal_ones_raise(rs, seed, policy_seed):
    rng = random.Random(policy_seed)
    for s in walk(rs, seed, policy_seed)[0]:
        legal = legal_actions(s)
        for a in legal:
            step(s, a)  # must not raise
        illegal = [a for a in range(rs.num_actions) if a not in legal]
        for a in rng.sample(illegal, min(5, len(illegal))) + [-1, rs.num_actions]:
            try:
                step(s, a)
            except IllegalAction:
                continue
            raise AssertionError(f"action {a} should be illegal in {s!r}")


@game
@walks
def test_3_determinism(rs, seed, policy_seed):
    states, actions = walk(rs, seed, policy_seed)
    assert deal(rs, seed) == deal(rs, seed)
    s = initial_state(rs, deal(rs, seed), seed)
    assert s == states[0]
    for a, expected in zip(actions, states[1:]):
        s = step(s, a)
        assert s == expected
        assert hash(s) == hash(expected)


@game
@walks
def test_4_immutability(rs, seed, policy_seed):
    states, actions = walk(rs, seed, policy_seed)
    for s, a in zip(states, actions):
        before = copy.deepcopy(s)
        step(s, a)
        assert s == before


@game
@walks
def test_5_termination_bound(rs, seed, policy_seed):
    states, actions = walk(rs, seed, policy_seed)
    s0 = states[0]
    if rs.reshuffle == "none":
        potential = 2 * len(s0.draw_pile) + len(s0.hands[0]) + len(s0.hands[1])
        assert len(actions) <= 2 * potential + 1
    else:
        assert len(actions) <= 2 * rs.max_turns + 1
    # every non-pass action lowers the potential, except a draw from an empty pile
    # that ends the game by exhaustion (no card moves)
    pot = lambda x: 2 * len(x.draw_pile) + len(x.hands[0]) + len(x.hands[1])  # noqa: E731
    for s, a, t in zip(states, actions, states[1:]):
        if rs.reshuffle == "none" and a != rs.pass_action:
            assert pot(t) < pot(s) or (t.end == "exhaustion" and pot(t) == pot(s))


@game
@walks
def test_6_terminal_correctness(rs, seed, policy_seed):
    states, _ = walk(rs, seed, policy_seed)
    for s in states[:-1]:
        assert not s.terminal and s.end is None and s.winner is None
    s = states[-1]
    assert s.terminal
    n0, n1 = len(s.hands[0]), len(s.hands[1])
    match s.end:
        case "win":
            assert (n0 == 0) != (n1 == 0)
            assert s.winner == (0 if n0 == 0 else 1)
        case "exhaustion":
            assert s.draw_pile == ()
            assert rs.reshuffle == "none" or len(s.discard) == 1
            assert s.winner == (None if n0 == n1 else (0 if n0 < n1 else 1))
        case "cap":
            assert s.turn == rs.max_turns and s.winner is None
        case _:
            raise AssertionError(s.end)


@game
@walks
def test_7_progress_and_phase_consistency(rs, seed, policy_seed):
    states, _ = walk(rs, seed, policy_seed)
    for s in states[:-1]:
        la = legal_actions(s)
        assert la
        assert len(set(la)) == len(la)
        if s.phase is Phase.AFTER_DRAW:
            assert s.drawn is not None and s.drawn in s.hands[s.player]
            assert rs.draw_then_may_play
            assert la == rs.play_actions[s.drawn] + (rs.pass_action,)
        else:
            assert s.drawn is None
            assert rs.pass_action not in la
    assert legal_actions(states[-1]) == ()


@game
@walks
def test_8_current_color_matches_top(rs, seed, policy_seed):
    for s in walk(rs, seed, policy_seed)[0]:
        top = rs.card_types[s.top]
        if not top.is_wild:
            assert s.current_color == top.color
        assert 0 <= s.current_color < rs.colors
        # public history always agrees with the discard pile
        for ev in s.public:
            if ev[0] == "play":
                assert 0 <= ev[3] < rs.colors


@game
@walks
def test_9_information_levels(rs, seed, policy_seed):
    rng = random.Random(policy_seed)
    for s in walk(rs, seed, policy_seed)[0]:
        for p in (0, 1):
            o = p ^ 1
            l1, l2, l3 = (observe(s, p, lvl) for lvl in (Level.L1, Level.L2, Level.L3))
            # L1: permuting the unseen cards changes nothing
            unseen = list(s.hands[o]) + list(s.draw_pile)
            rng.shuffle(unseen)
            hands = list(s.hands)
            hands[o] = tuple(sorted(unseen[: len(s.hands[o])]))
            shuffled = replace(s, hands=tuple(hands), draw_pile=tuple(unseen[len(s.hands[o]) :]))
            assert observe(shuffled, p, Level.L1) == l1
            # L2: permuting the pile changes nothing
            pile = list(s.draw_pile)
            rng.shuffle(pile)
            assert observe(replace(s, draw_pile=tuple(pile)), p, Level.L2) == l2
            # L3 determines the state
            assert reconstruct(rs, l3) == s
            # nesting and privacy
            assert l1.opp_hand is None and l2.opp_hand == s.hands[o] and l3.draw_pile == s.draw_pile
            # L2 carries the opponent's memory (perfect recall), L1 does not, L2 has no pile
            assert l1.opp_initial_hand is None and l1.opp_draws is None
            assert l2.opp_initial_hand == s.initial_hands[o] and l2.opp_draws == s.own_draws[o]
            assert l2.draw_pile is None and l2.seed is None
            assert l1.hand == s.hands[p] and l1.opp_hand_size == len(s.hands[o])
            if s.player != p:
                assert l1.drawn is None
            hash(l1), hash(l2), hash(l3)


@game
@walks
def test_10_perfect_recall(rs, seed, policy_seed):
    for s in walk(rs, seed, policy_seed)[0]:
        for p in (0, 1):
            plays = Counter(ev[2] for ev in s.public if ev[0] == "play" and ev[1] == p)
            draws = Counter(s.own_draws[p])
            assert Counter(s.initial_hands[p]) + draws - plays == Counter(s.hands[p])
            drawn_total = sum(ev[2] for ev in s.public if ev[0] == "draw" and ev[1] == p)
            assert drawn_total == len(s.own_draws[p])


@given(rs=rulesets())
def test_action_encoding_is_complete(rs):
    plays = [a for a in rs.actions if a.__class__.__name__ == "Play"]
    for t, ct in enumerate(rs.card_types):
        ids = rs.play_actions[t]
        if ct.is_wild:
            assert len(ids) == rs.colors
            assert [rs.actions[i].color for i in ids] == list(range(rs.colors))
        else:
            assert len(ids) == 1 and rs.actions[ids[0]].color is None
    assert len(plays) == sum(len(x) for x in rs.play_actions)
    assert rs.num_actions == len(plays) + 2
    assert sum(rs.deck_counts) == rs.deck_size
    assert all(n > 0 for n in rs.deck_counts)
    kinds = [ct.kind for ct in rs.card_types]
    assert kinds == sorted(kinds)  # numbers, skips, reverses, +2, W, W+4
    assert kinds[0] is Kind.NUMBER
