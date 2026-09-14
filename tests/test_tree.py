"""Lazy-chance tree (uno/tree.py): probabilities, canonical piles, brute-force cross-check."""

from __future__ import annotations

import itertools
import math
import random
from collections import Counter

from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from uno import Ruleset, initial_state, legal_actions, new_game, step
from uno.observe import info_state
from uno.tree import census, chance_outcomes, draw_count, initial_deals, pile_split

from .test_properties import rulesets, seeds


@st.composite
def lazy_rulesets(draw) -> Ruleset:
    rs = draw(rulesets())
    assume(rs.reshuffle == "none")
    return rs


@st.composite
def small_lazy_rulesets(draw) -> Ruleset:
    """Tiny decks (<= ~12 cards) so whole-tree enumeration stays cheap."""
    n_ranks = draw(st.integers(1, 3))
    try:
        return Ruleset(
            colors=draw(st.integers(1, 2)),
            rank_copies=tuple(draw(st.lists(st.integers(1, 2), min_size=n_ranks, max_size=n_ranks))),
            skip_per_color=draw(st.integers(0, 1)),
            reverse_per_color=draw(st.integers(0, 1)),
            draw_two_per_color=draw(st.integers(0, 1)),
            wild_count=draw(st.integers(0, 1)),
            wild_draw_four_count=draw(st.integers(0, 1)),
            hand_size=draw(st.integers(1, 2)),
            draw_then_may_play=draw(st.booleans()),
            forced_play=draw(st.booleans()),
            reshuffle="none",
            max_turns=None,
        )
    except ValueError:
        assume(False)


def multiset(s) -> Counter:
    return Counter(s.hands[0]) + Counter(s.hands[1]) + Counter(s.draw_pile) + Counter(s.discard)


@given(rs=lazy_rulesets(), seed=seeds, policy_seed=seeds)
@settings(deadline=None, max_examples=60)
def test_chance_outcomes_on_real_play(rs, seed, policy_seed):
    """On states from real play (non-canonical piles): probabilities sum to 1, every
    child is a legal step result, conservation holds, the drawn cards are the top of
    the constructed pile, and the pile decomposes into (unknown, known tail)."""
    assume(rs.deck_size <= 16)
    try:
        s = new_game(rs, seed)
    except ValueError:
        assume(False)
    rng = random.Random(policy_seed)
    while not s.terminal:
        unknown, tail = pile_split(s)
        assert unknown + Counter(tail) == Counter(s.draw_pile)
        # a +4 over many distinct types is thousands of ordered sequences: skip those
        acts = [a for a in legal_actions(s) if draw_count(s, a) < 4 or len(unknown) <= 3]
        ch = {a: chance_outcomes(s, a) for a in acts}
        for a, outcomes in ch.items():
            assert math.isclose(sum(p for _, p in outcomes), 1.0)
            assert all(p > 0 for _, p in outcomes)
            k = draw_count(s, a)
            if k == 0:
                assert len(outcomes) == 1
            drawer = s.player if k == 1 else s.player ^ 1
            for child, _ in outcomes:
                assert multiset(child) == multiset(s)
                n_drawn = len(child.own_draws[drawer]) - len(s.own_draws[drawer])
                assert n_drawn == min(k, len(s.draw_pile))
        a = rng.choice(legal_actions(s))
        s = step(s, a)


@given(rs=small_lazy_rulesets())
@settings(deadline=None, max_examples=50)
def test_initial_deals_mass(rs):
    assume(rs.deck_size <= 12)  # ordered setup prefixes explode with many non-number cards
    deals = initial_deals(rs)
    assert math.isclose(sum(p for _, p in deals), 1.0)
    keys = {(s.hands, s.top, s.public) for s, _ in deals}
    assert len(keys) == len(deals), "deals are merged by (hands, top, skipped)"
    for s, p in deals:
        assert p > 0
        _, tail = pile_split(s)
        assert tail == s.public[0][2]


# -- brute force over every permutation ---------------------------------------


def brute_force(rs: Ruleset):
    """Terminal (end, winner) mass under uniform-legal play, averaged over all valid
    permutations of the deck, plus the set of L1 information sets per seat."""
    perms = set(itertools.permutations(rs.deck))
    starts = []
    for perm in perms:
        try:
            starts.append(initial_state(rs, perm))
        except ValueError:
            pass
    mass: dict = {}
    infosets = (set(), set())
    histories = 0

    def rec(s, reach):
        nonlocal histories
        histories += 1
        if s.terminal:
            key = (s.end, s.winner)
            mass[key] = mass.get(key, 0.0) + reach
            return
        infosets[s.player].add(hash(info_state(s, s.player)))
        legal = legal_actions(s)
        for a in legal:
            rec(step(s, a), reach / len(legal))

    for s in starts:
        rec(s, 1.0 / len(starts))
    return mass, (len(infosets[0]), len(infosets[1]))


def check_against_brute_force(rs: Ruleset):
    mass, infosets = brute_force(rs)
    c = census(rs)
    assert c.complete
    assert c.info_sets == infosets
    assert set(c.end_mass) == set(mass)
    for k, v in mass.items():
        assert math.isclose(c.end_mass[k], v, abs_tol=1e-9), k
    assert math.isclose(sum(c.end_mass.values()), 1.0)


def test_brute_force_tiny():
    """1 color, ranks (2, 2), 1 wild, hand 1: 5 cards, 30 distinct permutations."""
    rs = Ruleset(
        colors=1, rank_copies=(2, 2), skip_per_color=0, reverse_per_color=0, draw_two_per_color=0,
        wild_count=1, wild_draw_four_count=0, hand_size=1, reshuffle="none", max_turns=None,
    )  # fmt: skip
    assert len(set(itertools.permutations(rs.deck))) == 30
    check_against_brute_force(rs)


def test_brute_force_with_action_cards():
    """Skip and +2 present, so penalty draws branch as ordered sequences."""
    rs = Ruleset(
        colors=2, rank_copies=(1,), skip_per_color=1, reverse_per_color=0, draw_two_per_color=1,
        wild_count=0, wild_draw_four_count=0, hand_size=1, reshuffle="none", max_turns=None,
    )  # fmt: skip
    assert rs.deck_size == 6
    check_against_brute_force(rs)


@given(rs=small_lazy_rulesets())
@settings(deadline=None, max_examples=40, suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow])
def test_brute_force_hypothesis(rs):
    assume(rs.deck_size <= 6)
    assume(sum(rs.deck_counts[t] for t, ct in enumerate(rs.card_types) if ct.rank is not None) > 0)
    check_against_brute_force(rs)
