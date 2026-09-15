"""Lazy-chance tree (uno/game/chance.py) and census (uno/reference/census.py): probabilities, canonical piles, brute-force cross-check."""

from __future__ import annotations

import itertools
import math
import random
from collections import Counter

from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from tests.strategies import lazy_rulesets, seeds, small_lazy_rulesets
from uno import Ruleset, initial_state, legal_actions, new_game, step
from uno.game.chance import _multisets, chance_outcomes, draw_count, initial_deals, pile_split
from uno.game.observe import info_state
from uno.reference.census import census


def multiset(s) -> Counter:
    return Counter(s.hands[0]) + Counter(s.hands[1]) + Counter(s.draw_pile) + Counter(s.discard)


@given(rs=lazy_rulesets(), seed=seeds, policy_seed=seeds)
@settings(deadline=None, max_examples=60, suppress_health_check=[HealthCheck.filter_too_much])
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


@given(counts=st.lists(st.integers(1, 3), min_size=1, max_size=5), n=st.integers(0, 4))
@settings(deadline=None, max_examples=60)
def test_multisets_are_hypergeometric(counts, n):
    """`_multisets` against every n-subset of the deck with copies told apart (step 8.9:
    the old recursion was wrong from n = 3 on with duplicates)."""
    counts = dict(enumerate(counts))
    deck = [t for t, c in counts.items() for _ in range(c)]
    assume(n <= len(deck))
    got = _multisets(counts, n)
    subsets = Counter(tuple(sorted(s)) for s in itertools.combinations(deck, n))
    assert [ms for ms, _ in got] == sorted(subsets), "every multiset once, in lexicographic order"
    for ms, p in got:
        assert math.isclose(p, subsets[ms] / math.comb(len(deck), n), abs_tol=1e-12), ms
    assert math.isclose(sum(p for _, p in got), 1.0)


def deals_by_permutation(rs: Ruleset) -> dict:
    """P(hands, setup event) over every permutation of the deck with copies told apart, given a valid deal."""
    out: Counter = Counter()
    n = 0
    for perm in itertools.permutations(rs.deck):
        try:
            s = initial_state(rs, perm)
        except ValueError:
            continue
        out[(s.hands, s.public[0])] += 1
        n += 1
    return {k: v / n for k, v in out.items()}


def check_deals_against_permutations(rs: Ruleset) -> None:
    exact = deals_by_permutation(rs)
    deals = initial_deals(rs)
    assert len(deals) == len(exact)
    for s, p in deals:
        assert math.isclose(p, exact[(s.hands, s.public[0])], abs_tol=1e-12), s


def test_initial_deals_with_duplicates_hand_3():
    """Regression (step 8.9): duplicate cards and a hand of 3, where the deal probabilities were wrong by up to 0.8 points."""
    for rs in (
        Ruleset(colors=2, rank_copies=(2, 1), skip_per_color=0, reverse_per_color=0, draw_two_per_color=0,
                wild_count=1, wild_draw_four_count=0, hand_size=3, forced_play=True, reshuffle="none", max_turns=None),
        Ruleset(colors=1, rank_copies=(2, 2), skip_per_color=1, reverse_per_color=0, draw_two_per_color=0,
                wild_count=2, wild_draw_four_count=0, hand_size=3, forced_play=False, reshuffle="none", max_turns=None),
    ):  # fmt: skip
        assert len(rs.deck) == 7 and len(set(rs.deck)) < 7
        check_deals_against_permutations(rs)


@given(rs=lazy_rulesets())
@settings(deadline=None, max_examples=20, suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow])
def test_initial_deals_match_permutations(rs):
    """Any hand size, duplicates allowed, decks of at most 7 cards (5040 permutations)."""
    assume(rs.deck_size <= 7)
    assume(any(ct.rank is not None for ct in rs.card_types))
    check_deals_against_permutations(rs)


@given(rs=small_lazy_rulesets())
@settings(deadline=None, max_examples=40, suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow])
def test_brute_force_hypothesis(rs):
    assume(rs.deck_size <= 6)
    assume(sum(rs.deck_counts[t] for t, ct in enumerate(rs.card_types) if ct.rank is not None) > 0)
    check_against_brute_force(rs)
