"""Shared test inputs: hypothesis strategies over rulesets and seeds, random walks, the
small reference ruleset and the census guard that keeps examples small."""

from __future__ import annotations

import random

from hypothesis import assume
from hypothesis import strategies as st

from uno import (
    Level,
    Ruleset,
    State,
    deal,
    initial_state,
    legal_actions,
    step,
)
from uno.game.chance import roots_and_expand
from uno.game.keys import KEYS
from uno.game.observe import observe
from uno.reference.census import census


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


def pair(rs, responder, opp, seat, level):
    agents = (responder, opp) if seat == 0 else (opp, responder)
    levels = (level, Level.L1) if seat == 0 else (Level.L1, level)
    return agents, levels


SMALL = Ruleset(
    colors=2, rank_copies=(1, 1, 1), skip_per_color=0, reverse_per_color=0, draw_two_per_color=0,
    wild_count=1, wild_draw_four_count=0, hand_size=2, reshuffle="none", max_turns=None,
)  # fmt: skip  # candidate D: 7 cards, pile 2, 365k histories


CAP = 5_000  # histories; the reference solvers walk States, so examples must stay small


def small(rs: Ruleset):
    """Census of a ruleset small enough for the reference solvers, or assume(False)."""
    assume(sum(rs.deck_counts[t] for t, ct in enumerate(rs.card_types) if ct.rank is not None) > 0)
    assume(rs.deck_size - 2 * rs.hand_size <= 5)  # the world tree has one root per ordering of the pile
    c = census(rs, node_cap=CAP)
    assume(c.complete)
    return c


def responder_infosets(rs, opp, seat, level) -> dict:
    """Every information set of the responder reachable under the opponent's
    support, with its legal actions. Independent of the solver's bookkeeping."""
    key = KEYS[level]
    roots, expand = roots_and_expand(rs, (level,))
    sets: dict = {}
    stack = [s for s, _ in roots]
    while stack:
        s = stack.pop()
        if s.terminal:
            continue
        ch = expand(s)
        if s.player == seat:
            sets.setdefault(key(s, seat), tuple(ch))
            for outs in ch.values():
                stack.extend(c for c, _ in outs)
        else:
            for a in opp.policy(observe(s, seat ^ 1, Level.L1), tuple(ch)):
                stack.extend(c for c, _ in ch[a])
    return sets
