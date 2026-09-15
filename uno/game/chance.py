"""Lazy-chance game tree over the engine: census now, exact solvers later.

The engine is a pure function of the deck permutation, but a solver does not
need the permutation: only the cards that actually get drawn matter. So here a
draw is a **chance node** over the distinct card types left in the unknown part
of the pile, weighted by multiplicity, and the initial deal is a chance node
over (hand multisets, setup prefix). Order that is never drawn never branches.

No engine change is needed: for each chance outcome we write a `draw_pile` with
the chosen cards on top (the rest in an arbitrary canonical order) and call
`step`, which only ever pops from the top.

The pile has two parts: an unknown multiset, and a publicly known *tail* (the
cards skipped at setup, placed at the bottom in order). Only `reshuffle ==
"none"` is supported: a seeded reshuffle would make the order a function of
the seed again.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import Callable

from uno.game.engine import State, initial_state, legal_actions, step
from uno.game.observe import Level
from uno.game.rules import Draw, Kind, Play, Ruleset

Outcome = tuple[State, float]


def _check(rs: Ruleset) -> None:
    if rs.reshuffle != "none":
        raise ValueError("the lazy-chance tree needs reshuffle == 'none'")


# -- pile decomposition -------------------------------------------------------


def pile_split(state: State) -> tuple[Counter, tuple[int, ...]]:
    """(unknown multiset, known tail) of the draw pile.

    The tail is whatever is left of the cards skipped at setup; they sit at the
    bottom of the pile in order, and this holds for any state, canonical or
    produced by real play, because the engine only ever pops from the top.
    """
    skipped = state.public[0][2]
    pile = state.draw_pile
    t = min(len(skipped), len(pile))
    cut = len(pile) - t
    tail = pile[cut:]
    assert tail == skipped[len(skipped) - t :], "pile bottom is not the setup's skipped cards"
    return Counter(pile[:cut]), tail


def ordered_picks(counts: dict[int, int], n: int) -> list[tuple[tuple[int, ...], float]]:
    """All ordered draws of n cards from a multiset, with their probabilities."""
    if n == 0:
        return [((), 1.0)]
    total = sum(counts.values())
    out = []
    for t in sorted(counts):
        c = counts[t]
        if c == 0:
            continue
        counts[t] -= 1
        for rest, p in ordered_picks(counts, n - 1):
            out.append(((t, *rest), p * c / total))
        counts[t] += 1
    return out


def draw_count(state: State, action: int) -> int:
    """How many cards the action will try to draw from the pile."""
    a = state.ruleset.actions[action]
    if isinstance(a, Draw):
        return 1
    if isinstance(a, Play):
        if len(state.hands[state.player]) == 1:
            return 0  # the play wins, no penalty draw
        kind = state.ruleset.card_types[a.card].kind
        if kind is Kind.DRAW_TWO:
            return 2
        if kind is Kind.WILD_DRAW_FOUR:
            return 4
    return 0


def chance_outcomes(state: State, action: int) -> list[Outcome]:
    """Children of `state` under `action`, with probabilities that sum to 1.

    Non-drawing actions have a single child. A drawing action branches over the
    ordered sequence of drawn types: from the unknown multiset first, then from
    the known tail, then nothing (the engine ends the game by exhaustion).
    """
    _check(state.ruleset)
    k = draw_count(state, action)
    if k == 0:
        return [(step(state, action, check=False), 1.0)]
    unknown, tail = pile_split(state)
    n_u = min(k, sum(unknown.values()))
    n_t = min(k - n_u, len(tail))
    from_tail, tail_rest = tail[:n_t], tail[n_t:]
    out: list[Outcome] = []
    for seq, p in ordered_picks(dict(unknown), n_u):
        rest = sorted((unknown - Counter(seq)).elements())
        pile = (*seq, *from_tail, *rest, *tail_rest)
        out.append((step(replace(state, draw_pile=pile), action, check=False), p))
    return out


def children(state: State) -> dict[int, list[Outcome]]:
    return {a: chance_outcomes(state, a) for a in legal_actions(state)}


# -- initial deal -------------------------------------------------------------


def _multisets(counts: dict[int, int], n: int, lo: int = 0) -> list[tuple[tuple[int, ...], float]]:
    """All multisets of n cards (as non-decreasing tuples) with P(drawing that multiset)."""
    if n == 0:
        return [((), 1.0)]
    total = sum(counts.values())
    out = []
    for t in sorted(counts):
        if t < lo:
            continue
        c = counts[t]
        if c == 0:
            continue
        counts[t] -= 1
        for rest, p in _multisets(counts, n - 1, t):
            # p is P(one specific ordering of rest); multiply by the number of
            # orderings of the whole multiset at the end, see _perms.
            out.append(((t, *rest), p * c / total))
        counts[t] += 1
    return [(ms, p * _perms(ms)) for ms, p in out]


def _perms(ms: tuple[int, ...]) -> int:
    from math import factorial

    n = factorial(len(ms))
    for m in Counter(ms).values():
        n //= factorial(m)
    return n


def initial_deals(rs: Ruleset, symmetric: bool = False) -> list[Outcome]:
    """All distinct initial states with their probabilities (sum to 1), or with
    `symmetric` one representative per color / rank orbit carrying the orbit's
    probability (`uno.game.symmetry`, RULES.md section 18).

    Conditioned on the deal being valid (a number card exists for the first
    discard); on the named toy rulesets every deal is valid.

    Cost: multisets for the hands, but the setup prefix (non-number cards
    popped until the first number) is an *ordered* sequence, so rulesets with
    many action cards and wilds blow up here. Fine for toys.
    """
    _check(rs)
    types = rs.card_types
    h = rs.hand_size
    counts = dict(enumerate(rs.deck_counts))
    out: list[Outcome] = []
    for hand0, p0 in _multisets(counts, h):
        c1 = dict(counts)
        for t in hand0:
            c1[t] -= 1
        for hand1, p1 in _multisets(c1, h):
            c2 = dict(c1)
            for t in hand1:
                c2[t] -= 1
            # setup prefix: ordered non-numbers until the first number
            for prefix, p2 in _prefixes(c2, types):
                c3 = dict(c2)
                for t in prefix:
                    c3[t] -= 1
                rest = sorted(Counter(c3).elements())
                s = initial_state(rs, (*hand0, *hand1, *prefix, *rest))
                out.append((s, p0 * p1 * p2))
    mass = sum(p for _, p in out)
    out = [(s, p / mass) for s, p in out]
    if symmetric:
        from uno.game.symmetry import canonical_roots

        return canonical_roots(rs, out, lazy=True)
    return out


def _prefixes(counts: dict[int, int], types) -> list[tuple[tuple[int, ...], float]]:
    total = sum(counts.values())
    out = []
    for t in sorted(counts):
        c = counts[t]
        if c == 0:
            continue
        p = c / total
        if types[t].kind is Kind.NUMBER:
            out.append(((t,), p))
        else:
            counts[t] -= 1
            out.extend(((t, *rest), p * q) for rest, q in _prefixes(counts, types))
            counts[t] += 1
    return out


Expand = Callable[[State], dict[int, list[Outcome]]]


# -- roots and expansion ------------------------------------------------------


def worlds(rs: Ruleset, symmetric: bool = False) -> list[Outcome]:
    """L3 roots: every deal x every order of the unknown part of the pile.
    `symmetric`: one representative per orbit under `uno.game.symmetry`, carrying
    the orbit's probability (RULES.md section 18)."""
    out: list[Outcome] = []
    for s, p in initial_deals(rs):
        unknown, tail = pile_split(s)
        for seq, q in ordered_picks(dict(unknown), sum(unknown.values())):
            out.append((replace(s, draw_pile=(*seq, *tail)), p * q))
    if symmetric:
        from uno.game.symmetry import canonical_roots

        return canonical_roots(rs, out, lazy=False)
    return out


def engine_children(state: State) -> dict[int, list[Outcome]]:
    """Children under the bare engine: the pile order is known, so no chance."""
    return {a: [(step(state, a, check=False), 1.0)] for a in legal_actions(state)}


def roots_and_expand(rs: Ruleset, levels) -> tuple[list[Outcome], Expand]:
    """Worlds + engine when anyone plays at L3, the lazy tree otherwise."""
    if Level.L3 in levels:
        return worlds(rs), engine_children
    return initial_deals(rs), children
