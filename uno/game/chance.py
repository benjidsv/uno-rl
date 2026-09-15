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

import random
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
    if rs.draws != "pile":
        raise ValueError("the lazy-chance tree and the worlds need draws == 'pile': they place the drawn card on top")


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


def draw_outcomes(state: State, action: int, rng: random.Random | None = None, samples: int = 64, enum_cap: int = 256) -> list[Outcome] | None:
    """The chance model of any ruleset, for AIVAT (docs/spec.md section 23, step 9.5b):
    children of `state` under `action` with probabilities summing to 1, the drawn cards
    uniform over the unknown part of the pile given everything but the pile order (the
    same for pile and ticket draws, section 8).

    Unlike `chance_outcomes` this handles `reshuffle == "seeded"`: after the first
    reshuffle the whole pile is unknown, and a voluntary draw on an empty pile draws
    uniformly from the discard minus its top, the reshuffle being written out by hand
    (discard down to its top, `n_reshuffles + 1`, `("reshuffle",)` before the draw event,
    as the engine does). A draw that would empty the pile part-way and then reshuffle (a
    +2 or +4 on a short pile) returns None: no chance model, the caller skips the
    corrections that need it. When the ordered draws number more than `enum_cap`, they
    are sampled `samples` times from `rng` (equal weights; unbiased provided `rng` is
    independent of the game), or None without an rng. Children are stepped with
    `draws = "pile"` so that the chosen cards are the ones drawn."""
    rs = state.ruleset
    k = draw_count(state, action)
    if k == 0:
        return [(step(state, action, check=False), 1.0)]
    base = state if rs.draws == "pile" else replace(state, ruleset=replace(rs, draws="pile"))
    pile = state.draw_pile
    if not pile:
        if rs.reshuffle == "seeded" and len(state.discard) > 1:
            if not isinstance(rs.actions[action], Draw):
                return None  # a penalty draw on an empty pile reshuffles after the play
            rest = Counter(state.discard[:-1])
            total = sum(rest.values())
            out = []
            for c in sorted(rest):
                others = sorted((rest - Counter((c,))).elements())
                s = replace(base, draw_pile=(c, *others), discard=state.discard[-1:], n_reshuffles=state.n_reshuffles + 1, public=(*state.public, ("reshuffle",)))
                out.append((step(s, action, check=False), rest[c] / total))
            return out
        return [(step(base, action, check=False), 1.0)]  # exhaustion
    if state.n_reshuffles == 0:
        unknown, tail = pile_split(state)
    else:
        unknown, tail = Counter(pile), ()
    if k > len(pile) and rs.reshuffle == "seeded" and len(state.discard) + (0 if isinstance(rs.actions[action], Draw) else 1) > 1:
        return None  # the pile runs out part-way through the draw and reshuffles
    n_u = min(k, sum(unknown.values()))
    n_t = min(k - n_u, len(tail))
    from_tail, tail_rest = tail[:n_t], tail[n_t:]
    if len(unknown) ** n_u <= enum_cap:
        picks = ordered_picks(dict(unknown), n_u)
    elif rng is not None:
        cards = sorted(unknown.elements())
        picks = [(tuple(rng.sample(cards, n_u)), 1.0 / samples) for _ in range(samples)]
    else:
        return None
    out = []
    for seq, p in picks:
        rest = sorted((unknown - Counter(seq)).elements())
        out.append((step(replace(base, draw_pile=(*seq, *from_tail, *rest, *tail_rest)), action, check=False), p))
    return out


def children(state: State) -> dict[int, list[Outcome]]:
    return {a: chance_outcomes(state, a) for a in legal_actions(state)}


# -- initial deal -------------------------------------------------------------


def _multisets(counts: dict[int, int], n: int) -> list[tuple[tuple[int, ...], float]]:
    """All multisets of n cards (as non-decreasing tuples, in lexicographic order) with
    P(n cards dealt from `counts` form that multiset), the multivariate hypergeometric
    prod_t C(counts[t], m_t) / C(total, n).

    Step 8.9: this used to multiply by the number of orderings at every level of a
    recursion instead of once, which is wrong for n >= 3 with duplicate cards (every
    named toy has one copy per card, so no solved number changed)."""
    from math import comb

    types = sorted(t for t, c in counts.items() if c > 0)
    total = comb(sum(counts.values()), n)
    out: list[tuple[tuple[int, ...], float]] = []

    def rec(i: int, left: int, cards: tuple[int, ...], ways: int) -> None:
        if left == 0:
            out.append((cards, ways / total))
            return
        if i == len(types):
            return
        t = types[i]
        for m in range(min(counts[t], left), -1, -1):  # more copies of the lower type first: lexicographic order
            rec(i + 1, left - m, cards + (t,) * m, ways * comb(counts[t], m))

    rec(0, n, (), 1)
    return out


def initial_deals(rs: Ruleset, symmetric: bool = False) -> list[Outcome]:
    """All distinct initial states with their probabilities (sum to 1), or with
    `symmetric` one representative per color / rank orbit carrying the orbit's
    probability (`uno.game.symmetry`, docs/spec.md section 18).

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
    the orbit's probability (docs/spec.md section 18)."""
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
