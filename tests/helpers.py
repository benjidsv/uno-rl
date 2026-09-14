"""Helpers to build hand-picked positions from card names (see CardType.__str__)."""

from __future__ import annotations

from collections import Counter

from uno import Play, Ruleset, State, initial_state


def tid(rs: Ruleset, name: str) -> int:
    """Type id from its display name: 'c0r1', 'c1S', 'c0R', 'c1+2', 'W', 'W+4'."""
    for i, ct in enumerate(rs.card_types):
        if str(ct) == name:
            return i
    raise KeyError(name)


def act(rs: Ruleset, name: str, color: int | None = None) -> int:
    return rs.action_index[Play(tid(rs, name), color)]


def names(rs: Ruleset, cards) -> list[str]:
    return [str(rs.card_types[c]) for c in cards]


def position(
    rs: Ruleset,
    hand0: list[str],
    hand1: list[str],
    pile: list[str] = (),
    seed: int | None = None,
) -> State:
    """Build the initial state whose deal is hand0, hand1, then `pile` on top of the
    draw pile (the first number card of `pile` becomes the first discard), with
    every remaining deck card appended below in sorted order."""
    assert len(hand0) == rs.hand_size and len(hand1) == rs.hand_size
    head = [tid(rs, n) for n in [*hand0, *hand1, *pile]]
    rest = Counter(rs.deck) - Counter(head)
    assert sum(rest.values()) == len(rs.deck) - len(head), "position uses cards not in the deck"
    perm = tuple(head + sorted(rest.elements()))
    return initial_state(rs, perm, seed)
