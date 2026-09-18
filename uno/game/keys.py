"""Information-set keys at L1 / L2 / L3 and their compact byte encoding.

See docs/foundations/spec.md sections 10 and 14 (the keys) and 16 (bytes keys in the compiled tree).
"""

from __future__ import annotations

from typing import Callable, Hashable

from uno.game.engine import State
from uno.game.observe import Level, Observation, info_state, reconstruct
from uno.game.rules import Ruleset

Key = Hashable


# -- information-set keys -----------------------------------------------------


def l1_key(state: State, seat: int) -> Key:
    return info_state(state, seat)


def l2_key(state: State, seat: int) -> Key:
    opp = seat ^ 1
    return (*info_state(state, seat), state.initial_hands[opp], state.own_draws[opp])


def l3_key(state: State, seat: int) -> Key:
    """The full state. Seed excluded: worlds carry none, real games do."""
    return (state.hands, state.draw_pile, state.player, state.phase, state.drawn, state.public, state.own_draws)


KEYS: dict[Level, Callable[[State, int], Key]] = {Level.L1: l1_key, Level.L2: l2_key, Level.L3: l3_key}


# -- keys as bytes: compact, hashable and storable (`uno.compiled` keeps one per set)

_EVENT_TAGS = {"setup": 0, "play": 1, "draw": 2, "reshuffle": 3}


def _put_public(out: bytearray, public: tuple) -> None:
    out.append(len(public))
    for ev in public:
        out.append(_EVENT_TAGS[ev[0]])
        match ev[0]:
            case "setup":
                out.append(ev[1]), out.append(len(ev[2])), out.extend(ev[2])
            case "play":
                out.extend((ev[1], ev[2], ev[3]))
            case "draw":
                out.extend((ev[1], ev[2]))


def _get_public(b: bytes, i: int) -> tuple[tuple, int]:
    n, i = b[i], i + 1
    out = []
    for _ in range(n):
        tag, i = b[i], i + 1
        if tag == 0:
            top, m = b[i], b[i + 1]
            out.append(("setup", top, tuple(b[i + 2 : i + 2 + m])))
            i += 2 + m
        elif tag == 1:
            out.append(("play", b[i], b[i + 1], b[i + 2]))
            i += 3
        elif tag == 2:
            out.append(("draw", b[i], b[i + 1]))
            i += 2
        else:
            out.append(("reshuffle",))
    return tuple(out), i


def _put_cards(out: bytearray, cards: tuple[int, ...]) -> None:
    out.append(len(cards)), out.extend(cards)


def _get_cards(b: bytes, i: int) -> tuple[tuple[int, ...], int]:
    n = b[i]
    return tuple(b[i + 1 : i + 1 + n]), i + 1 + n


def encode_key(level: Level, key: Key) -> bytes:
    """A key of `KEYS[level]` as bytes (type ids, colors and counts all < 255)."""
    out = bytearray()
    if level is Level.L3:
        hands, pile, player, phase, drawn, public, draws = key
        _put_cards(out, hands[0]), _put_cards(out, hands[1]), _put_cards(out, pile)
        out.extend((player, int(phase), 255 if drawn is None else drawn))
        _put_public(out, public)
        _put_cards(out, draws[0]), _put_cards(out, draws[1])
        return bytes(out)
    seat, hand, public, draws = key[:4]
    out.append(seat), _put_cards(out, hand), _put_public(out, public), _put_cards(out, draws)
    if level is Level.L2:
        _put_cards(out, key[4]), _put_cards(out, key[5])
    return bytes(out)


def decode_key(level: Level, b: bytes) -> Key:
    from uno.game.engine import Phase

    if level is Level.L3:
        h0, i = _get_cards(b, 0)
        h1, i = _get_cards(b, i)
        pile, i = _get_cards(b, i)
        player, phase, drawn = b[i], Phase(b[i + 1]), (None if b[i + 2] == 255 else b[i + 2])
        public, i = _get_public(b, i + 3)
        d0, i = _get_cards(b, i)
        d1, i = _get_cards(b, i)
        return ((h0, h1), pile, player, phase, drawn, public, (d0, d1))
    seat = b[0]
    hand, i = _get_cards(b, 1)
    public, i = _get_public(b, i)
    draws, i = _get_cards(b, i)
    if level is Level.L2:
        oh, i = _get_cards(b, i)
        od, i = _get_cards(b, i)
        return (seat, hand, public, draws, oh, od)
    return (seat, hand, public, draws)


def obs_key(rs: Ruleset, obs: Observation) -> Key:
    """The same key as `KEYS[obs.level]`, computed from an observation."""
    match obs.level:
        case Level.L1:
            return obs.info_state
        case Level.L2:
            return (*obs.info_state, obs.opp_initial_hand, obs.opp_draws)
        case Level.L3:
            return l3_key(reconstruct(rs, obs), obs.seat)
    raise ValueError(obs.level)
