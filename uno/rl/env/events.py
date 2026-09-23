"""The raw L1 event stream, and its per-decision count tables for the recurrent actor.

`Tape` is one seat's L1 event stream as small integer tokens: its initial hand, the
setup, every public event, and its own private draws placed right after the public
draw event they belong to. It is what `uno.experiments.belief_ceiling` fits its
sequence models on.

`EventCounter` bins the tokens a decision has not yet read -- everything since the
previous learner decision -- into fixed-size count tables (`event_width` numbers):

- `cards`  (6, num_types): card-carrying events by channel (`CARD_CHANNELS`) and type;
- `named`  (2, colors): plays by self / opponent, by the color they left active;
- `other`  (5,): draw events and drawn cards for self / opponent, and reshuffles.

These are all a recurrent actor needs from the interval
(`uno.rl.model.recurrent.EventExpansion` turns them into per-token inputs), so a
rollout stores one fixed-width row per decision rather than token streams.
"""

from __future__ import annotations

import numpy as np

from uno.game.rules import Ruleset

# Raw event tokens: (kind, actor, card, color, n) small integers.
PAD, INIT, SETUP, SKIP, PLAY, DRAW, OWN_DRAW, RESHUFFLE = range(8)
N_KINDS = 8
NOBODY, SELF, OPP = 0, 1, 2
MAX_N = 15

# Event kinds that carry a card: (kind, actor) -> channel.
CARD_CHANNELS = {(INIT, SELF): 0, (SETUP, NOBODY): 1, (SKIP, NOBODY): 2,
                 (PLAY, SELF): 3, (PLAY, OPP): 4, (OWN_DRAW, SELF): 5}
N_CHANNELS = len(CARD_CHANNELS)
N_OTHER = 5  # draw events (self, opp), drawn cards (self, opp), reshuffles


def event_width(rs: Ruleset) -> int:
    return N_CHANNELS * rs.num_types + 2 * rs.colors + N_OTHER


class Tape:
    """One game's L1 event stream for one seat, extended incrementally.

    Own private draws are placed right after the public draw event they belong to,
    using its actual count, as the observation contract requires for C-style history.
    """

    def __init__(self, rs, seat: int) -> None:
        self.rs, self.seat = rs, seat
        self.none_card, self.none_color = rs.num_types, rs.colors
        self.tokens: list[tuple[int, int, int, int, int]] = []
        self.public_seen = 0
        self.own_seen = 0

    def _who(self, p: int) -> int:
        return SELF if p == self.seat else OPP

    def advance(self, obs) -> int:
        """Append everything new in `obs`; return the prefix length a query may read."""
        nc, ncol = self.none_card, self.none_color
        if not self.tokens:
            self.tokens += [(INIT, SELF, c, ncol, 0) for c in sorted(obs.initial_hand)]
        for ev in obs.public[self.public_seen:]:
            match ev[0]:
                case "setup":
                    self.tokens.append((SETUP, NOBODY, ev[1], ncol, 0))
                    self.tokens += [(SKIP, NOBODY, c, ncol, 0) for c in ev[2]]
                case "play":
                    _, p, card, color = ev
                    self.tokens.append((PLAY, self._who(p), card, color, 0))
                case "draw":
                    _, p, n = ev
                    self.tokens.append((DRAW, self._who(p), nc, ncol, min(n, MAX_N)))
                    if p == self.seat:
                        cards = obs.own_draws[self.own_seen : self.own_seen + n]
                        if len(cards) != n:
                            raise ValueError("own private draws do not align with public draw events")
                        self.own_seen += n
                        self.tokens += [(OWN_DRAW, SELF, c, ncol, 0) for c in cards]
                case "reshuffle":
                    self.tokens.append((RESHUFFLE, NOBODY, nc, ncol, 0))
                case other:
                    raise ValueError(f"unknown public event {other!r}")
        self.public_seen = len(obs.public)
        return len(self.tokens)


def bin_events(rs: Ruleset, tokens) -> np.ndarray:
    """Count tables of `tokens` (one decision's interval), flattened to `event_width`."""
    k, colors = rs.num_types, rs.colors
    cards = np.zeros((N_CHANNELS, k), np.float32)
    named = np.zeros((2, colors), np.float32)
    other = np.zeros(N_OTHER, np.float32)
    for kind, actor, card, color, n in tokens:
        channel = CARD_CHANNELS.get((kind, actor))
        if channel is not None:
            cards[channel, card] += 1.0
        if kind == PLAY and color < colors:
            named[actor - 1, color] += 1.0
        elif kind == DRAW:
            other[actor - 1] += 1.0
            other[2 + actor - 1] += n
        elif kind == RESHUFFLE:
            other[4] += 1.0
    return np.concatenate([cards.ravel(), named.ravel(), other])


class EventCounter:
    """Per live game: the count tables of the events since the previous decision."""

    __slots__ = ("rs", "tape", "read")

    def __init__(self, rs: Ruleset, seat: int) -> None:
        self.rs, self.tape, self.read = rs, Tape(rs, seat), 0

    def advance(self, obs) -> np.ndarray:
        """Read `obs` (the seat's L1 observation at a decision); return its interval."""
        n = self.tape.advance(obs)
        row = bin_events(self.rs, self.tape.tokens[self.read : n])
        self.read = n
        return row


__all__ = [
    "CARD_CHANNELS", "DRAW", "EventCounter", "INIT", "MAX_N", "NOBODY", "N_KINDS", "OPP", "OWN_DRAW", "PAD",
    "PLAY", "RESHUFFLE", "SELF", "SETUP", "SKIP", "Tape", "bin_events", "event_width",
]  # fmt: skip
