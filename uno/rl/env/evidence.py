"""Arm B's structured public-history summary, incrementally and from scratch.

docs/rl/observation-contract.md section 2.2 and docs/rl/pipeline-plan.md section 5
("B evidence") fix what this carries:

- **voluntary-draw evidence**, indexed by the *full board* it was observed against
  -- the active color and the top card's whole identity (rank/kind/color), not the
  color alone -- with the number of cards the opponent has drawn since as its age;
- **exact ages** `0, 1, 2, 3, 4` and a `5+` overflow bin. Old evidence is retained
  in the overflow bin rather than deleted, and a play never clears history;
- **penalty events and penalty cards** counted separately from voluntary draws,
  because a +2/+4 draw is not evidence of lacking a playable card;
- **cards drawn since the opponent's last play**, a card count and not an event
  count;
- the opponent's **most recent play**, as full card identity plus chosen color;
- the **setup tail** still at the pile bottom and whether a reshuffle has happened.

Arm C adds the opponent's **play history**, which B omits beyond the last play:
per color, how often they named it with a wild, switched to it with a colored card,
or followed it; and per card type, what they played. Each is an exponentially
decayed count, `sum(gamma ** cards_drawn_since)` at every `HISTORY_DECAYS` rate. A
play reveals something about the hand that played it (the rulebots score colour
support), and the cards the opponent has drawn since are what make that stale --
their later plays are known cards leaving the hand, which the played-card count
already carries. Decay replaces bins here because these are soft preferences
that accumulate, not facts about one board: repeats reinforce instead of
colliding, and old events fade instead of dropping off an overflow bin.

Voluntary draws are soft evidence (contract section 2.2): they restrict no
legality and assert no void. `since` excludes the card drawn *at* the event, which
matches the existing `uno.eval.belief.voids` signal, whose scope caveat carries
over unchanged -- the learner is expected to be able to discount the assumption,
not to inherit it as a fact.

**Ages are stored exactly.** An item keeps the cumulative opponent draw count at
its insertion, so its age is a subtraction at read time; nothing is re-binned or
rounded as the game goes on. That is what makes `Evidence.from_public` -- a full
replay -- and the incremental `update` agree exactly rather than approximately
(`tests/rl/test_evidence.py`), which docs/rl/pipeline-plan.md section 5 requires.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from uno.game.engine import Event
from uno.game.rules import Kind, Ruleset

AGE_BINS = 6  # exact ages 0..4 then a 5+ overflow bin
MAX_AGE = AGE_BINS - 1

MULTIPLICITY_CAP = 8  # an evidence cell's count is exposed as min(count, 8) / 8
COUNTER_CAP = 32  # penalty events/cards and draws-since-play are capped at 32

HISTORY_DECAYS = np.array([0.5, 0.85])  # arm C: per card the opponent draws; fast and slow memory
HIST_WILD, HIST_SWITCH, HIST_FOLLOW = 0, 1, 2  # how the opponent's play set the active color
HIST_KINDS = 3


def _is_penalty(rs: Ruleset, public: tuple[Event, ...], i: int) -> bool:
    """Whether the draw event at index i was forced by the other player's +2 / +4.

    Same contract as `uno.eval.belief._is_penalty`, with one difference the
    contract asks for explicitly: a reshuffle may sit between the play and the
    penalty draw it caused (`uno.game.engine._Work.draw` appends `("reshuffle",)`
    when the pile runs out mid-penalty), so the scan skips reshuffle events rather
    than looking only at `public[i - 1]`.
    """
    j = i - 1
    while j >= 0 and public[j][0] == "reshuffle":
        j -= 1
    if j < 0 or public[j][0] != "play" or public[j][1] == public[i][1]:
        return False
    return rs.card_types[public[j][2]].kind in (Kind.DRAW_TWO, Kind.WILD_DRAW_FOUR)


@dataclass(frozen=True, slots=True)
class Item:
    """One voluntary-draw observation: the board it was seen against, and when.

    `mark` is the cumulative count of cards the opponent had drawn at the moment of
    insertion (that event's own cards included), so `age = cum_drawn - mark`.
    """

    top: int
    color: int
    mark: int


@dataclass(slots=True)
class Evidence:
    """The opponent-evidence summary at one point of one episode, for one seat.

    Mutable and advanced by `update`; `ages` and `setup_tail` are what a feature row
    reads. Reset per episode and per seat: it is a cache of the public history, not
    a channel of extra information.
    """

    rs: Ruleset
    opp: int
    items: list[Item] = field(default_factory=list)
    cum_opp_drawn: int = 0
    penalty_events: int = 0
    penalty_cards: int = 0
    drawn_since_opp_play: int = 0
    last_play_card: int = -1
    last_play_color: int = -1
    reshuffles: int = 0
    setup_skipped: int = 0
    # board in force, needed to index a voluntary draw by what it was seen against
    top: int = -1
    color: int = -1
    # arm C's decayed play history: (HIST_KINDS, decays, colors) and (decays, num_types)
    hist_color: np.ndarray | None = None
    hist_played: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.hist_color is None:
            self.hist_color = np.zeros((HIST_KINDS, len(HISTORY_DECAYS), self.rs.colors))
        if self.hist_played is None:
            self.hist_played = np.zeros((len(HISTORY_DECAYS), self.rs.num_types))

    def copy(self) -> Evidence:
        return replace(
            self,
            items=list(self.items),
            hist_color=self.hist_color.copy(),
            hist_played=self.hist_played.copy(),
        )

    # -- advancing ---------------------------------------------------------------

    def update(self, public: tuple[Event, ...], start: int = 0) -> int:
        """Process `public[start:]` and return the new event count.

        A collector keeps one `Evidence` per live game and calls this with the
        index it stopped at, so each event is processed exactly once however many
        decisions the learner takes in between.
        """
        for i in range(start, len(public)):
            self._event(public, i)
        return len(public)

    def _event(self, public: tuple[Event, ...], i: int) -> None:
        rs = self.rs
        ev = public[i]
        match ev[0]:
            case "setup":
                self.setup_skipped = len(ev[2])
                self.top, self.color = ev[1], rs.card_types[ev[1]].color
            case "play":
                _, p, card, color = ev
                if p == self.opp:
                    self.last_play_card, self.last_play_color = card, color
                    self.drawn_since_opp_play = 0
                    if rs.card_types[card].is_wild:
                        kind = HIST_WILD
                    elif color != self.color:
                        kind = HIST_SWITCH
                    else:
                        kind = HIST_FOLLOW
                    self.hist_color[kind, :, color] += 1.0
                    self.hist_played[:, card] += 1.0
                self.top, self.color = card, color
            case "draw":
                _, p, n = ev
                if p != self.opp:
                    return
                penalty = _is_penalty(rs, public, i)
                decay = HISTORY_DECAYS**n  # every drawn card, penalty or not, dilutes what a play showed
                self.hist_color *= decay[None, :, None]
                self.hist_played *= decay[:, None]
                self.cum_opp_drawn += n
                self.drawn_since_opp_play += n
                if penalty:
                    self.penalty_events += 1
                    self.penalty_cards += n
                elif n == 1 and self.top >= 0:
                    # `since` excludes this event's own card: the item is inserted
                    # after `cum_opp_drawn` already counts it, so its age is 0 now.
                    self.items.append(Item(self.top, self.color, self.cum_opp_drawn))
            case "reshuffle":
                self.reshuffles += 1

    @classmethod
    def from_public(cls, rs: Ruleset, public: tuple[Event, ...], opp: int) -> Evidence:
        """Full-history reconstruction. The oracle the incremental path must match."""
        ev = cls(rs, opp)
        ev.update(public)
        return ev

    # -- reading -----------------------------------------------------------------

    def ages(self) -> list[tuple[int, int, int]]:
        """Board identity for exact-age items and one overflow representative.

        Bins 0–4 are unique by construction: each voluntary single-card draw
        increments ``cum_opp_drawn`` before inserting its item, so marks are
        strictly monotone. Items can therefore share a bin only at 5+. That bin
        retains the most recent qualifying item (largest mark): the freshest board
        whose exact age no longer fits. Keeping one joint ``(top, color)`` pair
        avoids the pairing ambiguity created by projecting several pairs into
        separate token and color marginals.

        ``age_counts`` separately preserves how many items occupy every bin. Older
        overflow board identities are intentionally compressed away, not claimed
        to survive in the token/color channels.
        """
        overflow = None
        for it in self.items:
            if self.cum_opp_drawn - it.mark >= MAX_AGE:
                overflow = it  # items are insertion ordered, so this ends newest
        result: list[tuple[int, int, int]] = []
        for it in self.items:
            age = min(self.cum_opp_drawn - it.mark, MAX_AGE)
            if age < MAX_AGE or it is overflow:
                result.append((it.top, it.color, age))
        return result

    def age_counts(self) -> tuple[int, ...]:
        """Number of voluntary-draw items in each age bin, including all at 5+."""
        counts = [0] * AGE_BINS
        for it in self.items:
            counts[min(self.cum_opp_drawn - it.mark, MAX_AGE)] += 1
        return tuple(counts)

    def setup_tail(self, pile_size: int) -> int:
        """Public setup-skipped cards still at the pile bottom (contract section 1.3).

        Zero once a reshuffle has happened; before that, bounded by the pile as the
        tail is consumed. `pile_size` comes from the observation, so the caller does
        not depend on this object's own pile bookkeeping.
        """
        return 0 if self.reshuffles else min(self.setup_skipped, pile_size)
