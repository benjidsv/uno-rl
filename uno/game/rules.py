"""Ruleset: parameters, card types, deck multiset and action encoding.

See docs/spec.md sections 1, 2, 5 and 12.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from functools import cached_property
from typing import Literal


class Kind(IntEnum):
    NUMBER = 0
    SKIP = 1
    REVERSE = 2
    DRAW_TWO = 3
    WILD = 4
    WILD_DRAW_FOUR = 5


WILD_KINDS = frozenset({Kind.WILD, Kind.WILD_DRAW_FOUR})


@dataclass(frozen=True, slots=True)
class CardType:
    kind: Kind
    color: int | None  # None for wilds
    rank: int | None  # None unless NUMBER

    @property
    def is_wild(self) -> bool:
        return self.kind in WILD_KINDS

    def __str__(self) -> str:
        if self.kind is Kind.NUMBER:
            return f"c{self.color}r{self.rank}"
        if self.is_wild:
            return "W" if self.kind is Kind.WILD else "W+4"
        return f"c{self.color}{ {Kind.SKIP: 'S', Kind.REVERSE: 'R', Kind.DRAW_TWO: '+2'}[self.kind] }"


@dataclass(frozen=True, slots=True)
class Play:
    card: int  # type id
    color: int | None = None  # chosen color, wilds only


@dataclass(frozen=True, slots=True)
class Draw:
    pass


@dataclass(frozen=True, slots=True)
class Pass:
    pass


Action = Play | Draw | Pass

Reshuffle = Literal["none", "seeded"]
Draws = Literal["pile", "tickets"]


@dataclass(frozen=True)
class Ruleset:
    colors: int = 4
    rank_copies: tuple[int, ...] = (1, 2, 2, 2, 2, 2, 2, 2, 2, 2)
    skip_per_color: int = 2
    reverse_per_color: int = 2
    draw_two_per_color: int = 2
    wild_count: int = 4
    wild_draw_four_count: int = 4
    hand_size: int = 7
    draw_then_may_play: bool = True
    forced_play: bool = False
    reshuffle: Reshuffle = "seeded"
    max_turns: int | None = 1000
    # how a card is drawn (docs/spec.md section 8): "pile" pops the top of the pile, "tickets"
    # draws by per-draw tickets, same distribution, for coupled evaluation (step 9). Out of the
    # repr, which names cached trees: those are only ever compiled with "pile".
    draws: Draws = field(default="pile", repr=False)

    def __post_init__(self) -> None:
        if self.colors < 1:
            raise ValueError("need at least one color")
        if not self.rank_copies or any(c < 0 for c in self.rank_copies):
            raise ValueError("rank_copies must be non-empty and non-negative")
        if self.reshuffle not in ("none", "seeded"):
            raise ValueError(f"unknown reshuffle mode {self.reshuffle!r}")
        if self.draws not in ("pile", "tickets"):
            raise ValueError(f"unknown draws mode {self.draws!r}")
        if self.hand_size < 1:
            raise ValueError("hand_size must be >= 1")
        if self.deck_size < 2 * self.hand_size + 1:
            raise ValueError(
                f"deck of {self.deck_size} cards cannot deal {self.hand_size} each plus a discard"
            )

    # -- card types ---------------------------------------------------------

    @cached_property
    def card_types(self) -> tuple[CardType, ...]:
        types: list[CardType] = []
        for color in range(self.colors):
            for rank, copies in enumerate(self.rank_copies):
                if copies > 0:
                    types.append(CardType(Kind.NUMBER, color, rank))
        for kind, per_color in (
            (Kind.SKIP, self.skip_per_color),
            (Kind.REVERSE, self.reverse_per_color),
            (Kind.DRAW_TWO, self.draw_two_per_color),
        ):
            if per_color > 0:
                for color in range(self.colors):
                    types.append(CardType(kind, color, None))
        if self.wild_count > 0:
            types.append(CardType(Kind.WILD, None, None))
        if self.wild_draw_four_count > 0:
            types.append(CardType(Kind.WILD_DRAW_FOUR, None, None))
        return tuple(types)

    @cached_property
    def deck_counts(self) -> tuple[int, ...]:
        counts = []
        for t in self.card_types:
            match t.kind:
                case Kind.NUMBER:
                    counts.append(self.rank_copies[t.rank])
                case Kind.SKIP:
                    counts.append(self.skip_per_color)
                case Kind.REVERSE:
                    counts.append(self.reverse_per_color)
                case Kind.DRAW_TWO:
                    counts.append(self.draw_two_per_color)
                case Kind.WILD:
                    counts.append(self.wild_count)
                case Kind.WILD_DRAW_FOUR:
                    counts.append(self.wild_draw_four_count)
        return tuple(counts)

    @cached_property
    def deck(self) -> tuple[int, ...]:
        """The deck multiset expanded into a sorted tuple of type ids."""
        return tuple(t for t, n in enumerate(self.deck_counts) for _ in range(n))

    @cached_property
    def deck_start(self) -> tuple[int, ...]:
        """Index in `deck` of the first copy of each type."""
        return tuple(self.deck.index(t) for t in range(self.num_types))

    @property
    def deck_size(self) -> int:
        return len(self.deck)

    @property
    def num_types(self) -> int:
        return len(self.card_types)

    # -- actions ------------------------------------------------------------

    @cached_property
    def actions(self) -> tuple[Action, ...]:
        acts: list[Action] = []
        for t, ct in enumerate(self.card_types):
            if not ct.is_wild:
                acts.append(Play(t))
        for t, ct in enumerate(self.card_types):
            if ct.is_wild:
                for color in range(self.colors):
                    acts.append(Play(t, color))
        acts.append(Draw())
        acts.append(Pass())
        return tuple(acts)

    @cached_property
    def action_index(self) -> dict[Action, int]:
        return {a: i for i, a in enumerate(self.actions)}

    @property
    def num_actions(self) -> int:
        return len(self.actions)

    @property
    def draw_action(self) -> int:
        return self.num_actions - 2

    @property
    def pass_action(self) -> int:
        return self.num_actions - 1

    @cached_property
    def play_actions(self) -> tuple[tuple[int, ...], ...]:
        """For each type id, the action ids that play it (one per color for wilds)."""
        per_type: list[list[int]] = [[] for _ in self.card_types]
        for i, a in enumerate(self.actions):
            if isinstance(a, Play):
                per_type[a.card].append(i)
        return tuple(tuple(x) for x in per_type)


TOY_CANDIDATE = Ruleset(
    colors=2,
    rank_copies=(2, 2, 2),
    skip_per_color=0,
    reverse_per_color=0,
    draw_two_per_color=0,
    wild_count=1,
    wild_draw_four_count=0,
    hand_size=2,
    draw_then_may_play=True,
    forced_play=False,
    reshuffle="none",
    max_turns=None,
)

# Step 3 census (the `census` experiment, docs/results/03-08-foundations/03-census.md). Two lessons: the pile drives
# tree size (forced play removes the "draw with a play in hand" branch and cuts
# the tree ~65x), and per-card playability drives the first-player edge (with 2
# colors x 2 ranks nobody gets stuck, so seat 0 wins 72% of pure races; with 4
# colors x 2 ranks x 1 copy seat 0 is 59%). TOY_FULL is TOY without forced play
# (solved on the quotient since step 8.6).
TOY = Ruleset(
    colors=4,
    rank_copies=(1, 1),
    skip_per_color=0,
    reverse_per_color=0,
    draw_two_per_color=0,
    wild_count=1,
    wild_draw_four_count=0,
    hand_size=3,
    draw_then_may_play=True,
    forced_play=True,
    reshuffle="none",
    max_turns=None,
)

TOY_FULL = Ruleset(
    colors=4,
    rank_copies=(1, 1),
    skip_per_color=0,
    reverse_per_color=0,
    draw_two_per_color=0,
    wild_count=1,
    wild_draw_four_count=0,
    hand_size=3,
    draw_then_may_play=True,
    forced_play=False,
    reshuffle="none",
    max_turns=None,
)

FULL_UNO = Ruleset()

# Step 8.7 (docs/results/03-08-foundations/08.7-toy2.md): the position DP sized every 10-14 card candidate.
# Three colors x three ranks keeps the quotient at 312k histories with forced
# play and 28M without (the only candidate whose free-play version fits in
# memory), and moves the random-vs-random seat-0 score from 0.649 to 0.588.
# TOY2_SKIP (three colors, two ranks, one skip per color) is the action-card
# alternative at 923k histories forced; its free version is 77M, out of reach.
TOY2 = Ruleset(
    colors=3,
    rank_copies=(1, 1, 1),
    skip_per_color=0,
    reverse_per_color=0,
    draw_two_per_color=0,
    wild_count=1,
    wild_draw_four_count=0,
    hand_size=3,
    draw_then_may_play=True,
    forced_play=True,
    reshuffle="none",
    max_turns=None,
)

TOY2_FULL = Ruleset(
    colors=3,
    rank_copies=(1, 1, 1),
    skip_per_color=0,
    reverse_per_color=0,
    draw_two_per_color=0,
    wild_count=1,
    wild_draw_four_count=0,
    hand_size=3,
    draw_then_may_play=True,
    forced_play=False,
    reshuffle="none",
    max_turns=None,
)

TOY2_SKIP = Ruleset(
    colors=3,
    rank_copies=(1, 1),
    skip_per_color=1,
    reverse_per_color=0,
    draw_two_per_color=0,
    wild_count=1,
    wild_draw_four_count=0,
    hand_size=3,
    draw_then_may_play=True,
    forced_play=True,
    reshuffle="none",
    max_turns=None,
)

# Step 9.5b (`uv run uno aivat_uno sweep`, docs/results/09-evaluation/09.5b-results.md): intermediate rulesets
# between the toys and full UNO, sampled only, picked by a pre-declared rule on game length
# (Heuristic vs Random) and on the shares of the skill-carrying decisions against FULL_UNO's.
# Two candidate rounds: the first had nothing short enough with few enough wilds per card.
MID_12 = Ruleset(  # 26 cards, 13.7 actions per Heuristic vs Random game
    colors=3,
    rank_copies=(1, 2, 2),
    skip_per_color=1,
    reverse_per_color=1,
    draw_two_per_color=1,
    wild_count=1,
    wild_draw_four_count=1,
    hand_size=4,
)

MID_25 = Ruleset(  # 30 cards, 21.3 actions per Heuristic vs Random game
    colors=4,
    rank_copies=(1, 1, 1, 1),
    skip_per_color=1,
    reverse_per_color=1,
    draw_two_per_color=1,
    wild_count=1,
    wild_draw_four_count=1,
    hand_size=4,
)

# every named ruleset, in the order the cache names are looked up (`uno.store`)
NAMED: dict[str, Ruleset] = {
    "TOY": TOY,
    "TOY_FULL": TOY_FULL,
    "TOY2": TOY2,
    "TOY2_FULL": TOY2_FULL,
    "TOY2_SKIP": TOY2_SKIP,
    "TOY_CANDIDATE": TOY_CANDIDATE,
    "FULL_UNO": FULL_UNO,
    "MID_12": MID_12,
    "MID_25": MID_25,
}
