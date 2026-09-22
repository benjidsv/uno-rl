"""The token grid and action index tables the learner's encoder is built on.

docs/rl/pipeline-plan.md section 5 asks for "colored card/rank tokens, separate
semantic action-card kinds, wild tokens, and indexed evidence relations", with no
learned absolute color IDs and no distinguishable embeddings for interchangeable
ranks. This module fixes that indexing once per ruleset.

**The grid.** Every colored card type is a token at `(color, slot)`; `slot`
enumerates the per-color card kinds in a fixed order -- the number ranks that have
copies, then SKIP, REVERSE, DRAW_TWO where the ruleset has them. Wild types are
separate tokens with no color. On `FULL_UNO` that is a 4 x 13 grid plus 2 wilds,
i.e. exactly the 54 card types.

**What the group does to it** (docs/foundations/spec.md section 18, contract
section 1.4). A color permutation permutes the grid's first axis. A rank
permutation within a copy class permutes number slots *within that class* and
fixes the action-card slots; wild tokens are fixed throughout. So a network is
equivariant by construction if it

1. embeds a slot only through `slot_class` -- the (kind, copies) class, equal for
   interchangeable ranks and distinct for distinct semantic kinds -- never through
   the slot index itself, and never through the color index at all;
2. processes every token with shared weights, and pools only over whole axes
   (colors, or the slots of one copy class) or over everything;
3. scores an action from its own token and its own color's context, never from a
   flattened ordered color block.

`slot_class` is what point 1 needs; `class_slots` is what a copy-class-respecting
pool needs; the `act_*` tables are what point 3 needs. Nothing here is learned or
fitted: the tables are a pure function of the `Ruleset`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np

from uno.game.engine import is_playable
from uno.game.rules import Draw, Kind, Pass, Play, Ruleset

# action categories, as `Layout.act_kind`
ACT_COLORED, ACT_WILD, ACT_DRAW, ACT_PASS = 0, 1, 2, 3


@dataclass(frozen=True)
class Layout:
    """Per-ruleset token grid, slot classes and action tables (`layout`)."""

    rs: Ruleset
    colors: int
    slots: int  # per-color slots
    wilds: int  # wild tokens
    slot_kind: tuple[int, ...]  # (slots,) Kind of each slot
    slot_rank: tuple[int, ...]  # (slots,) rank of a number slot, else -1
    slot_class: tuple[int, ...]  # (slots,) embedding class: equal iff interchangeable
    wild_class: tuple[int, ...]  # (wilds,) embedding class, disjoint from slot_class
    n_classes: int
    class_slots: tuple[tuple[int, ...], ...]  # slots of each class that has any
    grid_type: np.ndarray  # (colors, slots) type id
    wild_type: np.ndarray  # (wilds,) type id
    type_color: np.ndarray  # (num_types,) color, -1 for wilds
    type_slot: np.ndarray  # (num_types,) slot, -1 for wilds
    type_wild: np.ndarray  # (num_types,) wild index, -1 for colored
    deck_counts: np.ndarray  # (num_types,) float
    playable: np.ndarray  # (num_types, colors, num_types) bool: playable[top, color, card]
    # action tables
    act_kind: np.ndarray  # (num_actions,) ACT_*
    act_token: np.ndarray  # (num_actions,) index into the flattened token list, -1
    act_color: np.ndarray  # (num_actions,) color whose context the action reads, -1
    act_card: np.ndarray  # (num_actions,) played type id, -1 for Draw/Pass
    act_feat: np.ndarray  # (num_actions, act_feat_width) invariant static descriptor

    @property
    def n_tokens(self) -> int:
        """Grid tokens then wild tokens, as the model flattens them."""
        return self.colors * self.slots + self.wilds

    @property
    def act_feat_width(self) -> int:
        return self.act_feat.shape[1]

    def token_index(self, color: int, slot: int) -> int:
        return color * self.slots + slot

    def wild_index(self, wild: int) -> int:
        return self.colors * self.slots + wild


@cache
def layout(rs: Ruleset) -> Layout:
    """The token grid and action tables of a ruleset. Cached: pure and reused hot."""
    types = rs.card_types
    # -- slots ------------------------------------------------------------------
    slot_kind: list[int] = []
    slot_rank: list[int] = []
    slot_class: list[int] = []
    copy_class: dict[int, int] = {}
    for rank, copies in enumerate(rs.rank_copies):
        if copies > 0:
            slot_kind.append(int(Kind.NUMBER))
            slot_rank.append(rank)
            slot_class.append(copy_class.setdefault(copies, len(copy_class)))
    n_class = len(copy_class)
    for kind, per_color in ((Kind.SKIP, rs.skip_per_color), (Kind.REVERSE, rs.reverse_per_color), (Kind.DRAW_TWO, rs.draw_two_per_color)):
        if per_color > 0:
            slot_kind.append(int(kind))
            slot_rank.append(-1)
            slot_class.append(n_class)
            n_class += 1
    wild_kinds = [k for k, n in ((Kind.WILD, rs.wild_count), (Kind.WILD_DRAW_FOUR, rs.wild_draw_four_count)) if n > 0]
    wild_class = []
    for _ in wild_kinds:
        wild_class.append(n_class)
        n_class += 1
    slots, wilds = len(slot_kind), len(wild_kinds)

    # -- the grid ---------------------------------------------------------------
    index = {(int(t.kind), -1 if t.color is None else t.color, -1 if t.rank is None else t.rank): i for i, t in enumerate(types)}
    grid = np.empty((rs.colors, slots), dtype=np.int64)
    for c in range(rs.colors):
        for s in range(slots):
            grid[c, s] = index[(slot_kind[s], c, slot_rank[s])]
    wild_type = np.array([index[(int(k), -1, -1)] for k in wild_kinds], dtype=np.int64)

    type_color = np.full(rs.num_types, -1, dtype=np.int64)
    type_slot = np.full(rs.num_types, -1, dtype=np.int64)
    type_wild = np.full(rs.num_types, -1, dtype=np.int64)
    for c in range(rs.colors):
        for s in range(slots):
            type_color[grid[c, s]] = c
            type_slot[grid[c, s]] = s
    for w, t in enumerate(wild_type):
        type_wild[t] = w

    playable = np.zeros((rs.num_types, rs.colors, rs.num_types), dtype=bool)
    for top in range(rs.num_types):
        for color in range(rs.colors):
            playable[top, color] = [is_playable(rs, card, top, color) for card in range(rs.num_types)]

    # -- actions ----------------------------------------------------------------
    n_a = rs.num_actions
    act_kind = np.empty(n_a, dtype=np.int64)
    act_token = np.full(n_a, -1, dtype=np.int64)
    act_color = np.full(n_a, -1, dtype=np.int64)
    act_card = np.full(n_a, -1, dtype=np.int64)
    max_copies = float(max(rs.deck_counts))
    feats: list[list[float]] = []
    for i, a in enumerate(rs.actions):
        f = [0.0] * (4 + len(Kind) + 1)
        match a:
            case Play(card=card, color=color):
                ct = types[card]
                act_card[i] = card
                f[4 + int(ct.kind)] = 1.0
                f[-1] = rs.deck_counts[card] / max_copies
                if ct.is_wild:
                    act_kind[i] = ACT_WILD
                    act_token[i] = rs.colors * slots + int(type_wild[card])
                    act_color[i] = color
                    f[ACT_WILD] = 1.0
                else:
                    act_kind[i] = ACT_COLORED
                    act_token[i] = int(type_color[card]) * slots + int(type_slot[card])
                    act_color[i] = int(type_color[card])
                    f[ACT_COLORED] = 1.0
            case Draw():
                act_kind[i] = ACT_DRAW
                f[ACT_DRAW] = 1.0
            case Pass():
                act_kind[i] = ACT_PASS
                f[ACT_PASS] = 1.0
        feats.append(f)

    class_slots = tuple(tuple(s for s in range(slots) if slot_class[s] == k) for k in range(n_class))
    return Layout(
        rs=rs,
        colors=rs.colors,
        slots=slots,
        wilds=wilds,
        slot_kind=tuple(slot_kind),
        slot_rank=tuple(slot_rank),
        slot_class=tuple(slot_class),
        wild_class=tuple(wild_class),
        n_classes=n_class,
        class_slots=tuple(cs for cs in class_slots if cs),
        grid_type=grid,
        wild_type=wild_type,
        type_color=type_color,
        type_slot=type_slot,
        type_wild=type_wild,
        deck_counts=np.asarray(rs.deck_counts, dtype=np.float64),
        playable=playable,
        act_kind=act_kind,
        act_token=act_token,
        act_color=act_color,
        act_card=act_card,
        act_feat=np.asarray(feats, dtype=np.float64),
    )
