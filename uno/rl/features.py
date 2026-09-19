"""The declared actor and critic inputs: arm A, arm B, and the separate critic.

This is the **production** feature path of docs/rl/observation-contract.md
sections 2.1-2.3 and 3. The test-only extractors of
`tests/game/test_observation_contract.py` and the distillation encoder
(`uno.eval.distill.arm_b_features`) certify nothing here; this module and
`uno.rl.model` are the single path used by collection, learning, evaluation and
checkpoint replay (contract section 5.6).

**Shape.** A row is three arrays, indexed rather than flattened
(`uno.rl.layout`):

| Array | Shape | What it carries |
|---|---|---|
| `tok` | `(n, colors * slots + wilds, tok_width)` | one token per card type |
| `col` | `(n, colors, col_width)` | one block per color |
| `glob` | `(n, glob_width)` | color- and rank-free context |

A color permutation permutes `tok`'s grid rows and `col`'s rows; a rank
permutation within a copy class permutes `tok`'s grid columns; `glob` is
invariant. `uno.rl.model` consumes exactly that structure, which is how
equivariance is achieved by construction rather than by canonicalization.

**Arms.** `A` is position plus discard, with no episode memory and no history
beyond the fields it declares. `B` is A plus the structured public evidence of
`uno.rl.evidence`, in extra channels -- A is B with that block absent, sharing
the position encoding exactly, so the arms differ in parameter count and that
difference is reported rather than denied (pipeline plan section 5). `CRITIC` is
B's fields plus the opponent's current hand multiplicities: an L2 subset, never
the pile order, the seed or the opponent's identity.

**Counts are scaled deterministically** -- by the ruleset's deck counts, hand
size and deck size -- so no statistics are fitted, nothing is standardized across
runs, and the scaling shares its constant across permutable coordinates by
construction (contract section 1.4).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np

from uno.game.engine import Phase
from uno.game.observe import Observation, discard_from_public
from uno.game.rules import Kind, Ruleset
from uno.rl.evidence import AGE_BINS, COUNTER_CAP, MULTIPLICITY_CAP, Evidence
from uno.rl.layout import Layout, layout

SCHEMA_VERSION = 3
EVALUATION_SCHEMA_VERSIONS = (2, SCHEMA_VERSION)

AGGREGATE_CAP = 8  # a per-color evidence aggregate is exposed as min(sum, 8) / 8

ARM_A, ARM_B, ARM_CRITIC = "A", "B", "CRITIC"
ARMS = (ARM_A, ARM_B, ARM_CRITIC)

# -- channel maps -------------------------------------------------------------
# Every arm shares the position block at offset 0; B's evidence block and the
# critic's opponent block are appended, so A's channels are a prefix of B's and
# B's a prefix of the critic's. Index constants are module-level because the
# tests pin individual channels by name.

TOK_HAND, TOK_IN_HAND, TOK_DISCARD, TOK_IS_TOP, TOK_IS_DRAWN, TOK_PLAYABLE = 0, 1, 2, 3, 4, 5
TOK_COPIES, TOK_IS_WILD, TOK_CUR_COLOR, TOK_TOP_COLOR = 6, 7, 8, 9
TOK_POSITION = 10
TOK_EV_SAME = TOK_POSITION  # ages 0..5, active color == the token's color
TOK_EV_DIFF = TOK_EV_SAME + AGE_BINS  # ages 0..5, active color != the token's color
TOK_LAST_PLAY = TOK_EV_DIFF + AGE_BINS
TOK_EVIDENCE_END = TOK_LAST_PLAY + 1
TOK_OPP_HAND, TOK_OPP_IN_HAND = TOK_EVIDENCE_END, TOK_EVIDENCE_END + 1
TOK_CRITIC_END = TOK_OPP_IN_HAND + 1

COL_CUR, COL_TOP, COL_DRAWN, COL_HAND, COL_DISCARD = 0, 1, 2, 3, 4
COL_POSITION = 5
COL_EV_DIAG = COL_POSITION  # active == this color, top of this color
COL_EV_ACTIVE = COL_EV_DIAG + AGE_BINS  # active == this color, top of another color
COL_EV_TOP = COL_EV_ACTIVE + AGE_BINS  # top of this color, active another color
COL_EV_WILD_TOP = COL_EV_TOP + AGE_BINS  # active == this color, wild top
COL_LAST_PLAY = COL_EV_WILD_TOP + AGE_BINS
COL_EVIDENCE_END = COL_LAST_PLAY + 1
COL_OPP_HAND = COL_EVIDENCE_END
COL_CRITIC_END = COL_OPP_HAND + 1

GLOB_HAND, GLOB_OPP_HAND, GLOB_PILE, GLOB_PILE_EMPTY, GLOB_TURN = 0, 1, 2, 3, 4
GLOB_RESHUFFLES, GLOB_AFTER_DRAW, GLOB_HAS_DRAWN, GLOB_SEAT = 5, 6, 7, 8
GLOB_TOP_KIND = 9  # one-hot over Kind
GLOB_DRAWN_KIND = GLOB_TOP_KIND + len(Kind)
GLOB_TOP_WILD = GLOB_DRAWN_KIND + len(Kind)
GLOB_DISCARD = GLOB_TOP_WILD + 1
GLOB_POSITION = GLOB_DISCARD + 1
GLOB_HAS_RESHUFFLED = GLOB_POSITION
GLOB_SETUP_TAIL = GLOB_HAS_RESHUFFLED + 1
GLOB_PENALTY_EVENTS = GLOB_SETUP_TAIL + 1
GLOB_PENALTY_CARDS = GLOB_PENALTY_EVENTS + 1
GLOB_DRAWS_SINCE_PLAY = GLOB_PENALTY_CARDS + 1
GLOB_HAS_LAST_PLAY = GLOB_DRAWS_SINCE_PLAY + 1
GLOB_EV_TOTAL = GLOB_HAS_LAST_PLAY + 1  # ages 0..5 over all evidence
GLOB_EVIDENCE_END = GLOB_EV_TOTAL + AGE_BINS
GLOB_CRITIC_END = GLOB_EVIDENCE_END


@dataclass(frozen=True, slots=True)
class Features:
    """One batch of declared inputs. Float32 rows, finite by contract."""

    tok: np.ndarray  # (n, n_tokens, tok_width)
    col: np.ndarray  # (n, colors, col_width)
    glob: np.ndarray  # (n, glob_width)

    def __len__(self) -> int:
        return len(self.glob)

    def __getitem__(self, idx) -> Features:
        return Features(self.tok[idx], self.col[idx], self.glob[idx])

    def check_finite(self) -> None:
        for name, a in (("tok", self.tok), ("col", self.col), ("glob", self.glob)):
            if not np.isfinite(a).all():
                raise ValueError(f"non-finite value in {name} features")


@dataclass(frozen=True)
class Spec:
    """The declared input schema of one arm on one ruleset."""

    rs: Ruleset
    arm: str
    layout: Layout
    tok_width: int
    col_width: int
    glob_width: int
    version: int = SCHEMA_VERSION

    @property
    def has_evidence(self) -> bool:
        return self.arm in (ARM_B, ARM_CRITIC)

    @property
    def has_opponent(self) -> bool:
        return self.arm == ARM_CRITIC

    @property
    def hash(self) -> str:
        """Stable schema hash, for checkpoints and run provenance."""
        from uno.eval.play import stable_hash

        return f"{stable_hash((self.version, repr(self.rs), self.arm, self.tok_width, self.col_width, self.glob_width)):016x}"


@cache
def spec(rs: Ruleset, arm: str, version: int = SCHEMA_VERSION) -> Spec:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
    if version not in EVALUATION_SCHEMA_VERSIONS:
        raise ValueError(
            f"unsupported learner feature schema {version}; "
            f"known evaluation schemas are {EVALUATION_SCHEMA_VERSIONS}"
        )
    lay = layout(rs)
    if arm == ARM_A:
        widths = (TOK_POSITION, COL_POSITION, GLOB_POSITION)
    elif arm == ARM_B:
        widths = (TOK_EVIDENCE_END, COL_EVIDENCE_END, GLOB_EVIDENCE_END)
    else:
        widths = (TOK_CRITIC_END, COL_CRITIC_END, GLOB_CRITIC_END)
    return Spec(rs, arm, lay, *widths, version=version)


# -- the production legality mask ---------------------------------------------


def legal_mask(rs: Ruleset, obs: Observation) -> np.ndarray:
    """The legal-action mask from the actor's **declared** inputs alone.

    `uno.game.engine.legal_actions` is the reference and reads the `State`; an
    actor never holds one, so the mask it actually applies has to be rebuilt from
    the observation's own hand, top, active color, phase and pending drawn card.
    `tests/rl/test_features.py` asserts equality with the engine over sampled and
    pinned states on several rulesets, which is stage 1's first acceptance check
    (pipeline plan section 13); it is not asserted to be true by construction.
    """
    lay = layout(rs)
    mask = np.zeros(rs.num_actions, dtype=bool)
    if obs.phase is Phase.AFTER_DRAW:
        mask[list(rs.play_actions[obs.drawn])] = True
        mask[rs.pass_action] = True
        return mask
    playable = lay.playable[obs.top, obs.current_color]
    any_play = False
    for t in sorted(set(obs.hand)):
        if playable[t]:
            mask[list(rs.play_actions[t])] = True
            any_play = True
    if not rs.forced_play or not any_play:
        mask[rs.draw_action] = True
    return mask


def legal_masks(rs: Ruleset, observations) -> np.ndarray:
    return np.stack([legal_mask(rs, o) for o in observations]) if observations else np.zeros((0, rs.num_actions), dtype=bool)


# -- extraction ----------------------------------------------------------------


def _multiplicities(cards, n_types: int) -> np.ndarray:
    out = np.zeros(n_types, dtype=np.float64)
    if cards:
        np.add.at(out, np.asarray(cards, dtype=np.int64), 1.0)
    return out


def extract(sp: Spec, observations, evidences=None) -> Features:
    """Declared features for a batch of decisions.

    `observations` are the acting seat's own observations (L1 for an actor arm,
    L2 for the critic). `evidences` is the matching `uno.rl.evidence.Evidence` per
    row for B and the critic, and is ignored for A -- passing one to A cannot
    change its output, which is what contract section 2.1's "no other historical
    information may affect its output" means and what the information test checks.
    """
    rs, lay = sp.rs, sp.layout
    n = len(observations)
    n_tok, n_col = lay.n_tokens, lay.colors
    tok = np.zeros((n, n_tok, sp.tok_width), dtype=np.float64)
    col = np.zeros((n, n_col, sp.col_width), dtype=np.float64)
    glob = np.zeros((n, sp.glob_width), dtype=np.float64)
    if n == 0:
        return Features(tok.astype(np.float32), col.astype(np.float32), glob.astype(np.float32))
    if sp.has_evidence and evidences is None:
        raise ValueError(f"arm {sp.arm} needs an Evidence per row")

    grid, wild_type = lay.grid_type, lay.wild_type
    type_color, type_slot, type_wild = lay.type_color, lay.type_slot, lay.type_wild
    deck = lay.deck_counts
    max_copies = float(deck.max())
    grid_tokens = grid.reshape(-1)  # token index c*slots+s holds type grid[c, s]
    order = np.concatenate([grid_tokens, wild_type])  # token index -> type id
    token_of_type = np.empty(rs.num_types, dtype=np.int64)
    token_of_type[order] = np.arange(len(order))
    is_wild_token = np.zeros(n_tok)
    is_wild_token[n_col * lay.slots :] = 1.0
    token_color = np.full(n_tok, -1, dtype=np.int64)
    token_color[: n_col * lay.slots] = np.repeat(np.arange(n_col), lay.slots)

    for i, obs in enumerate(observations):
        t, c, g = tok[i], col[i], glob[i]
        hand = _multiplicities(obs.hand, rs.num_types)[order]
        discard_cards = discard_from_public(obs.public)
        disc = _multiplicities(discard_cards, rs.num_types)[order]
        t[:, TOK_HAND] = hand / deck[order]
        t[:, TOK_IN_HAND] = hand > 0
        t[:, TOK_DISCARD] = disc / deck[order]
        t[token_of_type[obs.top], TOK_IS_TOP] = 1.0
        if obs.drawn is not None:
            t[token_of_type[obs.drawn], TOK_IS_DRAWN] = 1.0
        t[:, TOK_PLAYABLE] = lay.playable[obs.top, obs.current_color][order]
        t[:, TOK_COPIES] = deck[order] / max_copies
        t[:, TOK_IS_WILD] = is_wild_token
        t[: n_col * lay.slots, TOK_CUR_COLOR] = (token_color[: n_col * lay.slots] == obs.current_color)
        top_color = int(type_color[obs.top])
        if top_color >= 0:
            t[: n_col * lay.slots, TOK_TOP_COLOR] = (token_color[: n_col * lay.slots] == top_color)

        c[:, COL_CUR] = np.arange(n_col) == obs.current_color
        if top_color >= 0:
            c[top_color, COL_TOP] = 1.0
        if obs.drawn is not None and type_color[obs.drawn] >= 0:
            c[int(type_color[obs.drawn]), COL_DRAWN] = 1.0
        grid_hand = hand[: n_col * lay.slots].reshape(n_col, lay.slots)
        grid_disc = disc[: n_col * lay.slots].reshape(n_col, lay.slots)
        c[:, COL_HAND] = grid_hand.sum(1) / rs.hand_size
        c[:, COL_DISCARD] = grid_disc.sum(1) / rs.deck_size

        g[GLOB_HAND] = len(obs.hand) / rs.hand_size
        g[GLOB_OPP_HAND] = obs.opp_hand_size / rs.hand_size
        g[GLOB_PILE] = obs.pile_size / rs.deck_size
        g[GLOB_PILE_EMPTY] = obs.pile_size == 0
        g[GLOB_TURN] = obs.turn / rs.max_turns if rs.max_turns else 0.0
        reshuffles = sum(1 for ev in obs.public if ev[0] == "reshuffle")
        g[GLOB_RESHUFFLES] = min(reshuffles, 3) / 3.0
        g[GLOB_AFTER_DRAW] = obs.phase is Phase.AFTER_DRAW
        g[GLOB_HAS_DRAWN] = obs.drawn is not None
        g[GLOB_SEAT] = obs.seat == 0
        g[GLOB_TOP_KIND + int(rs.card_types[obs.top].kind)] = 1.0
        if obs.drawn is not None:
            g[GLOB_DRAWN_KIND + int(rs.card_types[obs.drawn].kind)] = 1.0
        g[GLOB_TOP_WILD] = rs.card_types[obs.top].is_wild
        g[GLOB_DISCARD] = len(discard_cards) / rs.deck_size

        if sp.has_evidence:
            _evidence_row(sp, obs, evidences[i], t, c, g, token_of_type, type_color)
        if sp.has_opponent:
            if obs.opp_hand is None:
                raise ValueError("the critic needs an L2 observation carrying opp_hand")
            opp = _multiplicities(obs.opp_hand, rs.num_types)[order]
            t[:, TOK_OPP_HAND] = opp / deck[order]
            t[:, TOK_OPP_IN_HAND] = opp > 0
            c[:, COL_OPP_HAND] = opp[: n_col * lay.slots].reshape(n_col, lay.slots).sum(1) / rs.hand_size

    out = Features(tok.astype(np.float32), col.astype(np.float32), glob.astype(np.float32))
    out.check_finite()
    return out


def _evidence_row(sp: Spec, obs: Observation, ev: Evidence, t, c, g, token_of_type, type_color) -> None:
    """Write B's evidence block for one row.

    Schema 3 returns at most one ``(top, active, age_bin)`` triple per bin and the
    5+ bin uses the most recent qualifying event. The read-only schema-2 evaluator
    reproduces its oldest-event rule. Each version therefore has one joint pair
    per bin, making the off-diagonal
    writes ``col[active, COL_EV_ACTIVE]`` and ``col[tc, COL_EV_TOP]`` together
    with ``tok[top_token, TOK_EV_DIFF]`` identify the specific board without
    collision.  (When multiple overflow-bin items existed under the old ``ages()``
    they could produce identical marginal projections for different histories;
    that is fixed at the ``Evidence`` level, not here.)

    A color permutation maps cell ``(a, top, age)`` to ``(g a, g top, age)``,
    so aggregating by its own indices keeps equivariance by construction.
    """
    from collections import Counter

    cells = Counter(_identity_ages(ev, sp.version))
    for (top, active, age), count in cells.items():
        v = min(count, MULTIPLICITY_CAP) / MULTIPLICITY_CAP
        tc = int(type_color[top])
        if tc < 0:  # a wild top has no color of its own
            t[token_of_type[top], TOK_EV_SAME + age] += v
            c[active, COL_EV_WILD_TOP + age] += v
        elif tc == active:
            t[token_of_type[top], TOK_EV_SAME + age] += v
            c[active, COL_EV_DIAG + age] += v
        else:
            t[token_of_type[top], TOK_EV_DIFF + age] += v
            c[active, COL_EV_ACTIVE + age] += v
            c[tc, COL_EV_TOP + age] += v

    # Aggregates are compressed to [0, 1] by declared, permutation-blind constants.
    # A token's block sums over at most `colors` cells, so dividing by that loses
    # nothing; the color and global blocks sum over an unbounded number of distinct
    # boards, so they are capped like the other counters. Caps are explicit
    # compression choices to diagnose if B stalls (pipeline plan section 5).
    t[:, TOK_EV_SAME:TOK_LAST_PLAY] /= sp.layout.colors
    block = c[:, COL_EV_DIAG:COL_LAST_PLAY]
    np.minimum(block, AGGREGATE_CAP, out=block)
    block /= AGGREGATE_CAP
    if sp.version == 2:
        # Exact compatibility for the already-running schema-2 diagnostic. It
        # accidentally derived totals from its one representative and scaled by
        # both multiplicity and counter caps. Keep this only for explicit replay.
        for (_, _, age), count in cells.items():
            g[GLOB_EV_TOTAL + age] += min(count, MULTIPLICITY_CAP) / MULTIPLICITY_CAP
        g[GLOB_EV_TOTAL:GLOB_EVIDENCE_END] /= COUNTER_CAP
    else:
        # Counts are independent of the representative board identities above. In
        # particular, the overflow total includes every 5+ item even though tok/col
        # retain one joint board pair. These are ordinary counters: cap once, then
        # scale once, like the penalty and draws-since-play fields below.
        g[GLOB_EV_TOTAL:GLOB_EVIDENCE_END] = (
            np.minimum(ev.age_counts(), COUNTER_CAP) / COUNTER_CAP
        )

    if ev.last_play_card >= 0:
        t[token_of_type[ev.last_play_card], TOK_LAST_PLAY] = 1.0
        if ev.last_play_color >= 0:
            c[ev.last_play_color, COL_LAST_PLAY] = 1.0
        g[GLOB_HAS_LAST_PLAY] = 1.0
    g[GLOB_HAS_RESHUFFLED] = ev.reshuffles > 0
    g[GLOB_SETUP_TAIL] = ev.setup_tail(obs.pile_size) / sp.rs.deck_size
    g[GLOB_PENALTY_EVENTS] = min(ev.penalty_events, COUNTER_CAP) / COUNTER_CAP
    g[GLOB_PENALTY_CARDS] = min(ev.penalty_cards, COUNTER_CAP) / COUNTER_CAP
    g[GLOB_DRAWS_SINCE_PLAY] = min(ev.drawn_since_opp_play, COUNTER_CAP) / COUNTER_CAP


def _identity_ages(ev: Evidence, version: int) -> list[tuple[int, int, int]]:
    """Versioned board-identity compression for checkpoint replay."""
    if version != 2:
        return ev.ages()
    result = []
    seen_overflow = False
    for it in ev.items:  # schema 2 retained the oldest, not the newest, 5+ item
        age = min(ev.cum_opp_drawn - it.mark, AGE_BINS - 1)
        if age < AGE_BINS - 1:
            result.append((it.top, it.color, age))
        elif not seen_overflow:
            result.append((it.top, it.color, age))
            seen_overflow = True
    return result


def extract_one(sp: Spec, obs: Observation, ev: Evidence | None = None) -> Features:
    """Single-row `extract`. The batched path is the production one; this exists so
    a scalar caller and a batch can be compared row for row."""
    return extract(sp, [obs], None if ev is None else [ev])


__all__ = [
    "ARM_A", "ARM_B", "ARM_CRITIC", "ARMS", "EVALUATION_SCHEMA_VERSIONS",
    "SCHEMA_VERSION", "Features", "Spec",
    "extract", "extract_one", "legal_mask", "legal_masks", "spec",
]  # fmt: skip
