"""AIVAT (Burch et al. 2016, arXiv:1612.06915) over recorded games: a fourth estimator
on the same traces as raw, mirrored and paired ("play once, estimate many ways",
`uno.eval.estimate`). See docs/spec.md section 23.

For a value model V (seat-0 return units) with node values `nu` and, for an action,
its pre-chance value `alpha(h, a)` (the expectation of `nu` over the action's chance
outcomes), the estimate of one game is

    X = u(z) - sum over the game's decisions of
          [ chance: nu(child) - alpha(h, a_taken) ]
          [ action, if the mover is corrected: alpha(h, a_taken) - sum_a sigma(a|h) alpha(h, a) ]
        - [ deal: nu(first state) - E_deals nu ]

Every bracket has conditional expectation zero given what came before it, provided
`alpha` is the expectation under the true chance model and `sigma` the policy that was
actually played, so X is unbiased for any V and any subset of corrections chosen before
the event it corrects. With V the exact values of the profile being played and every
correction on, the sum telescopes and X is the root value on every game.

V may condition on anything fixed *before* the chance event it integrates over: both
hands and the public history, the pile as a multiset, but never the pile order (a V that
reads the order makes every draw deterministic and leaves only the deal and reshuffle
permutations as chance events, whose expectations are intractable). On the toy that is
exactly the lazy tree's node value (`uno.compiled.aivat.TreeValues`).

The value model produces one `StepRecord` per decision; `game_terms` folds them into
per-game sums, so every subset of corrections costs one pass over the records.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterator, Protocol

import numpy as np

from uno.agents.baseline import Policy
from uno.eval.estimate import Estimate, Pairs, estimate
from uno.eval.play import Trace, play_from, stream_name
from uno.game.engine import new_game, returns
from uno.game.rules import Ruleset


@dataclass(frozen=True)
class Corrections:
    """Which corrections to apply, relative to the first-named (row) agent of a pair."""

    chance: bool = True
    row: bool = True
    column: bool = True
    deal: bool = True

    def players(self, seat: int) -> tuple[bool, bool]:
        """(seat 0 corrected, seat 1 corrected) when the row agent sits at `seat`."""
        return (self.row, self.column) if seat == 0 else (self.column, self.row)

    @property
    def label(self) -> str:
        parts = [name for name, on in (("chance", self.chance), ("row", self.row), ("column", self.column), ("deal", self.deal)) if on]
        return "+".join(parts) if parts else "none"


NONE = Corrections(False, False, False, False)
ALL = Corrections()


@dataclass(frozen=True, slots=True)
class StepRecord:
    """One decision of a game, as a value model sees it. Values in seat-0 return units."""

    player: int
    probs: Policy
    action: int
    node: float  # V at the decision's state
    alpha: dict[int, float | None]  # V after each action of the support, before chance; None: no chance model for it
    child: float  # V at the state after the step (at a terminal: V's terminal value)
    event: str = ""  # the chance event after the action: "", "voluntary draw", "penalty draw", "reshuffle draw" (for per-event checks)


class ValueModel(Protocol):
    exhaustion: str  # the scoring of exhaustion endings the model's values use (docs/spec.md section 9)
    deal_mean: float | None  # E over deals of V at the first state; None when not computable (full UNO)

    def records(self, trace: Trace) -> list[StepRecord]: ...


def rescored_return(trace: Trace, exhaustion: str = "hand") -> float:
    """Seat-0 return of a played game under a scoring of exhaustion endings, as
    `uno.compiled.tree.CompiledTree.rescored`: "hand" (the rule), "draw", or "stuck" (the
    player who had to draw from the empty pile loses: the mover on a `Draw`, the opponent
    on a play whose penalty draw ran out)."""
    last = trace.steps[-1]
    return terminal_return(last.state, last.action, trace.final, exhaustion)


def terminal_return(parent, action: int, child, exhaustion: str = "hand") -> float:
    """Seat-0 return of the terminal state `child` reached by `action` at `parent`, under a
    scoring of exhaustion endings (`rescored_return`)."""
    if exhaustion == "hand" or child.end != "exhaustion":
        return returns(child)[0]
    if exhaustion == "draw":
        return 0.0
    if exhaustion != "stuck":
        raise ValueError(f"unknown exhaustion scoring {exhaustion!r}")
    mover = parent.player
    drawer = mover if action == parent.ruleset.draw_action else mover ^ 1
    return 1.0 if drawer == 1 else -1.0


@dataclass(frozen=True, slots=True)
class GameTerms:
    """Per-game sums of the corrections (seat-0 return units); `estimate` applies a subset."""

    seed: int
    u: float  # seat-0 return under the model's scoring
    deal: float  # nan when the model has no deal mean
    chance: float
    action: tuple[float, float]  # per seat of the mover
    skipped_chance: int = 0  # decisions whose chance correction had no chance model
    skipped_action: int = 0  # decisions whose action correction had no chance model for some action of the support
    decisions: int = 0
    chance_by: dict[str, float] = field(default_factory=dict)  # the chance sum split by `StepRecord.event`

    def estimate(self, c: Corrections, seat: int) -> float:
        """Seat-0 return estimate when the row agent sits at `seat`."""
        x = self.u
        if c.chance:
            x -= self.chance
        for p, on in enumerate(c.players(seat)):
            if on:
                x -= self.action[p]
        if c.deal:
            if math.isnan(self.deal):
                raise ValueError("this value model has no deal mean: use a Corrections with deal=False")
            x -= self.deal
        return x


def game_terms(seed: int, records: list[StepRecord], u: float, deal_mean: float | None) -> GameTerms:
    chance = 0.0
    action = [0.0, 0.0]
    chance_by: dict[str, float] = {}
    skipped_chance = skipped_action = 0
    for r in records:
        taken = r.alpha.get(r.action)
        if taken is None:
            skipped_chance += 1
        else:
            chance += r.child - taken
            if r.event:
                chance_by[r.event] = chance_by.get(r.event, 0.0) + r.child - taken
        if taken is None or any(r.alpha.get(a) is None for a in r.probs):
            skipped_action += 1
        else:
            action[r.player] += taken - sum(p * r.alpha[a] for a, p in r.probs.items())
    deal = records[0].node - deal_mean if deal_mean is not None and records else math.nan
    return GameTerms(seed, u, deal, chance, (action[0], action[1]), skipped_chance, skipped_action, len(records), chance_by)


def trace_terms(model: ValueModel, seed: int, trace: Trace) -> GameTerms:
    return game_terms(seed, model.records(trace), rescored_return(trace, model.exhaustion), model.deal_mean)


def pair_terms(pairs: Pairs, models: tuple[ValueModel, ValueModel]) -> tuple[list[GameTerms], list[GameTerms]]:
    """Terms of every game of `pairs`; `models[s]` values the profile with the first-named
    agent at seat s (the two games of a pair are different profiles)."""
    out: tuple[list[GameTerms], list[GameTerms]] = ([], [])
    for g0, g1 in pairs.games:
        out[0].append(trace_terms(models[0], g0.seed, g0.trace))
        out[1].append(trace_terms(models[1], g1.seed, g1.trace))
    return out


def stream_pairs(rs: Ruleset, agents, seeds, models: tuple[ValueModel, ValueModel]) -> Iterator[tuple[tuple[GameTerms, GameTerms], tuple[list[StepRecord], list[StepRecord]]]]:
    """`play_pairs` that keeps no traces: yields, per valid seed, the terms and records of the two
    games (first-named agent at seat 0, then at seat 1). A `FULL_UNO` trace set of 4000 pairs holds
    ~1 GB of states (step 9.5a, bug 4); this holds one pair at a time."""
    names = (getattr(agents[0], "name", "?"), getattr(agents[1], "name", "?"))
    if agents[0] is not agents[1] and (names[0] == names[1] or stream_name(agents[0]) == stream_name(agents[1])):
        raise ValueError(f"two different agents named {names[0]!r} would share action rng streams (docs/spec.md section 13)")
    swapped = (agents[1], agents[0])
    for seed in seeds:
        try:
            s0 = new_game(rs, seed)
        except ValueError:
            continue
        t0, t1 = play_from(s0, seed, agents), play_from(s0, seed, swapped)
        r0, r1 = models[0].records(t0), models[1].records(t1)
        u0, u1 = rescored_return(t0, models[0].exhaustion), rescored_return(t1, models[1].exhaustion)
        yield (game_terms(seed, r0, u0, models[0].deal_mean), game_terms(seed, r1, u1, models[1].deal_mean)), (r0, r1)


def seat_scores(terms: list[GameTerms], c: Corrections, seat: int) -> np.ndarray:
    """The row agent's score per game at `seat` under corrections `c`."""
    sign = 1.0 if seat == 0 else -1.0
    return np.array([(sign * t.estimate(c, seat) + 1.0) / 2.0 for t in terms])


@dataclass(frozen=True)
class AivatEstimate:
    """AIVAT next to raw on the same games, first-named agent's score, per seat."""

    corrections: Corrections
    seat: tuple[Estimate, Estimate]
    raw: tuple[Estimate, Estimate]
    diff: tuple[Estimate, Estimate]  # per game AIVAT - raw: its mean is ~0 whatever the chance model's deal probabilities
    mirrored: Estimate  # over pairs, (AIVAT seat 0 + AIVAT seat 1) / 2 (seeds with both games only)
    scores: tuple[np.ndarray, np.ndarray]

    def var_ratio(self, seat: int) -> float:
        """Var(AIVAT) / Var(raw) at `seat`, same games: < 1 is the win."""
        return self.seat[seat].var / self.raw[seat].var if self.raw[seat].var else float("nan")

    def __str__(self) -> str:
        return "  ".join(f"seat{s} {self.seat[s]} (raw {self.raw[s]}, ratio {self.var_ratio(s):.3f})" for s in (0, 1)) + f"  mirrored {self.mirrored}"


def estimate_aivat(terms: tuple[list[GameTerms], list[GameTerms]], c: Corrections) -> AivatEstimate:
    scores = (seat_scores(terms[0], c, 0), seat_scores(terms[1], c, 1))
    raws = (seat_scores(terms[0], NONE, 0), seat_scores(terms[1], NONE, 1))
    n = min(len(scores[0]), len(scores[1]))
    return AivatEstimate(
        corrections=c,
        seat=(estimate(list(scores[0])), estimate(list(scores[1]))),
        raw=(estimate(list(raws[0])), estimate(list(raws[1]))),
        diff=(estimate(list(scores[0] - raws[0])), estimate(list(scores[1] - raws[1]))),
        mirrored=estimate(list((scores[0][:n] + scores[1][:n]) / 2)),
        scores=scores,
    )


def bootstrap_var(x: np.ndarray, reps: int = 2000, level: float = 0.998, seed: int = 0) -> tuple[float, float]:
    """Seeded percentile bootstrap CI of the sample variance of `x` (per-moment level
    99.8 %, as `uno.eval.estimate.bootstrap_moments`)."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    rng = np.random.default_rng(seed)
    out = []
    for start in range(0, reps, 100):
        xs = x[rng.integers(0, n, size=(min(100, reps - start), n))]
        out.append(xs.var(axis=1, ddof=1))
    q = ((1 - level) / 2, 1 - (1 - level) / 2)
    lo, hi = np.quantile(np.concatenate(out), q)
    return float(lo), float(hi)


def bootstrap_ratio(x: np.ndarray, y: np.ndarray, reps: int = 2000, level: float = 0.95, seed: int = 0) -> tuple[float, float]:
    """Seeded percentile bootstrap CI of Var(x) / Var(y), resampling the (x, y) pairs jointly."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    n = len(x)
    rng = np.random.default_rng(seed)
    out = []
    for start in range(0, reps, 100):
        idx = rng.integers(0, n, size=(min(100, reps - start), n))
        with np.errstate(invalid="ignore", divide="ignore"):
            out.append(x[idx].var(axis=1, ddof=1) / y[idx].var(axis=1, ddof=1))
    q = ((1 - level) / 2, 1 - (1 - level) / 2)
    lo, hi = np.nanquantile(np.concatenate(out), q)
    return float(lo), float(hi)


def mirror_regret(records: list[StepRecord], seat: int) -> float:
    """Sum over `seat`'s decisions of `V(h) - sum_a sigma(a|h) alpha(h, a)`: with V the
    values of a reference profile, the sampled edge decomposition of
    `uno.compiled.edge.sampled_mirror` (docs/spec.md section 21), read off the same
    records AIVAT uses. Seat-0 return units; negate for seat 1."""
    return sum(r.node - sum(p * r.alpha[a] for a, p in r.probs.items()) for r in records if r.player == seat)
