"""Estimators over recorded games: raw, mirrored (duplicate deals), paired
comparison, and their exact counterparts on the toy. See docs/spec.md section 15
and docs/plan.md ("A better evaluator").

Play once, estimate many ways. `play_pairs` plays every seed from both seats
(the first-named agent at seat 0, then at seat 1, same deal and pile order)
and keeps the traces; the estimators below are pure functions of the records,
so a later estimator (AIVAT, step 9) reads the same games.

What mirroring controls. The two games of a pair share the deal and the pile
order, so a deal that favours seat 0 favours it in both games and the
first-named agent gets it once from each side. The pair score `(seat-0 score +
seat-1 score) / 2` estimates the seat-averaged value, and its variance is
`(var0 + var1 + 2 cov) / 4` per pair against `(var0 + var1) / 4` for two
independent games: the whole gain is the negative covariance between the two
seats' scores on the same deal. Per-seat estimates are ordinary means over the
seat's games and gain nothing from the pairing; they are reported alongside
because seat is worth ~15 points on `TOY` and the pair score alone pools them.

Common random numbers. `uno.eval.play` seeds the action rng from (seed, information
state), so an opponent facing two different row agents on the same deal makes
the same choices until the public histories diverge. That is what makes
`compare` (paired difference of two row agents against the same column) tight.
A consequence: the two games of a self-play pair (same agent both seats) are
the same game, so every pair scores exactly 1/2 and the mirrored variance is
zero. The variance-reduction check therefore needs two different agents.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from uno.agents.baseline import Agent
from uno.eval.play import Trace, play_from
from uno.game.engine import new_game, returns
from uno.game.observe import Level
from uno.game.rules import Ruleset
from uno.progress import bar

Levels = tuple[Level, Level]


# -- records ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Game:
    seed: int
    seat: int  # seat of the first-named agent
    score: float  # first-named agent's score, draws worth 1/2
    trace: Trace


@dataclass(frozen=True)
class Pairs:
    """Every seed played from both seats by the first-named agent."""

    names: tuple[str, str]
    levels: Levels
    seeds: tuple[int, ...]  # valid seeds, in order
    games: tuple[tuple[Game, Game], ...]  # (first-named at seat 0, first-named at seat 1)
    invalid: int = 0

    @property
    def n(self) -> int:
        return len(self.games)

    def seat_scores(self, seat: int) -> list[float]:
        return [g[seat].score for g in self.games]

    def pair_scores(self) -> list[float]:
        return [(g0.score + g1.score) / 2 for g0, g1 in self.games]


def score_of(trace: Trace, seat: int) -> float:
    return (returns(trace.final)[seat] + 1.0) / 2.0


def play_pairs(rs: Ruleset, agents: tuple[Agent, Agent], seeds, levels: Levels = (Level.L1, Level.L1)) -> Pairs:
    """Play every seed twice: `agents[0]` at seat 0, then at seat 1, same deal.
    Seeds whose deal is invalid are skipped and counted.

    The action rng is keyed on the agent's name (`uno.eval.play`), so a name must
    identify a policy: two different agents with the same name would draw the same
    random numbers at the same seat of the two games and couple the pair beyond the
    deal. That raises; self-play passes the same object twice."""
    names = (getattr(agents[0], "name", "?"), getattr(agents[1], "name", "?"))
    if agents[0] is not agents[1] and names[0] == names[1]:
        raise ValueError(
            f"two different agents named {names[0]!r} would share action rng streams (docs/spec.md section 13): "
            "give them distinct names, or pass the same object twice for self-play"
        )
    swapped, lv_swapped = (agents[1], agents[0]), (levels[1], levels[0])
    games, valid, invalid = [], [], 0
    for seed in bar(seeds, desc=f"pairs {names[0]} vs {names[1]}", unit="pairs"):
        try:
            s0 = new_game(rs, seed)
        except ValueError:
            invalid += 1
            continue
        t0 = play_from(s0, seed, agents, levels)
        t1 = play_from(s0, seed, swapped, lv_swapped)
        games.append((Game(seed, 0, score_of(t0, 0), t0), Game(seed, 1, score_of(t1, 1), t1)))
        valid.append(seed)
    return Pairs(names, levels, tuple(valid), tuple(games), invalid)


# -- statistics ---------------------------------------------------------------


@dataclass(frozen=True)
class Estimate:
    """Sample mean with a normal 95 % CI. `var` is the per-sample variance."""

    n: int
    mean: float
    var: float

    @property
    def se(self) -> float:
        return math.sqrt(self.var / self.n) if self.n else float("nan")

    @property
    def ci(self) -> tuple[float, float]:
        return (self.mean - 1.96 * self.se, self.mean + 1.96 * self.se)

    @property
    def half(self) -> float:
        return 1.96 * self.se

    def __str__(self) -> str:
        return f"{self.mean:.4f} ± {self.half:.4f}"


def mean_var(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    m = sum(xs) / n
    return m, sum((x - m) ** 2 for x in xs) / max(n - 1, 1)


def estimate(xs: list[float]) -> Estimate:
    m, v = mean_var(xs)
    return Estimate(len(xs), m, v)


def correlation(xs: list[float], ys: list[float]) -> float:
    mx, vx = mean_var(xs)
    my, vy = mean_var(ys)
    if vx == 0 or vy == 0:
        return float("nan")
    c = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (len(xs) - 1)
    return c / math.sqrt(vx * vy)


def games_needed(edge: float, var_per_game: float, z: float = 3.0) -> float:
    """Games for an edge of `edge` score points to be `z` sigma, given the
    estimator's variance per game played."""
    return z * z * var_per_game / (edge * edge)


# -- estimators -----------------------------------------------------------------


@dataclass(frozen=True)
class Mirrored:
    """Raw and mirrored estimates from the same pairs, first-named agent's score.

    `seat[0]`, `seat[1]`: per-seat means (n games each; pairing does not help them).
    `raw`: the seat-averaged mean as if the 2n games were independent, i.e. what a
    raw design of n games per seat on different deals gives; its variance is
    estimated from the same games since pairing changes only the covariance.
    `mirrored`: the same mean with the CI over the n pair scores.
    """

    seat: tuple[Estimate, Estimate]
    raw: Estimate
    mirrored: Estimate
    corr: float  # sample correlation between the two seats' scores on the same deal
    splits: float  # share of pairs scoring exactly 1/2 (decided by the deal, or both drawn)

    @property
    def var_ratio(self) -> float:
        """Var(mirrored mean) / Var(raw mean) at equal games played; < 1 is the win."""
        return self.mirrored.var / self.raw.var if self.raw.var else float("nan")

    @property
    def var_per_game(self) -> tuple[float, float]:
        """(raw, mirrored) variance per game played, for `games_needed`. Both
        variances above are per pair, and a pair is two games."""
        return (2 * self.raw.var, 2 * self.mirrored.var)

    def __str__(self) -> str:
        return (
            f"seat0 {self.seat[0]}  seat1 {self.seat[1]}  mean raw {self.raw}  mirrored {self.mirrored}  "
            f"corr={self.corr:+.3f} splits={self.splits:.3f} var ratio={self.var_ratio:.3f}"
        )


def estimate_mirrored(pairs: Pairs) -> Mirrored:
    x, y = pairs.seat_scores(0), pairs.seat_scores(1)
    e0, e1 = estimate(x), estimate(y)
    ps = pairs.pair_scores()
    mean = (e0.mean + e1.mean) / 2
    # raw: mean of 2n independent games, Var = (var0 + var1) / (4n) = per-pair variance (var0 + var1) / 4
    raw = Estimate(pairs.n, mean, (e0.var + e1.var) / 4)
    mirrored = estimate(ps)
    assert math.isclose(mirrored.mean, mean, abs_tol=1e-12)
    return Mirrored(
        seat=(e0, e1),
        raw=raw,
        mirrored=mirrored,
        corr=correlation(x, y),
        splits=sum(1 for p in ps if p == 0.5) / pairs.n,
    )


@dataclass(frozen=True)
class Comparison:
    """Paired difference (first minus second row agent) against the same column, on the same deals."""

    seat: tuple[Estimate, Estimate]  # per-seat paired differences, over deals
    unpaired: tuple[Estimate, Estimate]  # same means, variance as if the two agents' games were independent
    mean: Estimate  # seat-averaged paired difference (mean of the two per-deal differences)
    corr: tuple[float, float]  # per seat, correlation of the two agents' scores on the same deal

    def var_ratio(self, seat: int) -> float:
        return self.seat[seat].var / self.unpaired[seat].var if self.unpaired[seat].var else float("nan")

    def __str__(self) -> str:
        return "  ".join(
            f"seat{s} {self.seat[s]} (unpaired ± {self.unpaired[s].half:.4f}, corr={self.corr[s]:+.3f}, ratio={self.var_ratio(s):.3f})"
            for s in (0, 1)
        ) + f"  mean {self.mean}"


def compare(a: Pairs, b: Pairs) -> Comparison:
    """Paired difference of two row agents that played the same seeds against
    the same column agent. Under common random numbers the column agent plays
    identically in both matchups until the histories diverge."""
    if a.seeds != b.seeds:
        raise ValueError("compare needs the same seeds in the same order")
    seat, unpaired, corr = [], [], []
    for s in (0, 1):
        xa, xb = a.seat_scores(s), b.seat_scores(s)
        d = estimate([p - q for p, q in zip(xa, xb)])
        ea, eb = estimate(xa), estimate(xb)
        seat.append(d)
        unpaired.append(Estimate(a.n, d.mean, ea.var + eb.var))
        corr.append(correlation(xa, xb))
    both = estimate([p - q for p, q in zip(a.pair_scores(), b.pair_scores())])
    return Comparison(tuple(seat), tuple(unpaired), both, tuple(corr))


# -- exact counterparts (toy only) -----------------------------------------------


@dataclass(frozen=True)
class ExactMirrored:
    """Exact moments of the first-named agent's score over worlds (deal x pile
    order), for both seats and for the pair. Given a world the two games are
    independent (the action rng streams differ by seat), so the covariance is
    the covariance of the per-world expected scores; for deterministic agents
    everything here is exact, for stochastic ones exact under that independence.
    """

    mean: tuple[float, float]  # per seat
    var: tuple[float, float]  # per seat, total (deal + pile + action randomness)
    cov: float  # between the two seats' scores on the same world
    splits: float  # P(pair score == 1/2)
    worlds: int

    @property
    def pair_mean(self) -> float:
        return (self.mean[0] + self.mean[1]) / 2

    @property
    def corr(self) -> float:
        return self.cov / math.sqrt(self.var[0] * self.var[1])

    @property
    def var_raw(self) -> float:
        """Variance of one pair's worth of the raw design (two independent games)."""
        return (self.var[0] + self.var[1]) / 4

    @property
    def var_mirrored(self) -> float:
        return (self.var[0] + self.var[1] + 2 * self.cov) / 4

    @property
    def var_ratio(self) -> float:
        return self.var_mirrored / self.var_raw

    def __str__(self) -> str:
        return (
            f"seat0 {self.mean[0]:.4f} (sd {math.sqrt(self.var[0]):.4f})  seat1 {self.mean[1]:.4f} (sd {math.sqrt(self.var[1]):.4f})  "
            f"mean {self.pair_mean:.4f}  corr={self.corr:+.3f} splits={self.splits:.3f} var ratio={self.var_ratio:.3f}"
        )


@dataclass(frozen=True)
class Bootstrap:
    """Percentile bootstrap CIs (resampling pairs, i.e. seeds) for the second moments
    of a mirrored estimate: the per-seat variances, the correlation between the two
    seats' scores, the split share and the variance ratio mirrored / raw."""

    level: float
    reps: int
    var: tuple[tuple[float, float], tuple[float, float]]
    corr: tuple[float, float]
    splits: tuple[float, float]
    var_ratio: tuple[float, float]

    def covers(self, exact: ExactMirrored) -> dict[str, bool]:
        """Whether each exact moment lies inside its CI."""
        inside = lambda ci, x: ci[0] <= x <= ci[1]  # noqa: E731
        return {
            "var0": inside(self.var[0], exact.var[0]),
            "var1": inside(self.var[1], exact.var[1]),
            "corr": inside(self.corr, exact.corr),
            "splits": inside(self.splits, exact.splits),
            "var_ratio": inside(self.var_ratio, exact.var_ratio),
        }


def bootstrap_moments(pairs: Pairs, reps: int = 4000, level: float = 0.998, seed: int = 0) -> Bootstrap:
    """Seeded percentile bootstrap over the pairs, with the definitions of
    `estimate_mirrored` (sample variances, pair score (s0 + s1) / 2, split = 1/2).

    The default level is 99.8 % per moment, a family-wise 99 % for the five
    moments of a matchup (Bonferroni): at 99 % per moment one of the five misses
    about one time in twenty by chance (step 8.8, the TOY split share on the dev
    seeds at z = +2.1 while 20,000 pooled pairs sit at z = +0.2). 4000 resamples
    leave a few draws in each 0.1 % tail."""
    x, y = np.array(pairs.seat_scores(0)), np.array(pairs.seat_scores(1))
    n = len(x)
    rng = np.random.default_rng(seed)
    stats = {name: [] for name in ("v0", "v1", "corr", "splits", "ratio")}
    for start in range(0, reps, 100):
        idx = rng.integers(0, n, size=(min(100, reps - start), n))
        xs, ys = x[idx], y[idx]
        dx, dy = xs - xs.mean(axis=1, keepdims=True), ys - ys.mean(axis=1, keepdims=True)
        vx, vy, cxy = (dx * dx).sum(axis=1) / (n - 1), (dy * dy).sum(axis=1) / (n - 1), (dx * dy).sum(axis=1) / (n - 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            stats["corr"].append(cxy / np.sqrt(vx * vy))
            stats["ratio"].append((vx + vy + 2 * cxy) / (vx + vy))
        stats["v0"].append(vx)
        stats["v1"].append(vy)
        stats["splits"].append((xs + ys == 1.0).mean(axis=1))
    q = ((1 - level) / 2, 1 - (1 - level) / 2)
    ci = {name: tuple(float(v) for v in np.nanquantile(np.concatenate(parts), q)) for name, parts in stats.items()}
    return Bootstrap(level, reps, (ci["v0"], ci["v1"]), ci["corr"], ci["splits"], ci["ratio"])
