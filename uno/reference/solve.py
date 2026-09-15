"""Exact solvers: policy evaluation and best response at L1, L2 and L3.

See docs/spec.md section 14 and docs/plan.md ("The oracle levels").

A best response is the responder's best pure strategy against a *fixed*
opponent (an `Agent` seen at L1), given the responder's information level:

- L1: information set = `info_state` (seat, initial hand, public history, own
  draws).
- L2: L1 plus the opponent's initial hand and draws in order, i.e. the
  opponent's hand as seen at every one of the responder's decisions. Keying on
  the *current* hand alone would merge histories the responder could tell apart
  earlier (which card the opponent drew), an imperfect-recall partition on
  which the local maximisation below is not guaranteed optimal.
- L3: the whole state, pile order included. Every information set is one
  history, so L3 is plain expectimax over a deterministic game.

L1 and L2 walk the lazy-chance tree (`uno.game.chance`): a draw is a chance node over
the distinct types left in the pile. L3 knows the pile order, so its roots are
*worlds* (every deal x every order of the unknown pile) and below a world the
tree is the bare engine, with no chance at all.

Algorithm (two passes over the tree):

1. forward: depth-first from the roots, accumulating for every history its
   counterfactual reach (deal x chance x opponent probabilities, never the
   responder's own) and grouping the responder's decision histories by
   information set;
2. backward: memoised values. At a responder history the action is the one
   that maximises the reach-weighted sum of continuation values over the whole
   information set. This is a best response only because every level has
   perfect recall (all histories of a set share the responder's past sets, so
   the responder's own reach is a common factor); the recursion is well founded
   because the public history grows strictly between two responder decisions.

Values are the responder's expected return in [-1, 1]; `score` is (v + 1) / 2,
i.e. P(win) with draws worth 1/2, comparable to `uno.eval.play.match`.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass

from uno.agents.baseline import Agent
from uno.game.chance import Expand, Outcome, children, engine_children, initial_deals, roots_and_expand, worlds
from uno.game.engine import State, new_game, returns
from uno.game.keys import KEYS, Key
from uno.game.observe import Level, observe
from uno.game.rules import Ruleset
from uno.progress import bar

# -- policy evaluation --------------------------------------------------------


def outcome_distribution(
    rs: Ruleset,
    agents: tuple[Agent, Agent],
    levels=(Level.L1, Level.L1),
    roots: list[Outcome] | None = None,
    expand: Expand | None = None,
) -> dict[float, float]:
    """Exact distribution of seat 0's return under two fixed agents: {+1: p, 0: p, -1: p}
    (seat 1's return is the negative). The moments of the score follow from it,
    which is what the exact variance of an estimator needs (`uno.eval.estimate`).

    `roots` defaults to the whole game (all deals) and `expand` to the tree
    those roots need (lazy chance tree, or the bare engine when someone plays
    at L3). Pass `roots=[(state, 1.0)]` and `expand=engine_children` for one
    real position: the pile order is then fixed and the distribution is over
    the agents' randomness only.
    """
    if expand is None:
        expand = engine_children if Level.L3 in levels else children
    if roots is None:  # only built when needed: the L3 worlds are expensive
        roots = worlds(rs) if expand is engine_children else initial_deals(rs)
    dist: dict[float, float] = defaultdict(float)
    stack = list(roots)
    progress, n = None, 0  # a bar only for walks of 100k histories and more: this runs once per seed in `scores_on_seeds`
    while stack:
        s, reach = stack.pop()
        n += 1
        if n % 100_000 == 0:
            if progress is None:  # not `progress or bar(...)`: a tqdm counter without a total has no truth value
                progress = bar(desc="exact walk", unit="histories")
            progress.update(100_000)
        if s.terminal:
            dist[returns(s)[0]] += reach
            continue
        p = s.player
        ch = expand(s)
        probs = agents[p].policy(observe(s, p, levels[p]), tuple(ch))
        for a, q in probs.items():
            for c, pc in ch[a]:
                stack.append((c, reach * q * pc))
    if progress is not None:
        progress.update(n - progress.n)
        progress.close()
    return dict(dist)


def evaluate(
    rs: Ruleset,
    agents: tuple[Agent, Agent],
    levels=(Level.L1, Level.L1),
    roots: list[Outcome] | None = None,
    expand: Expand | None = None,
) -> tuple[float, float]:
    """Exact expected returns of two fixed agents (seat 0, seat 1). Same
    arguments as `outcome_distribution`."""
    v0 = sum(r * p for r, p in outcome_distribution(rs, agents, levels, roots, expand).items())
    return (v0, -v0)


Z_MAX = 3.0  # |z| above this fails an on-seeds check: 0.27 % false alarms per check


@dataclass(frozen=True)
class OnSeeds:
    """Exact per-seat scores (draws worth 1/2) of two agents over the games of a seed
    set, each played on its real deal and pile order (`scores_on_seeds`).

    `mean[seat]` is exactly what a simulation on those seeds estimates: the deal
    and pile luck of the seed set is in it. `cond_var[seat]` is the mean over the
    seeds of Var(score | seed), the action noise that simulation adds, so
    `(sim - mean) / sqrt(cond_var / n)` is a z-score with the right scale. The CI
    of the simulated score is the wrong scale for this comparison: it also counts
    the deal and pile variance, which the seeds fix. Two deterministic agents have
    no action noise, and the simulation must then match to the digit."""

    n: int
    mean: tuple[float, float]
    cond_var: tuple[float, float]
    deal_var: tuple[float, float] = (math.nan, math.nan)  # variance over the seeds of the per-seed exact score

    def z_global(self, exact: float, seat: int) -> float:
        """z of the mean over these seeds against the exact score of the whole game,
        on the deal and pile variance (step 9). The on-seeds values start from the
        real deals, the whole-game value from the chance model's deal probabilities:
        a wrong chance model moves one and not the other, which no simulation
        check against `mean` can see."""
        return (self.mean[seat] - exact) / math.sqrt(self.deal_var[seat] / self.n)

    def z(self, sim: float, seat: int) -> float:
        diff = sim - self.mean[seat]
        se = math.sqrt(self.cond_var[seat] / self.n)
        if se < 1e-12:
            return 0.0 if abs(diff) <= 1e-9 else math.copysign(math.inf, diff)
        return diff / se

    def ok(self, sim: float, seat: int, z_max: float = Z_MAX) -> bool:
        return abs(self.z(sim, seat)) <= z_max


def scores_on_seeds(rs: Ruleset, agents: tuple[Agent, Agent], seeds, levels=(Level.L1, Level.L1)) -> OnSeeds:
    """Exact score moments over the games of `seeds` (invalid deals skipped), each
    played on its real deal *and* pile order: the expectation over the agents'
    randomness only, with the deal and pile luck of that seed set left in, and
    the variance of that randomness. Works at every level."""
    mean = [0.0, 0.0]
    var = [0.0, 0.0]
    sq = [0.0, 0.0]
    n = 0
    for seed in bar(seeds, desc="exact on seeds", unit="seeds"):
        try:
            s0 = new_game(rs, seed)
        except ValueError:
            continue
        dist = outcome_distribution(rs, agents, levels, roots=[(s0, 1.0)], expand=engine_children)
        for seat, sign in ((0, 1.0), (1, -1.0)):
            e = sum(p * (sign * r + 1.0) / 2.0 for r, p in dist.items())
            e2 = sum(p * ((sign * r + 1.0) / 2.0) ** 2 for r, p in dist.items())
            mean[seat] += e
            var[seat] += max(e2 - e * e, 0.0)
            sq[seat] += e * e
        n += 1
    deal_var = tuple(max(sq[s] - mean[s] ** 2 / n, 0.0) / max(n - 1, 1) for s in (0, 1))
    return OnSeeds(n, (mean[0] / n, mean[1] / n), (var[0] / n, var[1] / n), deal_var)


def evaluate_on_seeds(rs: Ruleset, agents: tuple[Agent, Agent], seeds, levels=(Level.L1, Level.L1)) -> tuple[float, float]:
    """Mean exact returns over the games of `seeds`, each on its real deal and pile
    order (the returns of `scores_on_seeds`)."""
    on = scores_on_seeds(rs, agents, seeds, levels)
    return (2.0 * on.mean[0] - 1.0, 2.0 * on.mean[1] - 1.0)


# -- best response ------------------------------------------------------------


@dataclass
class BestResponse:
    seat: int
    level: Level
    opponent: str
    value: float  # responder's expected return in [-1, 1]
    table: dict[Key, int]  # information set -> action (only sets reachable under the opponent)
    histories: int
    infosets: int
    seconds: float

    @property
    def score(self) -> float:
        return (self.value + 1.0) / 2.0

    def __str__(self) -> str:
        return (
            f"BR {self.level.name} seat {self.seat} vs {self.opponent}: score={self.score:.4f} "
            f"(histories={self.histories}, infosets={self.infosets}, {self.seconds:.1f}s)"
        )


def best_response(rs: Ruleset, opponent: Agent, seat: int, level: Level) -> BestResponse:
    t0 = time.perf_counter()
    key = KEYS[level]
    roots, expand = roots_and_expand(rs, (level,))
    opp = seat ^ 1

    # pass 1: reach probabilities and information sets. For a responder history
    # `kids` holds {action: outcomes}; for an opponent history a flat list of
    # (child, opponent prob x chance prob).
    kids: dict[State, object] = {}
    members: dict[Key, list[tuple[State, float]]] = defaultdict(list)
    histories = 0
    stack = list(roots)
    progress = bar(desc=f"BR {level.name} seat {seat} walk", unit="histories")
    while stack:
        s, reach = stack.pop()
        histories += 1
        if histories % 100_000 == 0:
            progress.update(100_000)
        if s.terminal:
            continue
        if s in kids:
            raise RuntimeError("the game tree is not a tree: a history was reached twice")
        ch = expand(s)
        if s.player == seat:
            kids[s] = ch
            group = members[key(s, seat)]
            if group and tuple(kids[group[0][0]]) != tuple(ch):
                raise RuntimeError(f"{level.name} information set with differing legal actions: {key(s, seat)!r}")
            group.append((s, reach))
            for outs in ch.values():
                for c, p in outs:
                    stack.append((c, reach * p))
        else:
            probs = opponent.policy(observe(s, opp, Level.L1), tuple(ch))
            flat = [(c, q * p) for a, q in probs.items() for c, p in ch[a]]
            kids[s] = flat
            for c, w in flat:
                stack.append((c, reach * w))

    progress.update(histories - progress.n)
    progress.close()

    # pass 2: values, choosing the best action per information set on demand
    memo: dict[State, float] = {}
    table: dict[Key, int] = {}

    def value(s: State) -> float:
        v = memo.get(s)
        if v is not None:
            return v
        if s.terminal:
            v = returns(s)[seat]
        elif s.player == seat:
            v = sum(p * value(c) for c, p in kids[s][choose(key(s, seat))])
        else:
            v = sum(w * value(c) for c, w in kids[s])
        memo[s] = v
        return v

    def choose(k: Key) -> int:
        a = table.get(k)
        if a is not None:
            return a
        group = members[k]
        best, best_v = -1, float("-inf")
        for a in kids[group[0][0]]:  # legal actions are the same across an information set
            va = sum(r * sum(p * value(c) for c, p in kids[h][a]) for h, r in group)
            if va > best_v + 1e-12:  # ties: lowest action id
                best, best_v = a, va
        table[k] = best
        return best

    total = sum(p * value(s) for s, p in roots)
    return BestResponse(
        seat=seat,
        level=level,
        opponent=getattr(opponent, "name", type(opponent).__name__),
        value=total,
        table=table,
        histories=histories,
        infosets=len(members),
        seconds=time.perf_counter() - t0,
    )
