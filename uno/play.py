"""Game runner and match statistics.

`play_game` records a full trace (state, observation, policy, action) per
decision, which is what the mirrored evaluator and AIVAT consume later. Action
sampling uses an rng seeded from (game seed, information state), so the same
agent makes the same choice in the same situation: common random numbers.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter
from dataclasses import dataclass, field

from uno.agents import Agent, Policy, sample
from uno.engine import State, legal_actions, new_game, returns, step
from uno.observe import Level, Observation, observe
from uno.rules import Ruleset


def stable_hash(obj: object) -> int:
    """Process-independent hash (str hashes are randomised per process)."""
    return int.from_bytes(hashlib.blake2b(repr(obj).encode(), digest_size=8).digest(), "little")


@dataclass(frozen=True, slots=True)
class Step:
    state: State
    obs: Observation
    probs: Policy
    action: int


@dataclass(frozen=True, slots=True)
class Trace:
    steps: tuple[Step, ...]
    final: State

    @property
    def actions(self) -> tuple[int, ...]:
        return tuple(s.action for s in self.steps)


def play_game(
    rs: Ruleset,
    seed: int,
    agents: tuple[Agent, Agent],
    levels: tuple[Level, Level] = (Level.L1, Level.L1),
) -> Trace:
    return play_from(new_game(rs, seed), seed, agents, levels)


def play_from(s: State, seed: int, agents: tuple[Agent, Agent], levels: tuple[Level, Level] = (Level.L1, Level.L1)) -> Trace:
    """Play `s` to the end. `seed` only seeds the action rng (common random numbers)."""
    steps: list[Step] = []
    while not s.terminal:
        p = s.player
        obs = observe(s, p, levels[p])
        legal = legal_actions(s)
        probs = agents[p].policy(obs, legal)
        rng = random.Random(stable_hash((seed, obs.info_state)))
        a = sample(probs, rng)
        steps.append(Step(s, obs, probs, a))
        s = step(s, a)
    return Trace(tuple(steps), s)


# -- statistics ---------------------------------------------------------------


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass(frozen=True)
class Result:
    """Outcome of `agents[0]` over a set of games."""

    n: int
    wins: int
    draws: int
    losses: int
    score: float  # mean of (r + 1) / 2, draws worth 1/2
    score_ci: tuple[float, float]  # normal approximation, 95%
    win_rate: float
    win_ci: tuple[float, float]  # Wilson, 95%
    mean_length: float
    ends: Counter = field(default_factory=Counter)  # end kind -> games
    invalid: int = 0  # seeds whose deal had no number card for the first discard

    def __str__(self) -> str:
        ends = " ".join(f"{k}={v}" for k, v in sorted(self.ends.items()))
        return (
            f"n={self.n} score={self.score:.3f} [{self.score_ci[0]:.3f}, {self.score_ci[1]:.3f}] "
            f"W/D/L={self.wins}/{self.draws}/{self.losses} len={self.mean_length:.1f} {ends}"
            + (f" invalid={self.invalid}" if self.invalid else "")
        )


def summarize(scores: list[float], lengths: list[int], ends: Counter | None = None, invalid: int = 0) -> Result:
    n = len(scores)
    wins = sum(1 for x in scores if x == 1.0)
    draws = sum(1 for x in scores if x == 0.5)
    mean = sum(scores) / n
    var = sum((x - mean) ** 2 for x in scores) / max(n - 1, 1)
    half = 1.96 * math.sqrt(var / n)
    return Result(
        n=n,
        wins=wins,
        draws=draws,
        losses=n - wins - draws,
        score=mean,
        score_ci=(mean - half, mean + half),
        win_rate=wins / n,
        win_ci=wilson(wins, n),
        mean_length=sum(lengths) / n,
        ends=ends or Counter(),
        invalid=invalid,
    )


def match(
    rs: Ruleset,
    agents: tuple[Agent, Agent],
    seeds,
    swap: bool = False,
    levels: tuple[Level, Level] = (Level.L1, Level.L1),
) -> Result:
    """Play every seed once. `swap` seats the first agent as player 1.

    The result is always from the first agent's point of view. Seeds whose deal
    is invalid (no number card for the first discard) are skipped and counted.
    """
    seat = 1 if swap else 0
    pair = (agents[1], agents[0]) if swap else agents
    lv = (levels[1], levels[0]) if swap else levels
    scores, lengths, ends, invalid = [], [], Counter(), 0
    for seed in seeds:
        try:
            s0 = new_game(rs, seed)
        except ValueError:
            invalid += 1
            continue
        t = play_from(s0, seed, pair, lv)
        scores.append((returns(t.final)[seat] + 1.0) / 2.0)
        lengths.append(t.final.turn)
        ends[t.final.end] += 1
    return summarize(scores, lengths, ends, invalid)
