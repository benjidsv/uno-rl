"""Game runner and match statistics.

`play_game` records a full trace (state, observation, policy, action) per
decision, which is what the mirrored evaluator and AIVAT consume later. Action
sampling uses an rng seeded from (game seed, agent name, information state),
so the same agent makes the same choice in the same situation: common random
numbers. The agent name is in the key so that two *different* stochastic
agents never share a stream: in a mirrored pair the row and the column agent
sit at the same seat in the two games and would otherwise draw the same
random numbers in the same information states, coupling the two games beyond
the shared deal (the exact mirrored moments assume they are independent given
the world). Same agent, same situation, same choice still holds across
matchups, which is what `uno.eval.estimate.compare` needs.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter
from dataclasses import dataclass, field

from uno.agents.baseline import Agent, Policy, sample
from uno.game.engine import State, legal_actions, new_game, returns, step
from uno.game.observe import Level, Observation, observe
from uno.game.rules import Ruleset
from uno.progress import bar

# Seed ranges. Everything is built, tested and (later) tuned on DEV_SEEDS; the
# ladder's reported numbers come from EVAL_SEEDS, which nothing is ever chosen
# on. A fixed seed range carries its own deal luck (seeds 0-4000 on TOY favour
# seat 0 by ~1.5 points): harmless while every number is compared on the same
# seeds, a bias as soon as an agent, a value function or a hyperparameter is
# picked because it scored well on them.
DEV_SEEDS = range(4000)
EVAL_SEEDS = range(1_000_000, 1_004_000)
# Step 9.5b: games a value function (the AIVAT critic, `uno.eval.critic`) is fitted on. A
# dev-class range, larger than DEV_SEEDS: nothing is reported on it, and it is disjoint from
# EVAL_SEEDS and from the fresh ranges earlier checks drew (3,000,000+, 5,000,000+).
FIT_SEEDS = range(10_000_000, 10_100_000)


def stable_hash(obj: object) -> int:
    """Process-independent hash (str hashes are randomised per process)."""
    return int.from_bytes(hashlib.blake2b(repr(obj).encode(), digest_size=8).digest(), "little")


def stream_name(agent: Agent) -> str:
    """The name that keys an agent's action rng: its `stream` attribute when set, its
    `name` otherwise. Two agents on one stream draw the same random numbers in the
    same information state. `uno.eval.estimate.compare` wants that for two variants
    of one policy (two checkpoints, step 9); the two agents of a mirrored pair must
    never have it (`play_pairs` raises)."""
    return getattr(agent, "stream", None) or getattr(agent, "name", "?")


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
        rng = random.Random(stable_hash((seed, stream_name(agents[p]), obs.info_state)))
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
    for seed in bar(seeds, desc=f"games {getattr(agents[0], 'name', '?')} vs {getattr(agents[1], 'name', '?')}", unit="games"):
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
