"""A symmetric version of any agent (docs/spec.md section 18).

The quotient tree evaluates a policy exactly only if the policy commutes with
the color / rank group. A learned agent does not, so it is wrapped: its
observation is canonicalised, it is asked on the canonical observation and
legal actions, its output is averaged within the action orbits of the canonical
key's stabilizer, and the result is mapped back to the real labels.

The averaging is what makes the wrapper a policy at all: when the stabilizer is
not trivial, several group elements reach the canonical key, `canonical` returns
one of them, and without the averaging the action played would depend on which.
"""

from __future__ import annotations

from collections import defaultdict

from uno.agents.baseline import Agent, Policy
from uno.game.keys import obs_key
from uno.game.observe import Observation, observation_from_key
from uno.game.rules import Ruleset
from uno.game.symmetry import Symmetry


class Symmetrized:
    """`agent` made symmetric: canonical observation in, stabilizer-averaged policy
    out, mapped back with the inverse element. On an agent that is symmetric
    already (Random, the Heuristic) it changes nothing."""

    def __init__(self, agent: Agent, rs: Ruleset, name: str | None = None) -> None:
        self.agent = agent
        self.rs = rs
        self.sym = Symmetry(rs)
        self.name = name or f"sym_{agent.name}"

    def policy(self, obs: Observation, legal: tuple[int, ...]) -> Policy:
        sym = self.sym
        key, g = sym.canonical(obs.level, obs_key(self.rs, obs))
        canonical_obs = observation_from_key(self.rs, obs.level, key, seat=obs.seat)
        canonical_legal = tuple(sorted(int(sym.actions[g, a]) for a in legal))
        probs = self.agent.policy(canonical_obs, canonical_legal)
        stabilizer = sym.stabilizer(obs.level, key)
        if len(stabilizer) > 1:
            averaged: dict[int, float] = defaultdict(float)
            for h in stabilizer:
                for a, p in probs.items():
                    averaged[int(sym.actions[h, a])] += p / len(stabilizer)
            probs = dict(averaged)
        return sym.map_policy(int(sym.inverse[g]), probs)
