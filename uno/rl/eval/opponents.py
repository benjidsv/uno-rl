"""The frozen training leagues, by name, with a hash for provenance.

The primary objective is the equal-weight score of the step-10C frozen four-agent
rulebot league, with balanced seats, one fixed opponent per episode and the
opponent's identity hidden from the actor (docs/rl/plan.md, starting-point section 3).
`uno.agents.rulebot.rulebot_league` stays the authority for its membership; this
module only names leagues so a config, a worker process and a checkpoint can refer
to the same one.

The toy league is `PlayFirstRandom` then `Heuristic`, the fixed-opponent order the
toy gates use (pipeline plan section 13). A league is training coverage, not
evidence that its members are optimal or individually unexploitable.
"""

from __future__ import annotations

from uno.agents.baseline import Agent, Heuristic, PlayFirstRandom
from uno.agents.rulebot import FROZEN_RULEBOT_LEAGUE, rulebot_league
from uno.game.rules import Ruleset

LEAGUES = ("rulebot", "toy", "toy_random", "toy_heuristic")


def league(rs: Ruleset, name: str = "rulebot") -> dict[str, Agent]:
    """The named league's agents, in display order. Built per process."""
    match name:
        case "rulebot":
            agents = rulebot_league(rs)
            if tuple(agents) != FROZEN_RULEBOT_LEAGUE:
                raise AssertionError("the frozen league changed membership")
            return agents
        case "toy":
            return {a.name: a for a in (PlayFirstRandom(rs), Heuristic(rs))}
        case "toy_random":
            return {a.name: a for a in (PlayFirstRandom(rs),)}
        case "toy_heuristic":
            return {a.name: a for a in (Heuristic(rs),)}
    raise KeyError(f"unknown league {name!r}; known: {', '.join(LEAGUES)}")


def league_hash(rs: Ruleset, name: str) -> str:
    """Stable hash of the league's membership, for checkpoints and run tags."""
    from uno.eval.play import stable_hash

    return f"{stable_hash((name, repr(rs), tuple(league(rs, name)))):016x}"


__all__ = ["LEAGUES", "league", "league_hash"]
