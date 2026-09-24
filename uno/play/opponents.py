"""Opponents for the play demo: every rulebot, plus actor exports loaded at runtime.

Agents return distributions (`uno.agents.baseline.Agent`); the demo samples them. A
checkpoint is wrapped per game because a recurrent actor carries memory that must
advance exactly once per decision of its own seat, in order.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from uno.agents.baseline import Agent, Heuristic, KeepsDrawn, PlayFirstRandom, UniformLegal
from uno.agents.rulebot import DistilledRuleBot, MemoryHeuristic, TransferRuleBot, UnpredictableRuleBot
from uno.game.engine import State, legal_actions
from uno.game.observe import Level, observe
from uno.game.rules import Ruleset


class Player(Protocol):
    """One seat of one game: the distribution it plays at the current decision."""

    def policy(self, state: State) -> dict[int, float]: ...


class AgentPlayer:
    """A stateless `Agent` on its L1 observation."""

    def __init__(self, agent: Agent, seat: int) -> None:
        self.agent, self.seat = agent, seat

    def policy(self, state: State) -> dict[int, float]:
        return self.agent.policy(observe(state, self.seat, Level.L1), legal_actions(state))


class NeuralPlayer:
    """An `ActorPolicy` in one game, keeping its evidence and recurrent memory like the collector."""

    def __init__(self, actor, seat: int) -> None:
        from uno.rl.env.evidence import Evidence

        self.actor, self.seat = actor, seat
        self.memory = actor.new_memory(seat)
        self.evidence = Evidence(actor.rs, seat ^ 1) if actor.spec.has_evidence else None
        self.seen = 0

    def policy(self, state: State) -> dict[int, float]:
        from uno.rl.algo.policy import played_probabilities
        from uno.rl.env.features import extract, legal_masks

        actor = self.actor
        obs1 = observe(state, self.seat, Level.L1)
        obs = observe(state, self.seat, Level.L2) if actor.spec.has_opponent else obs1
        evidences = None
        if self.evidence is not None:
            self.seen = self.evidence.update(state.public, self.seen)
            evidences = [self.evidence]
        feats = extract(actor.spec, [obs], evidences)
        masks = legal_masks(actor.rs, [obs1])
        memories = [self.memory] if actor.recurrent else None
        events = actor.observe_events(memories, [obs1]) if actor.recurrent else None
        p = played_probabilities(actor.log_probs(feats, masks, None, events, memories))[0]
        return {int(a): float(p[a]) for a in legal_actions(state)}


@dataclass
class Opponent:
    id: str
    label: str
    kind: str  # "rulebot" | "checkpoint"
    make: Callable[[int], Player]
    detail: str = ""


def rulebots(rs: Ruleset) -> list[Opponent]:
    agents: list[tuple[str, Agent]] = [
        ("Uniform legal", UniformLegal(rs)),
        ("Random (play-first)", PlayFirstRandom(rs)),
        ("Keeps drawn 50%", KeepsDrawn(rs)),
        ("Heuristic", Heuristic(rs)),
        ("Heuristic + memory", MemoryHeuristic(rs)),
        ("Rulebot transfer", TransferRuleBot(rs)),
        ("Rulebot distilled", DistilledRuleBot(rs)),
        ("Rulebot unpredictable", UnpredictableRuleBot(rs)),
    ]
    return [
        Opponent(agent.name, label, "rulebot", lambda seat, agent=agent: AgentPlayer(agent, seat), agent.name)
        for label, agent in agents
    ]


def checkpoint(rs: Ruleset, path: str | Path, label: str | None = None) -> Opponent:
    """An actor export (`*_actor.pt`), validated against `rs` by `load_actor`."""
    from uno.rl.core.checkpoint import load_actor

    path = Path(path).expanduser()
    agent, payload = load_actor(path, rs)
    actor = agent.actor
    kind = payload.get("model", {}).get("kind") or ("relational" if "width" in payload else "?")
    detail = (
        f"arm {payload['arm']}, {kind}, {payload.get('trained_decisions', 0):,} decisions, "
        f"T={payload['temperature']:g}, {path}"
    )
    return Opponent(f"ckpt:{path.resolve()}", label or f"{payload['name']}", "checkpoint",
                    lambda seat: NeuralPlayer(actor, seat), detail)
