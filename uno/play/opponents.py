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

    def copy(self) -> Player: ...


class AgentPlayer:
    """A stateless `Agent` on its L1 observation."""

    def __init__(self, agent: Agent, seat: int) -> None:
        self.agent, self.seat = agent, seat

    def policy(self, state: State) -> dict[int, float]:
        return self.agent.policy(observe(state, self.seat, Level.L1), legal_actions(state))

    def copy(self) -> AgentPlayer:
        return self


@dataclass
class Decision:
    """What one neural decision read: enough to re-run the network on it (`uno.play.inspect`)
    without touching the live memory. `memory` is the recurrent state *before* the decision."""

    features: object  # uno.rl.env.features.Features, one row
    mask: object  # (num_actions,) bool
    events: object | None
    memory: object | None  # (n_tokens, memory) tensor, or None
    probs: dict[int, float]


class NeuralPlayer:
    """An `ActorPolicy` in one game, keeping its evidence and recurrent memory like the collector.

    `policy` advances the memory: call it exactly once per decision of `seat`, in order.
    `copy` is an independent continuation (analysis, rollouts) that never touches this one.
    """

    def __init__(self, actor, seat: int) -> None:
        from uno.rl.env.evidence import Evidence

        self.actor, self.seat = actor, seat
        self.memory = actor.new_memory(seat)
        self.evidence = Evidence(actor.rs, seat ^ 1) if actor.spec.has_evidence else None
        self.seen = 0
        self.last: Decision | None = None

    def copy(self) -> NeuralPlayer:
        import copy

        other = object.__new__(NeuralPlayer)
        other.actor, other.seat, other.seen, other.last = self.actor, self.seat, self.seen, None
        other.evidence = None if self.evidence is None else self.evidence.copy()
        other.memory = copy.deepcopy(self.memory)  # EventCounter tape + state tensor
        return other

    def policy(self, state: State) -> dict[int, float]:
        return batch_policy([self], [state], keep=True)[0]


def batch_policy(players: list[NeuralPlayer], states: list[State], keep: bool = False) -> list[dict[int, float]]:
    """One decision for each `(player, state)`, all players sharing one actor: batched
    features and one forward. Advances every player's evidence and memory once."""
    from uno.rl.algo.policy import played_probabilities
    from uno.rl.env.features import extract, legal_masks

    actor = players[0].actor
    obs1 = [observe(s, p.seat, Level.L1) for p, s in zip(players, states)]
    obs = [observe(s, p.seat, Level.L2) for p, s in zip(players, states)] if actor.spec.has_opponent else obs1
    evidences = None
    if actor.spec.has_evidence:
        for p, s in zip(players, states):
            p.seen = p.evidence.update(s.public, p.seen)
        evidences = [p.evidence for p in players]
    feats = extract(actor.spec, obs, evidences)
    masks = legal_masks(actor.rs, obs1)
    memories = [p.memory for p in players] if actor.recurrent else None
    before = [None if m.state is None else m.state.clone() for m in memories] if (keep and memories) else None
    events = actor.observe_events(memories, obs1) if actor.recurrent else None
    probs = played_probabilities(actor.log_probs(feats, masks, None, events, memories))
    out = []
    for i, s in enumerate(states):
        d = {int(a): float(probs[i, a]) for a in legal_actions(s)}
        out.append(d)
        if keep:
            players[i].last = Decision(feats[i : i + 1], masks[i], None if events is None else events[i : i + 1],
                                       None if before is None else before[i], d)
    return out


@dataclass
class Opponent:
    id: str
    label: str
    kind: str  # "rulebot" | "checkpoint"
    make: Callable[[int], Player]
    detail: str = ""
    actor: object | None = None  # the ActorPolicy of a checkpoint opponent


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


def load_policy(rs: Ruleset, path: str | Path):
    """`(ActorPolicy, payload, detail)` of an actor export, validated against `rs` by `load_actor`."""
    from uno.rl.core.checkpoint import load_actor

    path = Path(path).expanduser()
    agent, payload = load_actor(path, rs)
    kind = payload.get("model", {}).get("kind") or ("relational" if "width" in payload else "?")
    detail = (
        f"arm {payload['arm']}, {kind}, {payload.get('trained_decisions', 0):,} decisions, "
        f"T={payload['temperature']:g}, {path}"
    )
    return agent.actor, payload, detail


def checkpoint(rs: Ruleset, path: str | Path, label: str | None = None) -> Opponent:
    """An actor export (`*_actor.pt`) as an opponent."""
    path = Path(path).expanduser()
    actor, payload, detail = load_policy(rs, path)
    return Opponent(f"ckpt:{path.resolve()}", label or f"{payload['name']}", "checkpoint",
                    lambda seat: NeuralPlayer(actor, seat), detail, actor)
