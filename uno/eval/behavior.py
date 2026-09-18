"""Behavioral fingerprints and common-state policy diversity for step 10C.

Rates use the probability mass an agent reported, not only the sampled action.
The visited positions are still on-policy, while the conditional choice estimate
has much less action-sampling noise and exactly describes stochastic agents.
Raw per-game counts use the realised trace.  A profiler consumes one trace and
then discards it; full-UNO experiments need not retain collections of States.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from uno.agents.rulebot import _denial
from uno.eval.belief import voids
from uno.eval.decisions import NODE_KINDS, node_kind
from uno.eval.play import Trace
from uno.game.engine import Phase, legal_actions, playable_table
from uno.game.observe import Observation, _voluntary
from uno.game.rules import Kind, Play, Ruleset


RATE_METRICS = (
    "voluntary_draw",
    "draw_with_wild",
    "keep_drawn",
    "keep_drawn_wild",
    "wild_hoard",
    "wild4_hoard",
    "draw_two_use",
    "block_switch_use",
    "sustainable_action",
    "void_target",
    "normalised_entropy",
    "deterministic",
)

COUNT_METRICS = (
    "decisions",
    "voluntary_draws",
    "forced_draws",
    "penalty_cards_received",
    "terminal_wilds",
)


@dataclass
class BehaviorStats:
    """Mergeable sufficient statistics for one agent's observed behavior."""

    numerators: Counter = field(default_factory=Counter)
    denominators: Counter = field(default_factory=Counter)
    totals: Counter = field(default_factory=Counter)
    games: int = 0

    def add_rate(self, name: str, value: float, weight: float = 1.0) -> None:
        self.numerators[name] += value
        self.denominators[name] += weight

    def merge(self, other: "BehaviorStats") -> "BehaviorStats":
        self.numerators.update(other.numerators)
        self.denominators.update(other.denominators)
        self.totals.update(other.totals)
        self.games += other.games
        return self

    def rate(self, name: str) -> float:
        den = self.denominators[name]
        return self.numerators[name] / den if den else float("nan")

    def per_game(self, name: str) -> float:
        return self.totals[name] / self.games if self.games else float("nan")

    def rates(self) -> dict[str, float]:
        return {name: self.rate(name) for name in RATE_METRICS}

    def counts(self) -> dict[str, float]:
        return {name: self.per_game(name) for name in COUNT_METRICS}


def _mass(probs: dict[int, float], actions) -> float:
    return sum(probs.get(a, 0.0) for a in actions)


def _play_groups(rs: Ruleset, legal: tuple[int, ...]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {k: [] for k in ("coloured", "wild", "wild4", "draw2", "block")}
    for a in legal:
        act = rs.actions[a]
        if not isinstance(act, Play):
            continue
        kind = rs.card_types[act.card].kind
        if not rs.card_types[act.card].is_wild:
            out["coloured"].append(a)
        if kind is Kind.WILD:
            out["wild"].append(a)
        elif kind is Kind.WILD_DRAW_FOUR:
            out["wild4"].append(a)
        elif kind is Kind.DRAW_TWO:
            out["draw2"].append(a)
        elif kind in (Kind.SKIP, Kind.REVERSE):
            out["block"].append(a)
    return out


def _sustainable(rs: Ruleset, obs: Observation, action: int) -> bool:
    play = rs.actions[action]
    assert isinstance(play, Play)
    ct = rs.card_types[play.card]
    colour = play.color if ct.is_wild else ct.color
    remaining = list(obs.hand)
    remaining.remove(play.card)
    playable = playable_table(rs)[play.card][colour]
    return any(c in playable for c in remaining)


def observe_trace(rs: Ruleset, trace: Trace, seat: int, out: BehaviorStats | None = None) -> BehaviorStats:
    """Accumulate one game for ``seat`` from the policy probabilities and realised actions."""

    if out is None:
        out = BehaviorStats()
    out.games += 1
    for st in trace.steps:
        if st.state.player != seat:
            continue
        obs, probs = st.obs, st.probs
        legal = legal_actions(st.state)
        groups = _play_groups(rs, legal)
        play_actions = [a for a in legal if isinstance(rs.actions[a], Play)]
        out.totals["decisions"] += 1

        if len(legal) > 1:
            entropy = -sum(p * math.log(p) for p in probs.values() if p > 0)
            out.add_rate("normalised_entropy", entropy / math.log(len(legal)))
            out.add_rate("deterministic", float(max(probs.values()) >= 1.0 - 1e-12))

        if obs.phase is Phase.TURN_START and play_actions and rs.draw_action in legal:
            out.add_rate("voluntary_draw", probs.get(rs.draw_action, 0.0))
            wild_in_hand = bool(groups["wild"] or groups["wild4"])
            if wild_in_hand:
                out.add_rate("draw_with_wild", probs.get(rs.draw_action, 0.0))
            if st.action == rs.draw_action:
                out.totals["voluntary_draws"] += 1
        elif obs.phase is Phase.TURN_START and not play_actions and st.action == rs.draw_action:
            out.totals["forced_draws"] += 1

        if obs.phase is Phase.AFTER_DRAW and rs.pass_action in legal:
            keep = probs.get(rs.pass_action, 0.0)
            out.add_rate("keep_drawn", keep)
            if rs.card_types[obs.drawn].is_wild:
                out.add_rate("keep_drawn_wild", keep)

        if groups["coloured"]:
            if groups["wild"]:
                out.add_rate("wild_hoard", 1.0 - _mass(probs, groups["wild"]))
            if groups["wild4"]:
                out.add_rate("wild4_hoard", 1.0 - _mass(probs, groups["wild4"]))

        other_plays = set(play_actions)
        for key, metric in (("draw2", "draw_two_use"), ("block", "block_switch_use")):
            acts = groups[key]
            if acts and other_plays - set(acts):
                out.add_rate(metric, _mass(probs, acts))

        tempo = groups["draw2"] + groups["block"] + groups["wild4"]
        tempo_mass = _mass(probs, tempo)
        if tempo_mass:
            good = sum(probs.get(a, 0.0) for a in tempo if _sustainable(rs, obs, a))
            out.add_rate("sustainable_action", good, tempo_mass)

        evidence = voids(rs, obs.public, seat ^ 1)
        if evidence and len(play_actions) > 1:
            scores = {}
            for a in play_actions:
                play = rs.actions[a]
                ct = rs.card_types[play.card]
                colour = play.color if ct.is_wild else ct.color
                scores[a] = _denial(rs, play.card, colour, evidence)
            best = max(scores.values())
            if best > 0 and any(x < best for x in scores.values()):
                out.add_rate("void_target", sum(probs.get(a, 0.0) for a, x in scores.items() if x == best))

    # Penalty draws are public events rather than decisions by their recipient.
    for i, ev in enumerate(trace.final.public):
        if ev[0] == "draw" and ev[1] == seat and not _voluntary(rs, trace.final.public, i):
            out.totals["penalty_cards_received"] += ev[2]
    out.totals["terminal_wilds"] += sum(rs.card_types[c].is_wild for c in trace.final.hands[seat])
    return out


Probe = tuple[str, Observation, tuple[int, ...]]


def collect_probes(rs: Ruleset, trace: Trace, buckets: dict[str, list[Probe]], cap: int) -> None:
    """Collect a bounded, node-kind-balanced bank of shared observations."""

    for st in trace.steps:
        legal = legal_actions(st.state)
        if len(legal) < 2:
            continue
        kind = node_kind(rs, st.obs, legal)
        if len(buckets.setdefault(kind, [])) < cap:
            buckets[kind].append((kind, st.obs, legal))


def policy_distances(agents: dict[str, object], probes: list[Probe]) -> tuple[list[str], np.ndarray]:
    """Mean total-variation distance, weighting every represented node kind equally."""

    names = list(agents)
    per_kind: dict[str, list[np.ndarray]] = {k: [] for k in NODE_KINDS}
    for kind, obs, legal in probes:
        rows = []
        for agent in agents.values():
            p = agent.policy(obs, legal)
            rows.append(np.array([p.get(a, 0.0) for a in legal], dtype=np.float64))
        per_kind[kind].append(np.stack(rows))
    mats = []
    for rows in per_kind.values():
        if not rows:
            continue
        acc = np.zeros((len(names), len(names)))
        for x in rows:
            acc += 0.5 * np.abs(x[:, None, :] - x[None, :, :]).sum(axis=2)
        mats.append(acc / len(rows))
    return names, np.mean(mats, axis=0) if mats else np.zeros((len(names), len(names)))
