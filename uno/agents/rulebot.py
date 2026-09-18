"""Cheap, symmetric rulebots for step 10C.

The bots are functions of the current position and public history only.  Their
"memory" is rebuilt from public draw events with the same :func:`voids`
contract used by the PIMC belief; no mutable episode state is hidden in the
agent.  This keeps exact tabulation, multiprocessing, and the response attack
honest.

Four policies are deliberately distinct:

``MemoryHeuristic``
    The old Heuristic with one change: inferred opponent voids help choose the
    resulting board.  This is the causal memory ablation.
``TransferRuleBot``
    A small tactical score intended to be frozen from exact-toy work.
``DistilledRuleBot``
    The same interpretable feature family with a more full-UNO-oriented fixed
    configuration.  Its weights are a DEV candidate until step 10C's fitting
    and held-out evaluation are complete.
``UnpredictableRuleBot``
    A stochastic DistilledRuleBot that sometimes draws or keeps a playable
    card and randomises between near-equal plays.  It returns those
    probabilities exactly; sampling remains in ``uno.eval.play``.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from uno.agents.baseline import Policy, uniform
from uno.eval.belief import Void, voids
from uno.game.engine import Phase, playable_table
from uno.game.observe import Observation
from uno.game.rules import Kind, Play, Ruleset


def _plays(rs: Ruleset, legal: tuple[int, ...]) -> list[tuple[int, Play]]:
    return [(a, act) for a in legal if isinstance((act := rs.actions[a]), Play)]


def _colours(rs: Ruleset, hand: tuple[int, ...]) -> Counter:
    return Counter(rs.card_types[c].color for c in hand if not rs.card_types[c].is_wild)


def _denial(rs: Ruleset, card: int, colour: int, evidence: list[Void]) -> float:
    """How much a resulting board reuses a live void.

    A void concerns the complete playable set of an old ``(top, colour)``
    board, not merely its colour.  The overlap is weighted by physical card
    multiplicity, normalised by the candidate board's playable mass, and
    weakened by cards the opponent drew later.  This is a cheap feature, not a
    posterior probability or a new hard constraint.
    """

    if not evidence:
        return 0.0
    table = playable_table(rs)
    candidate = table[card][colour]
    denom = sum(rs.deck_counts[c] for c in candidate)
    if not denom:
        return 0.0
    return max(
        sum(rs.deck_counts[c] for c in candidate & table[v.top][v.colour]) / denom / (1.0 + v.since)
        for v in evidence
    )


class MemoryHeuristic:
    """The Heuristic plus public-history void evidence, and no tactical rewrite.

    With no live opponent void this reproduces :class:`Heuristic` exactly.
    When evidence exists, denial breaks ties between equally supported colours.
    It cannot overrule the existing majority-colour rule; the first DEV profile
    showed that stronger insertion lost 4-5 score points.
    """

    name = "heuristic_memory"

    def __init__(self, rs: Ruleset, low_threshold: int = 2) -> None:
        self.rs, self.low_threshold = rs, low_threshold

    def policy(self, obs: Observation, legal: tuple[int, ...]) -> Policy:
        rs, types = self.rs, self.rs.card_types
        plays = _plays(rs, legal)
        if not plays:
            return {rs.draw_action: 1.0}
        evidence = voids(rs, obs.public, obs.seat ^ 1)
        colours = _colours(rs, obs.hand)
        opp_low = obs.opp_hand_size <= self.low_threshold
        coloured = [(a, p) for a, p in plays if not types[p.card].is_wild]
        if coloured:
            def key(ap: tuple[int, Play]) -> tuple:
                ct = types[ap[1].card]
                return (
                    opp_low and ct.kind is not Kind.NUMBER,
                    colours[ct.color],
                    _denial(rs, ap[1].card, ct.color, evidence),
                    ct.color == obs.current_color,
                    ct.kind is Kind.NUMBER,
                )

            best = max(key(ap) for ap in coloured)
            return uniform(a for a, p in coloured if key((a, p)) == best)

        def key(ap: tuple[int, Play]) -> tuple:
            ct = types[ap[1].card]
            colour = ap[1].color
            return (
                colours[colour],
                _denial(rs, ap[1].card, colour, evidence),
                colour == obs.current_color,
                opp_low == (ct.kind is Kind.WILD_DRAW_FOUR),
            )

        best = max(key(ap) for ap in plays)
        return uniform(a for a, p in plays if key((a, p)) == best)


@dataclass(frozen=True, slots=True)
class RuleWeights:
    """Interpretable coefficients for :class:`TacticalRuleBot`."""

    number: float = 0.0
    block_switch: float = 1.0
    draw_two: float = 1.5
    wild: float = -1.0
    wild_draw_four: float = 2.0
    colour_support: float = 0.6
    same_colour: float = 0.1
    singleton: float = 0.2
    pivot: float = 0.4
    followup: float = 0.7
    stranded_action: float = -1.5
    penalty_card: float = 0.25
    emergency: float = 1.0
    denial: float = 1.5
    wild_hold: float = -2.0


# A bounded coordinate sweep selected this preset by the unweighted exact mean
# against Random and Heuristic, at both seats, on TOY and TOY2_SKIP.  TOY_FULL
# was redundant for these deterministic play-first candidates.  The sweep and
# objective are recorded by the step-10C experiment; full-UNO data was not used.
TRANSFER_WEIGHTS = RuleWeights(singleton=0.0, denial=2.0)

# Selected from a deliberately small shared-stream DEV comparison.  The useful
# change was *less* trust in the void feature, plus a mild preference to shed a
# unique rank/action trait; more aggressive action-card and denial weights lost.
# This is a DEV-selected candidate, not a held-out result.
DISTILLED_WEIGHTS = RuleWeights(singleton=0.5, denial=0.5)


class TacticalRuleBot:
    """Linear action scorer over memory, hand structure, and two-player tempo."""

    def __init__(self, rs: Ruleset, weights: RuleWeights, name: str) -> None:
        self.rs, self.weights, self.name = rs, weights, name

    def _score(self, obs: Observation, action: int, evidence: list[Void], coloured_available: bool) -> float:
        rs, w = self.rs, self.weights
        play = rs.actions[action]
        assert isinstance(play, Play)
        ct = rs.card_types[play.card]
        colour = play.color if ct.is_wild else ct.color
        remaining = list(obs.hand)
        remaining.remove(play.card)
        if not remaining:  # every immediate win is equivalent
            return 1_000_000.0

        kinds = {
            Kind.NUMBER: w.number,
            Kind.SKIP: w.block_switch,
            Kind.REVERSE: w.block_switch,
            Kind.DRAW_TWO: w.draw_two,
            Kind.WILD: w.wild,
            Kind.WILD_DRAW_FOUR: w.wild_draw_four,
        }
        score = kinds[ct.kind]
        counts = _colours(rs, tuple(remaining))
        score += w.colour_support * counts[colour]
        score += w.same_colour * (colour == obs.current_color)
        # A "trait" is a number rank across colours, or an action kind.  A unique
        # trait is safe to shed; duplicates are useful bridges when colour changes.
        def trait(card: int) -> tuple:
            x = rs.card_types[card]
            return (x.kind, x.rank) if x.kind is Kind.NUMBER else (x.kind,)

        score += w.singleton * (sum(trait(c) == trait(play.card) for c in obs.hand) == 1)
        if not ct.is_wild and colour != obs.current_color:
            score += w.pivot
        score += w.denial * _denial(rs, play.card, colour, evidence)

        is_tempo = ct.kind in (Kind.SKIP, Kind.REVERSE, Kind.DRAW_TWO, Kind.WILD_DRAW_FOUR)
        if is_tempo:
            playable = playable_table(rs)[play.card][colour]
            followups = sum(c in playable for c in remaining)
            score += w.followup * min(followups, 3)
            if followups == 0:
                score += w.stranded_action
            if ct.kind is Kind.DRAW_TWO:
                score += 2 * w.penalty_card
            elif ct.kind is Kind.WILD_DRAW_FOUR:
                score += 4 * w.penalty_card
            score += w.emergency * max(0, 3 - obs.opp_hand_size)

        if ct.is_wild and coloured_available:
            score += w.wild_hold
        return score

    def scores(self, obs: Observation, legal: tuple[int, ...]) -> dict[int, float]:
        plays = _plays(self.rs, legal)
        evidence = voids(self.rs, obs.public, obs.seat ^ 1)
        coloured = any(not self.rs.card_types[p.card].is_wild for _, p in plays)
        return {a: self._score(obs, a, evidence, coloured) for a, _ in plays}

    def policy(self, obs: Observation, legal: tuple[int, ...]) -> Policy:
        scores = self.scores(obs, legal)
        if not scores:
            return {self.rs.draw_action: 1.0}
        best = max(scores.values())
        return uniform(a for a, s in scores.items() if s == best)


class TransferRuleBot(TacticalRuleBot):
    """The exact-toy transfer candidate; fixed before full-UNO evaluation."""

    def __init__(self, rs: Ruleset) -> None:
        super().__init__(rs, TRANSFER_WEIGHTS, "rulebot_transfer_toy_v1")


class DistilledRuleBot(TacticalRuleBot):
    """The separately named full-UNO DEV candidate, not evidence of transfer."""

    def __init__(self, rs: Ruleset) -> None:
        super().__init__(rs, DISTILLED_WEIGHTS, "rulebot_full_dev_v1")


class UnpredictableRuleBot(DistilledRuleBot):
    """A strong stochastic opponent that deliberately violates inference assumptions."""

    def __init__(
        self,
        rs: Ruleset,
        temperature: float = 0.15,
        margin: float = 0.20,
        draw_rate: float = 0.02,
        keep_rate: float = 0.04,
        keep_wild_rate: float = 0.10,
    ) -> None:
        super().__init__(rs)
        if temperature <= 0 or margin < 0 or not all(0 <= p < 1 for p in (draw_rate, keep_rate, keep_wild_rate)):
            raise ValueError("invalid stochastic rulebot configuration")
        self.temperature, self.margin = temperature, margin
        self.draw_rate, self.keep_rate, self.keep_wild_rate = draw_rate, keep_rate, keep_wild_rate
        self.name = f"rulebot_unpredictable_t{temperature:g}_m{margin:g}_d{draw_rate:g}_k{keep_rate:g}_kw{keep_wild_rate:g}"

    def _play_distribution(self, obs: Observation, legal: tuple[int, ...]) -> Policy:
        scores = self.scores(obs, legal)
        best = max(scores.values())
        kept = {a: s for a, s in scores.items() if best - s <= self.margin}
        weights = {a: math.exp((s - best) / self.temperature) for a, s in kept.items()}
        z = sum(weights.values())
        return {a: x / z for a, x in weights.items()}

    def policy(self, obs: Observation, legal: tuple[int, ...]) -> Policy:
        plays = _plays(self.rs, legal)
        if not plays:
            return {self.rs.draw_action: 1.0}
        out = self._play_distribution(obs, legal)
        if obs.phase is Phase.AFTER_DRAW and self.rs.pass_action in legal:
            wild = self.rs.card_types[obs.drawn].is_wild
            keep = self.keep_wild_rate if wild else self.keep_rate
            if keep == 0:
                return out
            return {**{a: (1.0 - keep) * p for a, p in out.items()}, self.rs.pass_action: keep}
        if self.rs.draw_action in legal:
            if self.draw_rate == 0:
                return out
            return {**{a: (1.0 - self.draw_rate) * p for a, p in out.items()}, self.rs.draw_action: self.draw_rate}
        return out


FROZEN_RULEBOT_LEAGUE = (
    "heuristic_memory",
    "rulebot_transfer_toy_v1",
    "rulebot_full_dev_v1",
    "rulebot_unpredictable_t0.15_m0.2_d0.02_k0.04_kw0.1",
)
FROZEN_RULEBOT_WEIGHTS = (0.25, 0.25, 0.25, 0.25)


def rulebot_league(rs: Ruleset) -> dict[str, object]:
    """The frozen, equally weighted step-10C training league in display order."""

    agents = (MemoryHeuristic(rs), TransferRuleBot(rs), DistilledRuleBot(rs), UnpredictableRuleBot(rs))
    out = {a.name: a for a in agents}
    if tuple(out) != FROZEN_RULEBOT_LEAGUE:
        raise AssertionError("a rulebot policy changed name without updating the frozen league")
    return out
