"""Baseline agents. See docs/spec.md section 13.

An agent maps (observation, legal actions) to a probability distribution over
actions. It never samples: best response tabulates the policy and AIVAT needs
the probabilities. Sampling lives in `uno.eval.play`.
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Protocol

from uno.game.engine import Phase
from uno.game.observe import Observation
from uno.game.rules import Kind, Play, Ruleset

Policy = dict[int, float]


class Agent(Protocol):
    name: str

    def policy(self, obs: Observation, legal: tuple[int, ...]) -> Policy: ...


def uniform(actions) -> Policy:
    actions = tuple(actions)
    p = 1.0 / len(actions)
    return {a: p for a in actions}


def sample(probs: Policy, rng: random.Random) -> int:
    r = rng.random()
    acc = 0.0
    last = None
    for a, p in probs.items():
        acc += p
        last = a
        if r < acc:
            return a
    return last  # rounding


class UniformLegal:
    """Uniform over every legal action, Draw and Pass included. Sanity opponent."""

    name = "uniform_legal"

    def __init__(self, rs: Ruleset) -> None:
        self.rs = rs

    def policy(self, obs: Observation, legal: tuple[int, ...]) -> Policy:
        return uniform(legal)


class PlayFirstRandom:
    """Uniform over legal plays; draws only when nothing plays; always plays a drawn card.

    The ladder's "Random" row.
    """

    name = "random"

    def __init__(self, rs: Ruleset) -> None:
        self.rs = rs
        self._is_play = tuple(isinstance(a, Play) for a in rs.actions)

    def policy(self, obs: Observation, legal: tuple[int, ...]) -> Policy:
        plays = [a for a in legal if self._is_play[a]]
        return uniform(plays if plays else legal)


class Heuristic:
    """Rulebot, symmetric under color and rank renaming (docs/spec.md section 18).

    Play over draw. Among playable non-wild cards prefer action cards when the
    opponent is low, then the color held most, then a card of the current color
    (so the color stays), then a number card (action cards are kept for when
    the opponent is low); what still ties is played uniformly at random, which
    is what keeps the policy symmetric. Hold wilds unless nothing else plays; a
    wild takes the color held most (the current color on ties, then uniform).
    After a draw, always play the drawn card.
    """

    name = "heuristic"

    def __init__(self, rs: Ruleset, low_threshold: int = 2) -> None:
        self.rs = rs
        self.low_threshold = low_threshold

    def _color_counts(self, hand: tuple[int, ...]) -> Counter:
        types = self.rs.card_types
        return Counter(types[c].color for c in hand if not types[c].is_wild)

    def policy(self, obs: Observation, legal: tuple[int, ...]) -> Policy:
        rs = self.rs
        types = rs.card_types
        plays = [(a, rs.actions[a]) for a in legal if isinstance(rs.actions[a], Play)]
        if not plays:
            return {rs.draw_action: 1.0}
        colors = self._color_counts(obs.hand)
        opp_low = obs.opp_hand_size <= self.low_threshold
        colored = [(a, p) for a, p in plays if not types[p.card].is_wild]
        if colored:
            def key(ap):
                ct = types[ap[1].card]
                return (opp_low and ct.kind is not Kind.NUMBER, colors[ct.color], ct.color == obs.current_color, ct.kind is Kind.NUMBER)

            best = max(key(ap) for ap in colored)
            return uniform(a for a, p in colored if key((a, p)) == best)
        # only wilds play: the color held most, the current color on ties
        color_key = lambda c: (colors[c], c == obs.current_color)  # noqa: E731
        best_color = max(color_key(c) for c in range(rs.colors))

        def wild_key(ap):
            ct = types[ap[1].card]
            return (color_key(ap[1].color) == best_color, opp_low == (ct.kind is Kind.WILD_DRAW_FOUR))

        best = max(wild_key(ap) for ap in plays)
        return uniform(a for a, p in plays if wild_key((a, p)) == best)


class KeepsDrawn:
    """`PlayFirstRandom`, except it keeps a playable drawn card with probability `p`.

    The opponent the `voids` belief assumes does not exist. That rung counts the cards drawn
    *strictly after* a void, so it assumes a playable drawn card is played at once (docs/spec.md
    section 26.2) -- true of every rulebot here, and not a rule: `draw_then_may_play` offers
    exactly this choice at every `AFTER_DRAW`, and a `Pass` writes no public event, so nothing in
    the history distinguishes "could not play" from "chose not to". An RL agent that learns to
    hold a card would not decline it.

    Lives here rather than in the tests because the closure test needs its *exact* posterior
    (`uv run uno pimc closure TOY ... passer`), which means tabulating it on a compiled tree.
    `p` is therefore part of `name`: a name identifies a policy (docs/spec.md section 13) and
    `uno.store.load_or_tabulate` is keyed on it, so two `p` would otherwise share one cache file.
    """

    def __init__(self, rs: Ruleset, p: float = 0.5) -> None:
        self.rs, self.p, self.inner = rs, p, PlayFirstRandom(rs)
        self.name = f"passer{p:g}"
        self._is_play = tuple(isinstance(a, Play) for a in rs.actions)

    def policy(self, obs: Observation, legal: tuple[int, ...]) -> Policy:
        if obs.phase is Phase.AFTER_DRAW:
            plays = [a for a in legal if self._is_play[a]]
            passes = [a for a in legal if not self._is_play[a]]
            if plays and passes:
                # The play mass keeps `PlayFirstRandom`'s own split over the drawn card's actions
                # (a wild has one per colour), so the agent stays symmetric under renaming.
                out = {a: self.p / len(passes) for a in passes}
                out.update({a: (1.0 - self.p) * q for a, q in self.inner.policy(obs, tuple(plays)).items()})
                return out
        return self.inner.policy(obs, legal)


class BySeat:
    """One agent per seat, dispatched on the observer's seat.

    Best responses are solved per seat, so a tabular best-response row agent of
    the ladder is `BySeat(TabularAgent(rs, br_seat0), TabularAgent(rs, br_seat1))`;
    the mirrored evaluator plays it from both seats of every deal.
    """

    def __init__(self, seat0: Agent, seat1: Agent, name: str | None = None) -> None:
        self.seats = (seat0, seat1)
        self.name = name or f"{seat0.name}|{seat1.name}"

    def policy(self, obs: Observation, legal: tuple[int, ...]) -> Policy:
        return self.seats[obs.seat].policy(obs, legal)


class Mixture:
    """`(1 - eps) * agent + eps * other` at every decision, over the legal actions in
    their order: a near-identical variant of `agent`, for the step 9 pairing checks.
    Symmetric when both agents are. `stream` (see `uno.eval.play.stream_name`) lets
    it share `agent`'s action rng, as two checkpoints of one policy would."""

    def __init__(self, agent: Agent, other: Agent, eps: float, name: str | None = None, stream: str | None = None) -> None:
        self.agent, self.other, self.eps = agent, other, eps
        self.name = name or f"{agent.name}+{eps:g}{other.name}"
        self.stream = stream

    def policy(self, obs: Observation, legal: tuple[int, ...]) -> Policy:
        p, q = self.agent.policy(obs, legal), self.other.policy(obs, legal)
        mix = {a: (1.0 - self.eps) * p.get(a, 0.0) + self.eps * q.get(a, 0.0) for a in legal}
        return {a: x for a, x in mix.items() if x > 0}
