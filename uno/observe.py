"""Observations at the three information levels. See RULES.md section 10.

The information level is a property of the observer, not of the game.
Observations are frozen and hashable so they can key information sets.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from uno.engine import Event, Hand, Phase, State
from uno.rules import Ruleset


class Level(IntEnum):
    L1 = 1  # own hand + public info
    L2 = 2  # + opponent hand
    L3 = 3  # + draw pile order and seed (perfect information)


@dataclass(frozen=True, slots=True)
class Observation:
    level: Level
    seat: int
    # public
    player: int  # who is to move
    phase: Phase
    current_color: int
    top: int
    opp_hand_size: int
    pile_size: int
    public: tuple[Event, ...]
    turn: int
    end: str | None
    winner: int | None
    # private to the observer
    initial_hand: Hand
    hand: Hand
    own_draws: tuple[int, ...]
    drawn: int | None  # pending drawn card, only meaningful when it is the observer's AFTER_DRAW
    # L2
    opp_hand: Hand | None = None
    # L3
    draw_pile: tuple[int, ...] | None = None
    seed: int | None = None
    opp_initial_hand: Hand | None = None
    opp_draws: tuple[int, ...] | None = None
    n_reshuffles: int | None = None

    @property
    def info_state(self) -> tuple:
        """The minimal L1 information state: everything else at L1 is derived from it."""
        return (self.seat, self.initial_hand, self.public, self.own_draws)


def info_state(state: State, player: int) -> tuple:
    """`observe(state, player).info_state` without building the Observation."""
    return (player, state.initial_hands[player], state.public, state.own_draws[player])


def observe(state: State, player: int, level: Level = Level.L1) -> Observation:
    opp = player ^ 1
    obs = dict(
        level=level,
        seat=player,
        player=state.player,
        phase=state.phase,
        current_color=state.current_color,
        top=state.top,
        opp_hand_size=len(state.hands[opp]),
        pile_size=len(state.draw_pile),
        public=state.public,
        turn=state.turn,
        end=state.end,
        winner=state.winner,
        initial_hand=state.initial_hands[player],
        hand=state.hands[player],
        own_draws=state.own_draws[player],
        drawn=state.drawn if state.player == player else None,
    )
    if level >= Level.L2:
        obs["opp_hand"] = state.hands[opp]
    if level >= Level.L3:
        obs["draw_pile"] = state.draw_pile
        obs["seed"] = state.seed
        obs["opp_initial_hand"] = state.initial_hands[opp]
        obs["opp_draws"] = state.own_draws[opp]
        obs["n_reshuffles"] = state.n_reshuffles
    return Observation(**obs)


def discard_from_public(public: tuple[Event, ...]) -> tuple[int, ...]:
    """Rebuild the discard pile (last = top) from the public history."""
    discard: list[int] = []
    for ev in public:
        match ev[0]:
            case "setup":
                discard = [ev[1]]
            case "play":
                discard.append(ev[2])
            case "reshuffle":
                discard = discard[-1:]
    return tuple(discard)


def reconstruct(ruleset: Ruleset, obs: Observation) -> State:
    """Rebuild the exact State from an L3 observation."""
    if obs.level is not Level.L3:
        raise ValueError("only an L3 observation determines the state")
    me, opp = obs.seat, obs.seat ^ 1
    hands = [None, None]
    hands[me], hands[opp] = obs.hand, obs.opp_hand
    initial = [None, None]
    initial[me], initial[opp] = obs.initial_hand, obs.opp_initial_hand
    draws = [None, None]
    draws[me], draws[opp] = obs.own_draws, obs.opp_draws
    return State(
        ruleset=ruleset,
        seed=obs.seed,
        hands=(hands[0], hands[1]),
        draw_pile=obs.draw_pile,
        discard=discard_from_public(obs.public),
        current_color=obs.current_color,
        player=obs.player,
        phase=obs.phase,
        drawn=obs.drawn if obs.player == me else (draws[opp][-1] if obs.phase is Phase.AFTER_DRAW else None),
        public=obs.public,
        own_draws=(draws[0], draws[1]),
        initial_hands=(initial[0], initial[1]),
        n_reshuffles=obs.n_reshuffles,
        turn=obs.turn,
        end=obs.end,
        winner=obs.winner,
    )
