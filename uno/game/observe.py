"""Observations at the three information levels. See RULES.md section 10.

The information level is a property of the observer, not of the game.
Observations are frozen and hashable so they can key information sets.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from uno.game.engine import Event, Hand, Phase, State
from uno.game.rules import Ruleset


class Level(IntEnum):
    L1 = 1  # own hand + public info
    L2 = 2  # + opponent hand (initial hand and draws, so the level has perfect recall)
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
    # L2: the opponent's hand as seen at every one of the observer's decisions.
    # Kept as (initial hand, draws in order) so that the L2 information state has
    # perfect recall; `opp_hand` is derived from them and the public history.
    opp_hand: Hand | None = None
    opp_initial_hand: Hand | None = None
    opp_draws: tuple[int, ...] | None = None
    # L3
    draw_pile: tuple[int, ...] | None = None
    seed: int | None = None
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
        obs["opp_initial_hand"] = state.initial_hands[opp]
        obs["opp_draws"] = state.own_draws[opp]
    if level >= Level.L3:
        obs["draw_pile"] = state.draw_pile
        obs["seed"] = state.seed
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


# -- observations from information-set keys -----------------------------------
#
# The compiled tree (`uno.compiled`) keeps no State objects, only the key of
# every information set (`uno.game.keys.KEYS`). To tabulate an agent over those
# sets its observation has to be rebuilt from the key alone, which is possible
# because a key is by definition everything the observer knows.


def _voluntary(rs: Ruleset, public: tuple[Event, ...], i: int) -> bool:
    """Whether the draw event at index i was a voluntary draw (vs a +2 / +4 penalty).
    A penalty draw is recorded right after the play that caused it (a reshuffle
    may sit in between); nothing else can follow such a play."""
    from uno.game.rules import Kind

    j = i - 1
    while j >= 0 and public[j][0] == "reshuffle":
        j -= 1
    if j < 0 or public[j][0] != "play":
        return True
    kind = rs.card_types[public[j][2]].kind
    return kind not in (Kind.DRAW_TWO, Kind.WILD_DRAW_FOUR) or public[j][1] == public[i][1]


def _public_summary(rs: Ruleset, public: tuple[Event, ...]) -> dict:
    """Top discard, current color, per-player plays and draws, pile size, turn count,
    reshuffles, and whether the last event was a voluntary single draw."""
    types = rs.card_types
    top, color = public[0][1], types[public[0][1]].color
    plays: tuple[list[int], list[int]] = ([], [])
    drawn = [0, 0]
    pile = rs.deck_size - 2 * rs.hand_size - 1  # skipped setup cards sit at the bottom of the pile
    discard = 1
    turn = reshuffles = 0
    last_voluntary_draw = None  # (player, n) of the last event if it is a voluntary draw
    for i, ev in enumerate(public[1:], start=1):
        last_voluntary_draw = None
        match ev[0]:
            case "play":
                _, p, t, c = ev
                plays[p].append(t)
                top, color = t, c
                discard += 1
                turn += 1
            case "draw":
                _, p, n = ev
                drawn[p] += n
                pile -= n
                if _voluntary(rs, public, i):
                    turn += 1
                    last_voluntary_draw = (p, n)
            case "reshuffle":
                reshuffles += 1
                pile += discard - 1
                discard = 1
    return dict(top=top, color=color, plays=plays, drawn=drawn, pile=pile, turn=turn, reshuffles=reshuffles, last_voluntary_draw=last_voluntary_draw)


def _hand(initial: Hand, draws: tuple[int, ...], plays: list[int]) -> Hand:
    from collections import Counter

    return tuple(sorted((Counter(initial) + Counter(draws) - Counter(plays)).elements()))


def observation_from_key(rs: Ruleset, level: Level, key: tuple, seat: int | None = None) -> Observation:
    """The observation of `seat` at one of its own decisions, from the information-set
    key of `uno.game.keys.KEYS[level]`. L1 / L2 keys carry the seat; an L3 key is the
    state, whose player to move is the observer unless `seat` says otherwise."""
    from uno.game.engine import is_playable

    if level is Level.L3:
        hands, draw_pile, player, phase, drawn, public, own_draws = key
        s = _public_summary(rs, public)
        initial = tuple(_hand(hands[p], tuple(s["plays"][p]), list(own_draws[p])) for p in (0, 1))
        state = State(
            ruleset=rs, seed=None, hands=hands, draw_pile=draw_pile, discard=discard_from_public(public),
            current_color=s["color"], player=player, phase=phase, drawn=drawn, public=public, own_draws=own_draws,
            initial_hands=initial, n_reshuffles=s["reshuffles"], turn=s["turn"], end=None, winner=None,
        )  # fmt: skip
        return observe(state, player if seat is None else seat, Level.L3)

    me, initial_hand, public, own_draws = key[:4]
    if seat is not None and seat != me:
        raise ValueError("an L1 / L2 key carries its own seat")
    opp = me ^ 1
    s = _public_summary(rs, public)
    phase, drawn = Phase.TURN_START, None
    if (
        rs.draw_then_may_play
        and s["last_voluntary_draw"] == (me, 1)
        and is_playable(rs, own_draws[-1], s["top"], s["color"])
    ):
        phase, drawn = Phase.AFTER_DRAW, own_draws[-1]
    obs = dict(
        level=level,
        seat=me,
        player=me,
        phase=phase,
        current_color=s["color"],
        top=s["top"],
        opp_hand_size=rs.hand_size + s["drawn"][opp] - len(s["plays"][opp]),
        pile_size=s["pile"],
        public=public,
        turn=s["turn"],
        end=None,
        winner=None,
        initial_hand=initial_hand,
        hand=_hand(initial_hand, own_draws, s["plays"][me]),
        own_draws=own_draws,
        drawn=drawn,
    )
    if level is Level.L2:
        opp_initial, opp_draws = key[4:6]
        obs["opp_initial_hand"] = opp_initial
        obs["opp_draws"] = opp_draws
        obs["opp_hand"] = _hand(opp_initial, opp_draws, s["plays"][opp])
    return Observation(**obs)
