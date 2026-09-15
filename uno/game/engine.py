"""Core engine: immutable State, deal / initial_state, legal_actions, step, returns.

See RULES.md sections 3, 4, 6, 7, 8, 9. Everything is a pure function of
(permutation, seed, action sequence).
"""

from __future__ import annotations

import random
from bisect import insort
from dataclasses import dataclass
from enum import IntEnum

from uno.game.rules import Draw, Kind, Pass, Play, Ruleset

Hand = tuple[int, ...]
Event = tuple  # ("setup", top, skipped) | ("play", p, type, color) | ("draw", p, n) | ("reshuffle",)


class Phase(IntEnum):
    TURN_START = 0
    AFTER_DRAW = 1


class IllegalAction(Exception):
    pass


@dataclass(frozen=True, slots=True)
class State:
    ruleset: Ruleset
    seed: int | None
    hands: tuple[Hand, Hand]  # sorted type ids
    draw_pile: tuple[int, ...]  # index 0 = top
    discard: tuple[int, ...]  # last = top
    current_color: int
    player: int
    phase: Phase
    drawn: int | None  # pending drawn card in AFTER_DRAW
    public: tuple[Event, ...]
    own_draws: tuple[tuple[int, ...], tuple[int, ...]]  # per player, in order
    initial_hands: tuple[Hand, Hand]
    n_reshuffles: int
    turn: int  # number of play + draw actions so far
    end: str | None  # None | "win" | "exhaustion" | "cap"
    winner: int | None

    @property
    def terminal(self) -> bool:
        return self.end is not None

    @property
    def top(self) -> int:
        return self.discard[-1]

    def __repr__(self) -> str:
        types = self.ruleset.card_types
        show = lambda cards: " ".join(str(types[c]) for c in cards)  # noqa: E731
        return (
            f"State(p{self.player} {self.phase.name} color={self.current_color} "
            f"top={types[self.top]} h0=[{show(self.hands[0])}] h1=[{show(self.hands[1])}] "
            f"pile={len(self.draw_pile)} turn={self.turn} end={self.end} winner={self.winner})"
        )


# -- construction -----------------------------------------------------------


def deal(ruleset: Ruleset, seed: int) -> tuple[int, ...]:
    """Shuffle the deck multiset into a permutation, index 0 = top."""
    deck = list(ruleset.deck)
    random.Random(seed).shuffle(deck)
    return tuple(deck)


def initial_state(ruleset: Ruleset, permutation: tuple[int, ...], seed: int | None = None) -> State:
    if tuple(sorted(permutation)) != ruleset.deck:
        raise ValueError("permutation is not a permutation of the ruleset's deck")
    if ruleset.reshuffle == "seeded" and seed is None:
        raise ValueError("seed is required when reshuffle == 'seeded'")
    h = ruleset.hand_size
    hands = (tuple(sorted(permutation[:h])), tuple(sorted(permutation[h : 2 * h])))
    pile = list(permutation[2 * h :])
    types = ruleset.card_types
    skipped: list[int] = []
    while pile and types[pile[0]].kind is not Kind.NUMBER:
        skipped.append(pile.pop(0))
    if not pile:
        raise ValueError("no number card available for the first discard")
    top = pile.pop(0)
    pile.extend(skipped)
    return State(
        ruleset=ruleset,
        seed=seed,
        hands=hands,
        draw_pile=tuple(pile),
        discard=(top,),
        current_color=types[top].color,
        player=0,
        phase=Phase.TURN_START,
        drawn=None,
        public=(("setup", top, tuple(skipped)),),
        own_draws=((), ()),
        initial_hands=hands,
        n_reshuffles=0,
        turn=0,
        end=None,
        winner=None,
    )


def new_game(ruleset: Ruleset, seed: int) -> State:
    return initial_state(ruleset, deal(ruleset, seed), seed)


# -- rules queries ----------------------------------------------------------


def is_playable(ruleset: Ruleset, card: int, top: int, current_color: int) -> bool:
    types = ruleset.card_types
    ct = types[card]
    if ct.is_wild or ct.color == current_color:
        return True
    tt = types[top]
    if ct.kind is Kind.NUMBER:
        return tt.kind is Kind.NUMBER and ct.rank == tt.rank
    return ct.kind is tt.kind


def legal_actions(state: State) -> tuple[int, ...]:
    if state.terminal:
        return ()
    rs = state.ruleset
    if state.phase is Phase.AFTER_DRAW:
        return rs.play_actions[state.drawn] + (rs.pass_action,)
    top, color = state.top, state.current_color
    acts: list[int] = []
    for t in sorted(set(state.hands[state.player])):
        if is_playable(rs, t, top, color):
            acts.extend(rs.play_actions[t])
    if not rs.forced_play or not acts:
        acts.append(rs.draw_action)
    return tuple(acts)


def returns(state: State) -> tuple[float, float]:
    """Zero-sum returns, +1 winner / -1 loser / 0 draw. Only defined at terminal states."""
    if not state.terminal:
        raise ValueError("returns() called on a non-terminal state")
    if state.winner is None:
        return (0.0, 0.0)
    return (1.0, -1.0) if state.winner == 0 else (-1.0, 1.0)


def win_prob(state: State, player: int) -> float:
    return (returns(state)[player] + 1.0) / 2.0


# -- transition -------------------------------------------------------------


class _Work:
    """Mutable scratch copy of a State used inside step()."""

    __slots__ = (
        "rs", "seed", "hands", "pile", "discard", "current_color", "player", "phase",
        "drawn", "public", "own_draws", "initial_hands", "n_reshuffles", "turn", "end", "winner",
    )  # fmt: skip

    def __init__(self, s: State) -> None:
        self.rs = s.ruleset
        self.seed = s.seed
        self.hands = [list(s.hands[0]), list(s.hands[1])]
        self.pile = list(s.draw_pile)
        self.discard = list(s.discard)
        self.current_color = s.current_color
        self.player = s.player
        self.phase = s.phase
        self.drawn = s.drawn
        self.public = list(s.public)
        self.own_draws = [list(s.own_draws[0]), list(s.own_draws[1])]
        self.initial_hands = s.initial_hands
        self.n_reshuffles = s.n_reshuffles
        self.turn = s.turn
        self.end = s.end
        self.winner = s.winner

    def freeze(self) -> State:
        return State(
            ruleset=self.rs,
            seed=self.seed,
            hands=(tuple(self.hands[0]), tuple(self.hands[1])),
            draw_pile=tuple(self.pile),
            discard=tuple(self.discard),
            current_color=self.current_color,
            player=self.player,
            phase=self.phase,
            drawn=self.drawn,
            public=tuple(self.public),
            own_draws=(tuple(self.own_draws[0]), tuple(self.own_draws[1])),
            initial_hands=self.initial_hands,
            n_reshuffles=self.n_reshuffles,
            turn=self.turn,
            end=self.end,
            winner=self.winner,
        )

    # -- primitives

    def next_player(self) -> None:
        self.player ^= 1
        self.phase = Phase.TURN_START
        self.drawn = None

    def end_exhaustion(self) -> None:
        n0, n1 = len(self.hands[0]), len(self.hands[1])
        self.end = "exhaustion"
        self.winner = None if n0 == n1 else (0 if n0 < n1 else 1)

    def draw(self, player: int, n: int) -> int:
        """Draw up to n cards for player. Returns the number actually drawn."""
        drawn = 0
        for _ in range(n):
            if not self.pile:
                if self.rs.reshuffle == "seeded" and len(self.discard) > 1:
                    self.n_reshuffles += 1
                    rest = self.discard[:-1]
                    self.discard = self.discard[-1:]
                    random.Random(f"{self.seed}:{self.n_reshuffles}").shuffle(rest)
                    self.pile = rest
                    self.public.append(("reshuffle",))
                else:
                    self.end_exhaustion()
                    break
            card = self.pile.pop(0)
            insort(self.hands[player], card)
            self.own_draws[player].append(card)
            drawn += 1
        if drawn:
            self.public.append(("draw", player, drawn))
        return drawn

    # -- actions

    def play(self, card: int, color: int | None) -> None:
        p = self.player
        hand = self.hands[p]
        hand.remove(card)
        self.discard.append(card)
        ct = self.rs.card_types[card]
        self.current_color = color if ct.is_wild else ct.color
        self.public.append(("play", p, card, self.current_color))
        self.phase = Phase.TURN_START
        self.drawn = None
        self.turn += 1
        if not hand:
            self.end = "win"
            self.winner = p
            return
        match ct.kind:
            case Kind.NUMBER | Kind.WILD:
                self.next_player()
            case Kind.SKIP | Kind.REVERSE:
                pass  # same player again
            case Kind.DRAW_TWO:
                self.draw(p ^ 1, 2)
            case Kind.WILD_DRAW_FOUR:
                self.draw(p ^ 1, 4)

    def voluntary_draw(self) -> None:
        p = self.player
        self.turn += 1
        if self.draw(p, 1) == 0:
            return  # exhaustion
        card = self.own_draws[p][-1]
        if self.rs.draw_then_may_play and is_playable(self.rs, card, self.discard[-1], self.current_color):
            self.phase = Phase.AFTER_DRAW
            self.drawn = card
        else:
            self.next_player()


def step(state: State, action: int, check: bool = True) -> State:
    """Apply an action. `check=False` skips the legality check for callers that
    just computed `legal_actions` themselves (tree walks, solvers)."""
    if check and action not in legal_actions(state):
        raise IllegalAction(f"action {action} is not legal in {state!r}")
    a = state.ruleset.actions[action]
    w = _Work(state)
    match a:
        case Play(card=card, color=color):
            w.play(card, color)
        case Draw():
            w.voluntary_draw()
        case Pass():
            w.next_player()
    if w.end is None and w.rs.max_turns is not None and w.turn >= w.rs.max_turns:
        w.end = "cap"
        w.winner = None
    return w.freeze()
