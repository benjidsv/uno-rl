"""Exact solvers: policy evaluation and best response at L1, L2 and L3.

See RULES.md section 14 and notes.md ("The oracle levels").

A best response is the responder's best pure strategy against a *fixed*
opponent (an `Agent` seen at L1), given the responder's information level:

- L1: information set = `info_state` (seat, initial hand, public history, own
  draws).
- L2: L1 plus the opponent's initial hand and draws in order, i.e. the
  opponent's hand as seen at every one of the responder's decisions. Keying on
  the *current* hand alone would merge histories the responder could tell apart
  earlier (which card the opponent drew), an imperfect-recall partition on
  which the local maximisation below is not guaranteed optimal.
- L3: the whole state, pile order included. Every information set is one
  history, so L3 is plain expectimax over a deterministic game.

L1 and L2 walk the lazy-chance tree (`uno.tree`): a draw is a chance node over
the distinct types left in the pile. L3 knows the pile order, so its roots are
*worlds* (every deal x every order of the unknown pile) and below a world the
tree is the bare engine, with no chance at all.

Algorithm (two passes over the tree):

1. forward: depth-first from the roots, accumulating for every history its
   counterfactual reach (deal x chance x opponent probabilities, never the
   responder's own) and grouping the responder's decision histories by
   information set;
2. backward: memoised values. At a responder history the action is the one
   that maximises the reach-weighted sum of continuation values over the whole
   information set. This is a best response only because every level has
   perfect recall (all histories of a set share the responder's past sets, so
   the responder's own reach is a common factor); the recursion is well founded
   because the public history grows strictly between two responder decisions.

Values are the responder's expected return in [-1, 1]; `score` is (v + 1) / 2,
i.e. P(win) with draws worth 1/2, comparable to `uno.play.match`.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Callable, Hashable

from uno.agents import Agent, Policy
from uno.engine import State, legal_actions, new_game, returns, step
from uno.observe import Level, Observation, info_state, observe, reconstruct
from uno.rules import Ruleset
from uno.tree import Outcome, children, initial_deals, ordered_picks, pile_split

Key = Hashable
Expand = Callable[[State], dict[int, list[Outcome]]]


# -- information-set keys -----------------------------------------------------


def l1_key(state: State, seat: int) -> Key:
    return info_state(state, seat)


def l2_key(state: State, seat: int) -> Key:
    opp = seat ^ 1
    return (*info_state(state, seat), state.initial_hands[opp], state.own_draws[opp])


def l3_key(state: State, seat: int) -> Key:
    """The full state. Seed excluded: worlds carry none, real games do."""
    return (state.hands, state.draw_pile, state.player, state.phase, state.drawn, state.public, state.own_draws)


KEYS: dict[Level, Callable[[State, int], Key]] = {Level.L1: l1_key, Level.L2: l2_key, Level.L3: l3_key}


def obs_key(rs: Ruleset, obs: Observation) -> Key:
    """The same key as `KEYS[obs.level]`, computed from an observation."""
    match obs.level:
        case Level.L1:
            return obs.info_state
        case Level.L2:
            return (*obs.info_state, obs.opp_initial_hand, obs.opp_draws)
        case Level.L3:
            return l3_key(reconstruct(rs, obs), obs.seat)
    raise ValueError(obs.level)


# -- roots and expansion ------------------------------------------------------


def worlds(rs: Ruleset) -> list[Outcome]:
    """L3 roots: every deal x every order of the unknown part of the pile."""
    out: list[Outcome] = []
    for s, p in initial_deals(rs):
        unknown, tail = pile_split(s)
        for seq, q in ordered_picks(dict(unknown), sum(unknown.values())):
            out.append((replace(s, draw_pile=(*seq, *tail)), p * q))
    return out


def engine_children(state: State) -> dict[int, list[Outcome]]:
    """Children under the bare engine: the pile order is known, so no chance."""
    return {a: [(step(state, a, check=False), 1.0)] for a in legal_actions(state)}


def roots_and_expand(rs: Ruleset, levels) -> tuple[list[Outcome], Expand]:
    """Worlds + engine when anyone plays at L3, the lazy tree otherwise."""
    if Level.L3 in levels:
        return worlds(rs), engine_children
    return initial_deals(rs), children


# -- policy evaluation --------------------------------------------------------


def evaluate(
    rs: Ruleset,
    agents: tuple[Agent, Agent],
    levels=(Level.L1, Level.L1),
    roots: list[Outcome] | None = None,
    expand: Expand | None = None,
) -> tuple[float, float]:
    """Exact expected returns of two fixed agents (seat 0, seat 1).

    `roots` defaults to the whole game (all deals) and `expand` to the tree
    those roots need (lazy chance tree, or the bare engine when someone plays
    at L3). Pass `roots=[(state, 1.0)]` and `expand=engine_children` for the
    value of one real position: the pile order is then fixed and the result is
    the expectation over the agents' randomness only.
    """
    if expand is None:
        expand = engine_children if Level.L3 in levels else children
    if roots is None:  # only built when needed: the L3 worlds are expensive
        roots = worlds(rs) if expand is engine_children else initial_deals(rs)
    total = [0.0, 0.0]
    stack = list(roots)
    while stack:
        s, reach = stack.pop()
        if s.terminal:
            r = returns(s)
            total[0] += reach * r[0]
            total[1] += reach * r[1]
            continue
        p = s.player
        ch = expand(s)
        probs = agents[p].policy(observe(s, p, levels[p]), tuple(ch))
        for a, q in probs.items():
            for c, pc in ch[a]:
                stack.append((c, reach * q * pc))
    return (total[0], total[1])


def evaluate_on_seeds(rs: Ruleset, agents: tuple[Agent, Agent], seeds, levels=(Level.L1, Level.L1)) -> tuple[float, float]:
    """Mean exact returns over the games of `seeds` (invalid deals skipped), each
    played on its real deal *and* pile order: the expectation over the agents'
    randomness only, with the deal and pile luck of that seed set left in. This
    is exactly what a simulation on those seeds estimates, so the difference
    between the two is action-sampling noise and nothing else (zero for two
    deterministic agents). Works at every level."""
    total = [0.0, 0.0]
    n = 0
    for seed in seeds:
        try:
            s0 = new_game(rs, seed)
        except ValueError:
            continue
        v = evaluate(rs, agents, levels, roots=[(s0, 1.0)], expand=engine_children)
        total[0] += v[0]
        total[1] += v[1]
        n += 1
    return (total[0] / n, total[1] / n)


# -- best response ------------------------------------------------------------


@dataclass
class BestResponse:
    seat: int
    level: Level
    opponent: str
    value: float  # responder's expected return in [-1, 1]
    table: dict[Key, int]  # information set -> action (only sets reachable under the opponent)
    histories: int
    infosets: int
    seconds: float

    @property
    def score(self) -> float:
        return (self.value + 1.0) / 2.0

    def __str__(self) -> str:
        return (
            f"BR {self.level.name} seat {self.seat} vs {self.opponent}: score={self.score:.4f} "
            f"(histories={self.histories}, infosets={self.infosets}, {self.seconds:.1f}s)"
        )


def best_response(rs: Ruleset, opponent: Agent, seat: int, level: Level) -> BestResponse:
    t0 = time.perf_counter()
    key = KEYS[level]
    roots, expand = roots_and_expand(rs, (level,))
    opp = seat ^ 1

    # pass 1: reach probabilities and information sets. For a responder history
    # `kids` holds {action: outcomes}; for an opponent history a flat list of
    # (child, opponent prob x chance prob).
    kids: dict[State, object] = {}
    members: dict[Key, list[tuple[State, float]]] = defaultdict(list)
    histories = 0
    stack = list(roots)
    while stack:
        s, reach = stack.pop()
        histories += 1
        if s.terminal:
            continue
        if s in kids:
            raise RuntimeError("the game tree is not a tree: a history was reached twice")
        ch = expand(s)
        if s.player == seat:
            kids[s] = ch
            group = members[key(s, seat)]
            if group and tuple(kids[group[0][0]]) != tuple(ch):
                raise RuntimeError(f"{level.name} information set with differing legal actions: {key(s, seat)!r}")
            group.append((s, reach))
            for outs in ch.values():
                for c, p in outs:
                    stack.append((c, reach * p))
        else:
            probs = opponent.policy(observe(s, opp, Level.L1), tuple(ch))
            flat = [(c, q * p) for a, q in probs.items() for c, p in ch[a]]
            kids[s] = flat
            for c, w in flat:
                stack.append((c, reach * w))

    # pass 2: values, choosing the best action per information set on demand
    memo: dict[State, float] = {}
    table: dict[Key, int] = {}

    def value(s: State) -> float:
        v = memo.get(s)
        if v is not None:
            return v
        if s.terminal:
            v = returns(s)[seat]
        elif s.player == seat:
            v = sum(p * value(c) for c, p in kids[s][choose(key(s, seat))])
        else:
            v = sum(w * value(c) for c, w in kids[s])
        memo[s] = v
        return v

    def choose(k: Key) -> int:
        a = table.get(k)
        if a is not None:
            return a
        group = members[k]
        best, best_v = -1, float("-inf")
        for a in kids[group[0][0]]:  # legal actions are the same across an information set
            va = sum(r * sum(p * value(c) for c, p in kids[h][a]) for h, r in group)
            if va > best_v + 1e-12:  # ties: lowest action id
                best, best_v = a, va
        table[k] = best
        return best

    total = sum(p * value(s) for s, p in roots)
    return BestResponse(
        seat=seat,
        level=level,
        opponent=getattr(opponent, "name", type(opponent).__name__),
        value=total,
        table=table,
        histories=histories,
        infosets=len(members),
        seconds=time.perf_counter() - t0,
    )


class TabularAgent:
    """Plays a best-response table. Must observe at `br.level`.

    Information sets missing from the table (unreachable under the opponent the
    table was computed against) fall back to the first legal action and are
    counted in `misses`.
    """

    def __init__(self, rs: Ruleset, br: BestResponse, name: str | None = None) -> None:
        self.rs = rs
        self.br = br
        self.level = br.level
        self.table = br.table
        self.name = name or f"br_{br.level.name.lower()}_vs_{br.opponent}"
        self.misses = 0

    def policy(self, obs: Observation, legal: tuple[int, ...]) -> Policy:
        if obs.level is not self.level:
            raise ValueError(f"{self.name} needs {self.level.name} observations, got {obs.level.name}")
        a = self.table.get(obs_key(self.rs, obs))
        if a is None or a not in legal:
            self.misses += 1
            a = legal[0]
        return {a: 1.0}


class PureStrategy:
    """A fixed action per information set, for brute-force checks. Same fallback as TabularAgent."""

    def __init__(self, rs: Ruleset, level: Level, table: dict[Key, int]) -> None:
        self.rs, self.level, self.table, self.name = rs, level, table, "pure"

    def policy(self, obs: Observation, legal: tuple[int, ...]) -> Policy:
        a = self.table.get(obs_key(self.rs, obs), legal[0])
        return {a: 1.0}
