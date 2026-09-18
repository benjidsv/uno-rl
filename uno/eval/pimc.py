"""PIMC values: Monte Carlo rollouts on determinized states, the teacher of step 9.5c's value
network (docs/foundations/spec.md section 24).

**Determinization at L2.** AIVAT's value function may see both hands, the public history and the
pile as a multiset, never the pile order (`uno.eval.aivat`). Given that, the order of the unknown
part of the pile is uniform (the setup tail stays at the bottom until the first reshuffle), which is
the chance model of `uno.game.chance.draw_outcomes`. `determinize` shuffles that part with an rng of
its own and gives the state a fresh seed, so the engine's reshuffles and ticket draws, and the
agents' action choices in the rollout, are independent of the real game. Draws become `"pile"`: a
uniform order drawn from the top is distributed like tickets (section 8). Unlike `draw_outcomes`,
nothing is skipped: a penalty draw across a reshuffle is played by the engine.

**Rollout values.** `rollout_value(state, agents, n)` is the mean seat-0 return of `n` games played
from `n` determinizations by the profile's own policies: an unbiased estimate of the profile's L2
value, with variance about (1 - V^2) / n. `rollout_q` takes an action first, estimating AIVAT's
pre-chance `alpha(h, a)`.

**Why a teacher and not AIVAT's V.** With rollouts of the played profile as V the corrections
telescope towards "n more games from the same hands", at the cost of those games; independent noise
per correction adds about 2 (1 - V^2) / n per decision. `RolloutValues` exists to measure that (and to
validate rollouts through AIVAT on the toys); the production V is a network fitted on rollout targets
(`uno.eval.value_net`), at the cost of a critic.
"""

from __future__ import annotations

import math
import os
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from functools import cache
from typing import Callable, Iterable

import numpy as np

from uno.agents.baseline import sample
from uno.eval.aivat import StepRecord, rescored_return, terminal_return
from uno.eval.play import Trace, play_game, stable_hash
from uno.game.chance import draw_count, pile_split
from uno.game.engine import State, legal_actions, new_game, returns, step
from uno.game.observe import Level, observe
from uno.game.rules import Ruleset


@cache
def pile_ruleset(rs: Ruleset) -> Ruleset:
    """`rs` with `draws="pile"` (one object per ruleset: card tables are cached per object)."""
    return rs if rs.draws == "pile" else replace(rs, draws="pile")


def unknown_cards(state: State) -> tuple[list[int], tuple[int, ...]]:
    """(the unknown part of the pile, sorted; the known tail), from L2 information only."""
    if state.n_reshuffles == 0:
        unknown, tail = pile_split(state)
        return sorted(unknown.elements()), tail
    return sorted(state.draw_pile), ()


def determinize(state: State, rng: random.Random) -> State:
    """`state` with the unknown part of its pile in a uniform random order, a fresh seed and pile
    draws. A function of `rng` and of the state up to its pile order and seed (tests check that)."""
    cards, tail = unknown_cards(state)
    rng.shuffle(cards)
    return replace(state, ruleset=pile_ruleset(state.ruleset), draw_pile=(*cards, *tail), seed=rng.getrandbits(62))


def rollout(state: State, agents, rng: random.Random, exhaustion: str = "hand") -> float:
    """Seat-0 return of one game played from `state` (already determinized) by `agents` = (seat 0,
    seat 1), actions sampled from `rng`, exhaustion endings scored as `exhaustion`."""
    s, parent, a = state, None, -1
    while not s.terminal:
        p = s.player
        legal = legal_actions(s)
        a = sample(agents[p].policy(observe(s, p, Level.L1), legal), rng)
        parent, s = s, step(s, a, check=False)
    return returns(s)[0] if parent is None else terminal_return(parent, a, s, exhaustion)


def rollout_rng(seed, label) -> random.Random:
    """The rng of the rollouts behind one value: keyed on the game's seed and a label that says
    which value of the game it is, never on anything the game's own chance or actions use."""
    return random.Random(stable_hash((seed, "pimc", label)))


def rollout_returns(state: State, agents, n: int, rng: random.Random, action: int | None = None, exhaustion: str = "hand") -> np.ndarray:
    """(n,) returns of rollouts from `n` determinizations of `state`, taking `action` first if given."""
    out = np.empty(n)
    for j in range(n):
        s = determinize(state, rng)
        if action is not None:
            parent, s = s, step(s, action, check=False)
            if s.terminal:
                out[j] = terminal_return(parent, action, s, exhaustion)
                continue
        out[j] = rollout(s, agents, rng, exhaustion)
    return out


def rollout_value(state: State, agents, n: int, rng: random.Random, exhaustion: str = "hand") -> tuple[float, float]:
    """(mean, variance of the mean) of `n` rollouts from `state`."""
    u = rollout_returns(state, agents, n, rng, None, exhaustion)
    return float(u.mean()), float(u.var(ddof=1) / n) if n > 1 else math.nan


def rollout_q(state: State, action: int, agents, n: int, rng: random.Random, exhaustion: str = "hand") -> tuple[float, float]:
    """(mean, variance of the mean) of `n` rollouts taking `action` at `state` first: the
    pre-chance value `alpha(state, action)`."""
    u = rollout_returns(state, agents, n, rng, action, exhaustion)
    return float(u.mean()), float(u.var(ddof=1) / n) if n > 1 else math.nan


# -- rollouts as AIVAT's value model (for measurement only) ---------------------------------


class RolloutValues:
    """`uno.eval.aivat.ValueModel` with rollout values: `n` rollouts for the node, for each action
    of the support (its `alpha`) and for the child, each value with its own rng (`rollout_rng` on
    the seed, the decision and what is valued). The noise of every value is independent of the
    game, so AIVAT stays unbiased; it is not independent across the values of one decision, which
    is fine for the same reason. Costs ~(support + 2) * n rollouts per decision."""

    def __init__(self, agents, n: int, exhaustion: str = "hand", deal_mean: float | None = None) -> None:
        self.agents, self.n, self.exhaustion, self.deal_mean = agents, n, exhaustion, deal_mean

    def records(self, trace: Trace) -> list[StepRecord]:
        steps = trace.steps
        u = rescored_return(trace, self.exhaustion)
        seed = steps[0].state.seed if steps else None
        out = []
        for i, st in enumerate(steps):
            s = st.state
            node = rollout_value(s, self.agents, self.n, rollout_rng(seed, (i, "node")), self.exhaustion)[0]
            alpha = {a: rollout_q(s, a, self.agents, self.n, rollout_rng(seed, (i, "alpha", a)), self.exhaustion)[0] for a in st.probs}
            nxt = steps[i + 1].state if i + 1 < len(steps) else trace.final
            child = u if nxt.terminal else rollout_value(nxt, self.agents, self.n, rollout_rng(seed, (i, "child")), self.exhaustion)[0]
            out.append(StepRecord(s.player, st.probs, st.action, node, alpha, child))
        return out


# -- parallel map ------------------------------------------------------------------------------


def chunks(seq, k: int) -> list:
    seq = list(seq)
    size = max(1, math.ceil(len(seq) / k))
    return [seq[i : i + size] for i in range(0, len(seq), size)]


def _init_worker() -> None:
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:
        pass


def pmap(fn: Callable, args: Iterable, workers: int | None = None) -> list:
    """`[fn(*a) for a in args]`, over processes when `workers` > 1 (default: every core). `fn` and
    its arguments must pickle (agents holding a compiled tree are better run with workers=1)."""
    args = list(args)
    workers = (os.cpu_count() or 1) if workers is None else workers
    if workers <= 1 or len(args) <= 1:
        return [fn(*a) for a in args]
    with ProcessPoolExecutor(max_workers=min(workers, len(args)), initializer=_init_worker) as ex:
        return list(ex.map(fn, *zip(*args)))


# -- the deal's share of the variance ---------------------------------------------------------


def _floor_chunk(rs: Ruleset, agents, seeds, replays: int, exhaustion: str) -> list[list[float]]:
    out = []
    for seed in seeds:
        try:
            s0 = new_game(rs, seed)
        except ValueError:
            continue
        rng = rollout_rng(seed, "floor")
        out.append([rollout(determinize(s0, rng), agents, rng, exhaustion) for _ in range(replays)])
    return out


@dataclass(frozen=True)
class Floor:
    """The variance of a game's return split into the deal's value and the rest (seat-0 returns)."""

    deals: int
    replays: int
    mean: float  # seat-0 return
    total: float  # Var(u)
    between: float  # Var over deals of E[u | deal]: AIVAT's residual with exact L2 values, chance+both, no deal
    share_ci: tuple[float, float]  # 95 % bootstrap over deals of between / total

    @property
    def share(self) -> float:
        return self.between / self.total


def floor_from(u: np.ndarray, reps: int = 1000, seed: int = 0) -> Floor:
    """From (deals, replays) returns: between = Var(deal means) - mean within-deal variance /
    replays (unbiased), total = between + mean within-deal variance."""
    d, r = u.shape

    def parts(x):
        within = x.var(axis=1, ddof=1).mean()
        between = x.mean(axis=1).var(ddof=1) - within / r
        return between, between + within

    between, total = parts(u)
    rng = np.random.default_rng(seed)
    shares = []
    for _ in range(reps):
        b, t = parts(u[rng.integers(0, d, d)])
        shares.append(b / t)
    lo, hi = np.quantile(shares, (0.025, 0.975))
    return Floor(d, r, float(u.mean()), float(total), float(between), (float(lo), float(hi)))


def deal_floor(rs: Ruleset, agents, seeds, replays: int, exhaustion: str = "hand", workers: int | None = None) -> Floor:
    """Replay every dealt hand (and setup) of `seeds` under `replays` determinizations of the pile
    with the profile `agents`: the deal's share of the outcome variance, which is the variance ratio
    AIVAT reaches with the exact L2 values and every correction but the deal's."""
    k = (os.cpu_count() or 1) * 32 if workers is None or workers > 1 else 1  # small chunks: full-UNO game lengths are heavy-tailed
    parts = pmap(_floor_chunk, [(rs, agents, c, replays, exhaustion) for c in chunks(seeds, k)], workers)
    return floor_from(np.array([row for p in parts for row in p]))


# -- teacher targets ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Targets:
    """Rollout targets on the states a value model is asked about along played games."""

    x: np.ndarray  # (states, features) float32
    y: np.ndarray  # (states,) rollout mean, seat-0 return
    noise: np.ndarray  # (states,) variance of that mean (sample variance / n)
    u: np.ndarray  # (states,) final seat-0 return of the state's game
    game: np.ndarray  # (states,) index of the game within its seed list
    on_path: np.ndarray  # (states,) bool: a state the game visited (else a sibling child)
    n: int

    @staticmethod
    def concat(parts: list[Targets]) -> Targets:
        offset, games = 0, []
        for p in parts:
            games.append(p.game + offset)
            offset += int(p.game.max()) + 1 if len(p.game) else 0
        return Targets(
            np.concatenate([p.x for p in parts]), np.concatenate([p.y for p in parts]), np.concatenate([p.noise for p in parts]),
            np.concatenate([p.u for p in parts]), np.concatenate(games), np.concatenate([p.on_path for p in parts]), parts[0].n,
        )  # fmt: skip


def _teacher_chunk(rs: Ruleset, agents, seeds, n: int, features: Callable, exhaustion: str, off_path: bool, rollout_agents=None) -> Targets:
    roll = agents if rollout_agents is None else rollout_agents
    rows, ys, noise, us, gs, on = [], [], [], [], [], []
    g = 0
    for seed in seeds:
        try:
            t = play_game(rs, seed, agents)
        except ValueError:
            continue
        u = rescored_return(t, exhaustion)
        pick = random.Random(stable_hash((seed, "pimc", "siblings")))
        for i, st in enumerate(t.steps):
            s = st.state
            todo = [(s, True, (i, "state"))]
            if off_path:
                other = [a for a in st.probs if a != st.action]
                if other:
                    todo.append((step(determinize(s, pick), pick.choice(other), check=False), False, (i, "other")))
                if draw_count(s, st.action) > 0:
                    todo.append((step(determinize(s, pick), st.action, check=False), False, (i, "outcome")))
            for state, visited, label in todo:
                if state.terminal:
                    continue
                m, v = rollout_value(state, roll, n, rollout_rng(seed, label), exhaustion)
                rows.append(features(state))
                ys.append(m)
                noise.append(v)
                us.append(u)
                gs.append(g)
                on.append(visited)
        g += 1
    return Targets(np.asarray(rows, dtype=np.float32), np.asarray(ys), np.asarray(noise), np.asarray(us), np.asarray(gs, dtype=np.int64), np.asarray(on, dtype=bool), n)


def teacher_targets(rs: Ruleset, agents, seeds, n: int, features: Callable, exhaustion: str = "hand", off_path: bool = True, workers: int | None = None, rollout_agents=None) -> Targets:
    """Play one game per seed (`agents` = (seat 0, seat 1)) and value, by `n` rollouts each, every
    decision state and (`off_path`) two siblings AIVAT also asks about: the child of one non-taken
    action of the support and another chance outcome of the taken action when it draws (both from a
    determinization of the state, so drawn from the true chance model). `features(state)` must be a
    module-level function (it crosses processes). Rollout rngs are keyed on the seed and the state's
    label, so the targets do not depend on `workers`.

    `rollout_agents` plays the **rollouts** when the profile that generates the *states* is too
    expensive to also play them out. A game contributes ~3 states per decision and each is valued
    by `n` rollouts, so a 39-decision full-UNO game costs ~230 rollouts against one played game:
    the rollouts, not the games, are the budget. Measured single-core on `FULL_UNO`, a PIMC
    w = 32 game is **469x** a rulebot game (543 ms against 1.16 ms), which puts 20,000 fit games
    with PIMC in the rollouts at ~43 hours on eight cores against ~30 minutes with rulebots there.

    Passing it makes V estimate the value of the state **under the rollout profile**, on the state
    distribution of the playing profile. That is the right trade when the complaint is
    distributional -- an "off-profile" leaf network is one whose R^2 was measured on states it was
    never fitted on -- and the wrong one if the continuation itself is what is being estimated. It
    is a different estimand, so a net fitted this way is labelled with both profiles.
    """
    k = (os.cpu_count() or 1) * 32 if workers is None or workers > 1 else 1  # small chunks: full-UNO game lengths are heavy-tailed
    parts = pmap(_teacher_chunk, [(rs, agents, c, n, features, exhaustion, off_path, rollout_agents) for c in chunks(seeds, k)], workers)
    return Targets.concat([p for p in parts if len(p.y)])
