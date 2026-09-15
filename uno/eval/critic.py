"""A fitted value function for AIVAT on rulesets with no exact tree (step 9.5b, docs/spec.md
section 23): colour- and rank-symmetric features of a position, a fractional logistic
regression fitted by IRLS (numpy only), and `CriticValues`, the value model that feeds it
to `uno.eval.aivat` through the chance model `uno.game.chance.draw_outcomes`.

**What the critic may see.** Both hands, the top card and colour, who moves, the phase and
the pile size: an L2-state value, never the pile order (`uno.eval.aivat`'s module
docstring). The features are functions of the hands as multisets, so every child the
chance model builds with an arbitrary pile order gets the value of the real one.

**Frame.** Seat-0 return; seat-0-frame hand blocks, plus the same blocks multiplied by
"seat 0 moves", so the model can tell a hand to move from a hand waiting. One critic per
profile (row, column, seat): AIVAT wants the value under the policies being played.
Fitted on `uno.eval.play.FIT_SEEDS`, never on the seeds it evaluates.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass

import numpy as np

from uno.eval.aivat import StepRecord, rescored_return, terminal_return
from uno.eval.play import Trace, play_game
from uno.game.chance import draw_count, draw_outcomes
from uno.game.engine import Phase, State, is_playable
from uno.game.observe import Observation
from uno.game.rules import Draw, Kind, Ruleset

N_ONEHOT = 8  # hand sizes 1..7, and 8 or more
_SLOT = {Kind.NUMBER: 0, Kind.SKIP: 1, Kind.REVERSE: 1, Kind.DRAW_TWO: 2, Kind.WILD: 3, Kind.WILD_DRAW_FOUR: 4}


def _block(rs: Ruleset, hand: tuple[int, ...], top: int, color: int) -> list[float]:
    types = rs.card_types
    kinds = [0.0] * 5
    colors: Counter = Counter()
    playable = 0
    for c in hand:
        ct = types[c]
        kinds[_SLOT[ct.kind]] += 1
        if ct.color is not None:
            colors[ct.color] += 1
        playable += is_playable(rs, c, top, color)
    n = len(hand)
    onehot = [0.0] * N_ONEHOT
    if n:
        onehot[min(n, N_ONEHOT) - 1] = 1.0
    return [float(n), *onehot, *kinds, float(len(colors)), float(max(colors.values(), default=0)), float(playable), float(playable > 0), float(colors[color])]


def position_features(rs: Ruleset, hands, top: int, color: int, player: int, phase: Phase, pile: int) -> list[float]:
    b0, b1 = _block(rs, hands[0], top, color), _block(rs, hands[1], top, color)
    m = 1.0 if player == 0 else 0.0
    after = 1.0 if phase is Phase.AFTER_DRAW else 0.0
    base = [*b0, *b1, m, after, pile / rs.deck_size, float(pile == 0), b0[0] - b1[0]]
    return base + [m * x for x in (*b0, *b1)] + [m * after]


def state_features(s: State) -> list[float]:
    return position_features(s.ruleset, s.hands, s.top, s.current_color, s.player, s.phase, len(s.draw_pile))


def observation_features(rs: Ruleset, obs: Observation) -> list[float]:
    """From an L2 observation (a compiled tree's L2 key, `uno.game.observe.observation_from_key`)."""
    hands = [None, None]
    hands[obs.seat], hands[obs.seat ^ 1] = obs.hand, obs.opp_hand
    return position_features(rs, hands, obs.top, obs.current_color, obs.player, obs.phase, obs.pile_size)


# -- fractional logistic regression --------------------------------------------------


def fit_logistic(x: np.ndarray, y: np.ndarray, lam: float, iterations: int = 50, tol: float = 1e-10, chunk: int = 100_000) -> np.ndarray:
    """Weights (bias last) minimising the cross-entropy of `sigmoid(x w + b)` against `y` in
    [0, 1] plus `lam / 2 |w|^2` (the bias unpenalised), by Newton's method (IRLS). Gradient
    and Hessian are accumulated over chunks of rows, so a million-state fit stays small."""
    n, d = x.shape
    w = np.zeros(d + 1)
    reg = np.full(d + 1, lam)
    reg[-1] = 0.0
    for _ in range(iterations):
        g = reg * w
        h = np.diag(reg) + 1e-9 * np.eye(d + 1)
        for i in range(0, n, chunk):
            xb = np.hstack([np.asarray(x[i : i + chunk], dtype=np.float64), np.ones((min(chunk, n - i), 1))])
            p = 1.0 / (1.0 + np.exp(-np.clip(xb @ w, -30, 30)))
            g += xb.T @ (p - y[i : i + chunk])
            h += (xb * (p * (1 - p))[:, None]).T @ xb
        step = np.linalg.solve(h, g)
        w -= step
        if float(np.max(np.abs(step))) < tol:
            break
    return w


@dataclass
class Critic:
    rs: Ruleset
    mean: np.ndarray
    scale: np.ndarray
    w: np.ndarray  # bias last
    lam: float

    def values(self, rows: list[list[float]] | np.ndarray) -> np.ndarray:
        """Seat-0 return in [-1, 1] per feature row."""
        x = (np.asarray(rows, dtype=np.float64) - self.mean) / self.scale
        z = x @ self.w[:-1] + self.w[-1]
        return 2.0 / (1.0 + np.exp(-np.clip(z, -30, 30))) - 1.0

    def value(self, s: State) -> float:
        return float(self.values([state_features(s)])[0])


@dataclass(frozen=True)
class Sample:
    """Feature rows of the decisions of played games, with the game's final seat-0 return."""

    x: np.ndarray  # (states, features) float32
    u: np.ndarray  # (states,) seat-0 return of the state's game
    game: np.ndarray  # (states,) index of the game
    games: int


def collect(rs: Ruleset, agents, seeds, exhaustion: str = "hand") -> Sample:
    """Play one game per seed (`agents` = (seat 0, seat 1)) and keep every decision's features."""
    rows: list[list[float]] = []
    us: list[float] = []
    gs: list[int] = []
    g = 0
    for seed in seeds:
        try:
            t = play_game(rs, seed, agents)
        except ValueError:
            continue
        u = rescored_return(t, exhaustion)
        for st in t.steps:
            rows.append(state_features(st.state))
            us.append(u)
            gs.append(g)
        g += 1
    return Sample(np.asarray(rows, dtype=np.float32), np.asarray(us), np.asarray(gs), g)


def r2(values: np.ndarray, u: np.ndarray) -> float:
    """Outcome R^2 over visited decisions: `1 - sum (u - V)^2 / sum (u - mean u)^2`
    (`uno.compiled.aivat.outcome_r2` is the exact counterpart)."""
    return 1.0 - float(np.sum((u - values) ** 2) / np.sum((u - u.mean()) ** 2))


@dataclass(frozen=True)
class FitReport:
    games: int
    states: int
    lam: float
    holdout_r2: dict[float, float]  # per ridge strength, on the last fifth of the games


def fit_critic(rs: Ruleset, sample: Sample, lams: tuple[float, ...] = (0.1, 10.0, 1000.0)) -> tuple[Critic, FitReport]:
    """Standardise, pick the ridge strength by outcome R^2 on the last fifth of the games (all
    from the fitting seeds), refit on everything."""
    x = sample.x
    mean = x.mean(axis=0, dtype=np.float64)
    scale = x.std(axis=0, dtype=np.float64)
    scale[scale == 0] = 1.0
    xs = ((x - mean) / scale).astype(np.float32)
    y = (sample.u + 1.0) / 2.0
    train = sample.game < int(0.8 * sample.games)
    scores = {}
    for lam in lams:
        c = Critic(rs, mean, scale, fit_logistic(xs[train], y[train], lam), lam)
        scores[lam] = r2(c.values(x[~train]), sample.u[~train]) if (~train).any() else float("nan")
    best = max(lams, key=lambda lam: scores[lam])
    return Critic(rs, mean, scale, fit_logistic(xs, y, best), best), FitReport(sample.games, len(sample.u), best, scores)


def fit(rs: Ruleset, agents, seeds, exhaustion: str = "hand", lams: tuple[float, ...] = (0.1, 10.0, 1000.0)) -> tuple[Critic, FitReport]:
    return fit_critic(rs, collect(rs, agents, seeds, exhaustion), lams)


# -- the value model ------------------------------------------------------------------


def event(parent: State, action: int, child: State) -> str:
    """The chance event between two states of a game (`uno.eval.aivat.StepRecord.event`)."""
    if draw_count(parent, action) == 0:
        return ""
    if child.n_reshuffles != parent.n_reshuffles:
        return "reshuffle draw"
    return "voluntary draw" if isinstance(parent.ruleset.actions[action], Draw) else "penalty draw"


class CriticValues:
    """`uno.eval.aivat.ValueModel` for a critic: V at each decision's state and its child, and
    for every action of the support the expectation of V over `draw_outcomes` (enumerated;
    sampled `samples` times from an rng keyed on the seed and the decision, independent of
    the game's own draws, when there are too many ordered draws). Decisions where the chance
    model has no answer (a penalty draw across a reshuffle) get `None` and are skipped.
    `deal_mean`: E over deals of V at the first state, when known (on a toy, from the tree);
    None on full UNO."""

    def __init__(self, critic: Critic, exhaustion: str = "hand", samples: int = 64, deal_mean: float | None = None) -> None:
        self.critic, self.exhaustion, self.samples, self.deal_mean = critic, exhaustion, samples, deal_mean

    def records(self, trace: Trace) -> list[StepRecord]:
        steps = trace.steps
        rows: list[list[float]] = []

        def row(s: State) -> int:
            rows.append(state_features(s))
            return len(rows) - 1

        plans = []
        for i, st in enumerate(steps):
            s = st.state
            rng = random.Random(f"{s.seed}:aivat:{i}")
            alpha: dict[int, list[tuple[int | None, float]] | None] = {}
            for a in st.probs:
                outs = draw_outcomes(s, a, rng, self.samples)
                if outs is None:
                    alpha[a] = None
                else:
                    alpha[a] = [(None, p * terminal_return(s, a, c, self.exhaustion)) if c.terminal else (row(c), p) for c, p in outs]
            nxt = steps[i + 1].state if i + 1 < len(steps) else trace.final
            plans.append((row(s), alpha, None if nxt.terminal else row(nxt), event(s, st.action, nxt)))
        v = self.critic.values(rows) if rows else np.zeros(0)
        u = rescored_return(trace, self.exhaustion)
        out = []
        for st, (node, alpha, child, ev) in zip(steps, plans):
            al = {a: None if items is None else math.fsum(x if j is None else x * v[j] for j, x in items) for a, items in alpha.items()}
            out.append(StepRecord(st.state.player, st.probs, st.action, float(v[node]), al, u if child is None else float(v[child]), ev))
        return out
