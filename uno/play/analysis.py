"""Rollout values for the play demo: the reference the critic failed to replace.

Phase 0 (`uno.play.calibration`, docs/rl/results) found the critics of the shipped
checkpoints miscalibrated as win probabilities -- by more than the declared 0.05
tolerance even after a per-opponent Platt fit -- so the demo shows **rollout values**
instead, on demand.

`rollout_values` answers, for one seat at one decision: *if the model played this
seat from here against the bot's real policy, how often would it win?* Only what the
seat does not know is resampled: `uno.eval.search.determinize_l1` splits the unseen
cards uniformly between the opponent's hand and the pile, which on free-play rules
(FULL_UNO) is exactly the deal model's conditional given the public history, before
any inference from the opponent's choices. The seat's L1 observations are unchanged
by that resampling, so a copy of its model's evidence and recurrent memory stays
consistent. The bot's copy is consistent only if its inputs are public (every rulebot
and every feed-forward export); a recurrent bot's memory also saw its real private
draws, which the resampled world contradicts.

All worlds of a query run in lockstep: each step groups the worlds by who is to move
and evaluates every neural player of one actor in a single batch.
"""

from __future__ import annotations

import random

import numpy as np

from uno.game.engine import State, returns, step
from uno.game.observe import Level, observe
from uno.play.opponents import NeuralPlayer, Player, batch_policy


def _sample(probs: dict[int, float], u: float) -> int:
    acc, last = 0.0, None
    for a, p in probs.items():
        acc += p
        last = a
        if u < acc:
            return a
    return last


def rollout_values(
    points: list[tuple[State, int, tuple[Player, Player]]], n: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Score `(return + 1) / 2` of `seat` from each point, mean and SE over `n` worlds.

    A point is `(state, seat, players)`: `players[s]` plays seat `s`, snapshotted at
    that state (they are copied per world, never advanced).
    """
    from uno.eval.search import determinize_l1

    rng = random.Random(seed)
    worlds: list[list] = []  # [state, point index, players]
    for i, (state, seat, players) in enumerate(points):
        obs = observe(state, seat, Level.L1)
        for _ in range(n):
            worlds.append([determinize_l1(state.ruleset, obs, rng), i, [players[0].copy(), players[1].copy()]])
    scores = np.zeros(len(worlds))
    active = list(range(len(worlds)))
    urng = np.random.default_rng(seed)
    while active:
        still = []
        neural: dict[int, list[int]] = {}
        for w in active:
            s, _, players = worlds[w]
            if s.terminal:
                seat = points[worlds[w][1]][1]
                scores[w] = (returns(s)[seat] + 1.0) / 2.0
                continue
            still.append(w)
            p = players[s.player]
            if isinstance(p, NeuralPlayer):
                neural.setdefault(id(p.actor), []).append(w)
        decided: dict[int, dict[int, float]] = {}
        for ws in neural.values():
            probs = batch_policy([worlds[w][2][worlds[w][0].player] for w in ws], [worlds[w][0] for w in ws])
            decided.update(zip(ws, probs))
        us = urng.random(len(still))
        for u, w in zip(us, still):
            s, _, players = worlds[w]
            probs = decided.get(w) or players[s.player].policy(s)
            worlds[w][0] = step(s, _sample(probs, u), check=False)
        active = still
    per = scores.reshape(len(points), n)
    return per.mean(1), per.std(1, ddof=1) / np.sqrt(n) if n > 1 else np.zeros(len(points))


__all__ = ["rollout_values"]
