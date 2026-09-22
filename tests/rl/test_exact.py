"""Stage 5 machinery: exact tabulation, and the exact gradient against finite differences.

The gate runs themselves live in `uno.experiments.rl_toy`; these are the bounded
regression checks that the exact path is what it claims to be. The finite-difference
check shares no code with `uno.rl.exact.gradient` beyond the tree's own evaluation,
so it is the definition of the derivative rather than a rearrangement of it.

These tests need the compiled `TOY` quotient tree, which `uno.store` caches.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.known import HEURISTIC_TOY
from tests.rl.helpers import SMALL_WIDTH
from uno import store
from uno.agents.baseline import Heuristic, PlayFirstRandom
from uno.game.observe import Level
from uno.game.rules import TOY
from uno.rl.eval import exact
from uno.rl.env import features as F
from uno.rl.model import Actor
from uno.rl.algo.policy import ActorPolicy, NeuralAgent


@pytest.fixture(scope="module")
def heuristic_sigma():
    return store.load_or_tabulate(exact.tree_for(TOY), Level.L1, Heuristic(TOY))


@pytest.fixture(scope="module")
def actor():
    torch.manual_seed(11)
    return Actor(F.spec(TOY, F.ARM_B), SMALL_WIDTH)


def test_best_response_ceilings_reproduce_the_pinned_toy_values(heuristic_sigma):
    """The exact ceiling the gate measures against is the pinned step-8.6 value."""
    for seat in (0, 1):
        assert abs(exact.br_score(TOY, heuristic_sigma, seat) - HEURISTIC_TOY[(seat, Level.L1)]) < 5e-5


def test_the_tabulated_network_is_symmetric_and_matches_the_agent_set_by_set(actor):
    sigma = exact.strategy(TOY, F.spec(TOY, F.ARM_B), actor, seat=0)
    k = exact.tree_for(TOY).keyings[Level.L1]
    assert k.is_symmetric(sigma, exact.SYMMETRY_TOL)
    agent = NeuralAgent(ActorPolicy(F.spec(TOY, F.ARM_B), actor), "test")
    from uno.game.observe import observation_from_key

    checked = 0
    for info in range(k.n_info):
        if int(k.info_player[info]) != 0:
            continue
        obs = observation_from_key(TOY, Level.L1, k.key(info))
        legal = k.actions(info)
        probs = agent.policy(obs, legal)
        lo = int(k.info_start[info])
        for j, a in enumerate(legal):
            assert abs(probs[a] - sigma[lo + j]) < 1e-5
        checked += 1
        if checked >= 40:
            break
    assert checked == 40


def test_exact_gradient_matches_a_finite_difference_of_the_exact_value(actor, heuristic_sigma):
    sp = F.spec(TOY, F.ARM_B)
    g = exact.gradient(TOY, sp, actor, heuristic_sigma, 0)
    flat = g.flat
    params = list(actor.parameters())
    rng = np.random.default_rng(0)
    # a handful of coordinates, chosen where the exact gradient is not ~0 so the
    # comparison can fail: a check on a zero gradient proves nothing
    sizes = [p.numel() for p in params]
    offsets = np.cumsum([0] + sizes)
    order = np.argsort(-flat.abs().numpy())[:200]
    picks = rng.choice(order, size=5, replace=False)
    h = 1e-4
    for idx in picks:
        which = int(np.searchsorted(offsets, idx, side="right") - 1)
        inside = int(idx - offsets[which])
        values = []
        for step in (+h, -h):
            with torch.no_grad():
                params[which].view(-1)[inside] += step
            sigma = exact.strategy(TOY, sp, actor, seat=0)
            values.append(2.0 * exact.value(TOY, sigma, heuristic_sigma, 0) - 1.0)  # in return units
            with torch.no_grad():
                params[which].view(-1)[inside] -= step
        fd = (values[0] - values[1]) / (2 * h)
        assert abs(fd - float(flat[idx])) <= 2e-3 * max(1.0, abs(fd)), f"coordinate {idx}: finite difference {fd}, exact {float(flat[idx])}"


def test_sampled_gradient_is_per_game_and_returns_the_decision_mean(actor):
    from uno.rl.algo.collect import collect
    from uno.rl.core.manifest import DealSchedule, build

    opponents = {a.name: a for a in (PlayFirstRandom(TOY),)}
    manifest, _ = build(0, list(opponents), DealSchedule(0), 40, 0, seats=(0,))
    rollout, _ = collect(manifest, ActorPolicy(F.spec(TOY, F.ARM_B), actor), F.spec(TOY, F.ARM_CRITIC), opponents, 0, 0, live_games=16)
    per_game, mean = exact.sampled_gradient(actor, rollout)
    assert per_game.shape[0] == rollout.n_games
    counts = torch.from_numpy(rollout.learner_decisions.astype(np.float32))
    expected = (per_game * counts[:, None]).sum(0) / counts.sum()
    assert torch.allclose(expected, mean)
    rows = exact.agreement(per_game, per_game.mean(0))
    assert rows[0].direction == "exact" and abs(rows[0].z) < 1e-4  # compared with itself
