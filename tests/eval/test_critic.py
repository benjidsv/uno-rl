"""The AIVAT critic (uno/eval/critic.py, step 9.5b): IRLS, symmetric features, and the played
path (`CriticValues` through `draw_outcomes`) against the tree path (`critic_values` +
`TreeValues`) for the same critic."""

from __future__ import annotations

import math
from functools import cache

import numpy as np

from tests.game.test_draw_outcomes import RESH, games
from tests.strategies import SMALL
from uno.agents.baseline import Heuristic, PlayFirstRandom
from uno.compiled.aivat import TreeValues, critic_values
from uno.compiled.lookup import build
from uno.compiled.tree import compile_tree
from uno.eval.critic import CriticValues, fit, fit_logistic, observation_features, state_features
from uno.eval.play import play_game
from uno.experiments.pairing import DUPS
from uno.game.observe import Level, observe
from uno.game.symmetry import Symmetry


def test_fit_logistic_recovers_coefficients():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4000, 3))
    w = np.array([0.8, -1.5, 0.3, 0.2])
    y = 1.0 / (1.0 + np.exp(-(x @ w[:3] + w[3])))  # fractional targets: the exact probabilities
    assert np.allclose(fit_logistic(x, y, lam=0.0), w, atol=1e-8)


def test_features_are_symmetric_and_agree_with_the_l2_observation():
    for rs in (SMALL, DUPS, RESH):
        sym = Symmetry(rs)
        n = 0
        for s, _, _ in games(rs, 25):
            f = state_features(s)
            assert f == observation_features(rs, observe(s, s.player, Level.L2))
            for g in range(sym.order):
                assert f == state_features(sym.state(g, s))
            n += 1
        assert n > 50


@cache
def small_critic():
    heu, rnd = Heuristic(SMALL), PlayFirstRandom(SMALL)
    critic, _ = fit(SMALL, (heu, rnd), range(10_000_000, 10_000_400), lams=(1.0,))
    return critic


def test_played_path_equals_tree_path():
    tree = compile_tree(SMALL, (Level.L1, Level.L2), symmetric=True)
    idx = build(tree, Level.L2)
    critic = small_critic()
    heu, rnd = Heuristic(SMALL), PlayFirstRandom(SMALL)
    for exhaustion in ("hand", "stuck"):
        nu = critic_values(tree, critic, exhaustion)
        tv = TreeValues(idx, nu, exhaustion)
        cv = CriticValues(critic, exhaustion, deal_mean=tv.deal_mean)
        n = 0
        for seed in range(150):
            t = play_game(SMALL, seed, (heu, rnd) if seed % 2 else (rnd, heu))
            for a, b in zip(tv.records(t), cv.records(t)):
                assert math.isclose(a.node, b.node, abs_tol=1e-12) and math.isclose(a.child, b.child, abs_tol=1e-12)
                assert a.alpha.keys() == b.alpha.keys()
                assert all(math.isclose(a.alpha[k], b.alpha[k], abs_tol=1e-12) for k in a.alpha)
                n += 1
        assert n > 300
