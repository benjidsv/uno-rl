"""The value network (uno/eval/value_net.py, step 9.5c): symmetric features that agree between a
state and its L2 observation, a fit that learns, batch-independent float64 inference, and the
played path equal to the tree path, as for the critic."""

from __future__ import annotations

import math
import random
from functools import cache

import numpy as np

from tests.eval.test_pimc import small_profile
from tests.game.test_draw_outcomes import RESH, games
from tests.strategies import SMALL
from uno.compiled.aivat import TreeValues, critic_values
from uno.compiled.lookup import build
from uno.eval.critic import CriticValues
from uno.eval.pimc import teacher_targets
from uno.eval.play import play_game
from uno.eval.value_net import ValueNet, fit_value_net, layout, observation_features, state_features
from uno.experiments.pairing import DUPS
from uno.game import rules
from uno.game.observe import Level, observe
from uno.game.symmetry import Symmetry


def invariant(rs, row):
    """The global block and the colour blocks as a multiset: renaming colours permutes the blocks."""
    g, b, c = layout(rs)
    return row[:g], sorted(tuple(row[g + i * b : g + (i + 1) * b]) for i in range(c))


def test_features_are_symmetric_and_agree_with_the_l2_observation():
    for rs, elements in ((SMALL, None), (DUPS, None), (RESH, None), (rules.MID_25, 12)):
        sym = Symmetry(rs)
        gs = range(sym.order) if elements is None else random.Random(0).sample(range(sym.order), elements)
        n = 0
        for s, _, _ in games(rs, 25):
            f = state_features(s)
            assert f == observation_features(rs, observe(s, s.player, Level.L2))
            for g in gs:
                assert invariant(rs, f) == invariant(rs, state_features(sym.state(g, s)))
            n += 1
        assert n > 50


def test_value_is_symmetric():
    net, _ = small_net()
    sym = Symmetry(SMALL)
    for s, _, _ in games(SMALL, 20):
        vals = net.values([state_features(sym.state(g, s)) for g in range(sym.order)])
        assert np.ptp(vals) <= 1e-12


@cache
def small_net():
    _, agents, _ = small_profile()
    t = teacher_targets(SMALL, agents, range(10_000_000, 10_000_600), 8, state_features, workers=1)
    return fit_value_net(SMALL, t.x, t.y, t.game, t.noise, width=32, max_epochs=40, patience=5, threads=1)


def test_fit_learns_and_inference_is_batch_independent(tmp_path):
    net, report = small_net()
    assert report.r2_values > 0.5, str(report)
    rows = [state_features(s) for s, _, _ in games(SMALL, 20)]
    batch = net.values(rows)
    alone = np.array([net.values([r])[0] for r in rows])
    assert np.max(np.abs(batch - alone)) <= 1e-12
    net.save(tmp_path / "net.pt")
    assert np.array_equal(ValueNet.load(tmp_path / "net.pt", SMALL).values(rows), batch)


def test_played_path_equals_tree_path():
    tree, agents, _ = small_profile()
    idx = build(tree, Level.L2)
    net, _ = small_net()
    nu = critic_values(tree, net)
    tv = TreeValues(idx, nu)
    cv = CriticValues(net, deal_mean=tv.deal_mean)
    n = 0
    for seed in range(150):
        t = play_game(SMALL, seed, agents if seed % 2 else agents[::-1])
        for a, b in zip(tv.records(t), cv.records(t)):
            assert math.isclose(a.node, b.node, abs_tol=1e-9) and math.isclose(a.child, b.child, abs_tol=1e-9)
            assert a.alpha.keys() == b.alpha.keys()
            assert all(math.isclose(a.alpha[k], b.alpha[k], abs_tol=1e-9) for k in a.alpha)
            n += 1
    assert n > 300
