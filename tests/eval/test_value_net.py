"""The value network (uno/eval/value_net.py, step 9.5c): symmetric features that agree between a
state and its L2 observation, a fit that learns, batch-independent float64 inference, and the
played path equal to the tree path, as for the critic."""

from __future__ import annotations

import math
import random
from functools import cache

import numpy as np
import pytest

from tests.eval.test_pimc import small_profile
from tests.game.test_draw_outcomes import RESH, games
from tests.strategies import SMALL
from uno.compiled.aivat import TreeValues, critic_values
from uno.compiled.lookup import build
from uno.eval.critic import CriticValues
from uno.eval.pimc import teacher_targets
from uno.eval.play import play_game
from uno.eval.value_net import ValueNet, batch_state_features, fit_value_net, layout, observation_features, state_features
from uno.experiments.pairing import DUPS
from uno.game import rules
from uno.game.engine import legal_actions, new_game, step
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


def test_at_a_fixed_batch_size_a_row_does_not_depend_on_its_neighbours():
    """The exact invariant `uno.eval.search.batched_search_q` rests on.

    The test above pins batch-independence to 1e-12, which is the right tolerance for it: float64
    inference is *not* bit-exact across row counts (batch 1 against batch 256 differ by ~1.7e-16,
    one ulp, because BLAS picks its kernel by the number of rows). What is exact, and what the
    search needs, is that at a **fixed** number of rows a row's value depends on that row alone --
    otherwise a decision's value would depend on which worlds happened to be sampled alongside it,
    and the agent would stop being the deterministic function of the information state that
    AIVAT's row correction assumes (docs/spec.md section 25).
    """
    net, _ = small_net()
    rows = [state_features(s) for s, _, _ in games(SMALL, 20)][:40]
    assert len(rows) == 40
    target, rest = rows[0], rows[1:]
    rng = random.Random(0)
    seen = set()
    for _ in range(5):
        others = rng.sample(rest, 15)
        seen.add(float(net.values([target, *others])[0]))
    for position in (0, 1, 8, 15):
        batch = list(rng.sample(rest, 15))
        batch.insert(position, target)
        seen.add(float(net.values(batch)[position]))
    assert len(seen) == 1, f"one row took {len(seen)} different values at a fixed batch size: {seen}"


@pytest.mark.parametrize("name", ("FULL_UNO", "TOY", "TOY2_SKIP", "TOY_FULL", "MID_12"))
def test_batch_features_equal_the_row_at_a_time_ones_bit_for_bit(name):
    """`batch_position_features` exists only to be faster, so it must produce the same array, not
    a close one -- `np.array_equal`, never `allclose`. Anything less and the network sees different
    inputs, the agent plays different games, and the rows on the ladder stop being comparable.

    Parametrised over rulesets because the tables are built per ruleset and the toys differ in the
    two dimensions the vectorisation folds into array shapes: the number of colours and the number
    of ranks.
    """
    rs = getattr(rules, name)
    states = []
    for seed in range(12):
        try:
            s = new_game(rs, seed)
        except ValueError:
            continue
        rng = random.Random(seed)
        while not s.terminal and len(states) < 400:
            states.append(s)
            s = step(s, rng.choice(legal_actions(s)))
    assert len(states) > 30
    assert np.array_equal(batch_state_features(states), np.array([state_features(x) for x in states]))


def test_batch_features_handle_a_wild_top_and_a_wild_drawn_card():
    """The trap the vectorisation has to dodge: a wild has `col == -1`, and `eye[-1]` is not "no
    colour", it is the *last* colour. Both `blk[15]` (does this block hold the top card's colour)
    and `blk[16]` (the drawn card's) index by a card's colour, so a wild top or a wild drawn card
    would silently light up the last colour block. The generic test above only catches this if its
    random games happen to leave a wild on top, so this one builds the states deliberately."""
    rs = rules.FULL_UNO
    wild_top, wild_drawn = [], []
    for seed in range(200):
        s = new_game(rs, seed)
        rng = random.Random(seed)
        while not s.terminal:
            if rs.card_types[s.top].is_wild:
                wild_top.append(s)
            if s.drawn is not None and rs.card_types[s.drawn].is_wild:
                wild_drawn.append(s)
            s = step(s, rng.choice(legal_actions(s)))
        if len(wild_top) > 40 and len(wild_drawn) > 5:
            break
    assert wild_top and wild_drawn, "no state with a wild on the board was produced"
    for states, what in ((wild_top, "a wild on top"), (wild_drawn, "a wild drawn card")):
        assert np.array_equal(batch_state_features(states), np.array([state_features(x) for x in states])), what


def test_an_empty_batch_is_refused_rather_than_guessed():
    """The ruleset comes from the batch, so there is nothing to build a zero-row array of."""
    with pytest.raises(ValueError, match="at least one state"):
        batch_state_features([])


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
