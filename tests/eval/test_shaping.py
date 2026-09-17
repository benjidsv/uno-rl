"""The shaping potentials (step 10a.2, item A4): the identities that make a potential
AIVAT's deal correction, on the 7-card SMALL ruleset.

The measurements are `uv run uno shaping`; these are the cheap invariants."""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest

from tests.strategies import SMALL
from uno.agents.baseline import Heuristic, PlayFirstRandom
from uno.compiled.aivat import TreeValues, aivat_moments, arc_expectation, exact_values
from uno.compiled.lookup import build
from uno.compiled.strategies import tabulate, transfer
from uno.compiled.tree import compile_tree
from uno.eval.aivat import NONE, Corrections, trace_terms
from uno.eval.play import play_game
from uno.eval.shaping import deal_term, potential, set_potential, values
from uno.game.observe import Level

L1, L2, L3 = Level.L1, Level.L2, Level.L3
NO_DEAL = Corrections(chance=True, row=True, column=True, deal=False)


@cache
def setup():
    lazy = compile_tree(SMALL, (L1, L2), symmetric=True)
    world = compile_tree(SMALL, (L1, L3), symmetric=True)
    heu, rnd = Heuristic(SMALL), PlayFirstRandom(SMALL)
    h = (L1, tabulate(lazy, L1, heu))
    r = (L1, tabulate(lazy, L1, rnd))
    hw = (L1, transfer(lazy, world, L1, h[1]))
    rw = (L1, transfer(lazy, world, L1, r[1]))
    return lazy, world, (h, r), (hw, rw), heu, rnd


def test_the_potential_is_zero_at_the_terminals_and_the_values_are_not():
    """The convention the telescoping identity rests on: with no discount, a game's shaping
    terms sum to `Phi(terminal) - Phi(s0)`, so `Phi(terminal) = 0` is what makes a potential
    contribute exactly `-Phi(s0)` to every return."""
    lazy, world, lz, wd, *_ = setup()
    for tree, pair in ((lazy, lz), (world, wd)):
        phi, v = potential(tree, *pair), values(tree, *pair)
        assert np.all(phi[tree.terminals] == 0.0)
        assert np.abs(v[tree.terminals]).max() == pytest.approx(1.0)
        # and they are the same array away from the terminals
        assert np.array_equal(np.delete(v, tree.terminals), np.delete(phi, tree.terminals))
    phi1 = potential(lazy, *lz, L1)
    assert np.all(phi1[lazy.terminals] == 0.0)


def test_the_deal_term_is_centred():
    lazy, _, lz, *_ = setup()
    for level in (L1, None):
        dt = deal_term(lazy, potential(lazy, *lz, level))
        assert abs(dt.mean) < 1e-12
        assert dt.roots == len(lazy.roots)


def test_the_l2_deal_term_is_aivats_residual_after_every_other_correction():
    """With an exact value function and every correction but the deal applied, the estimator
    *is* the deal term, so its variance is the potential's."""
    lazy, _, lz, *_ = setup()
    for seat in (0, 1):
        pair = lz if seat == 0 else lz[::-1]
        nu = exact_values(lazy, *pair)
        nodeal = aivat_moments(lazy, *pair, nu, NO_DEAL, seat).var
        assert deal_term(lazy, potential(lazy, *pair)).var == pytest.approx(nodeal, abs=1e-12)


def test_more_information_in_the_potential_removes_more_deal_luck():
    """L1 is a coarsening of L2 and L2 of L3 -- each is a conditional expectation of the
    next -- so the tower property orders the variances. A violation means a potential is not
    the conditional expectation it claims to be."""
    lazy, world, lz, wd, *_ = setup()
    for seat in (0, 1):
        pl = lz if seat == 0 else lz[::-1]
        pw = wd if seat == 0 else wd[::-1]
        v1 = deal_term(lazy, potential(lazy, *pl, L1)).var
        v2 = deal_term(lazy, potential(lazy, *pl)).var
        v3 = deal_term(world, potential(world, *pw)).var
        assert v1 <= v2 + 1e-12, (seat, v1, v2)
        assert v2 <= v3 + 1e-12, (seat, v2, v3)


def test_the_deal_potential_is_aivats_deal_correction_game_by_game():
    """A4's headline check, and the reason the item exists: the control variate and the
    shaping term are one object, not two similar ones -- so array-equal, not close."""
    lazy, _, lz, _, heu, rnd = setup()
    idx = build(lazy, L2)
    for seat in (0, 1):
        pair = lz if seat == 0 else lz[::-1]
        nu = exact_values(lazy, *pair)
        phi = potential(lazy, *pair)
        mean_phi = float(arc_expectation(lazy, phi)[0])
        model = TreeValues(idx, nu)
        agents = (heu, rnd) if seat == 0 else (rnd, heu)
        mine, theirs = [], []
        for seed in range(150):
            try:
                trace = play_game(SMALL, seed, agents)
            except ValueError:
                continue
            mine.append(phi[idx.node(trace.steps[0].state)] - mean_phi)
            theirs.append(trace_terms(model, seed, trace).deal)
        assert mine and np.array_equal(np.array(mine), np.array(theirs))


def test_the_set_potential_is_the_reach_weighted_mean_over_the_set():
    lazy, _, lz, *_ = setup()
    k = lazy.keyings[L1]
    sp = set_potential(lazy, *lz, L1)
    assert len(sp) == k.n_info
    # it agrees with the node-level L1 potential wherever the profile reaches the set
    phi = potential(lazy, *lz, L1)
    dec = np.nonzero(k.node_info >= 0)[0]
    reached = np.nonzero(sp != 0)[0]
    for i in reached[:50]:
        nodes = dec[k.node_info[dec] == i]
        assert np.allclose(phi[nodes], sp[i])


def test_a_potential_at_a_level_the_node_cannot_carry_is_refused():
    lazy, *_ = setup()
    _, _, lz, *_ = setup()
    with pytest.raises(ValueError, match="L1 only"):
        potential(lazy, *lz, L2)
