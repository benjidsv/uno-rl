"""Step 9 speed-ups: moving a strategy between keyings by key, and the tabulation cache."""

from __future__ import annotations

import numpy as np
import pytest

from tests.strategies import SMALL
from uno import store
from uno.agents.baseline import Heuristic, PlayFirstRandom
from uno.compiled.cfr import cfr
from uno.compiled.solvers import evaluate
from uno.compiled.strategies import tabulate, transfer
from uno.compiled.tree import compile_tree
from uno.game.observe import Level
from uno.game.rules import TOY

L1 = Level.L1


def test_transfer_between_lazy_and_world_trees():
    lazy = compile_tree(SMALL, (L1,))
    world = compile_tree(SMALL, (L1,), world=True)
    for agent in (PlayFirstRandom(SMALL), Heuristic(SMALL)):
        assert np.array_equal(transfer(lazy, world, L1, tabulate(lazy, L1, agent)), tabulate(world, L1, agent))
        assert np.array_equal(transfer(world, lazy, L1, tabulate(world, L1, agent)), tabulate(lazy, L1, agent))
    res = cfr(lazy, iterations=20, log_at=(20,))  # a mixed strategy no agent produces
    rnd = tabulate(lazy, L1, PlayFirstRandom(SMALL))
    moved, rnd_w = transfer(lazy, world, L1, res.avg), transfer(lazy, world, L1, rnd)
    assert np.allclose(transfer(world, lazy, L1, moved), res.avg)
    assert abs(evaluate(lazy, (L1, res.avg), (L1, rnd))[0] - evaluate(world, (L1, moved), (L1, rnd_w))[0]) < 1e-9
    with pytest.raises(ValueError):
        transfer(lazy, compile_tree(TOY, (L1,)), L1, res.avg)


def test_tabulation_cache(tmp_path):
    tree = compile_tree(SMALL, (L1,))
    rnd = PlayFirstRandom(SMALL)
    first = store.load_or_tabulate(tree, L1, rnd, cache=tmp_path)
    assert np.array_equal(first, tabulate(tree, L1, rnd))
    path = store.tabulation_path(tree, L1, rnd.name, tmp_path)
    assert path.exists() and store.is_current(path)

    class Refuses:
        name = "random"

        def policy(self, obs, legal):
            raise AssertionError("the cached tabulation should have been used")

    assert np.array_equal(store.load_or_tabulate(tree, L1, Refuses(), cache=tmp_path), first)
    # a file written for another keying is tabulated again
    np.savez(path, version=np.array(store.CACHE_VERSION), fingerprint=np.array("stale"), sigma=first)
    with pytest.raises(AssertionError):
        store.load_or_tabulate(tree, L1, Refuses(), cache=tmp_path)
    assert np.array_equal(store.load_or_tabulate(tree, L1, rnd, cache=tmp_path), first)
    assert store.tabulation_path(compile_tree(SMALL, (L1,), world=True), L1, rnd.name, tmp_path) != path
