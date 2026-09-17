"""Step 9 speed-ups: moving a strategy between keyings by key, and the tabulation cache.
Step 10a: the batched tabulation (item A2)."""

from __future__ import annotations

import numpy as np
import pytest

from tests.strategies import SMALL
from uno import store
from uno.agents.baseline import Heuristic, PlayFirstRandom
from uno.compiled.cfr import cfr
from uno.compiled.solvers import evaluate
from uno.compiled.strategies import tabulate, tabulate_batched, transfer
from uno.compiled.tree import compile_tree
from uno.game.observe import Level
from uno.game.rules import TOY, TOY2, Play

L1 = Level.L1


class BatchedRandom:
    """`PlayFirstRandom` answering a whole batch in `ia` order, fully vectorised:
    uniform over the legal plays, or over everything legal when nothing plays.

    The test double for `tabulate_batched`: a real policy whose row-at-a-time twin
    already exists, so the two paths must agree exactly. It is symmetric, so it also
    exercises the orbit check at the end of both."""

    name = "random"

    def __init__(self, rs) -> None:
        self.rs = rs
        self._is_play = np.array([isinstance(a, Play) for a in rs.actions])

    def policy_batch(self, batch, ia_action, info_start):
        n_ia = len(ia_action)
        is_play = self._is_play[ia_action]
        starts = np.asarray(info_start, dtype=np.int64)
        ends = np.append(starts[1:], n_ia)
        set_of = np.repeat(np.arange(len(starts)), ends - starts)
        has_play = np.bincount(set_of, weights=is_play.astype(float), minlength=len(starts)) > 0
        mask = np.where(has_play[set_of], is_play, True)
        cnt = np.bincount(set_of, weights=mask.astype(float), minlength=len(starts))
        return np.where(mask, 1.0 / cnt[set_of], 0.0)


@pytest.mark.parametrize("rs", [SMALL, TOY, TOY2], ids=["SMALL", "TOY", "TOY2"])
def test_batched_tabulation_equals_the_row_at_a_time_one(rs):
    """Array-equal, never merely close: a report card that reproduced its pinned values
    only to a tolerance would not be the "reproduces to the digit" device it is for."""
    tree = compile_tree(rs, (L1,), True, cache=store.CACHE)
    assert np.array_equal(tabulate_batched(tree, L1, BatchedRandom(rs)), tabulate(tree, L1, PlayFirstRandom(rs)))


def test_batched_tabulation_chunks_do_not_change_the_answer():
    """The chunk size is a memory knob, never a result knob: every chunking must give
    one vector, including a chunk smaller than a single set's action count."""
    tree = compile_tree(SMALL, (L1,), True, cache=store.CACHE)
    ref = tabulate_batched(tree, L1, BatchedRandom(SMALL))
    for chunk in (1, 2, 7, 100, 10_000):
        assert np.array_equal(tabulate_batched(tree, L1, BatchedRandom(SMALL), chunk=chunk), ref)


def test_batched_tabulation_refuses_an_asymmetric_policy():
    """The section 18 orbit check applies to the batched path too. A plain
    (non-equivariant) encoder is not orbit-uniform, which is why the encoding decision
    of step 10a is a decision at all."""
    tree = compile_tree(TOY, (L1,), True, cache=store.CACHE)

    class FirstAction:
        name = "first_action"

        def policy_batch(self, batch, ia_action, info_start):
            out = np.zeros(len(ia_action))
            out[np.asarray(info_start, dtype=np.int64)] = 1.0  # all mass on each set's first action
            return out

    with pytest.raises(ValueError, match="not symmetric"):
        tabulate_batched(tree, L1, FirstAction())


def test_batched_tabulation_is_l1_only():
    tree = compile_tree(SMALL, (L1, Level.L2), True, cache=store.CACHE)
    with pytest.raises(ValueError, match="L1 only"):
        tabulate_batched(tree, Level.L2, BatchedRandom(SMALL))


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
