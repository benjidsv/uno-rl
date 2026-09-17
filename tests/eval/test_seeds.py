"""The seed registry (step 10a, item A1): blocks are disjoint, claimed once, and a
spent block cannot carry a reported number."""

from __future__ import annotations

import pytest

from uno.eval import seeds
from uno.eval.play import DEV_SEEDS, EVAL_SEEDS


def test_the_pools_contain_the_historical_ranges():
    """Widening the pools must not strand a number already measured: `DEV_SEEDS` is the
    first dev block and `EVAL_SEEDS` is block 00."""
    assert DEV_SEEDS.start >= seeds.DEV_POOL.start and DEV_SEEDS.stop <= seeds.DEV_POOL.stop
    assert EVAL_SEEDS.start >= seeds.EVAL_POOL.start and EVAL_SEEDS.stop <= seeds.EVAL_POOL.stop
    block00 = seeds.REGISTRY["00"]
    assert (block00.start, block00.stop) == (EVAL_SEEDS.start, EVAL_SEEDS.stop)


def test_the_pools_are_disjoint():
    assert seeds.DEV_POOL.stop <= seeds.EVAL_POOL.start


def test_claimed_blocks_never_overlap():
    ordered = sorted(seeds.REGISTRY.values(), key=lambda b: b.start)
    for a, b in zip(ordered, ordered[1:]):
        assert a.stop <= b.start, f"{a.name} and {b.name} overlap"
    for b in ordered:
        assert seeds.EVAL_POOL.start <= b.start < b.stop <= seeds.EVAL_POOL.stop


def test_overlapping_blocks_are_refused():
    bad = {"a": seeds.Block("a", 1_000_000, 1_008_000, "x", "x"), "b": seeds.Block("b", 1_004_000, 1_012_000, "y", "y")}
    with pytest.raises(ValueError, match="overlap"):
        seeds._validate(bad)


def test_a_block_outside_the_pool_is_refused():
    with pytest.raises(ValueError, match="outside EVAL_POOL"):
        seeds._validate({"a": seeds.Block("a", seeds.EVAL_POOL.stop - 10, seeds.EVAL_POOL.stop + 10, "x", "x")})


def test_a_block_cannot_be_claimed_twice():
    """A1's own check: the registry is the declaration, so a second claim on one name
    is a mistake, not a redefinition."""
    name = "test-double-claim"
    try:
        first = seeds.claim(name, "test", "first")
        assert len(first) == seeds.BLOCK
        with pytest.raises(ValueError, match="already claimed"):
            seeds.claim(name, "test", "second")
    finally:
        seeds.REGISTRY.pop(name, None)


def test_a_claim_takes_fresh_seeds_and_stays_disjoint():
    names = ["test-fresh-1", "test-fresh-2"]
    try:
        a, b = (seeds.claim(n, "test", "fresh") for n in names)
        assert a.stop <= b.start
        assert a.start >= max(x.stop for x in seeds.REGISTRY.values() if x.name not in names)
        seeds._validate(seeds.REGISTRY)
    finally:
        for n in names:
            seeds.REGISTRY.pop(n, None)


def test_a_spent_block_refuses_to_hand_out_its_seeds():
    """Block 00 was selected on in 9.5d-9.5f, so it is a dev range now: asking for it
    by name is the mistake the registry exists to catch."""
    assert seeds.REGISTRY["00"].spent
    with pytest.raises(ValueError, match="spent"):
        seeds.block("00")


def test_an_unclaimed_block_is_a_keyerror():
    with pytest.raises(KeyError):
        seeds.block("no-such-block")


def test_dev_blocks_are_inside_the_dev_pool():
    assert seeds.dev_block(0, 4000) == range(0, 4000)
    with pytest.raises(ValueError, match="outside DEV_POOL"):
        seeds.dev_block(len(seeds.DEV_POOL) - 10, 4000)
