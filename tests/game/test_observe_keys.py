"""`observation_from_key` rebuilds the observation at every decision node from the key alone (docs/spec.md section 10).
Step 10a: and `batch_l1_observations` rebuilds many at once, to the same values (item A2)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from hypothesis import given, settings

from tests.strategies import rulesets, seeds, walk
from uno import Level, observe, store
from uno.compiled.tree import compile_tree
from uno.game.engine import Phase
from uno.game.keys import KEYS
from uno.game.observe import batch_l1_observations, discard_from_public, observation_from_key
from uno.game.rules import TOY, TOY2


def _assert_batch_matches(rs, keys):
    """Every field of the batch equals the row-at-a-time observation's, exactly.

    `np.array_equal`, never `allclose`: the batched path exists to make a report card
    cheap, and a report card that reproduced its pinned values only to a tolerance
    would not be the "reproduces to the digit" device it is for."""
    ref = [observation_from_key(rs, Level.L1, k) for k in keys]
    got = batch_l1_observations(rs, keys)
    assert len(got) == len(ref)
    hands = np.zeros((len(ref), rs.num_types), dtype=np.int64)
    for i, o in enumerate(ref):
        for c in o.hand:
            hands[i, c] += 1
    assert np.array_equal(got.hands, hands)
    assert np.array_equal(got.seat, np.array([o.seat for o in ref]))
    assert np.array_equal(got.top, np.array([o.top for o in ref]))
    assert np.array_equal(got.current_color, np.array([o.current_color for o in ref]))
    assert np.array_equal(got.opp_hand_size, np.array([o.opp_hand_size for o in ref]))
    assert np.array_equal(got.pile_size, np.array([o.pile_size for o in ref]))
    assert np.array_equal(got.turn, np.array([o.turn for o in ref]))
    assert np.array_equal(got.after_draw, np.array([o.phase is Phase.AFTER_DRAW for o in ref]))
    assert np.array_equal(got.drawn, np.array([-1 if o.drawn is None else o.drawn for o in ref]))
    assert got.discard == [discard_from_public(o.public) for o in ref]


@given(rs=rulesets(), seed=seeds, policy_seed=seeds)
@settings(deadline=None, max_examples=50)
def test_batch_l1_observations_match_one_at_a_time(rs, seed, policy_seed):
    states, _ = walk(rs, seed, policy_seed)
    keys = [KEYS[Level.L1](s, s.player) for s in states if not s.terminal]
    if keys:
        _assert_batch_matches(rs, keys)


@pytest.mark.parametrize("rs,limit", [(TOY, None), (TOY2, 20_000)], ids=["TOY", "TOY2"])
def test_batch_l1_observations_over_a_whole_keying(rs, limit):
    """At the scale a report card actually walks: whole sets of a compiled keying,
    where the keys are canonical rather than the ones a random game happens to reach."""
    tree = compile_tree(rs, (Level.L1,), True, cache=store.CACHE)
    k = tree.keyings[Level.L1]
    n = k.n_info if limit is None else min(limit, k.n_info)
    _assert_batch_matches(rs, [k.key(i) for i in range(n)])


def test_an_empty_batch_has_no_rows():
    got = batch_l1_observations(TOY, [])
    assert len(got) == 0 and got.hands.shape == (0, TOY.num_types)


@given(rs=rulesets(), seed=seeds, policy_seed=seeds)
@settings(deadline=None)
def test_observation_from_key_matches_observe(rs, seed, policy_seed):
    states, _ = walk(rs, seed, policy_seed)
    for s in states:
        if s.terminal:
            continue
        p = s.player
        for level in (Level.L1, Level.L2, Level.L3):
            key = KEYS[level](s, p)
            got = observation_from_key(rs, level, key)
            want = observe(s, p, level)
            if level is Level.L3:  # the L3 key excludes the seed
                want = replace(want, seed=None)
            assert got == want, (level, got, want)
