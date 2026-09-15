"""`observation_from_key` rebuilds the observation at every decision node from the key alone (docs/spec.md section 10)."""

from __future__ import annotations

from dataclasses import replace

from hypothesis import given, settings

from tests.strategies import rulesets, seeds, walk
from uno import Level, observe
from uno.game.keys import KEYS
from uno.game.observe import observation_from_key


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
