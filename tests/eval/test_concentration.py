"""The statistics behind `uv run uno pimc concentration` (step 9.5g).

The section's conclusions are arithmetic over recorded value matrices, so the arithmetic is what
needs pinning. Two of its pieces can be wrong in ways that would silently flatter or bury the
result:

- `_r2` is a *plug-in* variance ratio, upward biased with few draws per stratum. The section
  reads it against a permutation control rather than against zero, and the test that matters is
  that the control really sits where the bias is -- i.e. that a grouping carrying no signal scores
  well above zero. A test asserting `_r2 ~ 0` on noise would pass a broken implementation and
  would have hidden exactly the bias the control exists to remove.
- `_race` decides how many worlds an adaptive rule would spend. Its two ends (never stop, always
  stop) are the ones a bug would move, and they are pinned here.
"""

from __future__ import annotations

import math

import numpy as np

from uno.experiments.pimc import _best_two, _r2, _race


def test_r2_is_one_when_the_grouping_names_the_value_exactly() -> None:
    d = np.repeat([0.0, 1.0, 2.0, 3.0], 8)
    g = np.repeat(np.arange(4), 8)
    prop, ney = _r2(d, g)
    assert prop == 1.0 and ney == 1.0


def test_r2_is_zero_when_the_grouping_is_a_constant() -> None:
    rng = np.random.default_rng(0)
    d = rng.normal(size=256)
    prop, ney = _r2(d, np.zeros(256, dtype=int))
    assert abs(prop) < 1e-12 and abs(ney) < 1e-12


def test_r2_neyman_never_below_proportional() -> None:
    """Optimal allocation cannot do worse than proportional: `(sum p sd)^2 <= sum p var`."""
    rng = np.random.default_rng(1)
    for _ in range(50):
        g = rng.integers(0, 5, size=200)
        d = rng.normal(size=200) * (1 + g)  # deliberately unequal within-group spread
        prop, ney = _r2(d, g)
        assert ney >= prop - 1e-12


def test_the_permutation_control_sits_at_the_plug_in_bias_not_at_zero() -> None:
    """Pure noise scores roughly `(groups - 1) / (n - 1)`, which is why the section subtracts a
    control. If this ever came out near zero the correction would be over-correcting instead."""
    rng = np.random.default_rng(2)
    n, k = 128, 16
    g = np.repeat(np.arange(k), n // k)
    got = float(np.mean([_r2(rng.normal(size=n), g)[0] for _ in range(200)]))
    assert math.isclose(got, (k - 1) / (n - 1), rel_tol=0.25)


def test_race_stops_at_the_first_block_when_the_leader_is_clean() -> None:
    q = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (32, 1))  # every world agrees, zero spread
    used, leader = _race(q, t=3.0, block=4)
    assert used == 4 and leader == 0


def test_race_spends_the_whole_budget_on_a_tie() -> None:
    rng = np.random.default_rng(3)
    q = rng.normal(size=(32, 2)).astype(np.float32)  # two indistinguishable actions
    used, _ = _race(q, t=100.0, block=4)
    assert used == 32


def test_race_at_the_full_budget_returns_the_full_sample_answer() -> None:
    rng = np.random.default_rng(4)
    q = rng.normal(size=(32, 5)).astype(np.float32)
    assert _race(q, t=1e9, block=4)[1] == _best_two(q.mean(0))[0]


def test_best_two_breaks_ties_to_the_lowest_index_like_the_policy() -> None:
    """`PimcSearch.policy` breaks ties to the lowest action id and `legal_actions` is ascending,
    so the lowest index must win a tie here too, or the recorded answer is not the played one."""
    assert _best_two(np.array([1.0, 1.0, 0.0])) == (0, 1)
