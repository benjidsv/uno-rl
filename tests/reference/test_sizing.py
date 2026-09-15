"""Position DP (uno/reference/sizing.py) against the history walk (uno/reference/census.py) and the compiled tree."""

from __future__ import annotations

import math

from hypothesis import HealthCheck, assume, given, settings

from tests.strategies import small_lazy_rulesets
from uno.agents.baseline import PlayFirstRandom
from uno.compiled.solvers import evaluate
from uno.compiled.strategies import tabulate
from uno.compiled.tree import compile_tree
from uno.game import rules
from uno.game.observe import Level
from uno.reference.census import census
from uno.reference.sizing import size


def check_against_census(rs, symmetric: bool) -> None:
    c = census(rs, node_cap=20_000, symmetric=symmetric)
    assume(c.complete)
    z = size(rs, symmetric=symmetric)
    assert z.complete
    assert z.roots == c.deals
    assert z.histories == c.histories
    assert z.decision_nodes == c.decision_nodes
    assert z.terminals == c.terminals
    assert z.max_depth == c.max_depth
    assert z.positions <= z.histories
    assert set(z.end_mass[0]) == set(c.end_mass)
    for k, v in c.end_mass.items():
        assert math.isclose(z.end_mass[0][k], v, abs_tol=1e-9), k
    for mass in z.end_mass:
        assert math.isclose(sum(mass.values()), 1.0)


@given(rs=small_lazy_rulesets())
@settings(deadline=None, max_examples=40, suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow])
def test_matches_census(rs):
    """Every history-free count and the uniform-legal end mass equal the walk's, on the full tree."""
    assume(rs.deck_size <= 10)
    check_against_census(rs, symmetric=False)


@given(rs=small_lazy_rulesets())
@settings(deadline=None, max_examples=25, suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow])
def test_matches_census_quotient(rs):
    assume(rs.deck_size <= 10)
    check_against_census(rs, symmetric=True)


@given(rs=small_lazy_rulesets())
@settings(deadline=None, max_examples=25, suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow])
def test_random_score_matches_compiled_tree(rs):
    """The PlayFirstRandom end mass is the exact random-vs-random value of the compiled tree,
    and the world tree has the same value with the same or more histories."""
    assume(rs.deck_size <= 9)
    c = census(rs, node_cap=20_000, symmetric=True)
    assume(c.complete)
    z = size(rs, symmetric=True)
    tree = compile_tree(rs, (Level.L1,), symmetric=True)
    r = (Level.L1, tabulate(tree, Level.L1, PlayFirstRandom(rs)))
    assert math.isclose((evaluate(tree, r, r)[0] + 1) / 2, z.random_p0_score, abs_tol=1e-9)
    assert tree.n == z.histories + 1
    w = size(rs, symmetric=True, world=True)
    assert math.isclose(w.random_p0_score, z.random_p0_score, abs_tol=1e-9)
    assert math.isclose(w.uniform_legal_p0_score, z.uniform_legal_p0_score, abs_tol=1e-9)
    assert w.max_depth == z.max_depth


def test_position_cap():
    z = size(rules.TOY, symmetric=True, position_cap=1000)
    assert not z.complete
    assert 1000 <= z.positions <= 1010  # the recursion in flight may add a few
    assert z.histories < 33_646


def test_toy_pinned():
    """The step 8.6 sizes of TOY and TOY_FULL (quotient), and the exact random-vs-random value."""
    z = size(rules.TOY, symmetric=True)
    assert (z.roots, z.histories, z.decision_nodes) == (144, 33_646, 23_498)
    assert math.isclose(z.random_p0_score, 0.6493, abs_tol=5e-5)
    assert math.isclose(z.uniform_legal_p0_score, 0.6515, abs_tol=5e-5)
    w = size(rules.TOY, symmetric=True, world=True)
    assert (w.roots, w.histories) == (242, 33_997)
    f = size(rules.TOY_FULL, symmetric=True)
    assert f.histories == 774_007
    assert math.isclose(f.random_p0_score, z.random_p0_score, abs_tol=1e-12)  # Random never draws voluntarily
