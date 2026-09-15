"""TOY2 (notes_toy2.md): the sizes, best-response values and Nash value are pinned."""

from __future__ import annotations

import math

import pytest

from uno.agents.baseline import Heuristic, PlayFirstRandom
from uno.compiled.cfr import cfr
from uno.compiled.solvers import best_response, evaluate
from uno.compiled.strategies import tabulate
from uno.compiled.tree import compile_tree
from uno.game import rules
from uno.game.observe import Level
from uno.reference.sizing import size


def score(v: float) -> float:
    return (v + 1) / 2


def test_toy2_sizes():
    z = size(rules.TOY2, symmetric=True)
    assert (z.roots, z.histories, z.decision_nodes) == (584, 311_819, 225_345)
    assert math.isclose(z.random_p0_score, 0.5875, abs_tol=5e-5)
    f = size(rules.TOY2_FULL, symmetric=True, position_cap=200_000)
    assert not f.complete  # 28M histories, sized once by the `sizing` experiment
    s = size(rules.TOY2_SKIP, symmetric=True)
    assert (s.roots, s.histories) == (3_070, 922_744)
    assert math.isclose(s.random_p0_score, 0.5841, abs_tol=5e-5)


@pytest.fixture(scope="module")
def toy2_tree():
    return compile_tree(rules.TOY2, (Level.L1,), symmetric=True)


def test_toy2_best_responses(toy2_tree):
    """The step 8.7 values: BR L1 to Random / Heuristic per seat, and the row matchups."""
    tree = toy2_tree
    assert tree.n == 311_820
    assert tree.keyings[Level.L1].n_info == 129_428
    rnd = (Level.L1, tabulate(tree, Level.L1, PlayFirstRandom(rules.TOY2)))
    heu = (Level.L1, tabulate(tree, Level.L1, Heuristic(rules.TOY2)))
    expect = {("random", 0): 0.7305, ("random", 1): 0.5694, ("heuristic", 0): 0.6959, ("heuristic", 1): 0.4937}
    for (name, seat), v in expect.items():
        br = best_response(tree, Level.L1, rnd if name == "random" else heu, seat)
        assert math.isclose(br.score, v, abs_tol=5e-5), (name, seat, br.score)
    assert math.isclose(score(evaluate(tree, rnd, rnd)[0]), 0.5875, abs_tol=5e-5)
    assert math.isclose(score(evaluate(tree, heu, rnd)[0]), 0.6730, abs_tol=5e-5)
    assert math.isclose(score(evaluate(tree, rnd, heu)[1]), 0.4474, abs_tol=5e-5)
    assert math.isclose(score(evaluate(tree, heu, heu)[0]), 0.6385, abs_tol=5e-5)


def test_toy2_nash(toy2_tree):
    """Nash seat-0 score 0.5784 (1000 CFR+ iterations in the script); 100 here brackets it to 0.02 pts."""
    res = cfr(toy2_tree, iterations=100)
    lo, hi = res.final.bracket
    assert hi - lo < 0.0005
    assert lo - 1e-4 <= 0.5784 <= hi + 1e-4
