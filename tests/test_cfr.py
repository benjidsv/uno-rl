"""CFR (uno/cfr.py): exploitability goes to zero, and the pieces agree with the reference solvers."""

from __future__ import annotations

import math

import numpy as np

from uno import Level
from uno.agents import Heuristic, PlayFirstRandom
from uno.cfr import CFRResult, cfr, exploitability_of
from uno.compiled import best_response, compile_tree, evaluate, tabulate
from uno.play import match
from uno.solve import best_response as best_response_ref
from uno.solve import evaluate as evaluate_ref
from uno.solve import evaluate_on_seeds

from .test_solve import SMALL


def test_cfr_plus_converges_and_agrees_with_reference(tmp_path):
    rs = SMALL
    tree = compile_tree(rs, (Level.L1,))
    res = cfr(tree, iterations=60, log_at=(1, 5, 20, 60))
    nc = [e.nashconv for e in res.log]
    assert nc[-1] < 0.002, nc  # under 0.2 score points after 60 CFR+ iterations
    assert nc[0] > nc[1] > nc[-1]
    for e in res.log:
        assert e.bracket[0] <= e.value + 1e-9 and e.value <= e.bracket[1] + 1e-9  # the profile's value sits in its bracket
    assert res.iterations == 60
    k = tree.keyings[Level.L1]
    assert np.allclose(k.info_sum(res.avg), 1.0)

    # exploitability of the average strategy: reference best response to the played agent
    agent = res.agent(tree)
    ex = exploitability_of(tree, res)
    for seat in (0, 1):
        ref = best_response_ref(rs, agent, seat, Level.L1)
        assert math.isclose(ref.value, ex.br[seat].value, abs_tol=1e-9)
    # the profile's own value through the engine
    assert math.isclose(evaluate_ref(rs, (agent, agent))[0], ex.value, abs_tol=1e-9)
    assert agent.misses == 0

    # Nash value bracket: the row agents' best-response ceilings sit above it, since
    # their exploitability is non-negative by definition
    lo, hi = ex.bracket
    for opp in (PlayFirstRandom(rs), Heuristic(rs)):
        sig = (Level.L1, tabulate(tree, Level.L1, opp))
        assert best_response(tree, Level.L1, sig, 0).score >= hi - 1e-9
        assert 1 - best_response(tree, Level.L1, sig, 1).score <= lo + 1e-9

    # simulation on seeds matches the exact value on those seeds within the CI
    seeds = range(600)
    rnd = PlayFirstRandom(rs)
    r = match(rs, (agent, rnd), seeds)
    on = (evaluate_on_seeds(rs, (agent, rnd), seeds)[0] + 1) / 2
    assert abs(r.score - on) <= (r.score_ci[1] - r.score_ci[0]) / 2

    # save / load round trip
    res.save(tmp_path / "cfr.npz")
    back = CFRResult.load(tmp_path / "cfr.npz")
    assert np.array_equal(back.avg, res.avg) and back.iterations == res.iterations and len(back.log) == len(res.log)
    assert back.final.nashconv == res.final.nashconv


def test_vanilla_cfr_also_converges_but_slower():
    rs = SMALL
    tree = compile_tree(rs, (Level.L1,))
    plus = cfr(tree, iterations=30, log_at=(30,))
    vanilla = cfr(tree, iterations=30, plus=False, alternating=False, linear=False, log_at=(1, 30))
    assert vanilla.log[-1].nashconv < vanilla.log[0].nashconv
    assert plus.final.nashconv < vanilla.final.nashconv
    # both bracket the same Nash value
    assert plus.final.bracket[0] <= vanilla.final.bracket[1] and vanilla.final.bracket[0] <= plus.final.bracket[1]
