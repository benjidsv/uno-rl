"""The training target and the exact policy gradient (step 10a.2, item A6): the identities
that pin `uno.eval.target` and `uno.compiled.gradient` to the tree, on the 7-card SMALL
ruleset.

The measurements themselves are `uv run uno target`; these are the cheap invariants."""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest

from tests.strategies import SMALL
from uno.agents.baseline import Heuristic, PlayFirstRandom, UniformLegal
from uno.agents.tabular import CompiledAgent
from uno.compiled.aivat import TreeValues, exact_values, l1_set_values
from uno.compiled.gradient import estimator_mean, exact_gradient, finite_difference, orbit_gradient
from uno.compiled.lookup import build
from uno.compiled.strategies import tabulate
from uno.compiled.tree import compile_tree
from uno.eval.aivat import ALL, NONE, Corrections, trace_terms
from uno.eval.play import play_game
from uno.eval.target import TRAINING, Decisions, corrected_return, own_decisions, per_game, sampled_gradient
from uno.game.observe import Level

L1, L2 = Level.L1, Level.L2

# subsets of the corrections that leave the learner's own decisions alone, so every one of
# them must leave the gradient exactly where it was
KEEPS = (
    NONE,
    Corrections(True, False, False, False),
    Corrections(False, False, True, False),
    Corrections(False, False, False, True),
    TRAINING,
)


@cache
def setup():
    tree = compile_tree(SMALL, (L1, L2), symmetric=True)
    heu, rnd = Heuristic(SMALL), PlayFirstRandom(SMALL)
    # the learner is Random: a deterministic policy has a zero logit gradient by
    # construction, so a gradient check needs a mixed one
    return tree, (L1, tabulate(tree, L1, rnd)), (L1, tabulate(tree, L1, heu)), heu, rnd


def profile(learner, column, seat):
    return (learner, column) if seat == 0 else (column, learner)


def choices(tree, g, seat, n=6):
    """Orbits a differentiable, non-trivial choice actually rests on, largest gradient first.

    Two traps, both of which made this check vacuous while it was being written:

    - a set with one action orbit has `sigma == 1` there and a gradient that is identically
      zero, and those sets carry most of the reach, so ranking candidates by reach picks
      twelve of them and the check passes on `0.0 == 0.0` (step 9.5e's vacuous-test trap);
    - requiring *full* support on the set leaves nothing at all on a free-play ruleset,
      where `PlayFirstRandom` puts zero on `Draw` at every set with a play. Only the
      perturbed orbit needs positive probability (`finite_difference`)."""
    k = tree.keyings[L1]
    n_orbits = k.info_orbit_start[1:] - k.info_orbit_start[:-1]
    ex = orbit_gradient(k, g.logit)
    live = np.nonzero(
        (k.info_player[k.orbit_info] == seat)
        & (g.reach[k.orbit_info] > 1e-9)
        & (n_orbits[k.orbit_info] > 1)
        & (np.bincount(k.ia_orbit, weights=g.sigma, minlength=k.n_orbits) > 0)
    )[0]
    return sorted((int(o) for o in live), key=lambda o: -abs(ex[o]))[:n], ex


def test_the_exact_gradient_matches_a_finite_difference():
    """Against the definition, on both a policy with zeros (Random, which never draws with a
    play in hand) and one with full support (uniform over every legal action)."""
    tree, learner, column, *_ = setup()
    for lname, row in (("random", learner), ("uniform_legal", (L1, tabulate(tree, L1, UniformLegal(SMALL))))):
        for seat in (0, 1):
            g = exact_gradient(tree, row, column, seat)
            live, ex = choices(tree, g, seat)
            assert live, f"{lname} seat {seat}: no orbit with a real choice, the check would be vacuous"
            assert np.abs(ex[live]).max() > 1e-6, f"{lname} seat {seat}: every selected orbit has a zero gradient, the check would be vacuous"
            fd = finite_difference(tree, row, column, seat, live)
            assert np.allclose(fd, ex[live], rtol=1e-5, atol=1e-9), (lname, seat, fd, ex[live])


def test_the_logit_gradient_sums_to_zero_over_every_set():
    """A policy gradient is a direction inside the simplex, so each set's components sum to
    zero. It is what makes the corrections' per-set constants irrelevant."""
    tree, learner, column, *_ = setup()
    k = tree.keyings[L1]
    for seat in (0, 1):
        g = exact_gradient(tree, learner, column, seat)
        assert np.abs(k.info_sum(g.logit)).max() < 1e-12


def test_corrections_that_spare_the_row_leave_the_gradient_unchanged():
    """The unbiasedness the training target rests on, exactly rather than sampled: for any
    value function and any subset that does not correct the learner's own decisions, the
    sampled estimator's mean *is* the exact gradient."""
    tree, learner, column, *_ = setup()
    for seat in (0, 1):
        g = exact_gradient(tree, learner, column, seat)
        nus = (exact_values(tree, *profile(learner, column, seat)), l1_set_values(tree, *profile(learner, column, seat), exact_values(tree, *profile(learner, column, seat))))
        for nu in nus:
            for c in KEEPS:
                m = estimator_mean(tree, learner, column, seat, nu, c)
                assert np.abs(m.logit - g.logit).max() < 1e-12, (seat, c.label)
                if c.deal or c.chance or c.column:
                    # not vacuous: the corrections really are large before the centring
                    assert np.abs(m.prefix).max() > 1e-6, (seat, c.label)


def test_the_row_correction_with_exact_values_zeroes_the_gradient():
    """The trap, as an identity: with an exact V the row term is `Q - V`, so the corrected
    return's conditional mean does not depend on the action and the gradient vanishes."""
    tree, learner, column, *_ = setup()
    for seat in (0, 1):
        g = exact_gradient(tree, learner, column, seat)
        nu = exact_values(tree, *profile(learner, column, seat))
        m = estimator_mean(tree, learner, column, seat, nu, ALL)
        assert np.linalg.norm(g.logit) > 1e-6  # there was a gradient to destroy
        assert np.abs(m.logit).max() < 1e-12


def test_the_training_target_refuses_the_row_correction():
    tree, learner, column, heu, rnd = setup()
    idx = build(tree, L2)
    t = play_game(SMALL, 3, (rnd, heu))
    terms = trace_terms(TreeValues(idx, exact_values(tree, learner, column)), 3, t)
    assert corrected_return(terms, 0) == pytest.approx((1.0) * terms.estimate(TRAINING, 0))
    with pytest.raises(ValueError, match="own decisions"):
        corrected_return(terms, 0, ALL)


def test_the_corrected_return_flips_sign_at_seat_1():
    """Every AIVAT quantity is in seat-0 units; a learner maximises its own."""
    tree, learner, column, heu, rnd = setup()
    idx = build(tree, L2)
    t = play_game(SMALL, 5, (heu, rnd))
    terms = trace_terms(TreeValues(idx, exact_values(tree, column, learner)), 5, t)
    assert corrected_return(terms, 1) == pytest.approx(-terms.estimate(TRAINING, 1))


def played(tree, n=120, seat=0):
    """`n` games of the learner (as a tabulated agent, so its played policy *is* sigma)
    against the Heuristic, with the corrections and the learner's own decisions."""
    tree_, learner, column, heu, rnd = setup()
    k = tree.keyings[L1]
    idx = build(tree, L2)
    agent = CompiledAgent(tree, L1, learner[1], name="learner")
    model = TreeValues(idx, exact_values(tree, *profile(learner, column, seat)))
    game, info, ia, signal = [], [], [], []
    g = 0
    for seed in range(n):
        try:
            t = play_game(SMALL, seed, (agent, heu) if seat == 0 else (heu, agent))
        except ValueError:
            continue
        terms = trace_terms(model, seed, t)
        infos, ias = own_decisions(idx, k, t, seat)
        x = corrected_return(terms, seat)
        game += [g] * len(infos)
        info += infos
        ia += ias
        signal += [x] * len(infos)
        g += 1
    return Decisions(np.array(game), np.array(info), np.array(ia), np.array(signal, dtype=float), g)


def test_the_sampled_gradient_equals_a_plain_python_loop():
    """The vectorised accumulation against the definition, decision by decision."""
    tree, learner, *_ = setup()
    k = tree.keyings[L1]
    d = played(tree)
    got = sampled_gradient(k, d, learner[1])
    want = np.zeros(k.n_ia)
    for i in range(len(d)):
        lo, hi = int(k.info_start[d.info[i]]), int(k.info_start[d.info[i] + 1])
        want[lo:hi] -= learner[1][lo:hi] * d.signal[i]
        want[d.ia[i]] += d.signal[i]
    assert np.allclose(got, want / d.games, rtol=0, atol=1e-15)


def test_per_game_projections_average_to_the_gradient():
    tree, learner, column, *_ = setup()
    k = tree.keyings[L1]
    d = played(tree)
    g = sampled_gradient(k, d, learner[1])
    direction = exact_gradient(tree, learner, column, 0).logit
    assert per_game(k, d, learner[1], direction).mean() == pytest.approx(float(g @ direction), rel=1e-9, abs=1e-15)


def test_a_learner_decision_lands_on_the_set_it_played():
    """`own_decisions` must agree with what the agent was actually asked: the action taken
    is in the support the keying lists for that set."""
    tree, learner, column, heu, rnd = setup()
    k = tree.keyings[L1]
    idx = build(tree, L2)
    agent = CompiledAgent(tree, L1, learner[1], name="learner")
    for seed in range(40):
        try:
            t = play_game(SMALL, seed, (agent, heu))
        except ValueError:
            continue
        infos, ias = own_decisions(idx, k, t, 0)
        for i, a in zip(infos, ias):
            assert k.info_player[i] == 0
            assert int(k.info_start[i]) <= a < int(k.info_start[i + 1])
            assert learner[1][a] > 0  # the learner played it, so it had probability
