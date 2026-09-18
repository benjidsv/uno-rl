"""The attack instrument (`uno.eval.attack`, `uno.compiled.flaws`, step 10a.3C).

The load-bearing property is **soundness**: an attack's achieved score can never exceed the exact
best response, because a best response is the best any L1 policy can do against a fixed opponent.
Beside it sits the property that a *bound* cannot have: **sensitivity**. A vacuous attacker that
plays anything at all still produces a valid lower bound, so the tests that matter are the ones
that would fail if the instrument stopped detecting a weakness it is supposed to detect -- which is
why the flawed targets of `uno.compiled.flaws` exist and why the near-Nash target is tested
alongside them. A test suite that only checked validity would pass with the attack deleted.

Toy scale throughout: `TOY` is 12,036 information sets and a whole tabulated attacker costs a
couple of seconds at these budgets, which is what makes exact assertions affordable here.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from uno import store
from uno.agents.baseline import Heuristic, KeepsDrawn, PlayFirstRandom, UniformLegal
from uno.agents.symmetrized import Symmetrized
from uno.agents.tabular import CompiledAgent
from uno.compiled import flaws
from uno.compiled.solvers import best_response, evaluate, exploitability
from uno.compiled.strategies import tabulate
from uno.compiled.tree import compile_tree
from uno.eval import attack
from uno.eval.attack import Budget, UnsupportedTarget
from uno.game import rules
from uno.game.observe import Level

L1 = Level.L1
DEPTH = 40  # always terminal on the toys, so the leaf below is never consulted


def _leaf_must_not_fire(s):
    raise AssertionError("the search hit its depth limit: a toy world was not solved exactly")


def _score(v: float) -> float:
    return (v + 1) / 2


@pytest.fixture(scope="module")
def toy():
    rs = rules.TOY
    tree = compile_tree(rs, (L1,), True, cache=store.CACHE)
    nash = store.load_or_solve_nash(tree, 1000, "cfr+")
    return rs, tree, nash


def _attack_scores(rs, tree, target_agent, target_sigma, budget, **kw) -> tuple[float, float]:
    """The tabulated attacker's exact score at both seats: no game sampling anywhere."""
    att = attack.build(rs, target_agent, budget, leaf=_leaf_must_not_fire, **kw)
    sigma = tabulate(tree, L1, Symmetrized(att, rs, name=f"sym_{att.name}"))
    return (
        _score(evaluate(tree, (L1, sigma), (L1, target_sigma))[0]),
        _score(evaluate(tree, (L1, target_sigma), (L1, sigma))[1]),
    )


# -- which targets may be attacked ----------------------------------------------------------------


def test_supports_measures_the_target_rather_than_trusting_a_declaration(toy):
    """Position-only rulebots answer identically when only private history changes; a strategy
    keyed on the full L1 information state does not. Measured, not declared by an attribute the
    target could set wrongly."""
    rs, tree, nash = toy
    for agent in (PlayFirstRandom(rs), Heuristic(rs), UniformLegal(rs), KeepsDrawn(rs, 0.5)):
        sup = attack.supports(rs, agent, games=15)
        assert sup.conclusive, f"the probe never exercised the distinction for {sup.name}"
        assert sup.position_only, f"{sup.name} should be position-only: {sup}"
    sup = attack.supports(rs, CompiledAgent(tree, L1, nash.avg, name="nash"), games=15)
    assert sup.conclusive and not sup.position_only, f"a tabulated strategy reads its private history: {sup}"
    assert sup.changed > 0


def test_only_the_querying_continuation_refuses_a_memoryful_target(toy):
    rs, tree, nash = toy
    memoryful = CompiledAgent(tree, L1, nash.avg, name="nash")
    with pytest.raises(UnsupportedTarget):
        attack.check_supported(rs, memoryful, "response", games=15)
    for cont in ("surrogate", "minimax", "rollout"):
        attack.check_supported(rs, memoryful, cont, games=15)  # these never read the target
    for cont in attack.CONTINUATIONS:
        attack.check_supported(rs, PlayFirstRandom(rs), cont, games=15)


def test_the_refusal_is_not_vacuous_because_the_invalid_query_fails_silently(toy):
    """Why `response` refuses: inside a determinized world the opponent's private history is
    fabricated, so a tabulated target is asked about information sets that do not exist. It does
    not raise -- it misses the keying and returns a point mass on the first legal action, which
    would make the search best-respond to a policy the target does not follow. Without this test
    the guard above could be deleted and nothing would look wrong."""
    import random

    from uno.eval.search import determinize_l1
    from uno.game.engine import legal_actions, new_game, step
    from uno.game.observe import observe

    rs, tree, nash = toy
    target = CompiledAgent(tree, L1, nash.avg, name="nash_probe")
    queried = missed = point_mass = 0
    for seed in range(25):
        try:
            s = new_game(rs, seed)
        except ValueError:
            continue
        rng = random.Random(seed)
        while not s.terminal and queried < 40:
            obs = observe(s, s.player, L1)
            world = determinize_l1(rs, obs, random.Random(seed))
            for a in legal_actions(world):
                child = step(world, a, check=False)
                if child.terminal or child.player == obs.seat:
                    continue
                before = target.misses
                probs = target.policy(observe(child, child.player, L1), legal_actions(child))
                queried += 1
                if target.misses > before:
                    missed += 1
                    point_mass += len(probs) == 1 and next(iter(probs.values())) == 1.0
                break
            s = step(s, rng.choice(legal_actions(s)))
    assert queried > 20, "the probe never reached an opponent node inside a determinized world"
    assert missed > queried // 4, f"only {missed} of {queried} queries missed: is the fabricated history still invalid?"
    assert point_mass == missed, "a missed query no longer degrades silently to the first legal action"


# -- the flawed targets ---------------------------------------------------------------------------


def test_a_graft_keeps_the_strategy_symmetric(toy):
    """A symmetry-closed class keeps the strategy orbit-uniform, which is what lets the quotient
    tree evaluate it at all (docs/foundations/spec.md section 18)."""
    rs, tree, nash = toy
    k = tree.keyings[L1]
    built = flaws.flawed(tree, nash.avg)
    assert built, "no non-vacuous graft on TOY"
    for label, sigma in built.items():
        assert k.is_symmetric(sigma, 1e-9), f"{label} is not symmetric"
        sums = k.info_sum(sigma)
        assert np.allclose(sums, 1.0, atol=1e-9), f"{label} is not a strategy"


def test_a_no_op_graft_raises_instead_of_returning_an_unflawed_target(toy):
    """A calibration target that is secretly unflawed is worse than none: the attack would be
    scored against a target whose flaw does not exist."""
    rs, tree, nash = toy
    for mode in ("never", "always"):
        with pytest.raises(ValueError):
            flaws.graft(tree, nash.avg, "action_card", mode)  # TOY has no action cards
    with pytest.raises(ValueError):
        flaws.graft(tree, nash.avg, "pass", "sideways")
    with pytest.raises(ValueError):
        flaws.graft(tree, nash.avg, "no_such_class", "never")


def test_the_localized_flaws_bracket_the_near_nash_one(toy):
    """The calibration only works if the targets differ in exploitability by a lot: `never_pass`
    costs almost nothing (declining a playable drawn card is nearly free), while `never_wild_play`
    is severe. Pinned loosely, since these are exact values of a cached CFR+ strategy."""
    rs, tree, nash = toy
    nash_v = _score(evaluate(tree, (L1, nash.avg), (L1, nash.avg))[0])

    def nashconv(sigma):
        ex = exploitability(tree, L1, sigma)
        return sum(100 * (ex.br[s ^ 1].score - (1 - nash_v if s == 0 else nash_v)) for s in (0, 1))

    near = nashconv(flaws.graft(tree, nash.avg, "pass", "never"))
    severe = nashconv(flaws.graft(tree, nash.avg, "wild_play", "never"))
    assert near == pytest.approx(1.60, abs=0.3), f"near-Nash graft moved: {near:.2f}"
    assert severe == pytest.approx(27.47, abs=1.0), f"severe graft moved: {severe:.2f}"
    assert severe > near + 10


# -- soundness and sensitivity of the attack ------------------------------------------------------


@pytest.mark.parametrize("family", ("response", "surrogate", "minimax", "rollout"))
def test_the_attack_never_beats_the_exact_best_response(toy, family):
    """The soundness property: BR L1 is the best any L1 policy can do against a fixed opponent, so
    an attack that exceeded it would mean the attacker is reading something it must not see (or the
    tabulation is mis-keyed). This is the check that a plausible-looking wiring bug fails."""
    rs, tree, _ = toy
    target = PlayFirstRandom(rs)
    sigma = store.load_or_tabulate(tree, L1, target)
    kw = {"surrogate": dict(surrogate=Heuristic(rs)), "rollout": dict(rollout_agents=(Heuristic(rs), PlayFirstRandom(rs)))}.get(family, {})
    budget = Budget(continuation=family, worlds=4, depth=DEPTH, belief="prior", seed=0)
    got = _attack_scores(rs, tree, target, sigma, budget, **kw)
    for seat in (0, 1):
        ceiling = best_response(tree, L1, (L1, sigma), seat).score
        assert got[seat] <= ceiling + 1e-9, f"{family} at seat {seat}: {got[seat]:.4f} beat BR L1 {ceiling:.4f}"


def test_the_attack_detects_a_known_flaw_and_stays_quiet_against_near_nash(toy):
    """Sensitivity, the property a valid bound does not imply. A vacuous attacker would report a
    similar (small) number against both targets and pass every soundness test above; this fails for
    it. The flawed target's bound must be far larger than the near-Nash one's, and neither may
    exceed the exact NashConv it is bounding."""
    rs, tree, nash = toy
    nash_v = _score(evaluate(tree, (L1, nash.avg), (L1, nash.avg))[0])
    budget = Budget(continuation="surrogate", worlds=8, depth=DEPTH, belief="prior", seed=0)
    kw = dict(surrogate=Heuristic(rs))
    out = {}
    for label, sigma in (
        ("near_nash", flaws.graft(tree, nash.avg, "pass", "never")),
        ("flawed", flaws.graft(tree, nash.avg, "wild_play", "never")),
    ):
        got = _attack_scores(rs, tree, CompiledAgent(tree, L1, sigma, name=f"t_{label}"), sigma, budget, **kw)
        ex = exploitability(tree, L1, sigma)
        true_nc = sum(100 * (ex.br[s ^ 1].score - (1 - nash_v if s == 0 else nash_v)) for s in (0, 1))
        out[label] = (100 * (got[0] + got[1] - 1), true_nc)
    for label, (bound, true_nc) in out.items():
        assert bound <= true_nc + 1e-6, f"{label}: the bound {bound:.2f} exceeds the exact NashConv {true_nc:.2f} it bounds"
    assert out["flawed"][0] > out["near_nash"][0] + 5.0, f"the attack does not separate the targets: {out}"


def test_more_worlds_do_not_make_the_attack_worse(toy):
    """Pre-declared as monotone in the budget, not as reaching zero: PIMC's structural bias does not
    vanish with more worlds (docs/foundations/spec.md section 25), so the claim is only that a bigger budget
    does not miss more."""
    rs, tree, _ = toy
    target = Heuristic(rs)
    sigma = store.load_or_tabulate(tree, L1, target)
    prev = None
    for w in (1, 8):
        got = _attack_scores(rs, tree, target, sigma, Budget(continuation="response", worlds=w, depth=DEPTH, belief="prior", seed=0))
        total = got[0] + got[1]
        if prev is not None:
            assert total >= prev - 1e-9, f"w={w} scored {total:.4f} against w's predecessor {prev:.4f}"
        prev = total


def test_the_attacker_never_reads_the_hidden_state(toy):
    """L1 validity: the attacker may not read the opponent's hand or the pile order.

    `supports` is deliberately *not* used here. An L1 policy is entitled to read its own initial
    hand and own draws -- and since those plus the public plays determine its current hand, any
    agent that rebuilds its position from the L1 key (this attacker does, through `Symmetrized`)
    answers differently when they are perturbed, for a legitimate reason. The property that matters
    is the other one: swap in a different hidden world behind an identical L1 observation, and the
    answer must not move."""
    rs, tree, _ = toy
    target = PlayFirstRandom(rs)
    att = attack.build(rs, target, Budget(worlds=4, depth=DEPTH, belief="prior"), leaf=_leaf_must_not_fire)
    for agent in (att, Symmetrized(att, rs, name=f"sym_{att.name}")):
        probe = attack.hidden_state_probe(rs, agent, games=12)
        assert probe.tested > 10, f"the probe never exercised a real decision: {probe}"
        assert probe.observations_differed == 0, f"the engine handed out an L1 observation that depends on the hidden split: {probe}"
        assert probe.l1_valid, f"the attacker reads the hidden state: {probe}"


def test_the_mean_over_attacker_seeds_is_the_exact_behavioural_average(toy):
    """A built attacker is ONE realization of its keyed world sampling, not the averaged
    behavioural policy. `uno.compiled.pimc.pimc_n` computes that average exactly, so the seeds must
    straddle it -- which is also what makes the seed spread a reportable uncertainty rather than a
    nuisance."""
    from uno.compiled.pimc import pimc_n
    from uno.compiled.strategies import transfer

    rs, tree, _ = toy
    world = compile_tree(rs, (L1, Level.L3), True, cache=store.CACHE)
    target = PlayFirstRandom(rs)
    sigma = store.load_or_tabulate(tree, L1, target)
    target_w = (L1, transfer(tree, world, L1, sigma))
    n = 4
    vals = [_attack_scores(rs, tree, target, sigma, Budget(continuation="response", worlds=n, depth=DEPTH, belief="prior", seed=k)) for k in range(4)]
    for seat in (0, 1):
        xs = np.array([v[seat] for v in vals])
        exact = pimc_n(world, target_w, seat, n, realizations=64, weighting="prior", continuation="response", seed=1).score
        se = xs.std(ddof=1) / math.sqrt(len(xs))
        assert abs(xs.mean() - exact) <= max(3 * se, 0.02), f"seat {seat}: seeds mean {xs.mean():.4f} against pimc_n {exact:.4f} (se {se:.4f})"
        assert xs.std(ddof=1) > 0, "the attacker seed changed nothing: is the sampling actually keyed on the seed?"


# -- the reported bound ---------------------------------------------------------------------------


def test_the_bound_is_the_seat_sum_and_carries_the_covariance():
    """`NashConv >= b0 + b1 - 1`, with the uncertainty taken from the per-deal sum, so the seats are
    never assumed independent whatever the sign of their correlation (measured mild in this game:
    -0.30 on TOY, -0.10 on FULL_UNO, +0.10 +- 0.16 for one attacker). The constructed case below is
    the extreme that makes the difference visible: perfectly anti-correlated seats have a
    *zero*-variance sum, which an independence assumption cannot represent."""
    rng = np.random.default_rng(0)
    b0 = rng.binomial(1, 0.6, 400).astype(float)
    b1 = rng.binomial(1, 0.55, 400).astype(float)
    bound = attack.nashconv_bound(b0, b1)
    assert bound.estimate == pytest.approx(100 * (b0.mean() + b1.mean() - 1))
    assert bound.lower == pytest.approx(bound.estimate - bound.z * bound.se)

    x = rng.random(300)
    paired = attack.nashconv_bound(x, 1 - x)
    independent = 100 * math.sqrt(x.var(ddof=1) / 300 + (1 - x).var(ddof=1) / 300)
    assert paired.se < 1e-9 < independent, f"the covariance is not carried: paired se {paired.se}, independent {independent}"
    assert paired.corr == pytest.approx(-1.0, abs=1e-9)


def test_the_bound_never_reports_less_than_the_trivial_zero():
    """A weak attack gives a negative point estimate. Zero is always a valid lower bound, so that
    is what gets reported -- and a negative number is never evidence that a target is safe."""
    rng = np.random.default_rng(1)
    bound = attack.nashconv_bound(rng.binomial(1, 0.3, 200).astype(float), rng.binomial(1, 0.3, 200).astype(float))
    assert bound.estimate < 0 and bound.reported == 0.0


def test_the_bound_guards_its_inputs():
    with pytest.raises(ValueError):
        attack.nashconv_bound([0.5, 0.5], [0.5])  # not the same deals
    with pytest.raises(ValueError):
        attack.nashconv_bound([0.5], [0.5])  # no standard error from one deal


def test_a_budget_rejects_a_mislabelling_combination():
    """A name identifies a policy (docs/foundations/spec.md section 13), and `PimcRollout` has no belief rung,
    so a rollout budget carrying one would put a rung in the agent's name that its policy never
    uses."""
    rs = rules.TOY
    with pytest.raises(ValueError):
        Budget(continuation="teleport")
    with pytest.raises(ValueError):
        Budget(worlds=0)
    with pytest.raises(ValueError):
        attack.build(
            rs,
            PlayFirstRandom(rs),
            Budget(continuation="rollout", belief="voids"),
            leaf=_leaf_must_not_fire,
            rollout_agents=(Heuristic(rs), PlayFirstRandom(rs)),
        )
