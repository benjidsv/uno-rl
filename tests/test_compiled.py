"""Compiled tree (uno/compiled.py) against the reference walkers: census, uno.solve, uno.estimate."""

from __future__ import annotations

import math

import numpy as np
from hypothesis import HealthCheck, assume, given, settings

from uno import Level, Ruleset
from uno.agents import BySeat, Heuristic, PlayFirstRandom, UniformLegal
from uno.compiled import END_KINDS, CompiledAgent, best_response, compile_tree, evaluate, exact_mirrored, exploitability, outcome_distribution, tabulate, tabulate_table
from uno.estimate import exact_mirrored as exact_mirrored_ref
from uno.solve import best_response as best_response_ref
from uno.solve import evaluate as evaluate_ref
from uno.solve import outcome_distribution as outcome_distribution_ref
from uno.tree import census

from .test_solve import SMALL, pair
from .test_tree import small_lazy_rulesets

LEVELS = (Level.L1, Level.L2, Level.L3)


CAP = 5_000  # histories; the reference solvers walk States, so examples must stay small


def small(rs: Ruleset):
    """Census of a ruleset small enough for the reference solvers, or assume(False)."""
    assume(sum(rs.deck_counts[t] for t, ct in enumerate(rs.card_types) if ct.rank is not None) > 0)
    assume(rs.deck_size - 2 * rs.hand_size <= 5)  # the world tree has one root per ordering of the pile
    c = census(rs, node_cap=CAP)
    assume(c.complete)
    return c


@given(rs=small_lazy_rulesets())
@settings(deadline=None, max_examples=30, suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow])
def test_counts_and_uniform_value_match_census(rs):
    c = small(rs)
    tree = compile_tree(rs, (Level.L1, Level.L2), symmetric=False)
    assert tree.n == c.histories + 1  # the virtual root
    assert tree.n_terminals == c.terminals
    assert tree.n_decisions == c.decision_nodes
    assert tree.keyings[Level.L1].n_info == sum(c.info_sets)
    assert np.all(tree.level[tree.edge_child] > tree.level[tree.arc_node[tree.edge_arc]])
    k = tree.keyings[Level.L1]
    u = (Level.L1, k.uniform())
    assert math.isclose((evaluate(tree, u, u)[0] + 1) / 2, c.uniform_legal_p0_score, abs_tol=1e-9)
    d = outcome_distribution(tree, u, u)
    assert math.isclose(sum(d.values()), 1.0, abs_tol=1e-12)
    ends, _ = tree.values(tree.profile_weights(u, u), tree.end_columns())
    for i, kind in enumerate(END_KINDS[1:]):
        want = sum(p for (e, _), p in c.end_mass.items() if e == kind)
        assert math.isclose(ends[0, i], want, abs_tol=1e-9)
    # the world tree carries the same L1 sets and values
    w = compile_tree(rs, (Level.L1, Level.L3), symmetric=False)
    assert w.keyings[Level.L1].n_info == k.n_info
    assert w.keyings[Level.L3].n_info == w.n_decisions  # every L3 set is one history
    uw = (Level.L1, w.keyings[Level.L1].uniform())
    assert math.isclose(evaluate(w, uw, uw)[0], evaluate(tree, u, u)[0], abs_tol=1e-9)


@given(rs=small_lazy_rulesets())
@settings(deadline=None, max_examples=15, suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow])
def test_best_response_matches_reference_at_every_level(rs):
    small(rs)
    lazy = compile_tree(rs, (Level.L1, Level.L2), symmetric=False)
    world = compile_tree(rs, (Level.L1, Level.L3), symmetric=False)
    for opp in (UniformLegal(rs), PlayFirstRandom(rs), Heuristic(rs)):
        sig_lazy = (Level.L1, tabulate(lazy, Level.L1, opp))
        sig_world = (Level.L1, tabulate(world, Level.L1, opp))
        for seat in (0, 1):
            for level in LEVELS:
                ref = best_response_ref(rs, opp, seat, level)
                tree, sig = (world, sig_world) if level is Level.L3 else (lazy, sig_lazy)
                br = best_response(tree, level, sig, seat)
                assert math.isclose(br.value, ref.value, abs_tol=1e-9), (opp.name, seat, level, br.value, ref.value)
                # the pure strategy replays to its value, on the compiled tree and through the engine
                mine, theirs = (level, br.sigma), sig
                prof = (mine, theirs) if seat == 0 else (theirs, mine)
                assert math.isclose(evaluate(tree, *prof)[seat], br.value, abs_tol=1e-9)
                agent = CompiledAgent(tree, level, br.sigma)
                agents, levels = pair(rs, agent, opp, seat, level)
                assert math.isclose(evaluate_ref(rs, agents, levels)[seat], br.value, abs_tol=1e-9)
                assert agent.misses == 0
                # the reference table tabulated onto the keying has the same value
                if level is not Level.L3:
                    t = (level, tabulate_table(tree, level, ref.table))
                    prof = (t, theirs) if seat == 0 else (theirs, t)
                    assert math.isclose(evaluate(tree, *prof)[seat], ref.value, abs_tol=1e-9)
            # L1 best response on the world tree equals the lazy one
            assert math.isclose(
                best_response(world, Level.L1, sig_world, seat).value, best_response(lazy, Level.L1, sig_lazy, seat).value, abs_tol=1e-9
            )


def test_small_ruleset_values_and_moments_match_reference():
    rs = SMALL
    rnd, heu = PlayFirstRandom(rs), Heuristic(rs)
    world = compile_tree(rs, (Level.L1, Level.L3), symmetric=False)
    r, h = (Level.L1, tabulate(world, Level.L1, rnd)), (Level.L1, tabulate(world, Level.L1, heu))
    for a, b, aa, bb in ((r, r, rnd, rnd), (h, r, heu, rnd), (h, h, heu, heu)):
        assert math.isclose(evaluate(world, a, b)[0], evaluate_ref(rs, (aa, bb))[0], abs_tol=1e-9)
        d, dref = outcome_distribution(world, a, b), outcome_distribution_ref(rs, (aa, bb))
        assert all(math.isclose(d[k], dref.get(k, 0.0), abs_tol=1e-9) for k in (1.0, 0.0, -1.0))
        ex, ref = exact_mirrored(world, a, b), exact_mirrored_ref(rs, (aa, bb))
        assert ex.worlds == ref.worlds
        for x, y in ((ex.mean, ref.mean), (ex.var, ref.var), ((ex.cov, ex.splits), (ref.cov, ref.splits))):
            assert all(math.isclose(p, q, abs_tol=1e-9) for p, q in zip(x, y)), (x, y)
    # a best-response row agent (per seat) against the heuristic column, as the mirror script builds it
    brs = [best_response(world, Level.L1, h, seat) for seat in (0, 1)]
    row = BySeat(*(CompiledAgent(world, Level.L1, br.sigma) for br in brs), name="br")
    sig = (Level.L1, np.where(world.keyings[Level.L1].info_player[world.keyings[Level.L1].ia_info] == 0, brs[0].sigma, brs[1].sigma))
    ex, ref = exact_mirrored(world, sig, h), exact_mirrored_ref(rs, (row, heu))
    assert all(math.isclose(p, q, abs_tol=1e-9) for p, q in zip(ex.mean, ref.mean))
    assert math.isclose(ex.cov, ref.cov, abs_tol=1e-9)


def test_exploitability_of_uniform_is_best_response_to_uniform_legal():
    rs = SMALL
    tree = compile_tree(rs, (Level.L1,), symmetric=False)
    k = tree.keyings[Level.L1]
    ex = exploitability(tree, Level.L1, k.uniform())
    for seat in (0, 1):
        assert math.isclose(ex.br[seat].value, best_response_ref(rs, UniformLegal(rs), seat, Level.L1).value, abs_tol=1e-9)
    assert ex.bracket[0] <= ex.bracket[1]
    assert math.isclose(ex.score_points, sum(ex.per_seat), abs_tol=1e-12)
    assert ex.score_points > 0


def test_toy_reproduces_steps_5_6():
    """The eight L1 / L2 best-response values on the full tree of TOY: Random from
    notes.md steps 5-6, the Heuristic from step 8.6 (symmetric tie-breaks; the
    values were pinned with the pure-Python reference solver). Regression for a
    reach bug that every small-ruleset check missed (see notes, step 8). The
    quotient tree has its own copy in test_symmetry."""
    from uno import TOY

    known = {
        ("random", 0, Level.L1): 0.7934, ("random", 0, Level.L2): 0.8188, ("random", 1, Level.L1): 0.5513, ("random", 1, Level.L2): 0.5937,
        ("heuristic", 0, Level.L1): HEURISTIC_TOY[(0, Level.L1)], ("heuristic", 0, Level.L2): HEURISTIC_TOY[(0, Level.L2)],
        ("heuristic", 1, Level.L1): HEURISTIC_TOY[(1, Level.L1)], ("heuristic", 1, Level.L2): HEURISTIC_TOY[(1, Level.L2)],
    }  # fmt: skip
    tree = compile_tree(TOY, (Level.L1, Level.L2), symmetric=False)
    assert not tree.symmetric and tree.n == 1_252_041 and tree.keyings[Level.L1].n_info == 558_656
    for opp in (PlayFirstRandom(TOY), Heuristic(TOY)):
        sig = (Level.L1, tabulate(tree, Level.L1, opp))
        for (name, seat, level), want in known.items():
            if name == opp.name:
                assert abs(best_response(tree, level, sig, seat).score - want) < 5e-5, (name, seat, level)


# best responses to the step 8.6 Heuristic on TOY (scripts/symmetry.py, checked against uno.solve)
HEURISTIC_TOY = {(0, Level.L1): 0.7378, (0, Level.L2): 0.7730, (0, Level.L3): 0.7798, (1, Level.L1): 0.4361, (1, Level.L2): 0.4774, (1, Level.L3): 0.4786}


@given(rs=small_lazy_rulesets())
@settings(deadline=None, max_examples=10, suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow])
def test_keyings_attach_and_perfect_best_response(rs):
    """Compiles of the same tree with different keyings walk the nodes in the same
    order (so a keying can be attached to another compile), and the per-node
    perfect-information best response equals the keyed L3 one."""
    from uno.compiled import best_response_perfect

    assume(rs.deck_size <= 7)  # the world tree enumerates every pile order
    c = census(rs, node_cap=2_000, symmetric=True)
    assume(c.complete)
    fields = ("player", "level", "util", "end", "arc_node", "arc_action", "edge_arc", "edge_child", "edge_prob")
    l1 = compile_tree(rs, (Level.L1,), symmetric=True)
    l2 = compile_tree(rs, (Level.L2,), symmetric=True)
    assert all(np.array_equal(getattr(l1, f), getattr(l2, f)) for f in fields)
    l2.keyings[Level.L1] = l1.keyings[Level.L1]
    both = compile_tree(rs, (Level.L1, Level.L2), symmetric=True)
    w13 = compile_tree(rs, (Level.L1, Level.L3), symmetric=True)
    w1 = compile_tree(rs, (Level.L1,), symmetric=True, world=True)
    assert w1.roots_level is Level.L3
    assert all(np.array_equal(getattr(w13, f), getattr(w1, f)) for f in fields)
    for opp in (PlayFirstRandom(rs), Heuristic(rs)):
        s_attached = (Level.L1, tabulate(l2, Level.L1, opp))
        s_both = (Level.L1, tabulate(both, Level.L1, opp))
        s_w13 = (Level.L1, tabulate(w13, Level.L1, opp))
        s_w1 = (Level.L1, tabulate(w1, Level.L1, opp))
        for seat in (0, 1):
            assert math.isclose(best_response(l2, Level.L2, s_attached, seat).value, best_response(both, Level.L2, s_both, seat).value, abs_tol=1e-9)
            a, b = best_response_perfect(w1, s_w1, seat), best_response(w13, Level.L3, s_w13, seat)
            assert math.isclose(a.value, b.value, abs_tol=1e-9)
            # the per-arc profile (responder's choice, opponent's weights) replays to its value
            v, _ = w1.values(a.sigma, w1.seat_util(seat))
            assert math.isclose(float(v[0, 0]), a.value, abs_tol=1e-9)
