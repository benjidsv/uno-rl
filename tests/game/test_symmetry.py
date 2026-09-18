"""Color / rank symmetry (uno/game/symmetry.py, docs/spec.md section 18): the group acts on
the engine, canonical forms are invariant, the baseline agents are symmetric, and
the quotient tree reproduces the full tree's values."""

from __future__ import annotations

import math
import random

import numpy as np
from hypothesis import HealthCheck, given, settings

from tests.strategies import CAP, SMALL, pair, rulesets, seeds, small, small_lazy_rulesets, walk
from uno import Level, legal_actions, returns, step
from uno.agents.baseline import Heuristic, PlayFirstRandom, UniformLegal
from uno.agents.symmetrized import Symmetrized
from uno.agents.tabular import CompiledAgent
from uno.compiled.solvers import best_response, evaluate, exploitability
from uno.compiled.strategies import tabulate
from uno.compiled.tree import CompiledTree, compile_tree
from uno.game.keys import KEYS, decode_key, encode_key
from uno.game.observe import observe
from uno.game.rules import Play, Ruleset
from uno.game.symmetry import Symmetry
from uno.reference.census import census
from uno.reference.solve import evaluate as evaluate_ref

LEVELS = (Level.L1, Level.L2, Level.L3)


def profile(tree: CompiledTree, agent):
    return (Level.L1, tabulate(tree, Level.L1, agent))


# -- the group and its action on the engine --------------------------------------


@given(rs=rulesets(), seed=seeds, policy_seed=seeds, g_seed=seeds)
@settings(deadline=None, max_examples=80)
def test_group_acts_on_the_engine(rs, seed, policy_seed, g_seed):
    """g . step(s, a) == step(g . s, g . a); legal actions, returns, keys and
    observations-derived keys commute with the action; inverses invert."""
    sym = Symmetry(rs)
    assert sym.order == math.factorial(rs.colors) * math.prod(math.factorial(len([r for r in rs.rank_copies if r == c])) for c in set(rs.rank_copies) if c > 0)
    rng = random.Random(g_seed)
    states, actions = walk(rs, seed, policy_seed)
    for s, a in zip(states, actions):
        g = rng.randrange(sym.order)
        h = int(sym.inverse[g])
        assert sym.state(h, sym.state(g, s)) == s
        gs = sym.state(g, s)
        assert legal_actions(gs) == tuple(sorted(int(sym.actions[g, x]) for x in legal_actions(s)))
        assert sym.state(g, step(s, a)) == step(gs, int(sym.actions[g, a]))
        for lv in LEVELS:
            for p in (0, 1):
                k = KEYS[lv](s, p)
                assert KEYS[lv](gs, p) == sym.key(lv, g, k)
                assert decode_key(lv, encode_key(lv, k)) == k
    t = states[-1]
    assert returns(sym.state(g, t)) == returns(t)


@given(rs=rulesets(), seed=seeds, policy_seed=seeds, g_seed=seeds)
@settings(deadline=None, max_examples=60)
def test_canonical_form_is_invariant_and_reached(rs, seed, policy_seed, g_seed):
    sym = Symmetry(rs)
    rng = random.Random(g_seed)
    states, _ = walk(rs, seed, policy_seed)
    for s in states:
        if s.terminal:
            continue
        g = rng.randrange(sym.order)
        for lv in LEVELS:
            k = KEYS[lv](s, s.player)
            kc, gc = sym.canonical(lv, k)
            assert sym.key(lv, gc, k) == kc
            assert sym.canonical(lv, sym.key(lv, g, k))[0] == kc
            assert all(sym.key(lv, x, kc) == kc for x in sym.stabilizer(lv, kc))
            assert 0 in sym.stabilizer(lv, kc)
            # the canonical key is the minimum: no element goes below it
            assert all(not (sym.key(lv, x, kc) < kc) for x in range(sym.order))


@given(rs=rulesets(), seed=seeds, policy_seed=seeds, g_seed=seeds)
@settings(deadline=None, max_examples=80)
def test_baseline_agents_are_symmetric(rs, seed, policy_seed, g_seed):
    """policy(g . obs, g . legal) == g . policy(obs, legal) for Random, UniformLegal, Heuristic."""
    sym = Symmetry(rs)
    rng = random.Random(g_seed)
    states, _ = walk(rs, seed, policy_seed)
    agents = (UniformLegal(rs), PlayFirstRandom(rs), Heuristic(rs))
    for s in states:
        if s.terminal:
            continue
        g = rng.randrange(sym.order)
        gs = sym.state(g, s)
        legal, glegal = legal_actions(s), legal_actions(gs)
        for ag in agents:
            want = sym.map_policy(g, ag.policy(observe(s, s.player), legal))
            got = ag.policy(observe(gs, gs.player), glegal)
            assert set(got) == set(want) and all(math.isclose(got[a], want[a]) for a in got), ag.name


class LowestId:
    """Deterministic and *not* symmetric: the lowest playable type id."""

    name = "lowest_id"

    def __init__(self, rs: Ruleset) -> None:
        self.rs = rs

    def policy(self, obs, legal):
        plays = [a for a in legal if isinstance(self.rs.actions[a], Play)]
        return {(plays or legal)[0]: 1.0}


# -- the quotient ------------------------------------------------------------------


@given(rs=small_lazy_rulesets())
@settings(deadline=None, max_examples=30, suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow])
def test_quotient_census_matches_full_census(rs):
    c = small(rs)
    q = census(rs, node_cap=CAP, symmetric=True)
    assert q.complete and q.deals <= c.deals and q.histories <= c.histories
    assert sum(q.info_sets) <= sum(c.info_sets)
    assert math.isclose(q.uniform_legal_p0_score, c.uniform_legal_p0_score, abs_tol=1e-9)
    for key in set(c.end_mass) | set(q.end_mass):
        assert math.isclose(q.end_mass.get(key, 0.0), c.end_mass.get(key, 0.0), abs_tol=1e-9)
    assert q.max_depth == c.max_depth


@given(rs=small_lazy_rulesets())
@settings(deadline=None, max_examples=20, suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow])
def test_quotient_tree_reproduces_full_tree_values(rs):
    """Evaluate, best response at every level, exploitability and playback through
    the engine: quotient == full, for the symmetric baseline agents."""
    small(rs)
    full = {lv: compile_tree(rs, (Level.L1, lv), symmetric=False) for lv in (Level.L2, Level.L3)}
    quot = {lv: compile_tree(rs, (Level.L1, lv), symmetric=True) for lv in (Level.L2, Level.L3)}
    for lv in quot:
        assert quot[lv].symmetric and not full[lv].symmetric
        assert quot[lv].n <= full[lv].n and quot[lv].keyings[Level.L1].n_info <= full[lv].keyings[Level.L1].n_info
        assert math.isclose(float(quot[lv].root_prob.sum()), 1.0, abs_tol=1e-12)
        assert np.all(quot[lv].level[quot[lv].edge_child] > quot[lv].level[quot[lv].arc_node[quot[lv].edge_arc]])
    agents = (UniformLegal(rs), PlayFirstRandom(rs), Heuristic(rs))
    for a in agents:
        for b in agents:
            for lv in quot:
                f, q = full[lv], quot[lv]
                assert math.isclose(evaluate(q, profile(q, a), profile(q, b))[0], evaluate(f, profile(f, a), profile(f, b))[0], abs_tol=1e-9), (a.name, b.name)
    for opp in agents:
        for lv in LEVELS:
            f, q = (full[Level.L3], quot[Level.L3]) if lv is Level.L3 else (full[Level.L2], quot[Level.L2])
            sf, sq = profile(f, opp), profile(q, opp)
            for seat in (0, 1):
                brf, brq = best_response(f, lv, sf, seat), best_response(q, lv, sq, seat)
                assert math.isclose(brq.value, brf.value, abs_tol=1e-9), (opp.name, lv, seat, brq.value, brf.value)
                assert q.keyings[lv].is_symmetric(brq.sigma)
                mine, theirs = (lv, brq.sigma), sq
                prof = (mine, theirs) if seat == 0 else (theirs, mine)
                assert math.isclose(evaluate(q, *prof)[seat], brq.value, abs_tol=1e-9)
                # the quotient strategy played in the real (non-canonical) game
                agent = CompiledAgent(q, lv, brq.sigma)
                ag, lvs = pair(rs, agent, opp, seat, lv)
                assert math.isclose(evaluate_ref(rs, ag, lvs)[seat], brq.value, abs_tol=1e-9)
                assert agent.misses == 0
    f, q = full[Level.L2], quot[Level.L2]
    exf, exq = exploitability(f, Level.L1, f.keyings[Level.L1].uniform()), exploitability(q, Level.L1, q.keyings[Level.L1].uniform())
    assert math.isclose(exq.nashconv, exf.nashconv, abs_tol=1e-9) and math.isclose(exq.value, exf.value, abs_tol=1e-9)


def test_tabulate_rejects_a_non_symmetric_agent():
    from uno import TOY

    tree = compile_tree(TOY, (Level.L1,))
    assert tree.keyings[Level.L1].has_orbits
    tabulate(tree, Level.L1, Heuristic(TOY))
    try:
        tabulate(tree, Level.L1, LowestId(TOY))
    except ValueError as e:
        assert "not symmetric" in str(e)
    else:
        raise AssertionError("an asymmetric agent tabulated on a symmetric tree")


def test_stale_cache_is_recompiled(tmp_path, monkeypatch):
    """A cached tree written under another `CACHE_VERSION`, or with no version (before step 8.9), is rebuilt and overwritten."""
    from uno import store

    rs = SMALL
    path = tmp_path / f"tree_{store.cache_name(rs, (Level.L1,), True, False)}.npz"
    np.savez(path, player=np.zeros(1))  # an unstamped file under the right name
    assert not store.is_current(path)
    first = compile_tree(rs, (Level.L1,), cache=tmp_path)
    assert store.is_current(path) and first.n > 1
    monkeypatch.setattr(store, "CACHE_VERSION", store.CACHE_VERSION + 1)
    assert not store.is_current(path)
    second = compile_tree(rs, (Level.L1,), cache=tmp_path)
    assert store.is_current(path) and second.seconds != first.seconds  # compiled again, not loaded
    assert str(second).split(";")[0] == str(first).split(";")[0]


def test_cache_round_trip(tmp_path):
    rs = SMALL
    tree = compile_tree(rs, (Level.L1, Level.L3))
    tree.save(tmp_path / "t.npz")
    back = CompiledTree.load(tmp_path / "t.npz", rs)
    assert str(back) == str(tree) and back.symmetric
    for f in ("player", "level", "util", "end", "arc_node", "arc_action", "arc_player", "node_arc_start", "edge_arc", "edge_child", "edge_prob", "arc_edge_start"):
        assert np.array_equal(getattr(back, f), getattr(tree, f))
    for lv in tree.keyings:
        assert back.keyings[lv].keys == tree.keyings[lv].keys
        assert np.array_equal(back.keyings[lv].ia_orbit, tree.keyings[lv].ia_orbit)
    heu = Heuristic(rs)
    assert np.array_equal(tabulate(back, Level.L1, heu), tabulate(tree, Level.L1, heu))
    cached = compile_tree(rs, (Level.L1, Level.L3), cache=tmp_path)  # builds and saves
    again = compile_tree(rs, (Level.L1, Level.L3), cache=tmp_path)  # loads
    # `seconds` is compared only between a tree and its own round-trip (above): `again` and `tree`
    # are two *independent* compiles, and `__str__` prints the compile time to a tenth of a second,
    # so comparing the full repr fails whenever the two land either side of a rounding boundary
    # (seen under CPU load: 1.1s against 0.9s). Compare the structural part, as the test above does.
    assert again.seconds == cached.seconds and str(again).split(";")[0] == str(tree).split(";")[0]
    try:
        CompiledTree.load(tmp_path / "t.npz", Ruleset())  # FULL_UNO
    except ValueError:
        pass
    else:
        raise AssertionError("loaded a tree for another ruleset")


def test_toy_quotient_reproduces_the_random_column_and_the_nash_value():
    """Steps 5, 6 and 8 on the quotient of TOY: the twelve best-response values
    (Random from steps 5-6, the Heuristic from step 8.6) and the Nash value 0.6196
    (docs/results/03-08-foundations/08-cfr.md), on 37x fewer histories."""
    from uno import TOY
    from uno.compiled.cfr import cfr

    lazy = compile_tree(TOY, (Level.L1, Level.L2))
    world = compile_tree(TOY, (Level.L1, Level.L3))
    assert lazy.n == 33_647 and lazy.keyings[Level.L1].n_info == 12_036 and len(lazy.roots) == 144
    known = {(0, Level.L1): 0.7934, (0, Level.L2): 0.8188, (0, Level.L3): 0.8280, (1, Level.L1): 0.5513, (1, Level.L2): 0.5937, (1, Level.L3): 0.5989}
    from tests.known import HEURISTIC_TOY

    for agent, values in ((PlayFirstRandom(TOY), known), (Heuristic(TOY), HEURISTIC_TOY)):
        sl, sw = profile(lazy, agent), profile(world, agent)
        for (seat, lv), want in values.items():
            tree, sig = (world, sw) if lv is Level.L3 else (lazy, sl)
            assert abs(best_response(tree, lv, sig, seat).score - want) < 5e-5, (agent.name, seat, lv)
    res = cfr(lazy, iterations=300, log_at=(300,))
    assert res.final.nashconv < 0.005 and abs(res.final.value - 0.6196) < 5e-4
    assert lazy.keyings[Level.L1].is_symmetric(res.avg, 1e-9)


@given(rs=rulesets(), seed=seeds, policy_seed=seeds, g_seed=seeds)
@settings(deadline=None, max_examples=40)
def test_symmetrized_agent_commutes_with_the_group(rs, seed, policy_seed, g_seed):
    """`Symmetrized` turns an asymmetric agent into a symmetric one and leaves a
    symmetric agent unchanged (docs/spec.md section 18)."""
    sym = Symmetry(rs)
    rng = random.Random(g_seed)
    states, _ = walk(rs, seed, policy_seed)
    lowest, heu, sym_heu = Symmetrized(LowestId(rs), rs), Heuristic(rs), Symmetrized(Heuristic(rs), rs)
    for s in states:
        if s.terminal:
            continue
        g = rng.randrange(sym.order)
        gs = sym.state(g, s)
        obs, legal, glegal = observe(s, s.player), legal_actions(s), legal_actions(gs)
        want = sym.map_policy(g, lowest.policy(obs, legal))
        got = lowest.policy(observe(gs, gs.player), glegal)
        assert set(got) == set(want) and all(math.isclose(got[a], want[a]) for a in got)
        assert set(got) <= set(glegal) and math.isclose(sum(got.values()), 1.0)
        h, sh = heu.policy(obs, legal), sym_heu.policy(obs, legal)
        assert set(h) == set(sh) and all(math.isclose(h[a], sh[a]) for a in h)


def test_symmetrized_agent_is_exact_on_the_quotient():
    """A wrapped asymmetric agent tabulates on the quotient tree, and its value there
    against Random equals the full tree's and the engine's."""
    rs = SMALL
    agent, rnd = Symmetrized(LowestId(rs), rs), PlayFirstRandom(rs)
    values = []
    for symmetric in (True, False):
        tree = compile_tree(rs, (Level.L1,), symmetric=symmetric)
        values.append(evaluate(tree, (Level.L1, tabulate(tree, Level.L1, agent)), (Level.L1, tabulate(tree, Level.L1, rnd)))[0])
    assert math.isclose(values[0], values[1], abs_tol=1e-9)
    assert math.isclose(values[0], evaluate_ref(rs, (agent, rnd))[0], abs_tol=1e-9)
