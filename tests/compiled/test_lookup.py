"""State -> node/arc lookup (`uno.compiled.lookup`, step 9.5a): every state along
random games is found, the looked-up node's value matches an independent
reference, and on the quotient every node of an L2 set shares that value."""

from __future__ import annotations

import math

import numpy as np

from tests.strategies import SMALL, walk
from uno.agents.baseline import Heuristic, PlayFirstRandom
from uno.compiled.lookup import build
from uno.compiled.strategies import tabulate
from uno.compiled.tree import compile_tree
from uno.game.chance import chance_outcomes, children
from uno.game.engine import legal_actions
from uno.game.observe import Level
from uno.reference.solve import evaluate as evaluate_ref


def _setup(symmetric: bool):
    tree = compile_tree(SMALL, (Level.L1, Level.L2), symmetric=symmetric)
    idx = build(tree, Level.L2)
    heu, rnd = Heuristic(SMALL), PlayFirstRandom(SMALL)
    sl_heu = (Level.L1, tabulate(tree, Level.L1, heu))
    sl_rnd = (Level.L1, tabulate(tree, Level.L1, rnd))
    w = tree.profile_weights(sl_heu, sl_rnd)
    v, av = tree.values(w, tree.seat_util(0))
    return tree, idx, heu, rnd, v, av


def test_every_state_is_found_and_legal_actions_match():
    for symmetric in (False, True):
        tree, idx, *_ = _setup(symmetric)
        k = tree.keyings[Level.L2]
        for seed in range(40):
            states, _ = walk(SMALL, seed, policy_seed=seed + 200)
            for s in states:
                if s.terminal:
                    continue
                node = idx.node(s)
                a0, a1 = tree.node_arc_start[node], tree.node_arc_start[node + 1]
                # tree.arc_action is in the node's own frame (real, un-relabelled below a
                # canonical deal), not a frame shared across a set's nodes: compare via
                # k.ia_action[k.arc_ia[...]] instead, exactly what idx.arc() matches on.
                node_actions = set(int(a) for a in k.ia_action[k.arc_ia[a0:a1]])
                g = idx._lookup(s)[1]
                want = set(legal_actions(s)) if idx.sym is None else {int(idx.sym.actions[g, a]) for a in legal_actions(s)}
                assert node_actions == want
        assert (k.node_info >= 0).sum() > 0  # sanity: the keying actually has decision nodes


def test_node_value_matches_reference():
    for symmetric in (False, True):
        tree, idx, heu, rnd, v, av = _setup(symmetric)
        for seed in range(20):
            states, _ = walk(SMALL, seed, policy_seed=seed + 700)
            for s in states:
                if s.terminal:
                    continue
                node = idx.node(s)
                got = v[node, 0]
                ref = evaluate_ref(SMALL, (heu, rnd), roots=[(s, 1.0)], expand=children)[0]
                assert math.isclose(got, ref, abs_tol=1e-9), (symmetric, seed)


def test_arc_q_value_matches_reference_including_chance_branching_actions():
    for symmetric in (False, True):
        tree, idx, heu, rnd, v, av = _setup(symmetric)
        for seed in range(15):
            states, _ = walk(SMALL, seed, policy_seed=seed + 900)
            for s in states:
                if s.terminal:
                    continue
                for a in legal_actions(s):
                    arc = idx.arc(s, a)
                    got = av[arc, 0]
                    outs = chance_outcomes(s, a)
                    ref = sum(p * evaluate_ref(SMALL, (heu, rnd), roots=[(c, 1.0)], expand=children)[0] for c, p in outs)
                    assert math.isclose(got, ref, abs_tol=1e-9), (symmetric, seed, a)


def test_quotient_l2_set_nodes_share_the_value():
    tree, idx, heu, rnd, v, av = _setup(True)
    k = tree.keyings[Level.L2]
    dec = np.nonzero(k.node_info >= 0)[0]
    sets = k.node_info[dec]
    order = np.argsort(sets)
    dec_sorted, sets_sorted = dec[order], sets[order]
    boundaries = np.nonzero(np.diff(sets_sorted))[0] + 1
    groups = np.split(dec_sorted, boundaries)
    checked = 0
    for grp in groups:
        if len(grp) > 1:
            vals = v[grp, 0]
            assert np.allclose(vals, vals[0], atol=1e-12)
            checked += 1
    assert checked > 0  # sanity: at least one L2 set actually has multiple nodes on the quotient
