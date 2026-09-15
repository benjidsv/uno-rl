"""Exact edge attribution (`uno.compiled.edge`, step 9.5a): the two
performance-difference identities, non-negativity of the mirror terms against a
best response, and agreement of the grouped totals between the quotient and the
full tree."""

from __future__ import annotations

import numpy as np

from tests.strategies import SMALL
from uno.agents.baseline import Heuristic, PlayFirstRandom
from uno.compiled.edge import br_gap, forward_terms, mirror_terms, stakes, wild_split
from uno.compiled.solvers import best_response, evaluate
from uno.compiled.strategies import tabulate
from uno.compiled.tree import compile_tree
from uno.eval.decisions import ACTION_CLASSES, NODE_KINDS
from uno.game.observe import Level
from uno.store import load_or_label


def _profile(seat: int, mine, opp):
    return (mine, opp) if seat == 0 else (opp, mine)


def _random_strategy(k, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return k.normalize(rng.random(k.n_ia))


def test_identities_equal_the_exact_value_gap_and_vanish_at_self_comparison():
    tree = compile_tree(SMALL, (Level.L1,), symmetric=False)
    k = tree.keyings[Level.L1]
    labels = load_or_label(tree)
    opp = (Level.L1, tabulate(tree, Level.L1, PlayFirstRandom(SMALL)))
    row = (Level.L1, _random_strategy(k, seed=1))
    ref = (Level.L1, _random_strategy(k, seed=2))
    for seat in (0, 1):
        v_ref = evaluate(tree, *_profile(seat, ref, opp))[seat]
        v_row = evaluate(tree, *_profile(seat, row, opp))[seat]
        gap = v_ref - v_row
        m = mirror_terms(tree, row, ref, opp, seat, labels)
        f = forward_terms(tree, row, ref, opp, seat, labels)
        assert abs(m.total - gap) < 1e-9, (seat, m.total, gap)
        assert abs(f.total - gap) < 1e-9, (seat, f.total, gap)
        # a profile against itself: no gap. Individual arc terms need not vanish (an
        # arc with little of the row's own weight can still show a nonzero deviation
        # from the node's value), but every node's own terms sum to zero: `values`'s
        # backward pass makes v[node] the w-weighted average of its arcs' av, so
        # sum_a w(a) * (v[node] - av[a]) telescopes to zero at every node.
        m0 = mirror_terms(tree, row, row, opp, seat, labels)
        assert abs(m0.total) < 1e-9
        mine = tree.arc_player == seat
        per_node = np.bincount(tree.arc_node[mine], weights=m0.term[mine], minlength=tree.n)
        assert np.allclose(per_node, 0.0, atol=1e-9)


def test_br_gap_matches_best_response_minus_row():
    tree = compile_tree(SMALL, (Level.L1,), symmetric=False)
    opp = (Level.L1, tabulate(tree, Level.L1, PlayFirstRandom(SMALL)))
    row = (Level.L1, tabulate(tree, Level.L1, Heuristic(SMALL)))
    for seat in (0, 1):
        br = best_response(tree, Level.L1, opp, seat)
        v_row = evaluate(tree, *_profile(seat, row, opp))[seat]
        assert abs(br_gap(tree, row, opp, seat) - (br.value - v_row)) < 1e-9


def test_mirror_terms_are_nonnegative_when_ref_is_a_best_response():
    """B a best response to O: every mirror set-action total (grouped by A's own
    information sets) is >= 0, since B maximises the reach-weighted continuation
    value at every set and A's own reach is a positive common factor within a
    perfect-recall set."""
    tree = compile_tree(SMALL, (Level.L1,), symmetric=False)
    k = tree.keyings[Level.L1]
    labels = load_or_label(tree)
    opp = (Level.L1, tabulate(tree, Level.L1, PlayFirstRandom(SMALL)))
    row = (Level.L1, tabulate(tree, Level.L1, Heuristic(SMALL)))
    for seat in (0, 1):
        br = best_response(tree, Level.L1, opp, seat)
        ref = (Level.L1, br.sigma)
        m = mirror_terms(tree, row, ref, opp, seat, labels)
        mine = tree.arc_player == seat
        set_id = k.node_info[tree.arc_node]
        totals = np.bincount(np.where(mine, set_id, 0), weights=np.where(mine, m.term, 0.0), minlength=k.n_info)
        seat_sets = np.nonzero(k.info_player == seat)[0]
        assert np.all(totals[seat_sets] >= -1e-9), totals[seat_sets].min()


def test_labels_are_within_the_declared_vocabularies():
    tree = compile_tree(SMALL, (Level.L1,), symmetric=False)
    kinds, sits, actions = load_or_label(tree)
    assert set(kinds) <= set(NODE_KINDS)
    assert set(actions) <= set(ACTION_CLASSES)
    assert all(s.count("/") == 2 for s in sits)  # "seat/opp/pile"


def test_stakes_and_wild_split_are_nonnegative():
    """Both diagnostics are built from a max and a min (or a max and a subset max),
    so they must be non-negative by construction, on every seat's decision nodes."""
    tree = compile_tree(SMALL, (Level.L1,), symmetric=False)
    kinds, _, _ = load_or_label(tree)
    opp = (Level.L1, tabulate(tree, Level.L1, PlayFirstRandom(SMALL)))
    row = (Level.L1, tabulate(tree, Level.L1, Heuristic(SMALL)))
    saw_wild = False
    for seat in (0, 1):
        br = best_response(tree, Level.L1, opp, seat)
        ref = (Level.L1, br.sigma)
        st = stakes(tree, row, ref, opp, seat)
        assert len(st.node) > 0
        assert np.all(st.spread >= -1e-9)
        assert np.all(st.weight >= -1e-12)
        ws = wild_split(tree, row, ref, opp, seat, kinds)
        if len(ws.node) > 0:
            saw_wild = True
            assert np.all(ws.spend >= -1e-9)
            assert np.all(ws.color >= -1e-9)
    assert saw_wild  # SMALL has a wild, so this must exercise the wild branch


def test_quotient_and_full_tree_agree_on_grouped_totals():
    """Symmetric agents (Random, the Heuristic, BR L1 to Random) give the same
    br_gap and the same per-kind mirror totals whether computed on the full tree or
    on the quotient, since the mirror term is a property of the abstract game, not
    of which tree represents it."""
    full = compile_tree(SMALL, (Level.L1,), symmetric=False)
    quot = compile_tree(SMALL, (Level.L1,), symmetric=True)
    full_labels, quot_labels = load_or_label(full), load_or_label(quot)
    opp_full = (Level.L1, tabulate(full, Level.L1, PlayFirstRandom(SMALL)))
    opp_quot = (Level.L1, tabulate(quot, Level.L1, PlayFirstRandom(SMALL)))
    row_full = (Level.L1, tabulate(full, Level.L1, Heuristic(SMALL)))
    row_quot = (Level.L1, tabulate(quot, Level.L1, Heuristic(SMALL)))
    for seat in (0, 1):
        br_full = best_response(full, Level.L1, opp_full, seat)
        br_quot = best_response(quot, Level.L1, opp_quot, seat)
        assert abs(br_full.value - br_quot.value) < 1e-9
        gap_full = br_gap(full, row_full, opp_full, seat)
        gap_quot = br_gap(quot, row_quot, opp_quot, seat)
        assert abs(gap_full - gap_quot) < 1e-9
        m_full = mirror_terms(full, row_full, (Level.L1, br_full.sigma), opp_full, seat, full_labels)
        m_quot = mirror_terms(quot, row_quot, (Level.L1, br_quot.sigma), opp_quot, seat, quot_labels)
        by_kind_full = {kind: float(m_full.term[m_full.kind == kind].sum()) for kind in set(m_full.kind) - {""}}
        by_kind_quot = {kind: float(m_quot.term[m_quot.kind == kind].sum()) for kind in set(m_quot.kind) - {""}}
        for kind in set(by_kind_full) | set(by_kind_quot):
            assert abs(by_kind_full.get(kind, 0.0) - by_kind_quot.get(kind, 0.0)) < 1e-9, kind
