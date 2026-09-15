"""Exact edge attribution: where a row agent's score, against a fixed opponent,
is won or lost relative to a reference strategy (normally the best response), by
decision. See docs/plan.md ("An edge-by-decision tool") and docs/spec.md section 21
(step 9.5a, items D2 / D4 of the step 9 review's section D).

**The identity** (performance-difference lemma, Kakade & Langford 2002, adapted to
a fixed-opponent extensive-form game). For two row strategies A ("row"), B ("ref")
against the same fixed opponent O, both playing `seat`:

    V^B - V^A = sum_h rho^A(h) * sum_a sigma_A(a|h) * [V^B(h) - Q^B(h,a)]     (mirror)
              = sum_h rho^B(h) * sum_a sigma_B(a|h) * [Q^A(h,a) - V^A(h)]     (forward)

summed over A's (resp. B's) own decision nodes h. `rho^X(h)` is the **full** reach
probability of h under (X, O, chance) -- unlike a best response's counterfactual
reach, it includes the row player's own action probabilities along the path, since
it is a state-visitation frequency, not a best-response weight (`_reach` below).
`V^X`, `Q^X` are node and arc values under (X, O): well-defined at *every* node of
the tree, not only the ones X actually reaches, because `CompiledTree.values` is a
backward pass over the whole tree regardless of which weights reach a node.

Both identities are exact and, summed over all of A's (B's) own arcs, equal
`V^B - V^A` to numerical precision; the two need not agree term-by-term, since they
group the same total differently (by A's decisions or by B's). When B is a best
response to O, **every mirror term is non-negative**: B maximises the
reach-weighted sum of continuation values at every information set (docs/spec.md
section 16), and A's own reach is a positive common factor within a perfect-recall
set (section 14), so grouping mirror terms by set never changes sign. That is
tested in `tests/compiled/test_edge.py`.

`br_gap` (BR L1's return minus a row's, at `seat`) is the D4 metric: the gap to the
best response against a *fixed* training opponent, luck-free and independent of the
fill-in problems that make raw exploitability numbers upper bounds (docs/spec.md
section 16, step 8.8). `mirror_terms` grouped by `uno.eval.decisions.node_kind` /
`action_class` / `situation` (`uno.store.load_or_label`) is the edge-by-decision
report; `stakes` is a separate, always-defined diagnostic (how much a decision could
possibly be worth), not a term of the identity above.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from uno.compiled.solvers import best_response
from uno.compiled.tree import CompiledTree, Strategy
from uno.eval.estimate import Estimate
from uno.game.observe import Level

if TYPE_CHECKING:
    from uno.compiled.lookup import NodeIndex


def _profile(seat: int, mine: Strategy, opp: Strategy) -> tuple[Strategy, Strategy]:
    return (mine, opp) if seat == 0 else (opp, mine)


def _values(tree: CompiledTree, seat: int, mine: Strategy, opp: Strategy) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(v, av, w): node values, arc values and the combined per-arc profile weight,
    all under (`mine` at `seat`, `opp` at the other seat), seat-`seat` return."""
    w = tree.profile_weights(*_profile(seat, mine, opp))
    v, av = tree.values(w, tree.seat_util(seat))
    return v[:, 0], av[:, 0], w


def _reach(tree: CompiledTree, seat: int, mine: Strategy, opp: Strategy) -> np.ndarray:
    """Full reach probability (chance x mine's own x opp's own) to every node under
    (mine, opp): unlike `best_response`'s counterfactual reach, this *includes* the
    row player's own arc weights, since it is a state-visitation frequency."""
    opp_seat = seat ^ 1
    w_mine = tree.weights(mine[0], mine[1], seat)
    w_opp = tree.weights(opp[0], opp[1], opp_seat)
    w0, w1 = (w_mine, w_opp) if seat == 0 else (w_opp, w_mine)
    r = tree.reach(w0, w1)
    return r[:, 0] * r[:, 1] * r[:, 2]


def _minmax_per_node(tree: CompiledTree, av: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """max / min of `av` over each decision node's own arcs, by level (terminals get 0;
    the level-bounded `reduceat` avoids the out-of-range / empty-segment case a single
    tree-wide `reduceat` would hit at the last, terminal-only level)."""
    hi, lo = np.zeros(tree.n), np.zeros(tree.n)
    for L in tree.levels:
        if L.a1 == L.a0:
            continue
        starts = tree.node_arc_start[L.n0 : L.nd] - L.a0
        hi[L.n0 : L.nd] = np.maximum.reduceat(av[L.a0 : L.a1], starts)
        lo[L.n0 : L.nd] = np.minimum.reduceat(av[L.a0 : L.a1], starts)
    return hi, lo


@dataclass
class EdgeTerms:
    """One direction's exact decomposition. `term[arc]` is non-zero only on the
    grouping player's own arcs (`tree.arc_player == seat`); `total` is their sum,
    which should equal `V^ref - V^row` (mirror: grouped by A/row; forward: by B/ref)."""

    seat: int
    grouped_by: str  # "row" (mirror) or "ref" (forward)
    total: float
    term: np.ndarray  # (m,), return units
    kind: np.ndarray  # (m,), str: node_kind of the arc's node on the grouping player's own arcs, "" elsewhere
    action: np.ndarray  # (m,), str: action_class of the arc, "" elsewhere
    situation: np.ndarray  # (m,), str: situation of the arc's node, "" elsewhere


def _label_arcs(tree: CompiledTree, mine: np.ndarray, kinds: np.ndarray, sits: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Broadcast the L1 keying's per-set-action labels (`uno.eval.decisions.set_labels`
    via `uno.store.load_or_label`) to arcs: node labels via `node_info`, action labels
    via `arc_ia`, empty string off `mine` and at the (-1)-keyed chance/root arc."""
    k = tree.keyings[Level.L1]
    ni = np.where(mine, np.maximum(k.node_info[tree.arc_node], 0), 0)
    node_ok = mine & (k.node_info[tree.arc_node] >= 0)
    kind = np.where(node_ok, kinds[ni], "")
    sit = np.where(node_ok, sits[ni], "")
    ia = np.where(mine, np.maximum(k.arc_ia, 0), 0)
    act_ok = mine & (k.arc_ia >= 0)
    action = np.where(act_ok, actions[ia], "")
    return kind, action, sit


def mirror_terms(tree: CompiledTree, row: Strategy, ref: Strategy, opponent: Strategy, seat: int, labels: tuple) -> EdgeTerms:
    """`sum_a sigma_row(a|h) * [V^ref(h) - Q^ref(h,a)]`, weighted by `rho^row(h)`, per
    arc of `seat` (the mirror direction of the module docstring's identity)."""
    v_ref, av_ref, _ = _values(tree, seat, ref, opponent)
    rho_row = _reach(tree, seat, row, opponent)
    _, _, w_row = _values(tree, seat, row, opponent)
    mine = tree.arc_player == seat
    term = np.zeros(tree.m)
    nodes = tree.arc_node
    term[mine] = rho_row[nodes[mine]] * w_row[mine] * (v_ref[nodes[mine]] - av_ref[mine])
    kind, action, sit = _label_arcs(tree, mine, *labels)
    return EdgeTerms(seat=seat, grouped_by="row", total=float(term.sum()), term=term, kind=kind, action=action, situation=sit)


def forward_terms(tree: CompiledTree, row: Strategy, ref: Strategy, opponent: Strategy, seat: int, labels: tuple) -> EdgeTerms:
    """`sum_a sigma_ref(a|h) * [Q^row(h,a) - V^row(h)]`, weighted by `rho^ref(h)`, per
    arc of `seat` (the forward direction: grouped by the reference's own decisions)."""
    v_row, av_row, _ = _values(tree, seat, row, opponent)
    rho_ref = _reach(tree, seat, ref, opponent)
    _, _, w_ref = _values(tree, seat, ref, opponent)
    mine = tree.arc_player == seat
    term = np.zeros(tree.m)
    nodes = tree.arc_node
    term[mine] = rho_ref[nodes[mine]] * w_ref[mine] * (av_row[mine] - v_row[nodes[mine]])
    kind, action, sit = _label_arcs(tree, mine, *labels)
    return EdgeTerms(seat=seat, grouped_by="ref", total=float(term.sum()), term=term, kind=kind, action=action, situation=sit)


def br_gap(tree: CompiledTree, row: Strategy, opponent: Strategy, seat: int) -> float:
    """BR L1's return to `opponent` minus `row`'s return, at `seat` (return units, D4
    metric: the exact, luck-free gap to the ceiling against a *fixed* opponent)."""
    br = best_response(tree, Level.L1, opponent, seat)
    v_row, _, _ = _values(tree, seat, row, opponent)
    return br.value - v_row[0]


@dataclass
class Stakes:
    """How much a decision could possibly be worth under the *reference* strategy's
    values: `max_a Q^ref(h,a) - min_a Q^ref(h,a)` at every `seat`-owned decision node,
    weighted by `rho^row(h)` -- a diagnostic, not a term of the `mirror_terms` /
    `forward_terms` identity, and always >= 0 by construction."""

    seat: int
    node: np.ndarray  # decision-node ids, seat-owned
    weight: np.ndarray  # rho^row(h) at those nodes
    spread: np.ndarray  # max_a Q^ref - min_a Q^ref at those nodes


def stakes(tree: CompiledTree, row: Strategy, ref: Strategy, opponent: Strategy, seat: int) -> Stakes:
    _, av_ref, _ = _values(tree, seat, ref, opponent)
    rho_row = _reach(tree, seat, row, opponent)
    hi, lo = _minmax_per_node(tree, av_ref)
    node = np.nonzero(tree.player == seat)[0]
    return Stakes(seat=seat, node=node, weight=rho_row[node], spread=(hi - lo)[node])


@dataclass
class WildSplit:
    """At wild-kind decisions only (`uno.eval.decisions.node_kind == "wild"`), the
    reference's values split into a "spend" part (best action overall minus best
    wild) and a "color" part (best wild minus worst wild): both >= 0 by construction.
    A diagnostic, like `Stakes`, not a term of the exact identity."""

    seat: int
    node: np.ndarray
    weight: np.ndarray
    spend: np.ndarray
    color: np.ndarray


def wild_split(tree: CompiledTree, row: Strategy, ref: Strategy, opponent: Strategy, seat: int, kinds: np.ndarray) -> WildSplit:
    """Vectorized over arcs: a small (num_actions,) lookup for "is this action id a
    wild play", broadcast to arcs via `k.ia_action`, then a `np.maximum.at` /
    `np.minimum.at` scatter per node (no per-arc Python loop, so this stays cheap
    even on `TOY2_FULL`'s ~8M set-actions)."""
    from uno.game.rules import Play

    rs = tree.rs
    types = rs.card_types
    _, av_ref, _ = _values(tree, seat, ref, opponent)
    rho_row = _reach(tree, seat, row, opponent)
    k = tree.keyings[Level.L1]
    hi_all, _ = _minmax_per_node(tree, av_ref)
    is_wild_action = np.array([isinstance(a, Play) and types[a.card].is_wild for a in rs.actions])
    mine = tree.arc_player == seat
    ia = np.where(mine, k.arc_ia, 0)
    info = np.where(mine, k.ia_info[ia], -1)
    is_wild_info = np.zeros(k.n_info, dtype=bool)
    is_wild_info[np.nonzero((kinds == "wild") & (k.info_player == seat))[0]] = True
    is_wild_action_arc = is_wild_action[np.where(mine, k.ia_action[ia], 0)]
    wild_arc = mine & np.where(info >= 0, is_wild_info[np.maximum(info, 0)], False) & is_wild_action_arc
    nodes = tree.arc_node[wild_arc]
    vals = av_ref[wild_arc]
    if len(nodes) == 0:
        empty = np.array([], dtype=np.int64)
        return WildSplit(seat=seat, node=empty, weight=np.array([]), spend=np.array([]), color=np.array([]))
    uniq, inv = np.unique(nodes, return_inverse=True)
    wild_hi = np.full(len(uniq), -np.inf)
    wild_lo = np.full(len(uniq), np.inf)
    np.maximum.at(wild_hi, inv, vals)
    np.minimum.at(wild_lo, inv, vals)
    spend = hi_all[uniq] - wild_hi
    color = wild_hi - wild_lo
    return WildSplit(seat=seat, node=uniq, weight=rho_row[uniq], spend=spend, color=color)


@dataclass
class SampledMirror:
    """A sampled cross-check of `mirror_terms`' exact total (step 9.5a): played
    games under (row, opponent), summing at each of row's own decisions
    `sum_a sigma_row(a|h) * [V^ref(h) - Q^ref(h,a)]`, read off `tree`'s own node /
    arc values under (ref, opponent) via `uno.compiled.lookup.NodeIndex` -- the
    same per-node quantity `mirror_terms` computes exactly, so its game-by-game
    sample mean must equal the exact total to within sampling noise: a seed
    samples decision nodes with probability exactly `rho_row`, so summing this
    quantity over one game's own decisions and averaging over games is an
    unbiased estimator of `sum_h rho_row(h) * sum_a ... ` = the exact total."""

    n: int
    total: Estimate
    by_kind: dict[str, Estimate]  # over the SAME n games (zero on games with none of that kind)


def sampled_mirror(tree: CompiledTree, idx: NodeIndex, row_agent, opp_agent, ref: Strategy, seat: int, seeds) -> SampledMirror:
    """See `SampledMirror`. `idx`: a `uno.compiled.lookup.NodeIndex` built on `tree`
    (needs the L2 keying)."""
    from uno import store
    from uno.eval.decisions import NODE_KINDS, node_kind
    from uno.eval.estimate import estimate
    from uno.eval.play import play_game
    from uno.game.engine import legal_actions

    opp_sigma = (Level.L1, store.load_or_tabulate(tree, Level.L1, opp_agent))
    v_ref, av_ref, _ = _values(tree, seat, ref, opp_sigma)
    agents = (row_agent, opp_agent) if seat == 0 else (opp_agent, row_agent)
    totals: list[float] = []
    by_kind_totals: dict[str, list[float]] = {k: [] for k in NODE_KINDS}
    for seed in seeds:
        try:
            t = play_game(tree.rs, seed, agents)
        except ValueError:
            continue
        total = 0.0
        kind_totals = {k: 0.0 for k in NODE_KINDS}
        for st in t.steps:
            if st.state.player != seat:
                continue
            legal = legal_actions(st.state)
            kind = node_kind(tree.rs, st.obs, legal)
            node = idx.node(st.state)
            term = 0.0
            for a, p in st.probs.items():
                if p == 0.0:
                    continue
                arc = idx.arc(st.state, a)
                term += p * (v_ref[node] - av_ref[arc])
            total += term
            kind_totals[kind] += term
        totals.append(total)
        for k in NODE_KINDS:
            by_kind_totals[k].append(kind_totals[k])
    return SampledMirror(n=len(totals), total=estimate(totals), by_kind={k: estimate(v) for k, v in by_kind_totals.items()})
