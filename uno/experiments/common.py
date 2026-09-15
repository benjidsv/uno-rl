"""Helpers shared by the experiments: scores, best-response columns, row matchups,
exploitability rows, voluntary-draw counts."""

from __future__ import annotations

import time

import numpy as np

from uno.agents.baseline import Heuristic, PlayFirstRandom
from uno.compiled.cfr import CFRResult
from uno.compiled.solvers import best_response, evaluate
from uno.compiled.strategies import tabulate
from uno.compiled.tree import CompiledTree
from uno.game.observe import Level


def score(v: float) -> float:
    return (v + 1) / 2


def timed(f):
    t0 = time.perf_counter()
    out = f()
    return out, time.perf_counter() - t0


def by_seat(tree: CompiledTree, brs) -> tuple[Level, np.ndarray]:
    k = tree.keyings[Level.L1]
    return (Level.L1, np.where(k.info_player[k.ia_info] == 0, brs[0].sigma, brs[1].sigma))


def columns(lazy: CompiledTree, world: CompiledTree, label: str) -> dict:
    """Best responses at L1 / L2 (lazy tree) and L3 (world tree) to Random and the Heuristic, per seat."""
    rs = lazy.rs
    out = {}
    print(f"\n   {label}: best-response columns (score of the responder)")
    print(f"   {'opponent':10s} {'seat':4s} {'BR L1':>7s} {'BR L2':>7s} {'BR L3':>7s} {'L2-L1':>6s} {'L3-L2':>6s} {'sec':>5s}")
    for opp in (PlayFirstRandom(rs), Heuristic(rs)):
        sl, sw = (Level.L1, tabulate(lazy, Level.L1, opp)), (Level.L1, tabulate(world, Level.L1, opp))
        for seat in (0, 1):
            t0 = time.perf_counter()
            l1 = best_response(lazy, Level.L1, sl, seat)
            l2 = best_response(lazy, Level.L2, sl, seat)
            l3 = best_response(world, Level.L3, sw, seat)
            out[(opp.name, seat)] = (l1, l2, l3)
            print(f"   {opp.name:10s} {seat:<4d} {l1.score:7.4f} {l2.score:7.4f} {l3.score:7.4f} {100 * (l2.score - l1.score):6.1f} {100 * (l3.score - l2.score):6.1f} {time.perf_counter() - t0:5.2f}")
    return out


def matchups(tree: CompiledTree, label: str) -> dict:
    rs = tree.rs
    rnd, heu = PlayFirstRandom(rs), Heuristic(rs)
    r, h = (Level.L1, tabulate(tree, Level.L1, rnd)), (Level.L1, tabulate(tree, Level.L1, heu))
    out = {}
    print(f"\n   {label}: row matchups (score of the first-named, seat 0 / seat 1)")
    for name, a, b in (("random vs random", r, r), ("heuristic vs random", h, r), ("heuristic vs heuristic", h, h)):
        v0, v1 = score(evaluate(tree, a, b)[0]), score(evaluate(tree, b, a)[1])
        out[name] = (v0, v1)
        print(f"   {name:24s} {v0:.4f}  {v1:.4f}")
    return out


def exploitability_rows(tree: CompiledTree, res: CFRResult, cols: dict) -> None:
    rs = tree.rs
    k = tree.keyings[Level.L1]
    rows = {a.name: (Level.L1, tabulate(tree, Level.L1, a)) for a in (PlayFirstRandom(rs), Heuristic(rs))}
    rows["br_l1(random)"] = by_seat(tree, [cols[("random", seat)][0] for seat in (0, 1)])
    rows["nash_l1"] = (Level.L1, res.avg)
    nash_value = score(evaluate(tree, rows["nash_l1"], rows["nash_l1"])[0])
    print(f"\n   Nash seat-0 score {nash_value:.4f} (NashConv {100 * res.final.nashconv:.3f} pts); exploitability of the rows (score points)")
    print(f"   {'row':16s} {'seat':4s} {'keeps vs BR L1':>14s} {'BR to it':>9s} {'Nash (resp.)':>12s} {'exploitability':>14s}")
    for label, s in rows.items():
        for seat in (0, 1):
            br = best_response(tree, Level.L1, s, seat ^ 1)
            nash_resp = 1 - nash_value if seat == 0 else nash_value
            print(f"   {label:16s} {seat:<4d} {1 - br.score:14.4f} {br.score:9.4f} {nash_resp:12.4f} {100 * (br.score - nash_resp):14.2f}")
    print("\n   Nash row and column (exact, score of the first-named, seat 0 / seat 1)")
    for a, b in (("nash_l1", "random"), ("nash_l1", "heuristic"), ("nash_l1", "nash_l1"), ("random", "nash_l1"), ("heuristic", "nash_l1"), ("br_l1(random)", "nash_l1")):
        print(f"   {a + ' vs ' + b:26s} {score(evaluate(tree, rows[a], rows[b])[0]):.4f}  {score(evaluate(tree, rows[b], rows[a])[1]):.4f}")


def voluntary_draws(tree: CompiledTree, seat0, seat1) -> tuple[float, float]:
    """Expected number per game of 'draw with a play in hand' by each seat under the profile."""
    rs = tree.rs
    w = tree.profile_weights(seat0, seat1)
    r = tree.reach(tree.weights(seat0[0], seat0[1], 0), tree.weights(seat1[0], seat1[1], 1))
    reach = r[:, 0] * r[:, 1] * r[:, 2]
    is_draw = tree.arc_action == rs.draw_action
    n_arcs = tree.node_arc_start[1:] - tree.node_arc_start[:-1]
    has_play = (n_arcs[tree.arc_node] > 1) & (tree.arc_player >= 0)  # a draw arc next to a play arc: the draw is voluntary with a play in hand
    mask = is_draw & has_play
    out = []
    for p in (0, 1):
        m = mask & (tree.arc_player == p)
        out.append(float(np.sum(reach[tree.arc_node[m]] * w[m])))
    return (out[0], out[1])


def residual_ties(tree: CompiledTree) -> float:
    """Reach-weighted share of the Heuristic's decisions (against Random) with more than one action in its support."""
    rs = tree.rs
    k = tree.keyings[Level.L1]
    h = (Level.L1, tabulate(tree, Level.L1, Heuristic(rs)))
    r = (Level.L1, tabulate(tree, Level.L1, PlayFirstRandom(rs)))
    total = ties = 0.0
    for seat in (0, 1):
        prof = (h, r) if seat == 0 else (r, h)
        rr = tree.reach(tree.weights(prof[0][0], prof[0][1], 0), tree.weights(prof[1][0], prof[1][1], 1))
        reach = rr[:, 0] * rr[:, 1] * rr[:, 2]
        nodes = np.nonzero(tree.player == seat)[0]
        support = k.info_sum((h[1] > 0).astype(float))[k.node_info[nodes]]
        total += reach[nodes].sum()
        ties += reach[nodes][support > 1].sum()
    return ties / total
