"""Step 8.6: the symmetric (quotient) compiled tree, and TOY_FULL solved exactly.

    uv run python scripts/symmetry.py [cfr_iterations=1000]

Prints:

1. sizes and costs on TOY: full tree vs quotient (color x rank orbits, RULES.md
   section 18), compile seconds, bytes, load-from-cache seconds;
2. the twelve best-response values and the row matchups of steps 5-6 on both
   trees (the Heuristic has symmetric tie-breaks from this step on, so its
   values differ from the earlier notes; the Random column must not move);
3. CFR+ on the quotient: the exploitability curve and the Nash value (step 8
   found 0.6196 in 203 s on the full tree), and the exploitability of the
   row agents;
4. the Heuristic's residual ties: the share of its decisions (reach-weighted,
   against Random) where it plays uniformly over several cards;
5. TOY_FULL (no forced play, unreachable before this step): sizes, the
   best-response columns, the row matchups, Nash L1, the exploitability of the
   rows, and how often a best responder and the Nash agent draw with a play
   in hand, which is the worth of the voluntary draw that step 8.5 asked for.

Trees are cached in `.cache/` (`tree_*.npz`), CFR results too.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from uno import rules
from uno.agents import Heuristic, PlayFirstRandom
from uno.cfr import CFRResult, cfr
from uno.compiled import CompiledTree, best_response, cache_name, compile_tree, evaluate, tabulate
from uno.observe import Level
from uno.rules import Draw, Ruleset

CACHE = Path(__file__).resolve().parent.parent / ".cache"


def score(v: float) -> float:
    return (v + 1) / 2


def timed(f):
    t0 = time.perf_counter()
    out = f()
    return out, time.perf_counter() - t0


def nash(tree: CompiledTree, name: str, iterations: int, verbose: bool, variant: str = "cfr+") -> CFRResult:
    suffix = "" if variant == "cfr+" else f"_{variant.replace('+', 'p')}"
    path = CACHE / f"cfr_{cache_name(tree.rs, (Level.L1,), tree.symmetric)}{suffix}.npz"
    if path.exists():
        res = CFRResult.load(path)
        if res.iterations >= iterations and len(res.avg) == tree.keyings[Level.L1].n_ia:
            print(f"   loaded {path.name}: {res.iterations} iterations, NashConv {100 * res.final.nashconv:.3f} pts, {res.seconds:.0f}s")
            return res
    res = cfr(tree, iterations=iterations, verbose=verbose, variant=variant)
    res.save(path)
    print(f"   {variant} {res.iterations} iterations in {res.seconds:.1f}s ({1000 * res.seconds / res.iterations:.1f} ms each), NashConv {100 * res.final.nashconv:.3f} pts, saved to {path.name}")
    return res


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


def main() -> None:
    iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    CACHE.mkdir(exist_ok=True)
    rs = rules.TOY

    print("== 1. TOY: full tree vs quotient")
    print(f"   {'tree':22s} {'roots':>6s} {'nodes':>9s} {'L1 sets':>8s} {'other':>14s} {'compile s':>9s} {'MB':>6s} {'load s':>6s}")
    trees = {}
    for label, levels, symmetric in (("full lazy L1+L2", (Level.L1, Level.L2), False), ("quotient lazy L1+L2", (Level.L1, Level.L2), True), ("full world L1+L3", (Level.L1, Level.L3), False), ("quotient world L1+L3", (Level.L1, Level.L3), True)):
        path = CACHE / f"tree_{cache_name(rs, levels, symmetric)}.npz"
        if path.exists():
            path.unlink()
        tree, _ = timed(lambda: compile_tree(rs, levels, symmetric, cache=CACHE))
        _, tl = timed(lambda: compile_tree(rs, levels, symmetric, cache=CACHE))
        other = next(k for k in tree.keyings if k is not Level.L1)
        print(f"   {label:22s} {len(tree.roots):6d} {tree.n:9d} {tree.keyings[Level.L1].n_info:8d} {other.name + ' ' + str(tree.keyings[other].n_info):>14s} {tree.seconds:9.1f} {tree.nbytes / 2**20:6.1f} {tl:6.2f}")
        trees[(symmetric, other)] = tree
        sys.stdout.flush()
    print(f"   group order {trees[(True, Level.L2)].symmetry.order}; sets with a non-trivial action orbit: {int((trees[(True, Level.L2)].keyings[Level.L1].orbit_size > 1).sum())} orbits over {trees[(True, Level.L2)].keyings[Level.L1].n_info} L1 sets")

    print("\n== 2. steps 5-6 on both trees (Heuristic with symmetric tie-breaks)")
    cols = {}
    for symmetric in (False, True):
        label = "quotient" if symmetric else "full"
        cols[symmetric] = columns(trees[(symmetric, Level.L2)], trees[(symmetric, Level.L3)], label)
        matchups(trees[(symmetric, Level.L2)], label)
        sys.stdout.flush()
    del trees[(False, Level.L2)], trees[(False, Level.L3)]

    print(f"\n== 3. CFR+ on the quotient of TOY ({iterations} iterations)")
    lazy = trees[(True, Level.L2)]
    res = nash(lazy, "TOY", iterations, verbose=True)
    exploitability_rows(lazy, res, cols[True])
    sys.stdout.flush()

    print("\n== 4. the Heuristic's residual ties on TOY (reach-weighted share of its decisions against Random)")
    print(f"   {100 * residual_ties(lazy):.2f} % of decisions are played uniformly over 2+ cards")

    print("\n== 5. TOY_FULL on the quotient (no forced play)")
    rs = rules.TOY_FULL
    print(f"   {'tree':22s} {'roots':>6s} {'nodes':>9s} {'L1 sets':>8s} {'other':>14s} {'compile s':>9s} {'MB':>6s}")
    full_trees = {}
    for label, levels in (("quotient lazy L1+L2", (Level.L1, Level.L2)), ("quotient world L1+L3", (Level.L1, Level.L3))):
        tree = compile_tree(rs, levels, True, cache=CACHE)
        other = next(k for k in tree.keyings if k is not Level.L1)
        print(f"   {label:22s} {len(tree.roots):6d} {tree.n:9d} {tree.keyings[Level.L1].n_info:8d} {other.name + ' ' + str(tree.keyings[other].n_info):>14s} {tree.seconds:9.1f} {tree.nbytes / 2**20:6.1f}")
        full_trees[other] = tree
        sys.stdout.flush()
    lazy_f, world_f = full_trees[Level.L2], full_trees[Level.L3]
    cols_f = columns(lazy_f, world_f, "TOY_FULL")
    matchups(lazy_f, "TOY_FULL")
    sys.stdout.flush()
    res_f = nash(lazy_f, "TOY_FULL", iterations, verbose=True)
    exploitability_rows(lazy_f, res_f, cols_f)
    print("\n   voluntary draws with a play in hand, expected per game (seat 0 / seat 1 of the profile)")
    rnd = (Level.L1, tabulate(lazy_f, Level.L1, PlayFirstRandom(rs)))
    heu = (Level.L1, tabulate(lazy_f, Level.L1, Heuristic(rs)))
    br_r = by_seat(lazy_f, [cols_f[("random", seat)][0] for seat in (0, 1)])
    br_h = by_seat(lazy_f, [cols_f[("heuristic", seat)][0] for seat in (0, 1)])
    nash_f = (Level.L1, res_f.avg)
    for label, a, b in (("br_l1(random) vs random", br_r, rnd), ("random vs br_l1(random)", rnd, br_r), ("br_l1(heuristic) vs heuristic", br_h, heu), ("heuristic vs br_l1(heuristic)", heu, br_h), ("nash vs nash", nash_f, nash_f), ("nash vs random", nash_f, rnd), ("random vs nash", rnd, nash_f)):
        d0, d1 = voluntary_draws(lazy_f, a, b)
        print(f"   {label:32s} {d0:.4f}  {d1:.4f}")
    print("\n   TOY_FULL minus TOY, same responder and opponent (score points): the worth of the voluntary draw")
    print(f"   {'opponent':10s} {'seat':4s} {'BR L1':>7s} {'BR L2':>7s} {'BR L3':>7s}")
    for opp in ("random", "heuristic"):
        for seat in (0, 1):
            d = [100 * (cols_f[(opp, seat)][i].score - cols[True][(opp, seat)][i].score) for i in range(3)]
            print(f"   {opp:10s} {seat:<4d} {d[0]:7.2f} {d[1]:7.2f} {d[2]:7.2f}")


if __name__ == "__main__":
    main()
