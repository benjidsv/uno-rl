"""Step 8.6: the symmetric (quotient) compiled tree, and TOY_FULL solved exactly.

    uv run uno symmetry [cfr_iterations=1000]

Prints:

1. sizes and costs on TOY: full tree vs quotient (color x rank orbits, docs/foundations/spec.md
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

from uno.agents.baseline import Heuristic, PlayFirstRandom
from uno.compiled.strategies import tabulate
from uno.compiled.tree import compile_tree
from uno.experiments.common import by_seat, columns, exploitability_rows, matchups, residual_ties, timed, voluntary_draws
from uno.game import rules
from uno.game.observe import Level
from uno.store import CACHE, cache_name, load_or_solve_nash


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    iterations = int(argv[0]) if len(argv) > 0 else 1000
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
    res = load_or_solve_nash(lazy, iterations, verbose=True)
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
    res_f = load_or_solve_nash(lazy_f, iterations, verbose=True)
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
