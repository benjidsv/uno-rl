"""Step 8.7: TOY2 solved exactly on the quotient (sizes, ladder columns, Nash).

    uv run python scripts/toy2.py [ruleset=TOY2] [cfr_iterations=1000] [light|full] [variant=cfr+]

`light`: only the L1 keying on the lazy tree (no L2 / L3 columns, no world
tree), for TOY2_FULL whose 28M-node trees do not fit in memory with three
keyings. `variant`: a preset of `uno.cfr.VARIANTS` (scripts/cfr_variants.py
races them).

For the named ruleset (`uno.rules.<name>`): the position-DP sizes of the lazy
and world quotient trees, then the compiled trees (cached in `.cache/`), the
twelve best-response values, the row matchups, CFR+ to the Nash value, the
exploitability of the rows and, when play is not forced, the voluntary-draw
counts, all as in scripts/symmetry.py section 5. Ends with the seat gap of
every profile next to TOY's, which is what TOY2 is for.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.symmetry import CACHE, by_seat, columns, exploitability_rows, matchups, nash, voluntary_draws
from uno import rules
from uno.agents import Heuristic, PlayFirstRandom
from uno.compiled import compile_tree, evaluate, tabulate
from uno.observe import Level
from uno.sizing import size


def score(v: float) -> float:
    return (v + 1) / 2


def columns_l1(lazy, label: str) -> dict:
    """BR L1 only, per seat, against Random and the Heuristic (the `light` mode)."""
    from uno.compiled import best_response

    rs = lazy.rs
    out = {}
    print(f"\n   {label}: best-response column (score of the responder, L1 only)")
    print(f"   {'opponent':10s} {'seat':4s} {'BR L1':>7s} {'sec':>5s}")
    for opp in (PlayFirstRandom(rs), Heuristic(rs)):
        sl = (Level.L1, tabulate(lazy, Level.L1, opp))
        for seat in (0, 1):
            t0 = time.perf_counter()
            l1 = best_response(lazy, Level.L1, sl, seat)
            out[(opp.name, seat)] = (l1,)
            print(f"   {opp.name:10s} {seat:<4d} {l1.score:7.4f} {time.perf_counter() - t0:5.2f}")
    return out


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "TOY2"
    iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    light = len(sys.argv) > 3 and sys.argv[3] == "light"
    variant = sys.argv[4] if len(sys.argv) > 4 else "cfr+"
    rs = getattr(rules, name)
    CACHE.mkdir(exist_ok=True)
    print(f"== {name}: {rs.colors} colors, ranks {rs.rank_copies}, skips {rs.skip_per_color}, +2 {rs.draw_two_per_color}, wilds {rs.wild_count}, hand {rs.hand_size}, forced {rs.forced_play}: deck {rs.deck_size}, pile {rs.deck_size - 2 * rs.hand_size - 1}")

    print("\n== 1. sizes (position DP)")
    print(f"   {'tree':16s} {'roots':>7s} {'positions':>10s} {'histories':>12s} {'decisions':>12s} {'depth':>5s} {'R vs R p0':>9s} {'exh':>5s} {'ties':>5s} {'dec/game':>8s} {'sec':>5s}")
    for label, world in (("quotient lazy", False), ("quotient world", True))[: 1 if light else 2]:
        z = size(rs, symmetric=True, world=world)
        print(f"   {label:16s} {z.roots:7,} {z.positions:10,} {z.histories:12,} {z.decision_nodes:12,} {z.max_depth:5d} {z.random_p0_score:9.4f} {z.exhaustion_share():5.3f} {z.tie_share():5.3f} {z.decisions_per_game[1]:8.2f} {z.seconds:5.1f}")
        sys.stdout.flush()

    print("\n== 2. compiled trees")
    print(f"   {'tree':22s} {'roots':>6s} {'nodes':>10s} {'L1 sets':>9s} {'other':>16s} {'compile s':>9s} {'MB':>7s} {'load s':>6s}")
    trees = {}
    specs = [("quotient lazy L1", (Level.L1,))] if light else [("quotient lazy L1+L2", (Level.L1, Level.L2)), ("quotient world L1+L3", (Level.L1, Level.L3))]
    for label, levels in specs:
        t0 = time.perf_counter()
        tree = compile_tree(rs, levels, True, cache=CACHE)
        t_load = time.perf_counter() - t0
        other = next((k for k in tree.keyings if k is not Level.L1), None)
        other_s = "" if other is None else other.name + " " + f"{tree.keyings[other].n_info:,}"
        print(f"   {label:22s} {len(tree.roots):6d} {tree.n:10,} {tree.keyings[Level.L1].n_info:9,} {other_s:>16s} {tree.seconds:9.1f} {tree.nbytes / 2**20:7.1f} {t_load:6.2f}")
        trees[levels[-1]] = tree
        sys.stdout.flush()
    lazy = trees[Level.L1] if light else trees[Level.L2]
    k = lazy.keyings[Level.L1]
    print(f"   group order {lazy.symmetry.order}; L1 sets with a non-trivial action orbit: {int((k.orbit_size > 1).sum())} orbits")

    print("\n== 3. best-response columns and row matchups")
    if light:
        cols = columns_l1(lazy, name)
    else:
        cols = columns(lazy, trees[Level.L3], name)
    matchups(lazy, name)
    sys.stdout.flush()

    print(f"\n== 4. {variant} on the quotient ({iterations} iterations)")
    res = nash(lazy, name, iterations, verbose=True, variant=variant)
    exploitability_rows(lazy, res, cols)
    sys.stdout.flush()

    rnd = (Level.L1, tabulate(lazy, Level.L1, PlayFirstRandom(rs)))
    heu = (Level.L1, tabulate(lazy, Level.L1, Heuristic(rs)))
    br_r = by_seat(lazy, [cols[("random", seat)][0] for seat in (0, 1)])
    br_h = by_seat(lazy, [cols[("heuristic", seat)][0] for seat in (0, 1)])
    nash_s = (Level.L1, res.avg)
    if not rs.forced_play:
        print("\n   voluntary draws with a play in hand, expected per game (seat 0 / seat 1 of the profile)")
        for label, a, b in (("br_l1(random) vs random", br_r, rnd), ("random vs br_l1(random)", rnd, br_r), ("br_l1(heuristic) vs heuristic", br_h, heu), ("heuristic vs br_l1(heuristic)", heu, br_h), ("nash vs nash", nash_s, nash_s), ("nash vs random", nash_s, rnd), ("random vs nash", rnd, nash_s)):
            d0, d1 = voluntary_draws(lazy, a, b)
            print(f"   {label:32s} {d0:.4f}  {d1:.4f}")

    print("\n== 5. seat gap (seat-0 score minus seat-1 score of the same profile, points)")
    print(f"   {'profile':24s} {'seat 0':>7s} {'seat 1':>7s} {'gap':>6s}   TOY gap")
    toy_gaps = {"random vs random": 29.9, "heuristic vs heuristic": 38.7, "nash vs nash": 23.9, "br_l1(random) vs random": 24.2}
    for label, a, b in (("random vs random", rnd, rnd), ("heuristic vs heuristic", heu, heu), ("nash vs nash", nash_s, nash_s), ("br_l1(random) vs random", br_r, rnd)):
        v0, v1 = score(evaluate(lazy, a, b)[0]), score(evaluate(lazy, b, a)[1])
        print(f"   {label:24s} {v0:7.4f} {v1:7.4f} {100 * (v0 - v1):6.1f}   {toy_gaps[label]:.1f}")


if __name__ == "__main__":
    main()
