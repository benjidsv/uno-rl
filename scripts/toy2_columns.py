"""Step 8.7: the L2 and L3 columns of TOY2_FULL without a third keying in memory.

    uv run python scripts/toy2_columns.py [ruleset=TOY2_FULL]

Three cached trees, each with one keying: the lazy tree with L1 (the
opponents' strategies and the Nash agent), the lazy tree with L2 (the L2
responder; the L1 keying is attached to it, RULES.md section 16) and the
world tree with L1 (the L3 responder needs no keying, `best_response_perfect`).
Prints BR L1 / L2 / L3 per seat against Random, the Heuristic and the Nash
agent, with the information gaps, next to the forced-play ruleset's.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uno import rules
from uno.agents import Heuristic, PlayFirstRandom
from uno.cfr import CFRResult
from uno.compiled import best_response, best_response_perfect, cache_name, compile_tree, tabulate
from uno.observe import Level

CACHE = Path(__file__).resolve().parent.parent / ".cache"


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "TOY2_FULL"
    rs = getattr(rules, name)
    t0 = time.perf_counter()
    lazy = compile_tree(rs, (Level.L1,), True, cache=CACHE)
    l2 = compile_tree(rs, (Level.L2,), True, cache=CACHE)
    l2.keyings[Level.L1] = lazy.keyings[Level.L1]
    print(f"   lazy L1 {lazy.n:,} nodes, L2 keying {l2.keyings[Level.L2].n_info:,} sets, loaded in {time.perf_counter() - t0:.0f}s")
    opponents = {"random": PlayFirstRandom(rs), "heuristic": Heuristic(rs)}
    res = next((CFRResult.load(p) for p in sorted(CACHE.glob(f"cfr_{cache_name(rs, (Level.L1,), True)}*.npz"))), None)
    sig = {k: (Level.L1, tabulate(lazy, Level.L1, a)) for k, a in opponents.items()}
    if res is not None:
        sig["nash"] = (Level.L1, res.avg)
        opponents["nash"] = res.agent(lazy)
    out = {}
    for k, s in sig.items():
        for seat in (0, 1):
            out[(k, seat)] = [best_response(lazy, Level.L1, s, seat).score, best_response(l2, Level.L2, s, seat).score]
            print(f"   {k:10s} seat {seat}: L1 {out[(k, seat)][0]:.4f} L2 {out[(k, seat)][1]:.4f}", flush=True)
    del l2
    t0 = time.perf_counter()
    world = compile_tree(rs, (Level.L1,), True, cache=CACHE, world=True)
    print(f"   world L1 {world.n:,} nodes, loaded in {time.perf_counter() - t0:.0f}s")
    sigw = {k: (Level.L1, tabulate(world, Level.L1, a)) for k, a in opponents.items()}
    for k, s in sigw.items():
        for seat in (0, 1):
            out[(k, seat)].append(best_response_perfect(world, s, seat).score)
    print(f"\n   {name}: best-response columns (score of the responder)")
    print(f"   {'opponent':10s} {'seat':4s} {'BR L1':>7s} {'BR L2':>7s} {'BR L3':>7s} {'L2-L1':>6s} {'L3-L2':>6s}")
    for (k, seat), (l1, l2v, l3) in out.items():
        print(f"   {k:10s} {seat:<4d} {l1:7.4f} {l2v:7.4f} {l3:7.4f} {100 * (l2v - l1):6.1f} {100 * (l3 - l2v):6.1f}")


if __name__ == "__main__":
    main()
