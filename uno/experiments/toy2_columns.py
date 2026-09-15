"""Step 8.7: the L2 and L3 columns of TOY2_FULL without a third keying in memory.

    uv run uno toy2_columns [ruleset=TOY2_FULL]

Three cached trees, each with one keying: the lazy tree with L1 (the
opponents' strategies and the Nash agent), the lazy tree with L2 (the L2
responder; the L1 keying is attached to it, docs/spec.md section 16) and the
world tree with L1 (the L3 responder needs no keying, `best_response_perfect`).
Prints BR L1 / L2 / L3 per seat against Random, the Heuristic and the Nash
agent, with the information gaps, next to the forced-play ruleset's.
"""

from __future__ import annotations

import sys
import time

from uno import store
from uno.agents.baseline import Heuristic, PlayFirstRandom
from uno.compiled.solvers import best_response, best_response_perfect
from uno.compiled.strategies import transfer
from uno.compiled.tree import compile_tree
from uno.game import rules
from uno.game.observe import Level
from uno.store import CACHE, load_nash


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    name = argv[0] if len(argv) > 0 else "TOY2_FULL"
    rs = getattr(rules, name)
    t0 = time.perf_counter()
    lazy = compile_tree(rs, (Level.L1,), True, cache=CACHE)
    l2 = compile_tree(rs, (Level.L2,), True, cache=CACHE)
    l2.keyings[Level.L1] = lazy.keyings[Level.L1]
    print(f"   lazy L1 {lazy.n:,} nodes, L2 keying {l2.keyings[Level.L2].n_info:,} sets, loaded in {time.perf_counter() - t0:.0f}s")
    opponents = {"random": PlayFirstRandom(rs), "heuristic": Heuristic(rs)}
    res = load_nash(lazy)
    sig = {k: (Level.L1, store.load_or_tabulate(lazy, Level.L1, a)) for k, a in opponents.items()}
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
    sigw = {k: (Level.L1, transfer(lazy, world, Level.L1, s)) for k, (_, s) in sig.items()}  # same L1 sets, numbered differently
    for k, s in sigw.items():
        for seat in (0, 1):
            out[(k, seat)].append(best_response_perfect(world, s, seat).score)
    print(f"\n   {name}: best-response columns (score of the responder)")
    print(f"   {'opponent':10s} {'seat':4s} {'BR L1':>7s} {'BR L2':>7s} {'BR L3':>7s} {'L2-L1':>6s} {'L3-L2':>6s}")
    for (k, seat), (l1, l2v, l3) in out.items():
        print(f"   {k:10s} {seat:<4d} {l1:7.4f} {l2v:7.4f} {l3:7.4f} {100 * (l2v - l1):6.1f} {100 * (l3 - l2v):6.1f}")


if __name__ == "__main__":
    main()
