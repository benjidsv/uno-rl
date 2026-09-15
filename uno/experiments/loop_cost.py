"""Step 8.7: what one checkpoint of the step 10-11 loop costs on the quotient trees.

    uv run uno loop_cost

Per checkpoint the loop needs: load the cached tree, tabulate the agent over
the L1 keying (the only Python loop left, one policy call per canonical set),
its exploitability (two best responses), and, once per ruleset, CFR+ for the
Nash value. Timed on TOY and TOY_FULL with the Heuristic standing in for the
checkpoint (an RL policy costs one batched forward pass instead of 139k
Python calls, so the tabulate row is an upper bound).
"""

from __future__ import annotations

import sys
import time

from uno.agents.baseline import Heuristic, PlayFirstRandom
from uno.compiled.cfr import cfr
from uno.compiled.solvers import best_response, evaluate, exploitability
from uno.compiled.strategies import tabulate
from uno.compiled.tree import compile_tree
from uno.game import rules
from uno.game.observe import Level
from uno.store import CACHE


def timed(f, repeat: int = 1):
    t0 = time.perf_counter()
    for _ in range(repeat):
        out = f()
    return out, (time.perf_counter() - t0) / repeat


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    print(f"{'ruleset':9s} {'nodes':>9s} {'L1 sets':>8s} {'load s':>7s} {'tabulate s':>10s} {'per set us':>10s} {'BR s':>6s} {'exploit s':>9s} {'evaluate s':>10s} {'CFR+ it ms':>10s} {'nodes/s (CFR)':>13s}")
    for name, rs in (("TOY", rules.TOY), ("TOY_FULL", rules.TOY_FULL)):
        tree, t_load = timed(lambda: compile_tree(rs, (Level.L1,), True, cache=CACHE))
        k = tree.keyings[Level.L1]
        heu = Heuristic(rs)
        sig, t_tab = timed(lambda: tabulate(tree, Level.L1, heu))
        rnd = tabulate(tree, Level.L1, PlayFirstRandom(rs))
        _, t_br = timed(lambda: best_response(tree, Level.L1, (Level.L1, sig), 0), repeat=5)
        _, t_ex = timed(lambda: exploitability(tree, Level.L1, sig), repeat=5)
        _, t_ev = timed(lambda: evaluate(tree, (Level.L1, sig), (Level.L1, rnd)), repeat=5)
        res, t_cfr = timed(lambda: cfr(tree, iterations=20))
        it_ms = 1000 * t_cfr / 20
        print(f"{name:9s} {tree.n:9,} {k.n_info:8,} {t_load:7.2f} {t_tab:10.2f} {1e6 * t_tab / k.n_info:10.1f} {t_br:6.3f} {t_ex:9.3f} {t_ev:10.3f} {it_ms:10.1f} {tree.n / (t_cfr / 20):13,.0f}")
        sys.stdout.flush()
    print("\ncompile walk rate (no cache), TOY quotient lazy L1+L2:")
    tree, t = timed(lambda: compile_tree(rules.TOY, (Level.L1, Level.L2), True))
    print(f"   {tree.n:,} nodes in {t:.1f} s = {tree.n / t:,.0f} nodes/s")
    tree, t = timed(lambda: compile_tree(rules.TOY_FULL, (Level.L1,), True))
    print(f"   TOY_FULL L1 only: {tree.n:,} nodes in {t:.1f} s = {tree.n / t:,.0f} nodes/s")


if __name__ == "__main__":
    main()
