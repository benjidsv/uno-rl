"""Step 8.7: CFR variants raced on the cached quotient trees (notes_toy2.md).

    uv run uno cfr_variants [iterations=300] [rulesets=TOY,TOY_FULL,TOY2]

NashConv (score points) of the average strategy at the logged iterations for
every preset in `uno.compiled.cfr.VARIANTS`, with the seconds per iteration; the
exploitability computations are excluded from the timing. Each variant is
checked to bracket the same Nash value and to produce a symmetric strategy.
"""

from __future__ import annotations

import sys

from uno.compiled.cfr import VARIANTS, cfr
from uno.compiled.tree import compile_tree
from uno.game import rules
from uno.game.observe import Level
from uno.store import CACHE

LOG_AT = (10, 20, 30, 50, 70, 100, 150, 200, 300, 500, 700, 1000)


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    iterations = int(argv[0]) if len(argv) > 0 else 300
    names = argv[1].split(",") if len(argv) > 1 else ["TOY", "TOY_FULL", "TOY2"]
    log_at = tuple(t for t in LOG_AT if t <= iterations)
    for name in names:
        tree = compile_tree(getattr(rules, name), (Level.L1,), True, cache=CACHE)
        k = tree.keyings[Level.L1]
        print(f"\n== {name}: {tree.n:,} nodes, {k.n_info:,} L1 sets; NashConv in score points")
        print(f"   {'variant':8s} " + " ".join(f"{'t=' + str(t):>8s}" for t in log_at) + f" {'value':>7s} {'ms/it':>6s} sym")
        values = []
        for v in VARIANTS:
            res = cfr(tree, iterations=iterations, variant=v, log_at=log_at)
            by_t = {e.t: e for e in res.log}
            values.append(res.final.bracket)
            print(f"   {v:8s} " + " ".join(f"{100 * by_t[t].nashconv:8.3f}" for t in log_at) + f" {res.final.value:7.4f} {1000 * res.seconds / res.iterations:6.1f} {k.is_symmetric(res.avg)}")
            sys.stdout.flush()
        lo, hi = max(b[0] for b in values), min(b[1] for b in values)
        print(f"   all brackets intersect: {lo <= hi + 1e-9} ([{lo:.4f}, {hi:.4f}])")


if __name__ == "__main__":
    main()
