"""Step 9.5a: how much of BR L1's edge needs history at all (D3 of the step 9
review's section D: "Memory: BR L1 conditions on the full history. A backward pass
on a coarse, history-free key gives a valid policy, a lower bound on the
memoryless ceiling: does the RL observation need history?").

    uv run uno memory [ruleset=TOY] [iterations=20]

Best responses to Random and to the Heuristic, per seat, at three nested
history-free keys (`uno.compiled.coarse`, docs/spec.md section 20):

- K0: hand, top, current colour, opponent hand size, pile size, phase, drawn card;
- K1: K0 + the discard pile as a multiset;
- K2: K1 + a void hint (colours the opponent drew on since their last play).

`uno.compiled.coarse.memoryless_response` runs an iterated greedy best response on
each (`iterations` rounds, starting from the Heuristic's own play, itself
K0-measurable), and reports the best value seen. Every value is checked to lie
between the starting policy's own value and BR L1's (full history): both are
guaranteed (`uno.compiled.coarse.coarsen`'s docstring), so a failure here would be
a bug, not a borderline finding.

On a symmetric (quotient) tree, K0/K1/K2 have a known limitation (`coarsen`'s
docstring): they group by whichever canonical L1 representative `compile_tree`
picked for each set, not by one re-minimised for the coarse key's own, smaller
field set, so a quotient tree's numbers are not directly comparable to a full
tree's for the same keys (only `identity_key`, not used here, is exempt). Numbers
here are always reported on the tree passed in, never compared across trees.
"""

from __future__ import annotations

import sys

from uno import store
from uno.agents.baseline import Heuristic, PlayFirstRandom
from uno.compiled.coarse import KEYS, coarsen, memoryless_response
from uno.compiled.solvers import best_response, evaluate
from uno.compiled.tree import compile_tree
from uno.game import rules
from uno.game.observe import Level

L1 = Level.L1
FAILS: list[str] = []


def check(ok: bool, what: str) -> str:
    if not ok:
        FAILS.append(what)
    return "ok" if ok else "XX"


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    name = argv[0] if len(argv) > 0 else "TOY"
    iterations = int(argv[1]) if len(argv) > 1 else 20
    rs = getattr(rules, name)
    tree = compile_tree(rs, (L1,), True, cache=store.CACHE)
    print(f"== {name}: {tree}")
    rnd, heu = PlayFirstRandom(rs), Heuristic(rs)
    tab = {a.name: store.load_or_tabulate(tree, L1, a) for a in (rnd, heu)}
    coarsenings = {k: coarsen(tree, fn) for k, fn in KEYS.items()}
    for k, c in coarsenings.items():
        print(f"   {k}: {c.keying.n_info:,} coarse sets ({100 * c.keying.n_info / tree.keyings[L1].n_info:.1f} % of L1)")
    sys.stdout.flush()

    print(f"\n   {'opponent':10s} {'seat':4s} {'heuristic':>9s} {'K0':>9s} {'K1':>9s} {'K2':>9s} {'BR L1':>9s}  checks")
    for opponent_name, opp_sigma in (("random", tab["random"]), ("heuristic", tab["heuristic"])):
        opp = (L1, opp_sigma)
        for seat in (0, 1):
            br = best_response(tree, L1, opp, seat)
            heu_l1 = tab["heuristic"]
            heu_val = evaluate(tree, *(((L1, heu_l1), opp) if seat == 0 else (opp, (L1, heu_l1))))[seat]
            cells = []
            checks = []
            for k in ("K0", "K1", "K2"):
                res = memoryless_response(tree, coarsenings[k], opp, seat, heu_l1, iterations=iterations)
                cells.append(res.value)
                ok = check(heu_val - 1e-6 <= res.value <= br.value + 1e-6, f"{opponent_name} seat {seat} {k}: heuristic <= value <= BR L1")
                checks.append(ok)
            print(f"   {opponent_name:10s} {seat:<4d} {heu_val:9.4f} " + " ".join(f"{v:9.4f}" for v in cells) + f" {br.value:9.4f}  " + " ".join(checks))
            sys.stdout.flush()

    print(f"\n== {len(FAILS)} failed checks" + (": " + "; ".join(FAILS) if FAILS else ""))


if __name__ == "__main__":
    main()
