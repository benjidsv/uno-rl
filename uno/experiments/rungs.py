"""Step 10a.2 item A5b: what the shipped belief rungs are worth, and how exploitable they are.

    uv run uno rungs [ruleset=TOY] [columns=random,heuristic] [continuations=response]

9.5f closed the sampled rungs against the exact toy posterior as *distributions over hands*
(`uv run uno pimc closure`) and never asked the two questions a ladder row has to answer:
what is an agent that plays this belief **worth**, and how **exploitable** is it? The toys'
own warning is that fitting a belief to one opponent costs exploitability -- on `TOY2_SKIP`
`posterior/response` is 3.52 / 6.43 against `prior/response`'s 1.78 / 2.29 -- and that
matters for step 11 because its training league *is* this league, so the exploit an RL agent
finds will be the league's own.

`uno.compiled.rungs.rung_belief` builds each rung's belief as a weight per world and
`uno.compiled.pimc.pimc_response` takes it directly, so the rungs join the existing
belief axis rather than forming a parallel one.

**On the full world tree, and why.** A member world of an information set carries its own
canonicalising group element, so an opponent hand decoded from its L3 key on a *quotient*
tree is in that member's coordinates rather than the set's. So everything here runs on the
full tree, where a key is a raw key of the real game, which has two consequences to carry
with the numbers: the PIMC choice is a **pure** per-set action rather than an orbit played
uniformly (docs/spec.md section 25 measures those as genuinely different policies, 0.01-0.44
score points apart on `TOY`), and `prior` / `posterior` are therefore **recomputed here**
rather than quoted from the published quotient rows. Only `TOY` and `TOY_FULL` have a full
world tree; `TOY2`'s would be 36x its quotient.

Checks: the rung family must reproduce its own endpoints (`soft(0)` *is* `voids`, `soft(1)`
*is* the prior -- as beliefs and as policies, array-equal), `rung_belief("prior")` must
reproduce `uno.compiled.pimc.belief(..., "prior")`, every policy must sit at or below BR L1
and be a distribution, and the rungs must actually *differ* from the prior somewhere, or the
exploitability comparison below is vacuous (the step 9.5e lesson).
"""

from __future__ import annotations

import sys
import time

import numpy as np

from uno import store
from uno.agents.baseline import Heuristic, PlayFirstRandom
from uno.compiled.pimc import belief as exact_belief
from uno.compiled.pimc import pimc_response, pimc_row
from uno.compiled.rungs import rung_belief, rung_label
from uno.compiled.solvers import best_response, evaluate
from uno.compiled.tree import compile_tree
from uno.game import rules
from uno.game.observe import Level

L1, L2, L3 = Level.L1, Level.L2, Level.L3
FAILS: list[str] = []

# (kind, decay). The endpoints are in because they are the family's own identities; the
# working set is `voids`, and `soft(0.1)` is the maximin rung over the 9.5f league.
RUNGS = (("prior", 0.1), ("voids", 0.1), ("soft", 0.1), ("soft", 0.0), ("soft", 1.0))


def check(ok: bool, what: str) -> str:
    if not ok:
        FAILS.append(what)
    return "ok" if ok else "XX"


def score(v: float) -> float:
    return (v + 1.0) / 2.0


def nash_value(rs) -> float | None:
    """The cached Nash seat-0 score on the *quotient* lazy tree, for exploitability in points.

    The L1 Nash value of the quotient is the full game's (the quotient is exact for symmetric
    strategies, docs/spec.md section 18), so this is the right reference even though the rows
    below live on the full tree."""
    lazy = compile_tree(rs, (L1,), True, cache=store.CACHE)
    res = store.load_nash(lazy)
    if res is None:
        return None
    return score(evaluate(lazy, (L1, res.avg), (L1, res.avg))[0])


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    name = argv[0] if len(argv) > 0 else "TOY"
    columns = (argv[1] if len(argv) > 1 else "random,heuristic").split(",")
    continuations = (argv[2] if len(argv) > 2 else "response").split(",")
    rs = getattr(rules, name)

    t0 = time.perf_counter()
    tree = compile_tree(rs, (L1, L3), False, cache=store.CACHE)  # full: see the module docstring
    nash0 = nash_value(rs)
    print(f"== {name} belief rungs on the full world tree (item A5b)\n   {tree}")
    print(f"   {len(tree.roots):,} worlds; Nash seat-0 score {'unknown' if nash0 is None else f'{nash0:.4f}'}; loaded in {time.perf_counter() - t0:.1f}s")

    # The beliefs are policy-free -- what a voluntary draw proves does not depend on the
    # column -- so they are built once per (seat, rung) and reused for every column.
    print("\n== 1. the rung beliefs (built once per seat: they read no policy)")
    beliefs: dict[tuple[str, int], np.ndarray] = {}
    for kind, decay in RUNGS:
        label = rung_label(kind, decay)
        for seat in (0, 1):
            t1 = time.perf_counter()
            beliefs[(label, seat)] = b = rung_belief(tree, seat, kind, decay)
            dec = (tree.player == seat) & (b > 0)
            print(f"   {label:10s} seat {seat}: worlds with mass {int(dec.sum()):>9,d}  ({time.perf_counter() - t1:.1f}s)")
        sys.stdout.flush()
    # the family's own identities, as beliefs
    for seat in (0, 1):
        ok = check(np.array_equal(beliefs[("soft(0)", seat)], beliefs[("voids", seat)]), f"seat {seat}: soft(0) is not array-equal to voids as a belief")
        ok += check(np.array_equal(beliefs[("soft(1)", seat)], beliefs[("prior", seat)]), f"seat {seat}: soft(1) is not array-equal to the prior as a belief")
        print(f"   seat {seat}: soft(0) == voids and soft(1) == prior, as beliefs  {ok}")

    for column in columns:
        agent = {"random": PlayFirstRandom, "heuristic": Heuristic}[column](rs)
        opp = (L1, store.load_or_tabulate(tree, L1, agent))
        # the prior identity: rung_belief's own reference implementation
        for seat in (0, 1):
            same = np.array_equal(beliefs[("prior", seat)], exact_belief(tree, opp, seat, "prior"))
            check(same, f"seat {seat}: rung_belief('prior') does not reproduce uno.compiled.pimc.belief(..., 'prior')")
        # `posterior` is the only belief here that reads the column's policy, so it cannot be
        # built once per seat like the rungs. It is included because it is the published
        # warning's own row: on the quotient it is *more* exploitable than the prior
        # (`TOY` 2.91 / 4.03 against 0.81 / 0.34), and reproducing that ordering here is what
        # says the full tree's pure-tie-break family behaves like the quotient's.
        per_column = {("posterior", seat): exact_belief(tree, opp, seat, "posterior") for seat in (0, 1)}
        rungs_here = (*RUNGS, ("posterior", 0.1))
        for cont in continuations:
            print(f"\n== 2. vs {column}, {cont} continuation (score of the PIMC player; gap = BR L1 - PIMC, score points)")
            print(f"   {'seat':4s} {'rung':10s} {'score':>7s} {'gap':>6s} {'BR L1':>7s} {'differs from prior':>19s}  checks")
            rows: dict[str, tuple] = {}
            for seat in (0, 1):
                br = best_response(tree, L1, opp, seat)
                base = None
                for kind, decay in rungs_here:
                    label = rung_label(kind, decay)
                    b = per_column.get((label, seat), beliefs.get((label, seat)))
                    p = pimc_response(tree, opp, seat, label, cont, b=b)
                    k = tree.keyings[L1]
                    mine = k.info_player == seat
                    if base is None:
                        base = p.sigma
                        differs = 0
                    else:
                        per_set = k.info_sum(np.abs(p.sigma - base))
                        differs = int(((per_set > 1e-12) & mine).sum())
                    ok = check(p.value <= br.value + 1e-9, f"{column} {cont} seat {seat} {label}: PIMC {p.score:.6f} above BR L1 {br.score:.6f}")
                    ok += check(np.allclose(k.info_sum(p.sigma), 1.0), f"{column} {cont} seat {seat} {label}: the policy is not a distribution on every set")
                    rows[(label, seat)] = (p, differs)
                    print(f"   {seat:<4d} {label:10s} {p.score:7.4f} {100 * (br.score - p.score):6.2f} {br.score:7.4f} {differs:19,d}  {ok}")
                sys.stdout.flush()
            # the identities again, as policies
            for seat in (0, 1):
                ok = check(np.array_equal(rows[("soft(0)", seat)][0].sigma, rows[("voids", seat)][0].sigma), f"{column} {cont} seat {seat}: soft(0) and voids are not the same policy")
                ok += check(np.array_equal(rows[("soft(1)", seat)][0].sigma, rows[("prior", seat)][0].sigma), f"{column} {cont} seat {seat}: soft(1) and the prior are not the same policy")
                print(f"   seat {seat}: soft(0) == voids and soft(1) == prior, as policies  {ok}")
            # and the non-vacuity guard: a rung that changed nothing would make the table below meaningless
            moved = max(rows[(rung_label(*r), s)][1] for r in rungs_here for s in (0, 1))
            check(moved > 0, f"{column} {cont}: no rung ever chose differently from the prior, so the exploitability comparison is vacuous")

            print(f"\n== 3. vs {column}, {cont}: exploitability of the rung as a row agent (score points, row at seat 0 / seat 1)")
            print(f"   {'rung':10s} {'score s0/s1':>15s} {'exploitability s0/s1':>22s} {'NashConv':>9s}  checks")
            for kind, decay in rungs_here:
                label = rung_label(kind, decay)
                bs = tuple(per_column.get((label, s), beliefs.get((label, s))) for s in (0, 1))
                row, ps = pimc_row(tree, opp, label, cont, b=bs)
                ex = []
                for seat in (0, 1):
                    brr = best_response(tree, L1, row, seat ^ 1)
                    ex.append(100 * (brr.score - ((1 - nash0) if seat == 0 else nash0)) if nash0 is not None else float("nan"))
                print(f"   {label:10s} {ps[0].score:7.4f}/{ps[1].score:.4f} {ex[0]:10.2f} / {ex[1]:<10.2f} {ex[0] + ex[1]:9.2f}")
                sys.stdout.flush()

    print(f"\n== {len(FAILS)} failed checks" + (": " + "; ".join(FAILS) if FAILS else ""))


if __name__ == "__main__":
    main()
