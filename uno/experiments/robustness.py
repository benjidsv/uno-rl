"""Step 9: how robust the exact toy results are (splits, exhaustion rule, equilibrium choice, fill-in, deal model).

    uv run uno robustness [ruleset=TOY] [iterations=1000] [light] [seeds=20000]

The review of step 9 (docs/results/09-review.md) found readings that depend on
choices nobody had varied. This experiment varies them, on the cached trees:

1. (A1) the split share of the mirrored evaluator against the split share of two
   independent games with the same per-seat outcome distributions: the excess
   is what the shared deal adds (exact, world tree);
2. (A2, A6) everything under three scorings of exhaustion, "hand" (the rule:
   fewer cards wins), "draw" (every exhaustion ending is a draw; degenerate on
   free-play rulesets, where the trailing player stalls into a draw) and "stuck"
   (the player who has to draw from the empty pile loses), on the same
   tree (`CompiledTree.rescored`): row matchups, best-response columns against
   Random and the Heuristic, the Nash value (a CFR run per scoring), the
   exploitability of the rows, and the two-sided values (`two_sided`: both see
   both hands, both see everything) with the share of worlds whose result no
   play changes;
3. (B1) the equilibrium choice: the Nash-dependent cells (Nash row, best
   responses to Nash, the Nash-filled best response to Random) for several CFR
   variants, which reach different equilibria of the same value;
4. (B2) the fill-in of BR L1(random) at the sets Random never lets it reach: as
   solved (first orbit), Nash-filled, and least exploitable (`cfr` with the
   reached sets fixed), with the number of reached sets where orbits tie;
5. (B4) the deal model: the mean over many seeds of the exact score on each
   seed's real deal against the whole-game exact score, whose deals come from the
   chance model (z on the deal variance). Reference walkers, real deals.

`light` (TOY2_FULL): the lazy L1 tree and the world tree without an L3 keying,
no L2 column, DCFR / PCFR+ only. Every check prints ok / XX; the last line counts
the failures.
"""

from __future__ import annotations

import math
import sys
from itertools import pairwise

from uno import store
from uno.agents.baseline import Heuristic, PlayFirstRandom
from uno.compiled.cfr import cfr
from uno.compiled.solvers import (
    best_response,
    best_response_perfect,
    evaluate,
    exact_mirrored,
    two_sided,
)
from uno.compiled.strategies import br_row, transfer
from uno.compiled.tree import CompiledTree, compile_tree
from uno.experiments.common import by_seat, score
from uno.game import rules
from uno.game.observe import Level
from uno.reference.solve import scores_on_seeds
from uno.store import CACHE

L1, L2, L3 = Level.L1, Level.L2, Level.L3
FAILS: list[str] = []


def check(ok: bool, what: str) -> str:
    if not ok:
        FAILS.append(what)
    return "ok" if ok else "XX"


def independent_splits(tree: CompiledTree, a, b) -> float:
    """P(pair score = 1/2) if the two games of a pair were independent, with the
    per-seat outcome distributions of the first-named agent `a` (world tree)."""
    cols = tree.outcome_columns()
    d0 = tree.values(tree.profile_weights(a, b), cols)[0][0]  # a at seat 0: win, draw, loss
    d1 = tree.values(tree.profile_weights(b, a), cols)[0][0][::-1]  # a at seat 1: win, draw, loss
    return float(d0 @ d1[::-1])  # (win, loss) + (draw, draw) + (loss, win)


def exploit(tree: CompiledTree, row, seat: int, nash_value: float) -> float:
    """Exploitability in points of the L1 strategy `row` playing `seat`: its best
    responder's score minus the responder seat's Nash score."""
    br = best_response(tree, L1, (L1, row), seat ^ 1)
    return 100 * (br.score - (nash_value if seat == 1 else 1 - nash_value))


def pair_str(x: tuple[float, float], fmt: str = ".4f") -> str:
    return f"{x[0]:{fmt}} / {x[1]:{fmt}}"


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    name = argv[0] if len(argv) > 0 else "TOY"
    iterations = int(argv[1]) if len(argv) > 1 else 1000
    light = len(argv) > 2 and argv[2] == "light"
    n_seeds = int(argv[3]) if len(argv) > 3 else 20_000
    rs = getattr(rules, name)
    variant = "dcfr" if light else "cfr+"
    variants = ("dcfr", "pcfr+") if light else ("cfr+", "dcfr", "pcfr+")

    lazy = compile_tree(rs, (L1,) if light else (L1, L2), True, cache=CACHE)
    world = compile_tree(rs, (L1,), True, cache=CACHE, world=True) if light else compile_tree(rs, (L1, L3), True, cache=CACHE)
    print(f"== {name}: {lazy}\n   world: {world}")
    rnd, heu = PlayFirstRandom(rs), Heuristic(rs)
    sl = {a.name: (L1, store.load_or_tabulate(lazy, L1, a)) for a in (rnd, heu)}
    sw = {name: (L1, transfer(lazy, world, L1, s)) for name, (_, s) in sl.items()}  # same L1 sets, numbered differently
    nash = store.load_or_solve_nash(lazy, iterations, variant)
    v_nash = score(evaluate(lazy, (L1, nash.avg), (L1, nash.avg))[0])
    nash_w = (L1, transfer(lazy, world, L1, nash.avg))
    print(f"   Nash ({variant}, {nash.iterations} iterations, NashConv {100 * nash.final.nashconv:.4f} pts): seat-0 score {v_nash:.4f}")
    sys.stdout.flush()

    print("\n== 1. split share of mirrored pairs against two independent games (exact, world tree, first-named agent)")
    print(f"   {'matchup':30s} {'corr':>7s} {'splits':>7s} {'independent':>11s} {'excess':>7s}")
    brw = {o: by_seat(world, [best_response(world, L1, sw[o], s) for s in (0, 1)]) for o in ("random", "heuristic")}
    for label, a, b in (
        ("heuristic vs random", sw["heuristic"], sw["random"]),
        ("br_l1(random) vs random", brw["random"], sw["random"]),
        ("br_l1(heuristic) vs heuristic", brw["heuristic"], sw["heuristic"]),
        ("br_l1(random) vs heuristic", brw["random"], sw["heuristic"]),
        ("nash vs random", nash_w, sw["random"]),
        ("nash vs heuristic", nash_w, sw["heuristic"]),
    ):
        m = exact_mirrored(world, a, b)
        ind = independent_splits(world, a, b)
        print(f"   {label:30s} {m.corr:+7.3f} {m.splits:7.3f} {ind:11.3f} {m.splits - ind:+7.3f}")
    sys.stdout.flush()

    print("\n== 2. exhaustion scored on hand size (the rule), as a draw, and as a loss for the player stuck on the empty pile; same tree")
    summary = {}
    for mode in ("hand", "draw", "stuck"):
        L, W = lazy.rescored(mode), world.rescored(mode)
        res = nash if mode == "hand" else cfr(L, iterations=iterations, variant=variant, log_at=(iterations,))
        v = score(evaluate(L, (L1, res.avg), (L1, res.avg))[0])
        rr = score(evaluate(L, sl["random"], sl["random"])[0])
        hr = (score(evaluate(L, sl["heuristic"], sl["random"])[0]), score(evaluate(L, sl["random"], sl["heuristic"])[1]))
        hh = score(evaluate(L, sl["heuristic"], sl["heuristic"])[0])
        ts2, ts3 = two_sided(L), two_sided(W)
        print(f"\n   [{mode}] Nash seat-0 score {v:.4f} (NashConv {100 * res.final.nashconv:.4f} pts), seat gap {100 * (2 * v - 1):.1f} pts")
        print(f"   [{mode}] two-sided: both see both hands {ts2.score:.4f}, both see everything {ts3.score:.4f}; worlds whose result no play changes {ts3.decided:.3f}; under minimax play seat 0 wins / draw / seat 1 wins {ts3.outcome[0]:.3f} / {ts3.outcome[1]:.3f} / {ts3.outcome[2]:.3f}")
        print(f"   [{mode}] random vs random {rr:.4f} (gap {100 * (2 * rr - 1):.1f}); heuristic vs random {pair_str(hr)} (edge over random {100 * (hr[0] - rr):+.1f} / {100 * (hr[1] - (1 - rr)):+.1f}); heuristic vs heuristic gap {100 * (2 * hh - 1):.1f}")
        print(f"   [{mode}] {'opponent':10s} {'seat':4s} {'BR L1':>7s}" + ("" if light else f" {'BR L2':>7s}") + f" {'BR L3':>7s} {'L1 edge over random':>20s}")
        cols = {}
        for opp in ("random", "heuristic"):
            for seat in (0, 1):
                c = [best_response(L, L1, sl[opp], seat).score]
                if not light:
                    c.append(best_response(L, L2, sl[opp], seat).score)
                c.append(best_response_perfect(W, sw[opp], seat).score)
                cols[(opp, seat)] = c
                ok = check(all(x <= y + 1e-9 for x, y in pairwise(c)), f"[{mode}] L1 <= L2 <= L3 vs {opp} seat {seat}")
                base = rr if seat == 0 else 1 - rr
                print(f"   [{mode}] {opp:10s} {seat:<4d} " + " ".join(f"{x:7.4f}" for x in c) + f" {100 * (c[0] - base):+20.1f}  {ok}")
        ex = {row: tuple(exploit(L, s, seat, v) for seat in (0, 1)) for row, s in (("random", sl["random"][1]), ("heuristic", sl["heuristic"][1]), ("nash", res.avg))}
        print(f"   [{mode}] exploitability (pts, row at seat 0 / 1): " + ", ".join(f"{row} {pair_str(x, '.1f')}" for row, x in ex.items()))
        # the two-sided values bound the one-sided best responses: a fixed opponent can only do worse than minimax
        for tree_, ts, lv in ((L, ts2, L2), (W, ts3, L3)):
            if lv is L2 and light:
                continue
            for opp in ("random", "heuristic"):
                i = (1 if lv is L2 else 2) if not light else 1
                ok = check(1 - cols[(opp, 1)][i] - 1e-9 <= ts.score <= cols[(opp, 0)][i] + 1e-9, f"[{mode}] two-sided {lv.name} between the one-sided best responses to {opp}")
        summary[mode] = {"gap": 100 * (2 * v - 1), "rr": 100 * (2 * rr - 1), "br": cols, "ex": ex, "ts": (ts2.score, ts3.score, ts3.decided)}
        sys.stdout.flush()
    h = summary["hand"]
    for mode in ("draw", "stuck"):
        d = summary[mode]
        print(f"\n   {mode} minus hand: Nash seat gap {d['gap'] - h['gap']:+.1f} pts, random-vs-random gap {d['rr'] - h['rr']:+.1f}; BR L1 vs random {100 * (d['br'][('random', 0)][0] - h['br'][('random', 0)][0]):+.1f} / {100 * (d['br'][('random', 1)][0] - h['br'][('random', 1)][0]):+.1f}; "
              f"heuristic exploitability {d['ex']['heuristic'][0] - h['ex']['heuristic'][0]:+.1f} / {d['ex']['heuristic'][1] - h['ex']['heuristic'][1]:+.1f}")

    print("\n== 3. equilibrium choice: Nash-dependent cells per CFR variant (standard scoring; score of the first-named, seat 0 / seat 1)")
    cells: dict[str, list[tuple[float, float]]] = {}
    for var in variants:
        r = store.load_or_solve_nash(lazy, iterations, var)
        n_l = (L1, r.avg)
        n_w = (L1, transfer(lazy, world, L1, r.avg))
        (_, filled), _ = br_row(lazy, sl["random"], completion=r.avg)
        row = {
            "nash vs random": (score(evaluate(lazy, n_l, sl["random"])[0]), score(evaluate(lazy, sl["random"], n_l)[1])),
            "nash vs heuristic": (score(evaluate(lazy, n_l, sl["heuristic"])[0]), score(evaluate(lazy, sl["heuristic"], n_l)[1])),
            "BR L3 vs nash": (best_response_perfect(world, n_w, 0).score, best_response_perfect(world, n_w, 1).score),
            "BR L1(random)+nash exploitability (pts)": (exploit(lazy, filled, 0, v_nash), exploit(lazy, filled, 1, v_nash)),
        }
        if not light:
            row["BR L2 vs nash"] = (best_response(lazy, L2, n_l, 0).score, best_response(lazy, L2, n_l, 1).score)
        vr = score(evaluate(lazy, n_l, n_l)[0])
        print(f"   {var:6s} {r.iterations:5d} its, NashConv {100 * r.final.nashconv:.4f} pts, seat-0 score {vr:.4f}: " + "; ".join(f"{k} {pair_str(x, '.2f' if 'pts' in k else '.4f')}" for k, x in row.items()))
        for k, x in row.items():
            cells.setdefault(k, []).append(x)
        sys.stdout.flush()
    print("   spread over the variants (max - min, points): " + "; ".join(f"{k.replace(' (pts)', '')} {100 ** ('pts' not in k) * (max(x[0] for x in xs) - min(x[0] for x in xs)):.2f} / {100 ** ('pts' not in k) * (max(x[1] for x in xs) - min(x[1] for x in xs)):.2f}" for k, xs in cells.items()))

    print(f"\n== 4. BR L1(random) at the sets Random never reaches: as solved, Nash-filled, least exploitable (CFR {variant} with the reached sets fixed); exploitability in points")
    k = lazy.keyings[L1]
    marks = sorted({max(1, iterations // 4), max(1, iterations // 2), iterations})
    for seat in (0, 1):
        br = best_response(lazy, L1, sl["random"], seat)
        filled = best_response(lazy, L1, sl["random"], seat, completion=nash.avg)
        mask = (k.info_player[k.ia_info] == seat) & ~br.unreached_sets[k.ia_info]
        res = cfr(lazy, iterations=iterations, variant=variant, fixed=(mask, br.sigma), log_at=marks)
        resp = v_nash if seat == 1 else 1 - v_nash
        least = [100 * (e.br[seat ^ 1] - resp) for e in res.log]
        solved, with_nash = exploit(lazy, br.sigma, seat, v_nash), exploit(lazy, filled.sigma, seat, v_nash)
        n = int((k.info_player == seat).sum())
        ok = check(least[-1] <= with_nash + 0.05, f"least exploitable fill-in seat {seat} not above the Nash fill-in")
        print(f"   row at seat {seat}: {br.unreached:,} of {n:,} sets unreached, {br.ties:,} reached sets with tied orbits | as solved {solved:.2f}, Nash-filled {with_nash:.2f}, least exploitable {least[-1]:.2f} (t = {', '.join(map(str, marks))}: {', '.join(f'{x:.2f}' for x in least)})  {ok}")
        sys.stdout.flush()

    print(f"\n== 5. deal model: mean over {n_seeds:,} seeds of the exact score on each real deal, against the whole-game exact score (z on the deal variance)")
    seeds = range(2_000_000, 2_000_000 + n_seeds)
    for label, a, b in (("random vs random", rnd, rnd), ("heuristic vs random", heu, rnd)):
        exact = score(evaluate(lazy, sl[a.name], sl[b.name])[0])
        on = scores_on_seeds(rs, (a, b), seeds)
        z = on.z_global(exact, 0)
        ok = check(abs(z) <= 3, f"deal model {label} (z = {z:+.2f})")
        print(f"   {label:22s} whole game {exact:.4f}, mean on {on.n:,} seeds {on.mean[0]:.4f} (sd over seeds {math.sqrt(on.deal_var[0]):.3f}), z {z:+.2f}  {ok}")

    print(f"\n== {len(FAILS)} failed checks" + (": " + "; ".join(FAILS) if FAILS else ""))


if __name__ == "__main__":
    main()
