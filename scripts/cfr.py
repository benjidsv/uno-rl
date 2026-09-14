"""Step 8: Nash L1 by CFR+ on the compiled tree, and the ladder cells that use it.

    uv run python scripts/cfr.py [ruleset=TOY] [iterations=1000] [games=4000]

Prints, on the ruleset:

1. the compiled trees (lazy L1/L2 and world L1/L3) and a speed table against
   the pure-Python reference solvers;
2. the exploitability curve of CFR+ and of vanilla CFR (NashConv in score
   points against iteration count, with the bracket of the Nash seat-0 score);
3. the exploitability of every row agent per seat (BR L1 to it minus the Nash
   value at that seat), which the BR columns of steps 5-6 could not give;
4. the Nash row: exact value against Random and Heuristic per seat, the
   simulation on the held-out seeds with the on-seeds check, the mirrored
   estimate, and the exact mirrored moments;
5. the Nash column: Random, Heuristic and BR L1(random) against it;
6. the BR L1 / L2 / L3 ceilings against the Nash agent (information gaps).

The CFR+ result is saved to `.cache/cfr_<ruleset>.npz` (loaded if present and
computed with at least the requested iterations).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from uno import rules
from uno.agents import BySeat, Heuristic, PlayFirstRandom
from uno.cfr import CFRResult, cfr, exploitability_of
from uno.compiled import CompiledAgent, best_response, compile_tree, evaluate, exact_mirrored, tabulate
from uno.estimate import estimate_mirrored, play_pairs
from uno.observe import Level
from uno.play import EVAL_SEEDS, match
from uno.solve import best_response as best_response_ref
from uno.solve import evaluate as evaluate_ref
from uno.solve import evaluate_on_seeds

CACHE = Path(__file__).resolve().parent.parent / ".cache"


def score(v: float) -> float:
    return (v + 1) / 2


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "TOY"
    rs = getattr(rules, name)
    iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    games = int(sys.argv[3]) if len(sys.argv) > 3 else len(EVAL_SEEDS)
    seeds = range(EVAL_SEEDS.start, EVAL_SEEDS.start + games)
    rnd, heu = PlayFirstRandom(rs), Heuristic(rs)

    print(f"== compiled trees on {name}")
    lazy = compile_tree(rs, (Level.L1, Level.L2))
    print("  ", lazy)
    world = compile_tree(rs, (Level.L1, Level.L3))
    print("  ", world)
    t0 = time.perf_counter()
    sig = {a.name: (Level.L1, tabulate(lazy, Level.L1, a)) for a in (rnd, heu)}
    sigw = {a.name: (Level.L1, tabulate(world, Level.L1, a)) for a in (rnd, heu)}
    print(f"   tabulating Random and Heuristic on both trees: {time.perf_counter() - t0:.1f}s")

    print("\n== speed: compiled vs reference (seconds)")
    print(f"   {'computation':38s} {'compiled':>9s} {'reference':>10s} {'value':>8s} {'ref value':>10s}")
    for label, level, tree, sigs in (("BR L1 seat 1 vs random", Level.L1, lazy, sig), ("BR L2 seat 0 vs heuristic", Level.L2, lazy, sig), ("BR L3 seat 0 vs random", Level.L3, world, sigw)):
        opp = rnd if "random" in label else heu
        seat = int(label.split("seat ")[1][0])
        t0 = time.perf_counter()
        br = best_response(tree, level, sigs[opp.name], seat)
        tc = time.perf_counter() - t0
        t0 = time.perf_counter()
        ref = best_response_ref(rs, opp, seat, level)
        tr = time.perf_counter() - t0
        print(f"   {label:38s} {tc:9.2f} {tr:10.1f} {br.score:8.4f} {ref.score:10.4f}")
    t0 = time.perf_counter()
    v = score(evaluate(lazy, sig["heuristic"], sig["random"])[0])
    tc = time.perf_counter() - t0
    t0 = time.perf_counter()
    vr = score(evaluate_ref(rs, (heu, rnd))[0])
    tr = time.perf_counter() - t0
    print(f"   {'evaluate heuristic vs random':38s} {tc:9.2f} {tr:10.1f} {v:8.4f} {vr:10.4f}")
    sys.stdout.flush()

    print(f"\n== CFR+ ({iterations} iterations; NashConv in score points; bracket = Nash seat-0 score)")
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"cfr_{name}.npz"
    res = None
    if path.exists():
        res = CFRResult.load(path)
        if res.iterations < iterations or res.config.get("plus") != "True":
            res = None
        else:
            print(f"   loaded {path} ({res.iterations} iterations, {res.seconds:.0f}s)")
            for e in res.log:
                print("  ", e)
    if res is None:
        res = cfr(lazy, iterations=iterations, verbose=True)
        res.save(path)
        print(f"   {res.seconds:.0f}s for {res.iterations} iterations ({res.seconds / res.iterations:.2f}s each), saved to {path}")
    nash = res.agent(lazy)
    ex = exploitability_of(lazy, res)
    nash_value = (ex.bracket[0] + ex.bracket[1]) / 2  # midpoint; the bracket width is NashConv
    print(f"   Nash seat-0 score in [{ex.bracket[0]:.4f}, {ex.bracket[1]:.4f}], seat 1 = 1 - that; width {100 * ex.score_points:.3f} pts")
    sys.stdout.flush()

    print("\n== vanilla CFR for comparison (200 iterations)")
    van = cfr(lazy, iterations=min(200, iterations), plus=False, alternating=False, linear=False, log_at=(1, 2, 5, 10, 20, 50, 100, 200), verbose=True)
    print(f"   {van.seconds:.0f}s for {van.iterations} iterations")
    sys.stdout.flush()

    print("\n== exploitability of the row agents (BR L1 to the row minus the Nash value, per seat, score points)")
    print(f"   {'row':22s} {'seat':4s} {'own value':>10s} {'BR L1 to it':>12s} {'Nash':>8s} {'exploitability':>15s}")
    rows = {"random": sig["random"], "heuristic": sig["heuristic"]}
    brs = {}
    for seat in (0, 1):
        brs[("random", seat)] = best_response(lazy, Level.L1, sig["random"], seat)
    br_rnd_sigma = np.where(lazy.keyings[Level.L1].info_player[lazy.keyings[Level.L1].ia_info] == 0, brs[("random", 0)].sigma, brs[("random", 1)].sigma)
    rows["br_l1(random)"] = (Level.L1, br_rnd_sigma)
    rows["nash_l1"] = (Level.L1, res.avg)
    for label, s in rows.items():
        for seat in (0, 1):
            opp_seat = seat ^ 1
            br = best_response(lazy, Level.L1, s, opp_seat)  # the *opponent* best-responds to the row agent
            nash_seat = nash_value if seat == 0 else 1 - nash_value
            row_value = 1 - br.score  # what the row agent keeps against its best responder
            print(f"   {label:22s} {seat:<4d} {row_value:10.4f} {br.score:12.4f} {1 - nash_seat:8.4f} {(nash_seat - row_value) * 100:14.2f}")
    print("   (own value = score kept against the best responder; exploitability = Nash value at that seat minus it, in points)")
    sys.stdout.flush()

    print(f"\n== Nash row and column: exact values (score of the first-named, seat 0 / seat 1)")
    nash_w = (Level.L1, tabulate(world, Level.L1, nash))
    pairs_exact = {}
    for label, a, b, sa, sb in (
        ("nash vs random", nash, rnd, nash_w, sigw["random"]),
        ("nash vs heuristic", nash, heu, nash_w, sigw["heuristic"]),
        ("nash vs nash", nash, nash, nash_w, nash_w),
        ("random vs nash", rnd, nash, sigw["random"], nash_w),
        ("heuristic vs nash", heu, nash, sigw["heuristic"], nash_w),
    ):
        v0 = score(evaluate(world, sa, sb)[0])
        v1 = score(evaluate(world, sb, sa)[1])
        pairs_exact[label] = (v0, v1)
        print(f"   {label:24s} {v0:.4f}  {v1:.4f}")
    # BR L1(random) vs nash needs the per-seat tables on the world tree
    brw = [best_response(world, Level.L1, sigw["random"], seat) for seat in (0, 1)]
    kw = world.keyings[Level.L1]
    brw_sigma = (Level.L1, np.where(kw.info_player[kw.ia_info] == 0, brw[0].sigma, brw[1].sigma))
    v0, v1 = score(evaluate(world, brw_sigma, nash_w)[0]), score(evaluate(world, nash_w, brw_sigma)[1])
    print(f"   {'br_l1(random) vs nash':24s} {v0:.4f}  {v1:.4f}")
    sys.stdout.flush()

    print(f"\n== Nash row on {games} held-out pairs: sim vs exact on those seeds, mirrored estimate")
    print(f"   {'matchup':20s} {'seat':4s} {'exact':>7s} {'on seeds':>9s} {'sim ± 95%':>17s} {'ok':2s} | {'mean exact':>10s} {'raw ±':>7s} {'mirrored ±':>10s} {'corr':>7s} {'splits':>7s} {'var ratio':>9s}")
    for label, opp, s_opp in (("nash vs random", rnd, sigw["random"]), ("nash vs heuristic", heu, sigw["heuristic"])):
        pairs = play_pairs(rs, (nash, opp), seeds)
        m = estimate_mirrored(pairs)
        exm = exact_mirrored(world, nash_w, s_opp)
        on0 = score(evaluate_on_seeds(rs, (nash, opp), pairs.seeds)[0])
        on1 = score(evaluate_on_seeds(rs, (opp, nash), pairs.seeds)[1])
        for seat, on in ((0, on0), (1, on1)):
            e = m.seat[seat]
            ok = abs(e.mean - on) <= e.half
            tail = f" | {exm.pair_mean:10.4f} {m.raw.half:7.4f} {m.mirrored.half:10.4f} {m.corr:7.3f} {m.splits:7.3f} {m.var_ratio:9.3f}" if seat == 0 else ""
            print(f"   {label if seat == 0 else '':20s} {seat:<4d} {exm.mean[seat]:7.4f} {on:9.4f} {str(e):>17s} {'ok' if ok else 'XX'}{tail}")
        print(f"   {'':20s} exact corr={exm.corr:+.3f} splits={exm.splits:.3f} var ratio={exm.var_ratio:.3f}" + (f"  misses={nash.misses}" if nash.misses else ""))
    sys.stdout.flush()

    print("\n== best responses to the Nash agent by information level (score of the responder)")
    print(f"   {'seat':4s} {'BR L1':>7s} {'BR L2':>7s} {'BR L3':>7s} {'L2-L1':>7s} {'L3-L2':>7s}   Nash keeps at this seat")
    nash_lazy = (Level.L1, res.avg)
    for seat in (0, 1):
        l1 = best_response(lazy, Level.L1, nash_lazy, seat).score
        l2 = best_response(lazy, Level.L2, nash_lazy, seat).score
        l3 = best_response(world, Level.L3, nash_w, seat).score
        keeps = pairs_exact["nash vs nash"][seat ^ 1]
        print(f"   {seat:<4d} {l1:7.4f} {l2:7.4f} {l3:7.4f} {100 * (l2 - l1):7.1f} {100 * (l3 - l2):7.1f}   {keeps:.4f} (so seat {seat} gains {100 * (l1 - (1 - keeps)):.2f} / {100 * (l2 - (1 - keeps)):.2f} / {100 * (l3 - (1 - keeps)):.2f} pts at L1 / L2 / L3)")

    # steps 5-6 columns, replayed on the compiled tree for the record
    print("\n== steps 5-6 best-response columns on the compiled trees (should reproduce notes.md)")
    print(f"   {'opponent':10s} {'seat':4s} {'BR L1':>7s} {'BR L2':>7s} {'BR L3':>7s}")
    for opp in (rnd, heu):
        for seat in (0, 1):
            l1 = best_response(lazy, Level.L1, sig[opp.name], seat).score
            l2 = best_response(lazy, Level.L2, sig[opp.name], seat).score
            l3 = best_response(world, Level.L3, sigw[opp.name], seat).score
            print(f"   {opp.name:10s} {seat:<4d} {l1:7.4f} {l2:7.4f} {l3:7.4f}")


if __name__ == "__main__":
    main()
