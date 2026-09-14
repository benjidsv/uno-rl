"""Step 8.7: the validation checks of steps 5-8 on a ruleset, from the compiled trees.

    uv run python scripts/validate.py [ruleset=TOY] [games=4000] [light]

`scripts/ladder.py` and `scripts/mirror.py` run the pure-Python reference
solvers, which do not fit the TOY2 family (its full game is ~10M histories,
~10 GB in the reference walkers). This script does the same checks from the
cached quotient trees (RULES.md sections 16 and 18):

1. best-response columns (L1 / L2 on the lazy tree, L3 on the world tree)
   against Random and the Heuristic, per seat: the exact score, the exact
   score on the held-out seeds, the simulation of the best-response agent on
   those seeds inside its CI, no misses, L1 <= L2 <= L3;
2. the Nash agent (from `.cache/cfr_<ruleset>_L1_sym*.npz`): NashConv, the
   profile's value inside its bracket, every row's best-response ceiling
   above the bracket, and the Nash row on the seeds (sim vs on-seeds);
3. the mirrored evaluator on the step 7 matchups plus the Nash row: sampled
   per-seat means against the on-seeds exact values, and correlation, split
   share and variance ratio against the exact moments of the world tree;
4. the paired comparisons (sampled).

`light`: lazy tree with the L1 keying only (no L2 / L3 columns, no exact
mirrored moments), for TOY2_FULL.

Every check prints ok / XX and the last line counts the failures. A plain
"exact inside the CI" test fails one time in twenty by construction; the
on-seeds comparison is the test.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from uno import rules
from uno.agents import Heuristic, PlayFirstRandom
from uno.cfr import CFRResult
from uno.compiled import CompiledAgent, CompiledTree, best_response, cache_name, compile_tree, evaluate, exact_mirrored, exploitability, tabulate
from uno.estimate import compare, estimate_mirrored, play_pairs
from uno.observe import Level
from uno.play import EVAL_SEEDS, match
from uno.solve import evaluate_on_seeds

CACHE = Path(__file__).resolve().parent.parent / ".cache"
FAILS: list[str] = []


def score(v: float) -> float:
    return (v + 1) / 2


def check(ok: bool, what: str) -> str:
    if not ok:
        FAILS.append(what)
    return "ok" if ok else "XX"


def by_seat(tree: CompiledTree, br0, br1) -> np.ndarray:
    k = tree.keyings[Level.L1]
    return np.where(k.info_player[k.ia_info] == 0, br0.sigma, br1.sigma)


def load_nash(tree: CompiledTree, name: str) -> CFRResult | None:
    base = f"cfr_{cache_name(tree.rs, (Level.L1,), True)}"
    for path in sorted(CACHE.glob(base + "*.npz")):
        res = CFRResult.load(path)
        if len(res.avg) == tree.keyings[Level.L1].n_ia:
            print(f"   Nash strategy: {path.name}, {res.iterations} iterations of {res.config.get('variant') or 'cfr+'}, NashConv {100 * res.final.nashconv:.3f} pts")
            return res
    print("   no cached CFR result for this tree")
    return None


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "TOY"
    games = int(sys.argv[2]) if len(sys.argv) > 2 else len(EVAL_SEEDS)
    light = len(sys.argv) > 3 and sys.argv[3] == "light"
    rs = getattr(rules, name)
    seeds = range(EVAL_SEEDS.start, EVAL_SEEDS.start + games)
    rnd, heu = PlayFirstRandom(rs), Heuristic(rs)

    lazy = compile_tree(rs, (Level.L1,) if light else (Level.L1, Level.L2), True, cache=CACHE)
    world = None if light else compile_tree(rs, (Level.L1, Level.L3), True, cache=CACHE)
    print(f"== {name}: {lazy}" + ("" if world is None else f"\n   world: {world}"))
    sig = {a.name: (Level.L1, tabulate(lazy, Level.L1, a)) for a in (rnd, heu)}
    sigw = None if world is None else {a.name: (Level.L1, tabulate(world, Level.L1, a)) for a in (rnd, heu)}
    # the tree each level's best response lives on
    where = {Level.L1: lazy, Level.L2: lazy, Level.L3: world}
    levels = (Level.L1,) if light else (Level.L1, Level.L2, Level.L3)

    print("\n== 1. row matchups (exact, score of the first-named, seat 0 / seat 1)")
    for label, a, b in (("random vs random", "random", "random"), ("heuristic vs random", "heuristic", "random"), ("heuristic vs heuristic", "heuristic", "heuristic")):
        print(f"   {label:24s} {score(evaluate(lazy, sig[a], sig[b])[0]):.4f}  {score(evaluate(lazy, sig[b], sig[a])[1]):.4f}")

    print(f"\n== 2. best-response columns: exact | exact on the {games} held-out seeds | simulated on them")
    print(f"   {'opponent':10s} {'seat':4s} {'level':5s} {'exact':>7s} {'on seeds':>8s}  {'sim':>7s} {'95% CI':>16s}  ok  misses")
    brs = {}
    for opp in (rnd, heu):
        for seat in (0, 1):
            for level in levels:
                tree = where[level]
                s_opp = (sig if tree is lazy else sigw)[opp.name]
                br = best_response(tree, level, s_opp, seat)
                brs[(opp.name, seat, level)] = br
                agent = CompiledAgent(tree, level, br.sigma, name=f"br_{level.name.lower()}_{opp.name}")
                r = match(rs, (agent, opp), seeds, swap=(seat == 1), levels=(level, Level.L1))
                agents = (agent, opp) if seat == 0 else (opp, agent)
                lv = (level, Level.L1) if seat == 0 else (Level.L1, level)
                on = score(evaluate_on_seeds(rs, agents, seeds, lv)[seat])
                ok = check(abs(r.score - on) <= (r.score_ci[1] - r.score_ci[0]) / 2 and agent.misses == 0, f"BR {level.name} seat {seat} vs {opp.name} on seeds")
                print(f"   {opp.name:10s} {seat:<4d} {level.name:5s} {br.score:7.4f} {on:8.4f}  {r.score:7.4f} [{r.score_ci[0]:.4f}, {r.score_ci[1]:.4f}]  {ok}  {agent.misses}")
                sys.stdout.flush()
            if not light:
                l1, l2, l3 = (brs[(opp.name, seat, lv)].score for lv in levels)
                ok = check(l1 <= l2 + 1e-9 <= l3 + 2e-9, f"L1 <= L2 <= L3 seat {seat} vs {opp.name}")
                print(f"   {'':10s} {'':4s} gaps  L2-L1 {100 * (l2 - l1):5.1f}  L3-L2 {100 * (l3 - l2):5.1f}  {ok}")

    print("\n== 3. Nash L1")
    res = load_nash(lazy, name)
    nash = None
    if res is not None:
        ex = exploitability(lazy, Level.L1, res.avg)
        lo, hi = ex.bracket
        ok = check(lo - 1e-9 <= score(ex.value) <= hi + 1e-9, "Nash value inside its bracket")
        print(f"   seat-0 score {score(ex.value):.4f} in [{lo:.4f}, {hi:.4f}], NashConv {100 * ex.score_points:.3f} pts  {ok}")
        rows = {"random": sig["random"][1], "heuristic": sig["heuristic"][1], "br_l1(random)": by_seat(lazy, brs[("random", 0, Level.L1)], brs[("random", 1, Level.L1)]), "nash_l1": res.avg}
        print(f"   {'row':16s} {'seat':4s} {'BR L1 to it':>11s} {'Nash (resp.)':>12s} {'exploitability':>14s}  ok")
        for label, s in rows.items():
            for seat in (0, 1):
                br = best_response(lazy, Level.L1, (Level.L1, s), seat ^ 1)
                nash_resp = 1 - score(ex.value) if seat == 0 else score(ex.value)
                e = 100 * (br.score - nash_resp)
                ok = check(e >= -100 * ex.score_points - 1e-6, f"exploitability of {label} seat {seat} non-negative")
                print(f"   {label:16s} {seat:<4d} {br.score:11.4f} {nash_resp:12.4f} {e:14.2f}  {ok}")
        nash = res.agent(lazy)
        nash_w = None if world is None else (Level.L1, tabulate(world, Level.L1, nash))
        print("   exact Nash row / column (score of the first-named, seat 0 / seat 1)")
        for a, b in (("nash_l1", "random"), ("nash_l1", "heuristic"), ("nash_l1", "nash_l1"), ("random", "nash_l1"), ("heuristic", "nash_l1"), ("br_l1(random)", "nash_l1")):
            print(f"   {a + ' vs ' + b:26s} {score(evaluate(lazy, (Level.L1, rows[a]), (Level.L1, rows[b]))[0]):.4f}  {score(evaluate(lazy, (Level.L1, rows[b]), (Level.L1, rows[a]))[1]):.4f}")
        sys.stdout.flush()

    print(f"\n== 4. mirrored evaluator, {games} held-out pairs: sim per seat vs exact on those seeds; moments vs the world tree")
    # best-response row agents play from both seats; their strategies are computed on the world tree when it exists
    # (same L1 sets, same values) so the exact moments use the strategy that is played
    def br_row(opp_name: str):
        tree = lazy if world is None else world
        s_opp = (sig if tree is lazy else sigw)[opp_name]
        s = by_seat(tree, best_response(tree, Level.L1, s_opp, 0), best_response(tree, Level.L1, s_opp, 1))
        return CompiledAgent(tree, Level.L1, s, name=f"br_l1_{opp_name}"), (Level.L1, s)

    br_rnd, s_br_rnd = br_row("random")
    br_heu, s_br_heu = br_row("heuristic")
    col = sigw if world is not None else sig
    matchups = [("heuristic vs random", heu, rnd, col["heuristic"], col["random"]), ("br_l1(random) vs random", br_rnd, rnd, s_br_rnd, col["random"]), ("br_l1(heuristic) vs heuristic", br_heu, heu, s_br_heu, col["heuristic"]), ("br_l1(random) vs heuristic", br_rnd, heu, s_br_rnd, col["heuristic"])]
    if nash is not None:
        matchups += [("nash vs random", nash, rnd, nash_w, col["random"]), ("nash vs heuristic", nash, heu, nash_w, col["heuristic"])]
    print(f"   {'matchup':30s} {'seat':4s} {'on seeds':>8s} {'sim ± 95%':>17s} ok | {'corr ex/sim':>15s} {'splits ex/sim':>15s} {'var ratio ex/sim':>17s} misses")
    played = {}
    for label, a, b, sa, sb in matchups:
        pairs = play_pairs(rs, (a, b), seeds)
        played[label] = pairs
        m = estimate_mirrored(pairs)
        on = (score(evaluate_on_seeds(rs, (a, b), pairs.seeds)[0]), score(evaluate_on_seeds(rs, (b, a), pairs.seeds)[1]))
        misses = sum(getattr(x, "misses", 0) for x in (a, b))
        moments = ""
        if world is not None:
            exm = exact_mirrored(world, sa, sb)
            moments = f"{exm.corr:7.3f}/{m.corr:7.3f} {exm.splits:7.3f}/{m.splits:7.3f} {exm.var_ratio:8.3f}/{m.var_ratio:8.3f}"
            check(abs(exm.mean[0] - score(evaluate(world, sa, sb)[0])) < 1e-9, f"exact mirrored mean of {label}")
        for seat in (0, 1):
            e = m.seat[seat]
            ok = check(abs(e.mean - on[seat]) <= e.half and misses == 0, f"mirrored {label} seat {seat} on seeds")
            print(f"   {label if seat == 0 else '':30s} {seat:<4d} {on[seat]:8.4f} {e.mean:8.4f} ± {e.half:.4f} {ok} | {moments if seat == 0 else '':45s} {misses if seat == 0 else ''}")
        sys.stdout.flush()

    print("\n== 5. paired comparisons (first minus second row agent, same column, same deals; sampled)")
    rr = play_pairs(rs, (rnd, rnd), seeds)
    hh = play_pairs(rs, (heu, heu), seeds)
    for label, a, b in (("heuristic - random, vs random", played["heuristic vs random"], rr), ("br_l1(random) - heuristic, vs random", played["br_l1(random) vs random"], played["heuristic vs random"]), ("br_l1(heuristic) - heuristic, vs heuristic", played["br_l1(heuristic) vs heuristic"], hh), ("br_l1(heuristic) - br_l1(random), vs heuristic", played["br_l1(heuristic) vs heuristic"], played["br_l1(random) vs heuristic"])):
        c = compare(a, b)
        print(f"   {label:48s} " + "  ".join(f"seat {seat}: {str(c.seat[seat]):>18s} corr={c.corr[seat]:+.3f} var ratio={c.var_ratio(seat):.3f}" for seat in (0, 1)))

    print(f"\n== {len(FAILS)} failed checks" + (": " + "; ".join(FAILS) if FAILS else ""))


if __name__ == "__main__":
    main()
