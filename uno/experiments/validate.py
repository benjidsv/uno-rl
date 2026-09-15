"""Step 8.7: the validation checks of steps 5-8 on a ruleset, from the compiled trees.

    uv run uno validate [ruleset=TOY] [games=4000] [light]

The `ladder` and `mirror` experiments run the pure-Python reference
solvers, which do not fit the TOY2 family (its full game is ~10M histories,
~10 GB in the reference walkers). This experiment does the same checks from the
cached quotient trees (docs/spec.md sections 16 and 18):

1. the row matchups (exact);
2. best-response columns (L1 / L2 on the lazy tree, L3 on the world tree)
   against Random and the Heuristic, per seat: the exact score, the exact
   score on the held-out seeds, the simulation of the best-response agent on
   those seeds, no misses, L1 <= L2 <= L3;
3. the Nash agent (from the cache): NashConv, the profile's value inside its
   bracket, every row's exploitability non-negative (the best response to
   Random both as solved and with its unreached sets filled with Nash), the
   exact Nash row and column;
4. the mirrored evaluator on the step 7 matchups plus the Nash row: sampled
   per-seat means against the exact values on the same seeds, and the exact
   variances, correlation, split share and variance ratio of the world tree
   inside the bootstrap CIs of the sampled ones (99.8 % per moment, a
   family-wise 99 % per matchup);
5. the paired comparisons (sampled).

`light`: lazy tree with the L1 keying only (no L2 / L3 columns, no exact
mirrored moments), for TOY2_FULL.

Every check prints ok / XX and the last line counts the failures. A simulation
is compared with the exact value on the same seeds through
z = (sim - exact on seeds) / sqrt(action-noise variance / n), and must satisfy
|z| <= 3 (0.27 % false alarms per check, docs/spec.md section 14). The action
noise is exact (`scores_on_seeds`), so the check is calibrated, unlike a CI of
the total variance, which also counts the deal and pile luck the seeds fix.
"""

from __future__ import annotations

import sys

from uno import store
from uno.agents.baseline import Heuristic, PlayFirstRandom
from uno.agents.tabular import CompiledAgent
from uno.compiled.cfr import CFRResult
from uno.compiled.solvers import best_response, evaluate, exact_mirrored, exploitability
from uno.compiled.strategies import br_row, tabulate
from uno.compiled.tree import CompiledTree, compile_tree
from uno.eval.estimate import bootstrap_moments, compare, estimate_mirrored, play_pairs
from uno.eval.play import EVAL_SEEDS, match
from uno.experiments.common import by_seat, score
from uno.game import rules
from uno.game.observe import Level
from uno.reference.solve import scores_on_seeds
from uno.store import CACHE

FAILS: list[str] = []


def check(ok: bool, what: str) -> str:
    if not ok:
        FAILS.append(what)
    return "ok" if ok else "XX"


def load_nash(tree: CompiledTree) -> CFRResult | None:
    res = store.load_nash(tree)
    if res is None:
        print("   no cached CFR result for this tree")
    else:
        print(f"   Nash strategy: {res.iterations} iterations of {res.config.get('variant') or 'cfr+'}, NashConv {100 * res.final.nashconv:.3f} pts")
    return res


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    name = argv[0] if len(argv) > 0 else "TOY"
    games = int(argv[1]) if len(argv) > 1 else len(EVAL_SEEDS)
    light = len(argv) > 2 and argv[2] == "light"
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

    print(f"\n== 2. best-response columns: exact | exact on the {games} held-out seeds | simulated on them, z against the action noise")
    print(f"   {'opponent':10s} {'seat':4s} {'level':5s} {'exact':>7s} {'on seeds':>8s}  {'sim':>7s} {'z':>6s}  ok  misses")
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
                on = scores_on_seeds(rs, agents, seeds, lv)
                z = on.z(r.score, seat)
                ok = check(on.ok(r.score, seat) and agent.misses == 0, f"BR {level.name} seat {seat} vs {opp.name} on seeds (z = {z:+.2f})")
                print(f"   {opp.name:10s} {seat:<4d} {level.name:5s} {br.score:7.4f} {on.mean[seat]:8.4f}  {r.score:7.4f} {z:+6.2f}  {ok}  {agent.misses}")
                sys.stdout.flush()
            if not light:
                l1, l2, l3 = (brs[(opp.name, seat, lv)].score for lv in levels)
                ok = check(l1 <= l2 + 1e-9 <= l3 + 2e-9, f"L1 <= L2 <= L3 seat {seat} vs {opp.name}")
                print(f"   {'':10s} {'':4s} gaps  L2-L1 {100 * (l2 - l1):5.1f}  L3-L2 {100 * (l3 - l2):5.1f}  {ok}")

    print("\n== 3. Nash L1")
    res = load_nash(lazy)
    nash = None
    if res is not None:
        ex = exploitability(lazy, Level.L1, res.avg)
        lo, hi = ex.bracket
        ok = check(lo - 1e-9 <= score(ex.value) <= hi + 1e-9, "Nash value inside its bracket")
        print(f"   seat-0 score {score(ex.value):.4f} in [{lo:.4f}, {hi:.4f}], NashConv {100 * ex.score_points:.3f} pts  {ok}")
        (_, filled_sigma), filled = br_row(lazy, sig["random"], completion=res.avg)
        n_sets = [int((lazy.keyings[Level.L1].info_player == seat).sum()) for seat in (0, 1)]
        print(f"   BR L1(random) leaves {filled[0].unreached:,} / {n_sets[0]:,} seat-0 and {filled[1].unreached:,} / {n_sets[1]:,} seat-1 L1 sets unreached; '+nash' plays Nash there")
        rows = {
            "random": sig["random"][1],
            "heuristic": sig["heuristic"][1],
            "br_l1(random)": by_seat(lazy, [brs[("random", 0, Level.L1)], brs[("random", 1, Level.L1)]])[1],
            "br_l1(random)+nash": filled_sigma,
            "nash_l1": res.avg,
        }
        print(f"   {'row':20s} {'seat':4s} {'BR L1 to it':>11s} {'Nash (resp.)':>12s} {'exploitability':>14s}  ok")
        for label, s in rows.items():
            for seat in (0, 1):
                br = best_response(lazy, Level.L1, (Level.L1, s), seat ^ 1)
                nash_resp = 1 - score(ex.value) if seat == 0 else score(ex.value)
                e = 100 * (br.score - nash_resp)
                ok = check(e >= -100 * ex.score_points - 1e-6, f"exploitability of {label} seat {seat} non-negative")
                print(f"   {label:20s} {seat:<4d} {br.score:11.4f} {nash_resp:12.4f} {e:14.2f}  {ok}")
        nash = res.agent(lazy)
        nash_w = None if world is None else (Level.L1, tabulate(world, Level.L1, nash))
        print("   exact Nash row / column (score of the first-named, seat 0 / seat 1)")
        for a, b in (("nash_l1", "random"), ("nash_l1", "heuristic"), ("nash_l1", "nash_l1"), ("random", "nash_l1"), ("heuristic", "nash_l1"), ("br_l1(random)", "nash_l1"), ("br_l1(random)+nash", "nash_l1")):
            print(f"   {a + ' vs ' + b:30s} {score(evaluate(lazy, (Level.L1, rows[a]), (Level.L1, rows[b]))[0]):.4f}  {score(evaluate(lazy, (Level.L1, rows[b]), (Level.L1, rows[a]))[1]):.4f}")
        sys.stdout.flush()

    print(f"\n== 4. mirrored evaluator, {games} held-out pairs: sim per seat vs exact on those seeds (z); exact moments vs bootstrap CIs (99.8 % each)")
    # best-response row agents play from both seats; their strategies are computed on the world tree when it exists
    # (same L1 sets, same values) so the exact moments use the strategy that is played
    def br_agent(opp_name: str):
        tree = lazy if world is None else world
        s_opp = (sig if tree is lazy else sigw)[opp_name]
        s = by_seat(tree, [best_response(tree, Level.L1, s_opp, 0), best_response(tree, Level.L1, s_opp, 1)])[1]
        return CompiledAgent(tree, Level.L1, s, name=f"br_l1_{opp_name}"), (Level.L1, s)

    br_rnd, s_br_rnd = br_agent("random")
    br_heu, s_br_heu = br_agent("heuristic")
    col = sigw if world is not None else sig
    matchups = [("heuristic vs random", heu, rnd, col["heuristic"], col["random"]), ("br_l1(random) vs random", br_rnd, rnd, s_br_rnd, col["random"]), ("br_l1(heuristic) vs heuristic", br_heu, heu, s_br_heu, col["heuristic"]), ("br_l1(random) vs heuristic", br_rnd, heu, s_br_rnd, col["heuristic"])]
    if nash is not None:
        matchups += [("nash vs random", nash, rnd, nash_w, col["random"]), ("nash vs heuristic", nash, heu, nash_w, col["heuristic"])]
    print(f"   {'matchup':30s} {'seat':4s} {'on seeds':>8s} {'sim ± 95%':>17s} {'z':>6s} ok  misses")
    played = {}
    for label, a, b, sa, sb in matchups:
        pairs = play_pairs(rs, (a, b), seeds)
        played[label] = pairs
        m = estimate_mirrored(pairs)
        ons = (scores_on_seeds(rs, (a, b), pairs.seeds), scores_on_seeds(rs, (b, a), pairs.seeds))
        misses = sum(getattr(x, "misses", 0) for x in (a, b))
        for seat in (0, 1):
            e = m.seat[seat]
            z = ons[seat].z(e.mean, seat)
            ok = check(ons[seat].ok(e.mean, seat) and misses == 0, f"mirrored {label} seat {seat} on seeds (z = {z:+.2f})")
            print(f"   {label if seat == 0 else '':30s} {seat:<4d} {ons[seat].mean[seat]:8.4f} {e.mean:8.4f} ± {e.half:.4f} {z:+6.2f} {ok}  {misses if seat == 0 else ''}")
        if world is not None:
            exm = exact_mirrored(world, sa, sb)
            check(abs(exm.mean[0] - score(evaluate(world, sa, sb)[0])) < 1e-9, f"exact mirrored mean of {label}")
            bs = bootstrap_moments(pairs)
            covered = bs.covers(exm)
            ok = check(all(covered.values()), f"mirrored {label}: exact {', '.join(k for k, v in covered.items() if not v)} outside its bootstrap CI")
            print(
                f"   {'':30s} exact / sim [99.8 % CI]: corr {exm.corr:+.3f} / {m.corr:+.3f} [{bs.corr[0]:+.3f}, {bs.corr[1]:+.3f}]  "
                f"splits {exm.splits:.3f} / {m.splits:.3f} [{bs.splits[0]:.3f}, {bs.splits[1]:.3f}]  "
                f"var ratio {exm.var_ratio:.3f} / {m.var_ratio:.3f} [{bs.var_ratio[0]:.3f}, {bs.var_ratio[1]:.3f}]  {ok}"
            )
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
