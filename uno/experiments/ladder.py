"""Steps 5 and 6: exact best-response columns of the ladder on a ruleset.

    uv run uno ladder [ruleset=TOY] [games=4000]

For each row agent (Random, Heuristic), each seat and each level (L1, L2, L3):
the exact best-response score, and the score of the best-response agent
replayed through `match` on `games` seeds. A fixed seed range carries its own
deal and pile luck, so the simulation is compared with the exact value *on
those games* ("on seeds", real deal and real pile order): the difference is
action-sampling noise only, scored as z = difference / sqrt(action-noise
variance / n) with |z| <= 3 (docs/foundations/spec.md section 14); when both agents are
deterministic it is exactly zero. Also prints the exact values of
the row agents against each other, to compare with the sampled step 4 numbers.
"""

from __future__ import annotations

import sys

from uno.agents.baseline import Heuristic, PlayFirstRandom
from uno.agents.tabular import TabularAgent
from uno.eval.play import EVAL_SEEDS, match
from uno.game import rules
from uno.game.observe import Level
from uno.reference.solve import best_response, evaluate, scores_on_seeds


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    rs = getattr(rules, argv[0] if len(argv) > 0 else "TOY")
    games = int(argv[1]) if len(argv) > 1 else len(EVAL_SEEDS)
    rnd, heu = PlayFirstRandom(rs), Heuristic(rs)
    seeds = range(EVAL_SEEDS.start, EVAL_SEEDS.start + games)  # held out, see uno.eval.play

    print("== exact values of the row agents (score of the first-named, seat 0 / seat 1)")
    for name, a, b in (("random vs random", rnd, rnd), ("heuristic vs random", heu, rnd), ("heuristic vs heuristic", heu, heu)):
        v0 = (evaluate(rs, (a, b))[0] + 1) / 2
        v1 = (evaluate(rs, (b, a))[1] + 1) / 2
        print(f"   {name:24s} {v0:.4f}  {v1:.4f}")

    print(f"\n== best responses (exact score | exact on the {games} seeds | simulated on them)")
    print(f"   {'opponent':10s} {'seat':4s} {'level':5s} {'exact':>7s} {'on seeds':>8s}  {'sim':>7s} {'95% CI':>16s}  {'ok':2s} {'hist':>9s} {'sets':>7s} {'sec':>5s}")
    rows = {}
    for opp in (rnd, heu):
        for seat in (0, 1):
            for level in (Level.L1, Level.L2, Level.L3):
                br = best_response(rs, opp, seat, level)
                agent = TabularAgent(rs, br)
                r = match(rs, (agent, opp), seeds, swap=(seat == 1), levels=(level, Level.L1))
                agents = (agent, opp) if seat == 0 else (opp, agent)
                lv = (level, Level.L1) if seat == 0 else (Level.L1, level)
                on = scores_on_seeds(rs, agents, seeds, lv)
                on_seeds, z = on.mean[seat], on.z(r.score, seat)
                ok = on.ok(r.score, seat)
                rows[(opp.name, seat, level)] = br.score
                print(
                    f"   {opp.name:10s} {seat:<4d} {level.name:5s} {br.score:7.4f} {on_seeds:8.4f}  {r.score:7.4f} "
                    f"[{r.score_ci[0]:.4f}, {r.score_ci[1]:.4f}] z={z:+5.2f} {'ok' if ok else 'XX'} "
                    f"{br.histories:9d} {br.infosets:7d} {br.seconds:5.1f}"
                    + (f" misses={agent.misses}" if agent.misses else "")
                )
                sys.stdout.flush()

    print("\n== gaps (score points)")
    print(f"   {'opponent':10s} {'seat':4s} {'L2-L1 (opp hand)':>17s} {'L3-L2 (pile order)':>19s}")
    for opp in (rnd, heu):
        for seat in (0, 1):
            l1, l2, l3 = (rows[(opp.name, seat, lv)] for lv in (Level.L1, Level.L2, Level.L3))
            print(f"   {opp.name:10s} {seat:<4d} {100 * (l2 - l1):17.1f} {100 * (l3 - l2):19.1f}")


if __name__ == "__main__":
    main()
