"""Step 7: the mirrored evaluator against exact values on a ruleset.

    uv run python scripts/mirror.py [ruleset=TOY] [games=4000]

Plays `games` held-out seeds (uno.play.EVAL_SEEDS) from both seats for each
row-vs-column matchup and prints, per matchup: the exact per-seat values and
seat mean; the sampled per-seat estimates and the seat mean under the raw and
the mirrored analysis; the correlation between the two seats' scores on the
same deal, the share of split pairs and the variance ratio, exact and sampled;
and the games a 3-point edge needs at 3 sigma under each design. Then the
paired comparisons between row agents against the same column.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uno import rules
from uno.agents import BySeat, Heuristic, PlayFirstRandom
from uno.estimate import compare, estimate_mirrored, exact_mirrored, games_needed, play_pairs
from uno.observe import Level
from uno.play import EVAL_SEEDS
from uno.solve import TabularAgent, best_response, evaluate_on_seeds

EDGE = 0.03  # score points; the intro's "3-point edge near 60 %"


def br_agent(rs, opp, name):
    return BySeat(*(TabularAgent(rs, best_response(rs, opp, seat, Level.L1)) for seat in (0, 1)), name=name)


def main() -> None:
    rs = getattr(rules, sys.argv[1] if len(sys.argv) > 1 else "TOY")
    games = int(sys.argv[2]) if len(sys.argv) > 2 else len(EVAL_SEEDS)
    seeds = range(EVAL_SEEDS.start, EVAL_SEEDS.start + games)
    rnd, heu = PlayFirstRandom(rs), Heuristic(rs)
    br_rnd = br_agent(rs, rnd, "br_l1_to_random")
    br_heu = br_agent(rs, heu, "br_l1_to_heuristic")

    matchups = (
        ("heuristic vs random", heu, rnd),
        ("br_l1(random) vs random", br_rnd, rnd),
        ("br_l1(heuristic) vs heuristic", br_heu, heu),
        ("br_l1(random) vs heuristic", br_rnd, heu),
    )
    played = {}
    print(f"== mirrored evaluator on {sys.argv[1] if len(sys.argv) > 1 else 'TOY'}, {games} pairs of held-out seeds")
    print("   exact = whole game; on seeds = exact on the played deals and pile orders; sim = sampled")
    for name, a, b in matchups:
        pairs = play_pairs(rs, (a, b), seeds)
        played[name] = pairs
        m = estimate_mirrored(pairs)
        ex = exact_mirrored(rs, (a, b))
        on0 = (evaluate_on_seeds(rs, (a, b), pairs.seeds)[0] + 1) / 2
        on1 = (evaluate_on_seeds(rs, (b, a), pairs.seeds)[1] + 1) / 2
        ok0 = abs(m.seat[0].mean - on0) <= m.seat[0].half
        ok1 = abs(m.seat[1].mean - on1) <= m.seat[1].half
        print(f"\n   {name}")
        print(f"      {'':10s} {'seat 0':>16s} {'seat 1':>16s} {'mean':>16s}")
        print(f"      {'exact':10s} {ex.mean[0]:16.4f} {ex.mean[1]:16.4f} {ex.pair_mean:16.4f}")
        print(f"      {'on seeds':10s} {on0:16.4f} {on1:16.4f} {(on0 + on1) / 2:16.4f}")
        print(f"      {'sim raw':10s} {str(m.seat[0]):>16s} {str(m.seat[1]):>16s} {str(m.raw):>16s}  {'ok' if ok0 and ok1 else 'XX'}")
        misses = sum(t.misses for x in (a, b) if isinstance(x, BySeat) for t in x.seats)
        print(f"      {'mirrored':10s} {'':16s} {'':16s} {str(m.mirrored):>16s}" + (f"  misses={misses}" if misses else ""))
        print(f"      {'':10s} {'corr':>8s} {'splits':>8s} {'var ratio':>10s} {'sd0':>8s} {'sd1':>8s}")
        print(f"      {'exact':10s} {ex.corr:8.3f} {ex.splits:8.3f} {ex.var_ratio:10.3f} {ex.var[0] ** 0.5:8.4f} {ex.var[1] ** 0.5:8.4f}")
        print(f"      {'sim':10s} {m.corr:8.3f} {m.splits:8.3f} {m.var_ratio:10.3f} {m.seat[0].var ** 0.5:8.4f} {m.seat[1].var ** 0.5:8.4f}")
        g_raw, g_mir = (games_needed(EDGE, v) for v in (2 * ex.var_raw, 2 * ex.var_mirrored))
        g_seat = tuple(games_needed(EDGE, v) for v in ex.var)
        print(
            f"      games for a {100 * EDGE:.0f}-point edge at 3 sigma (exact variance): "
            f"seat 0 alone {g_seat[0]:.0f}, seat 1 alone {g_seat[1]:.0f}, seat mean raw {g_raw:.0f}, mirrored {g_mir:.0f}"
        )
        sys.stdout.flush()

    print("\n== paired comparisons (first minus second row agent, same column, same deals)")
    rr = play_pairs(rs, (rnd, rnd), seeds)
    hh = play_pairs(rs, (heu, heu), seeds)
    for name, a, b in (
        ("heuristic - random, vs random", played["heuristic vs random"], rr),
        ("br_l1(random) - heuristic, vs random", played["br_l1(random) vs random"], played["heuristic vs random"]),
        ("br_l1(heuristic) - heuristic, vs heuristic", played["br_l1(heuristic) vs heuristic"], hh),
        ("br_l1(heuristic) - br_l1(random), vs heuristic", played["br_l1(heuristic) vs heuristic"], played["br_l1(random) vs heuristic"]),
    ):
        c = compare(a, b)
        print(f"\n   {name}")
        for seat in (0, 1):
            d, u = c.seat[seat], c.unpaired[seat]
            print(
                f"      seat {seat}: paired {str(d):>18s}  unpaired ± {u.half:.4f}  corr={c.corr[seat]:+.3f} var ratio={c.var_ratio(seat):.3f}  "
                f"games for {100 * EDGE:.0f} pts at 3 sigma: unpaired {games_needed(EDGE, 2 * u.var):.0f}, paired {games_needed(EDGE, 2 * d.var):.0f}"
            )
        print(f"      mean:   paired {c.mean}")


if __name__ == "__main__":
    main()
