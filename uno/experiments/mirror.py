"""Step 7: the mirrored evaluator against exact values on a ruleset.

    uv run uno mirror [ruleset=TOY] [games=4000]

Plays `games` held-out seeds (uno.eval.play.EVAL_SEEDS) from both seats for each
row-vs-column matchup and prints, per matchup: the exact per-seat values and
seat mean; the sampled per-seat estimates and the seat mean under the raw and
the mirrored analysis; the correlation between the two seats' scores on the
same deal, the share of split pairs and the variance ratio, exact and sampled;
and the games a 3-point edge needs at 3 sigma under each design. Then the
paired comparisons between row agents against the same column.
"""

from __future__ import annotations

import sys

from uno.agents.baseline import BySeat, Heuristic, PlayFirstRandom
from uno.agents.tabular import TabularAgent
from uno.eval.estimate import bootstrap_moments, compare, estimate_mirrored, games_needed, play_pairs
from uno.eval.play import EVAL_SEEDS
from uno.game import rules
from uno.game.observe import Level
from uno.reference.mirrored import exact_mirrored
from uno.reference.solve import best_response, scores_on_seeds

EDGE = 0.03  # score points; the intro's "3-point edge near 60 %"


def br_agent(rs, opp, name):
    return BySeat(*(TabularAgent(rs, best_response(rs, opp, seat, Level.L1)) for seat in (0, 1)), name=name)


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    rs = getattr(rules, argv[0] if len(argv) > 0 else "TOY")
    games = int(argv[1]) if len(argv) > 1 else len(EVAL_SEEDS)
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
    print(f"== mirrored evaluator on {argv[0] if len(argv) > 0 else 'TOY'}, {games} pairs of held-out seeds")
    print("   exact = whole game; on seeds = exact on the played deals and pile orders; sim = sampled")
    for name, a, b in matchups:
        pairs = play_pairs(rs, (a, b), seeds)
        played[name] = pairs
        m = estimate_mirrored(pairs)
        ex = exact_mirrored(rs, (a, b))
        os0, os1 = scores_on_seeds(rs, (a, b), pairs.seeds), scores_on_seeds(rs, (b, a), pairs.seeds)
        on0, on1 = os0.mean[0], os1.mean[1]
        z0, z1 = os0.z(m.seat[0].mean, 0), os1.z(m.seat[1].mean, 1)
        ok0, ok1 = os0.ok(m.seat[0].mean, 0), os1.ok(m.seat[1].mean, 1)
        bs = bootstrap_moments(pairs)
        covered = bs.covers(ex)
        print(f"\n   {name}")
        print(f"      {'':10s} {'seat 0':>16s} {'seat 1':>16s} {'mean':>16s}")
        print(f"      {'exact':10s} {ex.mean[0]:16.4f} {ex.mean[1]:16.4f} {ex.pair_mean:16.4f}")
        print(f"      {'on seeds':10s} {on0:16.4f} {on1:16.4f} {(on0 + on1) / 2:16.4f}")
        print(f"      {'sim raw':10s} {str(m.seat[0]):>16s} {str(m.seat[1]):>16s} {str(m.raw):>16s}  z={z0:+.2f} / {z1:+.2f} {'ok' if ok0 and ok1 else 'XX'}")
        misses = sum(t.misses for x in (a, b) if isinstance(x, BySeat) for t in x.seats)
        print(f"      {'mirrored':10s} {'':16s} {'':16s} {str(m.mirrored):>16s}" + (f"  misses={misses}" if misses else ""))
        print(f"      {'':10s} {'corr':>8s} {'splits':>8s} {'var ratio':>10s} {'sd0':>8s} {'sd1':>8s}")
        print(f"      {'exact':10s} {ex.corr:8.3f} {ex.splits:8.3f} {ex.var_ratio:10.3f} {ex.var[0] ** 0.5:8.4f} {ex.var[1] ** 0.5:8.4f}")
        print(f"      {'sim':10s} {m.corr:8.3f} {m.splits:8.3f} {m.var_ratio:10.3f} {m.seat[0].var ** 0.5:8.4f} {m.seat[1].var ** 0.5:8.4f}")
        missed = [k for k, v in covered.items() if not v]
        print(f"      {'99.8% CI':10s} [{bs.corr[0]:+.3f}, {bs.corr[1]:+.3f}] [{bs.splits[0]:.3f}, {bs.splits[1]:.3f}] [{bs.var_ratio[0]:.3f}, {bs.var_ratio[1]:.3f}]  exact inside: {'ok' if not missed else 'XX ' + ', '.join(missed)}")
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
