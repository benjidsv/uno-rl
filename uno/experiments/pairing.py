"""Step 9: pairing on full UNO with near-identical policies, draws from the pile against tickets.

    uv run uno pairing [games=4000] [toy_seeds=20000]

Step 8.5 found that pairing removes ~10 % of the variance on full UNO, from
Heuristic against Random, two policies that part on the first move. This
measures what steps 10-11 need, two variants of one policy, and a draw rule that
keeps the games of a pair coupled after their draw counts part (`draws =
"tickets"`, docs/foundations/spec.md section 8):

1. tickets are the same game: on TOY, TOY_FULL and a small ruleset with duplicate
   cards, +2 and +4 (the ticket paths the named toys never take), the exact score on each seed's
   real deal and tickets, averaged over many seeds, against the whole-game exact
   value of the pile rules (z on the deal variance), and simulations against the
   exact values on their seeds (z on the action noise), for both draw rules;
2. FULL_UNO, 4000 held-out seeds, both draw rules: the per-seat means (the two
   rules estimate the same values), the mirrored evaluator (correlation, splits,
   variance ratio), and paired comparisons of the Heuristic against its
   variants, against the same column: variance ratio paired / unpaired per seat
   and deals for a 3-point difference at 3 sigma with 80 % power.

Variants of the Heuristic: `heuristic_t1` (the opponent counts as low at 1 card,
not 2), `heuristic_eps05` (5 % Random mixed in) on the Heuristic's action rng
stream as a checkpoint of one policy would be, and `heuristic_eps05_own` on its
own stream (`uno.eval.play.stream_name`).
"""

from __future__ import annotations

import sys
from dataclasses import replace

from uno.agents.baseline import Heuristic, Mixture, PlayFirstRandom, UniformLegal
from uno.compiled.solvers import evaluate
from uno.compiled.strategies import tabulate
from uno.compiled.tree import compile_tree
from uno.eval.estimate import Comparison, Pairs, compare, correlation, estimate, estimate_mirrored, play_pairs
from uno.eval.play import EVAL_SEEDS, match
from uno.experiments.common import score
from uno.game import rules
from uno.game.observe import Level
from uno.reference.solve import scores_on_seeds
from uno.store import CACHE, load_nash

# duplicate cards, a +2 per color and a wild +4: copy tickets, penalty draws and the setup tail (777k histories)
DUPS = rules.Ruleset(colors=2, rank_copies=(2, 1), skip_per_color=0, reverse_per_color=0, draw_two_per_color=1, wild_count=1, wild_draw_four_count=1, hand_size=3, forced_play=True, reshuffle="none", max_turns=None)
Z80 = 3.0 + 0.8416  # a 3-sigma test detecting the edge with 80 % power
EDGE = 0.03
FAILS: list[str] = []


def check(ok: bool, what: str) -> str:
    if not ok:
        FAILS.append(what)
    return "ok" if ok else "XX"


def same_game(n_seeds: int, games: int) -> None:
    print(f"== 1. tickets are the same game (exact whole-game value of the pile rules; {n_seeds:,} seeds for the deal model, {games:,} simulated)")
    print(f"   {'ruleset':9s} {'matchup':24s} {'draws':7s} {'whole game':>10s} {'mean on seeds':>13s} {'z deal':>7s} {'sim':>7s} {'z action':>8s}  ok")
    for name, rs in (("TOY", rules.TOY), ("TOY_FULL", rules.TOY_FULL), ("DUPS", DUPS)):
        tree = compile_tree(rs, (Level.L1,), True, cache=CACHE)
        rnd, heu = PlayFirstRandom(rs), Heuristic(rs)
        matchups = [("random vs random", rnd, rnd, None), ("heuristic vs random", heu, rnd, None)]
        if name == "TOY_FULL":
            matchups.append(("uniform_legal vs random", UniformLegal(rs), rnd, None))  # voluntary draws: draw counts part
            nash = load_nash(tree)
            if nash is not None:
                matchups.append(("nash vs random", nash.agent(tree), rnd, nash.avg))
        for label, a, b, sigma_a in matchups:
            sa = (Level.L1, sigma_a if sigma_a is not None else tabulate(tree, Level.L1, a))
            sb = (Level.L1, tabulate(tree, Level.L1, b))
            exact = score(evaluate(tree, sa, sb)[0])
            n = n_seeds if sigma_a is None else n_seeds // 4
            seeds = range(3_000_000, 3_000_000 + n)
            for draws in ("pile", "tickets"):
                r = replace(rs, draws=draws)
                on = scores_on_seeds(r, (a, b), seeds)
                zd = on.z_global(exact, 0)
                sim = match(r, (a, b), seeds[:games])
                on_g = scores_on_seeds(r, (a, b), seeds[:games])
                za = on_g.z(sim.score, 0)
                ok = check(abs(zd) <= 3 and abs(za) <= 3, f"{name} {label} {draws}: z deal {zd:+.2f}, z action {za:+.2f}")
                print(f"   {name:9s} {label:24s} {draws:7s} {exact:10.4f} {on.mean[0]:13.4f} {zd:+7.2f} {sim.score:7.4f} {za:+8.2f}  {ok}")
                sys.stdout.flush()


def _slim(p: Pairs) -> Pairs:
    """`p` with every game's trace dropped (`uno.experiments.transfer`'s
    `Played.pairs` pattern): a `FULL_UNO` pair set of games holds ~1 GB of
    `State`s in its traces, which only the step 9.5a divergence statistic
    (`compare(..., divergence=True)`) needs -- section 2a / 2b read only
    scores, already unaffected by slimming."""
    return replace(p, games=tuple((replace(a, trace=None), replace(b, trace=None)) for a, b in p.games))


def full_uno(games: int) -> None:
    """Collects `played` (2 draws x 2 columns x 5 rows = 20 `(row, column)` pairs)
    incrementally, one `(draws, cname)` group of 5 rows at a time, and slims each
    Pairs to scores-only (`_slim`) as soon as its traces are no longer needed:
    holding all 20 groups' full traces at once needs on the order of 20 GB (a
    `FULL_UNO` pair set is ~1 GB), which crashed a machine with ~17 GB of RAM once
    step 9.5a's divergence statistic (section 2c below) started actually reading
    them, though the traces were already being built and retained by `play_pairs`
    even before that, unused, by every earlier version of this function. Within one
    group, "heuristic" (collected first: `rows`' own iteration order) keeps its
    traces until every other row's comparison against it is computed and stored in
    `comparisons`, so at most two unslimmed Pairs (heuristic's, and whichever row is
    currently being compared) exist at once."""
    seeds = range(EVAL_SEEDS.start, EVAL_SEEDS.start + games)
    played: dict[tuple[str, str, str], Pairs] = {}
    comparisons: dict[tuple[str, str, str], Comparison] = {}  # (draws, other, cname) -> Comparison
    for draws in ("pile", "tickets"):
        rs = replace(rules.FULL_UNO, draws=draws)
        rnd, heu = PlayFirstRandom(rs), Heuristic(rs)
        col_heu = Heuristic(rs)
        col_heu.name = "heuristic_col"  # the column's own stream, so the Heuristic's variants can share the row's
        t1 = Heuristic(rs, low_threshold=1)
        t1.name, t1.stream = "heuristic_t1", "heuristic"
        rows = {
            "heuristic": heu,
            "heuristic_t1": t1,
            "heuristic_eps05": Mixture(heu, rnd, 0.05, name="heuristic_eps05", stream="heuristic"),
            "heuristic_eps05_own": Mixture(heu, rnd, 0.05, name="heuristic_eps05_own"),
            "random": rnd,
        }
        for cname, col in (("random", rnd), ("heuristic", col_heu)):
            heu_pairs: Pairs | None = None
            for rname, row in rows.items():
                p = play_pairs(rs, (row, col), seeds)
                if rname == "heuristic":
                    heu_pairs = p  # kept unslimmed until every other row of this group has been compared to it
                    continue
                comparisons[(draws, rname, cname)] = compare(heu_pairs, p)
                played[(draws, rname, cname)] = _slim(p)
                sys.stdout.flush()
            played[(draws, "heuristic", cname)] = _slim(heu_pairs)
            sys.stdout.flush()

    print(f"\n== 2a. FULL_UNO, {games} held-out seeds: per-seat score of the row agent, draws from the pile / tickets (z of the per-seed difference: both rules deal the same hands, so the two estimates are correlated)")
    for (draws, rname, cname), p in played.items():
        if draws != "pile":
            continue
        pt = played[("tickets", rname, cname)]
        mp, mt = estimate_mirrored(p), estimate_mirrored(pt)
        diffs = [estimate([a - b for a, b in zip(p.seat_scores(s), pt.seat_scores(s))]) for s in (0, 1)]
        zs = [d.mean / d.se if d.se > 0 else 0.0 for d in diffs]
        ok = check(all(abs(z) <= 3 for z in zs), f"pile and tickets agree on {rname} vs {cname}")
        print(f"   {rname + ' vs ' + cname:34s} seat 0 {mp.seat[0].mean:.4f} / {mt.seat[0].mean:.4f} (z {zs[0]:+.2f})  seat 1 {mp.seat[1].mean:.4f} / {mt.seat[1].mean:.4f} (z {zs[1]:+.2f})  {ok}")

    print("\n== 2b. mirrored evaluator (pile / tickets): correlation of the two seats, split share, variance ratio mirrored / raw")
    for (draws, rname, cname), p in played.items():
        if draws != "pile" or rname == cname:
            continue
        mp, mt = estimate_mirrored(p), estimate_mirrored(played[("tickets", rname, cname)])
        print(f"   {rname + ' vs ' + cname:34s} corr {mp.corr:+.3f} / {mt.corr:+.3f}  splits {mp.splits:.3f} / {mt.splits:.3f}  var ratio {mp.var_ratio:.3f} / {mt.var_ratio:.3f}")

    print(f"\n== 2c. paired comparisons, heuristic minus a variant against the same column: paired difference, var ratio paired / unpaired, deals for {100 * EDGE:.0f} pts at 3 sigma with 80 % power (paired / unpaired)")
    print("   step 9.5a: 'identical' is the share of deals whose two games play the exact same action sequence (common random numbers coupling them beyond the shared deal), 'diverge' the mean point (as a share of the shorter game's length) where the rest first differ")
    ratios: list[float] = []
    identicals: list[float] = []
    for cname in ("random", "heuristic"):
        for other in ("heuristic_t1", "heuristic_eps05", "heuristic_eps05_own", "random"):
            for draws in ("pile", "tickets"):
                c = comparisons[(draws, other, cname)]  # computed in full_uno, while both sides still had traces
                cells = [
                    f"seat {s}: {c.seat[s].mean:+.4f} ± {c.seat[s].half:.4f} ratio {c.var_ratio(s):.3f} deals {Z80**2 * c.seat[s].var / EDGE**2:6.0f} / {Z80**2 * c.unpaired[s].var / EDGE**2:6.0f} identical {c.identical[s]:.3f} diverge {c.diverge_frac[s]:.2f}"
                    for s in (0, 1)
                ]
                print(f"   vs {cname:9s} heuristic - {other:20s} {draws:7s} " + "   ".join(cells))
                for s in (0, 1):
                    ratios.append(c.var_ratio(s))
                    identicals.append(c.identical[s])
        sys.stdout.flush()

    def rank(xs: list[float]) -> list[float]:
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        out = [0.0] * len(xs)
        for r, i in enumerate(order):
            out[i] = float(r)
        return out

    rank_corr = correlation(rank(ratios), rank(identicals))
    print(f"\n   rank correlation of the variance ratio with the identical-games share, over the {len(ratios)} (comparison, seat) cells above: {rank_corr:+.3f}")


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    games = int(argv[0]) if len(argv) > 0 else len(EVAL_SEEDS)
    toy_seeds = int(argv[1]) if len(argv) > 1 else 20_000
    same_game(toy_seeds, games)
    full_uno(games)
    print(f"\n== {len(FAILS)} failed checks" + (": " + "; ".join(FAILS) if FAILS else ""))


if __name__ == "__main__":
    main()
