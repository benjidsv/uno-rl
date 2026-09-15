"""Step 9.5b: AIVAT with a fitted critic where no exact tree exists, and the transfer curve.

    uv run uno aivat_uno sweep [pairs=1000]
    uv run uno aivat_uno [games=4000] [fit_games=10000] [rulesets=TOY,TOY2,TOY2_SKIP,MID_12,MID_25,FULL_UNO]

`sweep`: the intermediate rulesets. Candidates are scaled-down full-UNO decks (seeded
reshuffle, free play, cap 1000). Heuristic vs Random, `pairs` DEV_SEEDS pairs: actions per game,
and the share of decisions offering a voluntary draw with a play in hand, an action card among
other plays, a wild to hold or spend, a choice of colour (`uno.eval.decisions.classify`).
Pre-declared rule: for each target length (12, 25 actions), among the candidates within +-20 %
of it whose four shares are all within 0.5-2x of FULL_UNO's, the one with the smallest largest
|log(share ratio)|. The picks are `MID_12` and `MID_25` in `uno.game.rules`.

Main run, per ruleset and matchup (heuristic vs random; heuristic_t1 vs random, a near-identical
row on the Heuristic's stream; heuristic vs heuristic_col):

1. a critic per profile (`uno.eval.critic`), fitted on FIT_SEEDS, with its holdout R^2;
2. AIVAT on `games` held-out pairs, streamed (no traces kept), per seat: raw and AIVAT means,
   the AIVAT - raw difference (z, a check), outcome R^2 over the visited decisions, the
   variance ratio AIVAT / raw with a 95 % bootstrap CI, deals for 3 points at 3 sigma with 80 %
   power, the pair (mirrored) variance ratio; per correction subset: chance, chance+row,
   chance+both (no deal correction: E over full-UNO deals is intractable);
3. mean zero of the chance corrections by event (voluntary, penalty, reshuffle draws) and of the
   action corrections (z, checks); the share of decisions skipped for want of a chance model;
4. on the toys: the same critic's exact moments on the tree (mean z, variance inside the 99.8 %
   bootstrap CI) and the exact ratio of the exact values without the deal correction;
5. the transfer curve: game length, critic R^2 and variance ratio, from the exact toys to
   FULL_UNO.
"""

from __future__ import annotations

import math
import sys
import time
from collections import Counter

import numpy as np

from uno.agents.baseline import Heuristic, PlayFirstRandom
from uno.eval.aivat import Corrections, GameTerms, bootstrap_ratio, bootstrap_var, estimate_aivat, stream_pairs
from uno.eval.critic import CriticValues, fit
from uno.eval.decisions import classify
from uno.eval.play import DEV_SEEDS, EVAL_SEEDS, FIT_SEEDS, play_from
from uno.game import rules
from uno.game.engine import legal_actions, new_game
from uno.game.observe import Level
from uno.game.rules import Ruleset

FAILS: list[str] = []
STOCHASTIC = [0]
Z80 = 3.0 + 0.8416
EDGE = 0.03
TOYS = ("TOY", "TOY_FULL", "TOY2", "TOY2_SKIP")
SHARES = ("draw with a play in hand", "action card among other plays", "wild: hold or spend", "which color: plays in 2+ colors")
TARGETS = (12, 25)
CORRECTIONS = {
    "chance": Corrections(True, False, False, False),
    "chance+row": Corrections(True, True, False, False),
    "chance+both": Corrections(True, True, True, False),
}


def _uno(colors, ranks, skip, reverse, draw_two, wild, wild4, hand) -> Ruleset:
    return Ruleset(colors=colors, rank_copies=ranks, skip_per_color=skip, reverse_per_color=reverse, draw_two_per_color=draw_two, wild_count=wild, wild_draw_four_count=wild4, hand_size=hand)


CANDIDATES = {
    "c3 r111 s1 d1 w1 f1 h4": _uno(3, (1, 1, 1), 1, 0, 1, 1, 1, 4),
    "c3 r1111 s1 d1 w2 f1 h4": _uno(3, (1, 1, 1, 1), 1, 0, 1, 2, 1, 4),
    "c4 r111 s1 d1 w1 f1 h4": _uno(4, (1, 1, 1), 1, 0, 1, 1, 1, 4),
    "c4 r122 s1 r1 d1 w1 f1 h4": _uno(4, (1, 2, 2), 1, 1, 1, 1, 1, 4),
    "c4 r1111 s1 r1 d1 w2 f2 h5": _uno(4, (1, 1, 1, 1), 1, 1, 1, 2, 2, 5),
    "c3 r1222 s1 r1 d1 w2 f2 h5": _uno(3, (1, 2, 2, 2), 1, 1, 1, 2, 2, 5),
    "c4 r11111 s1 r1 d1 w2 f2 h5": _uno(4, (1, 1, 1, 1, 1), 1, 1, 1, 2, 2, 5),
    "c4 r1222 s1 r1 d1 w2 f2 h5": _uno(4, (1, 2, 2, 2), 1, 1, 1, 2, 2, 5),
    "c4 r122222 s1 r1 d1 w2 f2 h7": _uno(4, (1, 2, 2, 2, 2, 2), 1, 1, 1, 2, 2, 7),
    "c4 r1222222 s2 r2 d2 w4 f4 h7": _uno(4, (1, 2, 2, 2, 2, 2, 2), 2, 2, 2, 4, 4, 7),
    # second round (the first had no candidate for 12 actions: every short one held too many
    # wilds per card, wild share 2.3-2.6x full UNO's); same rule, fewer wilds per card
    "c4 r11 s1 d1 w1 h3": _uno(4, (1, 1), 1, 0, 1, 1, 0, 3),
    "c4 r111 s1 d1 w1 h4": _uno(4, (1, 1, 1), 1, 0, 1, 1, 0, 4),
    "c4 r1111 s1 d1 w1 h4": _uno(4, (1, 1, 1, 1), 1, 0, 1, 1, 0, 4),
    "c3 r122 s1 r1 d1 w1 f1 h4": _uno(3, (1, 2, 2), 1, 1, 1, 1, 1, 4),
    "c4 r112 s1 r1 d1 w1 f1 h4": _uno(4, (1, 1, 2), 1, 1, 1, 1, 1, 4),
    "c4 r1111 s1 r1 d1 w1 f1 h4": _uno(4, (1, 1, 1, 1), 1, 1, 1, 1, 1, 4),
    "c4 r122 s1 r1 d1 w1 f1 h3": _uno(4, (1, 2, 2), 1, 1, 1, 1, 1, 3),
}


def check(ok: bool, what: str, stochastic: bool = True) -> str:
    STOCHASTIC[0] += stochastic
    if not ok:
        FAILS.append(what)
    return "ok" if ok else "XX"


def z_of(diff: float, se: float) -> float:
    if se < 1e-15:
        return 0.0 if abs(diff) <= 1e-9 else math.copysign(math.inf, diff)
    return diff / se


# -- 0. the intermediate rulesets --------------------------------------------------------


def mix(rs: Ruleset, pairs: int) -> tuple[float, dict[str, float]]:
    heu, rnd = Heuristic(rs), PlayFirstRandom(rs)
    games = turns = decisions = 0
    counts: Counter = Counter()
    for seed in DEV_SEEDS[:pairs]:
        try:
            s0 = new_game(rs, seed)
        except ValueError:
            continue
        for agents in ((heu, rnd), (rnd, heu)):
            t = play_from(s0, seed, agents)
            games += 1
            turns += t.final.turn
            decisions += len(t.steps)
            for st in t.steps:
                counts.update(classify(rs, st.obs, legal_actions(st.state)))
    return turns / games, {c: counts[c] / decisions for c in SHARES}


def sweep(pairs: int) -> None:
    print(f"== intermediate rulesets: heuristic vs random, {pairs} DEV_SEEDS pairs; actions per game and decision shares")
    rows = {"FULL_UNO": mix(rules.FULL_UNO, pairs)} | {name: mix(rs, pairs) for name, rs in CANDIDATES.items()}
    full = rows["FULL_UNO"][1]
    print(f"   {'ruleset':32s} {'deck':>4s} {'actions':>7s} " + " ".join(f"{c[:22]:>22s}" for c in SHARES))
    for name, (length, shares) in rows.items():
        rs = rules.FULL_UNO if name == "FULL_UNO" else CANDIDATES[name]
        print(f"   {name:32s} {rs.deck_size:4d} {length:7.1f} " + " ".join(f"{shares[c]:22.3f}" for c in SHARES))
    for target in TARGETS:
        ok = [n for n, (length, shares) in rows.items() if n != "FULL_UNO" and abs(length / target - 1) <= 0.2 and all(full[c] > 0 and 0.5 <= shares[c] / full[c] <= 2 for c in SHARES)]
        if ok:
            best = min(ok, key=lambda n: max(abs(math.log(rows[n][1][c] / full[c])) for c in SHARES))
            print(f"   target {target}: qualifying {ok}; pick {best}")
        else:
            best = min((n for n in rows if n != "FULL_UNO"), key=lambda n: abs(rows[n][0] - target))
            print(f"   target {target}: none qualifies; closest length {best} (does not meet the rule)")


# -- 1-4. critic, AIVAT, checks -------------------------------------------------------------


def ruleset(name: str) -> Ruleset:
    return getattr(rules, name)


def agents_for(rs: Ruleset) -> dict:
    heu, rnd = Heuristic(rs), PlayFirstRandom(rs)
    t1 = Heuristic(rs, low_threshold=1)
    t1.name, t1.stream = "heuristic_t1", "heuristic"
    col = Heuristic(rs)
    col.name = "heuristic_col"
    return {"heuristic": heu, "random": rnd, "heuristic_t1": t1, "heuristic_col": col}


MATCHUPS = (("heuristic", "random"), ("heuristic_t1", "random"), ("heuristic", "heuristic_col"))


def run_matchup(name: str, rs: Ruleset, agents: dict, row: str, col: str, games: int, fit_games: int, curve: list) -> None:
    t0 = time.perf_counter()
    critics = []
    for s in (0, 1):
        pair = (agents[row], agents[col]) if s == 0 else (agents[col], agents[row])
        critic, rep = fit(rs, pair, FIT_SEEDS[:fit_games])
        critics.append(critic)
        print(f"   {row} vs {col} seat {s}: critic on {rep.games:,} games / {rep.states:,} states, ridge {rep.lam:g}, holdout R^2 " + ", ".join(f"{k:g}: {v:.4f}" for k, v in rep.holdout_r2.items()))
    tree = nus = deal = None
    if name in TOYS:
        from uno import store
        from uno.compiled.aivat import arc_expectation, critic_values
        from uno.compiled.tree import compile_tree

        tree = compile_tree(rs, (Level.L1, Level.L2), True, cache=store.CACHE)
        nus = [critic_values(tree, c) for c in critics]
        deal = [float(arc_expectation(tree, nu)[0]) for nu in nus]
    models = tuple(CriticValues(critics[s], deal_mean=None if deal is None else deal[s]) for s in (0, 1))
    fit_seconds = time.perf_counter() - t0

    t1 = time.perf_counter()
    terms: tuple[list[GameTerms], list[GameTerms]] = ([], [])
    sq = [[0.0, 0.0, 0.0, 0] for _ in (0, 1)]  # per seat: sum (u - V)^2, sum u, sum u^2, visits
    for (g0, g1), (r0, r1) in stream_pairs(rs, (agents[row], agents[col]), EVAL_SEEDS[:games], models):
        for s, g, recs in ((0, g0, r0), (1, g1, r1)):
            terms[s].append(g)
            acc = sq[s]
            for r in recs:
                acc[0] += (g.u - r.node) ** 2
                acc[1] += g.u
                acc[2] += g.u * g.u
                acc[3] += 1
    play_seconds = time.perf_counter() - t1
    n = len(terms[0])
    decisions = sum(t.decisions for ts in terms for t in ts)
    skipped = sum(t.skipped_chance for ts in terms for t in ts)
    print(f"      {n} pairs in {play_seconds:.0f}s ({1000 * play_seconds / (2 * n):.1f} ms per game; fit {fit_seconds:.0f}s); {decisions / (2 * n):.1f} decisions per game; skipped chance corrections {skipped} ({100 * skipped / decisions:.3f} %)")
    r2s = []
    for s in (0, 1):
        e2, su, su2, k = sq[s]
        r2s.append(1.0 - e2 / (su2 - su * su / k))
    for label, c in CORRECTIONS.items():
        est = estimate_aivat(terms, c)
        for s in (0, 1):
            e, raw, d = est.seat[s], est.raw[s], est.diff[s]
            zd = z_of(d.mean, d.se)
            ok = check(abs(zd) <= 3, f"{name} {row} vs {col} {label} seat {s}: AIVAT - raw z {zd:+.2f}")
            lo, hi = bootstrap_ratio(est.scores[s], np.array([(1 if s == 0 else -1) * t.u / 2 + 0.5 for t in terms[s]]))
            print(
                f"      {label:12s} seat {s}: raw {raw}  AIVAT {e}  diff z {zd:+.2f} {ok}  ratio {est.var_ratio(s):.3f} [{lo:.3f}, {hi:.3f}]  "
                f"deals for 3 pts: raw {Z80**2 * raw.var / EDGE**2:5.0f}, AIVAT {Z80**2 * e.var / EDGE**2:5.0f}  R^2 {r2s[s]:.3f}"
            )
            if label == "chance+both":
                exact_ratio = None
                if tree is not None:
                    from uno.compiled.aivat import aivat_moments, exact_values
                    from uno.compiled.strategies import tabulate

                    pair = (agents[row], agents[col]) if s == 0 else (agents[col], agents[row])
                    prof = tuple((Level.L1, tabulate(tree, Level.L1, a)) for a in pair)
                    mean, var = aivat_moments(tree, *prof, nus[s], c, s).score(s)
                    raw_var = aivat_moments(tree, *prof, nus[s], Corrections(False, False, False, False), s).var / 4
                    ev = aivat_moments(tree, *prof, exact_values(tree, *prof), c, s).var / 4
                    zm = z_of(e.mean - mean, e.se)
                    blo, bhi = bootstrap_var(est.scores[s])
                    ok_m = check(abs(zm) <= 3, f"{name} {row} vs {col} seat {s}: critic mean vs exact z {zm:+.2f}")
                    ok_v = check(blo <= var <= bhi, f"{name} {row} vs {col} seat {s}: critic var {var:.5f} outside [{blo:.5f}, {bhi:.5f}]")
                    exact_ratio = (var / raw_var, ev / raw_var)
                    print(f"         exact (tree): mean {mean:.4f} (z {zm:+.2f}) {ok_m}  var {var:.5f} in CI [{blo:.5f}, {bhi:.5f}] {ok_v}  exact ratio {var / raw_var:.3f}; exact values, no deal: {ev / raw_var:.3f}")
                curve.append((name, f"{row} vs {col}", s, sum(t.decisions for t in terms[s]) / n, r2s[s], est.var_ratio(s), exact_ratio))
        print(f"      {label:12s} mirrored (pair) variance {est.mirrored.var:.5f}")
    print("      zero-mean corrections (chance+both), per game sums by event:")
    for s in (0, 1):
        events = sorted({k for t in terms[s] for k in t.chance_by} )
        cells = []
        for ev in events + ["action row", "action column"]:
            if ev == "action row":
                xs = [t.action[s] for t in terms[s]]
            elif ev == "action column":
                xs = [t.action[s ^ 1] for t in terms[s]]
            else:
                xs = [t.chance_by.get(ev, 0.0) for t in terms[s]]
            m = float(np.mean(xs))
            se = float(np.std(xs, ddof=1) / math.sqrt(len(xs)))
            z = z_of(m, se)
            cells.append(f"{ev} {m:+.4f} (z {z:+.2f}) {check(abs(z) <= 3, f'{name} {row} vs {col} seat {s} {ev}: mean z {z:+.2f}')}")
        print(f"         seat {s}: " + "  ".join(cells))
    sys.stdout.flush()


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "sweep":
        sweep(int(argv[1]) if len(argv) > 1 else 1000)
        return
    games = int(argv[0]) if len(argv) > 0 else len(EVAL_SEEDS)
    fit_games = int(argv[1]) if len(argv) > 1 else 10_000
    names = argv[2].split(",") if len(argv) > 2 else ["TOY", "TOY2", "TOY2_SKIP", "MID_12", "MID_25", "FULL_UNO"]
    curve: list = []
    for name in names:
        rs = ruleset(name)
        print(f"\n== {name} (deck {rs.deck_size}, hand {rs.hand_size}, reshuffle {rs.reshuffle}, forced play {rs.forced_play})")
        agents = agents_for(rs)
        for row, col in MATCHUPS:
            run_matchup(name, rs, agents, row, col, games, fit_games, curve)

    print("\n== transfer curve (chance+both corrections, no deal correction): decisions per game, critic outcome R^2, variance ratio AIVAT / raw; on the toys the exact ratio of the critic and of the exact values")
    print(f"   {'ruleset':10s} {'matchup':28s} {'seat':4s} {'decisions':>9s} {'R^2':>6s} {'ratio':>6s} {'exact critic':>12s} {'exact values':>12s}")
    for name, matchup, s, dec, r2v, ratio, exact_ratio in curve:
        ex = f"{exact_ratio[0]:12.3f} {exact_ratio[1]:12.3f}" if exact_ratio else f"{'':12s} {'':12s}"
        print(f"   {name:10s} {matchup:28s} {s:<4d} {dec:9.1f} {r2v:6.3f} {ratio:6.3f} {ex}")
    print(f"\n== {len(FAILS)} failed checks ({STOCHASTIC[0]} stochastic checks, ~{0.0027 * STOCHASTIC[0]:.1f} expected to fail by chance)" + (": " + "; ".join(FAILS) if FAILS else ""))


if __name__ == "__main__":
    main()
