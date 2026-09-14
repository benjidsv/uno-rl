"""Transfer check: how far `TOY` is from full UNO, before deciding to grow it.

    uv run python scripts/transfer.py [games=4000] [cfr_iterations=200]

Part 1, sampled, the same code on `TOY` and `FULL_UNO` over `games` held-out
pairs (uno.play.EVAL_SEEDS):

a. per-seat values of Random / Heuristic matchups, game length, endings;
b. luck: what mirroring buys on Heuristic vs Random, games for a 3-point edge;
c. skill proxy: the Heuristic's paired edge over Random, both against Random;
d. which skill-carrying decisions each ruleset actually offers (Heuristic vs
   Random games; a decision can fall in several categories).

Part 2, exact, `TOY` only (compiled tree): how games end under weak and strong
play, and through which ending each row agent collects its points, to see how
much of the toy's skill goes through exhaustion endings (the pile-parity
tactic of RULES.md section 9, which full UNO does not have). The Nash agent is
the CFR+ strategy cached by scripts/cfr.py (`.cache/cfr_TOY_L1_sym.npz`) if present,
otherwise `cfr_iterations` of CFR+ (not saved).
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from uno import rules
from uno.agents import Heuristic, PlayFirstRandom
from uno.cfr import CFRResult, cfr
from uno.compiled import END_KINDS, best_response, cache_name, compile_tree, evaluate, tabulate
from uno.engine import Phase, State, legal_actions
from uno.estimate import Pairs, compare, estimate_mirrored, games_needed, play_pairs
from uno.observe import Level
from uno.play import EVAL_SEEDS
from uno.rules import Kind, Play, Ruleset

CACHE = Path(__file__).resolve().parent.parent / ".cache"
EDGE = 0.03  # score points
RULESETS = ("TOY", "FULL_UNO")
CATEGORIES = (
    "any choice",
    "which color: plays in 2+ colors",
    "wild: hold or spend",
    "wild: pick a color",
    "action card among other plays",
    "draw with a play in hand",
    "play or keep the drawn card",
)


def classify(rs: Ruleset, s: State) -> set[str]:
    """Skill-carrying decision kinds at a decision node (see notes.md, "Which decisions carry skill")."""
    types = rs.card_types
    legal = legal_actions(s)
    plays = {rs.actions[a].card for a in legal if isinstance(rs.actions[a], Play)}
    cats = {"any choice"} if len(legal) > 1 else set()
    if s.phase is Phase.AFTER_DRAW:
        return cats | {"play or keep the drawn card"}
    colored = {c for c in plays if not types[c].is_wild}
    wild = len(plays) > len(colored)
    if len({types[c].color for c in colored}) > 1:
        cats.add("which color: plays in 2+ colors")
    if wild:
        cats.add("wild: pick a color")
        if colored:
            cats.add("wild: hold or spend")
    if len(plays) > 1 and any(types[c].kind not in (Kind.NUMBER, Kind.WILD) for c in plays):
        cats.add("action card among other plays")
    if plays and rs.draw_action in legal:
        cats.add("draw with a play in hand")
    return cats


@dataclass
class Played:
    pairs: Pairs  # traces dropped (a FULL_UNO pair set holds ~1 GB of States)
    ends: Counter  # end kind -> games, over the 2n games
    ties: int
    length: float  # play + draw actions per game
    decisions: Counter  # category -> decisions, over the 2n games
    n_decisions: int


def play(rs: Ruleset, agents, seeds, census: bool) -> Played:
    pairs = play_pairs(rs, agents, seeds)
    ends, decisions = Counter(), Counter()
    turns = n_dec = ties = 0
    for pair in pairs.games:
        for g in pair:
            ends[g.trace.final.end] += 1
            ties += g.score == 0.5
            turns += g.trace.final.turn
            if census:
                n_dec += len(g.trace.steps)
                for st in g.trace.steps:
                    decisions.update(classify(rs, st.state))
    slim = replace(pairs, games=tuple((replace(a, trace=None), replace(b, trace=None)) for a, b in pairs.games))
    return Played(slim, ends, ties, turns / (2 * pairs.n), decisions, n_dec)


def part1(games: int) -> None:
    seeds = range(EVAL_SEEDS.start, EVAL_SEEDS.start + games)
    results: dict[str, dict[str, Played]] = {}
    for name in RULESETS:
        rs = getattr(rules, name)
        rnd, heu = PlayFirstRandom(rs), Heuristic(rs)
        t0 = time.perf_counter()
        results[name] = {
            "random vs random": play(rs, (rnd, rnd), seeds, census=False),
            "heuristic vs random": play(rs, (heu, rnd), seeds, census=True),
            "heuristic vs heuristic": play(rs, (heu, heu), seeds, census=False),
        }
        print(f"   played {name}: {time.perf_counter() - t0:.0f}s", flush=True)

    print(f"\n== 1a. matchup values ({games} held-out pairs; score of the first-named ± 95 %)")
    print(f"   {'ruleset':9s} {'matchup':24s} {'seat 0':>16s} {'seat 1':>16s} {'actions/game':>12s} {'win end':>8s} {'exhaust':>8s} {'cap':>6s} {'ties':>6s}")
    for name, res in results.items():
        for label, p in res.items():
            m = estimate_mirrored(p.pairs)
            g = 2 * p.pairs.n
            print(
                f"   {name:9s} {label:24s} {str(m.seat[0]):>16s} {str(m.seat[1]):>16s} {p.length:12.1f} "
                f"{p.ends['win'] / g:8.3f} {p.ends['exhaustion'] / g:8.3f} {p.ends['cap'] / g:6.3f} {p.ties / g:6.3f}"
            )

    print(f"\n== 1b. luck: mirroring on heuristic vs random; games for a {100 * EDGE:.0f}-point edge at 3 sigma")
    print(f"   {'ruleset':9s} {'corr':>7s} {'splits':>7s} {'var ratio':>9s} {'seat 0 alone':>12s} {'seat 1 alone':>12s} {'mean raw':>9s} {'mirrored':>9s}")
    for name, res in results.items():
        m = estimate_mirrored(res["heuristic vs random"].pairs)
        raw, mir = m.var_per_game
        print(
            f"   {name:9s} {m.corr:7.3f} {m.splits:7.3f} {m.var_ratio:9.3f} {games_needed(EDGE, m.seat[0].var):12.0f} "
            f"{games_needed(EDGE, m.seat[1].var):12.0f} {games_needed(EDGE, raw):9.0f} {games_needed(EDGE, mir):9.0f}"
        )

    print("\n== 1c. skill proxy: heuristic - random, both against random, paired on the same deals")
    print(f"   {'ruleset':9s} {'seat':4s} {'paired diff':>17s} {'unpaired ±':>10s} {'corr':>7s} {'var ratio':>9s} {'games for 3 pts, unpaired / paired':>36s}")
    for name, res in results.items():
        c = compare(res["heuristic vs random"].pairs, res["random vs random"].pairs)
        for seat in (0, 1):
            d, u = c.seat[seat], c.unpaired[seat]
            g = f"{games_needed(EDGE, 2 * u.var):.0f} / {games_needed(EDGE, 2 * d.var):.0f}"
            print(f"   {name:9s} {seat:<4d} {str(d):>17s} {u.half:10.4f} {c.corr[seat]:+7.3f} {c.var_ratio(seat):9.3f} {g:>36s}")

    print("\n== 1d. decisions by kind (heuristic vs random games; per game = both players, share = of all decisions)")
    header = "".join(f" {name + ' /game':>15s} {'share':>6s}" for name in RULESETS)
    print(f"   {'decision kind':34s}{header}")
    rows = [("decisions", None)] + [(c, c) for c in CATEGORIES]
    for label, cat in rows:
        cells = ""
        for name in RULESETS:
            p = results[name]["heuristic vs random"]
            g = 2 * p.pairs.n
            count = p.n_decisions if cat is None else p.decisions[cat]
            cells += f" {count / g:15.2f} {count / p.n_decisions:6.3f}"
        print(f"   {label:34s}{cells}")
    sys.stdout.flush()


def part2(cfr_iterations: int) -> None:
    rs = rules.TOY
    print("\n== 2. TOY, exact: endings and where the points come from")
    tree = compile_tree(rs, cache=CACHE)
    k = tree.keyings[Level.L1]
    print("  ", tree, flush=True)
    rnd, heu = PlayFirstRandom(rs), Heuristic(rs)
    prof = {a.name: (Level.L1, tabulate(tree, Level.L1, a)) for a in (rnd, heu)}
    brs = [best_response(tree, Level.L1, prof["random"], seat) for seat in (0, 1)]
    prof["br_l1(random)"] = (Level.L1, np.where(k.info_player[k.ia_info] == 0, brs[0].sigma, brs[1].sigma))
    path = CACHE / f"cfr_{cache_name(rs, (Level.L1,), tree.symmetric)}.npz"
    res = CFRResult.load(path) if path.exists() else None
    if res is None or len(res.avg) != k.n_ia:
        res = cfr(tree, iterations=cfr_iterations)
        print(f"   no usable {path}: ran {res.iterations} CFR+ iterations ({res.seconds:.0f}s), not saved")
    print(f"   nash_l1 = CFR+ average strategy, {res.iterations} iterations, NashConv {100 * res.final.nashconv:.3f} pts", flush=True)
    prof["nash_l1"] = (Level.L1, res.avg)

    ends = tree.end_columns()  # win, exhaustion, cap
    u0 = tree.util[tree.terminals]

    def breakdown(row: str, col: str, seat: int) -> dict[str, float]:
        """Row agent at `seat`: P(each ending), and its score collected through each ending."""
        profile = (prof[row], prof[col]) if seat == 0 else (prof[col], prof[row])
        score = ((u0 if seat == 0 else -u0) + 1) / 2
        v, _ = tree.values(tree.profile_weights(*profile), np.hstack([ends, ends * score[:, None]]))
        p_win, p_exh, p_cap, s_win, s_exh, s_cap = v[0]
        total = s_win + s_exh + s_cap
        check = (evaluate(tree, *profile)[seat] + 1) / 2
        assert abs(total - check) < 1e-9 and abs(p_win + p_exh + p_cap - 1) < 1e-9
        return dict(score=total, p_exh=p_exh, s_win=s_win, s_exh=s_exh, s_exh_cond=s_exh / p_exh if p_exh else float("nan"))

    matchups = (
        ("random", "random"),
        ("heuristic", "random"),
        ("br_l1(random)", "random"),
        ("nash_l1", "random"),
        ("heuristic", "heuristic"),
        ("nash_l1", "heuristic"),
        ("nash_l1", "nash_l1"),
    )
    table = {(r, c, s): breakdown(r, c, s) for r, c in matchups for s in (0, 1)}
    print(f"\n   2a. row agent at the seat; 'from X' = score collected in games ending by X; 'within' = score given an exhaustion ending")
    print(f"   {'row vs column':28s} {'seat':4s} {'score':>7s} {'P(exhaust)':>10s} {'from emptied hand':>17s} {'from exhaustion':>15s} {'within exhaust':>14s}")
    for r, c in matchups:
        for s in (0, 1):
            b = table[(r, c, s)]
            print(f"   {r + ' vs ' + c if s == 0 else '':28s} {s:<4d} {b['score']:7.4f} {b['p_exh']:10.3f} {b['s_win']:17.4f} {b['s_exh']:15.4f} {b['s_exh_cond']:14.3f}")

    print("\n   2b. edge over Random at the same seat against Random, split by the ending the points arrive through (score points)")
    print(f"   {'row':16s} {'seat':4s} {'edge':>7s} {'via emptied hand':>16s} {'via exhaustion':>14s} {'exhaustion share':>16s}")
    for r in ("heuristic", "br_l1(random)", "nash_l1"):
        for s in (0, 1):
            b, base = table[(r, "random", s)], table[("random", "random", s)]
            edge, via_win, via_exh = (100 * (b[x] - base[x]) for x in ("score", "s_win", "s_exh"))
            print(f"   {r:16s} {s:<4d} {edge:7.2f} {via_win:16.2f} {via_exh:14.2f} {via_exh / edge:16.2f}")


def main() -> None:
    games = int(sys.argv[1]) if len(sys.argv) > 1 else len(EVAL_SEEDS)
    cfr_iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    assert END_KINDS[1:] == ("win", "exhaustion", "cap")
    part1(games)
    part2(cfr_iterations)


if __name__ == "__main__":
    main()
