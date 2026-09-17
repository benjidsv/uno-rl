"""Step 9.5d: the PIMC agent, exact on the toys and sampled on full UNO.

    uv run uno pimc exact  [rulesets=TOY,TOY2,TOY2_SKIP,TOY_FULL]
    uv run uno pimc pimcn  [rulesets=...] [realizations=64] [max_n=64]
    uv run uno pimc uno    [deals=400] [worlds=8] [depth=2] [workers] [cont=m|r]
    uv run uno pimc d2     [deals=200] [worlds=8] [depth=2] [workers]
    uv run uno pimc sweep  [deals=200] [workers] [configs=s2x2,s8x2,...]  (s<worlds>x<depth>[r][v|w], r<worlds>x<rollouts>, heu)
    uv run uno pimc belief [deals=1000] [worlds=32] [depth=2] [workers] [column=random|heuristic]
    uv run uno pimc calibration [deals=120] [samples=64]
    uv run uno pimc closure [ruleset=TOY] [sets=40] [samples=2000] [min_support=3] [opp=rulebot|passer]
    uv run uno pimc divergence [deals=300] [worlds=32] [depth=2] [workers] [column=random|heuristic]
    uv run uno pimc decay   [ruleset=TOY] [sets=20] [samples=4000] [min_support=3] [decays=0,0.02,...]

A trailing `r` on a search config (`s32x2r`), and the `cont=r` argument of `uno`, select step
9.5e's **response** continuation: each determinized world is solved as a best response to the
column's real policy instead of two-sided minimax. Worth +10.2 / +5.2 score points on full UNO at
w = 32, d = 2, and cheaper, because expectimax expands only the opponent's support.

A further `v` or `w` on a search config (`s32x2rv`) draws the opponent's hand from step 9.5f's
`voids` or `window` belief instead of the uniform split. It is a **separate axis from the
continuation letter**, so `sweep` groups its ladders by (continuation, belief) and never folds one
into the other: the groupings filter on `cont in ("m", "r")`, and a belief encoded there would
silently drop the rows it touched rather than compare them.

docs/spec.md section 25. Every check prints ok / XX; the last line counts the failures.

`pimcn`: PIMC with a finite number of sampled worlds, as a behavioural policy scored by one exact
`evaluate` (so the numbers carry no game-playing noise), against `exact`'s n -> infinity limit.

`uno`: the full-UNO PIMC row (search and rollout flavours) against Random and the Heuristic,
estimated with AIVAT using step 9.5c's value network as V. The network was fitted on rulebot
profiles, so it is off-profile here: AIVAT stays unbiased, it just reduces less. For a
deterministic agent the row correction is identically zero, so the gain comes from chance and the
opponent only.

`d2`: where a row's choices differ most from the search's preference, by decision kind. Selection
and valuation use disjoint halves of the worlds, and the control row draws its own; without both,
the winner's curse moves the totals by ~30 score points per game in either direction. Not a value
decomposition -- read the shares, not the totals.

`sweep`: raw score against Random as the world count and search depth grow. The toys' "n = 8
worlds is nearly the limit" does not transfer, because a toy belief holds 2-23 consistent worlds
and full UNO's holds astronomically more; this is where that shows up. Since step 9.5f the world
count can be climbed under a **belief** as well, which is what separates the two live explanations
for the ladder flattening past w = 32: an off-profile leaf network (9.5e's conclusion), or worlds
that were simply impossible (9.5f measured 79.9 % of them so, making w = 32 about six real worlds).

`belief`: step 9.5f's ladder over what the agent believes about the opponent's hand -- `prior` (the
uniform split, i.e. the 9.5e row unchanged), `voids` (the hard constraints a *voluntary draw*
implies, needing no policy *probabilities*, though not assumption-free -- see `closure`) and
`window` (plus the opponent's real action probabilities over their trailing run of plays, where
their hand is exactly recoverable) -- all at one budget, so the difference is the belief alone.
The particle filter over the deal that docs/plan.md originally specified **collapses**: a particle
survives the opponent's first play only if it holds the played card, ~7/98 on full UNO, so 64 of
them decay 0.156 -> 0.018 -> 0.002 -> 0.0004 -> 0 over their first five plays, and a version that
reset on collapse scored worse than the prior it replaced. These rungs therefore condition instead
of weighting. docs/spec.md section 26. The `column` argument picks the opponent the ladder is
scored against: step 9.5f ran only `random`, which is nearly uniform and so says little through its
play choices, while the exact toys put the belief axis at 0.0-0.6 points against Random and up to
2.6 against the Heuristic. `heuristic` is therefore where the soft `window` rung has to earn its
+29 % cost, having failed to beat `voids` against Random.

`calibration`: is the belief *correct*, or merely more confident? Over-pruning is the dangerous
failure of inference -- a belief that rules out the world the opponent is actually in looks sharp
and is wrong, and nothing downstream notices. Sharpness cannot detect that; calibration can. This
is the check that refuted the deal-time particle filter (Brier 0.167 against the uniform prior's
0.085, its "probability 1.000" bin observed at 0.112) while that design's *score* still looked
unremarkable, so it lives in the repo rather than in a scratch file.

`closure`: the toy closure test 9.5f went without -- the one place that step fell short of the
project's standard of checking a sampled method against an exact toy value before trusting it on
full UNO. The obstacle recorded then was that `window` is a *windowed* posterior with no exact
counterpart; but `voids` is the rung that ships, and it does have one. It is **not**
`uno.compiled.pimc`'s `"posterior"`: the constraint is a *necessary condition*, so what the rung
targets is the prior **restricted to the worlds the history does not rule out**. The `prior` rung
runs alongside as a control, and the distance it achieves is the sampling noise that sizes the
bound on `voids` -- with the gap between the two exact distributions printed next to both, because
without it a rung that ignored its constraints would pass whenever they happen not to bite.

`exact`: PIMC-infinity (`uno.compiled.pimc`) on the toy **world trees**, i.e. the n -> infinity
limit of "sample worlds consistent with what you know, solve each with perfect information,
play the best action on average", computed over the whole belief instead of a sample of it.
Four variants per (ruleset, opponent, seat): belief weighted by the opponent's reach
("posterior", PIMC *with* inference) or by the worlds' prior alone ("prior", no inference,
the only one that exists on full UNO), crossed with solving each world two-sided
("minimax", classic PIMC) or as a perfect-information best response to the fixed opponent
("response").

Reported against the exact ceilings already on the ladder (BR L1 / L2 / L3) and the
Heuristic's own score, plus each PIMC row's exploitability and `br_gap` (the step 9.5a D4
metric). Checks, all exact:

- `value <= BR L1` to 1e-9: PIMC-infinity is an L1 policy and BR L1 is the best of those, so a
  PIMC value above it would mean the policy is not one (the step 8.6 lesson, "a best response
  above the maximum is not a strategy");
- `believed` equals BR L3's value under "response" and the two-sided minimax value under
  "minimax": the per-arc perfect-information values are the ones those solvers produce;
- "prior" never has an unreached set (every information set of the tree contains at least one
  world, so the prior mass is always positive), while "posterior" inherits the best response's
  fill-in problem at the sets the opponent never allows (docs/spec.md section 16);
- every policy is orbit-uniform, which is what makes the quotient value the full game's
  (docs/spec.md section 18).

**Tie-breaking is value-relevant here, unlike a best response.** For a best response a tie
between action orbits means equal continuation value, so any tie-break scores the same. A PIMC
tie is a tie in the belief-weighted *perfect-information* score, and two actions can look
identical to that optimistic model while being worth different amounts in the real game. So
"PIMC" is a family indexed by its tie-break, not one policy: the quotient's orbit-uniform
choice and a full tree's pure per-set choice are genuinely different policies (measured on
`TOY`: they differ by 0.01-0.44 score points). The orbit-uniform one is canonical here because
it is the symmetric one; its quotient value was checked equal to its value lifted onto the full
`TOY` tree (16 of 16, `tests/compiled/test_pimc.py`).
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

import numpy as np

from uno import store
from uno.agents.baseline import Heuristic, KeepsDrawn, PlayFirstRandom
from uno.compiled.edge import br_gap
from uno.compiled.pimc import CONTINUATIONS, WEIGHTINGS, pimc_n, pimc_response, pimc_row
from uno.compiled.solvers import best_response, evaluate, two_sided
from uno.compiled.strategies import transfer
from uno.compiled.tree import compile_tree
from uno.eval.belief import Belief
from uno.eval.critic import CriticValues
from uno.eval.pimc import chunks, pmap
from uno.eval.search import PimcRollout, PimcSearch
from uno.eval.value_net import ValueNet
from uno.experiments import aivat_uno as au
from uno.game import rules
from uno.game.observe import Level

L1, L2, L3 = Level.L1, Level.L2, Level.L3
TOYS = ("TOY", "TOY2", "TOY2_SKIP", "TOY_FULL")
check = au.check


def score(v: float) -> float:
    return (v + 1.0) / 2.0


def nash_value(lazy) -> float | None:
    """The cached Nash seat-0 score on the lazy tree, if any (for exploitability in points)."""
    res = store.load_nash(lazy)
    if res is None:
        return None
    return score(evaluate(lazy, (L1, res.avg), (L1, res.avg))[0])


def exact_section(names: list[str]) -> None:
    for name in names:
        rs = getattr(rules, name)
        t0 = time.perf_counter()
        world = compile_tree(rs, (L1, L3), True, cache=store.CACHE)
        lazy = compile_tree(rs, (L1, L2), True, cache=store.CACHE)  # L2 for the BR L2 column
        ts = two_sided(world).value
        nash0 = nash_value(lazy)
        print(f"\n== {name} (deck {rs.deck_size}, hand {rs.hand_size}, forced play {rs.forced_play}); world roots {len(world.roots):,}, L1 sets {world.keyings[L1].n_info:,}")
        print(f"   two-sided minimax value {ts:+.6f} (score {score(ts):.4f}); Nash seat-0 score {'unknown' if nash0 is None else f'{nash0:.4f}'}  ({time.perf_counter() - t0:.1f}s to load)")

        for opp_agent in (PlayFirstRandom(rs), Heuristic(rs)):
            lazy_opp = (L1, store.load_or_tabulate(lazy, L1, opp_agent))
            opp = (L1, transfer(lazy, world, L1, lazy_opp[1]))
            heu = (L1, store.load_or_tabulate(lazy, L1, Heuristic(rs)))
            print(f"\n   vs {opp_agent.name} (score of the PIMC player; gap = BR L1 - PIMC, score points)")
            print(f"      {'seat':4s} {'variant':19s} {'PIMC':>7s} {'gap':>6s} {'BR L1':>7s} {'BR L2':>7s} {'BR L3':>7s} {'heur':>7s} {'believed':>8s} {'unreach':>8s} {'ties':>7s}")
            for seat in (0, 1):
                b1 = best_response(world, L1, opp, seat)
                b2 = best_response(lazy, L2, lazy_opp, seat)
                b3 = best_response(world, L3, opp, seat)
                hv = score(evaluate(lazy, *((heu, lazy_opp) if seat == 0 else (lazy_opp, heu)))[seat])
                for wt in WEIGHTINGS:
                    for co in CONTINUATIONS:
                        p = pimc_response(world, opp, seat, wt, co)
                        expected = b3.value if co == "response" else (ts if seat == 0 else -ts)
                        ok_le = check(p.value <= b1.value + 1e-9, f"{name} {opp_agent.name} seat {seat} {wt}/{co}: PIMC {p.score:.6f} above BR L1 {b1.score:.6f}", False)
                        ok_b = check(abs(p.believed - expected) <= 1e-9, f"{name} {opp_agent.name} seat {seat} {wt}/{co}: believed {p.believed:+.9f} != {expected:+.9f}", False)
                        ok_u = check(wt != "prior" or p.unreached == 0, f"{name} {opp_agent.name} seat {seat} prior/{co}: {p.unreached} unreached sets, expected none", False)
                        ok_s = check(world.keyings[L1].is_symmetric(p.sigma, 1e-12), f"{name} {opp_agent.name} seat {seat} {wt}/{co}: policy is not orbit-uniform", False)
                        print(
                            f"      {seat:<4d} {wt + '/' + co:19s} {p.score:7.4f} {100 * (b1.score - p.score):6.2f} {b1.score:7.4f} {b2.score:7.4f} {b3.score:7.4f} {hv:7.4f} "
                            f"{score(p.believed):8.4f}{ok_b} {p.unreached:8d}{ok_u} {p.ties:7d}  {ok_le}{ok_s}"
                        )
                sys.stdout.flush()

            print(f"      {'variant':19s} {'score s0/s1':>15s} {'exploitability s0/s1':>22s} {'br_gap s0/s1':>16s}")
            for wt in WEIGHTINGS:
                for co in CONTINUATIONS:
                    row, ps = pimc_row(world, opp, wt, co)
                    ex, gaps = [], []
                    for seat in (0, 1):
                        br = best_response(world, L1, row, seat ^ 1)
                        ex.append("n/a" if nash0 is None else f"{100 * (br.score - ((1 - nash0) if seat == 0 else nash0)):.2f}")
                        gaps.append(100 * br_gap(world, row, opp, seat) / 2)
                    print(f"      {wt + '/' + co:19s} {ps[0].score:7.4f}/{ps[1].score:.4f} {ex[0]:>10s} / {ex[1]:<10s} {gaps[0]:7.2f} /{gaps[1]:7.2f}")
            sys.stdout.flush()


def pimcn_section(names: list[str], realizations: int, max_n: int) -> None:
    grid = [n for n in (1, 2, 4, 8, 16, 32, 64) if n <= max_n]
    print(f"\n== PIMC-n: sampling n worlds per decision instead of the whole belief ({realizations} resamples per set to estimate the choice distribution,")
    print("   then one exact evaluate: the scores below carry no game-playing noise). posterior/response; gap = PIMC-infinity - PIMC-n, score points")
    for name in names:
        rs = getattr(rules, name)
        world = compile_tree(rs, (L1, L3), True, cache=store.CACHE)
        lazy = compile_tree(rs, (L1,), True, cache=store.CACHE)
        print(f"\n   {name} (L1 sets {world.keyings[L1].n_info:,})")
        for opp_agent in (PlayFirstRandom(rs), Heuristic(rs)):
            opp = (L1, transfer(lazy, world, L1, store.load_or_tabulate(lazy, L1, opp_agent)))
            for seat in (0, 1):
                limit = pimc_response(world, opp, seat, "posterior", "response")
                br = best_response(world, L1, opp, seat)
                cells, gaps = [], []
                for n in grid:
                    p = pimc_n(world, opp, seat, n, realizations, "posterior", "response", seed=1)
                    check(p.value <= br.value + 1e-9, f"pimcn {name} {opp_agent.name} seat {seat} n {n}: above BR L1", False)
                    gaps.append(limit.value - p.value)
                    cells.append(f"n={n}: {p.score:.4f} ({100 * gaps[-1] / 2:+.2f}, agr {p.agreement:.3f})")
                ok = check(gaps[-1] < gaps[0] + 1e-12, f"pimcn {name} {opp_agent.name} seat {seat}: more worlds did not close the gap ({gaps[0]:.4f} -> {gaps[-1]:.4f})", False)
                print(f"      vs {opp_agent.name:9s} seat {seat}: PIMC-inf {limit.score:.4f}, BR L1 {br.score:.4f} {ok}")
                print("         " + "  ".join(cells))
                sys.stdout.flush()


def net_for(rs, row: str, col: str, seat: int) -> ValueNet:
    """Step 9.5c's fitted value network for a profile, as AIVAT's V.

    The cached nets were fitted on *rulebot* profiles. AIVAT is unbiased for any V, so using the
    heuristic-vs-random net to evaluate a PIMC row is legitimate; it is only less effective, and
    the achieved ratio below says how much. Fitting one on PIMC games would cost 20,000 FIT games
    at ~0.3 s each (several hours) for a variance gain, not a correctness one.
    """
    path = Path(store.CACHE) / "value_net" / f"{rs_name(rs)}_heuristic_random_s{seat}_n2_g20000_w128_v1.pt"
    if not path.exists():
        raise SystemExit(f"no cached value network at {path}; run `uv run uno pimc_value aivat` first")
    return ValueNet.load(path, rs)


def rs_name(rs) -> str:
    return next(n for n, r in rules.NAMED.items() if r == rs)


def uno_section(games: int, worlds: int, depth: int, workers: int | None, cont: str = "m") -> None:
    rs = rules.FULL_UNO
    nets = [net_for(rs, "heuristic", "random", s) for s in (0, 1)]
    models = tuple(CriticValues(nets[s]) for s in (0, 1))
    leaves = (nets[0].leaf, nets[1].leaf)  # step 9.5e: 9.5d gave the agent nets[0] at both seats
    heu, rnd = Heuristic(rs), PlayFirstRandom(rs)

    def row_agent(row: str, col: str, column):
        """The row agent for this column. Under `cont == "r"` the PIMC row solves each world as a
        best response to the column, so it is a *different policy per column* and must carry a
        different name: a name identifies a policy (context.md section 3), and two agents sharing
        one would share an action rng stream."""
        if row == "pimc":
            return PimcSearch(rs, leaves, worlds=worlds, depth=depth, name=f"pimc_resp_{col}" if cont == "r" else "pimc", opponent=column if cont == "r" else None)
        if row == "pimc_rollout":
            return PimcRollout(rs, (heu, rnd), worlds=worlds, name="pimc_rollout")
        return heu

    rule = "each world solved as a best response to the column (9.5e)" if cont == "r" else "each world solved two-sided (classic PIMC)"
    print(f"\n== FULL_UNO PIMC row: {worlds} worlds per decision, depth {depth}, per-seat value network at the leaves; {rule}.")
    print(f"   {games} EVAL deals per matchup, both seatings; AIVAT's V is 9.5c's heuristic-vs-random network (off-profile, still unbiased).")
    curve: list = []
    for row in ("pimc", "pimc_rollout", "heuristic"):
        for col in ("random", "heuristic"):
            if row == col:
                continue  # self-play pairs are degenerate under common random numbers (context.md 5.4)
            column = rnd if col == "random" else Heuristic(rs)
            agents = {row: row_agent(row, col, column), col: column}
            print(f"\n   -- {row} vs {col}")
            t0 = time.perf_counter()
            au.evaluate_matchup("FULL_UNO", rs, agents, row, col, games, models, None, None, curve, 0.0, au.CORRECTIONS, row, workers)
            print(f"      matchup wall clock {time.perf_counter() - t0:.0f}s")
            sys.stdout.flush()


def _d2_chunk(rs, pair, seat: int, ref: PimcSearch, seeds) -> list[tuple[float, dict[str, float]]]:
    """Per game: the row's total loss against the search's own values, and that total split by
    decision kind. Only the row's own decisions with a real choice are scored."""
    from uno.eval.decisions import NODE_KINDS, node_kind
    from uno.eval.play import play_game
    from uno.game.engine import legal_actions

    out = []
    for seed in seeds:
        try:
            t = play_game(rs, seed, pair)
        except ValueError:
            continue
        total, by = 0.0, dict.fromkeys(NODE_KINDS, 0.0)
        for st in t.steps:
            if st.state.player != seat:
                continue
            legal = legal_actions(st.state)
            if len(legal) < 2:
                continue
            sel, val = ref.scores_split(st.obs, legal)
            best = max(legal, key=lambda a: (sel[a], -a))  # chosen on one half of the worlds...
            term = 50.0 * (val[best] - val[st.action])  # ...valued on the other, so no winner's curse
            total += term
            by[node_kind(rs, st.obs, legal)] += term
        out.append((total, by))
    return out


def d2_section(games: int, worlds: int, depth: int, workers: int | None) -> None:
    """Step 9.5a's D2 where no tree exists: the reference values come from the PIMC search.

    **What this is and is not.** The exact D2 (`uno.compiled.edge`) is a decomposition: the
    performance-difference lemma makes the per-decision terms of one *consistent* reference value
    function sum exactly to the value gap. A search Q is not a fixed policy's value function and
    the terms carry Monte Carlo noise, so nothing here telescopes: the totals below are **not** a
    value decomposition and must not be read as "the row is this many points worse". What the
    shares do say is where, by decision kind, the row's choices differ most from the search's
    preference -- a relative diagnostic.

    Three samples must stay independent, and getting this wrong is easy in both directions.
    Selection and valuation use disjoint halves of the world sample (`PimcSearch.scores_split`):
    without that, `max_a Qhat` overshoots at every decision and the per-game sum inflates by tens
    of points. The *control row* -- the search itself, whose total must sit near zero since it
    plays its own argmax -- additionally needs its **own** worlds, drawn from a different seed
    than the estimator's: sharing them lets the row's action be picked using the very half it is
    then valued on, which inflates `val[a_taken]` and drove the control to -32 points per game
    when this was first written. An ordinary row such as the Heuristic never sees the worlds, so
    its own term is free of both effects.
    """
    from uno.eval.decisions import NODE_KINDS
    from uno.eval.play import EVAL_SEEDS

    rs = rules.FULL_UNO
    nets = [net_for(rs, "heuristic", "random", s) for s in (0, 1)]
    heu, rnd = Heuristic(rs), PlayFirstRandom(rs)
    print(f"\n== FULL_UNO D2 with a search Q ({worlds} worlds, depth {depth}): where a row's choices differ most from the search's.")
    print(f"   {games} EVAL deals per seat, score points per game (mean +- se). Selection and valuation use disjoint halves of the worlds.")
    print("   Not a value decomposition (nothing telescopes): read the shares, not the totals. The search row is the control and should sit near zero.")
    for seat in (0, 1):
        ref = PimcSearch(rs, nets[seat].leaf, worlds=worlds, depth=depth, name="pimc", seed=0)
        # the control plays the same search but off an independent world sample (see the docstring)
        ctl = PimcSearch(rs, nets[seat].leaf, worlds=worlds, depth=depth, name="pimc_ctl", seed=9_999)
        for label, row_agent in (("heuristic", heu), ("pimc (control)", ctl)):
            pair = (row_agent, rnd) if seat == 0 else (rnd, row_agent)
            t0 = time.perf_counter()
            parts = pmap(_d2_chunk, [(rs, pair, seat, ref, c) for c in chunks(EVAL_SEEDS[:games], (os.cpu_count() or 1) * 8 if workers != 1 else 1)], workers)
            rowsg = [g for p in parts for g in p]
            totals = np.array([t for t, _ in rowsg])
            n = len(totals)
            se = totals.std(ddof=1) / math.sqrt(n)
            note = ""
            if label.startswith("pimc"):
                note = f"  {check(abs(totals.mean()) <= 3.0 * se + 2.0, f'd2 seat {seat}: control total {totals.mean():.2f} +- {se:.2f} is not near zero')}"
            print(f"\n   seat {seat}, row {label}: {n} games, {time.perf_counter() - t0:.0f}s; total {totals.mean():6.2f} +- {se:.2f} points per game{note}")
            # terms may be negative (the half-sample pick is not always the better one), so the
            # kinds do not partition a positive whole: a share is only printed when one exists.
            share = totals.sum() > 0
            print(f"      {'kind':10s} {'points/game':>12s} {'se':>6s} {'share' if share else '':>6s}")
            for kind in NODE_KINDS:
                xs = np.array([by[kind] for _, by in rowsg])
                if not np.any(xs):
                    continue
                pct = f"{100 * xs.sum() / totals.sum():5.1f} %" if share else ""
                print(f"      {kind:10s} {xs.mean():12.2f} {xs.std(ddof=1) / math.sqrt(n):6.2f} {pct:>7s}")
            sys.stdout.flush()


def _sweep_chunk(rs, pair, seeds) -> list[tuple[float, float]]:
    """The row's score at each seat, per deal, from the two seatings of one deal."""
    from uno.eval.play import play_from
    from uno.game.engine import new_game, returns

    out = []
    for seed in seeds:
        try:
            s0 = new_game(rs, seed)
        except ValueError:
            continue
        t0, t1 = play_from(s0, seed, pair), play_from(s0, seed, pair[::-1])
        out.append(((returns(t0.final)[0] + 1) / 2, (returns(t1.final)[1] + 1) / 2))
    return out


BELIEF_SUFFIX = {"p": "prior", "v": "voids", "w": "window"}


def _parse_config(rs, nets, heu, rnd, token: str):
    """`s32x2` = PimcSearch 32 worlds depth 2, each world solved two-sided; a trailing `r`
    (`s32x2r`) solves each world as a best response to the column instead (step 9.5e); a further
    `v` / `w` (`s32x2rv`) draws the opponent's hand from step 9.5f's `voids` / `window` belief
    instead of the uniform split; `r8x4` = PimcRollout 8 worlds 4 rollouts; `heu`.

    **The belief is a separate field of the meta tuple, never folded into the continuation
    letter.** The ladder groupings below filter on `cont in ("m", "r")`, so encoding a belief
    there would silently drop every row it touched instead of grouping it.
    """
    if token == "heu":
        return "Heuristic", heu, None
    kind, rest = token[0], token[1:]
    bel = "prior"
    if rest and rest[-1] in BELIEF_SUFFIX:
        bel, rest = BELIEF_SUFFIX[rest[-1]], rest[:-1]
    resp = rest.endswith("r")
    rest = rest[:-1] if resp else rest
    a, b = (int(x) for x in rest.split("x"))
    leaves = (nets[0].leaf, nets[1].leaf)  # step 9.5e: per seat, not nets[0] twice
    if kind == "s":
        # the sweep's column is always Random, so that is what a response config best-responds to
        # -- and what a `window` belief reads the action probabilities of.
        belief = None if bel == "prior" else Belief(rs, kind=bel, opponent=rnd)
        # A name identifies a policy (context.md section 3), so a belief needs its own. The prior
        # rows deliberately keep step 9.5e's bare names: same name, same world-sampling key, so at
        # the same deals they replay that step's games exactly and reproduce its numbers to the
        # digit -- the cross-check that this rewiring left the no-belief path alone.
        name = ("pimc_resp_random" if resp else "pimc") + ("" if bel == "prior" else f"_{bel}")
        label = f"PimcSearch w={a} d={b}" + (" resp" if resp else "") + ("" if bel == "prior" else f" {bel}")
        agent = PimcSearch(rs, leaves, worlds=a, depth=b, name=name, opponent=rnd if resp else None, belief=belief)
        return label, agent, ("search", a, b, "r" if resp else "m", bel)
    if kind == "r":
        if bel != "prior":
            raise SystemExit(f"bad sweep config {token!r}: the rollout player takes no belief (it never determinizes with one)")
        return f"PimcRollout w={a} r={b}", PimcRollout(rs, (heu, rnd), worlds=a, rollouts=b, name="pimc_rollout"), ("rollout", a, b, "m", "prior")
    raise SystemExit(f"bad sweep config {token!r}: s<worlds>x<depth>[r][v|w], r<worlds>x<rollouts>, or heu")


def sweep_section(games: int, workers: int | None, spec: str | None = None) -> None:
    """Is the full-UNO PIMC row weak because it is budget-starved, or because PIMC does not work
    here at all? Raw score against Random as the budget grows.

    The toys said n = 8 worlds recovers nearly all of PIMC-infinity, but a toy belief holds 2-23
    consistent worlds while full UNO's holds C(98, 5)-ish: the same n covers a vanishingly smaller
    share, so that conclusion is not expected to transfer. If more worlds and more depth do not
    move the score, the limit is PIMC itself (and the leaf value function), not the budget.
    """
    from uno.eval.play import EVAL_SEEDS

    rs = rules.FULL_UNO
    nets = [net_for(rs, "heuristic", "random", s) for s in (0, 1)]
    heu, rnd = Heuristic(rs), PlayFirstRandom(rs)
    tokens = (spec or "s2x2,s8x2,s32x2,s8x3,s32x3,r8x1,r32x1,r8x4,heu").split(",")
    seeds = EVAL_SEEDS[:games]
    k = (os.cpu_count() or 1) * 8 if workers != 1 else 1
    print(f"\n== FULL_UNO budget sweep: raw score against Random over {games} EVAL deals per config (both seatings).")
    print("   Every search config shares the world-sampling key, so at a shared information state they draw the")
    print("   *same* worlds. That couples the paired differences below only until the games diverge, which on a")
    print("   ~50-decision game is soon: measured at 3,000 deals under the response rule (step 9.5e), it removes")
    print("   11-19 % of the standard error on every axis -- worlds, depth and continuation alike, so 9.5d's")
    print("   '8 % for depth, 18-20 % for worlds' split does not survive a powered run. Never step 9's 77-85 %.")
    print(f"   {'row':34s} {'seat 0':>14s} {'seat 1':>14s} {'ms/game':>8s} {'sec':>6s}")
    scores: dict[str, np.ndarray] = {}
    meta: dict[str, tuple] = {}
    for token in tokens:
        label, agent, kind = _parse_config(rs, nets, heu, rnd, token)
        t0 = time.perf_counter()
        parts = pmap(_sweep_chunk, [(rs, (agent, rnd), c) for c in chunks(seeds, k)], workers)
        arr = np.array([g for p in parts for g in p])  # (deals, 2)
        secs = time.perf_counter() - t0
        scores[token], meta[token] = arr, (label, kind)
        cells = [f"{arr[:, s].mean():.4f}+-{arr[:, s].std(ddof=1) / math.sqrt(len(arr)):.4f}" for s in (0, 1)]
        print(f"   {label:34s} {cells[0]:>14s} {cells[1]:>14s} {1000 * secs / (2 * len(arr)):8.1f} {secs:6.0f}")
        sys.stdout.flush()

    # Depth, tested properly: the same deals and the same sampled worlds, so the difference is paired.
    # Comparing marginal means at 200 deals cannot see a 3-point effect (1 SE); this can.
    def paired_row(label: str, token: str, ref: str) -> None:
        cells = []
        for s in (0, 1):
            diff = scores[token][:, s] - scores[ref][:, s]
            se = diff.std(ddof=1) / math.sqrt(len(diff))
            cells.append(f"{100 * diff.mean():+6.2f}+-{100 * se:.2f} (z {diff.mean() / se if se else 0:+.1f})")
        print(f"      {label:28s} {cells[0]:>18s} {cells[1]:>18s}")

    search = {t: m[1] for t, m in meta.items() if m[1] and m[1][0] == "search"}
    seen = [b for b in ("prior", "voids", "window") if any(m[4] == b for m in search.values())]
    # The budget ladders are read *within* a continuation **and within a belief**: mixing either
    # would compare different agents, not different budgets. Step 9.5d's "flat past 32 worlds" and
    # "depth does not help" were both measured under the two-sided rule with the seat-0 network at
    # both seats, so under the response rule they are open questions again rather than settled ones
    # -- and step 9.5f reopened them a second time, since 79.9 % of the worlds those ladders were
    # counting were impossible (see the belief block at the end of this section).
    for cont, cname in (("m", "two-sided"), ("r", "response")):
        for bel in seen:
            tag = cname if bel == "prior" else f"{cname}, {bel}"
            base = {w: t for t, (_, w, d, c, b) in search.items() if d == 2 and c == cont and b == bel}
            deep = [(t, w, d) for t, (_, w, d, c, b) in search.items() if d > 2 and c == cont and b == bel]
            if deep and base:
                print(f"\n   [{tag}] paired effect of depth at equal worlds (deeper minus depth 2, score points; negative = deeper is worse).")
                print("   The two depths diverge at the first decision they disagree on, so 9.5d expected this pairing to be the")
                print("   weak one (~8 % of the SE); measured at 3,000 deals (9.5e) it removes 15-19 %, more than the worlds pairing.")
                print(f"      {'comparison':28s} {'seat 0':>18s} {'seat 1':>18s}")
                for t, w, d in sorted(deep, key=lambda x: (x[1], x[2])):
                    if w in base:
                        paired_row(f"w={w}: d={d} - d=2", t, base[w])
                sys.stdout.flush()
            if len(base) > 1:
                least = min(base)
                print(f"\n   [{tag}] paired effect of worlds at depth 2 (more worlds minus w={least}, score points; positive = more worlds is better).")
                print(f"   The larger config's first {least} worlds are exactly the smaller one's, so the two agree wherever those")
                print("   already decided the action. Measured at 3,000 deals (9.5e) this pairing removes ~14 % of the SE --")
                print("   *less* than the depth pairing, which reverses what 9.5d expected of the two axes.")
                print(f"      {'comparison':28s} {'seat 0':>18s} {'seat 1':>18s}")
                for w in sorted(base):
                    if w != least:
                        paired_row(f"d=2: w={w} - w={least}", base[w], base[least])
                # Successive doublings: "does the *next* doubling still pay". The rows above cannot
                # answer it -- they share a baseline, so the standard error of a difference between two
                # of them needs their covariance and is not recoverable from what they print. Step 9.5e
                # had to read "flat past 32" off unpaired per-config errors for exactly that reason.
                ws = sorted(base)
                if len(ws) > 2:
                    print(f"      {'-- each doubling against the previous one':28s}")
                    for a, b in zip(ws, ws[1:]):
                        paired_row(f"d=2: w={b} - w={a}", base[b], base[a])
                sys.stdout.flush()

    # Step 9.5e's headline: the continuation rule, at an otherwise identical budget and belief.
    for bel in seen:
        resp = {(w, d): t for t, (_, w, d, c, b) in search.items() if c == "r" and b == bel}
        mini = {(w, d): t for t, (_, w, d, c, b) in search.items() if c == "m" and b == bel}
        shared = sorted(set(resp) & set(mini))
        if shared:
            print(f"\n   [{bel}] paired effect of the continuation rule (response minus two-sided, score points; positive = responding is better).")
            print("   Exact on the toys: solving each world against the opponent's real policy instead of assuming a clairvoyant")
            print("   one is worth up to 8.8 points there, and most on free play, which full UNO is (09.5d section 1).")
            print("   The two share a world-sampling key only when they share a name, which they do not (a response agent is a")
            print("   different policy per column), so read this as unpaired-ish: the coupling here is the deal, not the worlds.")
            print(f"      {'comparison':28s} {'seat 0':>18s} {'seat 1':>18s}")
            for w, d in shared:
                paired_row(f"w={w} d={d}: resp - two-sided", resp[(w, d)], mini[(w, d)])
            sys.stdout.flush()

    # Step 9.5f's axis inside the sweep: the belief, at an otherwise identical budget.
    #
    # This is what separates the two live explanations for the worlds ladder flattening past
    # w = 32 (09.5f-results.md item 4). Step 9.5e promoted the off-profile leaf value network to
    # "the residual explanation" because the flattening survived fixing the continuation rule and
    # the seat networks. Step 9.5f then measured that the void constraint alone rules out 79.9 %
    # of `determinize_l1`'s worlds -- so w = 32 was carrying ~6 *possible* worlds, well inside the
    # steep part of that very ladder, and the ladder may have been flat because the extra worlds
    # were impossible rather than because the leaves cap what search can do. The two predict
    # different shapes for the `voids` ladder above: if it climbs past w = 32 where `prior` was
    # flat, world *quality* was binding; if it flattens in the same place, the leaf network stands.
    prior_at = {(w, d, c): t for t, (_, w, d, c, b) in search.items() if b == "prior"}
    for bel in seen:
        if bel == "prior":
            continue
        shared = sorted((w, d, c) for t, (_, w, d, c, b) in search.items() if b == bel and (w, d, c) in prior_at)
        if not shared:
            continue
        print(f"\n   paired effect of the `{bel}` belief at an equal budget ({bel} minus prior, score points; positive = inference helps).")
        print("   The two carry different names, so they do *not* share a world-sampling key: the coupling is the deal.")
        print("   Measured against Random at w = 32, d = 2 over 3,000 deals (step 9.5f): voids +2.83 / +4.40.")
        print(f"      {'comparison':28s} {'seat 0':>18s} {'seat 1':>18s}")
        for w, d, c in shared:
            t = next(t for t, m in search.items() if m[1:] == (w, d, c, bel))
            paired_row(f"w={w} d={d}{' resp' if c == 'r' else ''}: {bel} - prior", t, prior_at[(w, d, c)])
        sys.stdout.flush()


def _belief_chunk(rs, pair, seeds):
    """The row's score at each seat, per deal, from the two seatings of one deal."""
    from uno.eval.play import play_from
    from uno.game.engine import new_game, returns

    out = []
    for seed in seeds:
        try:
            s0 = new_game(rs, seed)
        except ValueError:
            continue
        t0 = play_from(s0, seed, pair)
        t1 = play_from(s0, seed, pair[::-1])
        out.append(((returns(t0.final)[0] + 1) / 2, (returns(t1.final)[1] + 1) / 2))
    return out


def belief_section(games: int, worlds: int, depth: int, workers: int | None, column: str = "random") -> None:
    """Step 9.5f: what the opponent's own actions are worth to the PIMC row.

    `determinize_l1` splits the unseen cards uniformly, which is `uno.compiled.pimc`'s `"prior"`
    weighting -- PIMC with no inference. Three rungs at an otherwise identical budget, so the
    difference is the belief and nothing else:

    - `prior`: the uniform split, i.e. the step 9.5e row unchanged;
    - `voids`: the hard constraints a *voluntary draw* implies (the drawer held nothing playable).
      No opponent policy, so unlike the `response` continuation this does not make the row
      opponent-specific -- the half that would survive into a general player;
    - `window`: additionally the opponent's real action probabilities over their trailing run of
      plays with no draw in between, where their hand is exactly recoverable.

    **Why not a particle filter over the deal**, which is what docs/plan.md's original design
    called for: it collapses. A particle survives the opponent's first play with probability
    `hand_size / unseen` ~ 7/98, and 64 particles decay 0.156 -> 0.018 -> 0.002 -> 0.0004 -> 0
    over their first five plays. A filter that resets on collapse scored *worse than the prior it
    replaced* (Brier 0.167 against 0.085, its 1.000 bin observed at 0.112). The rungs above
    condition instead of weighting, which is why they cannot degenerate.
    """
    from uno.eval.play import EVAL_SEEDS

    if column not in ("random", "heuristic"):
        raise SystemExit(f"unknown belief column {column!r}: random or heuristic")
    rs = rules.FULL_UNO
    nets = [net_for(rs, "heuristic", "random", s) for s in (0, 1)]
    leaves = (nets[0].leaf, nets[1].leaf)
    col = PlayFirstRandom(rs) if column == "random" else Heuristic(rs)
    seeds = EVAL_SEEDS[:games]
    k = (os.cpu_count() or 1) * 8 if workers != 1 else 1

    print(f"\n== FULL_UNO PIMC belief ladder vs {col.name}: w={worlds}, d={depth}, response continuation.")
    print(f"   {games} EVAL deals per rung, both seatings, paired on the same deals.")
    print(f"   {'belief':10s} {'seat 0':>16s} {'seat 1':>16s} {'ms/game':>9s}")
    scores: dict[str, np.ndarray] = {}
    beliefs: dict[str, object] = {}
    for label in ("prior", "voids", "window", "soft"):
        belief = None if label == "prior" else Belief(rs, kind=label, opponent=col if label == "window" else None)
        beliefs[label] = belief
        # A name identifies a policy (docs/context.md section 3). A response row is already a
        # different policy per column, so the column goes in the name beside the rung -- which
        # also keeps the vs-Random names byte-identical to step 9.5f's, so re-running that column
        # replays its games rather than merely resampling them.
        agent = PimcSearch(rs, leaves, worlds=worlds, depth=depth, name=f"pimc_resp_{column}_{label}", opponent=col, belief=belief)
        t0 = time.perf_counter()
        parts = pmap(_belief_chunk, [(rs, (agent, col), c) for c in chunks(seeds, k)], workers)
        arr = np.array([g for p in parts for g in p])
        secs = time.perf_counter() - t0
        scores[label] = arr
        cells = [f"{arr[:, s].mean():.4f}+-{arr[:, s].std(ddof=1) / math.sqrt(len(arr)):.4f}" for s in (0, 1)]
        print(f"   {label:10s} {cells[0]:>16s} {cells[1]:>16s} {1000 * secs / (2 * len(arr)):9.1f}")
        sys.stdout.flush()

    print("\n   paired against the prior (score points; positive = inference helps).")
    print("   The three rungs carry different names, so they do *not* share a world-sampling key:")
    print("   the coupling here is the deal, as it is for the continuation rule (step 9.5e).")
    print(f"      {'comparison':24s} {'seat 0':>20s} {'seat 1':>20s}")

    def diff_row(lo: str, hi: str) -> list[float]:
        cells, zs = [], []
        for s in (0, 1):
            d = scores[hi][:, s] - scores[lo][:, s]
            se = d.std(ddof=1) / math.sqrt(len(d))
            zs.append(au.z_of(d.mean(), se))
            cells.append(f"{100 * d.mean():+6.2f}+-{100 * se:.2f} (z {zs[-1]:+.1f})")
        print(f"      {f'{hi} - {lo}':24s} {cells[0]:>20s} {cells[1]:>20s}")
        return zs

    zs = {label: diff_row("prior", label) for label in ("voids", "window", "soft")}
    diff_row("voids", "window")
    diff_row("voids", "soft")  # `soft` believes the void where `voids` enforces it: what that costs

    # Real checks. Step 9.5f shipped this section with none, so its "0 failed checks" line was
    # vacuous (09.5f-results.md item 5). Both are sized to catch a *broken* belief, not to police
    # a close call: a threshold sitting just above the observed value is a test that fails on a
    # different seed range by construction (docs/context.md section 7).
    sizes = {label: len(a) for label, a in scores.items()}
    check(len(set(sizes.values())) == 1, f"belief: the rungs did not score the same deals ({sizes}), so the paired differences above are not paired")
    for label, z in zs.items():
        check(
            min(z) > -3.0,
            f"belief: {label} is {min(z):.1f} sigma *below* the uniform prior at a seat -- a belief that scores worse than no belief at all is over-pruning, not inference (the refuted particle filter's signature)",
        )

    # The fallback rate says whether the belief was *available*; a report that hides it hides
    # whether the rung ran at all. The counters live on the agent, which `pmap` pickles into each
    # worker, so they only come back when everything ran in this process.
    if workers == 1:
        for label in ("voids", "window", "soft"):
            b = beliefs[label]
            print(f"   {label:7s}: fallbacks {b.fallbacks:,}/{b.draws:,} draws ({100 * b.fallbacks / max(b.draws, 1):.1f} %)")
    else:
        print("   (fallback rates need workers=1: the counters do not cross processes)")
    sys.stdout.flush()


def calibration_section(games: int, samples: int) -> None:
    """Is the belief *correct*, or only more confident? (step 9.5f, docs/spec.md section 26.5)

    Over-pruning is the dangerous failure of any inference: a belief that rules out the world the
    opponent is actually in looks sharp and is wrong, and nothing downstream notices -- the search
    simply explores a set of worlds that excludes the truth. For every card type at every row
    decision each belief states P(the opponent holds at least one copy), and the played state says
    whether they do.

    - the **prior** is exact (hypergeometric), so it is the floor no belief may fall below;
    - a correct belief is **sharper** (lower Brier) *and* **calibrated** (predicted ~ observed);
    - an over-pruning one is sharper and mis-calibrated: it says 0.95 and is right 0.7 of the time.

    Two pre-declared checks, both sized against the design this refuted rather than against the
    numbers that passed: a rung may not be materially worse than no belief at all (the particle
    filter was 2x the prior's Brier), and no populated reliability bin may miss by more than 0.25
    (it missed by 0.89). The bound is deliberately loose: its job is to catch catastrophic
    miscalibration, not to police the gap between two already-calibrated rungs, and a threshold
    sitting just above the observed value is a test that fails on a different seed range by
    construction (docs/context.md section 7, first entry).

    Measured on the shipped rungs at the defaults: Brier prior 0.0846, `voids` 0.0828, `window`
    0.0825; worst reliability-bin gap 0.088 / 0.093 / 0.135. Note `window` is the **least**
    calibrated of the three while being marginally the sharpest, which is what a *windowed*
    posterior should look like -- it conditions on a truncated history, so it is slightly biased.
    That is a second mark against it, beside failing to beat `voids` on score.

    Runs single-process, which is what lets it report the fallback rates `belief_section` cannot
    (those counters do not cross `pmap`'s process boundary). DEV seeds: a diagnostic, never a
    reported number.
    """
    import random
    from collections import Counter

    from uno.eval.belief import Belief
    from uno.eval.play import DEV_SEEDS, play_game, stable_hash
    from uno.eval.search import unseen_cards

    rs = rules.FULL_UNO
    heu, rnd = Heuristic(rs), PlayFirstRandom(rs)
    seat = 0
    kinds = ("prior", "voids", "window", "soft")
    beliefs = {"voids": Belief(rs, kind="voids"), "window": Belief(rs, kind="window", opponent=rnd), "soft": Belief(rs, kind="soft")}
    n = dict.fromkeys(kinds, 0)
    brier = dict.fromkeys(kinds, 0.0)
    bins = {k: [[0, 0.0, 0.0] for _ in range(10)] for k in kinds}
    decisions = 0

    def add(kind: str, p: float, actual: float) -> None:
        p = min(max(p, 0.0), 1.0)
        n[kind] += 1
        brier[kind] += (p - actual) ** 2
        b = bins[kind][min(int(p * 10), 9)]
        b[0] += 1
        b[1] += p
        b[2] += actual

    print("\n== FULL_UNO belief calibration: is the belief correct, or only more confident?")
    print(f"   {games} DEV deals, {samples} worlds per decision, belief over Random's hand at seat 1.")
    t0 = time.perf_counter()
    for seed in list(DEV_SEEDS)[:games]:
        try:
            t = play_game(rs, seed, (heu, rnd))
        except ValueError:
            continue
        for st in t.steps:
            if st.state.player != seat:
                continue
            obs = st.obs
            truth = set(st.state.hands[seat ^ 1])
            try:
                unseen, _ = unseen_cards(rs, obs)
            except ValueError:
                continue
            k, u = obs.opp_hand_size, len(unseen)
            if not u or not k:
                continue
            decisions += 1
            mult = Counter(unseen)
            for t_id, m in mult.items():
                add("prior", 1.0 - math.comb(u - m, k) / math.comb(u, k), 1.0 if t_id in truth else 0.0)
            for kind, b in beliefs.items():
                rng = random.Random(stable_hash((seed, "calibration", st.state.turn)))
                held: Counter = Counter()
                for _ in range(samples):
                    held.update(set(b.hand(obs, unseen, rng)))
                for t_id in mult:
                    add(kind, held[t_id] / samples, 1.0 if t_id in truth else 0.0)

    print(f"   {decisions:,} row decisions, {n['prior']:,} (card type, decision) pairs each, {time.perf_counter() - t0:.0f}s")
    print(f"\n   {'belief':8s} {'Brier':>7s} {'vs prior':>9s} {'worst bin':>10s}  reliability, bins with n >= 200 (predicted -> observed)")
    for kind in kinds:
        cells, worst = [], 0.0
        for b in bins[kind]:
            if b[0] >= 200:
                pred, seen = b[1] / b[0], b[2] / b[0]
                worst = max(worst, abs(pred - seen))
                cells.append(f"{pred:.2f}->{seen:.2f}")
        rel = brier[kind] / brier["prior"] if brier["prior"] else float("nan")
        ok = ""
        if kind != "prior":
            # sized against the refuted design (2x the prior), not against the margin that passed
            ok += check(brier[kind] <= brier["prior"] * 1.05, f"calibration: {kind} Brier {brier[kind] / max(n[kind], 1):.4f} is worse than the prior's {brier['prior'] / max(n['prior'], 1):.4f}")
        ok += check(worst <= 0.25, f"calibration: {kind} worst reliability-bin gap {worst:.2f} > 0.25 -- over-pruning?")
        print(f"   {kind:8s} {brier[kind] / max(n[kind], 1):7.4f} {rel:9.3f} {worst:10.3f}  {'  '.join(cells)} {ok}")
    for kind, b in beliefs.items():
        print(f"   {kind:8s} fallbacks {b.fallbacks:,}/{b.draws:,} draws ({100 * b.fallbacks / max(b.draws, 1):.1f} %)")
    sys.stdout.flush()


def closure_section(name: str, sets: int, samples: int, min_support: int = 3, opponent: str = "rulebot", decays: tuple[float, ...] = (0.1,)) -> dict:
    """Step 9.5f item 2: close the *sampled* belief against the exact toys.

    Every other sampled method in 9.5b-9.5e was checked against an exact toy value before being
    trusted on full UNO, and the belief was not. The obstacle recorded at the time was that
    `window` is a *windowed* posterior with no exact counterpart, so there is no equality to
    assert -- but that is an argument about `window`, and `voids` is the rung that ships. `voids`
    does have an exact counterpart, and it is not `uno.compiled.pimc`'s `"posterior"`: the
    constraint is a **necessary condition**, so what the rung targets is the prior *restricted to
    the worlds the history does not rule out*. That object is computable on a toy world tree.

    **The three-way test.** Comparing a sampler against a distribution needs a noise scale, or the
    bound is arbitrary. So the `prior` rung is run as a **control**: it must reproduce the exact
    unrestricted belief, and the total-variation distance it achieves *is* the sampling noise at
    this many draws. `voids` must then match its own exact target to the same scale. The third
    number is the **separation** -- how far the two exact distributions are apart -- and without it
    the whole test is vacuous, since a rung that ignored its constraints entirely would pass the
    first two whenever the constraints happen not to bite (the step 9.5e lesson).

    **The full tree, not the quotient.** Member worlds of a set on a symmetric tree are each
    canonicalised by their *own* group element, so an opponent hand read off a member's L3 key
    would be in that member's coordinates and not the observation's -- the trap `uno.compiled.pimc`
    documents for arcs, one field over. `TOY` is compiled with `symmetric=False` here, where a
    member's key is a raw key of the real game.

    Several worlds give the opponent the same hand and differ only in the pile order, so the
    belief over *hands* is a marginal: it is summed over members, never read off one.
    """
    import random
    from collections import Counter

    from uno.compiled.pimc import _members
    from uno.compiled.pimc import belief as exact_belief
    from uno.eval.belief import Belief, satisfies, voids
    from uno.eval.play import stable_hash
    from uno.eval.search import unseen_cards
    from uno.game.keys import decode_key
    from uno.game.observe import observation_from_key

    if opponent not in ("rulebot", "passer"):
        raise SystemExit(f"unknown closure opponent {opponent!r}: rulebot or passer")
    rs = getattr(rules, name)
    opp_agent = PlayFirstRandom(rs) if opponent == "rulebot" else KeepsDrawn(rs)
    print(f"\n== {name} belief closure vs {opp_agent.name}: the sampled rungs against the exact belief on the full world tree.")
    t0 = time.perf_counter()
    tree = compile_tree(rs, (L1, L3), False, cache=store.CACHE)
    k1, k3 = tree.keyings[L1], tree.keyings[L3]
    inv3 = {i: kb for kb, i in k3.keys.items()}
    inv1 = {i: kb for kb, i in k1.keys.items()}
    print(f"   {len(tree.roots):,} worlds, {k1.n_info:,} L1 sets, loaded in {time.perf_counter() - t0:.1f}s; comparing sets with >= {min_support} consistent hands")

    def tv(p: dict, q: dict) -> float:
        return 0.5 * sum(abs(p.get(h, 0.0) - q.get(h, 0.0)) for h in set(p) | set(q))

    # The world prior, which is what `uno.compiled.pimc.belief(..., "prior")` reduces to: with both
    # players' weights identically one, `reach(...)[:, 2]` is the chance column alone. Computed
    # directly so the full tree never needs a tabulated opponent.
    prior = tree.reach(np.ones(tree.m), np.ones(tree.m))[:, 2]
    # The target for `voids` is **not** the policy-free prior. The rung encodes an assumption about
    # the opponent -- that a playable drawn card is played at once -- so the exact object it moves
    # towards is the *posterior* given such an opponent: `uno.compiled.pimc`'s `"posterior"`
    # weighting, the prior times the opponent's reach, which is what `uno.eval.belief` says the
    # module is for. Scored against the prior instead, the test measures the assumption rather than
    # the implementation and reports a correct rung as broken (it did, at 0.23 against 0.13).
    #
    # Which opponent is therefore the question, not a detail. Against `rulebot` the assumption is
    # true and the rung should be near-exact; against `passer` (`KeepsDrawn`, which keeps a
    # playable drawn card half the time) it is false, and this is where a hard filter has to pay
    # for it while `soft` -- the same constraint believed rather than enforced -- should not.
    opp_strategy = (L1, store.load_or_tabulate(tree, L1, opp_agent))
    # `prior`, two independent draws of `voids` (the noise scale), then one `soft` per decay. The
    # decay is ignored by the first three rungs; it is passed only because `Belief` takes one.
    rungs = [("prior", "a", 0.1), ("voids", "a", 0.1), ("voids", "b", 0.1)] + [("soft", f"{d:g}", d) for d in decays]
    results: dict = {}
    for seat in (0, 1):
        post = exact_belief(tree, opp_strategy, seat, "posterior")
        nodes, w, counts, starts = _members(tree, k1, seat, prior)
        live = [i for i in range(k1.n_info) if k1.info_player[i] == seat and counts[i] > 1]
        rng0 = random.Random(0)
        rng0.shuffle(live)
        rows, empty, impossible = [], 0, 0
        for i in live:
            kb = inv1[i]
            obs = observation_from_key(rs, L1, decode_key(L1, kb))
            vs = voids(rs, obs.public, seat ^ 1)
            if not vs:
                continue
            exact_p: dict = {}
            exact_q: dict = {}
            for j in range(starts[i], starts[i + 1]):
                hand = tuple(sorted(decode_key(L3, inv3[k3.node_info[nodes[j]]])[0][seat ^ 1]))
                exact_p[hand] = exact_p.get(hand, 0.0) + float(w[j])
                exact_q[hand] = exact_q.get(hand, 0.0) + float(post[nodes[j]])
            tot, tq = sum(exact_p.values()), sum(exact_q.values())
            if tq <= 0.0:
                continue  # the opponent never allows this set: there is no posterior to close against
            exact_p = {h: m / tot for h, m in exact_p.items()}
            exact_q = {h: m / tq for h, m in exact_q.items()}
            kept = {h: m for h, m in exact_p.items() if satisfies(rs, h, vs)}
            # Over-pruning, and the thing this test exists to find: the rung has ruled out every
            # world the observation allows, so the truth is certainly among the excluded. The set
            # **stays** in the comparison. Skipping it (as this did until the passer column was
            # added) would average `voids` over only the sets where its assumption happens to hold
            # -- the flattering half, and precisely the sets the passer column exists to reach.
            # Against an honest opponent it changes nothing: there the sets `voids` empties are
            # exactly those the posterior already zeroes, and those are skipped above.
            over_pruned = not kept
            s = sum(kept.values())
            exact_v = {h: m / s for h, m in kept.items()} if kept else {}
            # Most `TOY` sets admit one or two opponent hands, where every sampler agrees and the
            # comparison says nothing -- at `min_support = 1` the measured separation came out at
            # 1.6x the sampling noise and the vacuity guard below fired, correctly. Compare only
            # where there is something to discriminate.
            if len(exact_p) < min_support:
                continue
            try:
                unseen, _ = unseen_cards(rs, obs)
            except ValueError:
                continue
            emp = {}
            for kind, tag, decay in rungs:
                bel = Belief(rs, kind=kind, decay=decay)
                # Every `soft` rung draws from **one** stream, so the decays see an identical
                # sequence of candidate hands and differ only in how they weight it: the sweep
                # over decays is paired, where `voids|a` against `voids|b` deliberately is not
                # (that pair is the noise scale, and needs two independent draws).
                rng = random.Random(stable_hash((name, seat, i, kind, "shared" if kind == "soft" else tag)))
                c = Counter(bel.hand(obs, unseen, rng) for _ in range(samples))
                emp[f"{kind}|{tag}"] = {h: n / samples for h, n in c.items()}
            # Not an invariant, and it was wrongly written as one. The uniform split is not a
            # *sound* sampler: it proposes hands that no consistent world holds, because it
            # conditions on the hand sizes and the discard but never on the opponent's draw
            # *events*. Measured on `TOY` over a uniform sample of constrained sets, that happens
            # at 5 % (seat 0) and 16 % (seat 1) of them, and on every one of those the split does
            # walk into it. As an `assert` this would have killed the run the first time it drew
            # such a set; as a statistic it is the sound pruning `voids` does not do.
            if set(emp["prior|a"]) - set(exact_p):
                impossible += 1
            # The truth is `exact_p`: the tree's own conditional over the opponent's hand, which is
            # the prior over the *worlds the set actually contains*. Both rungs are scored against
            # it, so "does the belief help" is TV(voids) < TV(prior) -- a comparison, not a bound.
            #
            # The noise scale is two independent draws of the *same* rung, not the prior rung: an
            # earlier version used `prior` as the control and that is mis-specified, because the
            # distance from the uniform split to the truth is precisely what `voids` exists to
            # close. Using it as "noise" would have compared the rung against its own target.
            empty += over_pruned
            base = (tv(emp["prior|a"], exact_q), tv(emp["voids|a"], exact_q), tv(emp["voids|a"], emp["voids|b"]), tv(exact_p, exact_q))
            rows.append(base + tuple(tv(emp[f"soft|{d:g}"], exact_q) for d in decays) + (len(exact_p), len(exact_v)))
            if len(rows) >= sets:
                break
        # The severe failure first: a set where the rung kept *nothing*. The opponent's real hand
        # is one of the worlds the set contains, so ruling out all of them rules out the truth --
        # over-pruning, which is exactly what calibration alone would not necessarily catch.
        # Against the opponent it assumes this must be 0: the sets `voids` empties there are the
        # ones the posterior already zeroes, so an empty one would mean the rung contradicts the
        # very opponent it models. Against the passer it is the **finding** rather than a failure
        # -- that column is where the assumption is false, so emptying sets is what a hard filter
        # is expected to do, and the point is what it costs in TV against `soft`.
        honest = opponent == "rulebot"
        over = check(
            empty == 0 or not honest,
            f"closure {name} seat {seat}: `voids` excluded every consistent world at {empty} set(s): over-pruning, the opponent's real hand among them",
        )
        if empty:
            tail = over if honest else "(expected here: the assumption is false against this column)"
            print(f"   seat {seat}: **`voids` kept nothing at {empty} of the sets compared** {tail}")
        if not rows:
            print(f"   seat {seat}: no constrained set survived to be compared")
            continue
        nd = len(decays)
        a = np.array([r[: 4 + nd] for r in rows])
        d_prior, d_voids, noise, sep = (a[:, j].mean() for j in range(4))
        d_soft_by_decay = {d: float(a[:, 4 + j].mean()) for j, d in enumerate(decays)}
        # The checks below read a single `soft`: the shipped default where it was swept, else the
        # best of the sweep, so a sweep that omits 0.1 still states its claim about some rung.
        d_soft = d_soft_by_decay.get(0.1, min(d_soft_by_decay.values()))
        sup = np.array([r[4 + nd :] for r in rows]).mean(axis=0)
        print(f"   seat {seat}: {len(rows)} constrained sets compared, {samples} draws each, support {sup[0]:.1f} -> {sup[1]:.1f} hands")
        print(f"      TV(uniform split, exact posterior)   {d_prior:.4f}   <- the belief PIMC held before 9.5f")
        print(f"      TV(voids, exact posterior)           {d_voids:.4f}   <- the belief it holds now")
        for d in decays:
            tail = "  (decay 0 is `voids`)" if d == 0.0 else ("  (decay 1 is the prior)" if d == 1.0 else "")
            print(f"      TV(soft decay={d:<5g}, exact posterior) {d_soft_by_decay[d]:.4f}   <- believed, not enforced{tail}")
        print(f"      TV(exact prior, exact posterior)     {sep:.4f}   <- the distance the rung is trying to close")
        print(f"      sampling noise, two draws of voids   {noise:.4f}")
        print(f"      uniform split proposed an impossible hand: {impossible}/{len(rows)} of the sets compared")
        print("         (`min_support` selects against these -- more hands means less chance any is impossible;")
        print("          over a uniform sample of constrained TOY sets the rate is 5 % at seat 0, 16 % at seat 1)")
        # What is claimed depends on the column, because the rung's assumption does. Against the
        # opponent it models, `voids` must move the belief towards the posterior -- the one thing
        # it claims to do. Against `passer`, whose behaviour it gets wrong, that claim is *not*
        # expected to hold and its failure is the measurement, not a defect; what is claimed there
        # is that believing the void beats enforcing it, which is the case `soft` was added for.
        if honest:
            ok = check(d_voids < d_prior - 2.0 * noise, f"closure {name} seat {seat}: `voids` sits {d_voids:.4f} from the exact posterior against the uniform split's {d_prior:.4f}, so it does not move the belief towards the posterior -- which is the one thing the rung claims to do")
        else:
            ok = check(d_soft < d_voids - 2.0 * noise, f"closure {name} seat {seat} vs {opp_agent.name}: `soft` sits {d_soft:.4f} from the exact posterior against `voids`' {d_voids:.4f}, so believing the void buys nothing over enforcing it even against an opponent the enforcement is wrong about -- the one case the rung was added for")
        ok += check(sep >= 4.0 * noise, f"closure {name} seat {seat}: prior and posterior are only {sep:.4f} apart, {sep / max(noise, 1e-9):.1f}x the noise -- there is nothing here to close, so the test is near-vacuous")
        # The family property, end to end on the *sampled* rung rather than on `_void_weight`
        # alone: a sweep that does not reproduce its own endpoints is measuring something else.
        if 0.0 in decays:
            ok += check(
                abs(d_soft_by_decay[0.0] - d_voids) <= 4.0 * noise,
                f"closure {name} seat {seat}: `soft` at decay 0 sits {d_soft_by_decay[0.0]:.4f} from the posterior against `voids`' {d_voids:.4f}, but decay 0 *is* `voids` -- the one-parameter family does not reproduce its own endpoint",
            )
        if 1.0 in decays:
            ok += check(
                abs(d_soft_by_decay[1.0] - d_prior) <= 4.0 * noise,
                f"closure {name} seat {seat}: `soft` at decay 1 sits {d_soft_by_decay[1.0]:.4f} from the posterior against the uniform split's {d_prior:.4f}, but decay 1 *is* the uniform split -- the one-parameter family does not reproduce its own endpoint",
            )
        print(f"      {ok}{over if empty else ''}")
        results[seat] = {"prior": float(d_prior), "voids": float(d_voids), "sep": float(sep), "noise": float(noise), "soft": d_soft_by_decay}
        sys.stdout.flush()
    return results


def decay_section(name: str, sets: int, samples: int, min_support: int, decays: tuple[float, ...]) -> None:
    """Is there a decay good against **both** kinds of opponent? (the league question)

    `voids` is near-exact against an opponent that plays a playable drawn card at once, and close
    to worthless against one that does not (section 26.6, and the `passer` column here): the two
    columns rank the rungs in opposite orders. `soft` interpolates between them -- `decay = 0`
    *is* `voids`, `decay = 1` is the uniform prior -- so the question is whether some decay gives
    up little against the honest column while recovering most of what `voids` loses against the
    other. That is what a league needs from a single rung: an opponent it can field without
    knowing which kind it is about to meet.

    Scored by total variation to each column's **exact** posterior, so this costs minutes and no
    eval seeds. Settling the same question on score would take ~39,000 deals -- the 2,400-deal DEV
    sweep that picked `decay = 0.1` resolves about +-2 points -- and `divergence` shows the two
    rungs choose the same move at all but ~1 decision in 100 against an honest column, so there is
    very little score there to find in the first place.

    The summary criterion is **maximin over the two columns**: the worst column a rung faces,
    maximised. It is a choice, stated so it can be argued with. It suits a league where the row
    does not know who it is facing; a row that *does* know should switch rungs instead of
    compromising, since each column's own best rung is strictly better there.
    """
    res = {opponent: closure_section(name, sets, samples, min_support, opponent, decays) for opponent in ("rulebot", "passer")}

    def over_seats(col: str, key: str, d: float | None = None) -> float:
        vals = [r["soft"][d] if key == "soft" else r[key] for r in res[col].values()]
        return sum(vals) / len(vals) if vals else float("nan")

    prior = {c: over_seats(c, "prior") for c in res}
    table = [("voids", {c: over_seats(c, "voids") for c in res})]
    table += [(f"soft {d:g}", {c: over_seats(c, "soft", d) for c in res}) for d in decays]

    print(f"\n== {name} decay balance: TV to each column's exact posterior, mean over seats.")
    print("   `rulebot` is the column `voids` assumes; `passer` keeps a playable drawn card half")
    print("   the time, which is exactly the assumption broken. `closed` is the share of the")
    print("   uniform split's distance to that column's posterior that the rung removes.")
    print(f"      {'rung':12s} {'TV rulebot':>12s} {'closed':>8s} {'TV passer':>11s} {'closed':>8s} {'worst':>8s}")
    best, best_worst = None, -2.0
    for label, tvs in table:
        closed = {c: (prior[c] - tvs[c]) / prior[c] for c in tvs}
        worst = min(closed.values())
        if worst > best_worst:
            best, best_worst = label, worst
        print(f"      {label:12s} {tvs['rulebot']:12.4f} {100 * closed['rulebot']:7.1f}% {tvs['passer']:11.4f} {100 * closed['passer']:7.1f}% {100 * worst:7.1f}%")
    print(f"\n   maximin over the two columns: **{best}**, at {100 * best_worst:.1f} % of the distance closed in its worst column.")
    sys.stdout.flush()


def _divergence_chunk(rs, row, variants, col, seeds):
    """(seat, a void is live, one flag per variant: it would have chosen differently) per decision
    of the row's own games. One trajectory, several opinions about it -- the variants are asked at
    the row's states and never get to steer, which is what makes the shares comparable."""
    from uno.eval.belief import voids as void_constraints
    from uno.eval.play import play_game
    from uno.game.engine import legal_actions

    out = []
    for seed in seeds:
        for seat in (0, 1):
            pair = (row, col) if seat == 0 else (col, row)
            try:
                t = play_game(rs, seed, pair)
            except ValueError:
                continue
            for st in t.steps:
                if st.state.player != seat:
                    continue
                legal = legal_actions(st.state)
                if len(legal) < 2:
                    continue  # forced: no choice to differ on, and `policy` short-circuits anyway
                live = bool(void_constraints(rs, st.obs.public, seat ^ 1))
                diffs = tuple(max(v.policy(st.obs, legal).items(), key=lambda kv: kv[1])[0] != st.action for v in variants)
                out.append((seat, live, diffs))
    return out


def divergence_section(games: int, worlds: int, depth: int, workers: int | None, column: str = "random") -> None:
    """How often do `soft` and `voids` actually *choose differently*? (step 9.5f follow-up)

    The belief ladder separates rungs by score, and at 2,400 DEV deals that resolves about +-2
    points -- which is why `soft` at `decay = 0.1` sits in the "indistinguishable" bucket rather
    than being settled. This asks the cheaper question first: at how many decisions do the two
    rungs disagree at all? A rung that plays `voids`' move nearly everywhere cannot differ from it
    by much however many deals the ladder runs, so this bounds the score question by arithmetic
    instead of buying the answer in deals.

    **The control is the point of the section.** Each rung is a separate agent with its own name,
    and `PimcSearch._rng` keys the world sampler on (seed, name, information state) -- so two
    agents *of the same rung* under different names already disagree wherever 32 sampled worlds
    happen to produce different argmaxes. That floor is what a `soft - voids` disagreement must
    clear to mean anything, and a second `voids` under a different name measures it. `soft` is run
    twice for the same reason: under its own name (what the ladder compares) and under the row's
    name, where the sampler starts from the same seed, so what is left there is the belief rather
    than the stream.

    DEV seeds: this is a diagnostic, and `EVAL_SEEDS` is spent (09.5f-results.md section 7.3).
    """
    from uno.eval.play import DEV_SEEDS

    if column not in ("random", "heuristic"):
        raise SystemExit(f"unknown divergence column {column!r}: random or heuristic")
    rs = rules.FULL_UNO
    nets = [net_for(rs, "heuristic", "random", s) for s in (0, 1)]
    leaves = (nets[0].leaf, nets[1].leaf)
    col = PlayFirstRandom(rs) if column == "random" else Heuristic(rs)
    seeds = DEV_SEEDS[:games]
    k = (os.cpu_count() or 1) * 8 if workers != 1 else 1

    def agent(kind: str, name: str):
        return PimcSearch(rs, leaves, worlds=worlds, depth=depth, name=name, opponent=col, belief=Belief(rs, kind=kind))

    base = f"pimc_resp_{column}_voids"
    row = agent("voids", base)
    variants = (
        ("soft(0.1), row's world key", agent("soft", base)),
        ("soft(0.1), own world key", agent("soft", f"pimc_resp_{column}_soft")),
        ("voids, own world key (control)", agent("voids", f"pimc_resp_{column}_voids_b")),
    )

    print(f"\n== FULL_UNO belief divergence vs {col.name}: w={worlds}, d={depth}, response continuation.")
    print(f"   {games} DEV deals, both seatings. The games are the `voids` row's own; every variant")
    print("   is asked for its action at the row's decisions, so this is one trajectory, not three.")
    sys.stdout.flush()
    t0 = time.perf_counter()
    parts = pmap(_divergence_chunk, [(rs, row, [v for _, v in variants], col, c) for c in chunks(seeds, k)], workers)
    rows = [r for p in parts for r in p]
    secs = time.perf_counter() - t0
    n = len(rows)
    n_live = sum(1 for r in rows if r[1])
    print(f"   {n:,} row decisions with a choice, {n_live:,} of them with a void live ({100 * n_live / max(n, 1):.1f} %), {secs:.0f}s")
    print(f"      {'variant':32s} {'all decisions':>18s} {'void live':>12s} {'seat 0':>9s} {'seat 1':>9s}")

    def share(j: int, sel=None) -> float:
        sub = [r for r in rows if sel is None or sel(r)]
        return 100.0 * sum(1 for r in sub if r[2][j]) / len(sub) if sub else 0.0

    shares = []
    for j, (label, _) in enumerate(variants):
        allp = share(j)
        se = math.sqrt(max(allp, 1e-9) * (100.0 - allp) / max(n, 1))
        cells = (f"{allp:.1f} +- {se:.1f} %", f"{share(j, lambda r: r[1]):.1f} %", f"{share(j, lambda r: r[0] == 0):.1f} %", f"{share(j, lambda r: r[0] == 1):.1f} %")
        print(f"      {label:32s} {cells[0]:>18s} {cells[1]:>12s} {cells[2]:>9s} {cells[3]:>9s}")
        shares.append(allp)

    # Non-vacuity, the step 9.5e lesson: if re-seeding the world sampler changed no decision, this
    # instrument could not see a belief difference either and every share above would be noise.
    ok = check(shares[2] > 0.0, "divergence: the control never diverged, so re-seeding the world sampler changes nothing here and this instrument cannot detect a belief difference at all")
    ok += check(n > 1000, f"divergence: only {n} decisions compared, too few to read a share from")
    print(f"      {ok}")
    print("\n   Reading: `soft` under its own name -- the comparison the ladder makes -- differs at")
    print(f"   {shares[1]:.1f} % of decisions, against the control's {shares[2]:.1f} %, which is the share that")
    print("   re-seeding 32 worlds produces on its own, with no change of belief whatever.")
    sys.stdout.flush()


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    section = argv[0] if argv else "exact"
    args = argv[1:]
    if section == "exact":
        exact_section(args[0].split(",") if args else list(TOYS))
    elif section == "pimcn":
        names = args[0].split(",") if args else list(TOYS)
        pimcn_section(names, int(args[1]) if len(args) > 1 else 64, int(args[2]) if len(args) > 2 else 64)
    elif section == "uno":
        uno_section(
            int(args[0]) if args else 400,
            int(args[1]) if len(args) > 1 else 8,
            int(args[2]) if len(args) > 2 else 2,
            int(args[3]) if len(args) > 3 and args[3] else None,
            args[4] if len(args) > 4 else "m",
        )
    elif section == "d2":
        d2_section(
            int(args[0]) if args else 200,
            int(args[1]) if len(args) > 1 else 8,
            int(args[2]) if len(args) > 2 else 2,
            int(args[3]) if len(args) > 3 else None,
        )
    elif section == "sweep":
        sweep_section(
            int(args[0]) if args else 200,
            int(args[1]) if len(args) > 1 and args[1] else None,
            args[2] if len(args) > 2 and args[2] else None,
        )
    elif section == "belief":
        belief_section(
            int(args[0]) if args else 1000,
            int(args[1]) if len(args) > 1 else 32,
            int(args[2]) if len(args) > 2 else 2,
            int(args[3]) if len(args) > 3 and args[3] else None,
            args[4] if len(args) > 4 and args[4] else "random",
        )
    elif section == "calibration":
        calibration_section(int(args[0]) if args else 120, int(args[1]) if len(args) > 1 else 64)
    elif section == "closure":
        closure_section(
            args[0] if args else "TOY",
            int(args[1]) if len(args) > 1 else 40,
            int(args[2]) if len(args) > 2 else 2000,
            int(args[3]) if len(args) > 3 else 3,
            args[4] if len(args) > 4 and args[4] else "rulebot",
        )
    elif section == "decay":
        decay_section(
            args[0] if args else "TOY",
            int(args[1]) if len(args) > 1 else 20,
            int(args[2]) if len(args) > 2 else 4000,
            int(args[3]) if len(args) > 3 else 3,
            tuple(float(x) for x in args[4].split(",")) if len(args) > 4 and args[4] else (0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0),
        )
    elif section == "divergence":
        divergence_section(
            int(args[0]) if args else 300,
            int(args[1]) if len(args) > 1 else 32,
            int(args[2]) if len(args) > 2 else 2,
            int(args[3]) if len(args) > 3 and args[3] else None,
            args[4] if len(args) > 4 and args[4] else "random",
        )
    else:
        raise SystemExit(f"unknown section {section!r}: exact, pimcn, uno, d2, sweep, belief, calibration, closure, divergence, decay")
    print(f"\n== {len(au.FAILS)} failed checks" + (": " + "; ".join(au.FAILS) if au.FAILS else ""))


if __name__ == "__main__":
    main()
