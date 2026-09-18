"""Step 10a.3C: local best response (LBR) as a calibrated attack instrument.

    uv run uno lbr exact  [rulesets=TOY] [worlds=1,4,16] [seeds=3] [budget_worlds=16]
    uv run uno lbr games  [ruleset=TOY] [deals=200] [worlds=4]
    uv run uno lbr pilot  [deals=200] [worlds=32] [depth=2] [workers] [search_deals=100]
    uv run uno lbr eval   [deals=4000] [workers]

An attack's *achieved score in real games* lower-bounds the best-response score against a frozen
target, so a weak or biased planner still produces a valid bound (docs/foundations/results/10-readiness/10-review.md
section 4). The risk is an **insensitive** attack: a small number from a blunt instrument looks
exactly like a safe target. The toys are where that can be settled, because the exact best
response is computable there -- so `exact` measures how much of each target's *known*
exploitability the attack actually collects, and how much it misses.

`exact`: the calibration. Every attacker is **tabulated** on the quotient tree and scored by one
ordinary `evaluate`, so its value carries no game-playing noise whatsoever -- the attack score, the
exact BR L1 ceiling, and the difference between them are all exact. Cost is ~0.1-0.8 ms per
information set, so a whole `TOY` attacker is a couple of seconds.

`games`: the sampled path, checked against the exact one. The attacker that is *played* is the same
`Symmetrized` object that was tabulated, and its score is compared with `scores_on_seeds` -- the
exact value **on the same deals**, which is the calibrated scale (docs/foundations/spec.md section 14). A
comparison against the whole-game value would be the wrong scale: on 60 `TOY` deals the deal luck
alone is ~1.9 score points. This section also reports the attacker's internal **search Q** beside
its achieved score, because the two are different objects and only the second is a bound.

`pilot`: a bounded full-UNO DEV cost/sensitivity run. Nothing here is reported on EVAL seeds: the
configuration and the measurement budget are frozen from this pilot's variances *first*, and only
then does a confirmatory block get claimed from `uno.eval.seeds` (docs/foundations/spec.md section 27.1). Cost
measured at 0.26 s/game against a rulebot target and 8.11 s/game against a target that is itself a
search, which is why the search-target row runs on its own small deal count.

Every check prints ok / XX; the last line counts the failures. docs/foundations/spec.md section 32.
"""

from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

from uno import store
from uno.agents.baseline import Heuristic, KeepsDrawn, PlayFirstRandom
from uno.agents.symmetrized import Symmetrized
from uno.agents.tabular import CompiledAgent
from uno.compiled import flaws
from uno.compiled.pimc import pimc_n
from uno.compiled.solvers import best_response, evaluate, exploitability
from uno.compiled.strategies import tabulate, transfer
from uno.compiled.tree import compile_tree
from uno.eval import attack
from uno.eval import seeds as seed_registry
from uno.eval.attack import Budget, UnsupportedTarget
from uno.eval.belief import Belief
from uno.eval.pimc import chunks, pmap
from uno.eval.search import PimcSearch
from uno.game import rules
from uno.game.observe import Level

L1 = Level.L1
FAILS: list[str] = []

# A depth that always reaches a terminal on the toys, so every determinized world is solved
# exactly and the leaf is never consulted: the toy calibration then depends on no value network.
TOY_DEPTH = 40


def check(ok: bool, what: str) -> str:
    if not ok:
        FAILS.append(what)
    return "ok" if ok else "XX"


def score(v: float) -> float:
    return (v + 1) / 2


def no_leaf(s):
    raise AssertionError("the search hit its depth limit: a toy world was not solved exactly")


# -- the exact calibration ------------------------------------------------------------------------


def _targets(tree, rs, nash_sigma):
    """The calibration targets: `{label: (sigma, agent or None)}`.

    `agent` is a queryable position-only policy where one exists, which is what the `response`
    continuation needs; `None` marks a target that exists only as a strategy over the keying (Nash
    and the grafted flaws), attackable by the continuations that never read the target."""
    rnd, heu, passer = PlayFirstRandom(rs), Heuristic(rs), KeepsDrawn(rs, 0.5)
    out = {
        "random": (store.load_or_tabulate(tree, L1, rnd), rnd),
        "heuristic": (store.load_or_tabulate(tree, L1, heu), heu),
        "passer0.5": (store.load_or_tabulate(tree, L1, passer), passer),
        "nash": (nash_sigma, None),
    }
    # the near-Nash target (a graft that costs almost nothing) and the localized flaws
    grafts = flaws.flawed(tree, nash_sigma, specs=[("pass", "never"), ("pass", "always"), ("wild_play", "never"), ("action_card", "never")])
    for label, sigma in grafts.items():
        out[label] = (sigma, None)
    return out


def _families(rs):
    """The attacker families, as `{name: build kwargs}`.

    `response` is offered against *every* target, including the ones it cannot query: whether it may
    is `attack.check_supported`'s decision, and letting it refuse in the table is the point -- item 1
    of the plan asks for unsupported private-history queries to be rejected **explicitly**, not
    quietly omitted."""
    heu, rnd = Heuristic(rs), PlayFirstRandom(rs)
    return {
        "response": {},
        "surrogate": dict(surrogate=heu),
        "minimax": {},
        "rollout": dict(rollout_agents=(heu, rnd)),
    }


def _attack_sigma(tree, rs, target_agent, target_sigma, family: str, budget: Budget, kw) -> np.ndarray | None:
    """The tabulated attacker, or None when this family may not attack this target.

    The target handed to `attack.build` is the queryable agent when there is one; otherwise a
    `CompiledAgent` over the target's strategy, which `check_supported` will refuse for `response`
    -- deliberately, since that is the case the instrument must reject rather than answer badly."""
    queryable = target_agent if target_agent is not None else CompiledAgent(tree, L1, target_sigma, name="tabulated_target")
    try:
        att = attack.build(rs, queryable, budget, leaf=no_leaf, **kw)
    except UnsupportedTarget:
        return None
    return tabulate(tree, L1, Symmetrized(att, rs, name=f"sym_{att.name}"))


def _seat_scores(tree, att_sigma, target_sigma) -> tuple[float, float]:
    """The attacker's exact score at seat 0 and at seat 1 against the target."""
    return (
        score(evaluate(tree, (L1, att_sigma), (L1, target_sigma))[0]),
        score(evaluate(tree, (L1, target_sigma), (L1, att_sigma))[1]),
    )


def exact_section(names: list[str], ladder: tuple[int, ...], n_seeds: int, budget_worlds: int) -> None:
    for name in names:
        rs = getattr(rules, name)
        tree = compile_tree(rs, (L1,), True, cache=store.CACHE)
        nash = store.load_or_solve_nash(tree, 1000, "cfr+")
        nash_v = score(evaluate(tree, (L1, nash.avg), (L1, nash.avg))[0])
        targets = _targets(tree, rs, nash.avg)
        rnd_sigma = targets["random"][0]
        print(f"\n== {name}: exact attack calibration")
        print(f"   {tree}")
        print(f"   Nash seat-0 score {nash_v:.4f} (CFR+ {nash.iterations} its, NashConv {100 * nash.final.nashconv:.3f} pts); seeds: exact, none spent")

        # -- 1. the targets, and the exploitability that exists to be found -----------------
        print(f"\n   1. the targets. 'true NashConv' is exact; an attack can only ever bound it from below.")
        print(f"      {'target':26s} {'exploitability s0 / s1':>24s} {'true NashConv':>14s} {'BR L1 s0 / s1':>18s} {'random s0 / s1':>18s}")
        truth = {}
        for label, (sigma, _) in targets.items():
            ex = exploitability(tree, L1, sigma)
            per = tuple(100 * (ex.br[s ^ 1].score - (1 - nash_v if s == 0 else nash_v)) for s in (0, 1))
            ceil = tuple(best_response(tree, L1, (L1, sigma), s).score for s in (0, 1))
            floor = _seat_scores(tree, rnd_sigma, sigma)  # the same-seat Random baseline
            truth[label] = {"nashconv": per[0] + per[1], "per_seat": per, "ceiling": ceil, "floor": floor}
            print(f"      {label:26s} {per[0]:11.2f} /{per[1]:7.2f} {per[0] + per[1]:14.2f} {ceil[0]:9.4f} /{ceil[1]:8.4f} {floor[0]:9.4f} /{floor[1]:8.4f}")
            sys.stdout.flush()

        # -- 2. what each attacker family collects ------------------------------------------
        budget = Budget(worlds=budget_worlds, depth=TOY_DEPTH, belief="prior", seed=0)
        print(f"\n   2. attacks at w={budget_worlds} (belief 'prior', depth {TOY_DEPTH} so every world is solved exactly).")
        print("      'missed' = BR L1 - achieved, in score points; 'found' = the share of the range from the")
        print("      same-seat Random baseline to BR L1 (the report card's normalised scale, never selected on).")
        print(f"      {'target':26s} {'family':10s} {'achieved s0 / s1':>19s} {'missed s0 / s1':>16s} {'found':>13s} {'NashConv >=':>12s} {'of true':>8s}")
        for label, (sigma, agent) in targets.items():
            t = truth[label]
            for family, kw in _families(rs).items():
                b = Budget(continuation=family, worlds=budget.worlds, depth=budget.depth, belief="prior", seed=0)
                att_sigma = _attack_sigma(tree, rs, agent, sigma, family, b, kw)
                if att_sigma is None:
                    print(f"      {label:26s} {family:10s} refused: this target reads its own private history, so it cannot be queried inside a world")
                    continue
                got = _seat_scores(tree, att_sigma, sigma)
                missed = tuple(100 * (t["ceiling"][s] - got[s]) for s in (0, 1))
                found = tuple(
                    (got[s] - t["floor"][s]) / (t["ceiling"][s] - t["floor"][s]) if t["ceiling"][s] - t["floor"][s] > 1e-9 else float("nan") for s in (0, 1)
                )
                bound = 100 * (got[0] + got[1] - 1)
                # A share of a NashConv that is itself ~0 is meaningless (and prints as nonsense):
                # against an unexploitable target the bound is <= 0 by construction, which is the
                # finding, not a percentage of nothing.
                share_s = f"{bound / t['nashconv']:7.1%}" if t["nashconv"] > 0.05 else "     --"
                ok = check(
                    got[0] <= t["ceiling"][0] + 1e-9 and got[1] <= t["ceiling"][1] + 1e-9,
                    f"{name} {label} {family}: the attack beat the exact BR L1 ({got} against {t['ceiling']}), which is impossible",
                )
                print(
                    f"      {label:26s} {family:10s} {got[0]:9.4f} /{got[1]:8.4f} {missed[0]:8.2f} /{missed[1]:6.2f}"
                    f" {found[0]:6.2f} /{found[1]:5.2f} {bound:12.2f} {share_s}  {ok}"
                )
                sys.stdout.flush()

        # -- 3. the world ladder: a stronger attack must miss less -------------------------
        print(f"\n   3. the world ladder against the Heuristic (response) and the localized flaw (surrogate).")
        print("      Pre-declared on the **two-seat sum**, which is the quantity actually reported: the largest")
        print("      budget must miss less than the smallest. Not per seat, and not adjacent-pair monotone -- the")
        print("      attacker's own world-sampling seed moves a single seat by 0.6-0.8 score points at these")
        print("      budgets (section 5), so a per-seat wobble of a few tenths is below the instrument's own")
        print("      resolution and checking it would be checking noise. And not that it reaches zero: PIMC's")
        print("      structural bias does not vanish with more worlds (docs/foundations/spec.md section 25).")
        flaw_label = next((k for k in ("nash_never_action_card", "nash_never_wild_play", "nash_always_pass") if k in targets), None)
        rungs = [("heuristic", "response", {}), *([(flaw_label, "surrogate", dict(surrogate=Heuristic(rs)))] if flaw_label else [])]
        for label, family, kw in rungs:
            sigma, agent = targets[label]
            t = truth[label]
            print(f"      {label} / {family}:")
            sums: list[tuple[int, float]] = []
            for w in ladder:
                b = Budget(continuation=family, worlds=w, depth=TOY_DEPTH, belief="prior", seed=0)
                att_sigma = _attack_sigma(tree, rs, agent, sigma, family, b, kw)
                if att_sigma is None:
                    continue
                got = _seat_scores(tree, att_sigma, sigma)
                missed = tuple(100 * (t["ceiling"][s] - got[s]) for s in (0, 1))
                total = missed[0] + missed[1]
                note = "" if not sums or total <= sums[-1][1] + 1e-9 else "  (sum up on this rung: reported, not checked)"
                sums.append((w, total))
                print(f"         w={w:<4d} achieved {got[0]:.4f} / {got[1]:.4f}   missed {missed[0]:6.2f} / {missed[1]:6.2f} pts   sum {total:6.2f}{note}")
                sys.stdout.flush()
            if len(sums) > 1:
                ok = check(
                    sums[-1][1] <= sums[0][1] + 1e-9,
                    f"{name} {label} {family}: the largest budget (w={sums[-1][0]}, missed sum {sums[-1][1]:.2f}) is no better than the smallest (w={sums[0][0]}, {sums[0][1]:.2f})",
                )
                print(f"         w={sums[0][0]} -> w={sums[-1][0]}: missed sum {sums[0][1]:.2f} -> {sums[-1][1]:.2f} pts  {ok}")

        # -- 4. why a memoryful target is refused, measured ---------------------------------
        print("\n   4. the refusal is measured, not declared: does a target read its own private history?")
        for label, (sigma, agent) in targets.items():
            probe = agent if agent is not None else CompiledAgent(tree, L1, sigma, name=f"tab_{label}")
            sup = attack.supports(rs, probe, games=25)
            print(f"      {sup}")
        print("      A `response` attack on such a target would query it inside determinized worlds, where its")
        print("      private history is fabricated; measured on TOY, 79.5 % of those queries miss the keying and")
        print("      silently return the first legal action, so the search would best-respond to the wrong policy.")

        # -- 5. the attacker's own seed, against the exact behavioural limit ---------------
        print(f"\n   5. the attacker's own world-sampling seed ({n_seeds} seeds, response vs random, w={budget_worlds}).")
        print("      A built attacker is ONE realization keyed on the information state, not the averaged")
        print("      behavioural policy; `uno.compiled.pimc.pimc_n` is that average, and the seeds must straddle it.")
        world = compile_tree(rs, (L1, Level.L3), True, cache=store.CACHE)
        target_w = (L1, transfer(tree, world, L1, rnd_sigma))
        rnd = PlayFirstRandom(rs)
        # one tabulation per seed, not per (seat, seed): a tabulated attacker covers both seats
        per_seed = []
        for k in range(n_seeds):
            b = Budget(continuation="response", worlds=budget_worlds, depth=TOY_DEPTH, belief="prior", seed=k)
            per_seed.append(_seat_scores(tree, _attack_sigma(tree, rs, rnd, rnd_sigma, "response", b, {}), rnd_sigma))
        for seat in (0, 1):
            v = np.array([p[seat] for p in per_seed])
            pn = pimc_n(world, target_w, seat, budget_worlds, realizations=32, weighting="prior", continuation="response", seed=1)
            sd = float(v.std(ddof=1)) if len(v) > 1 else 0.0
            se = sd / math.sqrt(len(v)) if len(v) > 1 else 0.0
            # the mean over seeds estimates pimc_n's average; allow for both sides' Monte Carlo error
            tol = max(3 * se, 0.01)
            ok = check(
                abs(v.mean() - pn.score) <= tol,
                f"{name} seat {seat}: the mean over {n_seeds} attacker seeds ({v.mean():.4f}) is not pimc_n's average ({pn.score:.4f}, tol {tol:.4f})",
            )
            print(f"      seat {seat}: {' '.join(f'{x:.4f}' for x in v)}   mean {v.mean():.4f} sd {100 * sd:.2f} pts   pimc_n {pn.score:.4f}  {ok}")
            sys.stdout.flush()

        # -- 6. the calibration's own non-vacuity ------------------------------------------
        print("\n   6. non-vacuity: the instrument must separate a near-Nash target from a flawed one.")
        near = "nash_never_pass" if "nash_never_pass" in targets else None
        if near and flaw_label:
            pairs = {}
            for label in (near, flaw_label, "nash"):
                sigma, agent = targets[label]
                b = Budget(continuation="surrogate", worlds=budget_worlds, depth=TOY_DEPTH, belief="prior", seed=0)
                att_sigma = _attack_sigma(tree, rs, agent, sigma, "surrogate", b, dict(surrogate=Heuristic(rs)))
                got = _seat_scores(tree, att_sigma, sigma)
                pairs[label] = 100 * (got[0] + got[1] - 1)
                print(f"      {label:26s} NashConv >= {pairs[label]:7.2f} pts   (true {truth[label]['nashconv']:.2f})")
            ok = check(
                pairs[flaw_label] > pairs[near] + 2.0,
                f"{name}: the attack does not separate the flawed target from the near-Nash one ({pairs[flaw_label]:.2f} against {pairs[near]:.2f})",
            )
            print(f"      the flawed target's bound exceeds the near-Nash one by {pairs[flaw_label] - pairs[near]:.2f} pts  {ok}")


# -- the sampled path ----------------------------------------------------------------------------


def _pair_chunk(rs, pair, seeds):
    """Per deal, the first agent's score at each seat, from the two seatings of one deal."""
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
        out.append((score(returns(t0.final)[0]), score(returns(t1.final)[1])))
    return out


def games_section(name: str, deals: int, worlds: int) -> None:
    """Item 4: the attacker scored by actual games, validated against the exact toy values."""
    from uno.eval.play import play_from, play_game
    from uno.game.engine import legal_actions, new_game, returns
    from uno.game.observe import observe
    from uno.reference.solve import scores_on_seeds

    rs = getattr(rules, name)
    tree = compile_tree(rs, (L1,), True, cache=store.CACHE)
    rnd = PlayFirstRandom(rs)
    target_sigma = store.load_or_tabulate(tree, L1, rnd)
    budget = Budget(continuation="response", worlds=worlds, depth=TOY_DEPTH, belief="prior", seed=0)
    raw = attack.build(rs, rnd, budget, leaf=no_leaf)
    att = Symmetrized(raw, rs, name=f"sym_{raw.name}")
    seeds = list(seed_registry.dev_block(0, deals))

    print(f"\n== {name}: the sampled attack against Random, w={worlds} ({deals} DEV deals, both seatings)")
    att_sigma = tabulate(tree, L1, att)
    exact = _seat_scores(tree, att_sigma, target_sigma)
    print(f"   the attacker tabulated: exact whole-game score {exact[0]:.4f} / {exact[1]:.4f}")

    # 1. the played score against the exact value ON THE SAME DEALS
    print("\n   1. played games against the exact value on the same deals (spec section 14's calibrated scale).")
    for seat in (0, 1):
        pair = (att, rnd) if seat == 0 else (rnd, att)
        on = scores_on_seeds(rs, pair, seeds)
        played = []
        for sd in seeds:
            try:
                s0 = new_game(rs, sd)
            except ValueError:
                continue
            played.append(score(returns(play_from(s0, sd, pair).final)[seat]))
        xs = np.array(played)
        z = on.z(xs.mean(), seat)
        ok = check(abs(z) <= 3.0, f"{name} seat {seat}: played {xs.mean():.4f} against exact-on-seeds {on.mean[seat]:.4f}, z {z:+.2f}")
        print(
            f"      seat {seat}: played {xs.mean():.4f} over {len(xs)} deals   exact on these deals {on.mean[seat]:.4f}"
            f"   z {z:+.2f}  {ok}   (whole-game {exact[seat]:.4f}: {100 * abs(on.mean[seat] - exact[seat]):.2f} pts of deal luck)"
        )
        sys.stdout.flush()

    # 2. the search Q is not the achieved score
    print("\n   2. the attacker's internal search Q beside its achieved score. These are different objects:")
    print("      the Q is an optimistic perfect-information average over sampled worlds, not a value estimate,")
    print("      and only the achieved score is a bound (docs/foundations/spec.md section 25 on the winner's curse).")
    # The Q is read through `scores_split`, as `uno.experiments.pimc._d2_chunk` does: the action is
    # chosen on one half of the worlds and valued on the other, so this is not additionally inflated
    # by the winner's curse (docs/foundations/spec.md section 25). What remains is the optimism that *is* the
    # point -- a perfect-information average over sampled worlds is not the value of the real game.
    # Only the Q at the attacker's **first** decision is comparable to a game score: both are then
    # "the expected outcome of this game". Averaging Q over decisions at every depth and comparing
    # that to a root score compares two different quantities, and the difference means nothing in
    # either direction -- a first draft of this section did exactly that.
    # Q and achieved score must be collected as *pairs*, per game: a first draft appended the score
    # of every game but a Q only where the first decision had a real choice, so "the same games" was
    # not the same set of games at all.
    pairs: list[tuple[float, float]] = []
    for sd in seeds[: min(len(seeds), 60)]:
        try:
            t = play_game(rs, sd, (att, rnd))
        except ValueError:
            continue
        for st in t.steps:
            if st.state.player != 0:
                continue
            legal = legal_actions(st.state)
            if len(legal) < 2:
                continue
            sel, val = raw.scores_split(st.obs, legal)
            chosen = max(legal, key=lambda a: (sel[a], -a))
            pairs.append((score(val[chosen]), score(returns(t.final)[0])))
            break  # the root decision only, which is the one comparable to a whole-game score
    if pairs:
        q = np.array([p[0] for p in pairs])
        got = np.array([p[1] for p in pairs])
        d = q - got
        se = d.std(ddof=1) / math.sqrt(len(d))
        print(f"      search Q at the attacker's first decision: {q.mean():.4f} over {len(q)} games with a real root choice")
        print(f"      achieved score on those same {len(q)} games:      {got.mean():.4f}")
        print(f"      difference {100 * d.mean():+.2f} +- {100 * se:.2f} score points (paired), z {d.mean() / se if se > 0 else 0:+.2f}")
        print("      Reported, never used as the bound. At this sample the difference is not resolved, so its")
        print("      SIGN carries no finding: the point is only that the two are different objects -- a")
        print("      perfect-information average over sampled worlds against a realized score.")

    # 3. information boundary, honesty of the reported policy, reproducibility
    print("\n   3. the instrument's own properties.")
    # L1 validity is about what the attacker may NOT see -- the opponent's hand and the pile order.
    # `attack.supports` is the wrong instrument for it: an L1 policy is entitled to read its own
    # initial hand and draws, and those determine its current hand, so a key-rebuilding agent
    # legitimately answers differently there. `hidden_state_probe` substitutes a different hidden
    # world behind an identical L1 observation, which is the property that actually matters.
    boundary = attack.hidden_state_probe(rs, att, games=20)
    ok = check(boundary.l1_valid, f"{name}: the attacker's policy moved when only the hidden cards changed ({boundary})")
    print(f"      L1 validity: {boundary}  {ok}")
    s = new_game(rs, seeds[0])
    obs, legal = observe(s, s.player, L1), legal_actions(s)
    p1, p2 = att.policy(obs, legal), att.policy(obs, legal)
    det = len(p1) == 1 and abs(next(iter(p1.values())) - 1.0) < 1e-12
    ok = check(p1 == p2 and det, f"{name}: the attacker is not a deterministic function of its information state ({p1} then {p2})")
    print(f"      reported policy is deterministic and repeatable: {p1 == p2 and det}  {ok}   (AIVAT's row correction needs this)")
    twin_raw = attack.build(rs, rnd, budget, leaf=no_leaf)
    twin = Symmetrized(twin_raw, rs, name=att.name)
    a = np.array(_pair_chunk(rs, (att, rnd), seeds[:40]))
    b = np.array(_pair_chunk(rs, (twin, rnd), seeds[:40]))
    ok = check(a.shape == b.shape and np.array_equal(a, b), f"{name}: the same attack built twice did not replay its own games")
    print(f"      an attacker built twice replays its games exactly: {np.array_equal(a, b)}  {ok}")


# -- the full-UNO pilot ---------------------------------------------------------------------------


def pilot_section(deals: int, worlds: int, depth: int, workers: int | None, search_deals: int) -> None:
    """Item 5: a bounded DEV cost/sensitivity pilot. Nothing here is reported on EVAL seeds."""
    from uno.experiments.pimc import net_for

    rs = rules.FULL_UNO
    nets = [net_for(rs, "heuristic", "random", s) for s in (0, 1)]
    leaves = (nets[0].leaf, nets[1].leaf)
    heu, rnd = Heuristic(rs), PlayFirstRandom(rs)
    k = (os.cpu_count() or 1) * 8 if workers != 1 else 1
    seeds = list(seed_registry.dev_block(0, deals))

    nproc = (os.cpu_count() or 1) if workers is None else workers
    print(f"\n== FULL_UNO: bounded DEV pilot (d={depth}, voids belief, per-seat value networks at the leaves)")
    print(f"   {deals} DEV deals per rulebot target, {search_deals} for the search target, both seatings of each.")
    print("   DEV only: the configuration and budget are frozen from these variances BEFORE an EVAL block is claimed.")
    print(f"   {seed_registry.describe().splitlines()[-1]}")
    print(f"   'wall s/game' is elapsed time per game across {nproc} worker process(es), i.e. throughput, NOT cost.")
    print("   Single-core cost measured separately: 0.26 s/game against a rulebot target, 8.11 s/game against a")
    print("   target that is itself a search (~31x, since every attacker node re-runs the target's own search).")
    print(f"\n   {'target':30s} {'family':10s} {'w':>4s} {'n':>5s} {'b0':>16s} {'b1':>16s} {'wall s/g':>9s} {'NashConv lower bound':>34s}")

    # the world ladder is the sensitivity axis: how much the bound moves with the attacker's budget
    rows = [
        ("random", rnd, "response", {}, deals, worlds),
        ("heuristic", heu, "response", {}, deals, max(8, worlds // 4)),
        ("heuristic", heu, "response", {}, deals, worlds),
        ("heuristic", heu, "surrogate", dict(surrogate=heu), deals, worlds),
    ]
    # the column-free PIMC row: a target that is itself a search, so ~31x the cost per game
    cf = PimcSearch(rs, leaves, worlds=worlds, depth=depth, name="pimc_cf_heuristic_voids", opponent=heu, belief=Belief(rs, kind="voids"))
    rows.append(("pimc_cf_heuristic_voids", cf, "surrogate", dict(surrogate=heu), search_deals, worlds))

    for label, target, family, kw, n, w in rows:
        b = Budget(continuation=family, worlds=w, depth=depth, belief="voids", seed=0)
        try:
            att = attack.build(rs, target, b, leaf=leaves, **kw)
        except UnsupportedTarget as e:
            print(f"   {label:30s} {family:10s} refused: {str(e)[:60]}")
            continue
        use = seeds[:n]
        t0 = time.perf_counter()
        parts = pmap(_pair_chunk, [(rs, (att, target), c) for c in chunks(use, k)], workers)
        arr = np.array([g for p in parts for g in p])
        secs = time.perf_counter() - t0
        if len(arr) < 2:
            print(f"   {label:30s} {family:10s} too few valid deals ({len(arr)})")
            continue
        # The two seatings of a deal are different games. Identical per-deal scores would mean the
        # seatings are aliased -- one game scored twice -- which would silently halve the evidence
        # and make the paired covariance meaningless. Cheap to check, so checked.
        check(
            not np.array_equal(arr[:, 0], arr[:, 1]),
            f"{label} {family} w={w}: the two seatings produced identical per-deal scores, so they are aliased rather than played separately",
        )
        bound = attack.nashconv_bound(arr[:, 0], arr[:, 1])
        cells = [f"{arr[:, s].mean():.4f}+-{arr[:, s].std(ddof=1) / math.sqrt(len(arr)):.4f}" for s in (0, 1)]
        wide = "  (interval too wide to bound anything: a cost probe)" if bound.se > 10 else ""
        print(
            f"   {label:30s} {family:10s} {w:4d} {len(arr):5d} {cells[0]:>16s} {cells[1]:>16s} {secs / (2 * len(arr)):9.2f}"
            f"   >= {bound.estimate:+6.2f} +- {bound.se:.2f} (reported {bound.reported:.2f}){wide}"
        )
        sys.stdout.flush()

    print("\n   The bound is a lower bound: a negative point estimate is a weak attack, never evidence of safety,")
    print("   and `reported` clamps at the trivial bound of zero. The seat covariance is carried through the")
    print("   per-deal sum, so no independence is assumed; measured, that correlation is mild here (-0.30 on TOY,")
    print("   -0.10 on FULL_UNO for a rulebot row, +0.10 +- 0.16 for an attacker vs Random over 40 deals), so it")
    print("   is a correction that needs no assumption rather than a large one.")


def eval_section(deals: int, workers: int | None) -> None:
    """The frozen confirmatory measurement on **EVAL block 02**.

    Everything here was fixed by the DEV pilot above *before* this block was claimed: the
    `response` continuation against position-only targets, w = 32, d = 2, the `voids` belief,
    attacker seed 0, and 4,000 deals (one standard block) at both seatings. The pilot measured the
    bound's standard error at ~4.0 points on 200 deals, so one block scales it to ~0.9 points --
    ample for the 20-60 point bounds the rulebot targets produce, and enough to resolve even a
    5-point bound at z ~ 5.6.

    The column-free PIMC row is deliberately **not** on this block. At 8.11 s/game single-core it
    would need ~1,340 deals for a 5-point bound (~45 minutes), and the pilot's 100-deal reading was
    +1.00 +- 6.11 -- a weak bound is a reported result, not a reason to keep expanding the
    experiment until it turns positive."""
    from uno.experiments.pimc import net_for

    rs = rules.FULL_UNO
    nets = [net_for(rs, "heuristic", "random", s) for s in (0, 1)]
    leaves = (nets[0].leaf, nets[1].leaf)
    heu, rnd = Heuristic(rs), PlayFirstRandom(rs)
    k = (os.cpu_count() or 1) * 8 if workers != 1 else 1
    seeds = seed_registry.block("02")[:deals]

    print("\n== FULL_UNO: the LBR attack baseline on EVAL block 02 (frozen configuration)")
    print(f"   {seed_registry.REGISTRY['02']}")
    print(f"   frozen: response continuation, w=32, d=2, voids belief, attacker seed 0, {deals} deals, both seatings.")
    print(f"\n   {'target':14s} {'n':>5s} {'b0':>16s} {'b1':>16s} {'wall s/g':>9s} {'NashConv lower bound (pts)':>40s}")
    for label, target in (("random", rnd), ("heuristic", heu)):
        b = Budget(continuation="response", worlds=32, depth=2, belief="voids", seed=0)
        att = attack.build(rs, target, b, leaf=leaves)
        t0 = time.perf_counter()
        parts = pmap(_pair_chunk, [(rs, (att, target), c) for c in chunks(seeds, k)], workers)
        arr = np.array([g for p in parts for g in p])
        secs = time.perf_counter() - t0
        check(
            not np.array_equal(arr[:, 0], arr[:, 1]),
            f"{label}: the two seatings produced identical per-deal scores, so they are aliased rather than played separately",
        )
        bound = attack.nashconv_bound(arr[:, 0], arr[:, 1])
        cells = [f"{arr[:, s].mean():.4f}+-{arr[:, s].std(ddof=1) / math.sqrt(len(arr)):.4f}" for s in (0, 1)]
        print(
            f"   {label:14s} {len(arr):5d} {cells[0]:>16s} {cells[1]:>16s} {secs / (2 * len(arr)):9.2f}"
            f"   >= {bound.estimate:+6.2f} +- {bound.se:.2f} (reported {bound.reported:.2f})"
        )
        sys.stdout.flush()
    print("\n   These are lower bounds on NashConv, not exploitability: the full-UNO Nash seat values are")
    print("   unknown, and 0.5 was never substituted for one. The attacker is a PIMC search, so a larger")
    print("   bound would need a stronger attack, not more deals.")


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    section = argv[0] if argv else "exact"
    if section == "exact":
        names = (argv[1] if len(argv) > 1 else "TOY").split(",")
        ladder = tuple(int(x) for x in (argv[2] if len(argv) > 2 else "1,4,16").split(","))
        n_seeds = int(argv[3]) if len(argv) > 3 else 3
        budget_worlds = int(argv[4]) if len(argv) > 4 else 16
        exact_section(names, ladder, n_seeds, budget_worlds)
    elif section == "games":
        games_section(argv[1] if len(argv) > 1 else "TOY", int(argv[2]) if len(argv) > 2 else 200, int(argv[3]) if len(argv) > 3 else 4)
    elif section == "eval":
        eval_section(int(argv[1]) if len(argv) > 1 else 4000, int(argv[2]) if len(argv) > 2 and argv[2] else None)
    elif section == "pilot":
        pilot_section(
            int(argv[1]) if len(argv) > 1 else 120,
            int(argv[2]) if len(argv) > 2 else 32,
            int(argv[3]) if len(argv) > 3 else 2,
            int(argv[4]) if len(argv) > 4 and argv[4] else None,
            int(argv[5]) if len(argv) > 5 else 12,
        )
    else:
        raise SystemExit(f"unknown section {section!r}: exact | games | pilot")
    print(f"\n== {len(FAILS)} failed checks" + (": " + "; ".join(FAILS) if FAILS else ""))


if __name__ == "__main__":
    main()
