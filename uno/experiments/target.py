"""Step 10a.2 item A6: the training target, closed against the exact policy gradient.

    uv run uno target [ruleset=TOY] [games=2000] [iterations=1000] [cost_deals=200]

Step 11's objective is a policy gradient, and the question step 10 has to answer before any
checkpoint exists is which of AIVAT's corrections may appear in the return it is taken of.
The declared answer is `uno.eval.target.TRAINING` -- **chance + opponent + deal, never the
learner's own decisions** -- and the argument for the exclusion
(docs/results/10-brainstorm.md 5.2) is that with an exact V the row term *is* `Q(s, a) -
V(s)`, so subtracting it removes exactly the part of the return that depends on the action.

This makes that a measurement. On a compiled tree the exact policy gradient of a tabulated
policy is available, and so is the **exact mean of the sampled estimator** under any subset
of corrections (`uno.compiled.gradient`), so the bias is a number and not a limit:

1. **exact.** The exact gradient against a central finite difference of `evaluate` (an
   independent definition); it sums to zero over every set; then, per value function and
   subset, the estimator's exact mean against it. Zero bias is required for every subset
   that spares the row, and the row correction with an exact V is predicted to drive the
   gradient to **exactly** zero. The same table with an *imperfect* V says how much of the
   signal the row correction destroys when V is only approximate, which is the case step 11
   is actually in.
2. **sampled.** Games played with the learner, the corrected signal formed per game, and
   the sampled gradient compared against the exact one -- the project's own method, closing
   a sampled thing against an exact toy value. Bias is tested through two pre-declared
   scalar projections (onto the exact gradient, and onto one orbit's own component) so that
   thousands of components are not tested at once, and the variance ratio next to them is
   what says whether the corrections are worth their cost.
3. **cost.** Measured, not inherited: 9.5c's "AIVAT costs more CPU than it saves" was
   measured for *evaluation* with cheap rulebots and one leaf call per state. A training
   loop already computes `V(s)` at every state, and the siblings the corrections need
   batch. This section counts the extra value rows per decision on `FULL_UNO` and times the
   shipped path against the `V(s)` calls the step makes anyway.

Seeds: **dev** (`uno.eval.seeds.dev_block`). Nothing is selected here and no performance
number is claimed -- section 2 is a closure check against an exact value and section 3
measures wall-clock -- so no eval block is spent (docs/spec.md section 27.1).

Every check prints ok / XX; the last line counts the failures and the false alarms expected
at |z| > 3.
"""

from __future__ import annotations

import math
import sys
import time

import numpy as np

from uno import store
from uno.agents.baseline import Heuristic, PlayFirstRandom, UniformLegal
from uno.agents.tabular import CompiledAgent
from uno.compiled.aivat import TreeValues, exact_values, l1_set_values, noisy, outcome_r2
from uno.compiled.gradient import estimator_mean, exact_gradient, finite_difference, orbit_gradient
from uno.compiled.lookup import build
from uno.compiled.tree import CompiledTree, compile_tree
from uno.eval.aivat import ALL, NONE, Corrections, trace_terms
from uno.eval.play import play_game
from uno.eval.target import TRAINING, Decisions, corrected_return, own_decisions, per_game, sampled_gradient
from uno.eval import seeds as seed_registry
from uno.game import rules
from uno.game.observe import Level

L1, L2 = Level.L1, Level.L2
FAILS: list[str] = []
STOCHASTIC = [0]

# Matchups: the learner must be a *mixed* policy, or its logit gradient is identically zero
# (a deterministic policy has no gradient to estimate). `uniform_legal` mixes over every
# legal action, which is both the cleanest case for the finite-difference check and the
# shape an untrained network starts in; Random is uniform over its plays only (zero on
# `Draw`, so some logits are at -inf); the Nash strategy is mixed and far from uniform,
# which is the shape a partly trained one has.
MATCHUPS = (("uniform_legal", "heuristic"), ("random", "heuristic"), ("nash", "random"))

# Subsets that spare the learner's own decisions: every one must leave the gradient exactly
# where it was, whatever V is.
KEEPS = {
    "none": NONE,
    "chance": Corrections(True, False, False, False),
    "opponent": Corrections(False, False, True, False),
    "deal": Corrections(False, False, False, True),
    "TRAINING": TRAINING,
}
# and the trap.
ROW = {"+row (exact-V trap)": ALL, "chance+row": Corrections(True, True, False, False)}

# A z is only evidence when the statistic behind it has a central limit. The projection onto
# one orbit's component is zero in every game that does not visit that set, so at a low
# reach it is non-zero in a handful of games and its sample standard error means nothing:
# measured during the build, the orbit picked by largest gradient alone was visited **once
# in 200 games** and reported a confident z of -5.86 for a subset that is provably unbiased
# (23 visits at 2,000 games gave z -1.18, and 232 at 20,000 gave -1.13). So the projection
# orbit is chosen among sets the profile actually reaches, and the check is skipped, loudly,
# below a minimum number of contributing games.
REACH_FLOOR = 0.02  # of games, for an orbit to be worth projecting onto
MIN_VISITS = 30  # contributing games before a z is checked rather than only printed


def check(ok: bool, what: str, stochastic: bool = False) -> str:
    STOCHASTIC[0] += stochastic
    if not ok:
        FAILS.append(what)
    return "ok" if ok else "XX"


def z_of(diff: float, se: float) -> float:
    if se < 1e-15:
        return 0.0 if abs(diff) <= 1e-9 else math.copysign(math.inf, diff)
    return diff / se


def profile(tab: dict, learner: str, column: str, seat: int):
    return (tab[learner], tab[column]) if seat == 0 else (tab[column], tab[learner])


def strategies(tree: CompiledTree, iterations: int) -> tuple[dict, dict]:
    rs = tree.rs
    rnd, heu, uni = PlayFirstRandom(rs), Heuristic(rs), UniformLegal(rs)
    tab = {a.name: (L1, store.load_or_tabulate(tree, L1, a)) for a in (rnd, heu, uni)}
    nash = store.load_or_solve_nash(tree, iterations, "cfr+")
    tab["nash"] = (L1, nash.avg)
    print(f"   nash: cfr+ {nash.iterations} its, NashConv {100 * nash.final.nashconv:.3f} pts")
    agents = {"heuristic": heu, "random": rnd, "uniform_legal": uni}
    return tab, agents


def choices(tree: CompiledTree, g, seat: int, n: int = 8):
    """Orbits a differentiable, non-trivial choice actually rests on, largest gradient
    first, with the per-orbit exact gradient.

    Two ways this selection goes vacuous, both of which it did while being written:

    - a set with a single action orbit has `sigma == 1` there and a gradient that is
      identically zero, and those sets carry most of the reach -- so ranking candidates by
      reach picks only those and the finite-difference check passes on `0.0 == 0.0`, which
      is step 9.5e's vacuous-test trap one level over;
    - demanding *full* support on the set leaves **nothing** on a free-play ruleset, where
      `PlayFirstRandom` puts zero on `Draw` wherever a play exists. Only the perturbed
      orbit needs positive probability (`uno.compiled.gradient.finite_difference`)."""
    k = tree.keyings[L1]
    n_orbits = k.info_orbit_start[1:] - k.info_orbit_start[:-1]
    ex = orbit_gradient(k, g.logit)
    live = np.nonzero(
        (k.info_player[k.orbit_info] == seat)
        & (g.reach[k.orbit_info] > 1e-9)
        & (n_orbits[k.orbit_info] > 1)
        & (np.bincount(k.ia_orbit, weights=g.sigma, minlength=k.n_orbits) > 0)
    )[0]
    return sorted((int(o) for o in live), key=lambda o: -abs(ex[o]))[:n], ex


def exact_section(tree: CompiledTree, tab: dict) -> dict:
    """The exact gradient, and the exact mean of the sampled estimator per subset."""
    k = tree.keyings[L1]
    print("\n== 1. the exact policy gradient, and the exact mean of the sampled estimator under each subset of corrections")
    print("   (logit gradient: d/d theta of a softmax over the set's actions, summed within action orbits -- the parameters of a symmetric learner)")
    picked = {}
    for learner, column in MATCHUPS:
        for seat in (0, 1):
            s0, s1 = profile(tab, learner, column, seat)
            g = exact_gradient(tree, tab[learner], tab[column], seat)
            live, ex = choices(tree, g, seat)
            norm = float(np.linalg.norm(g.logit))
            # the orbit section 2 projects onto: the largest gradient among the sets the
            # profile actually reaches often enough for a z to mean anything (REACH_FLOOR)
            reached = [o for o in live if g.reach[k.orbit_info[o]] >= REACH_FLOOR]
            picked[(learner, column, seat)] = (g, (reached or live or [None])[0])

            print(f"\n   learner {learner} vs {column}, seat {seat}: value {g.value:+.6f}, |gradient| {norm:.6f}")
            ok_sum = check(float(np.abs(k.info_sum(g.logit)).max()) < 1e-12, f"{learner} vs {column} seat {seat}: the logit gradient does not sum to zero per set")
            print(f"      max |per-set sum| {np.abs(k.info_sum(g.logit)).max():.1e} (a gradient is a direction inside the simplex)  {ok_sum}")

            # the independent check: a central finite difference of `evaluate`
            ok_live = check(bool(live) and float(np.abs(ex[live]).max()) > 1e-6, f"{learner} vs {column} seat {seat}: no orbit with a real choice and a non-zero gradient (the finite-difference check would be vacuous)")
            if live:
                fd = finite_difference(tree, tab[learner], tab[column], seat, live)
                rel = float(np.abs(fd - ex[live]).max() / max(float(np.abs(ex[live]).max()), 1e-12))
                ok_fd = check(rel < 1e-4, f"{learner} vs {column} seat {seat}: finite difference disagrees with the exact gradient (rel {rel:.2e})")
                print(f"      finite difference on {len(live)} orbits with a real choice, |gradient| {np.abs(ex[live]).min():.2e}-{np.abs(ex[live]).max():.2e}: max rel err {rel:.1e}  {ok_fd} {ok_live}")

            nus = {"exact": exact_values(tree, s0, s1)}
            nus["l1 set values"] = l1_set_values(tree, s0, s1, nus["exact"])
            if L2 in tree.keyings:
                nus["exact + noise 0.1"] = noisy(tree, nus["exact"], 0.1, seed=0)
            print(f"      {'value function':20s} {'R^2':>7s} {'corrections':20s} {'max |mean - exact|':>19s} {'|mean| / |exact|':>17s} {'prefix |max|':>13s}")
            for vname, nu in nus.items():
                r2 = outcome_r2(tree, s0, s1, nu)
                for name, c in (*KEEPS.items(), *ROW.items()):
                    m = estimator_mean(tree, tab[learner], tab[column], seat, nu, c)
                    dev = float(np.abs(m.logit - g.logit).max())
                    rel = float(np.linalg.norm(m.logit) / norm) if norm else float("nan")
                    marks = ""
                    if not c.row:
                        # zero bias, for any V and any subset sparing the row
                        marks = check(dev < 1e-12, f"{learner} vs {column} seat {seat} {vname} {c.label}: biased gradient, max dev {dev:.2e}")
                        # and not vacuous: the corrections themselves are large before centring
                        if c is not NONE:
                            marks += " " + check(float(np.abs(m.prefix).max()) > 1e-9, f"{learner} vs {column} seat {seat} {vname} {c.label}: prefix corrections are zero, so the cancellation check is vacuous")
                    elif vname == "exact":
                        # the prediction: with an exact V the row correction zeroes it outright
                        marks = check(float(np.abs(m.logit).max()) < 1e-12, f"{learner} vs {column} seat {seat} exact V {c.label}: the row correction left a gradient, max {np.abs(m.logit).max():.2e}")
                    print(f"      {vname:20s} {r2:7.4f} {name:20s} {dev:19.2e} {rel:17.6f} {np.abs(m.prefix).max():13.3e}  {marks}")
            sys.stdout.flush()
    return picked


def decisions_for(k, records, c: Corrections, seat: int) -> Decisions:
    """The learner's decisions with the signal each game's corrected return gives them."""
    game, info, ia, signal = [], [], [], []
    for gi, terms, infos, ias in records:
        x = terms.estimate(c, seat) * (1.0 if seat == 0 else -1.0) if c.row else corrected_return(terms, seat, c)
        game += [gi] * len(infos)
        info += infos
        ia += ias
        signal += [x] * len(infos)
    return Decisions(np.array(game, dtype=np.int64), np.array(info, dtype=np.int64), np.array(ia, dtype=np.int64), np.array(signal, dtype=float), len(records))


def sampled_section(tree: CompiledTree, tab: dict, agents: dict, picked: dict, games: int) -> None:
    """Play games with the learner, form the corrected signal, compare the sampled gradient
    against the exact one through two pre-declared projections."""
    rs = tree.rs
    k = tree.keyings[L1]
    idx = build(tree, L2)
    seeds = seed_registry.dev_block(0, games)
    print(f"\n== 2. sampled: {games} dev games per matchup and seat; the sampled gradient against the exact one")
    print("   projections: onto the exact gradient (the direction a learner would step in), and onto one orbit's own component (a single parameter)")
    for learner, column in MATCHUPS:
        agent = CompiledAgent(tree, L1, tab[learner][1], name=f"learner_{learner}")
        col_agent = agents.get(column) or CompiledAgent(tree, L1, tab[column][1], name=f"column_{column}")
        for seat in (0, 1):
            g, top = picked[(learner, column, seat)]
            s0, s1 = profile(tab, learner, column, seat)
            model = TreeValues(idx, exact_values(tree, s0, s1))
            t0 = time.perf_counter()
            records = []
            n = 0
            for seed in seeds:
                pair = (agent, col_agent) if seat == 0 else (col_agent, agent)
                try:
                    trace = play_game(rs, seed, pair)
                except ValueError:
                    continue
                terms = trace_terms(model, seed, trace)
                infos, ias = own_decisions(idx, k, trace, seat)
                records.append((n, terms, infos, ias))
                n += 1
            dec = sum(len(r[2]) for r in records)
            print(f"\n   learner {learner} vs {column}, seat {seat}: {n} games, {dec} own decisions ({dec / max(n, 1):.1f} per game), {time.perf_counter() - t0:.1f}s")

            norm = float(np.linalg.norm(g.logit))
            dirs = {"exact gradient": g.logit / norm} if norm else {}
            if top is not None:
                e = np.zeros(k.n_ia)
                e[k.ia_orbit == top] = 1.0
                dirs[f"orbit {top}"] = e
            targets = {name: float(g.logit @ d) for name, d in dirs.items()}
            print(f"      {'corrections':20s} " + "  ".join(f"{name:>16s}: {'estimate':>21s} {'exact':>9s} {'z':>6s} {'var ratio':>9s} {'games':>6s}" for name in dirs))
            raw_var = {}
            for name, c in (*KEEPS.items(), *ROW.items()):
                d = decisions_for(k, records, c, seat)
                gh = sampled_gradient(k, d, tab[learner][1])
                cells, marks = [], ""
                for dname, direction in dirs.items():
                    xs = per_game(k, d, tab[learner][1], direction)
                    mean, se = float(xs.mean()), float(xs.std(ddof=1) / math.sqrt(len(xs)))
                    var = float(xs.var(ddof=1))
                    contributing = int(np.count_nonzero(xs))
                    if c is NONE:
                        raw_var[dname] = var
                    z = z_of(mean - targets[dname], se)
                    ratio = var / raw_var[dname] if raw_var.get(dname) else float("nan")
                    enough = contributing >= MIN_VISITS
                    cells.append(f"{dname:>16s}: {mean:+.6f} ± {1.96 * se:.6f} {targets[dname]:+9.6f} {z:+6.2f} {ratio:9.3f} {contributing:6d}")
                    if not enough:
                        # printed, never checked: see MIN_VISITS
                        marks += ".."
                    elif not c.row:
                        marks += check(abs(z) <= 3, f"sampled {learner} vs {column} seat {seat} {c.label} {dname}: z {z:+.2f}", True)
                    else:
                        # the prediction: the row correction drives the sampled gradient to zero
                        marks += check(abs(z_of(mean, se)) <= 3, f"sampled {learner} vs {column} seat {seat} {c.label} {dname}: the row-corrected gradient is not zero ({mean:+.6f}, z {z_of(mean, se):+.2f})", True)
                    # the two spellings of one sum must agree
                    check(abs(mean - float(gh @ direction)) < 1e-9, f"sampled {learner} vs {column} seat {seat} {c.label} {dname}: per-game projections disagree with the accumulated gradient")
                print(f"      {name:20s} " + "  ".join(cells) + f"  {marks}")
            print(f"      ('games' is how many of the {n} games contribute to that projection; '..' = fewer than {MIN_VISITS}, printed but not checked)")
            sys.stdout.flush()


def cost_section(deals: int) -> None:
    """What the corrections cost inside a training loop, on `FULL_UNO`, component by component.

    9.5c measured AIVAT's cost against a 2.6 ms rulebot pair and concluded the corrections
    cost more CPU than they save. A training step is a different regime on both sides: it
    already computes `V(s)` at every state it visits, and the sibling values the corrections
    need go through the network in one batch. So the comparison is not against the cost of
    *playing* a game but against the value calls the step makes anyway.

    Each part is **timed directly** rather than obtained by differencing the total, and the
    parts are then checked against the end-to-end time. That matters here: the first version
    of this section timed only the shipped path and the forward pass, and attributing the
    rest would have credited it to the network when in fact most of it is the engine
    building the sibling *states* -- which is the one part batching a network cannot
    touch."""
    import random as _random
    from pathlib import Path

    from uno.eval.critic import CriticValues
    from uno.eval.value_net import ValueNet, batch_state_features
    from uno.game.chance import draw_outcomes

    rs = rules.FULL_UNO
    print(f"\n== 3. cost on FULL_UNO, {deals} dev deals (heuristic vs random): the corrections against the V(s) calls a training step already makes")
    paths = [Path(store.CACHE) / "value_net" / f"FULL_UNO_heuristic_random_s{s}_n2_g20000_w128_v1.pt" for s in (0, 1)]
    if not all(p.exists() for p in paths):
        print(f"   no cached value network at {paths[0]}; run `uv run uno pimc_value aivat` first -- section skipped")
        return
    net = ValueNet.load(paths[0], rs)
    heu, rnd = Heuristic(rs), PlayFirstRandom(rs)
    seeds = seed_registry.dev_block(0, deals)

    traces, t_play = [], 0.0
    for seed in seeds:
        t0 = time.perf_counter()
        try:
            trace = play_game(rs, seed, (heu, rnd))
        except ValueError:
            continue
        t_play += time.perf_counter() - t0
        traces.append(trace)
    n = len(traces)
    own = sum(len(t.steps) for t in traces)

    # what the step itself pays: V at every state it visits, one batch per game
    t_step = 0.0
    for trace in traces:
        states = [st.state for st in trace.steps]
        t0 = time.perf_counter()
        net.values(batch_state_features(states))
        t_step += time.perf_counter() - t0

    # the shipped correction path, end to end
    model = CriticValues(net)
    t_shipped = 0.0
    for trace in traces:
        t0 = time.perf_counter()
        model.records(trace)
        t_shipped += time.perf_counter() - t0

    # and its three parts, each timed on its own, with the rows counted per correction:
    # they are wildly unequal, and a training loop can buy them separately
    t_build, rows, skipped, siblings = 0.0, 0, 0, []
    # Distinct rows per correction subset, never overlapping categories that a reader could
    # add up: the chance correction needs the *taken* action's outcomes at every decision,
    # an action correction needs every action of that mover's support, and the taken action
    # belongs to both -- so the subsets are counted as unions, one decision at a time.
    by = {"deal only": 0, "chance only": 0, "TRAINING (chance + opponent + deal)": 0, "every correction (+ own actions)": 0}
    for trace in traces:
        t0 = time.perf_counter()
        for i, st in enumerate(trace.steps):
            rng = _random.Random(f"{st.state.seed}:aivat:{i}")
            counts = {}
            for a in st.probs:
                outs = draw_outcomes(st.state, a, rng, 64)
                if outs is None:
                    skipped += 1
                    continue
                counts[a] = len(outs)
                rows += len(outs)
                siblings += [c for c, _ in outs if not c.terminal]
            opp = st.state.player == 1  # the row agent sits at seat 0 here
            by["chance only"] += counts.get(st.action, 0)
            by["TRAINING (chance + opponent + deal)"] += sum(v for a, v in counts.items() if opp or a == st.action)
            by["every correction (+ own actions)"] += sum(counts.values())
        t_build += time.perf_counter() - t0
    by["deal only"] = n  # Phi(s0), one row per game
    t0 = time.perf_counter()
    [net.features(s) for s in siblings]
    t_feat_row = time.perf_counter() - t0
    t0 = time.perf_counter()
    mat = batch_state_features(siblings)
    t_feat_batch = time.perf_counter() - t0
    t0 = time.perf_counter()
    net.values(mat)
    t_fwd = time.perf_counter() - t0

    per = lambda t: 1000 * t / n  # noqa: E731  ms per game
    floor = t_build + t_feat_batch + t_fwd
    parts = t_build + t_feat_row + t_fwd
    print(f"   {n} games, {own / n:.1f} decisions/game, {t_play / n * 1000:.2f} ms/game to play (both rulebots)")
    print(f"   the corrections need V at every chance outcome of every action of the support: {rows / own:.1f} rows/decision, {rows / n:.0f}/game")
    print(f"   against the {own / n:.1f} rows/game the step's own V(s) calls make -- a {rows / own:.0f}x row count, {skipped} decisions with no chance model")
    print("   and they are far from equal -- distinct value rows per game needed by each subset (the row agent at seat 0):")
    for label, count in by.items():
        note = "   <- Phi(s0): the shaping potential of uno.eval.shaping, one row a game" if label.startswith("deal") else ""
        print(f"      {label:36s} {count / n:9.1f}{note}")
    print(f"      {'what':46s} {'ms/game':>9s}")
    print(f"      {'a training step: V(s) at its own states (batched)':46s} {per(t_step):9.2f}")
    print(f"      {'the shipped corrections (CriticValues.records)':46s} {per(t_shipped):9.2f}")
    print(f"      {'  of which: building the sibling states':46s} {per(t_build):9.2f}   <- engine, does not batch")
    print(f"      {'  of which: features, row at a time':46s} {per(t_feat_row):9.2f}")
    print(f"      {'  of which: the forward pass':46s} {per(t_fwd):9.2f}")
    print(f"      {'the same features, vectorised':46s} {per(t_feat_batch):9.2f}   <- {t_feat_row / max(t_feat_batch, 1e-12):.1f}x")
    ok = check(abs(parts - t_shipped) <= 0.25 * t_shipped, f"cost: the parts ({per(parts):.1f} ms/game) do not account for the shipped path ({per(t_shipped):.1f})")
    print(f"      parts vs end to end: {per(parts):.2f} against {per(t_shipped):.2f} ms/game  {ok}")
    print(f"   => the corrections cost {t_shipped / max(t_step, 1e-12):.1f}x the step's own V calls as shipped, {floor / max(t_step, 1e-12):.1f}x with the features vectorised,")
    print(f"      and the floor is not the network: {100 * t_build / max(floor, 1e-12):.0f} % of that is the engine building {rows / n:.0f} sibling states per game.")
    print("   (9.5c's regime, for contrast: the corrections against *playing* a rulebot pair, where they dominated and the verdict was 'more CPU than it saves')")


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    name = argv[0] if len(argv) > 0 else "TOY"
    games = int(argv[1]) if len(argv) > 1 else 2000
    iterations = int(argv[2]) if len(argv) > 2 else 1000
    cost_deals = int(argv[3]) if len(argv) > 3 else 200
    rs = getattr(rules, name)

    tree = compile_tree(rs, (L1, L2), True, cache=store.CACHE)
    print(f"== {name}: {tree}")
    print(f"   target: {TRAINING.label} (uno.eval.target.TRAINING); seeds: dev, nothing selected and no performance number claimed")
    tab, agents = strategies(tree, iterations)
    picked = exact_section(tree, tab)
    sampled_section(tree, tab, agents, picked, games)
    if cost_deals:
        cost_section(cost_deals)
    print(f"\n== {len(FAILS)} failed checks ({STOCHASTIC[0]} stochastic checks, ~{0.0025 * STOCHASTIC[0]:.1f} expected to fail by chance)" + (": " + "; ".join(FAILS) if FAILS else ""))


if __name__ == "__main__":
    main()
