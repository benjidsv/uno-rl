"""Step 9.5b: AIVAT on the toys, validated against the exact tree.

    uv run uno aivat [ruleset=TOY] [games=4000] [iterations=1000] [light]

`ruleset` is a name of `uno.game.rules.NAMED` or `DUPS` (`uno.experiments.pairing`'s
10-card ruleset with duplicate cards, +2 and +4: multi-card chance edges). Matchups (row
vs column): heuristic vs random, BR L1(random) vs random, nash vs random, nash vs
heuristic, the row at both seats. Estimator: `uno.eval.aivat`; exact moments and value
functions: `uno.compiled.aivat` (docs/spec.md section 23).

0. The deal model through AIVAT's deal term (`deal_section`): 5 disjoint ranges of 1,000,000
   seeds, mean of `nu(first state) - E_model nu` for three profiles, pooled z (a check).

1. Exact moments per correction subset, for the exact values of the profile, under the
   three scorings of exhaustion: variance ratio AIVAT / raw. The mean must be the profile's
   value for every subset and the variance zero with every correction.
2. Sampled, `games` held-out pairs (EVAL_SEEDS), pile draws under the three scorings and
   ticket draws under the rule:
   - constant V reproduces raw to the digit; exact V with every correction gives the root
     value on every game (deterministic checks);
   - exact V with the column uncorrected, and L1-set values with every correction: the
     AIVAT mean against the exact value (z on the sampled variance: AIVAT removes the deal
     and pile luck, so the reference is the whole-game value, not the on-seeds one), the
     sampled variance inside the 99.8 % bootstrap CI against the exact variance;
   - the paired AIVAT - raw difference has mean zero (z): independent of the chance
     model's deal probabilities, which the exact-V check cannot see;
   - the raw mean against the exact value under each scoring.
3. The imperfect-V curve: exact values plus noise per L2 set, L1-set values, and other
   profiles' values; outcome R^2 against the exact variance ratio, with and without the
   deal correction (full UNO has none). Sampled spot check of one noisy point.

`light` (TOY2_FULL): the lazy L1 tree only, DCFR Nash: sections 1 and 3 under the rule,
without the noisy family and without sampled games (no L2 keying). Every check prints
ok / XX; the last line counts the failures and the false alarms expected at |z| > 3.
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import replace

import numpy as np

from uno import store
from uno.agents.baseline import Heuristic, PlayFirstRandom
from uno.agents.tabular import CompiledAgent
from uno.compiled.aivat import TreeValues, aivat_moments, exact_values, l1_set_values, noisy, outcome_r2
from uno.compiled.lookup import build as build_index
from uno.compiled.solvers import best_response, evaluate
from uno.compiled.tree import CompiledTree, compile_tree
from uno.eval.aivat import ALL, NONE, Corrections, bootstrap_var, estimate_aivat, pair_terms, seat_scores
from uno.eval.estimate import Pairs, play_pairs
from uno.eval.play import EVAL_SEEDS
from uno.experiments.common import by_seat
from uno.game import rules
from uno.game.observe import Level

L1, L2 = Level.L1, Level.L2
FAILS: list[str] = []
STOCHASTIC = [0]  # checks that fail by chance at |z| > 3 (or outside a 99.8 % CI)

SCORINGS = ("hand", "draw", "stuck")
MATCHUPS = (("heuristic", "random"), ("br_l1(random)", "random"), ("nash", "random"), ("nash", "heuristic"))
SUBSETS = {
    "chance": Corrections(True, False, False, False),
    "chance+row": Corrections(True, True, False, False),
    "chance+both": Corrections(True, True, True, False),
    "all (+deal)": ALL,
}
CHANCE_ROW = Corrections(chance=True, row=True, column=False, deal=True)
NO_DEAL = Corrections(deal=False)
SIGMAS = (0.02, 0.05, 0.1, 0.2, 0.5)
FIT_GAMES = 20_000  # FIT_SEEDS games per profile for the toy-fitted critic (section 4)


def check(ok: bool, what: str, stochastic: bool = False) -> str:
    STOCHASTIC[0] += stochastic
    if not ok:
        FAILS.append(what)
    return "ok" if ok else "XX"


def ruleset(name: str):
    if name == "DUPS":
        from uno.experiments.pairing import DUPS

        return DUPS
    return getattr(rules, name)


def profile(tab: dict, row: str, col: str, seat: int):
    return (tab[row], tab[col]) if seat == 0 else (tab[col], tab[row])


def strategies(tree: CompiledTree, iterations: int, light: bool) -> tuple[dict, dict]:
    rs = tree.rs
    rnd, heu = PlayFirstRandom(rs), Heuristic(rs)
    tab = {a.name: (L1, store.load_or_tabulate(tree, L1, a)) for a in (rnd, heu)}
    agents = {"random": rnd, "heuristic": heu}
    tab["br_l1(random)"] = by_seat(tree, [best_response(tree, L1, tab["random"], seat) for seat in (0, 1)])
    agents["br_l1(random)"] = CompiledAgent(tree, L1, tab["br_l1(random)"][1], name="br_l1_random")
    variant = "dcfr" if light else "cfr+"
    nash = store.load_or_solve_nash(tree, iterations, variant)
    print(f"   nash: {variant} {nash.iterations} its, NashConv {100 * nash.final.nashconv:.4f} pts")
    tab["nash"] = (L1, nash.avg)
    agents["nash"] = nash.agent(tree)
    return tab, agents


def z_of(diff: float, se: float) -> float:
    if se < 1e-15:
        return 0.0 if abs(diff) <= 1e-9 else math.copysign(math.inf, diff)
    return diff / se


def exact_section(tree: CompiledTree, tab: dict, scorings) -> None:
    print("\n== 1. exact moments with the exact values of the profile: variance ratio AIVAT / raw per correction subset (row = first-named)")
    print(f"   {'scoring':7s} {'row vs column':26s} {'seat':4s} {'score':>7s} {'raw sd':>7s} " + " ".join(f"{k:>12s}" for k in SUBSETS) + "  checks")
    for exhaustion in scorings:
        rt = tree.rescored(exhaustion)
        for row, col in MATCHUPS:
            for seat in (0, 1):
                s0, s1 = profile(tab, row, col, seat)
                nu = exact_values(rt, s0, s1)
                value = evaluate(rt, s0, s1)[0]
                raw = aivat_moments(rt, s0, s1, nu, NONE, seat)
                ms = {k: aivat_moments(rt, s0, s1, nu, c, seat) for k, c in SUBSETS.items()}
                ok_mean = check(all(abs(m.mean - value) < 1e-9 for m in [raw, *ms.values()]), f"{exhaustion} {row} vs {col} seat {seat}: exact means == value")
                ok_zero = check(ms["all (+deal)"].var <= 1e-9 * max(raw.var, 1e-12), f"{exhaustion} {row} vs {col} seat {seat}: zero variance with every correction")
                mean, var = raw.score(seat)
                ratios = " ".join(f"{m.var / raw.var:12.4f}" for m in ms.values())
                print(f"   {exhaustion:7s} {row + ' vs ' + col:26s} {seat:<4d} {mean:7.4f} {math.sqrt(var):7.4f} {ratios}  {ok_mean} {ok_zero}")
        sys.stdout.flush()


def sampled_section(rs, tree: CompiledTree, tab: dict, agents: dict, games: int) -> dict[tuple[str, str], Pairs]:
    idx = build_index(tree, L2)
    seeds = range(EVAL_SEEDS.start, EVAL_SEEDS.start + games)
    kept: dict[tuple[str, str], Pairs] = {}
    print(f"\n== 2. sampled, {games} held-out pairs: AIVAT against the exact moments (score of the row; z on the sampled standard error; var inside the 99.8 % bootstrap CI)")
    for row, col in MATCHUPS:
        for draws in ("pile", "tickets"):
            t0 = time.perf_counter()
            pairs = play_pairs(replace(rs, draws=draws), (agents[row], agents[col]), seeds)
            if draws == "pile":
                kept[(row, col)] = pairs
            print(f"\n   {row} vs {col}, {draws} draws ({pairs.n} pairs, {time.perf_counter() - t0:.1f}s)")
            if draws == "pile":
                zero = np.zeros(tree.n)
                terms = pair_terms(pairs, (TreeValues(idx, zero), TreeValues(idx, zero)))
                same = all(np.array_equal(seat_scores(terms[s], c, s), np.array(pairs.seat_scores(s))) for s in (0, 1) for c in (NONE, ALL))
                print(f"      constant V == raw on every game: {check(same, f'{row} vs {col}: constant V reproduces raw')}")
            for exhaustion in SCORINGS if draws == "pile" else ("hand",):
                t1 = time.perf_counter()
                rt = tree.rescored(exhaustion)
                profs = [profile(tab, row, col, s) for s in (0, 1)]
                exact = [exact_values(rt, *p) for p in profs]
                l1v = [l1_set_values(rt, *p, e) for p, e in zip(profs, exact)]
                t_exact = pair_terms(pairs, (TreeValues(idx, exact[0], exhaustion), TreeValues(idx, exact[1], exhaustion)))
                t_l1 = pair_terms(pairs, (TreeValues(idx, l1v[0], exhaustion), TreeValues(idx, l1v[1], exhaustion)))
                dev = max(abs(t.estimate(ALL, s) - exact[s][0]) for s in (0, 1) for t in t_exact[s])
                ok = check(dev <= 1e-9, f"{row} vs {col} {draws} {exhaustion}: exact V, every correction == root value")
                print(f"      {exhaustion}: exact V with every correction, max |estimate - root| over games {dev:.1e}  {ok}  ({time.perf_counter() - t1:.1f}s)")
                for vname, terms, nus, c in (("exact", t_exact, exact, CHANCE_ROW), ("l1_set", t_l1, l1v, ALL)):
                    est = estimate_aivat(terms, c)
                    for s in (0, 1):
                        mean, var = aivat_moments(rt, *profs[s], nus[s], c, s).score(s)
                        e = est.seat[s]
                        z = z_of(e.mean - mean, e.se)
                        lo, hi = bootstrap_var(est.scores[s])
                        ok_m = check(abs(z) <= 3, f"{row} vs {col} {draws} {exhaustion} {vname} {c.label} seat {s}: mean z {z:+.2f}", True)
                        ok_v = check(lo <= var <= hi or (var < 1e-15 and hi < 1e-15), f"{row} vs {col} {draws} {exhaustion} {vname} {c.label} seat {s}: var {var:.5f} outside [{lo:.5f}, {hi:.5f}]", True)
                        print(
                            f"      {exhaustion:5s} {vname:6s} {c.label:22s} seat {s}: mean {e.mean:.4f} (exact {mean:.4f}, z {z:+.2f}) {ok_m}  "
                            f"var {e.var:.5f} (exact {var:.5f}, CI [{lo:.5f}, {hi:.5f}]) {ok_v}  ratio to raw {est.var_ratio(s):.3f}"
                        )
                est = estimate_aivat(t_l1, ALL)
                for s in (0, 1):
                    zd = z_of(est.diff[s].mean, est.diff[s].se)
                    raw_mean = aivat_moments(rt, *profs[s], exact[s], NONE, s).score(s)[0]
                    zr = z_of(est.raw[s].mean - raw_mean, est.raw[s].se)
                    ok_d = check(abs(zd) <= 3, f"{row} vs {col} {draws} {exhaustion} seat {s}: AIVAT - raw z {zd:+.2f}", True)
                    ok_r = check(abs(zr) <= 3, f"{row} vs {col} {draws} {exhaustion} seat {s}: raw mean z {zr:+.2f}", True)
                    print(f"      {exhaustion:5s} seat {s}: AIVAT - raw {est.diff[s].mean:+.4f} (z {zd:+.2f}) {ok_d}  raw {est.raw[s].mean:.4f} vs exact {raw_mean:.4f} (z {zr:+.2f}) {ok_r}")
                raw_pair = np.var((seat_scores(t_l1[0], NONE, 0) + seat_scores(t_l1[1], NONE, 1)) / 2, ddof=1)
                print(f"      {exhaustion:5s} mirrored (pair score): AIVAT var {est.mirrored.var:.5f}, raw mirrored var {raw_pair:.5f}")
            sys.stdout.flush()
    return kept


def curve_section(tree: CompiledTree, tab: dict, kept: dict[tuple[str, str], Pairs] | None) -> None:
    has_l2 = L2 in tree.keyings
    print("\n== 3. imperfect-V curve (rule scoring): outcome R^2 over visited decisions, exact variance ratio AIVAT / raw with every correction, and without the deal correction")
    print(f"   {'row vs column':26s} {'seat':4s} {'value function':24s} {'R^2':>7s} {'ratio all':>9s} {'no deal':>9s}")
    others = {"random vs random": (tab["random"], tab["random"]), "heuristic vs heuristic": (tab["heuristic"], tab["heuristic"])}
    other_nus = {k: exact_values(tree, *p) for k, p in others.items()}
    for row, col in MATCHUPS:
        for seat in (0, 1):
            s0, s1 = profile(tab, row, col, seat)
            exact = exact_values(tree, s0, s1)
            families = {"exact": exact}
            if has_l2:
                families |= {f"exact + noise {sigma}": noisy(tree, exact, sigma, seed=i) for i, sigma in enumerate(SIGMAS)}
            families["l1 set values"] = l1_set_values(tree, s0, s1, exact)
            families |= {f"other: {k}": v for k, v in other_nus.items()}
            raw = aivat_moments(tree, s0, s1, exact, NONE, seat).var
            for fname, nu in families.items():
                r2 = outcome_r2(tree, s0, s1, nu)
                all_ = aivat_moments(tree, s0, s1, nu, ALL, seat).var / raw
                nd = aivat_moments(tree, s0, s1, nu, NO_DEAL, seat).var / raw
                print(f"   {row + ' vs ' + col:26s} {seat:<4d} {fname:24s} {r2:7.4f} {all_:9.4f} {nd:9.4f}")
            sys.stdout.flush()
    if not kept or not has_l2:
        return
    idx = build_index(tree, L2)
    row, col = MATCHUPS[0]
    pairs = kept[(row, col)]
    sigma = 0.1
    nus = [noisy(tree, exact_values(tree, *profile(tab, row, col, s)), sigma, seed=SIGMAS.index(sigma)) for s in (0, 1)]
    est = estimate_aivat(pair_terms(pairs, (TreeValues(idx, nus[0]), TreeValues(idx, nus[1]))), NO_DEAL)
    print(f"\n   sampled spot check, {row} vs {col}, exact + noise {sigma}, no deal correction:")
    for s in (0, 1):
        mean, var = aivat_moments(tree, *profile(tab, row, col, s), nus[s], NO_DEAL, s).score(s)
        e = est.seat[s]
        z = z_of(e.mean - mean, e.se)
        lo, hi = bootstrap_var(est.scores[s])
        ok_m = check(abs(z) <= 3, f"spot check noise {sigma} seat {s}: mean z {z:+.2f}", True)
        ok_v = check(lo <= var <= hi, f"spot check noise {sigma} seat {s}: var outside CI", True)
        print(f"      seat {s}: mean {e.mean:.4f} (exact {mean:.4f}, z {z:+.2f}) {ok_m}  var {e.var:.5f} (exact {var:.5f}, CI [{lo:.5f}, {hi:.5f}]) {ok_v}")


def deal_section(tree: CompiledTree, tab: dict, n: int = 1_000_000, starts=(20_000_000, 21_000_000, 22_000_000, 23_000_000, 24_000_000)) -> None:
    """The deal model through AIVAT's deal term: over real deals, the mean of `nu(first state) -
    E_model nu` (nu the exact values of a profile) is zero iff the chance model's deal
    distribution is right for nu. It needs no play, only a lookup per seed, so millions of seeds
    cost seconds: at 5,000,000 seeds 3 sigma is ~0.04 score points on TOY, against ~0.24 for the
    step 9.5a check at 100,000 (step 9.5b, prompted by positive deal terms on one 200,000-seed range
    that disjoint ranges showed to be that range's luck)."""
    import math

    from uno.compiled.aivat import arc_expectation
    from uno.game.engine import new_game

    idx = build_index(tree, L2)
    rs = tree.rs
    profiles = {"heuristic vs random": (tab["heuristic"], tab["random"]), "random vs heuristic": (tab["random"], tab["heuristic"]), "random vs random": (tab["random"], tab["random"])}
    terms = {k: (lambda nu: (nu, float(arc_expectation(tree, nu)[0])))(exact_values(tree, *p)) for k, p in profiles.items()}
    print(f"\n== 0. deal model through the deal term, {len(starts)} disjoint ranges of {n:,} seeds (seat-0 score points; z on the deal term's own sd)")
    sums = {k: [0.0, 0.0, 0] for k in profiles}
    for start in starts:
        nodes = []
        for seed in range(start, start + n):
            try:
                nodes.append(idx.node(new_game(rs, seed)))
            except ValueError:
                continue
        nodes = np.array(nodes)
        cells = []
        for k, (nu, mean) in terms.items():
            x = (nu[nodes] - mean) / 2
            cells.append(f"{k} {100 * x.mean():+.3f} (z {x.mean() / (x.std(ddof=1) / math.sqrt(len(x))):+.2f})")
            s = sums[k]
            s[0] += float(x.sum())
            s[1] += float((x * x).sum())
            s[2] += len(x)
        print(f"   seeds {start:,}+: " + "  ".join(cells))
        sys.stdout.flush()
    for k, (sx, sxx, m) in sums.items():
        mean = sx / m
        sd = math.sqrt((sxx - m * mean * mean) / (m - 1))
        z = mean / (sd / math.sqrt(m))
        ok = check(abs(z) <= 3, f"deal term {k}: pooled z {z:+.2f}", True)
        print(f"   pooled {m:,} seeds, {k:20s}: {100 * mean:+.4f} pts, 3 sigma = {300 * sd / math.sqrt(m):.3f} pts, z {z:+.2f}  {ok}")


def critic_section(tree: CompiledTree, tab: dict, agents: dict, kept: dict[tuple[str, str], Pairs], fit_games: int) -> None:
    """The full-UNO code path on the toy: a critic fitted on FIT_SEEDS (`uno.eval.critic`), its
    alpha from `draw_outcomes` for played games, against the exact moments of the same critic
    read off the tree (`uno.compiled.aivat.critic_values`)."""
    from uno.compiled.aivat import arc_expectation, critic_values
    from uno.eval.critic import CriticValues, fit
    from uno.eval.play import FIT_SEEDS

    idx = build_index(tree, L2)
    rs = tree.rs
    print(f"\n== 4. toy-fitted critic ({fit_games} FIT_SEEDS games per profile), played path (draw_outcomes) against the exact moments of the same critic")
    for row, col in (MATCHUPS[0], MATCHUPS[3]):
        t0 = time.perf_counter()
        critics, nus, deal = [], [], []
        for s in (0, 1):
            pair = (agents[row], agents[col]) if s == 0 else (agents[col], agents[row])
            critic, rep = fit(rs, pair, FIT_SEEDS[:fit_games])
            nu = critic_values(tree, critic)
            critics.append(critic)
            nus.append(nu)
            deal.append(float(arc_expectation(tree, nu)[0]))
            p = profile(tab, row, col, s)
            print(
                f"   {row} vs {col} seat {s}: {rep.states:,} states, ridge {rep.lam:g} (holdout R^2 " + ", ".join(f"{k:g}: {v:.4f}" for k, v in rep.holdout_r2.items()) + f"), "
                f"exact R^2 {outcome_r2(tree, *p, nu):.4f} (exact values {outcome_r2(tree, *p, exact_values(tree, *p)):.4f})"
            )
        terms = pair_terms(kept[(row, col)], (CriticValues(critics[0], deal_mean=deal[0]), CriticValues(critics[1], deal_mean=deal[1])))
        skipped = sum(t.skipped_chance + t.skipped_action for ts in terms for t in ts)
        print(f"      played in {time.perf_counter() - t0:.1f}s; decisions without a chance model: {skipped} {check(skipped == 0, f'{row} vs {col}: critic skipped decisions on a toy')}")
        for c in (ALL, NO_DEAL, Corrections(True, True, False, False)):
            est = estimate_aivat(terms, c)
            for s in (0, 1):
                p = profile(tab, row, col, s)
                mean, var = aivat_moments(tree, *p, nus[s], c, s).score(s)
                raw = aivat_moments(tree, *p, nus[s], NONE, s).var / 4
                e = est.seat[s]
                z = z_of(e.mean - mean, e.se)
                lo, hi = bootstrap_var(est.scores[s])
                ok_m = check(abs(z) <= 3, f"critic {row} vs {col} {c.label} seat {s}: mean z {z:+.2f}", True)
                ok_v = check(lo <= var <= hi, f"critic {row} vs {col} {c.label} seat {s}: var {var:.5f} outside [{lo:.5f}, {hi:.5f}]", True)
                print(f"      {c.label:22s} seat {s}: mean {e.mean:.4f} (exact {mean:.4f}, z {z:+.2f}) {ok_m}  var {e.var:.5f} (exact {var:.5f}, CI [{lo:.5f}, {hi:.5f}]) {ok_v}  exact ratio {var / raw:.3f}, sampled {est.var_ratio(s):.3f}")
        sys.stdout.flush()


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    name = argv[0] if len(argv) > 0 else "TOY"
    games = int(argv[1]) if len(argv) > 1 else len(EVAL_SEEDS)
    iterations = int(argv[2]) if len(argv) > 2 else 1000
    light = "light" in argv[3:]
    rs = ruleset(name)
    tree = compile_tree(rs, (L1,) if light else (L1, L2), True, cache=store.CACHE)
    print(f"== {name}: {tree}")
    tab, agents = strategies(tree, iterations, light)
    if not light:
        deal_section(tree, tab)
    exact_section(tree, tab, ("hand",) if light else SCORINGS)
    kept = None if light else sampled_section(rs, tree, tab, agents, games)
    curve_section(tree, tab, kept)
    if not light:
        critic_section(tree, tab, agents, kept, FIT_GAMES)
    print(f"\n== {len(FAILS)} failed checks ({STOCHASTIC[0]} stochastic checks, ~{0.0025 * STOCHASTIC[0]:.1f} expected to fail by chance)" + (": " + "; ".join(FAILS) if FAILS else ""))


if __name__ == "__main__":
    main()
