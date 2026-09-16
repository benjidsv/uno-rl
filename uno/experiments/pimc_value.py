"""Step 9.5c: PIMC rollout values as a teacher, a value network as AIVAT's V.

    uv run uno pimc_value check [states=400] [n=400]
    uv run uno pimc_value floor [deals=2000] [replays=32] [rulesets=TOY,TOY2,TOY2_SKIP,MID_12,MID_25,FULL_UNO]
    uv run uno pimc_value pilot [fit_games=2000]
    uv run uno pimc_value aivat [games=4000] [fit_games=20000] [n=2] [rulesets=TOY,TOY2,TOY2_SKIP,MID_12,MID_25,FULL_UNO] [all|first]
    uv run uno pimc_value mc [pairs=200] [n=16]

docs/spec.md section 24. Every check prints ok / XX; the last line counts the failures.

`check`: rollout values (`uno.eval.pimc`) against the exact tree. TOY, TOY2_SKIP and DUPS, pile and
ticket draws, heuristic vs random at both seats. At the decisions of EVAL games, `n` rollouts of the
state's value and of the taken action's pre-chance value against the exact node value and arc
expectation: pooled z and mean z^2 (a certain outcome must match exactly).

`floor`: the deal's share of the variance (`deal_floor`), the variance ratio AIVAT reaches with the
exact L2 values and every correction but the deal's. On the toys against the exact ratio (z); on
MID_12, MID_25, FULL_UNO for heuristic vs random and heuristic vs heuristic_col, with the deals an
exact V would need for 3 points at 3 sigma with 80 % power.

`pilot` (FIT and DEV seeds only, nothing reported is chosen on EVAL): FULL_UNO heuristic vs random,
seat 0. At a fixed rollout budget, rollouts per state n in {2, 8, 32} and network width in {64, 128,
256}; R^2 against the values on 300 DEV games valued with 128 rollouts per state.

`aivat`: per ruleset and matchup (`first`: heuristic vs random only), a critic and a value network
per profile. The network is fitted on the rollout targets of `fit_games` FIT games (`n` rollouts per
state). Then both run AIVAT on the same `games` EVAL pairs with every 9.5b check
(`uno.experiments.aivat_uno.evaluate_matchup`), plus the deal correction: exact on the toys, a mean
over DEAL_SEEDS elsewhere. On FULL_UNO heuristic vs random, the pre-declared criterion: at both
seats, the network's variance ratio (chance+both) has a 95 % CI entirely below the critic's and a point
value <= 0.5.

`mc`: rollout values used directly as V (`RolloutValues`). On TOY, n in {4, 16, 64}: mean against
exact, variance ratio against n. On FULL_UNO, `pairs` pairs: ratio and cost, next to raw, critic and
network on the same pairs, as CPU-seconds for 3 points at 3 sigma with 80 % power.
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from uno import store
from uno.agents.baseline import Heuristic, PlayFirstRandom
from uno.compiled.aivat import aivat_moments, arc_expectation, exact_values, outcome_r2
from uno.compiled.lookup import build as build_index
from uno.compiled.strategies import tabulate
from uno.compiled.tree import compile_tree
from uno.eval.aivat import ALL, NONE, Corrections, GameTerms, bootstrap_ratio, estimate_aivat, game_terms, rescored_return, seat_scores, stream_pairs
from uno.eval.critic import CriticValues
from uno.eval.pimc import RolloutValues, chunks, deal_floor, pmap, rollout_q, rollout_rng, rollout_value, teacher_targets
from uno.eval.play import DEAL_SEEDS, DEV_SEEDS, EVAL_SEEDS, FIT_SEEDS, play_from, play_game
from uno.eval.value_net import ValueNet, fit_value_net, state_features
from uno.experiments import aivat_uno as au
from uno.experiments.aivat import ruleset as toy_ruleset
from uno.game import rules
from uno.game.engine import new_game
from uno.game.observe import Level

L1, L2 = Level.L1, Level.L2
TOYS = ("TOY", "TOY_FULL", "TOY2", "TOY2_SKIP")
CHANCE_BOTH = au.CORRECTIONS["chance+both"]
WITH_DEAL = au.CORRECTIONS | {"+deal": ALL}
WIDTH = 128  # the pilot: 64 / 128 / 256 are within noise of each other
TEACHER_N = 2  # rollouts per state; at a fixed budget more states beat cleaner targets (the pilot)
FIT_GAMES = 20_000
DEAL_GAMES = 250_000
check = au.check


def ruleset(name: str):
    return toy_ruleset(name)


def pair_of(agents: dict, row: str, col: str, seat: int):
    return (agents[row], agents[col]) if seat == 0 else (agents[col], agents[row])


def zstats(zs: list[float]) -> tuple[float, float]:
    """(pooled z, mean z^2) of independent z-scores."""
    k = len(zs)
    return (sum(zs) / math.sqrt(k), sum(z * z for z in zs) / k) if k else (0.0, 1.0)


def toy_tab(tree, agents):
    return {a.name: (L1, store.load_or_tabulate(tree, L1, a)) for a in agents}


# -- check ------------------------------------------------------------------------------------


def check_section(states: int, n: int) -> None:
    print(f"== check: rollout values against the exact tree, {states} decisions of EVAL games per cell, {n} rollouts per value (z on the rollouts' own standard error)")
    for name in ("TOY", "TOY2_SKIP", "DUPS"):
        rs0 = ruleset(name)
        tree = compile_tree(rs0, (L1, L2), True, cache=store.CACHE)
        idx = build_index(tree, L2)
        tab = toy_tab(tree, (Heuristic(rs0), PlayFirstRandom(rs0)))
        for draws in ("pile", "tickets"):
            rs = replace(rs0, draws=draws)
            agents = (Heuristic(rs), PlayFirstRandom(rs))
            for seat in (0, 1):
                pair = agents if seat == 0 else agents[::-1]
                prof = (tab[pair[0].name], tab[pair[1].name])
                nu = exact_values(tree, *prof)
                alpha = arc_expectation(tree, nu)
                zv, zq, certain, exact_ok = [], [], 0, True
                t0 = time.perf_counter()
                for seed in EVAL_SEEDS:
                    if len(zv) + certain >= states:
                        break
                    try:
                        t = play_game(rs, seed, pair)
                    except ValueError:
                        continue
                    for i, st in enumerate(t.steps):
                        node, arcs = idx.arcs(st.state, (st.action,))
                        for (m, v), ex, out in (
                            (rollout_value(st.state, pair, n, rollout_rng(seed, (i, "v"))), nu[node], zv),
                            (rollout_q(st.state, st.action, pair, n, rollout_rng(seed, (i, "q"))), alpha[arcs[st.action]], zq),
                        ):
                            if v > 0:
                                out.append((m - ex) / math.sqrt(v))
                            else:
                                certain += out is zv
                                exact_ok &= abs(m - ex) < 1e-12
                cells = []
                for label, zs in (("value", zv), ("q", zq)):
                    pooled, chi = zstats(zs)
                    band = 3 * math.sqrt(2 / max(len(zs), 1))
                    ok_p = check(abs(pooled) <= 3, f"check {name} {draws} seat {seat} {label}: pooled z {pooled:+.2f}")
                    ok_c = check(abs(chi - 1) <= band, f"check {name} {draws} seat {seat} {label}: mean z^2 {chi:.3f}")
                    cells.append(f"{label}: {len(zs)} z, pooled {pooled:+.2f} {ok_p}, mean z^2 {chi:.3f} (1 +- {band:.3f}) {ok_c}")
                ok_e = check(exact_ok, f"check {name} {draws} seat {seat}: certain outcomes exact", False)
                print(f"   {name:9s} {draws:7s} heuristic at seat {seat}: " + "  ".join(cells) + f"  certain {certain} exact {ok_e}  ({time.perf_counter() - t0:.0f}s)")
                sys.stdout.flush()


# -- floor --------------------------------------------------------------------------------------


def floor_section(deals: int, replays: int, names: list[str]) -> None:
    print(f"\n== floor: the deal's share of the variance (between-deal variance of the value of the dealt hands), {replays} pile determinizations per deal, EVAL deals; score of the row")
    print(f"   {'ruleset':9s} {'matchup':26s} {'seat':4s} {'deals':>5s} {'score':>6s} {'Var(u)':>7s} {'share':>6s} {'95 % CI':>15s} {'deals 3 pts raw':>15s} {'exact V':>8s}  exact")
    for name in names:
        rs = ruleset(name)
        agents = au.agents_for(rs)
        toy = name in TOYS
        matchups = (("heuristic", "random"),) if toy else (("heuristic", "random"), ("heuristic", "heuristic_col"))
        tree = tab = None
        if toy:
            tree = compile_tree(rs, (L1,), True, cache=store.CACHE)
            tab = toy_tab(tree, (agents["heuristic"], agents["random"]))
        for row, col in matchups:
            for seat in (0, 1):
                pair = pair_of(agents, row, col, seat)
                t0 = time.perf_counter()
                f = deal_floor(rs, pair, EVAL_SEEDS[: min(deals, len(EVAL_SEEDS))], replays)
                score = ((f.mean if seat == 0 else -f.mean) + 1) / 2
                raw_deals = au.Z80**2 * (f.total / 4) / au.EDGE**2
                exact_deals = au.Z80**2 * (f.between / 4) / au.EDGE**2
                extra = ""
                if toy:
                    prof = (tab[pair[0].name], tab[pair[1].name])
                    nu = exact_values(tree, *prof)
                    raw = aivat_moments(tree, *prof, nu, NONE, seat).var
                    exact = aivat_moments(tree, *prof, nu, CHANCE_BOTH, seat).var / raw
                    sd = (f.share_ci[1] - f.share_ci[0]) / 3.92
                    z = (f.share - exact) / sd
                    extra = f"{exact:.4f} (z {z:+.2f}) {check(abs(z) <= 3, f'floor {name} seat {seat}: share {f.share:.4f} vs exact {exact:.4f} z {z:+.2f}')}; Var(u) exact {raw:.4f}"
                print(f"   {name:9s} {row + ' vs ' + col:26s} {seat:<4d} {f.deals:5d} {score:6.4f} {f.total:7.4f} {f.share:6.4f} [{f.share_ci[0]:.4f}, {f.share_ci[1]:.4f}] {raw_deals:15.0f} {exact_deals:8.0f}  {extra}  ({time.perf_counter() - t0:.0f}s)")
                sys.stdout.flush()


# -- pilot ------------------------------------------------------------------------------------


def pilot_section(fit_games: int) -> None:
    rs = rules.FULL_UNO
    agents = au.agents_for(rs)
    pair = (agents["heuristic"], agents["random"])
    t0 = time.perf_counter()
    val = teacher_targets(rs, pair, DEV_SEEDS[:300], 128, state_features)
    vnoise, vvar = float(val.noise.mean()), float(val.y.var())
    print(f"\n== pilot: FULL_UNO heuristic vs random seat 0, budget {8 * fit_games:,} game-rollouts x states; validation {len(val.y):,} DEV states at 128 rollouts (noise {vnoise:.5f}, var {vvar:.5f}; {time.perf_counter() - t0:.0f}s)")
    for n, games in ((2, 4 * fit_games), (8, fit_games), (32, fit_games // 4)):
        t0 = time.perf_counter()
        t = teacher_targets(rs, pair, FIT_SEEDS[:games], n, state_features)
        teach = time.perf_counter() - t0
        for width in (64, 128, 256):
            net, rep = fit_value_net(rs, t.x, t.y, t.game, t.noise, width=width)
            v = net.values(val.x)
            mse = float(np.mean((v - val.y) ** 2))
            r2v = 1 - (mse - vnoise) / (vvar - vnoise)
            on = val.on_path
            r2o = 1 - float(np.sum((val.u[on] - v[on]) ** 2) / np.sum((val.u[on] - val.u[on].mean()) ** 2))
            print(f"   n {n:2d}, {games:6,} games ({len(t.y):,} states, teacher {teach:.0f}s), width {width:3d}: {rep}; DEV R^2 vs values {r2v:.4f}, outcome R^2 {r2o:.4f}")
            sys.stdout.flush()


# -- aivat ------------------------------------------------------------------------------------


def net_path(name: str, row: str, col: str, seat: int, n: int, fit_games: int) -> Path:
    d = Path(store.CACHE) / "value_net"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{name}_{row}_{col}_s{seat}_n{n}_g{fit_games}_w{WIDTH}_v{store.CACHE_VERSION}.pt"


def fit_nets(name: str, rs, agents: dict, row: str, col: str, n: int, fit_games: int) -> list[ValueNet]:
    nets = []
    for s in (0, 1):
        path = net_path(name, row, col, s, n, fit_games)
        if path.exists():
            nets.append(ValueNet.load(path, rs))
            print(f"   {row} vs {col} seat {s}: network loaded from {path}")
            continue
        t0 = time.perf_counter()
        t = teacher_targets(rs, pair_of(agents, row, col, s), FIT_SEEDS[:fit_games], n, state_features)
        teach = time.perf_counter() - t0
        net, rep = fit_value_net(rs, t.x, t.y, t.game, t.noise, width=WIDTH)
        net.save(path)
        nets.append(net)
        on = t.on_path
        print(f"   {row} vs {col} seat {s}: teacher {len(t.y):,} states ({on.mean():.0%} on path, n {n}, mean noise {t.noise.mean():.5f}) in {teach:.0f}s; network {rep}")
        sys.stdout.flush()
    return nets


def mc_deal_mean(model, rs, seeds) -> tuple[float, float]:
    """Mean and standard error of the model's value at the first state over `seeds`' deals."""
    vals, rows = [], []
    for seed in seeds:
        try:
            rows.append(model.features(new_game(rs, seed)))
        except ValueError:
            continue
        if len(rows) == 50_000:
            vals.append(model.values(rows))
            rows = []
    if rows:
        vals.append(model.values(rows))
    v = np.concatenate(vals)
    return float(v.mean()), float(v.std(ddof=1) / math.sqrt(len(v)))


def criterion(name: str, row: str, col: str, terms_c, terms_n) -> None:
    print(f"   pre-declared criterion ({name} {row} vs {col}, chance+both): network ratio CI entirely below the critic's, and <= 0.5")
    for s in (0, 1):
        raw = seat_scores(terms_c[s], NONE, s)
        same = np.array_equal(raw, seat_scores(terms_n[s], NONE, s))
        ec, en = estimate_aivat(terms_c, CHANCE_BOTH), estimate_aivat(terms_n, CHANCE_BOTH)
        lo_c, hi_c = bootstrap_ratio(ec.scores[s], raw)
        lo_n, hi_n = bootstrap_ratio(en.scores[s], raw)
        rc, rn = ec.var_ratio(s), en.var_ratio(s)
        passed = hi_n < lo_c and rn <= 0.5
        print(
            f"      seat {s}: same raw games {check(same, f'{name} {row} vs {col} seat {s}: critic and network games differ', False)}  "
            f"critic {rc:.3f} [{lo_c:.3f}, {hi_c:.3f}]  network {rn:.3f} [{lo_n:.3f}, {hi_n:.3f}]  -> {'PASS' if passed else 'FAIL'}"
        )


def aivat_section(games: int, fit_games: int, n: int, names: list[str], first_only: bool) -> None:
    curve = []
    summary = []
    for name in names:
        rs = ruleset(name)
        print(f"\n== {name} (deck {rs.deck_size}, hand {rs.hand_size}, reshuffle {rs.reshuffle}, forced play {rs.forced_play})")
        agents = au.agents_for(rs)
        for row, col in au.MATCHUPS[:1] if first_only else au.MATCHUPS:
            t0 = time.perf_counter()
            critics = au.fit_critics(rs, agents, row, col, min(fit_games, 10_000))
            t_critic = time.perf_counter() - t0
            t0 = time.perf_counter()
            nets = fit_nets(name, rs, agents, row, col, n, fit_games)
            t_net = time.perf_counter() - t0
            tree, nus_c, deal_c = au.toy_values(name, rs, critics)
            _, nus_n, deal_n = au.toy_values(name, rs, nets)
            if tree is None:
                deal_c, deal_n = [], []
                for s in (0, 1):
                    for model, out in ((critics[s], deal_c), (nets[s], deal_n)):
                        m, se = mc_deal_mean(model, rs, DEAL_SEEDS[:DEAL_GAMES])
                        out.append(m)
                        print(f"   seat {s} {'critic' if model is critics[s] else 'network'}: deal mean over {DEAL_GAMES:,} DEAL_SEEDS deals {m:+.5f} (se {se:.5f} return, {50 * se:.3f} score points: a fixed error of every +deal estimate)")
            else:
                for s in (0, 1):
                    prof = tuple((L1, tabulate(tree, L1, a)) for a in pair_of(agents, row, col, s))
                    ex = exact_values(tree, *prof)
                    print(f"   seat {s} exact outcome R^2: critic {outcome_r2(tree, *prof, nus_c[s]):.4f}, network {outcome_r2(tree, *prof, nus_n[s]):.4f}, exact values {outcome_r2(tree, *prof, ex):.4f}")
            models_c = tuple(CriticValues(critics[s], deal_mean=deal_c[s]) for s in (0, 1))
            models_n = tuple(CriticValues(nets[s], deal_mean=deal_n[s]) for s in (0, 1))
            print("   -- critic")
            terms_c = au.evaluate_matchup(name, rs, agents, row, col, games, models_c, tree, nus_c, curve, t_critic, WITH_DEAL, "critic")
            print("   -- network")
            terms_n = au.evaluate_matchup(name, rs, agents, row, col, games, models_n, tree, nus_n, curve, t_net, WITH_DEAL, "network")
            for s in (0, 1):
                summary.append((name, f"{row} vs {col}", s, *(estimate_aivat(t, c).var_ratio(s) for t in (terms_c, terms_n) for c in (CHANCE_BOTH, ALL))))
            if name == "FULL_UNO" and (row, col) == ("heuristic", "random"):
                criterion(name, row, col, terms_c, terms_n)
    print("\n== transfer curve (chance+both): decisions per game, outcome R^2 over EVAL decisions, variance ratio; on the toys the exact ratio of the model and of the exact values")
    print(f"   {'ruleset':10s} {'matchup':38s} {'seat':4s} {'decisions':>9s} {'R^2':>6s} {'ratio':>6s} {'exact model':>11s} {'exact values':>12s}")
    for name, matchup, s, dec, r2v, ratio, exact_ratio in curve:
        ex = f"{exact_ratio[0]:11.3f} {exact_ratio[1]:12.3f}" if exact_ratio else f"{'':11s} {'':12s}"
        print(f"   {name:10s} {matchup:38s} {s:<4d} {dec:9.1f} {r2v:6.3f} {ratio:6.3f} {ex}")
    print("\n== summary: variance ratio AIVAT / raw, critic and network, without and with the deal correction")
    print(f"   {'ruleset':10s} {'matchup':28s} {'seat':4s} {'critic':>7s} {'+deal':>7s} {'network':>8s} {'+deal':>7s}")
    for name, matchup, s, c1, c2, n1, n2 in summary:
        print(f"   {name:10s} {matchup:28s} {s:<4d} {c1:7.3f} {c2:7.3f} {n1:8.3f} {n2:7.3f}")


# -- mc: rollouts as V -------------------------------------------------------------------------


def _mc_chunk(rs, agents, seeds, n: int, deal_mean: tuple) -> list[tuple[GameTerms, GameTerms]]:
    models = (RolloutValues(agents, n, deal_mean=deal_mean[0]), RolloutValues(agents[::-1], n, deal_mean=deal_mean[1]))
    return [g for g, _ in stream_pairs(rs, agents, seeds, models)]


def mc_terms(rs, agents, seeds, n: int, deal_mean=(None, None)):
    parts = pmap(_mc_chunk, [(rs, agents, c, n, deal_mean) for c in chunks(seeds, 64)])
    pairs = [g for p in parts for g in p]
    return [g0 for g0, _ in pairs], [g1 for _, g1 in pairs]


def mc_section(pairs: int, n_full: int) -> None:
    print(f"\n== mc: rollout values used directly as AIVAT's V")
    rs = rules.TOY
    tree = compile_tree(rs, (L1, L2), True, cache=store.CACHE)
    agents = au.agents_for(rs)
    pair = (agents["heuristic"], agents["random"])
    tab = toy_tab(tree, pair)
    seeds = EVAL_SEEDS[:pairs]
    profs = [(tab["heuristic"], tab["random"]), (tab["random"], tab["heuristic"])]
    for n in (4, 16, 64):
        t0 = time.perf_counter()
        terms = mc_terms(rs, pair, seeds, n)
        secs = time.perf_counter() - t0
        est = estimate_aivat(terms, CHANCE_BOTH)
        cells = []
        for s in (0, 1):
            nu = exact_values(tree, *profs[s])
            mean = aivat_moments(tree, *profs[s], nu, NONE, s).score(s)[0]
            floor = aivat_moments(tree, *profs[s], nu, CHANCE_BOTH, s).var / aivat_moments(tree, *profs[s], nu, NONE, s).var
            z = au.z_of(est.seat[s].mean - mean, est.seat[s].se)
            cells.append(f"seat {s}: mean {est.seat[s].mean:.4f} (exact {mean:.4f}, z {z:+.2f}) {check(abs(z) <= 3, f'mc TOY n {n} seat {s}: mean z {z:+.2f}')} ratio {est.var_ratio(s):.3f} (exact values {floor:.3f})")
        print(f"   TOY heuristic vs random, {len(terms[0])} pairs, n {n:2d} rollouts per value ({secs:.0f}s on every core): " + "  ".join(cells))
        sys.stdout.flush()

    rs = rules.FULL_UNO
    agents = au.agents_for(rs)
    row, col = "heuristic", "random"
    pair = (agents[row], agents[col])
    seeds = EVAL_SEEDS[:pairs]
    print(f"\n   FULL_UNO {row} vs {col}, {pairs} EVAL pairs; CPU-seconds for 3 points at 3 sigma with 80 % power = deals needed x seconds per pair (one core)")
    t0 = time.perf_counter()
    for seed in seeds:
        try:
            s0 = new_game(rs, seed)
        except ValueError:
            continue
        play_from(s0, seed, pair)
        play_from(s0, seed, pair[::-1])
    raw_pair = (time.perf_counter() - t0) / len(seeds)
    rows = []
    critics = au.fit_critics(rs, agents, row, col, 10_000)
    nets = [ValueNet.load(p, rs) if (p := net_path("FULL_UNO", row, col, s, TEACHER_N, FIT_GAMES)).exists() else None for s in (0, 1)]
    for label, models in (("critic", tuple(CriticValues(c) for c in critics)), ("network", tuple(CriticValues(m) for m in nets) if all(nets) else None)):
        if models is None:
            print("   (no cached network: run `uno pimc_value aivat` first)")
            continue
        t0 = time.perf_counter()
        terms = ([], [])
        for (g0, g1), _ in stream_pairs(rs, pair, seeds, models):
            terms[0].append(g0)
            terms[1].append(g1)
        rows.append((label, terms, (time.perf_counter() - t0) / len(terms[0])))
    t0 = time.perf_counter()
    terms = mc_terms(rs, pair, seeds, n_full)
    rows.append((f"rollouts n={n_full}", terms, (time.perf_counter() - t0) * 8 / len(terms[0])))  # wall clock on 8 processes -> one core
    for label, terms, per_pair in [("raw", rows[0][1], raw_pair)] + rows:
        c = NONE if label == "raw" else CHANCE_BOTH
        est = estimate_aivat(terms, c)
        cells = []
        for s in (0, 1):
            deals = au.Z80**2 * est.seat[s].var / au.EDGE**2
            cells.append(f"seat {s}: var {est.seat[s].var:.4f} ratio {est.var_ratio(s):.3f} deals {deals:5.0f} CPU-s {deals * per_pair:8.1f}")
        print(f"   {label:16s} {1000 * per_pair:9.1f} ms per pair  " + "  ".join(cells))


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    section = argv[0] if argv else "check"
    args = argv[1:]
    if section == "check":
        check_section(int(args[0]) if args else 400, int(args[1]) if len(args) > 1 else 400)
    elif section == "floor":
        names = args[2].split(",") if len(args) > 2 else ["TOY", "TOY2", "TOY2_SKIP", "MID_12", "MID_25", "FULL_UNO"]
        floor_section(int(args[0]) if args else 2000, int(args[1]) if len(args) > 1 else 32, names)
    elif section == "pilot":
        pilot_section(int(args[0]) if args else 2000)
    elif section == "aivat":
        games = int(args[0]) if args else len(EVAL_SEEDS)
        fit_games = int(args[1]) if len(args) > 1 else FIT_GAMES
        n = int(args[2]) if len(args) > 2 else TEACHER_N
        names = args[3].split(",") if len(args) > 3 else ["TOY", "TOY2", "TOY2_SKIP", "MID_12", "MID_25", "FULL_UNO"]
        aivat_section(games, fit_games, n, names, "first" in args[4:])
    elif section == "mc":
        mc_section(int(args[0]) if args else 200, int(args[1]) if len(args) > 1 else 16)
    else:
        raise SystemExit(f"unknown section {section!r}: check, floor, pilot, aivat, mc")
    print(f"\n== {len(au.FAILS)} failed checks ({au.STOCHASTIC[0]} checks, ~{0.0027 * au.STOCHASTIC[0]:.1f} expected to fail by chance)" + (": " + "; ".join(au.FAILS) if au.FAILS else ""))


if __name__ == "__main__":
    main()
