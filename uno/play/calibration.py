"""`uv run python -m uno.play.calibration CKPT [--games 1000] [--workers 4]`

Phase 0 of docs/play/plan.md: is `(v + 1) / 2` from a full checkpoint's critic a
calibrated win probability against each league member, and when in the game?

The checkpoint's own actor plays every league bot from both seats on two coupled
blocks of the DEV diagnostic partition (`uno.rl.core.manifest.diagnostic_panel`), with
the evaluator's keyed streams, so a worker split plays the same games as one process.
At every learner decision the critic is evaluated on the learner's L2 inputs, exactly
as the collector builds them. The critic was fitted to lambda-returns (gamma 1,
lambda 0.95) of the raw terminal return, whose fixed point is the expected terminal
return of the learner's policy against the league average; with draws near zero,
`(v + 1) / 2` is then that policy's win probability from this state. Per opponent it
is biased by construction (the opponent's identity is hidden), which is what this
measures.

Decision rule, declared before running (plan section 3): show the raw critic if every
per-opponent bin with >= 200 samples is within 0.05 of its empirical rate; else fit
a per-opponent Platt correction on block 0, check it with the same rule on block 1,
and show that; if even corrected it misses, show rollout values only.

Writes `.cache/play/calibration_<run>.npz` (per-decision records) and `.json` (the
report and any Platt parameters the demo loads).
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

CACHE = Path(".cache/play")
BINS = 10
MIN_BIN = 200
TOLERANCE = 0.05
PHASES = ((0, 10, "turn < 10"), (10, 30, "turn 10-29"), (30, 10**9, "turn >= 30"))


def load_full(path: str | Path):
    """`(config, ActorPolicy, critic)` from a full training checkpoint, schema-checked."""
    from uno.rl.algo.policy import ActorPolicy
    from uno.rl.core.checkpoint import load
    from uno.rl.core.config_io import from_dict
    from uno.rl.env.features import spec
    from uno.rl.model import make_actor, make_critic

    payload = load(path)
    config = from_dict(payload["config"]).config
    if config.hash != payload["config_hash"]:
        raise ValueError("the checkpoint's config does not round-trip through config_io")
    rs = config.rs
    actor_spec, critic_spec = spec(rs, config.arm), spec(rs, "CRITIC")
    if payload["actor_schema"] != actor_spec.hash or payload["critic_schema"] != critic_spec.hash:
        raise ValueError("the checkpoint's feature schemas differ from the current extractors")
    actor = make_actor(actor_spec, config.model)
    actor.load_state_dict(payload["actor"])
    critic = make_critic(critic_spec, config.model)
    critic.load_state_dict(payload["critic"])
    critic.eval()
    return config, ActorPolicy(actor_spec, actor, config.optim.temperature), critic


def _play(args) -> dict[str, np.ndarray]:
    """Play a list of episodes; one row per learner decision plus one per game."""
    import torch

    from uno.game.engine import returns
    from uno.rl.algo.collect import LiveGame
    from uno.rl.core.streams import EVAL, uniform
    from uno.rl.env.features import extract, legal_mask, spec
    from uno.rl.eval.opponents import league
    from uno.rl.model import to_tensors

    path, episodes, training_seed, live_games = args
    torch.set_num_threads(1)
    config, actor, critic = load_full(path)
    rs = config.rs
    critic_spec = spec(rs, "CRITIC")
    opponents = league(rs)
    names = list(opponents)
    queue = list(episodes)
    active: list[LiveGame] = []
    rows: dict[str, list] = {k: [] for k in ("episode", "decision", "turn", "v")}
    games: dict[str, list] = {k: [] for k in ("episode", "opponent", "seat", "deal", "ret", "turns")}
    while True:
        while queue and len(active) < live_games:
            ep = queue.pop(0)
            active.append(LiveGame(rs, ep, opponents[ep.opponent], actor.new_memory(ep.seat)))
        if not active:
            break
        ready = []
        for g in active:
            g.advance_to_learner(training_seed)
            if g.terminal:
                games["episode"].append(g.episode.episode_id)
                games["opponent"].append(names.index(g.episode.opponent))
                games["seat"].append(g.episode.seat)
                games["deal"].append(g.episode.deal_seed)
                games["ret"].append(float(returns(g.state)[g.seat]))
                games["turns"].append(g.state.turn)
            else:
                ready.append(g)
        active = ready
        if not active:
            continue
        pairs = [g.observations() for g in active]
        obs_l1 = [p[0] for p in pairs]
        obs_actor = [p[1] if actor.spec.has_opponent else p[0] for p in pairs]
        evidences = [g.evidence for g in active]
        masks = np.stack([legal_mask(rs, o) for o in obs_l1])
        feats = extract(actor.spec, obs_actor, evidences if actor.spec.has_evidence else None)
        memories = [g.memory for g in active] if actor.recurrent else None
        events = actor.observe_events(memories, obs_l1) if actor.recurrent else None
        log_probs = actor.log_probs(feats, masks, None, events, memories)
        with torch.no_grad():
            v = critic(*to_tensors(extract(critic_spec, [p[1] for p in pairs], evidences))).numpy()
        us = np.fromiter((uniform(training_seed, EVAL, g.episode.episode_id, g.decision_index) for g in active),
                         dtype=np.float64, count=len(active))
        chosen = actor.sample(log_probs, us)
        for i, g in enumerate(active):
            rows["episode"].append(g.episode.episode_id)
            rows["decision"].append(g.decision_index)
            rows["turn"].append(g.state.turn)
            rows["v"].append(float(v[i]))
            g.apply(int(chosen[i]))
    out = {f"d_{k}": np.asarray(v) for k, v in rows.items()}
    out.update({f"g_{k}": np.asarray(v) for k, v in games.items()})
    return out


def collect(path: str, games: int, start: int, blocks: int, workers: int, live_games: int, training_seed: int):
    from uno.rl.core.manifest import diagnostic_panel
    from uno.rl.eval.opponents import league
    from uno.rl.core.config_io import from_dict
    from uno.rl.core.checkpoint import load

    rs = from_dict(load(path)["config"]).config.rs
    names = list(league(rs))
    manifest = diagnostic_panel(names, games * blocks, start=start)
    eps = list(manifest.episodes)
    chunks = [eps[i::workers * 4] for i in range(workers * 4)]
    parts = []
    t0 = time.time()
    with ProcessPoolExecutor(workers) as pool:
        for part in pool.map(_play, [(path, c, training_seed, live_games) for c in chunks]):
            parts.append(part)
            print(f"  {sum(len(p['g_ret']) for p in parts):,}/{len(eps):,} games  {time.time() - t0:.0f}s", flush=True)
    data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    data["names"] = np.asarray(names)
    data["block_edges"] = np.asarray([start + b * games for b in range(blocks + 1)])
    return data


# -- analysis ------------------------------------------------------------------


def per_decision(data) -> dict[str, np.ndarray]:
    """Join each decision to its game: opponent, seat, block, score."""
    order = np.argsort(data["g_episode"])
    idx = order[np.searchsorted(data["g_episode"], data["d_episode"], sorter=order)]
    edges = data["block_edges"]
    deal = data["g_deal"][idx]
    return {
        "opponent": data["g_opponent"][idx], "seat": data["g_seat"][idx],
        "block": np.searchsorted(edges, deal, side="right") - 1,
        "score": (data["g_ret"][idx] + 1.0) / 2.0, "turn": data["d_turn"], "v": data["d_v"],
        "p": np.clip((data["d_v"] + 1.0) / 2.0, 0.0, 1.0), "episode": data["d_episode"],
    }  # fmt: skip


def reliability(p: np.ndarray, y: np.ndarray) -> list[dict]:
    """Ten equal-width bins: count, mean prediction, empirical rate and its SE."""
    b = np.minimum((p * BINS).astype(int), BINS - 1)
    out = []
    for i in range(BINS):
        m = b == i
        n = int(m.sum())
        if n == 0:
            out.append({"bin": i, "n": 0})
            continue
        rate = float(y[m].mean())
        out.append({"bin": i, "n": n, "pred": float(p[m].mean()), "rate": rate,
                    "se": float(np.sqrt(max(rate * (1 - rate), 1e-12) / n))})
    return out


def worst_gap(bins: list[dict]) -> float:
    gaps = [abs(b["pred"] - b["rate"]) for b in bins if b["n"] >= MIN_BIN]
    return max(gaps) if gaps else 0.0


def platt_fit(v: np.ndarray, y: np.ndarray, iters: int = 50) -> tuple[float, float]:
    """`p = sigmoid(a v + b)` by Newton's method on the log loss (score in [0, 1])."""
    a, b = 1.0, 0.0
    X = np.stack([v, np.ones_like(v)], 1)
    w = np.array([a, b])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (p - y)
        H = (X * (p * (1 - p))[:, None]).T @ X + 1e-9 * np.eye(2)
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-10:
            break
    return float(w[0]), float(w[1])


def platt(v, a, b):
    return 1 / (1 + np.exp(-(a * v + b)))


def ev(returns: np.ndarray, values: np.ndarray) -> float:
    return 1.0 - float(np.var(returns - values)) / float(np.var(returns))


def analyse(data) -> dict:
    d = per_decision(data)
    names = [str(n) for n in data["names"]]
    report = {"decisions": int(len(d["v"])), "games": int(len(data["g_ret"])), "opponents": {}}
    raw_ok, platt_ok = True, True
    for oi, name in enumerate(names):
        m = d["opponent"] == oi
        p, y, v, turn, block = d["p"][m], d["score"][m], d["v"][m], d["turn"][m], d["block"][m]
        entry = {"decisions": int(m.sum()), "games": int((data["g_opponent"] == oi).sum()),
                 "win_rate": float(y.mean()), "mean_pred": float(p.mean())}
        entry["seats"] = {int(s): {"win_rate": float(y[d["seat"][m] == s].mean()),
                                   "mean_pred": float(p[d["seat"][m] == s].mean())} for s in (0, 1)}
        entry["reliability"] = reliability(p, y)
        entry["worst_gap"] = worst_gap(entry["reliability"])
        entry["phases"] = {}
        for lo, hi, label in PHASES:
            pm = (turn >= lo) & (turn < hi)
            bins = reliability(p[pm], y[pm])
            entry["phases"][label] = {"n": int(pm.sum()), "reliability": bins, "worst_gap": worst_gap(bins),
                                      "brier": float(np.mean((p[pm] - y[pm]) ** 2)),
                                      "brier_const": float(np.mean((y[pm].mean() - y[pm]) ** 2))}
        entry["brier"] = float(np.mean((p - y) ** 2))
        entry["brier_const"] = float(np.mean((y.mean() - y) ** 2))
        entry["ev"] = ev(2 * y - 1, v)
        raw_ok &= entry["worst_gap"] <= TOLERANCE
        # Platt on block 0, checked on block 1
        fit, check = block == 0, block == 1
        a, b = platt_fit(v[fit], y[fit])
        pc = platt(v[check], a, b)
        entry["platt"] = {"a": a, "b": b, "check_reliability": reliability(pc, y[check]),
                          "check_brier": float(np.mean((pc - y[check]) ** 2)),
                          "check_brier_raw": float(np.mean((p[check] - y[check]) ** 2)),
                          "check_brier_const": float(np.mean((y[fit].mean() - y[check]) ** 2))}
        entry["platt"]["check_worst_gap"] = worst_gap(entry["platt"]["check_reliability"])
        platt_ok &= entry["platt"]["check_worst_gap"] <= TOLERANCE
        report["opponents"][name] = entry
    report["decision"] = "raw" if raw_ok else ("platt" if platt_ok else "rollout")
    return report


def print_report(report: dict) -> None:
    print(f"\n{report['games']:,} games, {report['decisions']:,} learner decisions")
    for name, e in report["opponents"].items():
        print(f"\n== {name}: win {e['win_rate']:.3f} (seat0 {e['seats'][0]['win_rate']:.3f}, "
              f"seat1 {e['seats'][1]['win_rate']:.3f}), mean pred {e['mean_pred']:.3f}")
        print(f"   Brier {e['brier']:.4f} vs const {e['brier_const']:.4f}; EV {e['ev']:.3f}; "
              f"worst gap (n>={MIN_BIN}) {e['worst_gap']:.3f}")
        print("   bin    n       pred   rate   se")
        for b in e["reliability"]:
            if b["n"]:
                print(f"   {b['bin'] / BINS:.1f}  {b['n']:7d}  {b['pred']:.3f}  {b['rate']:.3f}  {b['se']:.3f}")
        for label, ph in e["phases"].items():
            print(f"   {label:>11}: n {ph['n']:7d}  Brier {ph['brier']:.4f} vs const {ph['brier_const']:.4f}  "
                  f"worst gap {ph['worst_gap']:.3f}")
        pl = e["platt"]
        print(f"   Platt a={pl['a']:.3f} b={pl['b']:.3f}: held-out worst gap {pl['check_worst_gap']:.3f}, "
              f"Brier {pl['check_brier']:.4f} (raw {pl['check_brier_raw']:.4f}, const {pl['check_brier_const']:.4f})")
    print(f"\ndecision: {report['decision']}")


def main(argv: list[str] | None = None) -> None:
    from uno.rl.core.manifest import DIAGNOSTIC_DEALS

    ap = argparse.ArgumentParser(prog="uno.play.calibration", description=__doc__.split("\n\n")[1])
    ap.add_argument("ckpt", help="full training checkpoint (e.g. logs/<run>/checkpoints/best.pt)")
    ap.add_argument("--games", type=int, default=1000, help="games per cell per block (two blocks)")
    ap.add_argument("--start", type=int, default=DIAGNOSTIC_DEALS.start + 50_000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--live", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0, help="evaluation stream seed")
    ap.add_argument("--name", default=None, help="output name (default: the run directory)")
    ap.add_argument("--reanalyse", action="store_true", help="reuse the saved records")
    args = ap.parse_args(argv)

    path = Path(args.ckpt)
    name = args.name or path.parent.parent.name
    CACHE.mkdir(parents=True, exist_ok=True)
    npz, report_path = CACHE / f"calibration_{name}.npz", CACHE / f"calibration_{name}.json"
    if args.reanalyse:
        data = dict(np.load(npz))
    else:
        data = collect(str(path), args.games, args.start, 2, args.workers, args.live, args.seed)
        np.savez_compressed(npz, **data)
    report = analyse(data)
    report.update({"checkpoint": str(path), "games_per_cell_per_block": args.games,
                   "deals": [int(x) for x in data["block_edges"]], "eval_seed": args.seed,
                   "rule": f"every per-opponent bin with n >= {MIN_BIN} within {TOLERANCE}"})
    report_path.write_text(json.dumps(report, indent=1))
    print_report(report)
    print(f"wrote {npz} and {report_path}")


if __name__ == "__main__":
    main()
