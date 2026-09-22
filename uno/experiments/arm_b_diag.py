"""Why does arm B not beat arm A? Headroom, usage and a paired re-evaluation.

Four measurements, none of which trains anything:

a. **headroom**: `MemoryHeuristic` against `Heuristic` on one coupled panel. This
   is the causal value of void memory for a hand-coded policy.
b. **opportunity**: how often a learner decision has a fresh void (`since <= 2`),
   and how often two legal plays differ in `uno.agents.rulebot._denial`.
c. **usage**: on B's own decision stream, the KL between B's policy and B with its
   voluntary-draw items removed (and with the whole evidence removed), plus the
   expected denial of the chosen action under A and under B.
d. **paired A-vs-B**: both actors on the same fresh diagnostic panel.

    uv run python -m uno.experiments.arm_b_diag --a logs/..A../checkpoints/final_actor.pt \\
        --b logs/..B../checkpoints/final_actor.pt --deals 1024 --usage-deals 256
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from uno.agents.baseline import Heuristic
from uno.agents.rulebot import MemoryHeuristic, _denial
from uno.eval.belief import voids
from uno.game.engine import returns
from uno.game.rules import Play
from uno.rl.algo.collect import LiveGame
from uno.rl.core import manifest as mf
from uno.rl.core.checkpoint import load_actor
from uno.rl.core.streams import EVAL, uniform
from uno.rl.env.evidence import Evidence
from uno.rl.env.features import extract, legal_mask
from uno.rl.eval.evaluate import baseline_panel, paired, play_panel
from uno.rl.eval.opponents import league

FRESH = 2  # a void is "fresh" while the opponent has drawn at most this many cards since


def _kl(p_log: np.ndarray, q_log: np.ndarray) -> np.ndarray:
    """Row-wise KL(p || q) over the legal (finite) entries."""
    p = np.exp(p_log)
    legal = np.isfinite(p_log)
    diff = np.where(legal, p_log - np.where(legal, q_log, 0.0), 0.0)
    return (p * diff).sum(1)


def usage(rs, a_agent, b_agent, opponents, deals: int, seed: int) -> dict:
    """Play B on a panel and score A, B and ablated B on every B decision."""
    a_pol, b_pol = a_agent.actor, b_agent.actor
    man = mf.diagnostic_panel(opponents, deals, start=mf.DIAGNOSTIC_DEALS.stop - deals)
    queue, active = list(man.episodes), []
    plays = [isinstance(x, Play) for x in rs.actions]
    acc = {k: [] for k in ("kl_items", "kl_all", "fresh", "spread", "den_a", "den_b", "den_max", "den_min", "kl_ab")}
    while queue or active:
        while queue and len(active) < 64:
            e = queue.pop(0)
            active.append(LiveGame(rs, e, opponents[e.opponent]))
        for g in active:
            g.advance_to_learner(seed)
        active = [g for g in active if not g.terminal]
        if not active:
            continue
        obs = [g.observations()[0] for g in active]
        masks = np.stack([legal_mask(rs, o) for o in obs])
        evs = [g.evidence for g in active]
        no_items = []
        for ev in evs:
            c = ev.copy()
            c.items = []
            no_items.append(c)
        empty = [Evidence(rs, g.seat ^ 1) for g in active]
        lb = b_pol.log_probs(extract(b_pol.spec, obs, evs), masks)
        lb_items = b_pol.log_probs(extract(b_pol.spec, obs, no_items), masks)
        lb_all = b_pol.log_probs(extract(b_pol.spec, obs, empty), masks)
        la = a_pol.log_probs(extract(a_pol.spec, obs), masks)
        acc["kl_items"] += list(_kl(lb, lb_items))
        acc["kl_all"] += list(_kl(lb, lb_all))
        acc["kl_ab"] += list(_kl(lb, la))
        for i, o in enumerate(obs):
            vs = voids(rs, o.public, o.seat ^ 1)
            fresh = [v for v in vs if v.since <= FRESH]
            den = np.zeros(rs.num_actions)
            legal_plays = [x for x in np.flatnonzero(masks[i]) if plays[x]]
            for x in legal_plays:
                act = rs.actions[x]
                colour = act.color if rs.card_types[act.card].is_wild else rs.card_types[act.card].color
                den[x] = _denial(rs, act.card, colour, vs)
            acc["fresh"].append(bool(fresh))
            spread = len(legal_plays) >= 2 and den[legal_plays].max() > den[legal_plays].min() + 1e-9
            acc["spread"].append(spread)
            if spread:
                pa, pb = np.exp(la[i]), np.exp(lb[i])
                pa, pb = pa / pa[legal_plays].sum(), pb / pb[legal_plays].sum()
                # expected denial conditioned on playing (draws excluded)
                acc["den_a"].append(float((pa[legal_plays] * den[legal_plays]).sum()))
                acc["den_b"].append(float((pb[legal_plays] * den[legal_plays]).sum()))
                acc["den_max"].append(float(den[legal_plays].max()))
                acc["den_min"].append(float(den[legal_plays].min()))
        us = np.array([uniform(seed, EVAL, g.episode.episode_id, g.decision_index) for g in active])
        chosen = b_pol.sample(lb, us)
        for i, g in enumerate(active):
            g.apply(int(chosen[i]))
    out = {k: np.asarray(v, dtype=float) for k, v in acc.items()}
    fresh, spread = out["fresh"].astype(bool), out["spread"].astype(bool)
    print(f"\n[b/c] usage on B's stream: {len(fresh)} decisions over {len(man.episodes)} games")
    print(f"  fresh void (since<={FRESH}):          {fresh.mean():.3f} of decisions")
    print(f"  >=2 plays with different denial:   {spread.mean():.3f} of decisions")
    for name, k in (("B vs B-without-void-items", "kl_items"), ("B vs B-without-any-evidence", "kl_all"), ("B vs A (same obs)", "kl_ab")):
        v = out[k]
        print(f"  KL {name:30s} mean {v.mean():.4f} | fresh {v[fresh].mean() if fresh.any() else float('nan'):.4f} | spread {v[spread].mean() if spread.any() else float('nan'):.4f} | no-fresh {v[~fresh].mean():.4f}")
    if len(out["den_a"]):
        d = out["den_b"] - out["den_a"]
        se = d.std(ddof=1) / np.sqrt(len(d))
        print(f"  E[denial | play], spread rows: A {out['den_a'].mean():.4f}  B {out['den_b'].mean():.4f}  "
              f"(B-A {d.mean():+.4f} +- {1.96 * se:.4f}; min {out['den_min'].mean():.4f} max {out['den_max'].mean():.4f})")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--deals", type=int, default=1024, help="deals per cell for the paired panels (a, d)")
    ap.add_argument("--usage-deals", type=int, default=256, help="deals per cell for the usage stream (b, c)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip", default="", help="comma list of parts to skip: a,c,d")
    args = ap.parse_args()
    skip = set(filter(None, args.skip.split(",")))

    a_agent, a_meta = load_actor(args.a)
    b_agent, b_meta = load_actor(args.b)
    rs = a_agent.rs
    opponents = league(rs, "rulebot")
    print(f"A: {a_meta['name']} update {a_meta['update']} ({a_meta['trained_decisions']} decisions)")
    print(f"B: {b_meta['name']} update {b_meta['update']} ({b_meta['trained_decisions']} decisions)")
    man = mf.diagnostic_panel(opponents, args.deals)

    if "d" not in skip:
        t = time.time()
        ra = play_panel(man, a_agent.actor, opponents, args.seed)
        rb = play_panel(man, b_agent.actor, opponents, args.seed)
        print(f"\n[d] paired panel, {args.deals} deals/cell ({time.time() - t:.0f}s)")
        print(f"  A {ra}\n  B {rb}\n  B - A: {paired(rb, ra)}")
        for c in man.cells:
            print(f"    {c[0]}/s{c[1]}: A {ra.cell_means[c]:.4f} B {rb.cell_means[c]:.4f}")
    if "a" not in skip:
        t = time.time()
        rh = baseline_panel(man, Heuristic(rs), opponents, rs, args.seed)
        rm = baseline_panel(man, MemoryHeuristic(rs), opponents, rs, args.seed)
        print(f"\n[a] headroom, {args.deals} deals/cell ({time.time() - t:.0f}s)")
        print(f"  Heuristic       {rh}\n  MemoryHeuristic {rm}\n  memory - none: {paired(rm, rh)}")
    if "c" not in skip:
        usage(rs, a_agent, b_agent, opponents, args.usage_deals, args.seed)


if __name__ == "__main__":
    main()
