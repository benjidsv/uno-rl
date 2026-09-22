"""How much of the cheating arm's gain survives when its opponent hand is *inferred*?

The CRITIC-as-actor run reads the opponent's true hand (L2). Here that actor is
played on a coupled DEV panel with the true hand replaced by hands sampled from
public information only, so the gap to the true-hand score is what inference
cannot recover and the gap to arm A is what it can -- with no training at all.

Each decision samples `candidates` hands uniformly from the unseen cards
(`uno.eval.search.unseen_cards`), weights them by a belief rung, resamples `k`
of them and plays the **average** of the actor's policies over those `k` hands,
so every input it sees is an ordinary in-distribution hand:

- `truth`   the real hand (the ceiling, reproduces the run's own evaluation);
- `prior`   uniform over the unseen cards: the actor's value of *no* inference;
- `voids`   hard void filter (`uno.eval.belief.satisfies`), what arm B's evidence proves;
- `soft`    `void_weight` at decay 0.1, the same constraint believed rather than enforced;
- `window`  voids times the likelihood of the opponent's trailing no-draw plays under
            the **league mixture** (identity unknown, as for a real learner);
- `window_true`  the same under the episode's real opponent: an upper bound that
            also leaks the opponent's identity.

This is a lower bound on what a trained belief arm could reach: the actor was
trained to trust an exact hand, not to hedge over a posterior.

    uv run python -m uno.experiments.critic_recovery --critic path/to/latest_actor.pt \\
        --a logs/..A../checkpoints/final_actor.pt --deals 256 --modes truth,prior,voids,window
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import replace

import numpy as np

from uno.eval.belief import Belief, satisfies, void_weight, voids
from uno.eval.search import unseen_cards
from uno.game.engine import returns
from uno.rl.algo.collect import LiveGame
from uno.rl.core import manifest as mf
from uno.rl.core.checkpoint import load_actor
from uno.rl.core.streams import EVAL, uniform
from uno.rl.env.features import extract, legal_mask
from uno.rl.eval.evaluate import PanelResult, paired, play_panel
from uno.rl.eval.opponents import league

MODES = ("truth", "prior", "voids", "soft", "window", "window_true")
SOFT_DECAY = 0.1


class Sampler:
    """Draws `k` opponent hands for one decision under one belief rung."""

    def __init__(self, rs, mode: str, opponents: dict, candidates: int, k: int) -> None:
        self.rs, self.mode, self.candidates, self.k = rs, mode, candidates, k
        # Belief is only used for its exact window likelihood; one per league member.
        self.windows = {name: Belief(rs, "window", opponent=agent) for name, agent in opponents.items()}
        self.fallbacks = 0
        self.decisions = 0
        self.ess = []

    def hands(self, obs, opponent_name: str, rng: random.Random) -> list[tuple[int, ...]]:
        rs, k_opp = self.rs, obs.opp_hand_size
        unseen, _ = unseen_cards(rs, obs)
        self.decisions += 1
        if k_opp == 0:
            return [()] * self.k
        cands = [tuple(sorted(rng.sample(unseen, k_opp))) for _ in range(self.candidates)]
        if self.mode == "prior":
            return cands[: self.k]
        vs = voids(rs, obs.public, obs.seat ^ 1)
        if self.mode == "soft":
            w = np.array([void_weight(rs, h, vs, SOFT_DECAY) for h in cands])
        else:
            w = np.array([1.0 if not vs or satisfies(rs, h, vs) else 0.0 for h in cands])
        if self.mode in ("window", "window_true"):
            names = [opponent_name] if self.mode == "window_true" else list(self.windows)
            lik = np.zeros(len(cands))
            for i in np.flatnonzero(w):
                for name in names:
                    b = self.windows[name]
                    _, window = b._constraints(obs, obs.seat ^ 1)
                    lik[i] += b._window_likelihood(obs, cands[i], window) if window else 1.0
            if (w * lik).sum() > 0.0:
                w = w * lik
            else:  # no candidate explains the window: keep the void-consistent ones
                self.fallbacks += 1
        total = w.sum()
        if total <= 0.0:
            self.fallbacks += 1
            return cands[: self.k]
        p = w / total
        self.ess.append(1.0 / float((p**2).sum()))
        idx = rng.choices(range(len(cands)), weights=p, k=self.k)
        return [cands[i] for i in idx]


def play(man, critic, opponents, mode: str, seed: int, candidates: int, k: int, live: int = 64) -> tuple[PanelResult, Sampler]:
    rs, sp = critic.rs, critic.spec
    sampler = Sampler(rs, mode, opponents, candidates, k)
    by_cell = {c: {} for c in man.cells}
    queue, active = list(man.episodes), []
    while queue or active:
        while queue and len(active) < live:
            e = queue.pop(0)
            active.append(LiveGame(rs, e, opponents[e.opponent]))
        for g in active:
            g.advance_to_learner(seed)
        for g in [g for g in active if g.terminal]:
            v = float(returns(g.state)[g.seat])
            by_cell[(g.episode.opponent, g.episode.seat)][g.episode.deal_seed] = (v + 1.0) / 2.0
        active = [g for g in active if not g.terminal]
        if not active:
            continue
        pairs = [g.observations() for g in active]
        masks = np.stack([legal_mask(rs, l1) for l1, _ in pairs])
        if mode == "truth":
            rows, reps = [l2 for _, l2 in pairs], 1
        else:
            rows, reps = [], k
            for g, (l1, l2) in zip(active, pairs):
                rng = random.Random(f"{seed}:{g.episode.episode_id}:{g.decision_index}:{mode}")
                rows += [replace(l2, opp_hand=h) for h in sampler.hands(l1, g.episode.opponent, rng)]
        evs = [g.evidence for g in active for _ in range(reps)]
        lp = critic.log_probs(extract(sp, rows, evs), np.repeat(masks, reps, axis=0))
        probs = np.exp(lp).reshape(len(active), reps, -1).mean(1)
        with np.errstate(divide="ignore"):
            avg = np.where(masks, np.log(probs), -np.inf)
        us = np.array([uniform(seed, EVAL, g.episode.episode_id, g.decision_index) for g in active])
        chosen = critic.sample(avg, us)
        for i, g in enumerate(active):
            g.apply(int(chosen[i]))
    deals = np.asarray(sorted({e.deal_seed for e in man.episodes}), dtype=np.int64)
    scores = {c: np.asarray([by_cell[c][d] for d in deals]) for c in man.cells}
    return PanelResult(man.cells, scores, deals), sampler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--critic", required=True, help="actor export of the CRITIC-arm (L2) run")
    ap.add_argument("--a", help="arm A actor export, for the paired baseline")
    ap.add_argument("--b", help="arm B actor export, for the paired baseline")
    ap.add_argument("--deals", type=int, default=256)
    ap.add_argument("--modes", default="truth,prior,voids,window")
    ap.add_argument("--candidates", type=int, default=256)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    modes = args.modes.split(",")
    if bad := set(modes) - set(MODES):
        raise SystemExit(f"unknown modes {bad}; choose from {MODES}")

    critic, meta = load_actor(args.critic)
    if not critic.actor.spec.has_opponent:
        raise SystemExit("--critic must be a CRITIC-arm actor export")
    rs = critic.rs
    opponents = league(rs, "rulebot")
    man = mf.diagnostic_panel(opponents, args.deals)
    print(f"critic actor: {meta['name']} update {meta['update']} ({meta['trained_decisions']} decisions); "
          f"{args.deals} deals/cell, k={args.k}, candidates={args.candidates}", flush=True)

    refs = {}
    for name, path in (("A", args.a), ("B", args.b)):
        if path:
            agent, _ = load_actor(path)
            refs[name] = play_panel(man, agent.actor, opponents, args.seed)
            print(f"{name:12s} {refs[name].mean:.4f} +- {1.96 * refs[name].standard_error:.4f}", flush=True)
    results = {}
    for mode in modes:
        t = time.time()
        res, s = play(man, critic.actor, opponents, mode, args.seed, args.candidates, args.k)
        results[mode] = res
        extra = ""
        if mode not in ("truth", "prior"):
            ess = np.median(s.ess) if s.ess else float("nan")
            extra = f"  fallback {s.fallbacks / max(s.decisions, 1):.3f}, median ESS {ess:.1f}"
        line = f"{mode:12s} {res.mean:.4f} +- {1.96 * res.standard_error:.4f} ({time.time() - t:.0f}s){extra}"
        for name, ref in refs.items():
            line += f" | vs {name} {paired(res, ref)}"
        if "truth" in results and mode != "truth":
            line += f" | vs truth {paired(res, results['truth'])}"
        print(line, flush=True)


if __name__ == "__main__":
    main()
