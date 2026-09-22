"""Stage 5 exact-toy gates: gradient agreement, then the five-seed learning gates.

`uv run uno rl_toy gradient [RULESET] [OPPONENT] [GAMES]`
`uv run uno rl_toy tabular  [RULESET] [OPPONENT] [SEEDS] [CAP]`
`uv run uno rl_toy neural   [RULESET] [OPPONENT] [SEEDS] [CAP]`
`uv run uno rl_toy all`

docs/rl/pipeline-plan.md section 13, stage 5. Three things are measured, and they
are different claims:

**gradient** -- do the learner's *raw* sampled gradients agree with the exact
expected gradient of the same parameters, within game-level uncertainty? The exact
side is `uno.rl.exact.gradient`: the derivative of the exact game value on the
compiled quotient tree with respect to the network's own weights. The sampled side
is the collector's own rollouts. Agreement is tested by projection, with a standard
error over games, plus the exact identity that PPO's gradient at the collection
policy equals the score-function reference on the same advantages.

**tabular** -- the retained gate. Five training seeds, each within 0.5 score points
of its same-seat fixed-opponent best response at both seats, within a declared cap.
The actor is one logit per action orbit, so the representation question is removed
and what is being tested is the sampled optimizer and the pipeline around it.

**neural** -- the same sampled optimization against exact-gradient optimization of
the *same architecture*, plus the BR gap of both. Exact optimization is an achieved
comparator, not a certified architecture optimum, and the BR gap is reported rather
than asserted to be small.

Every number here is an exact toy value or a sampled estimate with its uncertainty
stated. Nothing in this module measures full UNO.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from uno import store
from uno.agents.baseline import Heuristic, PlayFirstRandom
from uno.game.observe import Level
from uno.game.rules import NAMED
from uno.rl.eval import exact
from uno.rl.env import features as F
from uno.rl.algo import ppo
from uno.rl.algo.collect import collect
from uno.rl.core.config import Optim
from uno.rl.core.manifest import DealSchedule, build
from uno.rl.model import make_actor, make_critic
from uno.rl.algo.policy import ActorPolicy
from uno.rl.core.streams import INIT, torch_seed
from uno.rl.eval.tabular import TabularActor, TabularPolicy

OPPONENTS = {"random": PlayFirstRandom, "heuristic": Heuristic}
WIDTH = 64  # toy-sized: the gates test the optimizer and the contract, not capacity

# Frozen after the bounded diagnostic below, before the five-seed gates were run.
TABULAR_CAP = 400_000  # learner decisions per seed
TABULAR_TOLERANCE = 0.005  # 0.5 score points, the retained gate
NEURAL_CAP = 100_000
# The exact-gradient comparator is run to its own achieved point rather than at a
# matched step count: one exact gradient forwards every information set of the tree
# (~1 s on TOY), so a step-for-step match with the sampled arm would cost hours and
# buy nothing the plan asks for. It is an achieved comparator, not an optimum.
EXACT_STEPS = 250
EXACT_LR = 3e-3
GRADIENT_Z = 3.0  # |z| on a projection with a game-level standard error


def _opponent(rs, name: str):
    return OPPONENTS[name](rs)


def _sigma(rs, opponent):
    return store.load_or_tabulate(exact.tree_for(rs), Level.L1, opponent)


def _rollout(rs, policy, critic_spec, opponents, seat, seed, games, set_id_fn=None, start_id=0):
    manifest, _ = build(0, list(opponents), DealSchedule(seed), games, start_id, seats=(seat,))
    return collect(manifest, policy, critic_spec, opponents, seed, 0, live_games=min(64, games), set_id_fn=set_id_fn)


# -- the gradient diagnostic ------------------------------------------------------


def gradient_check(ruleset: str = "TOY", opponent: str = "heuristic", games: int = 4000, seat: int = 0, seed: int = 0) -> dict:
    rs = NAMED[ruleset]
    opp = _opponent(rs, opponent)
    opp_sigma = _sigma(rs, opp)
    sp, spc = F.spec(rs, F.ARM_B), F.spec(rs, F.ARM_CRITIC)
    torch.manual_seed(torch_seed(seed, INIT))
    from uno.rl.core.config import RelationalModel
    actor, critic = make_actor(sp, RelationalModel(width=WIDTH)), make_critic(spc, RelationalModel(width=WIDTH))

    t0 = time.perf_counter()
    g = exact.gradient(rs, sp, actor, opp_sigma, seat)
    exact_time = time.perf_counter() - t0
    sigma = exact.strategy(rs, sp, actor, seat=seat)
    exact_score = exact.value(rs, sigma, opp_sigma, seat)

    rollout, stats = _rollout(rs, ActorPolicy(sp, actor), spc, {opp.name: opp}, seat, seed, games)
    per_game, mean = exact.sampled_gradient(actor, rollout)
    projections = exact.agreement(per_game, g.flat)

    sampled_score = float(((rollout.terminal_return + 1.0) / 2.0).mean())
    se_score = float(((rollout.terminal_return + 1.0) / 2.0).std(ddof=1) / np.sqrt(rollout.n_games))

    # PPO's gradient at the collection policy is the score-function gradient on the
    # same advantages: an exact identity, not a statistical one.
    values = ppo.critic_values(critic, rollout.critic)
    adv = ppo.advantages(rollout, values)
    optim = Optim(entropy_coef=0.0, normalize_advantages=False)
    ppo_grad = torch.cat([t.reshape(-1) for t in ppo.weighted_gradient(actor, rollout, adv, [np.arange(rollout.n_decisions)], optim)])
    # `weighted_gradient` differentiates the *loss*, which is minus the objective; the
    # objective's gradient at r = 1 is the decision-mean score-function gradient
    # weighted by the same advantages, matching `sampled_gradient`'s aggregate.
    _, ref_mean = exact.sampled_gradient(actor, rollout, baseline=values)
    reference = -ref_mean
    ppo_identity = float((ppo_grad - reference).norm() / max(float(reference.norm()), 1e-12))

    print(f"\n== gradient agreement: {ruleset} seat {seat} vs {opp.name}, {rollout.n_games:,} games, {rollout.n_decisions:,} learner decisions")
    print(f"   exact score {exact_score:.4f}   sampled score {sampled_score:.4f} +- {1.96 * se_score:.4f} (95% CI)")
    print(f"   exact gradient norm {float(g.flat.norm()):.5f} (computed in {exact_time:.1f}s over the compiled tree)")
    for p in projections:
        print(f"   {p}")
    print(f"   PPO gradient vs score-function reference at collection weights: relative error {ppo_identity:.2e}")
    worst = max(abs(p.z) for p in projections)
    ok = worst <= GRADIENT_Z and ppo_identity < 1e-5 and abs(sampled_score - exact_score) <= 1.96 * se_score + 1e-9
    print(f"   -> {'PASS' if ok else 'FAIL'} (worst |z| {worst:.2f} against the predeclared {GRADIENT_Z})")
    return {
        "ruleset": ruleset, "opponent": opp.name, "seat": seat, "games": rollout.n_games, "decisions": rollout.n_decisions,
        "exact_score": exact_score, "sampled_score": sampled_score, "score_se": se_score,
        "projections": [{"direction": p.direction, "exact": p.exact, "sampled": p.sampled, "se": p.standard_error, "z": p.z} for p in projections],
        "ppo_identity_relative_error": ppo_identity, "pass": bool(ok), "seconds": stats.seconds,
    }  # fmt: skip


# -- the learning gates -------------------------------------------------------------


def _train_tabular(rs, opponent, seat: int, seed: int, cap: int, games_per_iteration: int = 256, verbose: bool = False) -> dict:
    tree = exact.tree_for(rs)
    torch.manual_seed(torch_seed(seed, INIT))
    actor = TabularActor(tree)
    policy = TabularPolicy(tree, actor)
    spc = F.spec(rs, F.ARM_CRITIC)
    critic = Critic(spc, WIDTH)
    optim = Optim(actor_lr=0.05, critic_lr=1e-3, minibatch=512, actor_epochs=4, critic_epochs=2)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=optim.actor_lr, betas=optim.betas, eps=optim.eps)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=optim.critic_lr, betas=optim.betas, eps=optim.eps)
    opponents = {opponent.name: opponent}
    opp_sigma = _sigma(rs, opponent)
    schedule = DealSchedule(seed)
    decisions = update = next_id = 0
    history = []
    t0 = time.perf_counter()
    while decisions < cap:
        manifest, next_id = build(update, list(opponents), schedule, games_per_iteration, next_id, seats=(seat,))
        rollout, _ = collect(manifest, policy, spc, opponents, seed, update, live_games=64, set_id_fn=policy.index)
        ppo.update(actor, critic, actor_opt, critic_opt, rollout, optim, seed, update)
        decisions += rollout.n_decisions
        update += 1
        if update % 20 == 0 or decisions >= cap:
            v = exact.value(rs, actor.sigma(), opp_sigma, seat)
            history.append((decisions, v))
            if verbose:
                print(f"      {decisions:>8,} dec  exact {v:.4f}")
    final = exact.value(rs, actor.sigma(), opp_sigma, seat)
    return {"seed": seed, "seat": seat, "decisions": decisions, "updates": update, "exact": final, "history": history, "seconds": time.perf_counter() - t0}


def tabular_gate(ruleset: str = "TOY", opponent: str = "heuristic", seeds: int = 5, cap: int = TABULAR_CAP, verbose: bool = False) -> dict:
    rs = NAMED[ruleset]
    opp = _opponent(rs, opponent)
    opp_sigma = _sigma(rs, opp)
    br = {seat: exact.br_score(rs, opp_sigma, seat) for seat in (0, 1)}
    print(f"\n== tabular gate: {ruleset} vs {opp.name}, {seeds} seeds, cap {cap:,} learner decisions/seed/seat")
    print(f"   BR L1 ceiling: seat 0 {br[0]:.4f}, seat 1 {br[1]:.4f}; gate is within {100 * TABULAR_TOLERANCE:.1f} score points at both seats")
    rows = []
    for seed in range(seeds):
        for seat in (0, 1):
            r = _train_tabular(rs, opp, seat, seed, cap, verbose=verbose)
            r["br"] = br[seat]
            r["gap"] = br[seat] - r["exact"]
            r["pass"] = r["gap"] <= TABULAR_TOLERANCE
            rows.append(r)
            print(f"   seed {seed} seat {seat}: exact {r['exact']:.4f}  BR {br[seat]:.4f}  gap {100 * r['gap']:+.2f} pts  "
                  f"({r['updates']} updates, {r['seconds']:.0f}s)  {'ok' if r['pass'] else 'FAIL'}")
    by_seed = {s: all(r["pass"] for r in rows if r["seed"] == s) for s in range(seeds)}
    ok = all(by_seed.values())
    print(f"   -> {'PASS' if ok else 'FAIL'}: {sum(by_seed.values())}/{seeds} seeds within {100 * TABULAR_TOLERANCE:.1f} points at both seats")
    return {"ruleset": ruleset, "opponent": opp.name, "cap": cap, "tolerance": TABULAR_TOLERANCE, "br": br, "rows": rows, "pass": bool(ok)}


def _train_neural(rs, opponent, seat: int, seed: int, cap: int, games_per_iteration: int = 256, exact_mode: bool = False, verbose: bool = False, exact_steps: int = EXACT_STEPS, exact_lr: float = EXACT_LR) -> dict:
    sp, spc = F.spec(rs, F.ARM_B), F.spec(rs, F.ARM_CRITIC)
    torch.manual_seed(torch_seed(seed, INIT))
    from uno.rl.core.config import RelationalModel
    actor, critic = make_actor(sp, RelationalModel(width=WIDTH)), make_critic(spc, RelationalModel(width=WIDTH))
    optim = Optim(minibatch=512, actor_epochs=4, critic_epochs=2)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=exact_lr if exact_mode else optim.actor_lr, betas=optim.betas, eps=optim.eps)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=optim.critic_lr, betas=optim.betas, eps=optim.eps)
    opponents = {opponent.name: opponent}
    opp_sigma = _sigma(rs, opponent)
    schedule = DealSchedule(seed)
    decisions = update = next_id = 0
    history = []
    t0 = time.perf_counter()
    if exact_mode:
        # Exact-gradient optimization of the same architecture: ascent on the exact
        # game value, with no sampling at all. The comparator, not an optimum.
        steps = exact_steps
        for stepped in range(steps):
            g = exact.gradient(rs, sp, actor, opp_sigma, seat)
            actor_opt.zero_grad(set_to_none=True)
            for p, grad in zip(actor.parameters(), g.parameters):
                p.grad = -grad  # ascent on J
            torch.nn.utils.clip_grad_norm_(actor.parameters(), optim.actor_grad_norm)
            actor_opt.step()
            if (stepped + 1) % 25 == 0 or stepped + 1 == steps:
                v = exact.value(rs, exact.strategy(rs, sp, actor, seat=seat), opp_sigma, seat)
                history.append((stepped + 1, v))
                if verbose:
                    print(f"      exact step {stepped + 1:>5d}  exact {v:.4f}")
        update = steps
    else:
        while decisions < cap:
            manifest, next_id = build(update, list(opponents), schedule, games_per_iteration, next_id, seats=(seat,))
            rollout, _ = collect(manifest, ActorPolicy(sp, actor), spc, opponents, seed, update, live_games=64)
            ppo.update(actor, critic, actor_opt, critic_opt, rollout, optim, seed, update)
            decisions += rollout.n_decisions
            update += 1
            if update % 20 == 0 or decisions >= cap:
                v = exact.value(rs, exact.strategy(rs, sp, actor, seat=seat), opp_sigma, seat)
                history.append((decisions, v))
                if verbose:
                    print(f"      {decisions:>8,} dec  exact {v:.4f}")
    final = exact.value(rs, exact.strategy(rs, sp, actor, seat=seat), opp_sigma, seat)
    return {"seed": seed, "seat": seat, "decisions": decisions, "updates": update, "exact": final, "history": history, "seconds": time.perf_counter() - t0}


def neural_gate(ruleset: str = "TOY", opponent: str = "heuristic", seeds: int = 5, cap: int = NEURAL_CAP, verbose: bool = False) -> dict:
    rs = NAMED[ruleset]
    opp = _opponent(rs, opponent)
    opp_sigma = _sigma(rs, opp)
    br = {seat: exact.br_score(rs, opp_sigma, seat) for seat in (0, 1)}
    print(f"\n== neural gate: {ruleset} vs {opp.name}, {seeds} seeds, cap {cap:,} learner decisions/seed/seat")
    print(f"   sampled optimization against exact-gradient optimization of the same architecture; BR gap reported, not gated")
    rows = []
    for seed in range(seeds):
        for seat in (0, 1):
            sampled = _train_neural(rs, opp, seat, seed, cap, verbose=verbose)
            exact_run = _train_neural(rs, opp, seat, seed, cap, exact_mode=True, verbose=verbose)
            row = {
                "seed": seed, "seat": seat, "sampled": sampled["exact"], "exact_optimized": exact_run["exact"], "br": br[seat],
                "sampled_gap": br[seat] - sampled["exact"], "exact_gap": br[seat] - exact_run["exact"],
                "sampled_vs_exact": sampled["exact"] - exact_run["exact"], "seconds": sampled["seconds"] + exact_run["seconds"],
                "history": sampled["history"], "exact_history": exact_run["history"],
            }  # fmt: skip
            rows.append(row)
            print(f"   seed {seed} seat {seat}: sampled {row['sampled']:.4f}  exact-opt {row['exact_optimized']:.4f}  "
                  f"BR {br[seat]:.4f}  sampled-exact {100 * row['sampled_vs_exact']:+.2f} pts  BR gap {100 * row['sampled_gap']:+.2f} pts ({row['seconds']:.0f}s)")
    deficits = [r["sampled_vs_exact"] for r in rows]
    print(f"   -> sampled minus exact-gradient optimization: mean {100 * float(np.mean(deficits)):+.2f} pts, worst {100 * float(np.min(deficits)):+.2f} pts")
    return {"ruleset": ruleset, "opponent": opp.name, "cap": cap, "br": br, "rows": rows}


def main(argv: list[str]) -> None:
    what = argv[0] if argv else "all"
    out_dir = Path("logs/rl_toy")
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    if what in ("gradient", "all"):
        ruleset = argv[1] if len(argv) > 1 and what == "gradient" else "TOY"
        opponent = argv[2] if len(argv) > 2 and what == "gradient" else None
        games = int(argv[3]) if len(argv) > 3 and what == "gradient" else 4000
        names = [opponent] if opponent else ["random", "heuristic"]
        results["gradient"] = [gradient_check(ruleset, n, games, seat) for n in names for seat in (0, 1)]
    if what in ("tabular", "all"):
        ruleset = argv[1] if len(argv) > 1 and what == "tabular" else "TOY"
        opponent = argv[2] if len(argv) > 2 and what == "tabular" else None
        seeds = int(argv[3]) if len(argv) > 3 and what == "tabular" else 5
        cap = int(argv[4]) if len(argv) > 4 and what == "tabular" else TABULAR_CAP
        names = [opponent] if opponent else ["random", "heuristic"]
        results["tabular"] = [tabular_gate(ruleset, n, seeds, cap) for n in names]
    if what in ("neural", "all"):
        ruleset = argv[1] if len(argv) > 1 and what == "neural" else "TOY"
        opponent = argv[2] if len(argv) > 2 and what == "neural" else None
        seeds = int(argv[3]) if len(argv) > 3 and what == "neural" else 5
        cap = int(argv[4]) if len(argv) > 4 and what == "neural" else NEURAL_CAP
        names = [opponent] if opponent else ["random", "heuristic"]
        results["neural"] = [neural_gate(ruleset, n, seeds, cap) for n in names]
    if what not in ("gradient", "tabular", "neural", "all"):
        raise SystemExit(f"unknown subcommand {what!r}; expected gradient, tabular, neural or all")
    path = out_dir / f"rl_toy_{what}_{int(time.time())}.json"
    path.write_text(json.dumps(results, indent=2, default=float))
    print(f"\n   wrote {path}")
