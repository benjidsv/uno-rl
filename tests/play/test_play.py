"""The play demo's analysis boundaries (docs/play/plan.md sections 2 and 7).

A coach must give exactly what an independent player fed the same decisions gives,
and no analysis call -- a copy, a rollout -- may advance a recurrent memory. Reveal
mode changes what the page is sent, never the bot's inputs.
"""

from __future__ import annotations

import random

import numpy as np
import torch

from tests.rl.test_recurrent import small_recurrent
from uno.agents.baseline import Heuristic
from uno.game.engine import legal_actions
from uno.game.rules import NAMED
from uno.play.analysis import rollout_values
from uno.play.opponents import AgentPlayer, NeuralPlayer, Opponent
from uno.play.server import Coach, Game
from uno.rl.algo.policy import ActorPolicy

FULL = NAMED["FULL_UNO"]


def recurrent_policy(seed=0) -> ActorPolicy:
    actor = small_recurrent(FULL, seed=seed)
    return ActorPolicy(actor.spec, actor)


def heuristic_opponent() -> Opponent:
    agent = Heuristic(FULL)
    return Opponent("h", "Heuristic", "rulebot", lambda seat: AgentPlayer(agent, seat))


def neural_opponent(policy) -> Opponent:
    return Opponent("n", "net", "checkpoint", lambda seat: NeuralPlayer(policy, seat), actor=policy)


def play(game: Game, rng: random.Random, analyse: bool, max_human: int = 40) -> None:
    """Uniform human moves; with `analyse`, poke copies and rollouts between them."""
    game.run_bot()
    for k in range(max_human):
        if game.state.terminal:
            break
        if analyse and k < 6:
            point = game.history[-1]
            rollout_values([(point["state"], game.human, tuple(point["players"]))], 2, k)
            for _, p in game.coaches:
                p.copy().policy(game.state)
        game.act(rng.choice(legal_actions(game.state)))


def test_a_recurrent_coach_equals_an_independent_player_and_analysis_does_not_advance_it():
    pol = recurrent_policy()
    game = Game(FULL, heuristic_opponent(), 0, 12345, [Coach("rec", "L1", pol)])
    play(game, random.Random(0), analyse=True)
    human = [p for p in game.history if p["who"] == "you"]
    assert len(human) >= 5
    fresh = NeuralPlayer(pol, 0)
    for p in human:
        assert fresh.policy(p["state"]) == p["coach"]["rec"]
    # the live memory is exactly the fresh player's after the same decisions
    assert torch.equal(game.coaches[0][1].memory.state, fresh.memory.state)


def test_reveal_changes_what_is_sent_not_the_bot():
    pol = recurrent_policy(seed=1)
    runs = []
    for reveal in (False, True):
        game = Game(FULL, neural_opponent(pol), 1, 777, [], reveal)
        steps = game.run_bot()
        rng = random.Random(3)
        for _ in range(30):
            if game.state.terminal:
                break
            steps += game.act(rng.choice(legal_actions(game.state)))
        runs.append((game, steps))
    (g0, s0), (g1, s1) = runs
    assert [s["action"] for s in s0] == [s["action"] for s in s1]
    assert [s["prob"] for s in s0] == [s["prob"] for s in s1]
    bot = [s for s in s1 if s["who"] == "bot"]
    assert bot and all("policy" in s for s in bot)
    assert not any("policy" in s for s in s0)
    assert all(s["view"]["opp_hand"] is None for s in s0 if not s["view"]["over"])


def test_rollout_values_are_keyed_and_bounded():
    pol = recurrent_policy(seed=2)
    game = Game(FULL, heuristic_opponent(), 0, 99, [Coach("rec", "L1", pol)])
    game.run_bot()
    point = game.history[-1]
    args = [(point["state"], 0, tuple(point["players"]))]
    m1, se1 = rollout_values(args, 8, 5)
    m2, se2 = rollout_values(args, 8, 5)
    assert np.array_equal(m1, m2) and np.array_equal(se1, se2)
    assert 0.0 <= m1[0] <= 1.0


def _decisions(pol, n_games=3):
    from uno.play.inspect import sample_decisions

    return sample_decisions(pol, games=n_games, keep=60)


def test_the_trace_reproduces_the_logits_and_the_decomposition_is_exact():
    from uno.play.inspect import _body, _inputs, blocks, decompose, project, stack, token_attribution, trace
    from uno.rl.env.features import ARM_B, spec
    from uno.rl.model import make_actor
    from uno.rl.core.config import RelationalModel

    torch.manual_seed(0)
    b = make_actor(spec(FULL, ARM_B), RelationalModel(width=32, blocks=2))
    for pol in (recurrent_policy(seed=4), ActorPolicy(b.spec, b)):
        ds = [d for d in _decisions(pol) if len(d.probs) > 1]
        f, e, m = stack(ds, pol.actor)
        tr = trace(pol.actor, f, e, m)  # raises if the logits differ
        assert tr["_error"] <= 1e-5
        if pol.recurrent:
            # memory after the traced decision equals the live player's next state
            assert tr["memory"].shape[1:] == (pol.spec.layout.n_tokens, pol.actor.memory)
        d = ds[0]
        f, e, m = stack([d], pol.actor)
        body = _body(pol.actor)
        with torch.no_grad():
            h, hw = project(body.encoder, *_inputs(pol.actor, f, e, m))
            layers, cc, gc = blocks(body.encoder, h, hw)
        a, bb = sorted(d.probs, key=d.probs.get, reverse=True)[:2]
        dec = decompose(body, *layers[-1], cc, gc, a, bb)  # raises unless exact
        logp = np.log([d.probs[a], d.probs[bb]])
        assert abs(dec["total"] - (logp[0] - logp[1])) < 1e-4
        att = token_attribution(body, h, hw, a, bb, np.zeros((pol.spec.layout.n_tokens, body.width), np.float32))
        assert abs(att["ig_sum"] - att["completeness"]) < 0.1 * max(1.0, abs(att["completeness"]))


def test_auroc_matches_the_pairwise_definition():
    from uno.play.inspect import auroc

    rng = np.random.default_rng(0)
    y = rng.random(200) < 0.3
    p = np.round(rng.random(200), 1)  # many ties
    pos, neg = p[y], p[~y]
    brute = ((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg))
    assert abs(auroc(y, p) - brute) < 1e-12


def test_the_prior_belief_is_the_uniform_split_marginal():
    from uno.eval.search import determinize_l1
    from uno.game.observe import Level, observe
    from uno.play.inspect import opponent_counts, prior_belief

    game = Game(FULL, heuristic_opponent(), 0, 31, [])
    game.run_bot()
    s = game.state
    exact = prior_belief(FULL, s, 0)
    obs, rng, n = observe(s, 0, Level.L1), random.Random(0), 4000
    mc = np.mean([opponent_counts(FULL, determinize_l1(FULL, obs, rng), 0) > 0 for _ in range(n)], axis=0)
    assert np.abs(mc - exact).max() < 4 * np.sqrt(0.25 / n)


class _Spy:
    def __init__(self):
        self.contexts = []

    def view(self, policy, d, chosen, context):
        self.contexts.append(context)
        return {}


def test_the_true_hand_is_sent_only_when_you_may_see_it():
    from uno.play.server import App

    pol = recurrent_policy(seed=5)
    app = App(FULL, None, rollouts=2)
    app.coaches.append(Coach("rec", "L1", pol))
    app.opponents["n"] = neural_opponent(pol)
    spy = app.inspector = _Spy()
    for reveal in (False, True):
        g = app.new_game("n", 0, reveal)
        game = app.games[g["game"]]
        app.inspect(g["game"], len(game.history) - 1, "rec")
        assert spy.contexts[-1]["truth"] is reveal and spy.contexts[-1]["seat"] == 0
    bot = [i for i, p in enumerate(game.history) if p["who"] == "bot"]
    if bot:  # the bot's opponent is you: your own hand is always yours to see
        app.inspect(g["game"], bot[0], "bot")
        assert spy.contexts[-1]["truth"] is True and spy.contexts[-1]["seat"] == 1
