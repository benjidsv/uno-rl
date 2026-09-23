"""The recurrent actor: event tables, equivariance, and step/sequence agreement.

The event path must reproduce `uno.experiments.belief_ceiling.Recurrent._summaries`,
the model whose hand-inference gain motivates this actor; the actor must be
equivariant through its memory, not just at one decision; and the whole-episode
forward PPO trains through must equal the per-decision steps collection samples from.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.rl.helpers import SMALL_WIDTH, decision_games
from uno.game.observe import Level, observe
from uno.game.rules import MID_25, NAMED, TOY, TOY2_SKIP
from uno.game.symmetry import Symmetry
from uno.rl.env import features as F
from uno.rl.env.events import EventCounter, Tape, event_width
from uno.rl.env.evidence import Evidence
from uno.rl.model import Actor, masked_log_probs, to_tensors
from uno.rl.model.recurrent import EventExpansion, RecurrentActor

FULL = NAMED["FULL_UNO"]


def seat_sequences(rs, seeds, seat: int = 0, max_decisions: int = 80):
    """Each game's decision states for one seat, in order."""
    out = []
    for game in decision_games(rs, seeds, max_decisions):
        states = [s for s in game if s.player == seat]
        if len(states) >= 2:
            out.append(states)
    return out


def inputs(rs, sp, states):
    """Features, event rows and masks of one seat's decisions in one game."""
    obs = [observe(s, s.player, Level.L1) for s in states]
    ev = [Evidence.from_public(rs, s.public, s.player ^ 1) for s in states]
    counter = EventCounter(rs, states[0].player)
    events = np.stack([counter.advance(o) for o in obs])
    return F.extract(sp, obs, ev), events, F.legal_masks(rs, obs)


def small_recurrent(rs, arm=F.ARM_C, seed=0, scramble=True) -> RecurrentActor:
    torch.manual_seed(seed)
    actor = RecurrentActor(F.spec(rs, arm), width=SMALL_WIDTH, blocks=2, memory=8, events=6)
    if scramble:  # the fresh actor ignores memory and events; make them matter
        with torch.no_grad():
            proj = actor.body.encoder.proj.weight
            t = actor.spec.tok_width
            proj[:, t : t + actor.memory + actor.events].normal_(0.0, 0.3)
    return actor


@pytest.mark.parametrize("rs", (MID_25, FULL))
def test_event_tables_reproduce_the_belief_ceiling_summaries(rs):
    from uno.experiments.belief_ceiling import Recurrent

    sp = F.spec(rs, F.ARM_C)
    reference = Recurrent(rs, sp, SMALL_WIDTH, 2)
    expand = EventExpansion(sp)
    checked = 0
    for states in seat_sequences(rs, range(12)):
        tape, counter, pos, rows = Tape(rs, 0), EventCounter(rs, 0), [], []
        for s in states:
            o = observe(s, 0, Level.L1)
            pos.append(tape.advance(o))
            rows.append(counter.advance(o))
        tokens = torch.tensor(tape.tokens, dtype=torch.long)[None]
        want = reference._summaries(tokens, torch.tensor(pos)[None])[0]
        got = expand(torch.from_numpy(np.stack(rows)))
        assert torch.allclose(got, want, atol=1e-6)
        checked += len(states)
    assert checked > 50


def test_a_fresh_recurrent_actor_is_its_feedforward_actor():
    """Zero weights on memory and events: `load_feedforward` gives the same policy."""
    rs = MID_25
    sp = F.spec(rs, F.ARM_C)
    torch.manual_seed(1)
    ff = Actor(sp, width=SMALL_WIDTH)
    rec = small_recurrent(rs, scramble=False)
    rec.load_feedforward(ff.state_dict())
    states = seat_sequences(rs, range(3))[0]
    feats, events, _ = inputs(rs, sp, states)
    tok, col, glob = to_tensors(feats)
    with torch.no_grad():
        want = ff(tok, col, glob)
        got = rec(tok, col, glob, torch.from_numpy(events), [len(states)])
    assert torch.allclose(got, want, atol=1e-5)


def test_the_episode_forward_equals_per_decision_steps():
    """What PPO recomputes is what collection sampled from, in any episode order."""
    rs = MID_25
    sp = F.spec(rs, F.ARM_C)
    actor = small_recurrent(rs)
    games = seat_sequences(rs, range(10))[:6]
    per_game = [inputs(rs, sp, g) for g in games]
    with torch.no_grad():
        stepped = []
        for feats, events, _ in per_game:
            tok, col, glob = to_tensors(feats)
            m = actor.initial_memory(1)
            for i in range(len(feats)):
                logits, m = actor.step(tok[i : i + 1], col[i : i + 1], glob[i : i + 1],
                                       torch.from_numpy(events[i : i + 1]), m)
                stepped.append(logits[0])
        stepped = torch.stack(stepped)
        feats = F.Features(*(np.concatenate([getattr(f, k) for f, _, _ in per_game]) for k in ("tok", "col", "glob")))
        events = torch.from_numpy(np.concatenate([e for _, e, _ in per_game]))
        lengths = [len(g) for g in games]
        assert len(set(lengths)) > 1
        whole = actor(*to_tensors(feats), events, lengths)
    assert torch.allclose(whole, stepped, atol=1e-5)
    # memory matters: the same rows without their history give different logits
    with torch.no_grad():
        alone = actor(*to_tensors(feats), events, [1] * len(feats))
    assert not torch.allclose(alone, stepped, atol=1e-3)


@pytest.mark.parametrize("rs", (TOY2_SKIP, MID_25))
def test_probabilities_are_equivariant_through_the_memory(rs):
    """`policy(g . history)(g . a) = policy(history)(a)` at every decision of a game,
    so memory built from mapped events stays the mapped memory."""
    sym = Symmetry(rs)
    sp = F.spec(rs, F.ARM_C)
    actor = small_recurrent(rs, seed=4)
    worst, compared = 0.0, 0
    for states in seat_sequences(rs, range(8))[:4]:
        feats, events, masks = inputs(rs, sp, states)
        with torch.no_grad():
            base = masked_log_probs(actor(*to_tensors(feats), torch.from_numpy(events), [len(states)]),
                                    torch.from_numpy(masks)).exp().numpy()
        for g in range(0, sym.order, max(1, sym.order // 24)):  # a spread of the group, for time
            mapped = [sym.state(g, s) for s in states]
            mf, me, mm = inputs(rs, sp, mapped)
            with torch.no_grad():
                got = masked_log_probs(actor(*to_tensors(mf), torch.from_numpy(me), [len(mapped)]),
                                       torch.from_numpy(mm)).exp().numpy()
            want = np.zeros_like(base)
            want[:, np.asarray(sym.actions[g], dtype=np.int64)] = base
            worst = max(worst, float(np.abs(want - got).max()))
            compared += len(states)
    assert compared > 20
    assert worst < 1e-5, f"max mapped-probability error {worst}"


# -- the pipeline ------------------------------------------------------------------


def recurrent_setup(rs=TOY, games_per_cell: int = 6, seed: int = 0, live: int = 8):
    from uno.rl.algo.collect import collect
    from uno.rl.algo.policy import ActorPolicy
    from uno.rl.core.manifest import DealSchedule, build
    from uno.rl.eval.opponents import league

    actor = small_recurrent(rs, seed=seed)
    opponents = league(rs, "toy")
    manifest, _ = build(0, list(opponents), DealSchedule(seed), games_per_cell, 0)
    rollout, _ = collect(manifest, ActorPolicy(actor.spec, actor), F.spec(rs, F.ARM_CRITIC), opponents, seed, 0,
                         live_games=live)
    return actor, rollout, manifest


def test_recurrent_collection_stores_events_and_is_independent_of_batching():
    """Each game carries its own memory: one live game or many give the same data."""
    _, wide, _ = recurrent_setup(live=16)
    _, narrow, _ = recurrent_setup(live=1)
    assert wide.events is not None and wide.events.shape == (wide.n_decisions, event_width(TOY))
    assert np.array_equal(wide.actions, narrow.actions)
    assert np.array_equal(wide.events, narrow.events)
    assert np.allclose(wide.old_logp, narrow.old_logp, atol=1e-6)
    # every episode's first row reads the initial hand
    first = wide.events[wide.offset]
    assert (first[:, : TOY.num_types].sum(1) == TOY.hand_size).all()


def test_ppo_recomputes_the_collection_probabilities_at_frozen_weights():
    """Stage 2's frozen old/new agreement, through the whole-episode recurrent forward."""
    from uno.rl.algo import ppo

    actor, rollout, _ = recurrent_setup()
    t = rollout.n_decisions
    zero = torch.zeros(t)
    prepared = ppo._prepare_rollout(rollout, zero, zero, zero, "cpu")
    for chunk in (7, 10_000):
        got = []
        for sl, lengths in ppo._chunks(prepared, chunk):
            with torch.no_grad():
                logp, _ = ppo._new_log_probs_tensors(
                    actor, prepared.actor_tok[sl], prepared.actor_col[sl], prepared.actor_glob[sl],
                    prepared.actor_masks[sl], prepared.actor_actions[sl], 1.0,
                    events=prepared.actor_events[sl], lengths=lengths)
            got.append(logp)
        assert np.allclose(torch.cat(got).numpy(), rollout.old_logp, atol=1e-5)
    rng = np.random.default_rng(0)
    batches = ppo._actor_minibatches(rng, prepared, 4, "cpu")
    assert sorted(np.concatenate([b[0] for b in batches]).tolist()) == list(range(t))
    for index, rows, lengths in batches:
        loss, diag = ppo._prepared_actor_loss(actor, prepared, index, rows, ppo.Optim(), lengths)
        assert float(diag["approx_kl"]) < 1e-10


def test_a_recurrent_update_trains_the_memory():
    from uno.rl.algo import ppo
    from uno.rl.core.config import Optim
    from uno.rl.model import Critic

    actor, rollout, _ = recurrent_setup()
    torch.manual_seed(0)
    critic = Critic(F.spec(TOY, F.ARM_CRITIC), width=SMALL_WIDTH)
    before = {k: v.clone() for k, v in actor.state_dict().items()}
    optim = Optim(n_minibatches=4, actor_epochs=2, critic_epochs=1, target_kl=None, actor_lr=1e-3)
    stats = ppo.update(actor, critic, torch.optim.Adam(actor.parameters(), lr=optim.actor_lr),
                       torch.optim.Adam(critic.parameters(), lr=optim.critic_lr), rollout, optim, 0, 0)
    assert stats.actor_steps == 8 and stats.actor_presentations == 2 * rollout.n_decisions
    assert stats.approx_kl > 0 and np.isfinite(stats.approx_kl)
    for key in ("cell.weight_ih", "phi.0.weight"):
        assert not torch.equal(before[key], actor.state_dict()[key]), key


def test_a_recurrent_actor_plays_the_panel_and_refuses_stateless_queries():
    from uno.rl.algo.policy import ActorPolicy, NeuralAgent
    from uno.rl.core.manifest import panel
    from uno.rl.eval.evaluate import play_panel
    from uno.rl.eval.opponents import league

    actor = small_recurrent(TOY)
    pol = ActorPolicy(actor.spec, actor)
    opponents = league(TOY, "toy")
    result = play_panel(panel(list(opponents), 4, 0), pol, opponents, 0, live_games=3)
    assert result.behavior.games == 4 * len(opponents) * 2
    state = decision_games(TOY, [1])[0][0]
    with pytest.raises(ValueError, match="per-game memory"):
        NeuralAgent(pol, "rec").policy(observe(state, state.player, Level.L1), ())


def test_warm_start_begins_at_the_relational_run_s_policy(tmp_path):
    from uno.rl.algo.train import Run, warm_start
    from uno.rl.core import checkpoint as ckpt
    from uno.rl.core.config import Config, RecurrentModel, RelationalModel

    base = Config(ruleset="TOY", arm=F.ARM_C, model=RelationalModel(width=SMALL_WIDTH, blocks=2))
    source = Run(base, tmp_path / "rel", league_name="toy")
    path = tmp_path / "rel.pt"
    ckpt.save(path, base, source.actor, source.critic, source.actor_opt, source.critic_opt, source.progress)
    cfg = base.with_(model=RecurrentModel(width=SMALL_WIDTH, blocks=2, memory=8, events=6, init_from=str(path)))
    run = Run(cfg, tmp_path / "rec", league_name="toy")
    warm_start(run)
    states = seat_sequences(TOY, range(4))[0]
    feats, events, _ = inputs(TOY, run.actor.spec, states)
    with torch.no_grad():
        want = source.actor(*to_tensors(feats))
        got = run.actor(*to_tensors(feats), torch.from_numpy(events), [len(states)])
    assert torch.allclose(got, want, atol=1e-5)
    for a, b in zip(source.critic.parameters(), run.critic.parameters()):
        assert torch.equal(a, b)
    with pytest.raises(ValueError, match="relational run of width"):
        other = Run(cfg.with_(model=RecurrentModel(width=SMALL_WIDTH * 2, blocks=2, init_from=str(path))),
                    tmp_path / "bad", league_name="toy")
        warm_start(other)


@pytest.mark.slow
def test_parallel_recurrent_collection_equals_the_single_worker_reference():
    from uno.rl.algo.parallel import ParallelCollector, WorkerConfig

    actor, reference, manifest = recurrent_setup(games_per_cell=8)
    cfg = WorkerConfig(ruleset="TOY", arm=F.ARM_C, league="toy", training_seed=0, live_games=8, temperature=1.0,
                       model_config={"kind": "recurrent", "width": SMALL_WIDTH, "blocks": 2, "memory": 8, "events": 6})
    with ParallelCollector(cfg, workers=2, bundle=3) as pool:
        parallel, stats = pool.collect(manifest, actor, F.spec(TOY, F.ARM_CRITIC), behavior_version=0, timeout=120)
    assert stats.retried_bundles == 0
    assert np.array_equal(reference.actions, parallel.actions)
    assert np.array_equal(reference.events, parallel.events)
    assert np.allclose(reference.old_logp, parallel.old_logp, atol=1e-6)
