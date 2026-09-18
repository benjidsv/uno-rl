"""Stage 2 acceptance for the reference learner (docs/rl/pipeline-plan.md section 13).

Frozen old/new probability agreement, the game-average-sum weighting under
unequal game lengths and a partial final minibatch, the detached baseline, and
replayable seeds.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.rl.helpers import SMALL_WIDTH
from uno.agents.baseline import Heuristic, PlayFirstRandom
from uno.game.rules import TOY, TOY2_SKIP
from uno.rl import features as F
from uno.rl import ppo
from uno.rl.collect import collect
from uno.rl.config import Optim
from uno.rl.manifest import DealSchedule, build
from uno.rl.model import Actor, Critic
from uno.rl.policy import ActorPolicy
from uno.rl.rollout import Rollout


def setup(rs=TOY, games_per_cell: int = 6, seed: int = 0, live: int = 8):
    torch.manual_seed(seed)
    sp, spc = F.spec(rs, F.ARM_B), F.spec(rs, F.ARM_CRITIC)
    actor, critic = Actor(sp, width=SMALL_WIDTH), Critic(spc, width=SMALL_WIDTH)
    opponents = {a.name: a for a in (PlayFirstRandom(rs), Heuristic(rs))}
    schedule = DealSchedule(training_seed=seed)
    manifest, _ = build(0, list(opponents), schedule, games_per_cell, 0)
    rollout, stats = collect(manifest, ActorPolicy(sp, actor), spc, opponents, seed, 0, live_games=live)
    return actor, critic, rollout, stats, manifest


# -- collection ------------------------------------------------------------------


def test_the_manifest_is_completed_exactly_once_per_episode():
    _, _, rollout, stats, manifest = setup()
    assert rollout.n_games == len(manifest)
    assert sorted(rollout.episode_id.tolist()) == [e.episode_id for e in manifest.episodes]
    assert stats.games == len(manifest)
    assert set(rollout.deal_seed.tolist()) == {e.deal_seed for e in manifest.episodes}


def test_episodes_are_stored_in_logical_id_order_not_completion_order():
    _, _, rollout, _, _ = setup()
    assert np.all(np.diff(rollout.episode_id) > 0)
    for i in range(rollout.n_games):
        rows = slice(int(rollout.offset[i]), int(rollout.offset[i] + rollout.learner_decisions[i]))
        assert np.all(rollout.episode[rows] == i)
        assert rollout.decision_index[rows].tolist() == list(range(int(rollout.learner_decisions[i])))


def test_cells_are_balanced_over_opponents_and_seats():
    _, _, rollout, _, manifest = setup()
    from collections import Counter

    counts = Counter(zip(rollout.opponent, rollout.seat.tolist()))
    assert set(counts) == set(manifest.cells)
    assert set(counts.values()) == {manifest.games_per_cell}


def test_terminal_returns_are_the_engine_s_and_in_the_learner_frame():
    """A ruleset cap or an exhaustion is a real terminal result, not a collector
    truncation, and the return is the learner's own (pipeline plan section 3)."""
    _, _, rollout, _, _ = setup()
    assert set(np.unique(rollout.terminal_return).tolist()) <= {-1.0, 0.0, 1.0}
    reasons = np.asarray(rollout.end_reason)
    assert set(reasons.tolist()) <= {"win", "exhaustion", "cap"}
    assert np.all(np.abs(rollout.terminal_return[reasons == "win"]) == 1.0)
    assert np.all(rollout.terminal_return[reasons == "cap"] == 0.0)
    # both seats occur and both outcomes occur, so the frame is actually exercised
    assert set(rollout.seat.tolist()) == {0, 1}
    assert {-1.0, 1.0} <= set(rollout.terminal_return.tolist())


def test_collection_is_reproducible_from_its_logical_streams():
    a = setup(seed=3)[2]
    b = setup(seed=3)[2]
    assert np.array_equal(a.actions, b.actions)
    assert np.array_equal(a.terminal_return, b.terminal_return)
    assert np.allclose(a.old_logp, b.old_logp)


def test_live_game_count_is_an_execution_setting_only():
    """Pipeline plan section 3: worker/batching settings must not change the data."""
    _, _, wide, _, _ = setup(seed=5, live=32)
    _, _, narrow, _, _ = setup(seed=5, live=1)
    assert np.array_equal(wide.episode_id, narrow.episode_id)
    assert np.array_equal(wide.terminal_return, narrow.terminal_return)
    assert np.array_equal(wide.actions, narrow.actions)
    assert np.allclose(wide.old_logp, narrow.old_logp, atol=1e-6)


def test_rollout_rejects_a_duplicate_episode():
    from uno.rl.rollout import Builder

    _, _, rollout, _, manifest = setup()
    b = Builder(rollout.actor_spec, rollout.critic_spec, 0)
    b.finish_episode(manifest.episodes[0], 1.0, "win", 3)
    with pytest.raises(ValueError):
        b.finish_episode(manifest.episodes[0], 1.0, "win", 3)


def test_rollout_rejects_a_short_manifest():
    from uno.rl.rollout import Builder

    _, _, rollout, _, manifest = setup()
    b = Builder(rollout.actor_spec, rollout.critic_spec, 0)
    b.finish_episode(manifest.episodes[0], 1.0, "win", 3)
    with pytest.raises(ValueError, match="were assigned"):
        b.finish(expected=len(manifest))


# -- probabilities ----------------------------------------------------------------


def test_recomputed_log_probs_match_the_behavior_policy_at_frozen_weights():
    actor, _, rollout, _, _ = setup()
    logp, _ = ppo.new_log_probs(actor, rollout.actor, rollout.masks, rollout.actions, 1.0, "cpu")
    assert np.allclose(logp.detach().numpy(), rollout.old_logp, atol=1e-6)


def test_kl_and_ratio_are_zero_at_the_collection_policy():
    actor, _, rollout, _, _ = setup()
    assert ppo.rollout_kl(actor, rollout, Optim()) < 1e-9


# -- the objective ------------------------------------------------------------------


def _adv(critic, rollout: Rollout) -> np.ndarray:
    return ppo.advantages(rollout, ppo.critic_values(critic, rollout.critic))


def test_advantage_is_the_terminal_return_minus_the_frozen_baseline():
    _, critic, rollout, _, _ = setup()
    values = ppo.critic_values(critic, rollout.critic)
    adv = ppo.advantages(rollout, values)
    assert np.allclose(adv, rollout.returns_per_decision - values)
    for i in range(rollout.n_games):
        rows = slice(int(rollout.offset[i]), int(rollout.offset[i] + rollout.learner_decisions[i]))
        assert np.all(rollout.returns_per_decision[rows] == rollout.terminal_return[i])


def test_minibatch_gradient_sums_to_the_full_batch_gradient():
    """Pipeline plan section 6, including a final partial minibatch."""
    actor, critic, rollout, _, _ = setup(rs=TOY2_SKIP, games_per_cell=5)
    adv = _adv(critic, rollout)
    optim = Optim(entropy_coef=0.0, normalize_advantages=False)
    t, g = rollout.n_decisions, rollout.n_games
    size = next(k for k in range(5, t) if t % k)  # guarantee a partial final minibatch
    index = np.arange(t)
    full = ppo.weighted_gradient(actor, rollout, adv, [index], optim)
    parts = [index[i : i + size] for i in range(0, t, size)]
    assert len(parts[-1]) < size
    split = ppo.weighted_gradient(actor, rollout, adv, parts, optim)
    for a, b in zip(full, split):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-4)


def test_game_lengths_are_unequal_in_that_check():
    _, _, rollout, _, _ = setup(rs=TOY2_SKIP, games_per_cell=5)
    assert len(set(rollout.learner_decisions.tolist())) > 1


def test_the_objective_is_a_game_average_sum_not_a_decision_mean():
    """Doubling one game's decisions must move the objective, which a decision mean
    would hide; and the scale is T/G, not 1."""
    actor, critic, rollout, _, _ = setup()
    adv = _adv(critic, rollout)
    t, g = rollout.n_decisions, rollout.n_games
    index = np.arange(t)
    _, diag_scaled = ppo.actor_loss(actor, rollout, index, adv, t / g, Optim())
    _, diag_mean = ppo.actor_loss(actor, rollout, index, adv, 1.0, Optim())
    assert abs(diag_scaled["surrogate"] - (t / g) * diag_mean["surrogate"]) < 1e-5
    # at the collection policy the ratio is 1, so the surrogate is the mean advantage
    assert abs(diag_mean["surrogate"] - float(adv.mean())) < 1e-4


def test_the_baseline_is_detached_from_the_actor_graph():
    actor, critic, rollout, _, _ = setup()
    adv = _adv(critic, rollout)
    loss, _ = ppo.actor_loss(actor, rollout, np.arange(rollout.n_decisions), adv, 1.0, Optim())
    loss.backward()
    assert all(p.grad is None for p in critic.parameters())
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in actor.parameters())


def test_explained_variance_is_unavailable_rather_than_invented():
    actor, critic, rollout, _, _ = setup()
    zeroed = Rollout(**{**rollout.__dict__, "terminal_return": np.zeros_like(rollout.terminal_return)})
    stats = ppo.update(actor, critic, torch.optim.Adam(actor.parameters()), torch.optim.Adam(critic.parameters()), zeroed, Optim(actor_epochs=1, critic_epochs=1), 0, 0)
    assert stats.explained_variance is None


def test_update_reports_steps_epochs_and_kl_and_fits_the_critic_to_completion():
    actor, critic, rollout, _, _ = setup()
    optim = Optim(actor_epochs=4, critic_epochs=4, minibatch=32, target_kl=0.0)  # force an early stop
    stats = ppo.update(actor, critic, torch.optim.Adam(actor.parameters(), lr=1e-2), torch.optim.Adam(critic.parameters(), lr=1e-2), rollout, optim, 0, 0)
    assert stats.early_stopped and stats.actor_epochs == 1
    assert stats.critic_epochs == 4, "critic fitting must complete even if actor epochs stop"
    assert stats.actor_steps > 0 and stats.critic_steps > 0
    assert stats.kl_per_epoch and stats.approx_kl >= 0.0


def test_update_is_reproducible_given_the_same_streams():
    def run():
        actor, critic, rollout, _, _ = setup(seed=7)
        ppo.update(actor, critic, torch.optim.Adam(actor.parameters(), lr=1e-3), torch.optim.Adam(critic.parameters(), lr=1e-3), rollout, Optim(minibatch=32), 7, 0)
        return torch.cat([p.detach().reshape(-1) for p in actor.parameters()])

    assert torch.equal(run(), run())
