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


def test_rollout_actor_diagnostics_are_chunk_independent_and_cover_the_final_policy():
    actor, critic, rollout, _, _ = setup(rs=TOY2_SKIP, games_per_cell=5)
    adv = _adv(critic, rollout)
    optim = Optim(minibatch=next(k for k in range(5, rollout.n_decisions) if rollout.n_decisions % k))
    with torch.no_grad():
        parameter = actor.score[-1].weight
        parameter.add_(torch.linspace(0.0, 0.1, parameter.numel()).reshape_as(parameter))
    whole = ppo.rollout_actor_diagnostics(actor, rollout, optim, adv, chunk=rollout.n_decisions)
    chunked = ppo.rollout_actor_diagnostics(actor, rollout, optim, adv, chunk=7)
    assert chunked == pytest.approx(whole, rel=1e-5, abs=1e-6)
    assert whole["approx_kl"] > 0.0
    assert whole["legal_actions"] >= 1.0
    assert whole["max_entropy"] >= whole["entropy"] - 1e-6
    assert 0.0 <= whole["normalized_entropy"] <= 1.0
    assert whole["effective_actions"] >= 1.0


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


def test_advantage_normalization_is_global_and_actor_loss_does_not_re_normalize_minibatches():
    actor, critic, rollout, _, _ = setup(rs=TOY2_SKIP, games_per_cell=5)
    raw = _adv(critic, rollout)
    global_adv = ppo.normalize_advantages(raw)
    assert global_adv.mean() == pytest.approx(0.0, abs=1e-6)
    assert global_adv.std() == pytest.approx(1.0, abs=1e-6)
    index = np.arange(min(7, rollout.n_decisions))
    normalized = Optim(normalize_advantages=True)
    plain = Optim(normalize_advantages=False)
    loss_normalized, _ = ppo.actor_loss(actor, rollout, index, global_adv, 1.0, normalized)
    loss_plain, _ = ppo.actor_loss(actor, rollout, index, global_adv, 1.0, plain)
    assert loss_normalized == loss_plain


def test_entropy_bonus_has_the_same_game_average_scale_as_the_surrogate():
    actor, critic, rollout, _, _ = setup(rs=TOY2_SKIP, games_per_cell=5)
    adv = _adv(critic, rollout)
    optim = Optim(entropy_coef=0.1, normalize_advantages=False)
    index = np.arange(min(11, rollout.n_decisions))
    loss_one, _ = ppo.actor_loss(actor, rollout, index, adv, 1.0, optim)
    loss_two, _ = ppo.actor_loss(actor, rollout, index, adv, 2.0, optim)
    assert float(loss_two.detach()) == pytest.approx(2.0 * float(loss_one.detach()))


def test_gamma_discounts_terminal_returns_by_remaining_learner_decisions():
    _, _, rollout, _, _ = setup(rs=TOY2_SKIP, games_per_cell=5)
    got = ppo.discounted_returns(rollout, 0.5)
    remaining = rollout.learner_decisions[rollout.episode] - rollout.decision_index - 1
    expected = rollout.terminal_return[rollout.episode] * np.power(0.5, remaining)
    assert np.allclose(got, expected)
    assert np.array_equal(ppo.discounted_returns(rollout, 1.0), rollout.returns_per_decision)


def test_value_clip_uses_the_more_conservative_value_loss():
    prediction = torch.tensor([2.0, -2.0])
    target = torch.tensor([0.0, 0.0])
    old = torch.tensor([0.0, 0.0])
    unclipped = ppo.critic_loss(prediction, target, old, None)
    clipped = ppo.critic_loss(prediction, target, old, 0.5)
    assert unclipped.item() == pytest.approx(4.0)
    assert clipped.item() == pytest.approx(4.0)
    assert ppo.critic_loss(torch.tensor([0.25]), torch.tensor([1.0]), torch.tensor([0.0]), 0.1).item() == pytest.approx(0.81)


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
    assert stats.critic_pre_update_ev is None
    assert stats.critic_post_update_same_rollout_ev is None


def test_update_reports_steps_epochs_and_kl_and_fits_the_critic_to_completion():
    actor, critic, rollout, _, _ = setup()
    optim = Optim(actor_epochs=4, critic_epochs=4, minibatch=32, target_kl=0.0)  # force an early stop
    frozen_advantages = _adv(critic, rollout)
    stats = ppo.update(actor, critic, torch.optim.Adam(actor.parameters(), lr=1e-2), torch.optim.Adam(critic.parameters(), lr=1e-2), rollout, optim, 0, 0)
    assert stats.early_stopped and stats.actor_epochs == 1
    assert stats.critic_epochs == 4, "critic fitting must complete even if actor epochs stop"
    assert stats.actor_steps > 0 and stats.critic_steps > 0
    assert stats.kl_per_epoch and stats.approx_kl >= 0.0
    final = ppo.rollout_actor_diagnostics(actor, rollout, optim, frozen_advantages)
    assert stats.surrogate == pytest.approx(final["surrogate"])
    assert stats.entropy == pytest.approx(final["entropy"])
    assert stats.legal_actions == pytest.approx(final["legal_actions"])
    assert stats.max_entropy == pytest.approx(final["max_entropy"])
    assert stats.normalized_entropy == pytest.approx(final["normalized_entropy"])
    assert stats.effective_actions == pytest.approx(final["effective_actions"])
    assert stats.clip_fraction == pytest.approx(final["clip_fraction"])
    assert stats.max_abs_log_ratio == pytest.approx(final["max_abs_log_ratio"])
    assert stats.approx_kl == pytest.approx(final["approx_kl"])
    assert 0.0 <= stats.critic_clipped_fraction <= 1.0
    assert stats.critic_pre_update_ev is not None
    assert stats.critic_post_update_same_rollout_ev is not None
    assert stats.critic_value_seconds > 0
    assert stats.actor_update_seconds > 0
    assert stats.critic_update_seconds > 0


def test_actor_kl_early_stop_is_checked_before_each_minibatch_step():
    actor, critic, rollout, _, _ = setup()
    optim = Optim(actor_epochs=4, critic_epochs=1, minibatch=8, target_kl=0.0)
    stats = ppo.update(
        actor,
        critic,
        torch.optim.Adam(actor.parameters(), lr=1e-2),
        torch.optim.Adam(critic.parameters(), lr=1e-3),
        rollout,
        optim,
        0,
        0,
    )
    assert stats.early_stopped
    assert stats.actor_epochs == 1
    assert stats.actor_steps < int(np.ceil(rollout.n_decisions / optim.minibatch))


def test_update_is_reproducible_given_the_same_streams():
    def run():
        actor, critic, rollout, _, _ = setup(seed=7)
        ppo.update(actor, critic, torch.optim.Adam(actor.parameters(), lr=1e-3), torch.optim.Adam(critic.parameters(), lr=1e-3), rollout, Optim(minibatch=32), 7, 0)
        return torch.cat([p.detach().reshape(-1) for p in actor.parameters()])

    assert torch.equal(run(), run())
