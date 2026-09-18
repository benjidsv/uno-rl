"""Stage 4 acceptance: checkpoints, resume, tracking outages, panels and selection.

docs/rl/pipeline-plan.md section 13: "Save/load and interrupted continuation;
logging outage recovery; worker retry; cross-device short continuation". The
cross-device case is CPU-to-CPU here -- what is tested is that a checkpoint is
portable, remaps optimizer state and refuses an experiment branch, not that two
devices produce identical learning curves, which the plan explicitly does not claim.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from tests.rl.helpers import SMALL_WIDTH
from uno.agents.baseline import Heuristic
from uno.game.rules import TOY
from uno.rl import checkpoint as ckpt
from uno.rl import evaluate as ev
from uno.rl import features as F
from uno.rl import manifest as mf
from uno.rl.config import Config, Optim
from uno.rl.opponents import league, league_hash
from uno.rl.policy import ActorPolicy
from uno.rl.track import Tracker
from uno.rl.train import Run, iteration


def toy_config(**kw) -> Config:
    base = Config(ruleset="TOY", arm=F.ARM_B, width=SMALL_WIDTH, blocks=2, training_seed=0, data_cap=2000,
                  optim=Optim(minibatch=64, actor_epochs=2, critic_epochs=2))  # fmt: skip
    return base.with_(**kw) if kw else base


def make_run(tmp_path, **kw) -> Run:
    return Run(toy_config(**kw), tmp_path, league_name="toy")


# -- checkpoints -----------------------------------------------------------------


def test_checkpoint_round_trip_restores_parameters_counters_and_schedule(tmp_path):
    run = make_run(tmp_path)
    run.progress.update = 3
    run.progress.learner_decisions = 1234
    run.progress.deal_schedule = run.schedule.state
    manifest = ckpt.save(tmp_path / "c.pt", run.config, run.actor, run.critic, run.actor_opt, run.critic_opt, run.progress, {"league": "toy"})
    assert ckpt.verify(tmp_path / "c.pt")
    assert json.loads((tmp_path / "c.pt.manifest.json").read_text())["checksum"] == manifest["checksum"]

    other = make_run(tmp_path / "second")
    with torch.no_grad():  # the same training seed initializes identically, so move it first
        for p in other.actor.parameters():
            p.add_(1.0)
    before = torch.cat([p.detach().reshape(-1) for p in other.actor.parameters()]).clone()
    payload = ckpt.load(tmp_path / "c.pt")
    progress = ckpt.restore(payload, other.config, other.actor, other.critic, other.actor_opt, other.critic_opt)
    after = torch.cat([p.detach().reshape(-1) for p in other.actor.parameters()])
    assert not torch.equal(before, after)
    assert torch.equal(after, torch.cat([p.detach().reshape(-1) for p in run.actor.parameters()]))
    assert progress.update == 3 and progress.learner_decisions == 1234
    assert progress.deal_schedule == run.schedule.state


def test_checkpoint_refuses_a_different_schema_or_configuration(tmp_path):
    run = make_run(tmp_path)
    ckpt.save(tmp_path / "c.pt", run.config, run.actor, run.critic, run.actor_opt, run.critic_opt, run.progress)
    payload = ckpt.load(tmp_path / "c.pt")
    branched = make_run(tmp_path / "branch", arm=F.ARM_A)
    with pytest.raises(ValueError, match="schema|configuration"):
        ckpt.restore(payload, branched.config, branched.actor, branched.critic)
    retuned = make_run(tmp_path / "lr", optim=Optim(actor_lr=1e-2))
    with pytest.raises(ValueError, match="configuration"):
        ckpt.restore(payload, retuned.config, retuned.actor, retuned.critic)


def test_actor_export_excludes_the_critic_and_replays_the_same_policy(tmp_path):
    run = make_run(tmp_path)
    ckpt.export_actor(tmp_path / "actor.pt", run.config, run.actor, "toy_actor", run.progress)
    payload = torch.load(tmp_path / "actor.pt", weights_only=False)
    assert "critic" not in payload and "actor_opt" not in payload
    agent, meta = ckpt.load_actor(tmp_path / "actor.pt", TOY)
    assert meta["arm"] == F.ARM_B

    from uno.game.engine import legal_actions, new_game
    from uno.game.observe import Level, observe

    s = new_game(TOY, 0)
    obs, legal = observe(s, s.player, Level.L1), legal_actions(s)
    direct = ActorPolicy(F.spec(TOY, F.ARM_B), run.actor)
    lp, _ = direct.distributions([obs], [__import__("uno.rl.evidence", fromlist=["Evidence"]).Evidence.from_public(TOY, s.public, obs.seat ^ 1)])
    exported = agent.policy(obs, legal)
    for a in legal:
        assert abs(exported[a] - float(np.exp(lp[0, a]))) < 1e-6


def test_interrupted_continuation_reaches_the_same_state_as_an_uninterrupted_run(tmp_path):
    """Two iterations straight through, against one-then-resume-then-one."""
    straight = make_run(tmp_path / "straight")
    iteration(straight)
    iteration(straight)

    first = make_run(tmp_path / "first")
    iteration(first)
    first.progress.deal_schedule = first.schedule.state
    ckpt.save(tmp_path / "boundary.pt", first.config, first.actor, first.critic, first.actor_opt, first.critic_opt, first.progress, {"league": "toy"})

    from uno.rl.train import resume

    resumed = resume(tmp_path / "boundary.pt", toy_config(), tmp_path / "resumed", league_name="toy")
    iteration(resumed)

    assert resumed.progress.update == straight.progress.update
    assert resumed.progress.next_episode_id == straight.progress.next_episode_id
    assert resumed.progress.learner_decisions == straight.progress.learner_decisions
    assert resumed.schedule.state == straight.schedule.state
    a = torch.cat([p.detach().reshape(-1) for p in resumed.actor.parameters()])
    b = torch.cat([p.detach().reshape(-1) for p in straight.actor.parameters()])
    assert torch.allclose(a, b, atol=1e-6)


# -- tracking ----------------------------------------------------------------------


def test_tracking_outage_keeps_the_local_metrics_and_marks_pending_uploads(tmp_path):
    tracker = Tracker(tmp_path / "t", run_name="r", tags={"seed": 0}, params={"x": 1}, use_mlflow=False)
    tracker.log({"loss": 1.0, "explained_variance": None}, step=10)
    tracker.log({"loss": 0.5, "explained_variance": 0.25}, step=20)
    summary = tracker.finish()
    rows = (tmp_path / "t" / "metrics.csv").read_text().splitlines()
    assert len(rows) == 3
    assert rows[1].split(",")[0] == "10"
    assert ",," in rows[1] or rows[1].endswith(","), "an undefined statistic is written empty, not as a number"
    assert summary["pending_uploads"] == 2


def test_tracker_records_provenance_and_params(tmp_path):
    from uno.rl.track import provenance

    p = provenance("deadbeef")
    assert set(p) >= {"base_commit", "dirty", "torch", "numpy", "platform", "machine"}
    tracker = Tracker(tmp_path / "t", run_name="r", params=p, use_mlflow=False)
    assert json.loads((tmp_path / "t" / "params.json").read_text())["torch"] == p["torch"]
    tracker.finish()


# -- panels, pairing and selection ---------------------------------------------------


def test_the_panel_is_balanced_coupled_and_inside_the_dev_partition():
    names = list(league(TOY, "toy"))
    panel = mf.panel(names, 32)
    assert len(panel) == len(names) * 2 * 32
    per_cell = {c: sorted(e.deal_seed for e in panel.episodes if (e.opponent, e.seat) == c) for c in panel.cells}
    first = next(iter(per_cell.values()))
    assert all(v == first for v in per_cell.values()), "cells must share deals, or the difference is not paired"
    assert all(d in mf.PANEL_DEALS for d in first)
    assert not set(first) & set(mf.TRAIN_DEALS)


def test_panel_refuses_to_run_past_its_partition():
    with pytest.raises(ValueError):
        mf.panel(["a"], 1, start=mf.PANEL_DEALS.stop - 1 + 1)


def test_paired_difference_groups_the_cells_of_a_deal(tmp_path):
    run = make_run(tmp_path, evaluation=__import__("uno.rl.config", fromlist=["Evaluation"]).Evaluation(deals_per_cell=24))
    result = ev.play_panel(run.panel, run.policy, run.opponents, run.config.training_seed)
    baseline = ev.baseline_panel(run.panel, Heuristic(TOY), run.opponents, TOY, run.config.training_seed)
    assert len(result.deals) == 24 and np.array_equal(result.deals, baseline.deals)
    diff = ev.paired(result, baseline)
    assert abs(diff.difference - (result.mean - baseline.mean)) < 1e-9
    assert diff.n == 24
    # the aggregate is the equal-weight mean of the cells, not a decision-weighted one
    assert abs(result.mean - float(np.mean(list(result.cell_means.values())))) < 1e-12
    assert set(result.behavior.as_dict()) >= {"draw_rate", "action_cap_rate", "mean_game_length", "entropy"}


def test_paired_difference_refuses_mismatched_panels(tmp_path):
    run = make_run(tmp_path)
    a = ev.play_panel(mf.panel(list(run.opponents), 4), run.policy, run.opponents, 0)
    b = ev.play_panel(mf.panel(list(run.opponents), 4, start=mf.PANEL_DEALS.start + 100), run.policy, run.opponents, 0)
    with pytest.raises(ValueError):
        ev.paired(a, b)


def test_selection_prefers_the_earlier_checkpoint_on_a_tie():
    assert ev.select([(0, 0.5), (1, 0.5000001), (2, 0.4)]) == (0, 0.5)
    assert ev.select([(0, 0.5), (1, 0.6), (2, 0.6)]) == (1, 0.6)


def test_league_hash_is_stable_and_names_the_frozen_members():
    from uno.agents.rulebot import FROZEN_RULEBOT_LEAGUE
    from uno.game.rules import FULL_UNO

    assert tuple(league(FULL_UNO, "rulebot")) == FROZEN_RULEBOT_LEAGUE
    assert league_hash(FULL_UNO, "rulebot") == league_hash(FULL_UNO, "rulebot")
    assert league_hash(FULL_UNO, "rulebot") != league_hash(TOY, "toy")


# -- the deal schedule -----------------------------------------------------------------


def test_deal_schedule_cycles_without_repeating_or_leaving_the_training_partition():
    small = mf.DealSchedule(0, pool=range(0, 10))
    first = small.take(10)
    assert sorted(s for s, _ in first) == list(range(10))
    assert {c for _, c in first} == {0}
    second = small.take(10)
    assert {c for _, c in second} == {1}
    assert [s for s, _ in second] != [s for s, _ in first], "a new cycle draws a new permutation"
    assert small.state["cycle"] == 2


def test_deal_schedule_state_round_trips():
    s = mf.DealSchedule(3)
    s.take(100)
    restored = mf.DealSchedule.from_state(3, s.state)
    assert restored.take(5) == mf.DealSchedule(3, s.position, s.cycle).take(5)


def test_a_real_mlflow_run_records_tags_params_metrics_and_artifacts(tmp_path):
    """The store is local, explicit and complete: the plan's section 10 groups."""
    mlflow = pytest.importorskip("mlflow")
    from uno.rl.track import EXPERIMENT, local_store
    from uno.rl.train import start_tracker, train

    cfg = toy_config(data_cap=300, evaluation=__import__("uno.rl.config", fromlist=["Evaluation"]).Evaluation(deals_per_cell=8))
    run = Run(cfg, tmp_path / "run", league_name="toy")
    run.tracker = start_tracker(run, store_root=tmp_path)
    if not run.tracker.available:  # a broken local store must not fail the suite
        pytest.skip(run.tracker.unavailable_reason)
    train(run, run.tracker, verbose=False)
    run_id = run.tracker.run_id
    assert run.tracker.pending_uploads == 0

    uri, artifacts = local_store(tmp_path)
    mlflow.set_tracking_uri(uri)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT)
    assert experiment.artifact_location == artifacts  # explicit, not the process's cwd
    stored = client.get_run(run_id)
    assert {"study_id", "arm", "phase", "seed", "machine", "kind", "league"} <= set(stored.data.tags)
    assert {"config_hash", "league_hash", "actor_schema", "critic_schema", "stream_version", "base_commit", "mlflow"} <= set(stored.data.params)
    assert {"surrogate", "critic_mse", "entropy", "approx_kl", "dev_league_mean", "games", "engine_decisions"} <= set(stored.data.metrics)
    steps = [m.step for m in client.get_metric_history(run_id, "surrogate")]
    assert steps == sorted(steps) and steps[0] > 0  # the step is cumulative learner decisions
    assert {"run_manifest.json", "summary.json"} <= {a.path for a in client.list_artifacts(run_id)}
    assert stored.info.status == "FINISHED"
