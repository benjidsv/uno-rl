"""Stage 4 acceptance: checkpoints, resume, tracking outages, panels and selection.

docs/rl/pipeline-plan.md section 13: "Save/load and interrupted continuation;
logging outage recovery; worker retry; cross-device short continuation". The
cross-device case is CPU-to-CPU here -- what is tested is that a checkpoint is
portable, remaps optimizer state and refuses an experiment branch, not that two
devices produce identical learning curves, which the plan explicitly does not claim.
"""

from __future__ import annotations

import json
import signal
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.rl.helpers import SMALL_WIDTH
from uno.agents.baseline import Heuristic
from uno.game.engine import Phase
from uno.game.rules import Play, TOY
from uno.rl.core import checkpoint as ckpt
from uno.rl.eval import evaluate as ev
from uno.rl.env import features as F
from uno.rl.core import manifest as mf
from uno.rl.core.config import Collection, Config, Evaluation, Optim, RelationalModel
from uno.rl.eval.opponents import league, league_hash
from uno.rl.algo.policy import ActorPolicy
from uno.rl.core.track import Tracker
from uno.rl.algo.train import Run, evaluate_now, iteration


def toy_config(**kw) -> Config:
    base = Config(ruleset="TOY", arm=F.ARM_B, model=RelationalModel(width=SMALL_WIDTH, blocks=2), training_seed=0, data_cap=2000,
                  optim=Optim(n_minibatches=64, actor_epochs=2, critic_epochs=2))  # fmt: skip
    return base.with_(**kw) if kw else base


def make_run(tmp_path, **kw) -> Run:
    return Run(toy_config(**kw), tmp_path, league_name="toy")


# -- checkpoints -----------------------------------------------------------------


def test_checkpoint_round_trip_restores_parameters_counters_and_schedule(tmp_path):
    run = make_run(tmp_path)
    run.progress.update = 3
    run.progress.learner_decisions = 1234
    run.progress.best_score = 0.75
    run.progress.best_checkpoint = "checkpoints/best.pt"
    run.progress.best_update = 2
    run.progress.best_decisions = 900
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
    assert (progress.best_score, progress.best_update, progress.best_decisions) == (0.75, 2, 900)
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


def test_checkpoint_allows_new_budgets_evaluation_and_execution_settings(tmp_path):
    run = make_run(tmp_path)
    ckpt.save(
        tmp_path / "c.pt", run.config, run.actor, run.critic,
        run.actor_opt, run.critic_opt, run.progress, {"league": "toy"},
    )
    payload = ckpt.load(tmp_path / "c.pt")
    continued_config = run.config.with_(
        data_cap=50_000,
        time_cap_s=600.0,
        device="cpu",
        collection=Collection(
            games_per_cell=run.config.collection.games_per_cell,
            target_decisions=run.config.collection.target_decisions,
            workers=3,
            live_games_per_worker=7,
        ),
        evaluation=Evaluation(deals_per_cell=31, every_fraction=0.25),
    )
    assert continued_config.hash != run.config.hash
    assert continued_config.algorithm_hash == run.config.algorithm_hash
    other = Run(continued_config, tmp_path / "continued", league_name="toy")
    ckpt.restore(
        payload, continued_config, other.actor, other.critic,
        other.actor_opt, other.critic_opt,
    )


def test_legacy_checkpoint_derives_algorithm_identity_from_its_full_config(tmp_path):
    run = make_run(tmp_path)
    ckpt.save(
        tmp_path / "c.pt", run.config, run.actor, run.critic,
        run.actor_opt, run.critic_opt, run.progress,
    )
    payload = ckpt.load(tmp_path / "c.pt")
    payload.pop("algorithm_hash")
    continued = run.config.with_(data_cap=run.config.data_cap * 2)
    other = Run(continued, tmp_path / "legacy", league_name="toy")
    ckpt.restore(payload, continued, other.actor, other.critic)


def test_current_full_uno_configuration_hash_is_pinned():
    """Feature semantics must move the resolved configuration identity."""
    cfg = Config(
        ruleset="FULL_UNO",
        arm=F.ARM_B,
        training_seed=0,
        data_cap=80_000,
        device="cpu",
        collection=Collection(
            games_per_cell=64, live_games_per_worker=64, workers=1
        ),
        evaluation=Evaluation(deals_per_cell=16),
    )
    # Schema 3 makes the 5+ representative and its all-event count explicit.
    # Earlier B checkpoints are intentionally incompatible with this extractor.
    assert cfg.hash == "933facfac719a18f"



def test_iteration_uses_and_restores_runtime_thread_budgets(tmp_path):
    previous = torch.get_num_threads()
    run = make_run(
        tmp_path,
        collection=Collection(games_per_cell=1, live_games_per_worker=4),
    )
    run.execution = {"collection_torch_threads": 1, "update_torch_threads": 2}
    stats, timings = iteration(run)
    assert timings["collection_torch_threads"] == 1
    assert timings["update_torch_threads"] == 2
    assert timings["iteration_fresh_decisions_per_s"] > 0
    assert timings["update_fresh_decisions_per_s"] > 0
    assert timings["actor_presentations"] == run.progress.learner_decisions * stats.actor_epochs
    assert timings["critic_presentations"] == run.progress.learner_decisions * stats.critic_epochs
    assert timings["actor_presentations_per_s"] > 0
    assert timings["critic_presentations_per_s"] > 0
    assert timings["actor_steps"] == stats.actor_steps
    assert timings["critic_steps"] == stats.critic_steps
    assert torch.get_num_threads() == previous


def test_evaluation_uses_and_restores_collection_thread_budget(tmp_path, monkeypatch):
    previous = torch.get_num_threads()
    run = make_run(tmp_path)
    run.execution = {"collection_torch_threads": 1}
    sentinel = object()
    seen = []

    def fake_panel(*args, **kwargs):
        seen.append(torch.get_num_threads())
        return sentinel

    monkeypatch.setattr(ev, "play_panel", fake_panel)
    assert evaluate_now(run) is sentinel
    assert seen == [1]
    assert torch.get_num_threads() == previous


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
    lp, _ = direct.distributions([obs], [__import__("uno.rl.env.evidence", fromlist=["Evidence"]).Evidence.from_public(TOY, s.public, obs.seat ^ 1)])
    exported = agent.policy(obs, legal)
    for a in legal:
        assert abs(exported[a] - float(np.exp(lp[0, a]))) < 1e-6


def test_actor_export_refuses_an_incompatible_schema_or_format(tmp_path):
    run = make_run(tmp_path)
    path = tmp_path / "actor.pt"
    ckpt.export_actor(path, run.config, run.actor, "toy_actor", run.progress)
    payload = torch.load(path, weights_only=False)

    payload["actor_schema"] = "old-schema"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="feature schema differs"):
        ckpt.load_actor(path, TOY)

    payload["actor_schema"] = F.spec(TOY, F.ARM_B, version=2).hash
    torch.save(payload, path)
    legacy, _ = ckpt.load_actor(path, TOY)
    assert legacy.actor.spec.version == 2

    payload["actor_schema"] = F.spec(TOY, F.ARM_B).hash
    payload["format_version"] = ckpt.FORMAT_VERSION + 1
    torch.save(payload, path)
    with pytest.raises(ValueError, match="export format"):
        ckpt.load_actor(path, TOY)


def test_interrupted_continuation_reaches_the_same_state_as_an_uninterrupted_run(tmp_path):
    """Two iterations straight through, against one-then-resume-then-one."""
    straight = make_run(tmp_path / "straight")
    iteration(straight)
    iteration(straight)

    first = make_run(tmp_path / "first")
    iteration(first)
    first.progress.deal_schedule = first.schedule.state
    ckpt.save(tmp_path / "boundary.pt", first.config, first.actor, first.critic, first.actor_opt, first.critic_opt, first.progress, {"league": "toy"})

    from uno.rl.algo.train import resume

    resumed = resume(tmp_path / "boundary.pt", toy_config(), tmp_path / "resumed", league_name="toy")
    iteration(resumed)

    assert resumed.progress.update == straight.progress.update
    assert resumed.progress.next_episode_id == straight.progress.next_episode_id
    assert resumed.progress.learner_decisions == straight.progress.learner_decisions
    assert resumed.schedule.state == straight.schedule.state
    a = torch.cat([p.detach().reshape(-1) for p in resumed.actor.parameters()])
    b = torch.cat([p.detach().reshape(-1) for p in straight.actor.parameters()])
    assert torch.allclose(a, b, atol=1e-6)


def test_resume_carries_the_selected_artifact_and_metadata(tmp_path):
    from uno.rl.algo import train as training

    source = make_run(tmp_path / "source")
    source.progress.best_score = 0.7
    source.progress.best_checkpoint = "checkpoints/best.pt"
    source.progress.best_update = 0
    source.progress.best_decisions = 0
    training._save(source, "best")
    training._save(source, "latest")

    resumed = training.resume(
        source.directory / "checkpoints/latest.pt",
        source.config,
        tmp_path / "resumed-best",
        league_name="toy",
    )
    selected = resumed.directory / resumed.progress.best_checkpoint
    assert selected.exists()
    assert ckpt.verify(selected)
    assert resumed.progress.best_score == 0.7
    assert resumed.progress.best_update == 0


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
    from uno.rl.core.track import provenance

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


def test_diagnostic_panel_is_coupled_and_separate_from_selection_and_training():
    names = list(league(TOY, "toy"))
    panel = mf.diagnostic_panel(names, 8)
    per_cell = {
        c: [e.deal_seed for e in panel.episodes if (e.opponent, e.seat) == c]
        for c in panel.cells
    }
    first = next(iter(per_cell.values()))
    assert all(v == first for v in per_cell.values())
    assert all(d in mf.DIAGNOSTIC_DEALS for d in first)
    assert not set(first) & set(mf.PANEL_DEALS)
    assert not set(first) & set(mf.TRAIN_DEALS)
    with pytest.raises(ValueError):
        mf.diagnostic_panel(names, 2, start=mf.DIAGNOSTIC_DEALS.stop - 1)


def test_diagnostic_panel_stream_ids_do_not_change_when_it_is_extended():
    names = list(league(TOY, "toy"))
    short = mf.diagnostic_panel(names, 8)
    long = mf.diagnostic_panel(names, 16)
    short_ids = {
        (e.opponent, e.seat, e.deal_seed): e.episode_id for e in short.episodes
    }
    long_ids = {
        (e.opponent, e.seat, e.deal_seed): e.episode_id for e in long.episodes
    }
    assert all(long_ids[key] == episode_id for key, episode_id in short_ids.items())


def test_panel_sizing_prefixes_concentrate_near_the_requested_maximum():
    from uno.experiments.rl import _panel_sizes

    assert _panel_sizes(5_760) == (4_032, 4_896, 5_760)
    assert _panel_sizes(256) == (179, 218, 256)


def test_paired_difference_groups_the_cells_of_a_deal(tmp_path):
    run = make_run(tmp_path, evaluation=__import__("uno.rl.core.config", fromlist=["Evaluation"]).Evaluation(deals_per_cell=24))
    result = ev.play_panel(run.panel, run.policy, run.opponents, run.config.training_seed)
    baseline = ev.baseline_panel(run.panel, Heuristic(TOY), run.opponents, TOY, run.config.training_seed)
    assert len(result.deals) == 24 and np.array_equal(result.deals, baseline.deals)
    diff = ev.paired(result, baseline)
    assert abs(diff.difference - (result.mean - baseline.mean)) < 1e-9
    assert diff.n == 24
    # the aggregate is the equal-weight mean of the cells, not a decision-weighted one
    assert abs(result.mean - float(np.mean(list(result.cell_means.values())))) < 1e-12
    assert set(result.behavior.as_dict()) >= {
        "game_draw_rate", "forced_draw_rate", "optional_draw_opportunity_rate",
        "optional_draw_rate", "action_cap_rate", "mean_game_length", "entropy",
        "legal_actions", "max_entropy", "normalized_entropy", "effective_actions",
        "wild_play_rate", "action_card_rate", "action_choice_play_rate",
        "action_only_draw_rate", "wild_choice_play_rate", "wild_only_draw_rate",
    }
    assert baseline.behavior.learner_decisions > 0


def test_draw_behavior_separates_forced_and_optional_draws():
    behavior = ev.Behavior()
    is_play = tuple(isinstance(action, Play) for action in TOY.actions)
    play = next(i for i, value in enumerate(is_play) if value)
    obs = SimpleNamespace(phase=Phase.TURN_START)
    ev._record_action(
        behavior, TOY, obs, (TOY.draw_action,), TOY.draw_action, 0.0, is_play
    )
    ev._record_action(
        behavior, TOY, obs, (TOY.draw_action, play), TOY.draw_action, 0.0, is_play
    )
    ev._record_action(
        behavior, TOY, obs, (TOY.draw_action, play), play, 0.0, is_play
    )
    metrics = behavior.as_dict()
    assert behavior.forced_draws == 1
    assert behavior.optional_draw_opportunities == 2
    assert behavior.optional_draws == 1
    assert metrics["forced_draw_rate"] == pytest.approx(1 / 3)
    assert metrics["optional_draw_opportunity_rate"] == pytest.approx(2 / 3)
    assert metrics["optional_draw_rate"] == 0.5


def test_panel_progress_callbacks_count_completed_games(tmp_path):
    run = make_run(tmp_path)
    panel = mf.diagnostic_panel(list(run.opponents), 4)
    actor_ticks, reference_ticks = [], []
    ev.play_panel(panel, run.policy, run.opponents, 0, live_games=3, on_games=actor_ticks.append)
    ev.baseline_panel(panel, Heuristic(TOY), run.opponents, TOY, 0, on_games=reference_ticks.append)
    assert sum(actor_ticks) == len(panel)
    assert sum(reference_ticks) == len(panel)


def test_paired_difference_refuses_mismatched_panels(tmp_path):
    run = make_run(tmp_path)
    a = ev.play_panel(mf.panel(list(run.opponents), 4), run.policy, run.opponents, 0)
    b = ev.play_panel(mf.panel(list(run.opponents), 4, start=mf.PANEL_DEALS.start + 100), run.policy, run.opponents, 0)
    with pytest.raises(ValueError):
        ev.paired(a, b)


def test_selection_prefers_the_earlier_checkpoint_on_a_tie():
    assert ev.select([(0, 0.5), (1, 0.5000001), (2, 0.4)]) == (0, 0.5)
    assert ev.select([(0, 0.5), (1, 0.6), (2, 0.6)]) == (1, 0.6)


def test_training_reuses_a_boundary_evaluation_as_the_final_result(tmp_path):
    from uno.rl.algo.train import train

    cfg = toy_config(
        data_cap=1,
        collection=Collection(games_per_cell=1, live_games_per_worker=4),
        evaluation=Evaluation(deals_per_cell=2, every_fraction=1.0),
    )
    run = Run(cfg, tmp_path / "deduplicated-final", league_name="toy")
    summary = train(run, verbose=False)
    assert summary["evaluation_points"] == 2  # initial and one update boundary
    assert summary["final_evaluation_reused"] is True
    assert len(run.eval_history) == 2


def test_every_evaluation_path_updates_the_persistent_best(tmp_path, monkeypatch):
    from uno.rl.algo import train as training

    run = make_run(tmp_path)
    saved = []
    monkeypatch.setattr(training, "_save", lambda run, label: saved.append(label))

    first = SimpleNamespace(mean=0.4)
    assert training._record_evaluation(run, None, first, 0, 0.0)
    assert saved == ["best"]
    assert run.progress.best_update == 0
    assert run.progress.best_decisions == 0
    assert run.progress.best_checkpoint == "checkpoints/best.pt"

    run.progress.update = 3
    run.progress.learner_decisions = 123
    assert not training._record_evaluation(
        run, None, SimpleNamespace(mean=0.3), 123, 1.0
    )
    assert saved == ["best"]
    assert training._record_evaluation(
        run, None, SimpleNamespace(mean=0.5), 123, 1.0
    )
    assert saved == ["best", "best"]
    assert run.progress.best_score == 0.5
    assert run.progress.best_update == 3
    assert run.progress.best_decisions == 123


def test_training_progress_bar_tracks_learner_decisions(tmp_path, monkeypatch):
    from uno.rl.algo import train as training

    bars = []

    class RecordingBar:
        def __init__(self, total, desc):
            self.total = total
            self.desc = desc
            self.n = 0
            self.disable = False
            self.postfix = None
            self.closed = False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

        def update(self, n):
            self.n += n

        def set_postfix(self, **kwargs):
            self.postfix = kwargs

        def close(self):
            self.closed = True

    def recording_bar(*args, total, desc, **kwargs):
        progress = RecordingBar(total, desc)
        bars.append(progress)
        return progress

    monkeypatch.setattr(training, "bar", recording_bar)
    cfg = toy_config(
        data_cap=1,
        collection=Collection(games_per_cell=1, live_games_per_worker=4),
        evaluation=Evaluation(deals_per_cell=2, every_fraction=1.0),
    )
    run = Run(cfg, tmp_path / "progress", league_name="toy")
    summary = training.train(run, verbose=True)

    progress = next(p for p in bars if p.desc == "PPO training")
    assert progress.total == 1
    assert progress.n == summary["learner_decisions"]
    assert progress.postfix["update"] == summary["updates"]
    assert {"collect", "ppo", "train", "dev"} <= set(progress.postfix)
    assert progress.closed


def test_first_sigint_stops_after_iteration_and_writes_final_artifacts(
    tmp_path, monkeypatch
):
    from uno.rl.algo import train as training

    cfg = toy_config(
        data_cap=1_000_000,
        collection=Collection(games_per_cell=1, live_games_per_worker=4),
        evaluation=Evaluation(deals_per_cell=2, every_fraction=1.0),
    )
    run = Run(cfg, tmp_path / "interrupted", league_name="toy")
    tracker = Tracker(
        run.directory / "tracking",
        run_name="interrupt-test",
        use_mlflow=False,
    )
    real_iteration = training.iteration

    def interrupt_after_iteration(*args, **kwargs):
        result = real_iteration(*args, **kwargs)
        signal.raise_signal(signal.SIGINT)
        return result

    previous = signal.getsignal(signal.SIGINT)
    monkeypatch.setattr(training, "iteration", interrupt_after_iteration)
    summary = training.train(run, tracker=tracker, verbose=False)

    assert summary["stop_reason"] == "interrupt"
    assert summary["updates"] == 1
    assert summary["evaluation_points"] == 2  # initial plus graceful final
    assert (run.directory / "checkpoints" / "final.pt").exists()
    assert (run.directory / "summary.json").exists()
    tracker_summary = json.loads(
        (run.directory / "tracking" / "run_summary.json").read_text()
    )
    assert tracker_summary["status"] == "FINISHED"
    assert signal.getsignal(signal.SIGINT) is previous


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
    from uno.rl.core.track import EXPERIMENT, local_store
    from uno.rl.algo.train import start_tracker, train

    cfg = toy_config(data_cap=300, evaluation=__import__("uno.rl.core.config", fromlist=["Evaluation"]).Evaluation(deals_per_cell=8))
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
    assert {
        "config_hash", "algorithm_hash", "league_hash", "actor_schema",
        "critic_schema", "stream_version", "base_commit", "mlflow",
    } <= set(stored.data.params)
    metric_names = set(stored.data.metrics)
    assert {
        "primary/dev_score",
        "primary/train_score",
        "primary/iteration_decisions_per_second",
        "primary/entropy",
        "primary/normalized_entropy",
        "primary/effective_actions",
        "primary/approx_kl",
        "primary/critic_pre_update_mc_ev",
        "progress/games",
        "progress/engine_decisions",
        "actor/surrogate",
        "actor/policy_loss",
        "actor/legal_actions",
        "actor/max_entropy",
        "actor/normalized_entropy",
        "actor/effective_actions",
        "critic/mse",
        "critic/value_loss",
        "critic/post_update_mc_ev",
        "critic/grad_clipped_fraction",
        "timing/collection_total_seconds",
        "timing/update_total_seconds",
        "evaluation/score_league_mean",
        "evalbehavior/game_draw_rate",
        "rollout/games",
        "rollout/reward_mean",
        "rollout/engine_length_mean",
    } <= metric_names
    assert all(name.count("/") == 1 for name in metric_names)
    assert not any(name.startswith("diagnostic/") for name in metric_names)
    assert not any(name.startswith("evaluation/behavior_") for name in metric_names)
    assert "surrogate" not in metric_names  # flat names remain CSV-only
    steps = [
        m.step
        for m in client.get_metric_history(
            run_id, "actor/surrogate"
        )
    ]
    assert steps == sorted(steps) and steps[0] > 0  # the step is cumulative learner decisions
    csv_header = (run.directory / "tracking" / "metrics.csv").read_text().splitlines()[0]
    assert "surrogate" in csv_header and "actor/surrogate" not in csv_header
    assert {"run_manifest.json", "summary.json"} <= {a.path for a in client.list_artifacts(run_id)}
    assert stored.info.status == "FINISHED"
