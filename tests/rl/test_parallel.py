"""Stage 3 acceptance: the parallel collector produces the reference's rollout.

docs/rl/pipeline-plan.md section 13: "Same logical manifests/streams at 1 versus
multiple workers; no missing/duplicate games; no completion-order selection;
bounded memory". Spawning processes is slow, so this is one end-to-end test on the
smallest ruleset rather than a matrix.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.rl.helpers import SMALL_WIDTH
from uno.game.rules import TOY
from uno.rl.env import features as F
from uno.rl.algo.collect import collect
from uno.rl.core.manifest import DealSchedule, build
from uno.rl.model import make_actor
from uno.rl.eval.opponents import league
from uno.rl.algo.parallel import ParallelCollector, WorkerConfig
from uno.rl.algo.policy import ActorPolicy


@pytest.mark.slow
def test_parallel_collection_equals_the_single_worker_reference():
    rs = TOY
    torch.manual_seed(0)
    sp, spc = F.spec(rs, F.ARM_B), F.spec(rs, F.ARM_CRITIC)
    from uno.rl.core.config import RelationalModel
    actor = make_actor(sp, RelationalModel(width=SMALL_WIDTH, blocks=2))
    opponents = league(rs, "toy")
    manifest, _ = build(0, list(opponents), DealSchedule(0), 8, 0)
    reference, _ = collect(
        manifest, ActorPolicy(sp, actor), spc, opponents, 0, 7, live_games=8
    )

    cfg = WorkerConfig(
        ruleset="TOY",
        arm=F.ARM_B,
        model_config={"kind": "relational", "width": SMALL_WIDTH, "blocks": 2},
        league="toy",
        training_seed=0,
        live_games=8,
        temperature=1.0,
    )
    with ParallelCollector(cfg, workers=2, bundle=3) as pool:
        parallel, stats = pool.collect(
            manifest, actor, spc, behavior_version=7, timeout=120
        )

    assert stats.retried_bundles == 0
    assert stats.transfer_bytes > parallel.nbytes
    assert stats.serialization_seconds >= 0
    assert stats.deserialization_seconds >= 0
    assert stats.merge_seconds >= 0
    assert np.array_equal(reference.episode_id, parallel.episode_id)
    assert np.array_equal(reference.terminal_return, parallel.terminal_return)
    assert np.array_equal(reference.actions, parallel.actions)
    assert np.array_equal(reference.decision_index, parallel.decision_index)
    assert np.allclose(reference.old_logp, parallel.old_logp, atol=1e-6)
    assert np.array_equal(reference.actor.tok, parallel.actor.tok)
    assert np.array_equal(reference.critic.glob, parallel.critic.glob)
    assert parallel.n_games == len(manifest)


def test_worker_config_names_its_thread_budget():
    """Numerical-library threads times worker count must not silently oversubscribe."""
    cfg = WorkerConfig(
        ruleset="TOY",
        arm=F.ARM_B,
        model_config={"kind": "relational", "width": 8, "blocks": 1},
        league="toy",
        training_seed=0,
        live_games=4,
        temperature=1.0,
    )
    assert cfg.threads == 1


@pytest.mark.slow
def test_parallel_evaluation_equals_the_single_process_panel():
    """The DEV panel split across workers plays the same games: every draw is keyed
    by episode and decision, so scores and behavior counters match `play_panel`."""
    from uno.rl.core.manifest import panel
    from uno.rl.eval.evaluate import play_panel

    rs = TOY
    torch.manual_seed(1)
    sp = F.spec(rs, F.ARM_B)
    from uno.rl.core.config import RelationalModel
    actor = make_actor(sp, RelationalModel(width=SMALL_WIDTH, blocks=2))
    opponents = league(rs, "toy")
    manifest = panel(list(opponents), 12)
    reference = play_panel(manifest, ActorPolicy(sp, actor), opponents, 0, live_games=8)

    cfg = WorkerConfig(
        ruleset="TOY",
        arm=F.ARM_B,
        model_config={"kind": "relational", "width": SMALL_WIDTH, "blocks": 2},
        league="toy",
        training_seed=0,
        live_games=8,
        temperature=1.0,
    )
    seen = []
    with ParallelCollector(cfg, workers=3, bundle=5) as pool:
        result = pool.evaluate(manifest, actor, version=0, on_games=seen.append, timeout=120)
        shared = pool.evaluate(manifest, actor, version=0, timeout=120,
                               local_policy=ActorPolicy(sp, actor), local_share=0.34)  # fmt: skip
        # the same pool then collects: task kinds do not leak between calls
        rollout, _ = pool.collect(build(0, list(opponents), DealSchedule(0), 2, 0)[0], actor, F.spec(rs, F.ARM_CRITIC), 0, timeout=120)

    assert sum(seen) == len(manifest)
    assert rollout.n_games == 2 * len(manifest.cells)
    assert np.array_equal(result.deals, reference.deals)
    ref_b = reference.behavior.as_dict()
    for played in (result, shared):  # all on the pool, and split with the caller
        assert np.array_equal(played.deals, reference.deals)
        for cell in manifest.cells:
            assert np.array_equal(played.scores[cell], reference.scores[cell])
        par_b = played.behavior.as_dict()
        assert ref_b.keys() == par_b.keys()
        for k in ref_b:
            assert par_b[k] == pytest.approx(ref_b[k], rel=1e-9, abs=1e-12), k


@pytest.mark.slow
def test_overlapped_collection_trains_the_same_run_as_sequential_iterations(tmp_path):
    """Collecting k+1 while critic k trains changes only timing: parameters,
    counters and the deal schedule match the sequential order exactly. A
    prefetch that is then discarded commits nothing."""
    from uno.rl.core.config import Config, Optim, RelationalModel
    from uno.rl.algo.train import Run, _discard_prefetch, iteration

    cfg = Config(ruleset="TOY", arm=F.ARM_B, model=RelationalModel(width=SMALL_WIDTH, blocks=2), training_seed=0,
                 data_cap=2000, optim=Optim(n_minibatches=8, actor_epochs=2, critic_epochs=2))  # fmt: skip
    wc = WorkerConfig(ruleset="TOY", arm=F.ARM_B, model_config={"kind": "relational", "width": SMALL_WIDTH, "blocks": 2},
                      league="toy", training_seed=0, live_games=8, temperature=1.0)  # fmt: skip
    sequential = Run(cfg, tmp_path / "seq", league_name="toy")
    overlapped = Run(cfg, tmp_path / "ovl", league_name="toy")
    with ParallelCollector(wc, workers=2, bundle=4) as pool:
        for _ in range(3):
            iteration(sequential, pool)
        prefetched = []
        for _ in range(3):
            _, extra = iteration(overlapped, pool, continue_after=lambda d: True)
            prefetched.append(extra["prefetched"])
        assert prefetched == [0, 1, 1]
        assert overlapped.prefetch is not None  # a fourth collection is in flight
        _discard_prefetch(overlapped)

    for a, b in ((sequential, overlapped),):
        assert a.progress.update == b.progress.update == 3
        assert a.progress.next_episode_id == b.progress.next_episode_id
        assert a.progress.learner_decisions == b.progress.learner_decisions
        assert a.schedule.state == b.schedule.state
        for x, y in zip(a.actor.parameters(), b.actor.parameters()):
            assert torch.equal(x, y)
        for x, y in zip(a.critic.parameters(), b.critic.parameters()):
            assert torch.equal(x, y)
