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
from uno.rl import features as F
from uno.rl.collect import collect
from uno.rl.manifest import DealSchedule, build
from uno.rl.model import Actor
from uno.rl.opponents import league
from uno.rl.parallel import ParallelCollector, WorkerConfig
from uno.rl.policy import ActorPolicy


@pytest.mark.slow
def test_parallel_collection_equals_the_single_worker_reference():
    rs = TOY
    torch.manual_seed(0)
    sp, spc = F.spec(rs, F.ARM_B), F.spec(rs, F.ARM_CRITIC)
    actor = Actor(sp, width=SMALL_WIDTH)
    opponents = league(rs, "toy")
    manifest, _ = build(0, list(opponents), DealSchedule(0), 8, 0)
    reference, _ = collect(manifest, ActorPolicy(sp, actor), spc, opponents, 0, 7, live_games=8)

    cfg = WorkerConfig(ruleset="TOY", arm=F.ARM_B, width=SMALL_WIDTH, blocks=2, league="toy", training_seed=0, live_games=8)
    with ParallelCollector(cfg, workers=2, bundle=3) as pool:
        parallel, stats = pool.collect(manifest, actor, spc, behavior_version=7, timeout=120)

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
    cfg = WorkerConfig(ruleset="TOY", arm=F.ARM_B, width=8, blocks=1, league="toy", training_seed=0, live_games=4)
    assert cfg.threads == 1
