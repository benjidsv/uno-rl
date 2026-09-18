"""Persistent collector processes: the same manifest, in parallel.

docs/rl/pipeline-plan.md section 2. Workers are persistent **processes** under an
explicit `spawn` context, not threads: the engine and the feature extractor are
Python, and process startup is kept out of per-update work. Each worker owns its
engine states, its frozen opponents, its incremental public summaries, its action
sampling and a read-only CPU actor replica. Workers never update models and never
write MLflow runs.

Work is queued in **small dynamic bundles** rather than partitioned up front, so a
slow opponent cell is not bound to one worker. Which worker runs which bundle is
therefore not reproducible -- but the *data* is: episodes carry their own logical
streams (`uno.rl.streams`), the coordinator reassembles in logical ID order, and
`tests/rl/test_parallel.py` asserts that 1 worker and N workers produce the same
rollout. Worker count is an execution setting, not a data-allocation setting.

Transfers are plain contiguous arrays, one chunk per bundle, bounded by the bundle
size. A shared-memory buffer pool is a later optimization to make only if a
measured transfer cost justifies it; no such measurement exists yet.

**Failure handling.** A bundle whose worker died is retried -- in the coordinator,
with the same logical streams, so the retry produces the same episodes. Duplicate
episode results are rejected by `uno.rl.rollout.Builder`. A whole iteration is
replayed from the last saved boundary if recovery is uncertain.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import time
from dataclasses import dataclass

import torch

from uno.game.rules import NAMED
from uno.rl.collect import Stats, collect
from uno.rl.features import Features, spec
from uno.rl.manifest import Manifest
from uno.rl.model import Actor
from uno.rl.opponents import league
from uno.rl.policy import ActorPolicy
from uno.rl.rollout import Builder, Rollout

BUNDLE = 8  # episodes per queued bundle
STOP = "stop"


def _payload(rollout: Rollout) -> dict:
    """A bundle's result as plain arrays: no Spec, no Ruleset, no engine states."""
    return {
        "tok": rollout.actor.tok, "col": rollout.actor.col, "glob": rollout.actor.glob,
        "ctok": rollout.critic.tok, "ccol": rollout.critic.col, "cglob": rollout.critic.glob,
        "masks": rollout.masks, "actions": rollout.actions, "old_logp": rollout.old_logp,
        "decision_index": rollout.decision_index, "episode": rollout.episode,
        "episode_id": rollout.episode_id, "terminal_return": rollout.terminal_return,
        "end_reason": rollout.end_reason, "engine_decisions": rollout.engine_decisions,
        "learner_decisions": rollout.learner_decisions, "offset": rollout.offset,
    }  # fmt: skip


def _merge(builder: Builder, payload: dict, episodes: dict) -> None:
    """Add a bundle's rows and episodes to the coordinator's builder."""
    for i, eid in enumerate(payload["episode_id"].tolist()):
        start, n = int(payload["offset"][i]), int(payload["learner_decisions"][i])
        for j in range(start, start + n):
            actor = Features(payload["tok"][j : j + 1], payload["col"][j : j + 1], payload["glob"][j : j + 1])
            critic = Features(payload["ctok"][j : j + 1], payload["ccol"][j : j + 1], payload["cglob"][j : j + 1])
            builder.add_decision(eid, int(payload["decision_index"][j]), actor, critic, payload["masks"][j], int(payload["actions"][j]), float(payload["old_logp"][j]))
        builder.finish_episode(episodes[eid], float(payload["terminal_return"][i]), payload["end_reason"][i], int(payload["engine_decisions"][i]))


@dataclass(frozen=True)
class WorkerConfig:
    ruleset: str
    arm: str
    width: int
    blocks: int
    league: str
    training_seed: int
    live_games: int
    threads: int = 1  # BLAS/torch threads per collector: workers x threads must not oversubscribe


def _worker(cfg: WorkerConfig, tasks, results, ctl) -> None:
    """One persistent collector process."""
    torch.set_num_threads(max(1, cfg.threads))
    os.environ.setdefault("OMP_NUM_THREADS", str(max(1, cfg.threads)))
    rs = NAMED[cfg.ruleset]
    actor_spec, critic_spec = spec(rs, cfg.arm), spec(rs, "CRITIC")
    actor = Actor(actor_spec, cfg.width, cfg.blocks)
    policy = ActorPolicy(actor_spec, actor)
    opponents = league(rs, cfg.league)
    while True:
        message = ctl.get()
        if message[0] == STOP:
            return
        _, version, state = message
        actor.load_state_dict(state)
        actor.eval()
        while True:
            item = tasks.get()
            if item is None:
                results.put(("done", version, None, None))
                break
            bundle_id, episodes = item
            sub = Manifest(-1, tuple(episodes), (), 0)
            rollout, stats = collect(sub, policy, critic_spec, opponents, cfg.training_seed, version, cfg.live_games)
            results.put(("bundle", bundle_id, _payload(rollout), stats))


class ParallelCollector:
    """A pool of persistent collector processes, reused across iterations."""

    def __init__(self, cfg: WorkerConfig, workers: int, bundle: int = BUNDLE) -> None:
        self.cfg, self.workers, self.bundle = cfg, workers, bundle
        self._ctx = mp.get_context("spawn")
        self._tasks = self._ctx.Queue()
        self._results = self._ctx.Queue()
        self._ctl = [self._ctx.Queue() for _ in range(workers)]
        self._procs = [self._ctx.Process(target=_worker, args=(cfg, self._tasks, self._results, self._ctl[i]), daemon=True) for i in range(workers)]
        for p in self._procs:
            p.start()

    def close(self) -> None:
        for q in self._ctl:
            q.put((STOP,))
        for p in self._procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()

    def __enter__(self) -> ParallelCollector:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def collect(self, manifest: Manifest, actor: Actor, critic_spec, behavior_version: int, timeout: float = 600.0) -> tuple[Rollout, Stats]:
        rs = NAMED[self.cfg.ruleset]
        actor_spec = spec(rs, self.cfg.arm)
        builder = Builder(actor_spec, critic_spec, behavior_version)
        stats = Stats()
        t0 = time.perf_counter()
        by_id = {e.episode_id: e for e in manifest.episodes}

        state = {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}
        for q in self._ctl:
            q.put(("weights", behavior_version, state))
        bundles = {}
        episodes = list(manifest.episodes)
        for i in range(0, len(episodes), self.bundle):
            bundles[i // self.bundle] = episodes[i : i + self.bundle]
            self._tasks.put((i // self.bundle, episodes[i : i + self.bundle]))
        for _ in self._procs:
            self._tasks.put(None)

        outstanding = set(bundles)
        finished_workers = 0
        deadline = time.perf_counter() + timeout
        while finished_workers < len(self._procs):
            try:
                kind, a, payload, worker_stats = self._results.get(timeout=1.0)
            except queue.Empty:
                # A dead pool would otherwise be waited on for the whole timeout.
                if not any(p.is_alive() for p in self._procs) or time.perf_counter() > deadline:
                    break
                continue
            if kind == "done":
                finished_workers += 1
                continue
            _merge(builder, payload, by_id)
            outstanding.discard(a)
            _accumulate(stats, worker_stats)

        retried = sorted(outstanding)
        if retried:
            # A worker died or timed out. Retry its logical work here: the episodes
            # carry their own streams, so the retry is the same data.
            local = ActorPolicy(actor_spec, actor)
            for bundle_id in retried:
                sub = Manifest(-1, tuple(bundles[bundle_id]), (), 0)
                rollout, worker_stats = collect(sub, local, critic_spec, league(rs, self.cfg.league), self.cfg.training_seed, behavior_version, self.cfg.live_games)
                _merge(builder, _payload(rollout), by_id)
                _accumulate(stats, worker_stats)
        stats.seconds = time.perf_counter() - t0
        stats.retried_bundles = len(retried)
        return builder.finish(expected=len(manifest)), stats


def _accumulate(total: Stats, part: Stats) -> None:
    total.games += part.games
    total.learner_decisions += part.learner_decisions
    total.engine_decisions += part.engine_decisions
    total.inference_calls += part.inference_calls
    total.inference_rows += part.inference_rows
    total.batch_sizes += part.batch_sizes
    total.drain_seconds = max(total.drain_seconds, part.drain_seconds)
    total.worker_seconds += part.seconds
    for k, v in part.ends.items():
        total.ends[k] = total.ends.get(k, 0) + v


__all__ = ["BUNDLE", "ParallelCollector", "WorkerConfig"]
