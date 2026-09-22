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

Transfers use named shared-memory blocks for NumPy arrays and a small serialized
descriptor over the result queue, one chunk per bundle. The result queue is
bounded to one pending descriptor per worker, and the coordinator records block
creation, backpressure, waiting, mapping/copying, byte volume and assembly
separately. A block is unlinked after the coordinator has copied its arrays.

**Failure handling.** A bundle whose worker died is retried -- in the coordinator,
with the same logical streams, so the retry produces the same episodes. Duplicate
episode results are rejected by `uno.rl.rollout.Builder`. A whole iteration is
replayed from the last saved boundary if recovery is uncertain.
"""

from __future__ import annotations

import math
import multiprocessing as mp
from multiprocessing import shared_memory
import os
import queue
import signal
import time
from dataclasses import dataclass

import torch
import numpy as np

from uno.game.rules import NAMED
from uno.rl.collect import Stats, collect
from uno.rl.features import spec
from uno.rl.manifest import Manifest
from uno.rl.model import make_actor
from uno.rl.opponents import league
from uno.rl.policy import ActorPolicy
from uno.rl.rollout import Builder, Rollout

TARGET_BUNDLES_PER_WORKER = 3  # measured
STOP = "stop"


def _payload(rollout: Rollout) -> dict:
    """A bundle's result as plain arrays: no Spec, no Ruleset, no engine states."""
    return {
        "tok": rollout.actor.tok, "col": rollout.actor.col, "glob": rollout.actor.glob,
        "ctok": rollout.critic.tok, "ccol": rollout.critic.col, "cglob": rollout.critic.glob,
        "masks": rollout.masks, "actions": rollout.actions, "old_logp": rollout.old_logp,
        "decision_index": rollout.decision_index, "episode": rollout.episode, "set_id": rollout.set_id,
        "episode_id": rollout.episode_id, "terminal_return": rollout.terminal_return,
        "end_reason": rollout.end_reason, "engine_decisions": rollout.engine_decisions,
        "learner_decisions": rollout.learner_decisions, "offset": rollout.offset,
    }  # fmt: skip


_ARRAY_FIELDS = (
    "tok",
    "col",
    "glob",
    "ctok",
    "ccol",
    "cglob",
    "masks",
    "actions",
    "old_logp",
    "decision_index",
    "episode",
    "set_id",
    "episode_id",
    "terminal_return",
    "learner_decisions",
    "offset",
    "engine_decisions",
)


def _shared_payload(rollout: Rollout) -> tuple[dict, list[shared_memory.SharedMemory]]:
    """Copy a bundle's arrays into named shared-memory blocks."""
    plain = _payload(rollout)
    descriptors = {}
    handles = []
    transferred = 0
    for name in _ARRAY_FIELDS:
        array = plain[name]
        shm = shared_memory.SharedMemory(create=True, size=max(1, array.nbytes))
        if array.nbytes:
            np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)[...] = array
        descriptors[name] = (shm.name, array.shape, array.dtype.str, array.nbytes)
        handles.append(shm)
        transferred += array.nbytes
    return {
        "shared": descriptors,
        "bytes": transferred,
        "end_reason": plain["end_reason"],
    }, handles


def _load_shared_payload(payload: dict) -> dict:
    """Map, copy, close, and unlink one shared-memory result bundle."""
    result = {"end_reason": payload["end_reason"]}
    opened = {}
    try:
        for field, (name, shape, dtype, nbytes) in payload["shared"].items():
            opened[field] = shared_memory.SharedMemory(name=name)
            source = np.ndarray(shape, dtype=np.dtype(dtype), buffer=opened[field].buf)
            result[field] = (
                source.copy() if nbytes else np.empty(shape, dtype=np.dtype(dtype))
            )
        return result
    finally:
        for shm in opened.values():
            shm.close()
            try:
                shm.unlink()
            except FileNotFoundError:
                pass


def _merge(builder: Builder, payload: dict, episodes: dict) -> None:
    """Add a whole bundle without retaining one set of views per decision."""
    builder.add_payload(payload, episodes)


@dataclass(frozen=True)
class WorkerConfig:
    ruleset: str
    arm: str
    model_config: dict
    league: str
    training_seed: int
    live_games: int
    temperature: float
    threads: int = (
        1  # BLAS/torch threads per collector: workers x threads must not oversubscribe
    )


def _worker(cfg: WorkerConfig, tasks, results, ctl) -> None:
    """One persistent collector process."""
    # Ignore SIGINT in workers: the coordinator handles Ctrl-C and waits for workers to end
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    torch.set_num_threads(max(1, cfg.threads))
    os.environ.setdefault("OMP_NUM_THREADS", str(max(1, cfg.threads)))
    rs = NAMED[cfg.ruleset]
    actor_spec, critic_spec = spec(rs, cfg.arm), spec(rs, "CRITIC")
    from uno.rl.config_io import model_from_dict

    model_config = model_from_dict(cfg.model_config)
    actor = make_actor(actor_spec, model_config)
    policy = ActorPolicy(actor_spec, actor, temperature=cfg.temperature, device="cpu")
    opponents = league(rs, cfg.league)

    # On Windows, shared memory is tied to process handles. If a worker closes
    # its handle before the coordinator opens it, the OS destroys the memory block.
    # We accumulate handles and close them at the start of the next iteration,
    # long after the coordinator is guaranteed to have mapped and unlinked them.
    handles_to_close: list[shared_memory.SharedMemory] = []

    while True:
        message = ctl.get()

        for shm in handles_to_close:
            shm.close()
        handles_to_close.clear()

        if message[0] == STOP:
            return
        _, version, state = message
        actor.load_state_dict(state)
        actor.eval()
        queue_put_seconds = 0.0
        while True:
            item = tasks.get()
            if item is None:
                results.put(("done", version, None, queue_put_seconds))
                break
            bundle_id, episodes = item
            sub = Manifest(-1, tuple(episodes), (), 0)
            rollout, stats = collect(
                sub,
                policy,
                critic_spec,
                opponents,
                cfg.training_seed,
                version,
                cfg.live_games,
            )
            started = time.perf_counter()
            payload, handles = _shared_payload(rollout)
            stats.serialization_seconds = time.perf_counter() - started
            started = time.perf_counter()
            sent = False
            try:
                results.put(("bundle", bundle_id, payload, stats))
                sent = True
            finally:
                if not sent:
                    # If we failed to send, we must clean up ourselves
                    for shm in handles:
                        shm.close()
                        try:
                            shm.unlink()
                        except FileNotFoundError:
                            pass
                else:
                    # Delay closing handles until the next iteration to avoid Windows destroying them prematurely
                    handles_to_close.extend(handles)
            queue_put_seconds += time.perf_counter() - started
            del rollout, payload, handles


class ParallelCollector:
    """A pool of persistent collector processes, reused across iterations."""

    def __init__(
        self,
        cfg: WorkerConfig,
        workers: int,
        bundle: int | None = None,
    ) -> None:
        self.cfg, self.workers, self.bundle = cfg, workers, bundle
        self._ctx = mp.get_context("spawn")
        self._tasks = self._ctx.Queue()
        # One queued payload per worker bounds the coordinator-side backlog. A
        # worker that outruns ingestion blocks here instead of allowing rollout
        # arrays and multiprocessing serialization buffers to grow without limit.
        self._results = self._ctx.Queue(maxsize=max(1, workers))
        self._ctl = [self._ctx.Queue() for _ in range(workers)]
        self._procs = [
            self._ctx.Process(
                target=_worker,
                args=(cfg, self._tasks, self._results, self._ctl[i]),
                daemon=True,
            )
            for i in range(workers)
        ]
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

    def collect(
        self,
        manifest: Manifest,
        actor: nn.Module,
        critic_spec,
        behavior_version: int,
        timeout: float = 600.0,
    ) -> tuple[Rollout, Stats]:
        rs = NAMED[self.cfg.ruleset]
        actor_spec = spec(rs, self.cfg.arm)
        builder = Builder(
            actor_spec,
            critic_spec,
            behavior_version,
            decision_capacity=max(1024, len(manifest) * 16),
        )
        stats = Stats()
        t0 = time.perf_counter()
        by_id = {e.episode_id: e for e in manifest.episodes}

        state = {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}
        for q in self._ctl:
            q.put(("weights", behavior_version, state))

        # Adaptive bundle sizing targetting around three per worker, never more than live games, min at 8
        # I measured this empirically and it gives good load balancing
        # I also rounded to nearest mutiple of 8 in order not to break anything but it's probably not necessary
        if self.bundle is None:
            bundle = max(
                8,
                min(
                    self.cfg.live_games,
                    math.ceil(len(manifest) / (self.workers * 3)),
                ),
            )
        else:
            bundle = self.bundle
        bundles = {}
        episodes = list(manifest.episodes)
        for i in range(0, len(episodes), bundle):
            bundles[i // bundle] = episodes[i : i + bundle]
            self._tasks.put((i // bundle, episodes[i : i + bundle]))
        for _ in self._procs:
            self._tasks.put(None)

        outstanding = set(bundles)
        finished_workers = 0
        deadline = time.perf_counter() + timeout
        while finished_workers < len(self._procs):
            wait_started = time.perf_counter()
            try:
                kind, a, payload, worker_stats = self._results.get(timeout=1.0)
            except queue.Empty:
                stats.result_queue_wait_seconds += time.perf_counter() - wait_started
                # A dead pool would otherwise be waited on for the whole timeout.
                if (
                    not any(p.is_alive() for p in self._procs)
                    or time.perf_counter() > deadline
                ):
                    break
                continue
            stats.result_queue_wait_seconds += time.perf_counter() - wait_started
            if kind == "done":
                finished_workers += 1
                stats.result_queue_put_seconds += float(worker_stats)
                continue
            stats.transfer_bytes += int(payload.get("bytes", len(payload)))
            started = time.perf_counter()
            decoded = _load_shared_payload(payload) if "shared" in payload else payload
            stats.deserialization_seconds += time.perf_counter() - started
            started = time.perf_counter()
            _merge(builder, decoded, by_id)
            stats.merge_seconds += time.perf_counter() - started
            outstanding.discard(a)
            _accumulate(stats, worker_stats)

        retried = sorted(outstanding)
        if retried:
            # A worker died or timed out. Retry its logical work here: the episodes
            # carry their own streams, so the retry is the same data.
            local = ActorPolicy(
                actor_spec, actor, temperature=self.cfg.temperature, device="cpu"
            )
            for bundle_id in retried:
                sub = Manifest(-1, tuple(bundles[bundle_id]), (), 0)
                rollout, worker_stats = collect(
                    sub,
                    local,
                    critic_spec,
                    league(rs, self.cfg.league),
                    self.cfg.training_seed,
                    behavior_version,
                    self.cfg.live_games,
                )
                _merge(builder, _payload(rollout), by_id)
                _accumulate(stats, worker_stats)
        stats.seconds = time.perf_counter() - t0
        stats.retried_bundles = len(retried)
        started = time.perf_counter()
        rollout = builder.finish(expected=len(manifest))
        stats.merge_seconds += time.perf_counter() - started
        return rollout, stats


def _accumulate(total: Stats, part: Stats) -> None:
    total.games += part.games
    total.learner_decisions += part.learner_decisions
    total.engine_decisions += part.engine_decisions
    total.inference_calls += part.inference_calls
    total.inference_rows += part.inference_rows
    total.batch_sizes += part.batch_sizes
    total.drain_seconds = max(total.drain_seconds, part.drain_seconds)
    total.simulation_seconds += part.simulation_seconds
    total.feature_seconds += part.feature_seconds
    total.inference_seconds += part.inference_seconds
    total.assembly_seconds += part.assembly_seconds
    total.worker_seconds += part.seconds
    total.serialization_seconds += part.serialization_seconds
    for k, v in part.ends.items():
        total.ends[k] = total.ends.get(k, 0) + v


__all__ = ["BUNDLE", "ParallelCollector", "WorkerConfig"]
