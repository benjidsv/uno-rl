"""The reference collector: many live games per worker, one batched actor call.

docs/rl/pipeline-plan.md section 2: a worker "advances several live games, batches
ready learner observations, runs one actor call, samples actions, and advances
games through opponent decisions to the next learner decision or terminal state".
Consecutive learner turns (a skip or reverse grants another turn in two-player
play) and after-draw choices need no alternating-seat assumption: a game simply
reports whether it is waiting on the learner.

This module is the single-process reference implementation and the oracle for
`uno.rl.algo.parallel`. It keeps the immutable engine as the authority for legality,
transitions and terminal returns, and it plays **every assigned game to a true
terminal state**: a ruleset action cap is a terminal result, while a process
interruption or a wall-time cutoff is not, and is never recorded as a draw.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from uno.agents.baseline import Agent
from uno.game.engine import State, legal_actions, new_game, returns, step
from uno.game.observe import Level, observe
from uno.game.rules import Ruleset
from uno.rl.env.evidence import Evidence
from uno.rl.env.features import Spec, extract, legal_mask
from uno.rl.core.manifest import Episode, Manifest
from uno.rl.algo.policy import ActorPolicy
from uno.rl.algo.rollout import Builder, Rollout
from uno.rl.core.streams import ACTOR, OPPONENT, uniform

MAX_FAILURE_SAMPLES = 16  # a bounded replay sample, kept outside the training buffer


@dataclass
class Stats:
    """What one collection pass measured (pipeline plan section 7)."""

    games: int = 0
    learner_decisions: int = 0
    engine_decisions: int = 0
    seconds: float = 0.0
    inference_calls: int = 0
    inference_rows: int = 0
    batch_sizes: list[int] = field(default_factory=list)
    drain_seconds: float = 0.0  # time after the last game started, the straggler tail
    simulation_seconds: float = 0.0
    feature_seconds: float = 0.0
    inference_seconds: float = 0.0
    assembly_seconds: float = 0.0
    ends: dict[str, int] = field(default_factory=dict)
    # parallel collection only: worker stage times overlap, so their sum is not
    # elapsed wall time (pipeline plan section 7) and is reported separately.
    worker_seconds: float = 0.0
    retried_bundles: int = 0
    serialization_seconds: float = 0.0
    result_queue_put_seconds: float = 0.0
    result_queue_wait_seconds: float = 0.0
    deserialization_seconds: float = 0.0
    merge_seconds: float = 0.0
    transfer_bytes: int = 0

    @property
    def decisions_per_second(self) -> float:
        return self.learner_decisions / self.seconds if self.seconds else 0.0

    @property
    def mean_batch(self) -> float:
        return self.inference_rows / self.inference_calls if self.inference_calls else 0.0

    def __str__(self) -> str:
        ends = " ".join(f"{k}={v}" for k, v in sorted(self.ends.items()))
        return (
            f"{self.games} games, {self.learner_decisions:,} learner / {self.engine_decisions:,} engine decisions in {self.seconds:.2f}s "
            f"({self.decisions_per_second:,.0f} learner dec/s), mean inference batch {self.mean_batch:.1f}, drain {self.drain_seconds:.2f}s, {ends}"
        )


class LiveGame:
    """One assigned episode in flight, with its own engine state and evidence."""

    __slots__ = ("episode", "state", "seat", "opponent", "evidence", "seen", "decision_index", "opponent_decisions", "engine_decisions")

    def __init__(self, rs: Ruleset, episode: Episode, opponent: Agent) -> None:
        try:
            self.state: State = new_game(rs, episode.deal_seed)
        except ValueError as exc:  # no number card for the first discard
            raise ValueError(
                f"deal seed {episode.deal_seed} is not a valid {rs.colors}-color deal ({exc}); "
                "filter the deal pool before building a manifest rather than dropping an assigned game"
            ) from exc
        self.episode = episode
        self.seat = episode.seat
        self.opponent = opponent
        self.evidence = Evidence(rs, episode.seat ^ 1)
        self.seen = 0
        self.decision_index = 0
        self.opponent_decisions = 0
        self.engine_decisions = 0

    @property
    def terminal(self) -> bool:
        return self.state.terminal

    def advance_to_learner(self, training_seed: int) -> None:
        """Play opponent decisions until the learner is to move, or the game ends."""
        while not self.state.terminal and self.state.player != self.seat:
            legal = legal_actions(self.state)
            probs = self.opponent.policy(observe(self.state, self.state.player, Level.L1), legal)
            u = uniform(training_seed, OPPONENT, self.episode.episode_id, self.opponent_decisions)
            self.state = step(self.state, sample_action(probs, u), check=False)
            self.opponent_decisions += 1
            self.engine_decisions += 1

    def observations(self):
        """The learner's own L1 and L2 observations at the current decision."""
        self.seen = self.evidence.update(self.state.public, self.seen)
        return observe(self.state, self.seat, Level.L1), observe(self.state, self.seat, Level.L2)

    def apply(self, action: int) -> None:
        self.state = step(self.state, action, check=False)
        self.decision_index += 1
        self.engine_decisions += 1

    def result(self) -> tuple[float, str]:
        return float(returns(self.state)[self.seat]), str(self.state.end)


def sample_action(probs: dict[int, float], u: float) -> int:
    """Inverse-CDF sampling from an agent's distribution, in its own key order."""
    acc = 0.0
    last = None
    for a, p in probs.items():
        acc += p
        last = a
        if u < acc:
            return a
    return last  # rounding


def collect(
    manifest: Manifest,
    actor: ActorPolicy,
    critic_spec: Spec,
    opponents: dict[str, Agent],
    training_seed: int,
    behavior_version: int,
    live_games: int = 64,
    failures: list | None = None,
    set_id_fn=None,
) -> tuple[Rollout, Stats]:
    """Collect every episode of `manifest` under the frozen actor, and assemble it.

    Returns the rollout in logical episode order plus the pass's `Stats`. Raises if
    a game is missing or duplicated: completing the manifest is the contract, and a
    long game is never silently dropped to keep the rest of its iteration.
    """
    rs = actor.rs
    spec = actor.spec
    builder = Builder(spec, critic_spec, behavior_version, decision_capacity=max(1024, len(manifest) * 16))
    stats = Stats()
    t0 = time.perf_counter()

    queue = list(manifest.episodes)
    started = 0
    active: list[LiveGame] = []
    last_start = t0

    def start_next() -> None:
        nonlocal started, last_start
        while queue and len(active) < live_games:
            episode = queue.pop(0)
            if episode.opponent not in opponents:
                raise KeyError(f"episode {episode.episode_id} names opponent {episode.opponent!r}, which was not supplied")
            active.append(LiveGame(rs, episode, opponents[episode.opponent]))
            started += 1
            last_start = time.perf_counter()

    while True:
        # Refill first, then advance: a game started this pass must reach its own
        # first learner decision before it can join an inference batch.
        simulation_started = time.perf_counter()
        start_next()
        stats.simulation_seconds += time.perf_counter() - simulation_started
        if not active:
            break
        ready: list[LiveGame] = []
        done: list[LiveGame] = []
        simulation_started = time.perf_counter()
        for game in active:
            game.advance_to_learner(training_seed)
            (done if game.terminal else ready).append(game)
        finished = [(game, *game.result()) for game in done]
        stats.simulation_seconds += time.perf_counter() - simulation_started
        assembly_started = time.perf_counter()
        for game, value, reason in finished:
            builder.finish_episode(game.episode, value, reason, game.engine_decisions)
        stats.assembly_seconds += time.perf_counter() - assembly_started
        for game, value, reason in finished:
            stats.games += 1
            stats.engine_decisions += game.engine_decisions
            stats.ends[reason] = stats.ends.get(reason, 0) + 1
        active = ready
        if not active:
            continue
        feature_started = time.perf_counter()
        pairs = [g.observations() for g in active]
        actor_rows = extract(spec, [p[0] for p in pairs], [g.evidence for g in active] if spec.has_evidence else None)
        critic_rows = extract(critic_spec, [p[1] for p in pairs], [g.evidence for g in active])
        masks = np.stack([legal_mask(rs, p[0]) for p in pairs])
        set_ids = None if set_id_fn is None else np.fromiter((set_id_fn(p[0]) for p in pairs), dtype=np.int64, count=len(pairs))
        stats.feature_seconds += time.perf_counter() - feature_started
        inference_started = time.perf_counter()
        log_probs = actor.log_probs(actor_rows, masks, set_ids)
        stats.inference_calls += 1
        stats.inference_rows += len(active)
        stats.batch_sizes.append(len(active))
        us = np.fromiter((uniform(training_seed, ACTOR, g.episode.episode_id, g.decision_index) for g in active), dtype=np.float64, count=len(active))
        chosen = actor.sample(log_probs, us)
        stats.inference_seconds += time.perf_counter() - inference_started
        assembly_started = time.perf_counter()
        for i, game in enumerate(active):
            a = int(chosen[i])
            if not masks[i, a]:
                raise ValueError(f"sampled an illegal action {a} in episode {game.episode.episode_id}")
            set_id = -1 if set_ids is None else int(set_ids[i])
            builder.add_decision(game.episode.episode_id, game.decision_index, actor_rows[i : i + 1], critic_rows[i : i + 1], masks[i], a, float(log_probs[i, a]), set_id)
            stats.learner_decisions += 1
            if failures is not None and len(failures) < MAX_FAILURE_SAMPLES and game.decision_index == 0:
                # A small, explicitly bounded replay sample, outside the training
                # buffer: enough to reproduce a decision, never thousands of traces.
                failures.append({
                    "episode_id": game.episode.episode_id, "deal_seed": game.episode.deal_seed,
                    "seat": game.seat, "opponent": game.episode.opponent, "public": game.state.public,
                    "legal": tuple(np.nonzero(masks[i])[0].tolist()), "action": a,
                })  # fmt: skip
        stats.assembly_seconds += time.perf_counter() - assembly_started
        simulation_started = time.perf_counter()
        for i, game in enumerate(active):
            a = int(chosen[i])
            game.apply(a)
        stats.simulation_seconds += time.perf_counter() - simulation_started

    stats.seconds = time.perf_counter() - t0
    stats.drain_seconds = time.perf_counter() - last_start
    assembly_started = time.perf_counter()
    rollout = builder.finish(expected=len(manifest))
    stats.assembly_seconds += time.perf_counter() - assembly_started
    return rollout, stats


__all__ = ["LiveGame", "Stats", "collect", "sample_action"]
