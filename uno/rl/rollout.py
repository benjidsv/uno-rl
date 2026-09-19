"""The rollout: compact typed arrays for one iteration's episodes and decisions.

docs/rl/pipeline-plan.md section 4 fixes what is stored. Per episode: logical ID,
deal seed, action-stream IDs, training seed, opponent version, learner seat,
behavior-policy version, terminal return and reason, total engine decisions,
learner decisions, and row offsets. Per learner decision: the exact declared
actor features, the separate pre-action critic features, the engine legal mask,
the chosen action, the old log probability, and its episode/sequence reference.

Only the **selected action's** log probability is kept: that is all ordinary PPO
needs. The complete distribution stays available from the public agent API and
from a bounded diagnostic sample, never as a per-decision training array.

No full game traces are retained. `Rollout.nbytes` and `estimate_bytes` are there
so a batch is sized before it is allocated, and a small bounded failure sample
(`uno.rl.collect`) lives outside this buffer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uno.rl.features import Features, Spec

ROLLOUT_SCHEMA = 1


@dataclass(frozen=True)
class Rollout:
    """One iteration's collected data, in logical episode order."""

    actor_spec: Spec
    critic_spec: Spec
    behavior_version: int
    # per decision
    actor: Features
    critic: Features
    masks: np.ndarray  # (T, num_actions) bool
    actions: np.ndarray  # (T,) int32
    old_logp: np.ndarray  # (T,) float32
    episode: np.ndarray  # (T,) int32, index into the per-episode arrays
    decision_index: np.ndarray  # (T,) int32, sequence position within its episode
    # per episode
    episode_id: np.ndarray  # (G,) int64
    deal_seed: np.ndarray  # (G,) int64
    opponent: tuple[str, ...]  # (G,)
    seat: np.ndarray  # (G,) int8
    cycle: np.ndarray  # (G,) int32
    terminal_return: np.ndarray  # (G,) float32, in the learner's frame
    end_reason: tuple[str, ...]  # (G,)
    engine_decisions: np.ndarray  # (G,) int32, both players
    learner_decisions: np.ndarray  # (G,) int32
    offset: np.ndarray  # (G,) int64, first row of the episode
    # (T,) int64 information-set ids, or all -1. Only a tabular actor
    # (`uno.rl.tabular`) needs them; a network reads its features instead.
    set_id: np.ndarray | None = None
    schema: int = ROLLOUT_SCHEMA

    @property
    def n_decisions(self) -> int:
        return len(self.actions)

    @property
    def n_games(self) -> int:
        return len(self.episode_id)

    @property
    def returns_per_decision(self) -> np.ndarray:
        """`R_g` broadcast to its decisions: the undiscounted terminal return."""
        return self.terminal_return[self.episode]

    @property
    def nbytes(self) -> int:
        arrays = (self.actor.tok, self.actor.col, self.actor.glob, self.critic.tok, self.critic.col, self.critic.glob,
                  self.masks, self.actions, self.old_logp, self.episode, self.decision_index)  # fmt: skip
        return int(sum(a.nbytes for a in arrays))

    def validate(self) -> None:
        """Structural checks the plan requires before an update touches the batch."""
        t = self.n_decisions
        for name, a in (("masks", self.masks), ("actions", self.actions), ("old_logp", self.old_logp), ("episode", self.episode)):
            if len(a) != t:
                raise ValueError(f"{name} has {len(a)} rows, expected {t}")
        if len(self.actor) != t or len(self.critic) != t:
            raise ValueError("feature rows do not match the decision count")
        if self.n_games and int(self.learner_decisions.sum()) != t:
            raise ValueError("per-episode learner decision counts do not sum to the rows")
        ids = self.episode_id
        if len(set(ids.tolist())) != len(ids):
            raise ValueError("duplicate episode ids in one rollout")
        if not np.all(np.diff(ids) > 0):
            raise ValueError("episodes are not in logical id order")
        if not np.all(self.masks[np.arange(t), self.actions]):
            raise ValueError("a stored action is illegal under its own stored mask")
        if not np.isfinite(self.old_logp).all():
            raise ValueError("a stored behavior log probability is not finite")
        if np.any(self.old_logp > 1e-6):
            raise ValueError("a stored behavior log probability is positive")
        self.actor.check_finite()
        self.critic.check_finite()


def estimate_bytes(actor_spec: Spec, critic_spec: Spec, decisions: int, games: int) -> int:
    """Bytes a rollout of this size will take, before allocating it (section 4)."""
    lay = actor_spec.layout
    per_row = 4 * (lay.n_tokens * (actor_spec.tok_width + critic_spec.tok_width) + lay.colors * (actor_spec.col_width + critic_spec.col_width) + actor_spec.glob_width + critic_spec.glob_width)
    per_row += lay.rs.num_actions + 4 + 4 + 4 + 4  # mask, action, logp, episode, index
    return decisions * per_row + games * 64


class Builder:
    """Accumulates decisions per episode, then assembles them in logical ID order.

    Completion order is deliberately not the storage order: the manifest fixes
    which games exist and `finish` sorts by `episode_id`, so a rollout cannot
    inherit a selection effect from which worker finished first (section 3).
    """

    def __init__(self, actor_spec: Spec, critic_spec: Spec, behavior_version: int) -> None:
        self.actor_spec, self.critic_spec, self.behavior_version = actor_spec, critic_spec, behavior_version
        self._rows: dict[int, list[tuple]] = {}
        self._episodes: dict[int, dict] = {}
        self._payloads: list[dict] = []

    def add_decision(self, episode_id: int, decision_index: int, actor: Features, critic: Features, mask: np.ndarray, action: int, old_logp: float, set_id: int = -1) -> None:
        self._rows.setdefault(episode_id, []).append((decision_index, actor, critic, mask, action, old_logp, set_id))

    def finish_episode(self, episode, terminal_return: float, end_reason: str, engine_decisions: int) -> None:
        if episode.episode_id in self._episodes:
            raise ValueError(f"episode {episode.episode_id} was completed twice")
        self._episodes[episode.episode_id] = {
            "episode": episode, "terminal_return": terminal_return, "end_reason": end_reason, "engine_decisions": engine_decisions,
        }  # fmt: skip

    def add_payload(self, payload: dict, episodes: dict[int, object]) -> None:
        """Ingest one complete parallel-worker payload without per-row views.

        The payload remains one compact group of arrays until final assembly.
        This avoids retaining a Python tuple plus feature/mask views for every
        decision received from a worker; completion order is still discarded.
        """
        if self._rows:
            raise ValueError("cannot mix row-wise and bulk rollout ingestion")
        for i, eid in enumerate(payload["episode_id"].tolist()):
            eid = int(eid)
            if eid not in episodes:
                raise ValueError(f"worker returned unassigned episode {eid}")
            self.finish_episode(
                episodes[eid],
                float(payload["terminal_return"][i]),
                payload["end_reason"][i],
                int(payload["engine_decisions"][i]),
            )
        self._payloads.append(payload)

    def finish(self, expected: int | None = None) -> Rollout:
        if self._payloads:
            return self._finish_payloads(expected)
        ids = sorted(self._episodes)
        if expected is not None and len(ids) != expected:
            raise ValueError(f"{len(ids)} episodes completed, {expected} were assigned")
        rows: list[tuple] = []
        offsets, counts = [], []
        for eid in ids:
            got = sorted(self._rows.get(eid, []), key=lambda r: r[0])
            if [r[0] for r in got] != list(range(len(got))):
                raise ValueError(f"episode {eid} has a gap or a duplicate in its decision sequence")
            offsets.append(len(rows))
            counts.append(len(got))
            rows += got
        t = len(rows)
        n_actions = self.actor_spec.rs.num_actions
        if t:
            actor = Features(
                np.concatenate([r[1].tok for r in rows]), np.concatenate([r[1].col for r in rows]), np.concatenate([r[1].glob for r in rows])
            )
            critic = Features(
                np.concatenate([r[2].tok for r in rows]), np.concatenate([r[2].col for r in rows]), np.concatenate([r[2].glob for r in rows])
            )
            masks = np.stack([r[3] for r in rows])
        else:
            lay = self.actor_spec.layout
            empty = lambda sp: Features(  # noqa: E731
                np.zeros((0, lay.n_tokens, sp.tok_width), np.float32), np.zeros((0, lay.colors, sp.col_width), np.float32), np.zeros((0, sp.glob_width), np.float32)
            )
            actor, critic, masks = empty(self.actor_spec), empty(self.critic_spec), np.zeros((0, n_actions), bool)
        meta = [self._episodes[eid] for eid in ids]
        out = Rollout(
            actor_spec=self.actor_spec, critic_spec=self.critic_spec, behavior_version=self.behavior_version,
            actor=actor, critic=critic, masks=masks,
            actions=np.asarray([r[4] for r in rows], dtype=np.int32),
            old_logp=np.asarray([r[5] for r in rows], dtype=np.float32),
            episode=np.repeat(np.arange(len(ids), dtype=np.int32), counts) if ids else np.zeros(0, np.int32),
            decision_index=np.asarray([r[0] for r in rows], dtype=np.int32),
            set_id=np.asarray([r[6] for r in rows], dtype=np.int64),
            episode_id=np.asarray(ids, dtype=np.int64),
            deal_seed=np.asarray([m["episode"].deal_seed for m in meta], dtype=np.int64),
            opponent=tuple(m["episode"].opponent for m in meta),
            seat=np.asarray([m["episode"].seat for m in meta], dtype=np.int8),
            cycle=np.asarray([m["episode"].cycle for m in meta], dtype=np.int32),
            terminal_return=np.asarray([m["terminal_return"] for m in meta], dtype=np.float32),
            end_reason=tuple(m["end_reason"] for m in meta),
            engine_decisions=np.asarray([m["engine_decisions"] for m in meta], dtype=np.int32),
            learner_decisions=np.asarray(counts, dtype=np.int32),
            offset=np.asarray(offsets, dtype=np.int64),
        )
        out.validate()
        return out

    def _finish_payloads(self, expected: int | None) -> Rollout:
        """Assemble bulk worker arrays once, in logical episode order."""
        ids = sorted(self._episodes)
        if expected is not None and len(ids) != expected:
            raise ValueError(f"{len(ids)} episodes completed, {expected} were assigned")
        locations: dict[int, tuple[dict, int]] = {}
        for payload in self._payloads:
            for i, eid in enumerate(payload["episode_id"].tolist()):
                eid = int(eid)
                if eid in locations:
                    raise ValueError(f"episode {eid} was returned in two payloads")
                locations[eid] = payload, i
        if set(locations) != set(ids):
            raise ValueError("bulk payload episodes do not match completed episodes")

        counts = [int(locations[eid][0]["learner_decisions"][locations[eid][1]]) for eid in ids]
        pieces: dict[str, list[np.ndarray]] = {
            name: [] for name in (
                "tok", "col", "glob", "ctok", "ccol", "cglob", "masks",
                "actions", "old_logp", "decision_index", "set_id",
            )
        }
        for eid in ids:
            payload, i = locations[eid]
            start, n = int(payload["offset"][i]), int(payload["learner_decisions"][i])
            sl = slice(start, start + n)
            for name in pieces:
                pieces[name].append(payload[name][sl])

        def joined(name: str, empty: np.ndarray) -> np.ndarray:
            return np.concatenate(pieces[name], axis=0) if pieces[name] else empty

        lay = self.actor_spec.layout
        actor = Features(
            joined("tok", np.zeros((0, lay.n_tokens, self.actor_spec.tok_width), np.float32)),
            joined("col", np.zeros((0, lay.colors, self.actor_spec.col_width), np.float32)),
            joined("glob", np.zeros((0, self.actor_spec.glob_width), np.float32)),
        )
        critic = Features(
            joined("ctok", np.zeros((0, lay.n_tokens, self.critic_spec.tok_width), np.float32)),
            joined("ccol", np.zeros((0, lay.colors, self.critic_spec.col_width), np.float32)),
            joined("cglob", np.zeros((0, self.critic_spec.glob_width), np.float32)),
        )
        meta = [self._episodes[eid] for eid in ids]
        out = Rollout(
            actor_spec=self.actor_spec, critic_spec=self.critic_spec, behavior_version=self.behavior_version,
            actor=actor, critic=critic,
            masks=joined("masks", np.zeros((0, self.actor_spec.rs.num_actions), bool)),
            actions=joined("actions", np.zeros(0, np.int32)),
            old_logp=joined("old_logp", np.zeros(0, np.float32)),
            episode=np.repeat(np.arange(len(ids), dtype=np.int32), counts) if ids else np.zeros(0, np.int32),
            decision_index=joined("decision_index", np.zeros(0, np.int32)),
            set_id=joined("set_id", np.zeros(0, np.int64)),
            episode_id=np.asarray(ids, dtype=np.int64),
            deal_seed=np.asarray([m["episode"].deal_seed for m in meta], dtype=np.int64),
            opponent=tuple(m["episode"].opponent for m in meta),
            seat=np.asarray([m["episode"].seat for m in meta], dtype=np.int8),
            cycle=np.asarray([m["episode"].cycle for m in meta], dtype=np.int32),
            terminal_return=np.asarray([m["terminal_return"] for m in meta], dtype=np.float32),
            end_reason=tuple(m["end_reason"] for m in meta),
            engine_decisions=np.asarray([m["engine_decisions"] for m in meta], dtype=np.int32),
            learner_decisions=np.asarray(counts, dtype=np.int32),
            offset=np.cumsum(np.asarray([0, *counts[:-1]], dtype=np.int64)),
        )
        out.validate()
        return out


__all__ = ["ROLLOUT_SCHEMA", "Builder", "Rollout", "estimate_bytes"]
