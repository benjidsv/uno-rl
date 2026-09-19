"""Episode manifests, deal partitions and the DEV evaluation panel.

docs/rl/pipeline-plan.md section 3: the complete episode manifest for an
iteration is created **before** any work is dispatched, with equal completed-game
quotas on the eight opponent x learner-seat cells. A fixed game count, not a race
to the first N decisions: selecting the fastest completed games would favor short
trajectories, and a predetermined manifest makes worker count an execution
setting rather than a data-allocation setting.

Section 4's proposed DEV partition, which this module owns:

| Deal seeds | Use |
|---|---|
| `[0, 800000)` | training deals |
| `[800000, 900000)` | fixed scheduled selection panels |
| `[900000, 1000000)` | pilot sizing and diagnostic/holdout panels |

All three are **DEV** partitions inside `uno.eval.seeds.DEV_POOL`; none of them is
confirmatory data. A frozen claim still claims a fresh block of `EVAL_POOL`.
Training deals cycle through a seeded permutation when the finite pool is
exhausted; the cycle count is logged and the schedule never drifts into EVAL.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uno.eval.seeds import DEV_POOL
from uno.rl.streams import SCHEDULE, generator

TRAIN_DEALS = range(0, 800_000)
PANEL_DEALS = range(800_000, 900_000)
DIAGNOSTIC_DEALS = range(900_000, 1_000_000)

assert TRAIN_DEALS.start >= DEV_POOL.start and DIAGNOSTIC_DEALS.stop <= DEV_POOL.stop


@dataclass(frozen=True, slots=True)
class Episode:
    """One assigned game. Training metadata, never actor input (section 4)."""

    episode_id: int  # logical, globally increasing within a training lineage
    deal_seed: int
    opponent: str
    seat: int  # the learner's seat
    cycle: int  # how many times the training deal pool had been exhausted


@dataclass(frozen=True)
class Manifest:
    """The complete episode assignment of one iteration."""

    update: int
    episodes: tuple[Episode, ...]
    cells: tuple[tuple[str, int], ...]
    games_per_cell: int

    def __len__(self) -> int:
        return len(self.episodes)

    @property
    def deal_seeds(self) -> tuple[int, ...]:
        return tuple(e.deal_seed for e in self.episodes)


class DealSchedule:
    """The training deal order: a seeded permutation of `TRAIN_DEALS`, cycled.

    Position and cycle count are checkpoint state, so a resumed run continues the
    schedule rather than replaying deals it has already trained on.
    """

    def __init__(self, training_seed: int, position: int = 0, cycle: int = 0, pool: range = TRAIN_DEALS) -> None:
        self.training_seed, self.pool = training_seed, pool
        self.position, self.cycle = position, cycle
        self._order: np.ndarray | None = None

    def _permutation(self, cycle: int) -> np.ndarray:
        return generator(self.training_seed, SCHEDULE, "deals", cycle).permutation(len(self.pool)) + self.pool.start

    def take(self, n: int) -> list[tuple[int, int]]:
        """The next `n` (deal seed, cycle) pairs, advancing the schedule."""
        out: list[tuple[int, int]] = []
        while len(out) < n:
            if self._order is None:
                self._order = self._permutation(self.cycle)
            remaining = len(self._order) - self.position
            take = min(n - len(out), remaining)
            out += [(int(s), self.cycle) for s in self._order[self.position : self.position + take]]
            self.position += take
            if self.position >= len(self._order):
                self.cycle += 1
                self.position = 0
                self._order = None
        return out

    @property
    def state(self) -> dict:
        return {"position": self.position, "cycle": self.cycle, "pool_start": self.pool.start, "pool_stop": self.pool.stop}

    @classmethod
    def from_state(cls, training_seed: int, state: dict) -> DealSchedule:
        return cls(training_seed, state["position"], state["cycle"], range(state["pool_start"], state["pool_stop"]))


def cells(opponents, seats: tuple[int, ...] = (0, 1)) -> tuple[tuple[str, int], ...]:
    """The opponent x learner-seat cells, in a fixed order.

    `seats` is `(0, 1)` for the league objective, which balances both seats. A toy
    diagnostic that reports the two seats separately passes one of them (pipeline
    plan section 3: "For fixed-opponent toys, balance the two seats" -- balanced
    across the pair of runs, not inside one gradient check)."""
    return tuple((name, seat) for name in opponents for seat in seats)


def build(update: int, opponents, schedule: DealSchedule, games_per_cell: int, next_episode_id: int, seats: tuple[int, ...] = (0, 1)) -> tuple[Manifest, int]:
    """The next iteration's complete manifest, and the next free episode id.

    Every cell gets the same number of games and its own deals. A deal seed is
    used by exactly one cell within an iteration, so a cell's score is not
    coupled to another cell's by a shared deal during *training*; paired
    evaluation, where coupling is wanted, is `uno.rl.evaluate`'s job.
    """
    cs = cells(opponents, seats)
    drawn = schedule.take(len(cs) * games_per_cell)
    episodes: list[Episode] = []
    eid = next_episode_id
    for i, (name, seat) in enumerate(cs):
        for j in range(games_per_cell):
            deal, cycle = drawn[i * games_per_cell + j]
            episodes.append(Episode(eid, deal, name, seat, cycle))
            eid += 1
    return Manifest(update, tuple(episodes), cs, games_per_cell), eid


def panel(opponents, deals_per_cell: int, start: int = PANEL_DEALS.start, seats: tuple[int, ...] = (0, 1)) -> Manifest:
    """The fixed, disjoint DEV evaluation panel: the same deals in every cell.

    Contract for section 9: balanced over opponents and seats, and coupled -- all
    cells share the deal block, so a paired difference can group every seat and
    opponent that shares a deal. It is fixed across checkpoints and variants, and
    it is not resized because a checkpoint looks good.
    """
    if start + deals_per_cell > PANEL_DEALS.stop:
        raise ValueError("the panel would run past the DEV panel partition")
    cs = cells(opponents, seats)
    episodes = []
    for i, (name, seat) in enumerate(cs):
        for j in range(deals_per_cell):
            episodes.append(Episode(i * deals_per_cell + j, start + j, name, seat, 0))
    return Manifest(-1, tuple(episodes), cs, deals_per_cell)


def diagnostic_panel(
    opponents,
    deals_per_cell: int,
    start: int = DIAGNOSTIC_DEALS.start,
    seats: tuple[int, ...] = (0, 1),
) -> Manifest:
    """A coupled DEV panel reserved for sizing and other bounded diagnostics.

    This deliberately does not consume the fixed scheduled-selection panel.  It
    has the same opponent/seat/deal coupling, so its paired variance estimates
    transfer to choosing that panel's size without selecting on its outcomes.
    """
    if start < DIAGNOSTIC_DEALS.start or start + deals_per_cell > DIAGNOSTIC_DEALS.stop:
        raise ValueError("the diagnostic panel would run outside its DEV partition")
    cs = cells(opponents, seats)
    episodes = []
    offset = start - DIAGNOSTIC_DEALS.start
    for i, (name, seat) in enumerate(cs):
        for j in range(deals_per_cell):
            # Independent of the requested width, so extending a sizing panel
            # preserves every existing policy and opponent random stream.
            episode_id = i * len(DIAGNOSTIC_DEALS) + offset + j
            episodes.append(Episode(episode_id, start + j, name, seat, 0))
    return Manifest(-1, tuple(episodes), cs, deals_per_cell)


__all__ = [
    "DIAGNOSTIC_DEALS", "PANEL_DEALS", "TRAIN_DEALS", "DealSchedule", "Episode",
    "Manifest", "build", "cells", "diagnostic_panel", "panel",
]  # fmt: skip
