"""Agents that play a table: a reference best-response table (`TabularAgent`),
a fixed pure strategy for brute-force checks (`PureStrategy`) and a strategy
over a compiled keying (`CompiledAgent`).
"""

from __future__ import annotations

import numpy as np

from uno.agents.baseline import Policy
from uno.compiled.tree import CompiledTree
from uno.game.keys import Key, encode_key, obs_key
from uno.game.observe import Level, Observation
from uno.game.rules import Ruleset
from uno.reference.solve import BestResponse


class TabularAgent:
    """Plays a best-response table. Must observe at `br.level`.

    Information sets missing from the table (unreachable under the opponent the
    table was computed against) fall back to the first legal action and are
    counted in `misses`.
    """

    def __init__(self, rs: Ruleset, br: BestResponse, name: str | None = None) -> None:
        self.rs = rs
        self.br = br
        self.level = br.level
        self.table = br.table
        self.name = name or f"br_{br.level.name.lower()}_vs_{br.opponent}"
        self.misses = 0

    def policy(self, obs: Observation, legal: tuple[int, ...]) -> Policy:
        if obs.level is not self.level:
            raise ValueError(f"{self.name} needs {self.level.name} observations, got {obs.level.name}")
        a = self.table.get(obs_key(self.rs, obs))
        if a is None or a not in legal:
            self.misses += 1
            a = legal[0]
        return {a: 1.0}


class PureStrategy:
    """A fixed action per information set, for brute-force checks. Same fallback as TabularAgent."""

    def __init__(self, rs: Ruleset, level: Level, table: dict[Key, int]) -> None:
        self.rs, self.level, self.table, self.name = rs, level, table, "pure"

    def policy(self, obs: Observation, legal: tuple[int, ...]) -> Policy:
        a = self.table.get(obs_key(self.rs, obs), legal[0])
        return {a: 1.0}


class CompiledAgent:
    """Plays a strategy over a keying through `uno.eval.play`. Must observe at `level`.
    Sets missing from the keying (never reached in the compiled tree, e.g. under
    a different ruleset) fall back to the first legal action and are counted."""

    def __init__(self, tree: CompiledTree, level: Level, sigma: np.ndarray, name: str = "compiled") -> None:
        self.rs = tree.rs
        self.keying = tree.keyings[level]
        self.symmetry = tree.symmetry
        self.level = level
        self.sigma = sigma
        self.name = name
        self.misses = 0

    def policy(self, obs: Observation, legal: tuple[int, ...]) -> Policy:
        if obs.level is not self.level:
            raise ValueError(f"{self.name} needs {self.level.name} observations, got {obs.level.name}")
        k = self.keying
        key = obs_key(self.rs, obs)
        g = 0
        if self.symmetry is not None:
            key, g = self.symmetry.canonical(self.level, key)
        i = k.keys.get(encode_key(self.level, key))
        if i is None:
            self.misses += 1
            return {legal[0]: 1.0}
        s, t = k.info_start[i], k.info_start[i + 1]
        out = {int(a): float(p) for a, p in zip(k.ia_action[s:t], self.sigma[s:t]) if p > 0}
        if g:
            out = self.symmetry.map_policy(int(self.symmetry.inverse[g]), out)  # back from canonical to the real labels
        assert all(a in legal for a in out), "keying and game disagree on legal actions"
        return out
