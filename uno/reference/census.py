"""Census of the lazy-chance tree (`uno.game.chance`): walk every history and count.

See docs/foundations/results/03-08-foundations/03-census.md and docs/foundations/spec.md section 19 (the position DP that replaced it for sizing).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from uno.game.chance import chance_outcomes, initial_deals
from uno.game.engine import State, legal_actions
from uno.game.observe import info_state
from uno.game.rules import Ruleset
from uno.progress import bar

# -- census -------------------------------------------------------------------


@dataclass
class Census:
    ruleset: Ruleset
    deals: int
    histories: int  # every node of the lazy tree, terminals included
    decision_nodes: int
    terminals: int
    info_sets: tuple[int, int]  # L1 information sets at decision nodes, per seat
    max_depth: int
    end_mass: dict[tuple[str, int | None], float]  # (end, winner) -> reach prob under uniform-legal play
    complete: bool
    seconds: float

    @property
    def uniform_legal_p0_score(self) -> float:
        """P(win) for seat 0 with draws worth 1/2, both seats playing UniformLegal."""
        return sum(p * (1.0 if w == 0 else 0.5 if w is None else 0.0) for (_, w), p in self.end_mass.items())

    @property
    def exhaustion_share(self) -> float:
        return sum(p for (e, _), p in self.end_mass.items() if e == "exhaustion")

    def row(self) -> dict[str, object]:
        return {
            "deals": self.deals,
            "histories": self.histories,
            "decision": self.decision_nodes,
            "terminals": self.terminals,
            "infosets0": self.info_sets[0],
            "infosets1": self.info_sets[1],
            "depth": self.max_depth,
            "exhaust%": round(100 * self.exhaustion_share, 1),
            "p0_score": round(self.uniform_legal_p0_score, 3),
            "complete": self.complete,
            "sec": round(self.seconds, 1),
        }


def census(rs: Ruleset, node_cap: int | None = None, symmetric: bool = False) -> Census:
    """Walk the whole lazy tree (depth first) and count.

    Reach probabilities use uniform-legal play at decision nodes, so `end_mass`
    is the exact terminal distribution of UniformLegal vs UniformLegal.
    `symmetric` walks the quotient tree (one subtree per deal orbit) and counts
    canonical L1 information sets; `deals` is then the number of orbits.
    """
    t0 = time.perf_counter()
    deals = initial_deals(rs, symmetric)
    sym = None
    if symmetric:
        from uno.game.observe import Level
        from uno.game.symmetry import Symmetry

        sym = Symmetry(rs)
    stack: list[tuple[State, int, float]] = [(s, 0, p) for s, p in reversed(deals)]
    histories = decision = terminals = depth_max = 0
    infosets: tuple[set[int], set[int]] = (set(), set())
    end_mass: dict[tuple[str, int | None], float] = {}
    complete = True
    progress = bar(total=node_cap, desc="census", unit="histories")
    while stack:
        s, d, reach = stack.pop()
        histories += 1
        if histories % 100_000 == 0:
            progress.update(100_000)
        depth_max = max(depth_max, d)
        if s.terminal:
            terminals += 1
            key = (s.end, s.winner)
            end_mass[key] = end_mass.get(key, 0.0) + reach
            continue
        decision += 1
        key = info_state(s, s.player)
        infosets[s.player].add(hash(key if sym is None else sym.canonical(Level.L1, key)[0]))
        if node_cap is not None and histories >= node_cap:
            complete = False
            break
        legal = legal_actions(s)
        w = reach / len(legal)
        for a in legal:
            for child, p in chance_outcomes(s, a):
                stack.append((child, d + 1, w * p))
    progress.update(histories - progress.n)
    progress.close()
    return Census(
        ruleset=rs,
        deals=len(deals),
        histories=histories,
        decision_nodes=decision,
        terminals=terminals,
        info_sets=(len(infosets[0]), len(infosets[1])),
        max_depth=depth_max,
        end_mass=end_mass,
        complete=complete,
        seconds=time.perf_counter() - t0,
    )
