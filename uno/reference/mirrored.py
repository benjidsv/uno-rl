"""Exact moments of the mirrored evaluator by walking every world with the reference
solver: the cross-check for `uno.compiled.solvers.exact_mirrored`. See docs/foundations/spec.md section 15.
"""

from __future__ import annotations

from uno.agents.baseline import Agent
from uno.eval.estimate import ExactMirrored, Levels
from uno.game.chance import engine_children, worlds
from uno.game.observe import Level
from uno.game.rules import Ruleset
from uno.progress import bar
from uno.reference.solve import outcome_distribution


def _score_dist(rs: Ruleset, agents, levels, root, seat: int) -> dict[float, float]:
    """Distribution of `agents[seat]`'s score on one world."""
    d = outcome_distribution(rs, agents, levels, roots=[(root, 1.0)], expand=engine_children)
    sign = 1.0 if seat == 0 else -1.0
    return {(sign * r + 1.0) / 2.0: p for r, p in d.items()}


def exact_mirrored(rs: Ruleset, agents: tuple[Agent, Agent], levels: Levels = (Level.L1, Level.L1)) -> ExactMirrored:
    swapped, lv_swapped = (agents[1], agents[0]), (levels[1], levels[0])
    m = [0.0, 0.0]
    q = [0.0, 0.0]
    cross = split = 0.0
    ws = worlds(rs)
    for w, p in bar(ws, desc="exact mirrored", unit="worlds"):
        d0 = _score_dist(rs, agents, levels, w, 0)
        d1 = _score_dist(rs, swapped, lv_swapped, w, 1)
        e = [sum(s * pr for s, pr in d.items()) for d in (d0, d1)]
        for i, d in enumerate((d0, d1)):
            m[i] += p * e[i]
            q[i] += p * sum(s * s * pr for s, pr in d.items())
        cross += p * e[0] * e[1]
        split += p * sum(p0 * p1 for s0, p0 in d0.items() for s1, p1 in d1.items() if s0 + s1 == 1.0)
    return ExactMirrored(
        mean=(m[0], m[1]),
        var=(q[0] - m[0] ** 2, q[1] - m[1] ** 2),
        cov=cross - m[0] * m[1],
        splits=split,
        worlds=len(ws),
    )
