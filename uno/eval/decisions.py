"""Skill-carrying decision kinds, action classes and situations, as functions of
an observation and legal actions rather than a `State`. See docs/plan.md ("Which
decisions carry skill") and docs/spec.md section 21.

Working from an observation (not a `State`) means the same classifiers apply to
played games (`observe`) and to the compiled tree's information sets, whose only
record of a decision is its key (`uno.game.observe.observation_from_key`, step 9.5a).
`uno.compiled.edge` groups the edge-attribution decomposition by `node_kind` and
`action_class`; `uno.experiments.transfer` uses `classify` (moved here unchanged)
for its decision-kind census.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from uno.game.engine import Phase
from uno.game.observe import Level, Observation
from uno.game.rules import Kind, Pass, Play, Ruleset

if TYPE_CHECKING:
    from uno.compiled.tree import CompiledTree

CATEGORIES = (
    "any choice",
    "which color: plays in 2+ colors",
    "wild: hold or spend",
    "wild: pick a color",
    "action card among other plays",
    "draw with a play in hand",
    "play or keep the drawn card",
)


def classify(rs: Ruleset, obs: Observation, legal: tuple[int, ...]) -> set[str]:
    """`uno.experiments.transfer`'s decision-kind census (step 8.5), generalised from a
    `State` to an observation + legal actions so it also works on information-set keys."""
    types = rs.card_types
    plays = {rs.actions[a].card for a in legal if isinstance(rs.actions[a], Play)}
    cats = {"any choice"} if len(legal) > 1 else set()
    if obs.phase is Phase.AFTER_DRAW:
        return cats | {"play or keep the drawn card"}
    colored = {c for c in plays if not types[c].is_wild}
    wild = len(plays) > len(colored)
    if len({types[c].color for c in colored}) > 1:
        cats.add("which color: plays in 2+ colors")
    if wild:
        cats.add("wild: pick a color")
        if colored:
            cats.add("wild: hold or spend")
    if len(plays) > 1 and any(types[c].kind not in (Kind.NUMBER, Kind.WILD) for c in plays):
        cats.add("action card among other plays")
    if plays and rs.draw_action in legal:
        cats.add("draw with a play in hand")
    return cats


NODE_KINDS = ("single", "drawn", "wild", "action", "color", "other")


def node_kind(rs: Ruleset, obs: Observation, legal: tuple[int, ...]) -> str:
    """One label per decision node (a partition, unlike `classify`'s categories), by
    fixed priority: a single legal action; the drawn card (after a voluntary draw);
    a wild among the legal plays; an action card among 2+ plays; plays in 2+ colors;
    else "other" (a single-color choice, or a draw against one kind of play)."""
    if len(legal) == 1:
        return "single"
    if obs.phase is Phase.AFTER_DRAW:
        return "drawn"
    types = rs.card_types
    plays = [rs.actions[a].card for a in legal if isinstance(rs.actions[a], Play)]
    if any(types[c].is_wild for c in plays):
        return "wild"
    if len(plays) > 1 and any(types[c].kind is not Kind.NUMBER for c in plays):
        return "action"
    if len({types[c].color for c in plays}) > 1:
        return "color"
    return "other"


ACTION_CLASSES = (
    "draw with a play in hand",
    "forced draw",
    "pass",
    "play drawn card",
    "wild with a colored play available",
    "wild when only wilds play",
    "action card",
    "number",
)


def action_class(rs: Ruleset, obs: Observation, legal: tuple[int, ...], a: int) -> str:
    """One label per (node, action): which of the D2 report's action classes `a` falls
    under at this decision."""
    act = rs.actions[a]
    types = rs.card_types
    if isinstance(act, Pass):
        return "pass"
    if a == rs.draw_action:
        plays = [ac for ac in legal if isinstance(rs.actions[ac], Play)]
        return "draw with a play in hand" if plays else "forced draw"
    if obs.phase is Phase.AFTER_DRAW:
        return "play drawn card"
    if types[act.card].is_wild:
        colored = [ac for ac in legal if isinstance(rs.actions[ac], Play) and not types[rs.actions[ac].card].is_wild]
        return "wild with a colored play available" if colored else "wild when only wilds play"
    if types[act.card].kind is not Kind.NUMBER:
        return "action card"
    return "number"


def situation(obs: Observation) -> tuple[int, str, str]:
    """(seat, opponent hand size bucket, pile size bucket), the grouping for the D2
    report's "by situation" breakdown."""
    opp = "1" if obs.opp_hand_size == 1 else "2" if obs.opp_hand_size == 2 else "3+"
    pile = "0" if obs.pile_size == 0 else "1" if obs.pile_size == 1 else "2+"
    return (obs.seat, opp, pile)


def set_labels(tree: CompiledTree, level: Level | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per information set of `tree.keyings[level]` (default `Level.L1`): `node_kind`
    (str array, `n_info`) and `situation` packed as `"seat/opp/pile"` (str array,
    `n_info`); per set-action (`ia`), `action_class` (str array, `n_ia`). One
    `observation_from_key` call per set: cache with `uno.store.load_or_label` on a
    tree this size (`uno.compiled.strategies.tabulate`'s docstring quotes ~50s per
    agent per tree on `TOY2_FULL`'s 3.75M sets; this walks the same number)."""
    from uno.game.observe import observation_from_key

    level = Level.L1 if level is None else level
    k = tree.keyings[level]
    kinds = np.empty(k.n_info, dtype=object)
    sits = np.empty(k.n_info, dtype=object)
    actions = np.empty(k.n_ia, dtype=object)
    for i in range(k.n_info):
        key = k.key(i)
        obs = observation_from_key(tree.rs, level, key)
        legal = k.actions(i)
        kinds[i] = node_kind(tree.rs, obs, legal)
        sits[i] = "/".join(str(x) for x in situation(obs))
        s = k.info_start[i]
        for j, a in enumerate(legal):
            actions[s + j] = action_class(tree.rs, obs, legal, a)
    return kinds.astype(str), sits.astype(str), actions.astype(str)
