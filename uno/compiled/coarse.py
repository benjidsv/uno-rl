"""Coarse (history-free) information-set keys and an iterated memoryless best
response: how much of BR L1's edge needs history at all. Step 9.5a, D3 of the step
9 review's section D ("Memory: BR L1 conditions on the full history. A backward
pass on a coarse, history-free key gives a valid policy, a lower bound on the
memoryless ceiling: does the RL observation need history?"). See docs/spec.md
section 20.

**Three keys**, each a deterministic function of the L1 observation rebuilt from a
canonical L1 set's key (`uno.game.observe.observation_from_key`), nested:

- K0: (seat, tree level, hand, top, current color, opponent hand size, pile size,
  phase, drawn card). The tree level (`2 * len(public) + [phase == TURN_START]`,
  docs/spec.md section 16, `uno.compiled.tree.node_level`) is included so a coarse
  set never spans two tree levels, which the per-level backward pass of
  `best_response` needs (a set whose arcs span two levels would have its choice
  finalised, wrongly, at the first level it is seen); it also acts as a weak clock,
  a little memory of how long the game has run.
- K1: K0 + the discard pile as a multiset (order dropped, contents kept).
- K2: K1 + a void hint -- the colors that were current at each of the opponent's
  draws since their last play (the list clears whenever they play), a coarse proxy
  for "what they might be avoiding". Counts every draw event once, not every card
  of a multi-card penalty draw.

Because K0's fields (hand, top, color, phase, drawn) are exactly what determines
legality (given the ruleset's static flags: `opp_hand_size` and `pile_size` do not
affect which of *this* player's actions are legal), two L1 sets sharing a K0 value
are *guaranteed* to share the same legal actions; K1 and K2 only add fields, so the
guarantee carries over. `coarsen` asserts it anyway, as `compile_tree` does for its
own keyings, and raises if it ever fails.

**Orbits.** On the quotient tree, several L1 sets can merge into one coarse set,
and each carries its own action-orbit partition (docs/spec.md section 18) --
possibly a *different* one, since a set's orbit structure depends on the
stabilizer of its whole canonical key (history included), not only on K0/K1/K2. A
coarse policy is one distribution per coarse set, so for it to be valid for every
merged L1 set at once, it must be uniform within the **transitive join** of their
orbit partitions (union two actions whenever *any* merged set's own orbits group
them together): the join is never finer than any single constituent's partition
(so it never asks for *less* uniformity than a constituent independently
requires), which is what keeps the resulting strategy achievable and never an
over-estimate (docs/spec.md section 18, "a best response above the maximum is not
a strategy"). It can force a little *more* uniformity than the coarse-optimal
strategy strictly needs -- `coarsen` picks the safe join rather than search for
the tightest valid partition, which only matters for `memoryless_response`'s
value as a lower bound, never for its validity.

`Coarsening.keying` has no `.keys` (no byte-encoding exists for K0/K1/K2 shapes,
unlike the L1/L2/L3 keys of `uno.game.keys`): `best_response`, `tree.weights` /
`tree.reach` / `tree.values` and `evaluate` never touch a keying's `.keys`, so this
is enough for everything here, but `tabulate` / `CompiledAgent` do not work on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from uno.compiled.solvers import best_response, evaluate
from uno.compiled.tree import CompiledTree, Keying, Strategy
from uno.game.engine import Phase
from uno.game.observe import Level, Observation, discard_from_public, observation_from_key
from uno.game.rules import Ruleset
from uno.game.symmetry import Symmetry

CoarseKey = tuple


def _node_level(obs: Observation) -> int:
    """`uno.compiled.tree.node_level`, from an observation instead of a `State`."""
    return 2 * len(obs.public) + (1 if obs.phase is Phase.TURN_START else 0)


def k0(rs: Ruleset, obs: Observation, sym: Symmetry | None = None) -> CoarseKey:
    return (obs.seat, _node_level(obs), obs.hand, obs.top, obs.current_color, obs.opp_hand_size, obs.pile_size, int(obs.phase), obs.drawn)


def k1(rs: Ruleset, obs: Observation, sym: Symmetry | None = None) -> CoarseKey:
    return k0(rs, obs) + (tuple(sorted(discard_from_public(obs.public))),)


def _void_hint(rs: Ruleset, obs: Observation) -> tuple[int, ...]:
    """Colors that were current at each of the opponent's draws since their last
    play, in the order drawn; cleared at every opponent play (each multi-card draw
    event counts once, not once per card)."""
    opp = obs.seat ^ 1
    colors: list[int] = []
    color = None
    for ev in obs.public:
        if ev[0] == "setup":
            color = rs.card_types[ev[1]].color
        elif ev[0] == "play":
            color = ev[3]
            if ev[1] == opp:
                colors = []
        elif ev[0] == "draw" and ev[1] == opp:
            colors.append(color)
    return tuple(colors)


def k2(rs: Ruleset, obs: Observation, sym: Symmetry | None = None) -> CoarseKey:
    return k1(rs, obs) + (_void_hint(rs, obs),)


def identity_key(rs: Ruleset, obs: Observation, sym: Symmetry | None = None) -> CoarseKey:
    """The full L1 information state: `coarsen(tree, identity_key)` is structurally
    the same partition as `tree.keyings[Level.L1]` (one coarse set per L1 set), used
    to test the machinery in this module against ordinary `best_response`."""
    return obs.info_state


KEYS: dict[str, Callable[..., CoarseKey]] = {"K0": k0, "K1": k1, "K2": k2}


@dataclass
class Coarsening:
    keying: Keying
    representative: np.ndarray  # (n_coarse,) an L1 set id representing each coarse set
    l1_to_coarse: np.ndarray  # (k1.n_info,) coarse set id of every L1 set


def _union_find_orbits(k: Keying, members: list[int], acts_sorted: tuple[int, ...]) -> np.ndarray:
    """The transitive join of every member's own action-orbit partition (see the
    module docstring's "Orbits"), as an (len(acts_sorted),) array of orbit ids."""
    n = len(acts_sorted)
    pos_of_action = {a: p for p, a in enumerate(acts_sorted)}
    parent = list(range(n))

    def find(x: int) -> int:
        r = x
        while parent[r] != r:
            r = parent[r]
        while parent[x] != r:
            parent[x], x = r, parent[x]
        return r

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for m in members:
        s, e = k.info_start[m], k.info_start[m + 1]
        local_orbit = k.ia_orbit[s:e] - k.info_orbit_start[m]
        by_orbit: dict[int, list[int]] = {}
        for j, a in enumerate(k.ia_action[s:e]):
            by_orbit.setdefault(int(local_orbit[j]), []).append(pos_of_action[int(a)])
        for positions in by_orbit.values():
            for p in positions[1:]:
                union(positions[0], p)
    roots = [find(x) for x in range(n)]
    root_to_orbit = {r: idx for idx, r in enumerate(sorted(set(roots)))}
    return np.array([root_to_orbit[r] for r in roots], dtype=np.int32)


def _none_safe(x):
    """`x` with every `None` (the "no drawn card" case) replaced by `-1`, recursively
    through nested tuples: a sort key for `CoarseKey` tuples, which otherwise raise
    on the mixed `None` / `int` comparisons Python's `sorted` can hit at the `drawn`
    position once some keys have a drawn card and others do not."""
    if x is None:
        return -1
    if isinstance(x, tuple):
        return tuple(_none_safe(v) for v in x)
    return x


def coarsen(tree: CompiledTree, keyfn: Callable[..., CoarseKey], level: Level = Level.L1) -> Coarsening:
    """Group `tree.keyings[level]`'s information sets by `keyfn` applied to each
    set's own (canonical) observation, into a new `Keying` over the tree's own
    nodes and arcs (usable directly with `best_response`, `tree.weights`, `evaluate`
    via `CompiledTree.keying_of`). `keyfn` also gets a `Symmetry` instance (`None`
    on a non-symmetric tree), currently unused by K0 / K1 / K2 / `identity_key`.

    **Known limitation on a symmetric (quotient) tree**: K0 / K1 / K2 read their
    fields off whichever canonical L1 representative `compile_tree` happened to
    pick for each L1 set (minimising the *whole* key, history included) -- not a
    representative re-minimised for K0/K1/K2's own, smaller field set. The same
    abstract coarse position can therefore land in different coarse groups
    depending on which L1 representative reached it, so the quotient tree's K0 /
    K1 / K2 partition can be a *finer* (less merged) one than the "ideal" quotient
    of the abstract K0/K1/K2 partition, and `coarsen(quotient_tree, k0)`'s value
    is not expected to numerically match `coarsen(full_tree, k0)`'s (unlike
    `identity_key`, where every group has one member and no such gap can arise).
    Properly re-canonicalising K0/K1/K2 under the group would need every
    constituent's action list remapped through its own canonicalising element at
    the arc level, not just the key; not done (docs/results/09.5a-results.md logs
    this as a correction to the step 9.5a plan, which expected the two to agree).
    The value computed here is still a valid, achievable lower bound on *that
    tree's* own BR L1 either way (`test_coarse_response_replays_through_evaluate_
    and_never_exceeds_br_l1`); only the full/quotient comparison is affected."""
    k = tree.keyings[level]
    sym = Symmetry(tree.rs) if tree.symmetric else None
    groups: dict[CoarseKey, list[int]] = {}
    for i in range(k.n_info):
        obs = observation_from_key(tree.rs, level, k.key(i))
        key = keyfn(tree.rs, obs, sym)
        groups.setdefault(key, []).append(i)
    # Sets must be numbered so that every tree level's sets are contiguous, as
    # `compile_tree`'s own keyings are by construction ("in order of first
    # appearance", spec section 16): `best_response`'s per-level loop slices a
    # level's responder arcs by `ia.min() : ia.max() + 1` and assumes that range
    # covers exactly the sets touched at that level, nothing from another level
    # interleaved. A coarse key does not include `_node_level` (it does for K0 /
    # K1 / K2, but not `identity_key`), and even when it does, a plain sort by key
    # value need not group same-level keys together (e.g. K0/K1/K2 sort by hand
    # before level). Sorting by `(tree level, key)` restores contiguity without
    # constraining `keyfn`'s own field order.
    keys_sorted = sorted(groups, key=lambda key: (int(k.info_level[groups[key][0]]), _none_safe(key)))
    n_coarse = len(keys_sorted)

    info_player = np.zeros(n_coarse, dtype=np.int8)
    info_level = np.zeros(n_coarse, dtype=np.int32)
    n_actions = np.zeros(n_coarse, dtype=np.int32)
    representative = np.zeros(n_coarse, dtype=np.int64)
    l1_to_coarse = np.full(k.n_info, -1, dtype=np.int32)
    ia_action_chunks: list[np.ndarray] = []
    ia_orbit_chunks: list[np.ndarray] = []

    for c, key in enumerate(keys_sorted):
        members = groups[key]
        representative[c] = members[0]
        for m in members:
            l1_to_coarse[m] = c
        acts_sorted = tuple(sorted(k.actions(members[0])))
        for m in members:
            if tuple(sorted(k.actions(m))) != acts_sorted:
                raise ValueError(f"coarse key merges sets with different legal actions: {key!r}")
            if k.info_level[m] != k.info_level[members[0]]:
                raise ValueError(f"coarse key merges sets at different tree levels: {key!r}")
        info_player[c] = k.info_player[members[0]]
        info_level[c] = k.info_level[members[0]]
        n_actions[c] = len(acts_sorted)
        ia_action_chunks.append(np.array(acts_sorted, dtype=np.int32))
        orbit_ids = np.arange(len(acts_sorted), dtype=np.int32) if not tree.symmetric else _union_find_orbits(k, members, acts_sorted)
        ia_orbit_chunks.append(orbit_ids)

    info_start = np.zeros(n_coarse + 1, dtype=np.int32)
    np.cumsum(n_actions, out=info_start[1:])
    ia_action = np.concatenate(ia_action_chunks) if ia_action_chunks else np.zeros(0, dtype=np.int32)
    ia_info = np.repeat(np.arange(n_coarse, dtype=np.int32), n_actions)

    n_orbits = np.array([int(o.max()) + 1 if len(o) else 0 for o in ia_orbit_chunks], dtype=np.int32)
    info_orbit_start = np.zeros(n_coarse + 1, dtype=np.int32)
    np.cumsum(n_orbits, out=info_orbit_start[1:])
    ia_orbit = np.concatenate([chunk + info_orbit_start[c] for c, chunk in enumerate(ia_orbit_chunks)]) if ia_orbit_chunks else np.zeros(0, dtype=np.int32)
    orbit_info = np.repeat(np.arange(n_coarse, dtype=np.int32), n_orbits)
    orbit_size = np.bincount(ia_orbit, minlength=int(info_orbit_start[-1])).astype(np.int32)

    # a lookup table (coarse set, real/canonical action id) -> position in ia_action's sorted, per-set chunk
    num_actions = len(tree.rs.actions)
    pos_table = np.full((max(n_coarse, 1), num_actions), -1, dtype=np.int32)
    pos_within_set = np.arange(len(ia_action), dtype=np.int32) - np.repeat(info_start[:-1], n_actions)
    if len(ia_action):
        pos_table[ia_info, ia_action] = pos_within_set

    node_coarse = np.where(k.node_info >= 0, l1_to_coarse[np.maximum(k.node_info, 0)], -1).astype(np.int32)
    arc_coarse = node_coarse[tree.arc_node]
    arc_l1_ia = k.arc_ia
    valid = arc_l1_ia >= 0
    arc_action_canonical = np.where(valid, k.ia_action[np.maximum(arc_l1_ia, 0)], -1)
    arc_ia = np.full(tree.m, -1, dtype=np.int32)
    ok = valid & (arc_coarse >= 0)
    arc_ia[ok] = info_start[arc_coarse[ok]] + pos_table[arc_coarse[ok], arc_action_canonical[ok]]

    keying = Keying(
        level=level,
        node_info=node_coarse,
        arc_ia=arc_ia,
        ia_info=ia_info,
        ia_action=ia_action,
        info_player=info_player,
        info_level=info_level,
        info_start=info_start,
        keys={},
        ia_orbit=ia_orbit,
        orbit_info=orbit_info,
        orbit_size=orbit_size,
        info_orbit_start=info_orbit_start,
    )
    return Coarsening(keying=keying, representative=representative, l1_to_coarse=l1_to_coarse)


def project_onto_coarse(tree: CompiledTree, coarsening: Coarsening, l1_sigma: np.ndarray) -> np.ndarray:
    """An L1-level strategy (e.g. `store.load_or_tabulate(tree, Level.L1, agent)` for
    a K0-measurable agent such as Random or the Heuristic) restricted to a coarse
    keying via one representative L1 set per coarse set: valid only when the L1
    strategy is *actually* constant across every coarse group's members, which is
    not checked here (K0-measurable agents guarantee it).

    On a symmetric tree the representative's own values need not already be uniform
    within the *coarse* keying's own action orbits (the transitive join across every
    merged L1 set, module docstring): they are only guaranteed uniform within the
    representative's own, possibly finer, orbits. `orbit_mean` enforces the coarser
    requirement, as `tabulate` / `cfr` do elsewhere whenever a strategy must be
    symmetric for the compiled tree's machinery to treat it as one well-defined
    real-game strategy (docs/spec.md section 18); skipping it reproduces the
    historical "pure choice on an orbit scores above the true value" bug there."""
    k1 = tree.keyings[Level.L1]
    kc = coarsening.keying
    sigma = np.zeros(kc.n_ia)
    for c in range(kc.n_info):
        m = int(coarsening.representative[c])
        s1, e1 = k1.info_start[m], k1.info_start[m + 1]
        acts_m, vals_m = k1.ia_action[s1:e1], l1_sigma[s1:e1]
        sc, ec = kc.info_start[c], kc.info_start[c + 1]
        acts_c = kc.ia_action[sc:ec]
        order = np.argsort(acts_m)
        pos = np.searchsorted(acts_m[order], acts_c)
        sigma[sc:ec] = vals_m[order][pos]
    return kc.orbit_mean(sigma)


def _profile(seat: int, mine: Strategy, opp: Strategy) -> tuple[Strategy, Strategy]:
    return (mine, opp) if seat == 0 else (opp, mine)


@dataclass
class MemorylessResult:
    coarsening: Coarsening
    sigma: np.ndarray  # the best coarse strategy seen, over `coarsening.keying`'s ia
    value: float  # its exact return at `seat` against the opponent
    log: list[float]  # value at every iterate, iterate 0 first


def memoryless_response(
    tree: CompiledTree,
    coarsening: Coarsening,
    opponent: Strategy,
    seat: int,
    start_l1_sigma: np.ndarray,
    iterations: int = 20,
) -> MemorylessResult:
    """Iterated greedy best response on a coarse (imperfect-recall) keying: from a
    starting policy projected onto the coarse keying (`project_onto_coarse`),
    repeatedly compute the best response to `opponent` with `own` fixed at the
    *current* iterate's weights (`best_response(..., own=tree.weights(keying,
    sigma, seat))`, docs/spec.md section 21), then evaluate the new policy exactly
    and keep going. Every iterate is a real (history-free) policy, so the best
    value seen is a valid lower bound on the best achievable memoryless policy
    against `opponent`; `best_response(tree, Level.L1, opponent, seat)` (full
    history) is the upper bound. The iteration need not be monotone (an
    imperfect-recall local best response is not guaranteed to improve every step),
    so the best (value, sigma) pair across the whole run is kept, not just the
    last. Stops early when the policy repeats an earlier iterate exactly."""
    kc = coarsening.keying
    sigma = project_onto_coarse(tree, coarsening, start_l1_sigma)
    seen: list[bytes] = []

    def value_of(s: np.ndarray) -> float:
        return evaluate(tree, *_profile(seat, (kc, s), opponent))[seat]

    v = value_of(sigma)
    log = [v]
    best_v, best_sigma = v, sigma
    seen.append(sigma.tobytes())
    for _ in range(iterations):
        own = tree.weights(kc, sigma, seat)
        br = best_response(tree, kc, opponent, seat, own=own)
        sigma = br.sigma
        v = value_of(sigma)
        log.append(v)
        if v > best_v:
            best_v, best_sigma = v, sigma
        b = sigma.tobytes()
        if b in seen:
            break
        seen.append(b)
    return MemorylessResult(coarsening=coarsening, sigma=best_sigma, value=best_v, log=log)
