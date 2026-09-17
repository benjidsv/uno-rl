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

**Symmetry and orbits.** A quotient L1 set is already in coordinates chosen by
canonicalising its *whole* history. Those coordinates cannot be reused for a
coarse observation: two observations with the same K0/K1/K2 fields may have been
put in different frames by history that the coarse policy is not allowed to see.
`coarsen` therefore canonicalises the coarse key again under its own group action
and maps every legal action through the same element. Its action orbits come from
the stabilizer of that canonical coarse key, rather than from the stabilizers of
the constituent full-L1 keys. Consequently the resulting strategy is both a
valid symmetric strategy and a function of the declared coarse input alone.

`Coarsening.keying` has no byte-encoded `.keys` (K0/K1/K2 have variable tuple
shapes unlike the L1/L2/L3 keys of `uno.game.keys`). `Coarsening.keys` and
`CoarseAgent` provide the corresponding typed lookup and deployed policy path;
`lift_to_l1` produces an ordinary L1 strategy for exact replay.
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


def _map_k0(sym: Symmetry, g: int, key: CoarseKey) -> CoarseKey:
    """`g . key` for K0. The field positions are documented at the top of the
    module; only cards and colors move under the group."""
    seat, level, hand, top, color, opp_n, pile_n, phase, drawn = key
    tm, cm = sym._types[g], sym._colors[g]
    return (
        seat,
        level,
        sym.hand(g, hand),
        tm[top],
        cm[color],
        opp_n,
        pile_n,
        phase,
        None if drawn is None else tm[drawn],
    )


def _map_k1(sym: Symmetry, g: int, key: CoarseKey) -> CoarseKey:
    return _map_k0(sym, g, key[:9]) + (tuple(sorted(sym._types[g][card] for card in key[9])),)


def _map_k2(sym: Symmetry, g: int, key: CoarseKey) -> CoarseKey:
    return _map_k1(sym, g, key[:10]) + (tuple(sym._colors[g][color] for color in key[10]),)


def _map_identity(sym: Symmetry, g: int, key: CoarseKey) -> CoarseKey:
    return sym.key(Level.L1, g, key)


_SYMMETRY_ACTION: dict[Callable[..., CoarseKey], Callable[[Symmetry, int, CoarseKey], CoarseKey]] = {
    k0: _map_k0,
    k1: _map_k1,
    k2: _map_k2,
    identity_key: _map_identity,
}


def _coarse_canonical(
    sym: Symmetry, keyfn: Callable[..., CoarseKey], key: CoarseKey
) -> tuple[CoarseKey, int]:
    """The minimum image of a coarse key and the first element reaching it.

    A symmetric coarsening must declare its group action explicitly. Silently
    canonicalising an arbitrary key through its source L1 history is exactly the
    information leak this module is meant to prevent.
    """
    try:
        action = _SYMMETRY_ACTION[keyfn]
    except KeyError as exc:
        raise ValueError("a custom key function on a symmetric tree needs a registered coarse symmetry action") from exc
    best, bg = key, 0
    for g in range(1, sym.order):
        image = action(sym, g, key)
        if image < best:
            best, bg = image, g
    return best, bg


def _coarse_stabilizer(
    sym: Symmetry, keyfn: Callable[..., CoarseKey], canonical_key: CoarseKey
) -> tuple[int, ...]:
    action = _SYMMETRY_ACTION[keyfn]
    return tuple(g for g in range(sym.order) if action(sym, g, canonical_key) == canonical_key)


@dataclass
class Coarsening:
    keying: Keying
    keyfn: Callable[..., CoarseKey]
    keys: tuple[CoarseKey, ...]  # coarse set id -> raw key (full tree) or canonical key (quotient)
    key_to_info: dict[CoarseKey, int]
    symmetry: Symmetry | None
    representative: np.ndarray  # (n_coarse,) an L1 set id representing each coarse set
    l1_to_coarse: np.ndarray  # (k1.n_info,) coarse set id of every L1 set
    l1_to_symmetry: np.ndarray  # (k1.n_info,) element mapping each L1 frame to its coarse frame
    l1_ia_to_coarse_ia: np.ndarray  # (k1.n_ia,) exact action-coordinate map

    def canonical(self, key: CoarseKey) -> tuple[CoarseKey, int]:
        """Canonical coarse key and the element mapping `key` to it."""
        if self.symmetry is None:
            return key, 0
        return _coarse_canonical(self.symmetry, self.keyfn, key)


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
    via `CompiledTree.keying_of`). On a quotient tree, both the coarse key and its
    actions are put in the coordinate frame selected from the coarse fields alone.
    The chosen element is retained for projection, lifting, and arc construction;
    no discarded L1 history can select the deployed policy's coordinates.

    Custom key functions remain supported on a full tree. A quotient coarsening
    rejects one without an explicit entry in `_SYMMETRY_ACTION`, because guessing
    its group action would risk recreating the hidden-history leak."""
    k = tree.keyings[level]
    sym = Symmetry(tree.rs) if tree.symmetric else None
    groups: dict[CoarseKey, list[int]] = {}
    l1_to_symmetry = np.zeros(k.n_info, dtype=np.int16)
    canonical_cache: dict[CoarseKey, tuple[CoarseKey, int]] = {}
    for i in range(k.n_info):
        obs = observation_from_key(tree.rs, level, k.key(i))
        raw_key = keyfn(tree.rs, obs, sym)
        if sym is None:
            key, g = raw_key, 0
        else:
            cached = canonical_cache.get(raw_key)
            if cached is None:
                cached = canonical_cache[raw_key] = _coarse_canonical(sym, keyfn, raw_key)
            key, g = cached
        l1_to_symmetry[i] = g
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
    stabilizer_cache: dict[CoarseKey, tuple[int, ...]] = {}

    def mapped_actions(m: int) -> tuple[int, ...]:
        actions = k.actions(m)
        if sym is None:
            return tuple(sorted(actions))
        return tuple(sorted(int(sym.actions[l1_to_symmetry[m], a]) for a in actions))

    for c, key in enumerate(keys_sorted):
        members = groups[key]
        representative[c] = members[0]
        for m in members:
            l1_to_coarse[m] = c
        acts_sorted = mapped_actions(members[0])
        for m in members:
            if mapped_actions(m) != acts_sorted:
                raise ValueError(f"coarse key merges sets with different legal actions: {key!r}")
            if k.info_level[m] != k.info_level[members[0]]:
                raise ValueError(f"coarse key merges sets at different tree levels: {key!r}")
        info_player[c] = k.info_player[members[0]]
        info_level[c] = k.info_level[members[0]]
        n_actions[c] = len(acts_sorted)
        ia_action_chunks.append(np.array(acts_sorted, dtype=np.int32))
        if sym is None:
            orbit_ids = np.arange(len(acts_sorted), dtype=np.int32)
        else:
            stabilizer = stabilizer_cache.get(key)
            if stabilizer is None:
                stabilizer = stabilizer_cache[key] = _coarse_stabilizer(sym, keyfn, key)
            orbit_ids = np.array(sym.action_orbits(stabilizer, acts_sorted), dtype=np.int32)
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

    l1_info = k.ia_info
    coarse_of_l1_ia = l1_to_coarse[l1_info]
    mapped_l1_action = k.ia_action.copy()
    if sym is not None:
        mapped_l1_action = sym.actions[l1_to_symmetry[l1_info], mapped_l1_action].astype(np.int32)
    l1_ia_to_coarse_ia = info_start[coarse_of_l1_ia] + pos_table[coarse_of_l1_ia, mapped_l1_action]
    if np.any(l1_ia_to_coarse_ia < 0):
        raise RuntimeError("an L1 action has no image in its coarse set")

    node_coarse = np.where(k.node_info >= 0, l1_to_coarse[np.maximum(k.node_info, 0)], -1).astype(np.int32)
    arc_ia = np.where(k.arc_ia >= 0, l1_ia_to_coarse_ia[np.maximum(k.arc_ia, 0)], -1).astype(np.int32)

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
    return Coarsening(
        keying=keying,
        keyfn=keyfn,
        keys=tuple(keys_sorted),
        key_to_info={key: i for i, key in enumerate(keys_sorted)},
        symmetry=sym,
        representative=representative,
        l1_to_coarse=l1_to_coarse,
        l1_to_symmetry=l1_to_symmetry,
        l1_ia_to_coarse_ia=l1_ia_to_coarse_ia.astype(np.int32),
    )


def project_onto_coarse(tree: CompiledTree, coarsening: Coarsening, l1_sigma: np.ndarray) -> np.ndarray:
    """An L1-level strategy (e.g. `store.load_or_tabulate(tree, Level.L1, agent)` for
    a K0-measurable agent such as Random or the Heuristic) restricted to a coarse
    keying via one representative L1 set per coarse set: valid only when the L1
    strategy is *actually* constant across every coarse group's members, which is
    not checked here (K0-measurable agents guarantee it).

    On a symmetric tree the representative's actions are first mapped from its L1
    canonical frame into the independently canonical coarse frame. `orbit_mean`
    then enforces the coarse key's own stabilizer, as required for a deployable
    equivariant policy (docs/spec.md section 18)."""
    k1 = tree.keyings[Level.L1]
    kc = coarsening.keying
    sigma = np.zeros(kc.n_ia)
    for c in range(kc.n_info):
        m = int(coarsening.representative[c])
        s1, e1 = k1.info_start[m], k1.info_start[m + 1]
        mapped = coarsening.l1_ia_to_coarse_ia[s1:e1]
        sigma[mapped] = l1_sigma[s1:e1]
    return kc.orbit_mean(sigma)


def lift_to_l1(tree: CompiledTree, coarsening: Coarsening, sigma: np.ndarray) -> np.ndarray:
    """Lift a coarse strategy into the tree's ordinary L1 action layout.

    This is an exact gather through the action-coordinate map built by `coarsen`.
    On a quotient tree the result is a symmetric L1 strategy, suitable for
    `CompiledAgent`, `evaluate`, and cross-checking the coarse keying's value.
    """
    if len(sigma) != coarsening.keying.n_ia:
        raise ValueError("sigma must be a strategy over the coarse keying")
    out = np.asarray(sigma)[coarsening.l1_ia_to_coarse_ia]
    k1 = tree.keyings[Level.L1]
    if tree.symmetric and not k1.is_symmetric(out):
        raise ValueError("the coarse strategy is not uniform within its coarse stabilizer")
    return out


class CoarseAgent:
    """Play a strategy as a function of its declared raw coarse observation.

    Unlike lifting through a full-L1 table, this is the direct deployment path:
    canonicalise K0/K1/K2 itself, query one coarse set, and map the distribution
    back to the observation's real labels. Equal raw keys therefore take exactly
    equal distributions, including when their full histories choose different L1
    canonical frames.
    """

    def __init__(self, tree: CompiledTree, coarsening: Coarsening, sigma: np.ndarray, name: str = "coarse") -> None:
        if len(sigma) != coarsening.keying.n_ia:
            raise ValueError("sigma must be a strategy over the coarse keying")
        if coarsening.symmetry is not None and not coarsening.keying.is_symmetric(sigma):
            raise ValueError("a quotient coarse policy must be uniform within every coarse action orbit")
        self.rs = tree.rs
        self.level = coarsening.keying.level
        self.coarsening = coarsening
        self.sigma = sigma
        self.name = name
        self.misses = 0
        self._canonical_cache: dict[CoarseKey, tuple[CoarseKey, int]] = {}

    def policy(self, obs: Observation, legal: tuple[int, ...]) -> dict[int, float]:
        if obs.level is not self.level:
            raise ValueError(f"{self.name} needs {self.level.name} observations, got {obs.level.name}")
        c = self.coarsening
        raw_key = c.keyfn(self.rs, obs, c.symmetry)
        mapped = self._canonical_cache.get(raw_key)
        if mapped is None:
            mapped = self._canonical_cache[raw_key] = c.canonical(raw_key)
        key, g = mapped
        i = c.key_to_info.get(key)
        if i is None:
            self.misses += 1
            return {legal[0]: 1.0}
        s, e = c.keying.info_start[i], c.keying.info_start[i + 1]
        out = {int(a): float(p) for a, p in zip(c.keying.ia_action[s:e], self.sigma[s:e]) if p > 0}
        if g:
            assert c.symmetry is not None
            out = c.symmetry.map_policy(int(c.symmetry.inverse[g]), out)
        assert all(a in legal for a in out), "coarse keying and game disagree on legal actions"
        return out


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
