"""Color and rank symmetry of a ruleset. See docs/spec.md section 18.

The rules never mention a particular color or a particular rank: renaming the
colors (any permutation) and renaming the ranks (any permutation among ranks
with the same number of copies) maps deals to equally likely deals, legal
actions to legal actions and terminal returns to themselves. So the game is
invariant under the group G = S_colors x (rank permutations within copy
classes); wilds are fixed. On `TOY` (4 colors, 2 ranks of one copy) |G| = 48.

What that buys: one subtree per *orbit* of deals instead of one per deal
(37x fewer histories on `TOY`), and one information set per orbit of keys
(46x fewer L1 sets). Values computed on that quotient equal the full-game
values whenever every policy involved is **symmetric**, i.e. commutes with
the group: `policy(g . obs) = g . policy(obs)`. Random, UniformLegal and the
Heuristic (section 13, symmetric tie-breaks) are; CFR and best responses on
the quotient produce symmetric strategies by construction; an RL agent must
be made symmetric by canonicalising its observation and mapping the action
back (`canonical` + `map_policy`), which is also a free |G|-fold gain in data
efficiency. Symmetrising any policy never raises its exploitability.

Canonical form. `canonical(level, key)` returns the lexicographic minimum of
the key over the group and an element `g` that reaches it: `g . key ==
canonical`. The search is pruned on the observer's initial hand(s), the first
component of every key after the seat: only elements achieving the minimal
hand image are tried on the rest of the key. Hands are few, so the candidate
lists are cached per hand.
"""

from __future__ import annotations

import itertools
from dataclasses import replace
from typing import Hashable

import numpy as np

from uno.game.engine import Event, State
from uno.game.observe import Level
from uno.game.rules import CardType, Kind, Play, Ruleset

Key = Hashable


class Symmetry:
    """The group of a ruleset as lookup tables: element `g` maps type `t` to
    `types[g, t]`, color `c` to `colors[g, c]`, action `a` to `actions[g, a]`.
    Element 0 is the identity and `inverse[g]` is the inverse of `g`."""

    def __init__(self, rs: Ruleset) -> None:
        self.rs = rs
        ct = rs.card_types
        index = {t: i for i, t in enumerate(ct)}
        classes: dict[int, list[int]] = {}
        for r, copies in enumerate(rs.rank_copies):
            if copies > 0:
                classes.setdefault(copies, []).append(r)
        rank_maps: list[dict[int, int]] = []
        for combo in itertools.product(*(itertools.permutations(v) for v in classes.values())):
            m: dict[int, int] = {}
            for cls, perm in zip(classes.values(), combo):
                m.update(zip(cls, perm))
            rank_maps.append(m)
        types, colors = [], []
        for pc in itertools.permutations(range(rs.colors)):
            for pr in rank_maps:
                tm = []
                for t in ct:
                    if t.is_wild:
                        tm.append(index[t])
                    elif t.kind is Kind.NUMBER:
                        tm.append(index[CardType(Kind.NUMBER, pc[t.color], pr[t.rank])])
                    else:
                        tm.append(index[CardType(t.kind, pc[t.color], None)])
                types.append(tm)
                colors.append(pc)
        self.types = np.array(types, dtype=np.int16)  # (G, num_types)
        self.colors = np.array(colors, dtype=np.int16)  # (G, colors)
        acts = np.empty((len(types), rs.num_actions), dtype=np.int16)
        for g in range(len(types)):
            for i, a in enumerate(rs.actions):
                if isinstance(a, Play):
                    color = None if a.color is None else int(self.colors[g, a.color])
                    acts[g, i] = rs.action_index[Play(int(self.types[g, a.card]), color)]
                else:
                    acts[g, i] = i
        self.actions = acts
        ident = np.arange(rs.num_types)
        self.inverse = np.array(
            [next(h for h in range(len(types)) if np.array_equal(self.types[h][self.types[g]], ident) and np.array_equal(self.colors[h][self.colors[g]], np.arange(rs.colors))) for g in range(len(types))],
            dtype=np.int16,
        )
        assert np.array_equal(self.types[0], ident) and np.array_equal(self.colors[0], np.arange(rs.colors))
        self._types = [tuple(int(x) for x in row) for row in self.types]  # tuples index faster than numpy scalars
        self._colors = [tuple(int(x) for x in row) for row in self.colors]
        self._hand_cache: dict[tuple[int, ...], tuple[tuple[int, ...], tuple[int, ...]]] = {}
        self._hands_cache: dict[tuple, tuple[tuple, tuple[int, ...]]] = {}

    @property
    def order(self) -> int:
        return len(self.types)

    # -- the action of an element ----------------------------------------------

    def hand(self, g: int, hand: tuple[int, ...]) -> tuple[int, ...]:
        tm = self._types[g]
        return tuple(sorted(tm[x] for x in hand))

    def cards(self, g: int, cards: tuple[int, ...]) -> tuple[int, ...]:
        tm = self._types[g]
        return tuple(tm[x] for x in cards)

    def public(self, g: int, public: tuple[Event, ...]) -> tuple[Event, ...]:
        tm, pc = self._types[g], self._colors[g]
        out = []
        for ev in public:
            match ev[0]:
                case "setup":
                    out.append(("setup", tm[ev[1]], tuple(tm[x] for x in ev[2])))
                case "play":
                    out.append(("play", ev[1], tm[ev[2]], pc[ev[3]]))
                case _:
                    out.append(ev)
        return tuple(out)

    def key(self, level: Level, g: int, key: Key) -> Key:
        """`g . key` for a key of `uno.game.keys.KEYS[level]`."""
        if level is Level.L3:
            hands, pile, player, phase, drawn, public, draws = key
            tm = self._types[g]
            return (
                (self.hand(g, hands[0]), self.hand(g, hands[1])),
                self.cards(g, pile),
                player,
                phase,
                None if drawn is None else tm[drawn],
                self.public(g, public),
                (self.cards(g, draws[0]), self.cards(g, draws[1])),
            )
        seat, hand, public, draws = key[:4]
        out = (seat, self.hand(g, hand), self.public(g, public), self.cards(g, draws))
        if level is Level.L2:
            out += (self.hand(g, key[4]), self.cards(g, key[5]))
        return out

    def state(self, g: int, s: State) -> State:
        tm, pc = self._types[g], self._colors[g]
        return replace(
            s,
            hands=(self.hand(g, s.hands[0]), self.hand(g, s.hands[1])),
            draw_pile=self.cards(g, s.draw_pile),
            discard=self.cards(g, s.discard),
            current_color=pc[s.current_color],
            drawn=None if s.drawn is None else tm[s.drawn],
            public=self.public(g, s.public),
            own_draws=(self.cards(g, s.own_draws[0]), self.cards(g, s.own_draws[1])),
            initial_hands=(self.hand(g, s.initial_hands[0]), self.hand(g, s.initial_hands[1])),
        )

    def map_policy(self, g: int, probs: dict[int, float]) -> dict[int, float]:
        am = self.actions[g]
        return {int(am[a]): p for a, p in probs.items()}

    # -- canonical forms ----------------------------------------------------------

    def _candidates(self, hand: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """(minimal image of the hand, elements reaching it)."""
        c = self._hand_cache.get(hand)
        if c is None:
            best, gs = None, []
            for g in range(self.order):
                h = self.hand(g, hand)
                if best is None or h < best:
                    best, gs = h, [g]
                elif h == best:
                    gs.append(g)
            c = self._hand_cache[hand] = (best, tuple(gs))
        return c

    def _candidates2(self, hands: tuple) -> tuple[tuple, tuple[int, ...]]:
        c = self._hands_cache.get(hands)
        if c is None:
            best, gs = None, []
            for g in range(self.order):
                h = (self.hand(g, hands[0]), self.hand(g, hands[1]))
                if best is None or h < best:
                    best, gs = h, [g]
                elif h == best:
                    gs.append(g)
            c = self._hands_cache[hands] = (best, tuple(gs))
        return c

    def canonical(self, level: Level, key: Key) -> tuple[Key, int]:
        """(min over the group of `g . key`, one `g` reaching it)."""
        if level is Level.L3:
            _, gs = self._candidates2(key[0])
        else:
            _, gs = self._candidates(key[1])
        if len(gs) == 1:
            return self.key(level, gs[0], key), gs[0]
        best, bg = None, 0
        for g in gs:
            k = self.key(level, g, key)
            if best is None or k < best:
                best, bg = k, g
        return best, bg

    def stabilizer(self, level: Level, canonical_key: Key) -> tuple[int, ...]:
        """Elements fixing a canonical key (identity included)."""
        gs = self._candidates2(canonical_key[0])[1] if level is Level.L3 else self._candidates(canonical_key[1])[1]
        return tuple(g for g in gs if self.key(level, g, canonical_key) == canonical_key)

    def action_orbits(self, stabilizer: tuple[int, ...], actions: tuple[int, ...]) -> tuple[int, ...]:
        """Orbit index (in order of first appearance) of each action under the stabilizer.
        A symmetric strategy is uniform within every orbit (docs/spec.md section 18)."""
        if len(stabilizer) == 1:
            return tuple(range(len(actions)))
        rep: dict[int, int] = {}
        return tuple(rep.setdefault(min(int(self.actions[g, a]) for g in stabilizer), len(rep)) for a in actions)

    def canonical_state(self, s: State) -> tuple[State, int]:
        """The canonical representative of a state's orbit (by its L3 key) and the element reaching it."""
        from uno.game.keys import l3_key

        _, g = self.canonical(Level.L3, l3_key(s, s.player))
        return self.state(g, s), g


def canonical_roots(rs: Ruleset, roots: list[tuple[State, float]], lazy: bool, sym: Symmetry | None = None) -> list[tuple[State, float]]:
    """One representative per orbit of `roots`, carrying the orbit's total
    probability; the representative is the canonical state itself. `lazy`: the
    roots are deals of the lazy tree, whose unknown pile is a multiset (its
    order never matters), so it is kept sorted and ignored by the orbit; worlds
    (`lazy=False`) keep their pile order."""
    from uno.game.chance import pile_split
    from uno.game.keys import l3_key

    sym = sym or Symmetry(rs)
    reps: dict[Key, tuple[State, float]] = {}
    for s, p in roots:
        if lazy:
            n_tail = len(pile_split(s)[1])
            best, rep = None, s
            for g in sym._candidates2(s.hands)[1]:
                t = sym.state(g, s)
                cut = len(t.draw_pile) - n_tail
                t = replace(t, draw_pile=(*sorted(t.draw_pile[:cut]), *t.draw_pile[cut:]))
                k = l3_key(t, t.player)
                if best is None or k < best:
                    best, rep = k, t
            kc = best
        else:
            kc, g = sym.canonical(Level.L3, l3_key(s, s.player))
            rep = sym.state(g, s)
        rec = reps.get(kc)
        reps[kc] = (rep, p) if rec is None else (rec[0], rec[1] + p)
    return list(reps.values())
