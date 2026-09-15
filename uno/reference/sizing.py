"""Exact tree sizes without walking the tree: a dynamic program over positions.

`census` (uno/reference/census.py) visits every history, ~140k per second in pure Python,
so a candidate toy of 20-50M histories costs minutes to *count* and cannot be
told from one of 500M. But the subtree below a history depends only on its
**position**: the hands, the pile (unknown multiset plus known tail), the top
card and current color, whose turn it is, the phase and the pending drawn
card. The public history, the initial hands and the draw records only feed
the information-set keys, not the transitions. So every count that does not
mention information sets (histories, decision nodes, terminals, depth, the
end distribution and game length under a fixed history-free policy) is a
function of the position, and the whole tree is sized by memoising over
positions, of which there are far fewer than histories.

Two policies are tracked: UniformLegal (what `census` reports, the
cross-check) and PlayFirstRandom (the ladder's Random row, whose random-vs-
random seat-0 score is the seat-parity readout of the census rounds in
docs/results/03-census.md). Their end masses are exact, not sampled.

What this cannot count: L1 / L2 information sets (they depend on the
history). Those come out of `compile_tree` once a candidate is chosen.

The symmetric quotient tree (docs/spec.md section 18) is one subtree per orbit of
deals, so its size is the sum over the orbit representatives; positions are
shared across roots, so the quotient costs no more than one root's worth of
new positions per orbit.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass

from uno.game.chance import children, engine_children, initial_deals, pile_split, worlds
from uno.game.engine import State
from uno.game.rules import Play, Ruleset

# end-mass columns: (end, winner)
ENDS: tuple[tuple[str, int | None], ...] = (
    ("win", 0),
    ("win", 1),
    ("exhaustion", 0),
    ("exhaustion", 1),
    ("exhaustion", None),
    ("cap", None),
)
_END_INDEX = {e: i for i, e in enumerate(ENDS)}
N_POLICIES = 2  # uniform legal, play-first random


@dataclass
class Size:
    ruleset: Ruleset
    roots: int  # deals (or worlds), or orbits of them when symmetric
    positions: int  # distinct positions visited (the DP's cost)
    histories: int
    decision_nodes: int
    terminals: int
    max_depth: int
    end_mass: tuple[dict[tuple[str, int | None], float], ...]  # per policy
    decisions_per_game: tuple[float, ...]  # per policy, expected number of decisions
    complete: bool  # False when the position cap was hit (every count is then a lower bound)
    seconds: float

    @staticmethod
    def _score(mass: dict, seat: int) -> float:
        return sum(p * (1.0 if w == seat else 0.5 if w is None else 0.0) for (_, w), p in mass.items())

    @property
    def uniform_legal_p0_score(self) -> float:
        return self._score(self.end_mass[0], 0)

    @property
    def random_p0_score(self) -> float:
        """Seat-0 score under PlayFirstRandom vs PlayFirstRandom, draws worth 1/2."""
        return self._score(self.end_mass[1], 0)

    def exhaustion_share(self, policy: int = 1) -> float:
        return sum(p for (e, _), p in self.end_mass[policy].items() if e == "exhaustion")

    def tie_share(self, policy: int = 1) -> float:
        return sum(p for (_, w), p in self.end_mass[policy].items() if w is None)


def _position(s: State, lazy: bool) -> tuple:
    if lazy:
        unknown, tail = pile_split(s)
        pile = (tuple(sorted(unknown.elements())), tail)
    else:
        pile = s.draw_pile
    turn = s.turn if s.ruleset.max_turns is not None else 0
    return (s.hands, pile, s.top, s.current_color, s.player, s.phase, s.drawn, turn)


class _Cap(Exception):
    pass


def size(rs: Ruleset, symmetric: bool = True, world: bool = False, position_cap: int | None = None) -> Size:
    """Size the lazy tree (or the world tree of L3, `world=True`) exactly.
    `symmetric`: the quotient tree, one root per deal (world) orbit.
    `position_cap`: stop once that many positions are memoised (memory is
    ~0.5 KB per position); the result is then incomplete."""
    t0 = time.perf_counter()
    lazy = not world
    roots = worlds(rs, symmetric) if world else initial_deals(rs, symmetric)
    expand = engine_children if world else children
    is_play = tuple(isinstance(a, Play) for a in rs.actions)
    n_ends = len(ENDS)
    zero_mass = [0.0] * n_ends
    # memo: position -> (histories, decisions, terminals, depth, mass_ul, mass_rnd, len_ul, len_rnd)
    memo: dict[tuple, tuple] = {}

    def rec(s: State) -> tuple:
        if s.terminal:
            m = list(zero_mass)
            m[_END_INDEX[(s.end, s.winner)]] = 1.0
            return (1, 0, 1, 0, m, m, 0.0, 0.0)
        key = _position(s, lazy)
        hit = memo.get(key)
        if hit is not None:
            return hit
        if position_cap is not None and len(memo) >= position_cap:
            raise _Cap
        ch = expand(s)
        legal = tuple(ch)
        plays = [a for a in legal if is_play[a]]
        rnd_support = plays if plays else legal
        w_ul, w_rnd = 1.0 / len(legal), 1.0 / len(rnd_support)
        h, d, t, depth = 1, 1, 0, 0
        m_ul, m_rnd = list(zero_mass), list(zero_mass)
        l_ul = l_rnd = 1.0
        for a in legal:
            in_rnd = not plays or is_play[a]
            for c, p in ch[a]:
                ch_, cd, ct, cdepth, cm_ul, cm_rnd, cl_ul, cl_rnd = rec(c)
                h += ch_
                d += cd
                t += ct
                if cdepth + 1 > depth:
                    depth = cdepth + 1
                q = w_ul * p
                for i in range(n_ends):
                    m_ul[i] += q * cm_ul[i]
                l_ul += q * cl_ul
                if in_rnd:
                    q = w_rnd * p
                    for i in range(n_ends):
                        m_rnd[i] += q * cm_rnd[i]
                    l_rnd += q * cl_rnd
        out = (h, d, t, depth, m_ul, m_rnd, l_ul, l_rnd)
        memo[key] = out
        return out

    H = D = T = depth = 0
    mass = [list(zero_mass) for _ in range(N_POLICIES)]
    length = [0.0] * N_POLICIES
    complete = True
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old_limit, 10_000))
    try:
        for s, p in roots:
            try:
                h, d, t, dep, m_ul, m_rnd, l_ul, l_rnd = rec(s)
            except _Cap:
                complete = False
                break
            H += h
            D += d
            T += t
            depth = max(depth, dep)
            for i in range(n_ends):
                mass[0][i] += p * m_ul[i]
                mass[1][i] += p * m_rnd[i]
            length[0] += p * l_ul
            length[1] += p * l_rnd
    finally:
        sys.setrecursionlimit(old_limit)
    if not complete:
        # what was summed so far is a lower bound; the masses are meaningless
        mass = [list(zero_mass) for _ in range(N_POLICIES)]
    return Size(
        ruleset=rs,
        roots=len(roots),
        positions=len(memo),
        histories=H,
        decision_nodes=D,
        terminals=T,
        max_depth=depth,
        end_mass=tuple({e: m[i] for i, e in enumerate(ENDS) if m[i] > 0} for m in mass),
        decisions_per_game=tuple(length),
        complete=complete,
        seconds=time.perf_counter() - t0,
    )
