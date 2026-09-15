"""`uno.game.chance.draw_outcomes` (step 9.5b): the chance model AIVAT integrates over, with
seeded reshuffles and ticket draws."""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import replace

from tests.strategies import SMALL
from uno.experiments.pairing import DUPS
from uno.game.chance import chance_outcomes, draw_outcomes
from uno.game.engine import legal_actions, new_game, step
from uno.game.rules import Draw, Ruleset

# 10 cards, a 3-card pile, +2 and +4: reshuffles in most games
RESH = Ruleset(colors=2, rank_copies=(1, 1, 1), skip_per_color=0, reverse_per_color=0, draw_two_per_color=1, wild_count=1, wild_draw_four_count=1, hand_size=3, reshuffle="seeded", max_turns=60)


def games(rs, n, seed0=0):
    """(state, action taken, child) along random-policy games."""
    for seed in range(seed0, seed0 + n):
        try:
            s = new_game(rs, seed)
        except ValueError:
            continue
        rng = random.Random(seed + 10_000)
        while not s.terminal:
            a = rng.choice(legal_actions(s))
            c = step(s, a)
            yield s, a, c
            s = c


def sig(s):
    """A state up to the order of its pile."""
    return (s.hands, s.discard, s.current_color, s.player, s.phase, s.drawn, s.public, s.own_draws, s.n_reshuffles, s.turn, s.end, s.winner, tuple(sorted(s.draw_pile)))


def conserved(rs, s) -> bool:
    return Counter(s.hands[0]) + Counter(s.hands[1]) + Counter(s.draw_pile) + Counter(s.discard) == Counter(rs.deck)


def test_equals_chance_outcomes_without_reshuffle():
    for rs in (SMALL, DUPS):
        n = 0
        for s, a, _ in games(rs, 60):
            got = draw_outcomes(s, a, enum_cap=10**6)
            want = chance_outcomes(s, a)
            assert [(c, p) for c, p in got] == want
            n += 1
        assert n > 200


def test_reshuffle_model_contains_the_real_child_and_sums_to_one():
    enumerated = reshuffled = skipped = 0
    for draws in ("pile", "tickets"):
        rs = replace(RESH, draws=draws)
        for s, a, child in games(rs, 300):
            outs = draw_outcomes(s, a, rng=random.Random(1))
            if outs is None:
                skipped += 1
                continue
            assert math.isclose(sum(p for _, p in outs), 1.0, abs_tol=1e-12)
            assert all(conserved(rs, c) for c, _ in outs)
            if len(outs) > 1 or s.n_reshuffles != child.n_reshuffles:
                assert sig(child) in {sig(c) for c, _ in outs}
                enumerated += 1
            if not s.draw_pile and isinstance(rs.actions[a], Draw) and len(s.discard) > 1:
                assert {c.own_draws[s.player][-1] for c, _ in outs} == set(s.discard[:-1])
                reshuffled += 1
    assert enumerated > 100 and reshuffled > 20 and skipped > 0


def test_chance_corrections_have_mean_zero_with_reshuffles():
    """For a fixed V of the hands, sum per game of V(real child) - E V(child): mean zero."""

    def v(s):
        return len(s.hands[0]) - 0.7 * len(s.hands[1]) + 0.3 * sum(RESH.card_types[c].is_wild for c in s.hands[0])

    for draws in ("pile", "tickets"):
        rs = replace(RESH, draws=draws)
        per_game: dict[int, float] = {}
        reshuffle_terms = []
        seed = 0
        for seed in range(2000):
            try:
                s = new_game(rs, seed)
            except ValueError:
                continue
            rng = random.Random(seed + 10_000)
            total = 0.0
            while not s.terminal:
                a = rng.choice(legal_actions(s))
                c = step(s, a)
                outs = draw_outcomes(s, a, rng=random.Random(f"{seed}:{s.turn}"))
                if outs is not None:
                    term = v(c) - sum(p * v(x) for x, p in outs)
                    total += term
                    if c.n_reshuffles != s.n_reshuffles:
                        reshuffle_terms.append(term)
                s = c
            per_game[seed] = total
        xs = list(per_game.values())
        m = sum(xs) / len(xs)
        se = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1) / len(xs))
        assert abs(m / se) <= 3.0, (draws, m, se)
        rm = sum(reshuffle_terms) / len(reshuffle_terms)
        rse = math.sqrt(sum((x - rm) ** 2 for x in reshuffle_terms) / (len(reshuffle_terms) - 1) / len(reshuffle_terms))
        assert len(reshuffle_terms) > 100 and abs(rm / rse) <= 3.0, (draws, rm, rse)
