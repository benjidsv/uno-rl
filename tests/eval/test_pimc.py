"""PIMC rollout values (uno/eval/pimc.py, step 9.5c): determinization keeps everything but the pile
order and never reads it, its first draw is `draw_outcomes`' chance model, and rollout values,
rollout Q and the deal floor match the exact tree on a small ruleset."""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import replace
from functools import cache

import numpy as np

from tests.game.test_draw_outcomes import RESH, games, sig
from tests.strategies import SMALL
from uno.agents.baseline import Heuristic, PlayFirstRandom
from uno.compiled.aivat import arc_expectation, aivat_moments, exact_values
from uno.compiled.lookup import build
from uno.compiled.strategies import tabulate
from uno.compiled.tree import compile_tree
from uno.eval.aivat import NONE, Corrections
from uno.eval.pimc import deal_floor, determinize, rollout_q, rollout_rng, rollout_value, unknown_cards
from uno.experiments.pairing import DUPS
from uno.game import rules
from uno.game.chance import draw_count, draw_outcomes, pile_split
from uno.game.engine import new_game, step
from uno.game.observe import Level


def test_determinize_keeps_everything_but_the_order():
    n = 0
    for rs in (SMALL, DUPS, RESH, replace(RESH, draws="tickets")):
        for s, _, _ in games(rs, 40):
            d = determinize(s, random.Random(n))
            assert sig(d) == sig(s) and d.ruleset.draws == "pile"
            if s.n_reshuffles == 0:
                assert pile_split(d)[1] == pile_split(s)[1], "the setup tail stays at the bottom"
            n += 1
    assert n > 500


def _shuffled(s, rng):
    cards, tail = unknown_cards(s)
    rng.shuffle(cards)
    return replace(s, draw_pile=(*cards, *tail), seed=rng.getrandbits(40))


def test_rollout_values_never_read_the_pile_order_or_the_seed():
    n = reshuffled = 0
    for rs, count in ((RESH, 40), (replace(RESH, draws="tickets"), 20), (rules.FULL_UNO, 2)):
        agents = (Heuristic(rs), PlayFirstRandom(rs))
        for i, (s, a, _) in enumerate(games(rs, count)):
            if rs is rules.FULL_UNO and i % 10:
                continue
            other = _shuffled(s, random.Random(i))
            assert rollout_value(s, agents, 3, rollout_rng(7, i)) == rollout_value(other, agents, 3, rollout_rng(7, i))
            assert rollout_q(s, a, agents, 3, rollout_rng(8, i)) == rollout_q(other, a, agents, 3, rollout_rng(8, i))
            n += 1
            reshuffled += s.n_reshuffles > 0
    assert n > 100 and reshuffled > 20


def test_first_draw_is_the_chance_model():
    """The children of a drawing action over many determinizations are distributed as
    `draw_outcomes` says, reshuffles and empty-pile voluntary draws included."""
    checked = empty = 0
    for s, a, _ in games(RESH, 200):
        if draw_count(s, a) == 0 or checked >= 25:
            continue
        model = draw_outcomes(s, a, enum_cap=10**6)
        if model is None:
            continue
        want = Counter()
        for c, p in model:
            want[sig(c)] += p
        if len(want) < 2:
            continue
        rng = random.Random(checked)
        m = 3000
        got = Counter(sig(step(determinize(s, rng), a, check=False)) for _ in range(m))
        assert set(got) <= set(want)
        for k, p in want.items():
            assert abs(got[k] / m - p) <= 5 * math.sqrt(p * (1 - p) / m) + 1e-9
        checked += 1
        empty += not s.draw_pile
    assert checked >= 25 and empty >= 1


@cache
def small_profile():
    tree = compile_tree(SMALL, (Level.L1, Level.L2), symmetric=True)
    heu, rnd = Heuristic(SMALL), PlayFirstRandom(SMALL)
    prof = ((Level.L1, tabulate(tree, Level.L1, heu)), (Level.L1, tabulate(tree, Level.L1, rnd)))
    return tree, (heu, rnd), prof


def test_rollouts_match_the_exact_values():
    tree, agents, prof = small_profile()
    idx = build(tree, Level.L2)
    nu = exact_values(tree, *prof)
    alpha = arc_expectation(tree, nu)
    n, zs = 1000, []
    for i, (s, a, _) in enumerate(games(SMALL, 60)):
        if i % 4:
            continue
        node, arcs = idx.arcs(s, (a,))
        for (m, v), exact in ((rollout_value(s, agents, n, rollout_rng(1, i)), nu[node]), (rollout_q(s, a, agents, n, rollout_rng(2, i)), alpha[arcs[a]])):
            if v > 0:
                zs.append((m - exact) / math.sqrt(v))
            else:
                assert abs(m - exact) < 1e-12, "a certain outcome must be the exact value"
    assert len(zs) > 20 and max(abs(z) for z in zs) <= 4.5


def test_deal_floor_matches_the_exact_residual():
    tree, agents, prof = small_profile()
    nu = exact_values(tree, *prof)
    raw = aivat_moments(tree, *prof, nu, NONE, 0).var
    exact = aivat_moments(tree, *prof, nu, Corrections(True, True, True, False), 0).var / raw
    f = deal_floor(SMALL, agents, range(20_000, 24_000), 8, workers=1)
    sd = (f.share_ci[1] - f.share_ci[0]) / 3.92
    assert abs(f.share - exact) <= 4 * sd, (f.share, exact, f.share_ci)
    assert abs(f.total - raw) <= 0.05 * raw
