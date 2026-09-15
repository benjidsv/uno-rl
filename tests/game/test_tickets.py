"""Ticket draws (docs/spec.md section 8, step 9): the same game as the pile, coupled across games."""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import replace

import pytest

from tests.strategies import SMALL
from uno.agents.baseline import Heuristic, Mixture, PlayFirstRandom, UniformLegal
from uno.compiled.solvers import evaluate
from uno.compiled.strategies import tabulate
from uno.compiled.tree import compile_tree
from uno.eval.estimate import play_pairs
from uno.eval.play import play_game
from uno.game.chance import initial_deals, worlds
from uno.game.engine import ticket_index
from uno.game.observe import Level
from uno.game.rules import FULL_UNO
from uno.reference.solve import scores_on_seeds


def test_ticket_draw_is_uniform_over_the_pile():
    rs = FULL_UNO
    pile = [0, 0, 1, 5]  # two copies of type 0 when the deck has them
    if rs.deck_counts[0] < 2:
        pile = [1, 1, 2, 5]
    n = 40_000
    counts = Counter(pile[ticket_index(rs, 7, 0, k, pile)] for k in range(n))
    for t, c in Counter(pile).items():
        p = c / len(pile)
        assert abs(counts[t] / n - p) < 4 * (p * (1 - p) / n) ** 0.5, (t, counts[t] / n, p)


def test_tickets_couple_piles_that_differ_by_one_card():
    rs = FULL_UNO
    rng = random.Random(3)
    same = 0
    for k in range(2000):
        pile = rng.sample(rs.deck, 40)
        other = pile[:]
        other.pop(rng.randrange(40))
        rng.shuffle(other)
        same += pile[ticket_index(rs, 11, 1, k, pile)] == other[ticket_index(rs, 11, 1, k, other)]
    assert same / 2000 > 0.93  # the top of a reshuffled pile would agree ~1 time in 20


def test_ticket_games_are_valid_and_deterministic():
    rs = replace(FULL_UNO, draws="tickets")
    for seed in range(20):
        agents = (Heuristic(rs), PlayFirstRandom(rs))
        t = play_game(rs, seed, agents)
        assert t.actions == play_game(rs, seed, agents).actions
        s = t.final
        assert s.terminal
        assert Counter(s.hands[0]) + Counter(s.hands[1]) + Counter(s.draw_pile) + Counter(s.discard) == Counter(rs.deck)


def test_exact_trees_refuse_tickets():
    rt = replace(SMALL, draws="tickets")
    with pytest.raises(ValueError):
        initial_deals(rt)
    with pytest.raises(ValueError):
        worlds(rt)
    assert repr(rt) == repr(SMALL)  # cached trees are named by the repr, and only ever compiled with pile draws


@pytest.mark.parametrize("draws", ["pile", "tickets"])
def test_deal_model_on_seeds(draws):
    """The exact score over many real deals (and tickets) averages to the whole-game
    exact value of the chance model, within the deal variance; a value off by 3
    points is rejected. With voluntary draws, so ticket games part in draw counts."""
    rs = SMALL
    tree = compile_tree(rs, (Level.L1,))
    a, b = UniformLegal(rs), PlayFirstRandom(rs)
    exact = (evaluate(tree, (Level.L1, tabulate(tree, Level.L1, a)), (Level.L1, tabulate(tree, Level.L1, b)))[0] + 1) / 2
    on = scores_on_seeds(replace(rs, draws=draws), (a, b), range(2500))
    assert abs(on.z_global(exact, 0)) <= 3, on.z_global(exact, 0)
    assert abs(on.z_global(exact + 0.03, 0)) > 3


def test_shared_streams():
    rs = SMALL
    rnd, heu = PlayFirstRandom(rs), Heuristic(rs)
    mix = Mixture(heu, rnd, 0.0, name="heuristic_eps0", stream="heuristic")
    for seed in range(30):  # eps = 0 on the Heuristic's stream is the Heuristic, choice for choice
        assert play_game(rs, seed, (mix, rnd)).actions == play_game(rs, seed, (heu, rnd)).actions
    with pytest.raises(ValueError):
        play_pairs(rs, (mix, heu), range(3))  # one stream in both games of a mirrored pair
