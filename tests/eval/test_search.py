"""The PIMC agent on rulesets with no tree (`uno.eval.search`, step 9.5d): determinization at L1
keeps everything the observer can see, samples the rest uniformly, and always contains the real
world; the depth-limited search equals plain minimax; and the agents report the policy they
actually follow, which is what keeps AIVAT unbiased.
"""

from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np
import pytest

from tests.game.test_draw_outcomes import RESH
from uno.agents.baseline import Heuristic, PlayFirstRandom
from uno.eval.aivat import terminal_return
from uno.eval.search import PimcRollout, PimcSearch, determinize_l1, search_q, search_value, unseen_cards
from uno.game import rules
from uno.game.chance import pile_split
from uno.game.engine import legal_actions, new_game, returns, step
from uno.game.observe import Level, observe


def _decisions(rs, games: int, cap: int = 10**9):
    """(state, the mover's L1 observation) along random-policy games."""
    for seed in range(games):
        try:
            s = new_game(rs, seed)
        except ValueError:
            continue
        rng = random.Random(seed)
        n = 0
        while not s.terminal and n < cap:
            yield s, observe(s, s.player, Level.L1)
            s = step(s, rng.choice(legal_actions(s)))
            n += 1


@pytest.mark.parametrize("name", ("TOY", "TOY_FULL", "FULL_UNO"))
def test_determinization_keeps_everything_the_observer_can_see(name):
    """A determinized world must be a *possible* world: same own hand, discard, top, colour, phase
    and counts, the deck conserved, the same legal actions, the setup tail still at the bottom of
    the pile, and the real opponent hand among the cards it could have dealt."""
    rs = getattr(rules, name)
    n = 0
    for s, obs in _decisions(rs, 6 if name == "FULL_UNO" else 20):
        p = obs.seat
        d = determinize_l1(rs, obs, random.Random(n))
        assert d.hands[p] == obs.hand and len(d.hands[p ^ 1]) == obs.opp_hand_size
        assert len(d.draw_pile) == obs.pile_size and d.discard[-1] == obs.top
        assert (d.current_color, d.player, d.phase, d.drawn) == (obs.current_color, obs.player, obs.phase, obs.drawn)
        assert Counter(d.hands[0]) + Counter(d.hands[1]) + Counter(d.draw_pile) + Counter(d.discard) == Counter(rs.deck)
        assert legal_actions(d) == legal_actions(s), "the determinization changed what the mover may do"
        if d.n_reshuffles == 0:
            pile_split(d)  # asserts the publicly known skipped cards are still at the bottom
        unseen, _ = unseen_cards(rs, obs)
        assert not (Counter(s.hands[p ^ 1]) - Counter(unseen)), "the real world is not in the support"
        n += 1
    assert n > 40


def test_determinization_samples_the_unseen_cards_uniformly():
    """The opponent's hand is a uniform draw from the unseen multiset, so each card type reaches
    it with the hypergeometric probability. A biased split would make PIMC's belief wrong and, on
    the toys, break its agreement with the exact belief of `uno.compiled.pimc`."""
    rs = rules.FULL_UNO
    s = new_game(rs, 7)
    for _ in range(4):
        s = step(s, legal_actions(s)[0])
    obs = observe(s, s.player, Level.L1)
    unseen, _ = unseen_cards(rs, obs)
    mult, k, N = Counter(unseen), obs.opp_hand_size, 4000
    seen = Counter()
    for i in range(N):
        seen.update(set(determinize_l1(rs, obs, random.Random(i)).hands[obs.seat ^ 1]))
    for t, m in mult.items():
        expected = 1.0 - math.comb(len(unseen) - m, k) / math.comb(len(unseen), k)
        se = math.sqrt(expected * (1.0 - expected) / N)
        assert abs(seen[t] / N - expected) <= 4.0 * se, f"type {t} appears with the wrong probability"


def _leaf(s):
    """A deterministic, non-constant stand-in for a value function."""
    return math.sin(len(s.hands[0]) * 1.7 + len(s.hands[1]) * 2.3 + s.current_color + len(s.draw_pile) * 0.11)


def _plain(s, seat, depth, parent=None, action=-1):
    if s.terminal:
        u = returns(s)[0] if parent is None else terminal_return(parent, action, s, "hand")
        return u if seat == 0 else -u
    if depth <= 0:
        return _leaf(s) if seat == 0 else -_leaf(s)
    vals = [_plain(step(s, a, check=False), seat, depth - 1, s, a) for a in legal_actions(s)]
    return max(vals) if s.player == seat else min(vals)


@pytest.mark.parametrize("name,depth", (("TOY", 5), ("TOY_FULL", 4), ("FULL_UNO", 3)))
def test_alpha_beta_equals_plain_minimax(name, depth):
    """With two players a skip or reverse means the same player moves again, so the maximising
    side is read off `s.player` and not from the parity of the depth; pruning must still not
    change a value."""
    rs = getattr(rules, name)
    n = 0
    for s, _ in _decisions(rs, 4 if name == "FULL_UNO" else 10, cap=8):
        for seat in (0, 1):
            assert search_value(s, seat, depth, _leaf) == pytest.approx(_plain(s, seat, depth), abs=1e-12)
            n += 1
    assert n > 20


def test_search_q_is_the_value_of_each_action():
    rs = rules.TOY
    n = 0
    for s, _ in _decisions(rs, 10, cap=6):
        for seat in (0, 1):
            for a, v in search_q(s, seat, 4, _leaf).items():
                assert v == pytest.approx(_plain(step(s, a, check=False), seat, 3, s, a), abs=1e-12)
                n += 1
    assert n > 40


def test_the_sampled_agent_reproduces_the_exact_pimc_on_a_toy():
    """The loop the toys exist to close: the sampled agent must land on the exact tree's answer.

    On `TOY` a depth of 12 always reaches a terminal, so every determinized world is solved
    exactly and the leaf is never consulted -- the leaf here raises, which proves it. Splitting
    the unseen cards uniformly *is* the prior belief, and solving a world two-sided *is* the
    "minimax" continuation, so the agent is exactly
    `uno.compiled.pimc.pimc_n(weighting="prior", continuation="minimax")`.

    Agreement is statistical, not exact: on the quotient the exact version plays a whole action
    orbit uniformly while the agent picks one action (the lowest id), and 284 of TOY's 12,036 L1
    sets have a non-trivial orbit.
    """
    from uno import store
    from uno.compiled.pimc import pimc_n
    from uno.compiled.strategies import transfer
    from uno.compiled.tree import compile_tree
    from uno.eval.play import EVAL_SEEDS, play_from
    from uno.game.engine import new_game, returns

    def no_leaf(s):
        raise AssertionError("the search hit its depth limit: the world was not solved exactly")

    rs = rules.TOY
    world = compile_tree(rs, (Level.L1, Level.L3), True, cache=store.CACHE)
    lazy = compile_tree(rs, (Level.L1,), True, cache=store.CACHE)
    rnd = PlayFirstRandom(rs)
    opp = (Level.L1, transfer(lazy, world, Level.L1, store.load_or_tabulate(lazy, Level.L1, rnd)))
    n = 4
    agent = PimcSearch(rs, no_leaf, worlds=n, depth=12, name="pimc")
    scores = ([], [])
    for seed in EVAL_SEEDS[:200]:
        s0 = new_game(rs, seed)
        scores[0].append((returns(play_from(s0, seed, (agent, rnd)).final)[0] + 1) / 2)
        scores[1].append((returns(play_from(s0, seed, (rnd, agent)).final)[1] + 1) / 2)
    for seat in (0, 1):
        exact = pimc_n(world, opp, seat, n, realizations=32, weighting="prior", continuation="minimax", seed=1)
        xs = np.array(scores[seat])
        z = (xs.mean() - exact.score) / (xs.std(ddof=1) / math.sqrt(len(xs)))
        assert abs(z) <= 3.5, f"seat {seat}: sampled {xs.mean():.4f} vs exact {exact.score:.4f}, z {z:+.2f}"


def test_scores_split_is_the_same_worlds_cut_in_two():
    """The halves must average back to the agent's own full-sample scores: `scores_split` replays
    the identical rng sequence, taking the first half to choose with and the second to value with.
    That is what buys independence between selection and valuation at no extra worlds.

    Pinned because reading a reference value off the worlds that chose the action is the winner's
    curse, and summing it over a game's decisions inflated a D2 total by ~34 score points per game
    (docs/spec.md section 25)."""
    rs = RESH
    agent = PimcSearch(rs, _leaf, worlds=6, depth=2, name="pimc")  # even, so the halves are exhaustive
    n = 0
    for s, obs in _decisions(rs, 8, cap=8):
        legal = legal_actions(s)
        if len(legal) < 2:
            continue
        sel, val = agent.scores_split(obs, legal)
        full = agent._scores(obs, legal)
        assert set(sel) == set(val) == set(legal)
        for a in legal:
            assert (sel[a] + val[a]) / 2 == pytest.approx(full[a], abs=1e-12)
        n += 1
    assert n > 10


@pytest.mark.parametrize("build", (
    lambda rs: PimcSearch(rs, _leaf, worlds=3, depth=2, name="pimc"),
    lambda rs: PimcRollout(rs, (Heuristic(rs), PlayFirstRandom(rs)), worlds=3, name="pimc_rollout"),
))  # fmt: skip
def test_the_reported_policy_is_the_one_actually_followed(build):
    """The agents sample worlds inside `policy`, so the policy they report would be a lie -- and
    AIVAT's row correction would stop being zero-mean -- if the sampling were not keyed on the
    information state. Asking twice must give the same answer, and it must be a legal point mass.
    """
    rs = RESH  # reshuffles, +2 and +4: the awkward chance paths
    agent = build(rs)
    n = 0
    for s, obs in _decisions(rs, 12, cap=10):
        legal = legal_actions(s)
        p = agent.policy(obs, legal)
        assert p == agent.policy(obs, legal), "the policy is not a function of the information state"
        assert sum(p.values()) == pytest.approx(1.0) and set(p) <= set(legal)
        n += 1
    assert n > 40
