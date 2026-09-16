"""The opponent-hand belief (`uno.eval.belief`, step 9.5f).

The dangerous failure of any inference is **over-pruning**: a belief that rules out the world the
opponent is actually in looks confident and is wrong, and nothing downstream would notice -- PIMC
would simply search a set of worlds that excludes the truth. So the load-bearing tests here are
the two that assert the truth survives, one per rung:

- `voids` must never exclude the hand the opponent really holds;
- the `window` likelihood must be strictly positive for that hand, since the opponent demonstrably
  made those plays from it.

Both are checks a wrong implementation fails immediately, and neither can be satisfied by simply
disabling the feature -- which is what the non-vacuity test pins (the step 9.5e lesson: a test
that would pass with the feature turned off is not a test of it).
"""

from __future__ import annotations

import math
import random
from collections import Counter

import pytest

from uno.agents.baseline import Heuristic, PlayFirstRandom
from uno.eval.belief import Belief, satisfies, trailing_window, voids
from uno.eval.play import play_game
from uno.eval.search import PimcSearch, determinize_l1, unseen_cards
from uno.game import rules
from uno.game.engine import legal_actions, new_game, step
from uno.game.observe import Level, observe


def _row_decisions(rs, games: int, seat: int = 0):
    """(state, the row's L1 observation) at the row's own decisions of rulebot games."""
    heu, rnd = Heuristic(rs), PlayFirstRandom(rs)
    for seed in range(games):
        try:
            t = play_game(rs, seed, (heu, rnd) if seat == 0 else (rnd, heu))
        except ValueError:
            continue
        for st in t.steps:
            if st.state.player == seat:
                yield st.state, st.obs


@pytest.mark.parametrize("name", ("FULL_UNO", "TOY_FULL"))
def test_voids_never_exclude_the_hand_the_opponent_really_holds(name):
    """The guard against over-pruning, and the reason the rung is trustworthy at all.

    A voluntary draw proves the drawer held nothing playable against the board then in force; if
    they have drawn `k` cards since, at most `k` of their cards now may be playable against it.
    That is a *necessary* condition, so the real hand must satisfy every accumulated constraint.
    Measured on full UNO before this rung was written: 0 violations in 2,487 decisions.
    """
    rs = getattr(rules, name)
    checked = constrained = 0
    # the toy needs many more games to clear the non-vacuity guard below: a `TOY_FULL` game is
    # ~6 decisions on a 9-card deck, so voluntary draws (the only thing that creates a
    # constraint) are rare per game, where a full-UNO game carries ~41 decisions
    for state, obs in _row_decisions(rs, 12 if name == "FULL_UNO" else 200):
        vs = voids(rs, obs.public, obs.seat ^ 1)
        if not vs:
            continue
        constrained += 1
        true_hand = state.hands[obs.seat ^ 1]
        assert satisfies(rs, true_hand, vs), f"the void constraints exclude the opponent's real hand {true_hand}"
        checked += 1
    assert constrained > 20, "no decision carried a void constraint: the test is vacuous"
    assert checked == constrained


def test_the_void_constraint_actually_rules_worlds_out():
    """Non-vacuity (the step 9.5e lesson). If the constraint excluded nothing, the test above
    would pass with `voids` returning `[]` and the rung would be a no-op. Measured on full UNO:
    79.9 % of uniform determinizations are impossible, median surviving share 15.6 %."""
    rs = rules.FULL_UNO
    rng = random.Random(0)
    total = bad = decisions = 0
    for _state, obs in _row_decisions(rs, 10):
        vs = voids(rs, obs.public, obs.seat ^ 1)
        if not vs:
            continue
        unseen, _ = unseen_cards(rs, obs)
        decisions += 1
        for _ in range(16):
            h = tuple(sorted(rng.sample(unseen, obs.opp_hand_size)))
            total += 1
            bad += not satisfies(rs, h, vs)
    assert decisions > 20
    assert bad / total > 0.10, f"the void constraint excluded only {100 * bad / total:.1f} % of uniform worlds"


def test_the_window_likelihood_is_positive_for_the_hand_the_opponent_really_held():
    """The soft rung's version of the same guard. Over the trailing run of plays with no draw in
    between, the opponent's hand is recoverable as "the hand now plus the cards played since", so
    the hand they really hold must explain those plays with non-zero probability. A sign or
    ordering error in the roll-back makes this zero at once."""
    rs = rules.FULL_UNO
    rnd = PlayFirstRandom(rs)
    b = Belief(rs, kind="window", opponent=rnd)
    checked = 0
    for state, obs in _row_decisions(rs, 12):
        window = trailing_window(rs, obs.public, obs.seat ^ 1)
        if not window:
            continue
        true_hand = state.hands[obs.seat ^ 1]
        p = b._window_likelihood(obs, true_hand, window)
        assert p > 0.0, f"the real hand {true_hand} cannot explain the opponent's own trailing plays {window}"
        assert p <= 1.0 + 1e-12
        checked += 1
    assert checked > 20, "no decision had a trailing window: the test is vacuous"


@pytest.mark.parametrize("kind", ("voids", "window"))
def test_a_sampled_hand_satisfies_the_constraints_it_was_sampled_under(kind):
    """What the rung promises the caller: the hand it hands back is consistent, except on the
    counted fallback where no consistent hand was found within `tries`."""
    rs = rules.FULL_UNO
    rnd = PlayFirstRandom(rs)
    b = Belief(rs, kind=kind, opponent=rnd if kind == "window" else None)
    rng = random.Random(3)
    checked = 0
    for _state, obs in _row_decisions(rs, 8):
        vs = voids(rs, obs.public, obs.seat ^ 1)
        if not vs:
            continue
        unseen, _ = unseen_cards(rs, obs)
        before = b.fallbacks
        h = b.hand(obs, unseen, rng)
        assert len(h) == obs.opp_hand_size
        assert not (Counter(h) - Counter(unseen)), "the hand uses cards the observer has already seen"
        if b.fallbacks == before:
            assert satisfies(rs, h, vs)
            checked += 1
    assert checked > 20


@pytest.mark.parametrize("kind", ("voids", "window"))
def test_a_determinized_world_is_still_consistent_with_the_observation(kind):
    """The belief only chooses *which* hand; everything `determinize_l1` guaranteed before must
    still hold, or the search would be solving worlds the observer can rule out directly."""
    rs = rules.FULL_UNO
    rnd = PlayFirstRandom(rs)
    b = Belief(rs, kind=kind, opponent=rnd if kind == "window" else None)
    n = 0
    for state, obs in _row_decisions(rs, 6):
        p = obs.seat
        d = determinize_l1(rs, obs, random.Random(n), b)
        assert d.hands[p] == obs.hand and len(d.hands[p ^ 1]) == obs.opp_hand_size
        assert len(d.draw_pile) == obs.pile_size and d.discard[-1] == obs.top
        assert (d.current_color, d.player, d.phase) == (obs.current_color, obs.player, obs.phase)
        assert Counter(d.hands[0]) + Counter(d.hands[1]) + Counter(d.draw_pile) + Counter(d.discard) == Counter(rs.deck)
        assert legal_actions(d) == legal_actions(state), "the determinization changed what the mover may do"
        n += 1
    assert n > 40


def _leaf(s):
    return math.sin(len(s.hands[0]) * 1.7 + len(s.hands[1]) * 2.3 + s.current_color + len(s.draw_pile) * 0.11)


def test_the_agent_stays_a_deterministic_function_of_the_information_state():
    """AIVAT subtracts `sum_a sigma(a|h) alpha(h, a)` using the probabilities the agent reports,
    so a search that sampled worlds afresh would report a deterministic policy while following a
    stochastic one and the row correction would stop being zero-mean (docs/spec.md section 25).
    The belief adds a second source of sampling inside `policy`, so the invariant is re-checked
    with it wired in."""
    rs = rules.TOY_FULL
    rnd = PlayFirstRandom(rs)
    agent = PimcSearch(rs, _leaf, worlds=3, depth=3, name="pimc", opponent=rnd, belief=Belief(rs, kind="voids"))
    n = 0
    for seed in range(20):
        s = new_game(rs, seed)
        rng = random.Random(seed)
        while not s.terminal and n < 400:
            obs = observe(s, s.player, Level.L1)
            legal = legal_actions(s)
            if len(legal) > 1:
                p = agent.policy(obs, legal)
                assert p == agent.policy(obs, legal), "the policy is not a function of the information state"
                assert sum(p.values()) == pytest.approx(1.0) and set(p) <= set(legal)
                n += 1
            s = step(s, rng.choice(legal))
    assert n > 40
