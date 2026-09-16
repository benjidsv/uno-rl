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
from uno.eval.belief import Belief
from uno.eval.search import PimcRollout, PimcSearch, batched_search_q, determinize_l1, leaf_for, response_value, search_q, search_value, unseen_cards
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


def _plain_response(s, seat, depth, opp, parent=None, action=-1):
    """Reference expectimax: max at the responder's nodes, the opponent's own probabilities at
    its nodes. Independent of `response_value` only in its recursion, which is what it pins."""
    if s.terminal:
        u = returns(s)[0] if parent is None else terminal_return(parent, action, s, "hand")
        return u if seat == 0 else -u
    if depth <= 0:
        return _leaf(s) if seat == 0 else -_leaf(s)
    legal = legal_actions(s)
    vals = {a: _plain_response(step(s, a, check=False), seat, depth - 1, opp, s, a) for a in legal}
    if s.player == seat:
        return max(vals.values())
    probs = opp.policy(observe(s, s.player, Level.L1), legal)
    return sum(p * vals[a] for a, p in probs.items())


@pytest.mark.parametrize("name,depth", (("TOY", 5), ("TOY_FULL", 4), ("FULL_UNO", 3)))
def test_response_continuation_equals_a_plain_expectimax(name, depth):
    rs = getattr(rules, name)
    opp = Heuristic(rs)
    n = 0
    for s, _ in _decisions(rs, 4 if name == "FULL_UNO" else 10, cap=8):
        for seat in (0, 1):
            assert response_value(s, seat, depth, _leaf, opp) == pytest.approx(_plain_response(s, seat, depth, opp), abs=1e-12)
            n += 1
    assert n > 20


@pytest.mark.parametrize("name,depth", (("TOY", 5), ("TOY_FULL", 4), ("FULL_UNO", 3)))
def test_responding_to_the_real_opponent_is_never_worse_than_assuming_a_clairvoyant_one(name, depth):
    """The whole point of the `response` continuation, as an inequality that cannot come out of
    the same code twice: minimax takes the *minimum* over the opponent's children, and any
    expectation over those same children is at least that. A seat or sign error in either path
    breaks it immediately.

    It bounds the *searched* value, not the achieved score -- what the agent then wins is an
    empirical question (step 9.5e), and on the toys `uno.compiled.pimc` measures the gap.
    """
    rs = getattr(rules, name)
    opp = Heuristic(rs)
    n = 0
    for s, _ in _decisions(rs, 4 if name == "FULL_UNO" else 10, cap=8):
        for seat in (0, 1):
            assert response_value(s, seat, depth, _leaf, opp) >= search_value(s, seat, depth, _leaf) - 1e-12
            n += 1
    assert n > 20


def test_the_leaf_is_chosen_by_the_observers_seat():
    """Step 9.5d handed the agent the seat-0 network at both seats, so at seat 1 it searched with
    a value function fitted on a different state distribution. The seat-1 leaf here raises, which
    is what proves the dispatch actually happens."""
    rs = rules.TOY

    def boom(s):
        raise AssertionError("the seat-0 leaf was used at seat 1")

    assert leaf_for((_leaf, boom), 0) is _leaf and leaf_for((_leaf, boom), 1) is boom
    assert leaf_for(_leaf, 0) is _leaf and leaf_for(_leaf, 1) is _leaf, "a single leaf serves both seats"
    seen = 0
    for s, obs in _decisions(rs, 20, cap=8):
        legal = legal_actions(s)
        if len(legal) < 2:
            continue
        # the leaf that must never fire is the *other* seat's, so each observation exercises one
        leaves = (_leaf, boom) if obs.seat == 0 else (boom, _leaf)
        PimcSearch(rs, leaves, worlds=2, depth=2, name="pimc").policy(obs, legal)
        seen += 1
    assert seen > 20


def test_the_continuation_rule_changes_what_the_agent_plays():
    """Non-vacuity guard for the closure test below, and the sharpest check of the axis there is.

    On `TOY` against Random the two continuations are only 0.2-0.5 score points apart (exact,
    `uno.compiled.pimc`), while 200 sampled games carry a standard error of ~3.2 points, so a
    value comparison there *cannot* tell them apart: the closure test below would pass unchanged
    with the wrong continuation wired in. This asserts the axis is live by the one thing that
    carries no noise at all -- the actions themselves.

    Both agents are given the same `name` and `seed` on purpose. That is what makes them draw the
    identical worlds (`PimcSearch._rng` keys on the name), so any disagreement is the continuation
    rule and nothing else. Two agents sharing a name must never be *played* against each other
    (`uno.eval.play` raises); here they are only asked for a policy.
    """
    rs = rules.TOY_FULL  # free play, where the toys measured the continuation mattering most
    rnd = PlayFirstRandom(rs)
    mini = PimcSearch(rs, _leaf, worlds=4, depth=3, name="pimc", seed=0)
    resp = PimcSearch(rs, _leaf, worlds=4, depth=3, name="pimc", seed=0, opponent=rnd)
    assert (mini.continuation, resp.continuation) == ("minimax", "response")
    differ = n = 0
    for s, obs in _decisions(rs, 30, cap=10):
        legal = legal_actions(s)
        if len(legal) < 2:
            continue
        differ += mini.policy(obs, legal) != resp.policy(obs, legal)
        n += 1
    assert n > 40
    assert differ / n > 0.02, f"the continuation rule changed nothing in {n} decisions: is `opponent` reaching the search?"


def test_the_sampled_response_agent_reproduces_the_exact_pimc_on_a_toy():
    """The 9.5e loop, closed the way 9.5d closed the minimax one: solving each world against the
    opponent's real policy is `uno.compiled.pimc`'s `continuation="response"`, and splitting the
    unseen cards uniformly is its `weighting="prior"`, so the sampled agent must land on that
    exact number. Depth 12 always reaches a terminal on `TOY`, so the leaf is never consulted and
    the worlds are solved exactly -- the leaf raises, which proves it."""
    from uno import store
    from uno.compiled.pimc import pimc_n
    from uno.compiled.strategies import transfer
    from uno.compiled.tree import compile_tree
    from uno.eval.play import EVAL_SEEDS, play_from
    from uno.game.engine import new_game

    def no_leaf(s):
        raise AssertionError("the search hit its depth limit: the world was not solved exactly")

    rs = rules.TOY
    world = compile_tree(rs, (Level.L1, Level.L3), True, cache=store.CACHE)
    lazy = compile_tree(rs, (Level.L1,), True, cache=store.CACHE)
    rnd = PlayFirstRandom(rs)
    opp = (Level.L1, transfer(lazy, world, Level.L1, store.load_or_tabulate(lazy, Level.L1, rnd)))
    n = 4
    agent = PimcSearch(rs, no_leaf, worlds=n, depth=12, name="pimc", opponent=rnd)
    assert agent.continuation == "response"
    scores = ([], [])
    for seed in EVAL_SEEDS[:200]:
        s0 = new_game(rs, seed)
        scores[0].append((returns(play_from(s0, seed, (agent, rnd)).final)[0] + 1) / 2)
        scores[1].append((returns(play_from(s0, seed, (rnd, agent)).final)[1] + 1) / 2)
    for seat in (0, 1):
        exact = pimc_n(world, opp, seat, n, realizations=32, weighting="prior", continuation="response", seed=1)
        xs = np.array(scores[seat])
        z = (xs.mean() - exact.score) / (xs.std(ddof=1) / math.sqrt(len(xs)))
        assert abs(z) <= 3.5, f"seat {seat}: sampled {xs.mean():.4f} vs exact {exact.score:.4f}, z {z:+.2f}"


def test_the_response_agent_matches_the_exact_value_on_the_toy_that_separates_the_two_rules():
    """The closure test with power. `TOY` cannot separate the continuations (0.2-0.5 points apart,
    against ~3.2 points of sampling noise at 200 games), but `TOY_FULL` can: free play is where
    assuming a clairvoyant opponent is expensive, and exactly there the two rules sit 7-9 points
    apart (measured, n = 4 prior: 0.7158 two-sided against 0.7870 responding, vs Random at seat 0).

    So this asserts both halves: the sampled agent agrees with the exact `response` number, *and*
    it is nearer that than the `minimax` one -- which is what would fail if the continuation were
    mis-wired, and what the `TOY` test cannot see.
    """
    from uno import store
    from uno.compiled.pimc import pimc_n
    from uno.compiled.strategies import transfer
    from uno.compiled.tree import compile_tree
    from uno.eval.play import EVAL_SEEDS, play_from
    from uno.game.engine import new_game

    def no_leaf(s):
        raise AssertionError("the search hit its depth limit: the world was not solved exactly")

    rs = rules.TOY_FULL
    world = compile_tree(rs, (Level.L1, Level.L3), True, cache=store.CACHE)
    lazy = compile_tree(rs, (Level.L1,), True, cache=store.CACHE)
    rnd = PlayFirstRandom(rs)
    opp = (Level.L1, transfer(lazy, world, Level.L1, store.load_or_tabulate(lazy, Level.L1, rnd)))
    n = 4
    agent = PimcSearch(rs, no_leaf, worlds=n, depth=40, name="pimc", opponent=rnd)  # 40 > any TOY_FULL game
    scores = ([], [])
    for seed in EVAL_SEEDS[:200]:
        s0 = new_game(rs, seed)
        scores[0].append((returns(play_from(s0, seed, (agent, rnd)).final)[0] + 1) / 2)
        scores[1].append((returns(play_from(s0, seed, (rnd, agent)).final)[1] + 1) / 2)
    for seat in (0, 1):
        exact = {c: pimc_n(world, opp, seat, n, realizations=16, weighting="prior", continuation=c, seed=1).score for c in ("minimax", "response")}
        xs = np.array(scores[seat])
        z = (xs.mean() - exact["response"]) / (xs.std(ddof=1) / math.sqrt(len(xs)))
        assert abs(z) <= 3.5, f"seat {seat}: sampled {xs.mean():.4f} vs exact response {exact['response']:.4f}, z {z:+.2f}"
        if abs(exact["response"] - exact["minimax"]) > 0.03:  # only where the two rules really differ
            near = abs(xs.mean() - exact["response"]) < abs(xs.mean() - exact["minimax"])
            assert near, f"seat {seat}: sampled {xs.mean():.4f} is nearer the two-sided value {exact['minimax']:.4f} than {exact['response']:.4f}"


def test_a_precomputed_split_gives_the_identical_world():
    """`unseen_cards` depends on the observation alone, so an agent sampling `worlds` worlds of one
    decision recomputed the same multiset difference over the same history that many times.
    Passing it in must change nothing -- including the rng, which the uniform branch shuffles the
    list with, so the list has to be copied and not aliased."""
    rs = rules.FULL_UNO
    n = 0
    for _, obs in _decisions(rs, 4, cap=10):
        split = unseen_cards(rs, obs)
        for belief in (None, Belief(rs, kind="voids")):
            a = determinize_l1(rs, obs, random.Random(5), belief)
            b = determinize_l1(rs, obs, random.Random(5), belief, split)
            assert a == b
        # the caller's copy must survive being handed over twice
        assert split == unseen_cards(rs, obs)
        n += 1
    assert n > 10


# -- batched leaves (the leaf is consulted once per decision, not once per leaf state) -----------


class _BatchCountingLeaf:
    """`_leaf`, plus the batched form `batched_search_q` looks for, counting how often it is asked."""

    def __init__(self) -> None:
        self.calls = 0
        self.rows = 0

    def __call__(self, s):
        self.calls += 1
        self.rows += 1
        return _leaf(s)

    def batch(self, states):
        self.calls += 1
        self.rows += len(states)
        return [_leaf(s) for s in states]


@pytest.mark.parametrize("continuation", ("response", "minimax"))
@pytest.mark.parametrize("name,depth", (("TOY", 4), ("TOY2_SKIP", 4), ("TOY_FULL", 3), ("FULL_UNO", 3)))
def test_batched_search_q_equals_search_q_bit_for_bit(name, depth, continuation):
    """The batching may not change a value by so much as an ulp, so this asserts `==` on floats
    and not `approx`.

    It can hold because `_plan` walks the nodes `response_value` walks, in the order it walks them
    -- `probs.items()` at an opponent node, zero-probability actions skipped -- and `_fold` combines
    them with the same `max` and the same left-to-right `sum`. A tolerance here would pass on a plan
    that had quietly reordered an expectation, which is exactly the kind of drift that makes a
    'faster' row incomparable with the numbers already on the ladder.

    Under `minimax` the batched walk also *drops* `search_value`'s alpha-beta, so this is the check
    that pruning was only ever a speed-up: a min or a max selects one of its children's floats
    without arithmetic, and `search_q` opens a fresh window per root action, so every root value it
    returns is the exact one and not a fail-soft bound.
    """
    rs = getattr(rules, name)
    opp = Heuristic(rs) if continuation == "response" else None
    n = 0
    for s, _ in _decisions(rs, 4 if name == "FULL_UNO" else 8, cap=6):
        for seat in (0, 1):
            one = search_q(s, seat, depth, _leaf, opponent=opp)
            many = batched_search_q([s], seat, depth, _leaf, opponent=opp)[0]
            assert many == one, f"{name} seat {seat} {continuation}: batched {many} != {one}"
            n += 1
    assert n > 20


def test_batched_search_q_batches_across_worlds():
    """Non-vacuity: the point of the change is one leaf call for a whole *sample* of worlds, so a
    version that merely looped `search_q` would pass the equality test above and buy nothing."""
    rs = rules.FULL_UNO
    opp = Heuristic(rs)
    rng = random.Random(0)
    for s, obs in _decisions(rs, 3, cap=4):
        worlds = [determinize_l1(rs, obs, rng) for _ in range(8)]
        leaf = _BatchCountingLeaf()
        batched_search_q(worlds, obs.seat, 2, leaf, opponent=opp)
        if leaf.rows < 8:
            continue  # a decision whose whole tree terminates inside the depth has nothing to batch
        assert leaf.calls == 1, f"{leaf.rows} leaf states took {leaf.calls} calls, not one batch"
        return
    pytest.fail("no decision produced leaf states to batch")


def test_a_leaf_that_cannot_batch_still_works():
    """`batch_leaf` falls back to one call per state, so any plain callable is still a leaf."""
    rs = rules.FULL_UNO
    opp = PlayFirstRandom(rs)
    rng = random.Random(1)
    for s, obs in _decisions(rs, 2, cap=3):
        worlds = [determinize_l1(rs, obs, rng) for _ in range(4)]
        assert batched_search_q(worlds, obs.seat, 2, _leaf, opponent=opp) == batched_search_q(
            worlds, obs.seat, 2, _BatchCountingLeaf(), opponent=opp
        )


@pytest.mark.parametrize("continuation", ("response", "minimax"))
@pytest.mark.parametrize("name", ("TOY_FULL", "FULL_UNO"))
def test_the_batched_agent_plays_the_unbatched_agent_s_games(name, continuation):
    """The whole point of the speed-up: the row must **replay** the rows already on the ladder, not
    merely resample them. So the agent's scores must equal, to the bit, what the same agent would
    have scored calling its leaf once per leaf state -- which needs the worlds to be the same
    worlds, i.e. the rng consumed in the same order (`PimcSearch._worlds_q`).
    """
    rs = getattr(rules, name)
    opp = Heuristic(rs) if continuation == "response" else None
    agent = PimcSearch(rs, _leaf, worlds=6, depth=2, name=f"pimc_{continuation}", opponent=opp)

    def unbatched(obs, legal):
        rng = agent._rng(obs)
        total = dict.fromkeys(legal, 0.0)
        for _ in range(agent.worlds):
            world = determinize_l1(rs, obs, rng, agent.belief)
            q = search_q(world, obs.seat, agent.depth, leaf_for(agent.leaf, obs.seat), agent.exhaustion, opp)
            for a in legal:
                total[a] += q[a]
        return {a: v / agent.worlds for a, v in total.items()}

    n = 0
    for s, obs in _decisions(rs, 3 if name == "FULL_UNO" else 6, cap=8):
        legal = legal_actions(s)
        if len(legal) < 2:
            continue
        assert agent._scores(obs, legal) == unbatched(obs, legal)
        assert agent.policy(obs, legal) == {max(legal, key=lambda a: (unbatched(obs, legal)[a], -a)): 1.0}
        n += 1
    assert n > 10


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
