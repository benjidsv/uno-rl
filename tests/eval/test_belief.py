"""The opponent-hand belief (`uno.eval.belief`, step 9.5f).

The dangerous failure of any inference is **over-pruning**: a belief that rules out the world the
opponent is actually in looks confident and is wrong, and nothing downstream would notice -- PIMC
would simply search a set of worlds that excludes the truth. So the load-bearing tests here are
the two that assert the truth survives, one per rung:

- `voids` must never exclude the hand the opponent really holds **against an opponent that plays a
  playable drawn card at once** -- which every play-first rulebot does, and which is an assumption,
  not a rule (see `test_voids_over_prunes_against_an_opponent_that_keeps_a_playable_drawn_card`);
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

from uno.agents.baseline import Heuristic, KeepsDrawn, PlayFirstRandom
from uno.eval.belief import Belief, satisfies, trailing_window, voids
from uno.eval.play import play_game
from uno.eval.search import PimcSearch, determinize_l1, unseen_cards
from uno.game import rules
from uno.game.engine import Phase, legal_actions, new_game, step
from uno.game.observe import Level, observe
from uno.game.rules import Play


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
    """The guard against over-pruning -- **for the opponents this plays against**, which is the
    scope the rung is actually trustworthy over.

    A voluntary draw proves the drawer held nothing playable against the board then in force; if
    they have drawn `k` cards since, at most `k` of their cards now may be playable against it.
    Measured on full UNO before this rung was written: 0 violations in 2,487 decisions, and this
    reproduces that.

    That condition is necessary **given that the card drawn at the void never stays in hand as a
    playable one** -- i.e. that a playable drawn card is played at once. `_row_decisions` plays
    rulebots, which always do. It is not a rule: `draw_then_may_play` lets a player keep it by
    passing, and a Pass writes no public event, so nothing distinguishes the two. Drop the
    assumption and this test fails 12.1 % of the time; that is the companion test below, and the
    reason this one is scoped rather than universal.
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


# The passer moved to `uno.agents.baseline.KeepsDrawn`: the closure test needs its *exact*
# posterior, which means tabulating it on a compiled tree, so it could no longer live in a test
# file. It also spreads the play mass over the drawn card's actions instead of taking the first,
# which keeps it symmetric; nothing here depends on that.


def test_voids_over_prunes_against_an_opponent_that_keeps_a_playable_drawn_card():
    """The scope of `voids`, pinned: it is a *policy* assumption, not a rules fact.

    The rung's own module docstring once claimed it "needs no policy assumption whatever" on a
    forced-play ruleset, and that claim is what justified preferring it over `window` as the half
    that survives into a general player. It is wrong. The constraint counts the cards drawn
    *strictly after* the void, so it implicitly assumes the card drawn **at** the void did not
    stay in hand as a playable one -- true of every play-first rulebot, false in general.

    The sound version is provably vacuous, which is why the assumption cannot simply be removed:
    at the void the hand holds nothing playable, and every card that could be playable since was
    drawn since, so "playable now <= drawn since" is a tautology and excludes nothing at all
    (measured exactly on the `TOY` world tree: it keeps 100 % of worlds at both seats).

    Both opponents run here, because the contrast is the point: a test that only ever plays
    rulebots verifies a scoped fact while reading like a universal one, and that is precisely how
    this went unnoticed through step 9.5f.
    """
    rs = rules.FULL_UNO
    heu = Heuristic(rs)
    out = {}
    for label, opp in (("rulebot", PlayFirstRandom(rs)), ("passer", KeepsDrawn(rs))):
        checked = violations = 0
        for seed in range(25):
            try:
                t = play_game(rs, seed, (heu, opp))
            except ValueError:
                continue
            for st in t.steps:
                if st.state.player != 0:
                    continue
                vs = voids(rs, st.obs.public, 1)
                if not vs:
                    continue
                checked += 1
                violations += not satisfies(rs, st.state.hands[1], vs)
        out[label] = (violations, checked)
    assert out["rulebot"][1] > 50 and out["passer"][1] > 50, f"too few constrained decisions to conclude: {out}"
    assert out["rulebot"][0] == 0, f"voids excluded the true hand against a rulebot: {out['rulebot']}"
    assert out["passer"][0] > 0, (
        "voids never excluded the true hand against an opponent that keeps playable drawn cards, "
        f"so this test no longer pins the assumption: {out['passer']}"
    )


class _Bluffer:
    """`PlayFirstRandom`, except it sometimes draws *while holding a playable card*.

    This is the move that breaks the void premise itself, as opposed to `_Passer`, which breaks
    only the accounting of the card drawn at the void. Legal on any free-play ruleset."""

    name = "bluffer"

    def __init__(self, rs, p: float = 0.4) -> None:
        self.rs, self.p, self.inner = rs, p, PlayFirstRandom(rs)

    def policy(self, obs, legal):
        if obs.phase is Phase.TURN_START:
            draw = self.rs.draw_action
            plays = [a for a in legal if isinstance(self.rs.actions[a], Play)]
            if draw in legal and plays:
                rest = self.inner.policy(obs, tuple(plays))
                return {draw: self.p, **{a: (1.0 - self.p) * q for a, q in rest.items()}}
        return self.inner.policy(obs, legal)


def _void_cases(rs, opp, games: int = 12):
    """(true opponent hand, the voids in force) at the row's constrained decisions."""
    heu = Heuristic(rs)
    for seed in range(games):
        try:
            t = play_game(rs, seed, (heu, opp))
        except ValueError:
            continue
        for st in t.steps:
            if st.state.player != 0:
                continue
            vs = voids(rs, st.obs.public, 1)
            if vs:
                yield tuple(sorted(st.state.hands[1])), vs


@pytest.mark.parametrize("who", ("rulebot", "passer", "bluffer"))
def test_soft_never_puts_zero_on_the_hand_the_opponent_really_holds(who):
    """The reason the rung exists. `voids` is a hard filter, so when its assumption breaks it
    assigns the truth probability *zero* and no amount of search recovers it -- measured at 12.3 %
    of constrained decisions against `KeepsDrawn` and 68.5 % against `_Bluffer`. `soft` weights
    instead of rejecting, so the truth is never excluded whatever the opponent does.

    The `bluffer` case doubles as the non-vacuity guard: `voids` must actually hit zero there, or
    this would pass with `soft` doing nothing.
    """
    rs = rules.FULL_UNO
    opp = {"rulebot": PlayFirstRandom(rs), "passer": KeepsDrawn(rs), "bluffer": _Bluffer(rs)}[who]
    soft = Belief(rs, kind="soft")  # the shipped default, so this guards what actually runs
    n = hard_zero = 0
    for truth, vs in _void_cases(rs, opp):
        assert soft._void_weight(truth, vs) > 0.0, "the soft rung excluded the hand the opponent really holds"
        hard_zero += not satisfies(rs, truth, vs)
        n += 1
    assert n > 50, f"too few constrained decisions to conclude ({n})"
    if who == "bluffer":
        assert hard_zero > 0, "`voids` never excluded the truth here, so this test no longer pins anything"


def test_soft_spans_voids_and_the_prior():
    """`decay = 0` must reproduce `voids` exactly and `decay = 1` the uniform prior, which is what
    makes the rung a one-parameter family between the two rather than a third unrelated thing."""
    rs = rules.FULL_UNO
    hard, none_ = Belief(rs, kind="soft", decay=0.0), Belief(rs, kind="soft", decay=1.0)
    n = 0
    for truth, vs in _void_cases(rs, _Bluffer(rs)):
        assert hard._void_weight(truth, vs) == (1.0 if satisfies(rs, truth, vs) else 0.0)
        assert none_._void_weight(truth, vs) == 1.0
        n += 1
    assert n > 50


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


def _observations(rs, games: int, cap: int = 10**9):
    """The mover's L1 observation along random-policy games."""
    for seed in range(games):
        s = new_game(rs, seed)
        rng = random.Random(seed)
        n = 0
        while not s.terminal and n < cap:
            yield observe(s, s.player, Level.L1)
            s = step(s, rng.choice(legal_actions(s)))
            n += 1


def test_the_constraint_memo_follows_the_decision_it_was_derived_from():
    """`Belief` memoises what a decision derives from its public history, because every world of
    that decision asks for the same thing. The failure a memo has is not being slow, it is being
    *stale*: a belief reused across decisions -- which is exactly how one is used, one object per
    rung for a whole run -- must answer for the history in front of it and not the one before.

    So this walks real decisions with a single reused `Belief` and demands the same answers a
    fresh one gives each time, and it checks the histories actually differ, or a memo that never
    invalidated would pass."""
    rs = rules.FULL_UNO
    rnd = PlayFirstRandom(rs)
    shared = {k: Belief(rs, kind=k, opponent=rnd if k == "window" else None) for k in ("voids", "window")}
    seen, n = set(), 0
    for obs in _observations(rs, 3, cap=14):
        opp = obs.seat ^ 1
        seen.add(len(obs.public))
        for kind, bel in shared.items():
            fresh = Belief(rs, kind=kind, opponent=rnd if kind == "window" else None)
            assert bel._constraints(obs, opp) == fresh._constraints(obs, opp), kind
            unseen, _ = unseen_cards(rs, obs)
            assert bel.hand(obs, list(unseen), random.Random(3)) == fresh.hand(obs, list(unseen), random.Random(3))
        n += 1
    assert n > 20 and len(seen) > 5, "the decisions did not have differing histories, so staleness could not show"
