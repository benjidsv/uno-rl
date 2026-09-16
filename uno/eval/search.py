"""The PIMC agent on rulesets with no tree (step 9.5d, docs/spec.md section 25).

Section 24 determinizes at **L2** to *value* a state under a fixed profile. A PIMC player is an
L1 player: it cannot see the opponent's hand either, so `determinize_l1` samples that too, from
the cards the observer has not seen, and the choice is then driven by solving each sampled world.

**Determinization at L1.** The observer knows its own hand, the discard (rebuilt from the public
history) and, until the first reshuffle, the cards skipped at setup, which sit at the bottom of
the pile and are public. Everything else -- the opponent's hand and the pile -- is a uniformly
random split of the unseen multiset, which is the conditional the deal model implies once the
public history is fixed. `determinize_l1` deals `opp_hand_size` of them to the opponent, shuffles
the rest into the pile above the known tail, and sets `draws="pile"` with a fresh seed, so
reshuffles and rollouts inside the search are independent of the real game.

**The belief (step 9.5f).** That uniform split is `uno.compiled.pimc`'s `"prior"` weighting: PIMC
with no inference at all. Given a `uno.eval.belief.Belief`, `determinize_l1` instead draws the
opponent's hand from what their own actions implied -- a voluntary draw proves they held nothing
playable, and their trailing run of plays is scored by their real policy. The belief chooses only
the *hand*; the pile is the remainder in a uniform order either way. See docs/spec.md section 26,
including why the particle filter over the deal that docs/plan.md originally called for collapses
(a particle survives the opponent's first play with probability `hand_size / unseen` ~ 7/98).

The opponent's `initial_hands` / `own_draws` are fabricated to match the hand we just dealt them:
nothing downstream reads them (`uno.eval.value_net.state_features` and the baseline agents work
off `hands`, `discard`, `top`, `current_color`, `phase`, `drawn` and the pile length), but they
must not be *inconsistent with the hand*, or a later `observe` of that seat would disagree with
itself. They are not a reconstruction of what really happened, and no L1/L2 key of the opponent's
seat should be taken from a determinized state.

**The continuation rule (step 9.5e).** Solving each sampled world two-sided assumes the opponent
is clairvoyant too, which is classic PIMC and what step 9.5d shipped. `uno.compiled.pimc` measured
that choice exactly on the toys and it is the **worse** of the two rules by up to 8.8 score points,
worst on free play -- which full UNO is. So `PimcSearch` takes an optional `opponent`: given one,
each world is solved as a perfect-information best response to the policy it really follows
(`response_value`), the sampled analogue of `best_response_perfect`. Every ladder matchup knows its
column's policy, so this needs no training; it does make the agent opponent-aware, and on the toys
that *costs* exploitability, so the row must be labelled with the opponent it was built against.

**Leaves are per seat.** Step 9.5c fits one value network per seat and one agent object plays both
seatings of a deal, so `leaf` may be a `(seat 0, seat 1)` pair resolved per observation
(`leaf_for`); step 9.5d used the seat-0 network at both seats.

**Leaves are evaluated in one batch per decision.** A decision searches `worlds` determinized
worlds, all independent, so `batched_search_q` plans every one of their depth-limited trees before
consulting the leaf and then values all the leaves together. For a network leaf that is 2.9-3.0x
end to end, because a batch-of-1 torch forward is dominated by dispatch (78.5 us a row at batch 1,
7.0 us at batch 32-256). It changes no decision: the plan visits the same nodes in the same order
as `response_value` and combines them with the same operations, and the worlds are drawn from the
same rng in the same order, so the batched agent replays the unbatched one's games.

**Why the sampling is keyed, not free.** AIVAT's row correction subtracts
`sum_a sigma(a|h) alpha(h, a)` using the probabilities the agent reports. An agent that sampled
worlds afresh and returned a point mass on the winner would be reporting a *deterministic* policy
while actually following a stochastic one, and the correction would stop being zero-mean --
AIVAT would silently become biased. Keying the world sampling on the information state
(`stable_hash((seed, name, obs.info_state))`) makes the policy genuinely deterministic, so
`{a: 1.0}` is the truth, and it gives common random numbers across matchups for free, exactly as
`uno.eval.play` does for action sampling.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import replace
from typing import Callable

from uno.eval.aivat import terminal_return
from uno.eval.pimc import pile_ruleset, rollout
from uno.eval.play import stable_hash
from uno.game.engine import Phase, State, legal_actions, returns, step
from uno.game.observe import Level, Observation, discard_from_public, observe
from uno.game.rules import Ruleset


def unseen_cards(rs: Ruleset, obs: Observation) -> tuple[list[int], tuple[int, ...]]:
    """(the cards the observer has not seen, sorted; the publicly known tail at the pile bottom).

    Unseen = deck - own hand - discard - the setup's skipped cards while they are still at the
    bottom of the pile. After a reshuffle the skipped cards have been dealt back into the pile and
    nothing of it is known, which is the same split `uno.eval.pimc.unknown_cards` makes at L2.
    """
    reshuffles = sum(1 for ev in obs.public if ev[0] == "reshuffle")
    skipped = tuple(obs.public[0][2])
    # the skipped cards sit at the *bottom* of the pile in order and are drawn last, so what is
    # still known is their last `t` (`uno.game.chance.pile_split`), not their first.
    t = 0 if reshuffles else min(len(skipped), obs.pile_size)
    tail = skipped[len(skipped) - t :] if t else ()
    left = Counter(rs.deck) - Counter(obs.hand) - Counter(discard_from_public(obs.public)) - Counter(tail)
    unseen = sorted(left.elements())
    if len(unseen) != obs.opp_hand_size + obs.pile_size - len(tail):
        raise ValueError(f"unseen cards ({len(unseen)}) do not fill the opponent's hand ({obs.opp_hand_size}) and the pile ({obs.pile_size - len(tail)})")
    return unseen, tail


def determinize_l1(rs: Ruleset, obs: Observation, rng: random.Random, belief=None) -> State:
    """A state consistent with `obs`: the unseen cards split into the opponent's hand and the
    pile. See the module docstring on the fabricated opponent memory.

    Without `belief` the split is **uniform**, which is `uno.compiled.pimc`'s `"prior"` weighting:
    PIMC with no inference at all. A `uno.eval.belief.Belief` instead draws the opponent's hand
    from what their own actions implied (step 9.5f), which is the `"posterior"` direction. The
    belief only chooses the *hand*; the pile is the remainder in a uniform order either way.
    """
    unseen, tail = unseen_cards(rs, obs)
    me, opp = obs.seat, obs.seat ^ 1
    if belief is None:
        rng.shuffle(unseen)
        opp_hand = tuple(sorted(unseen[: obs.opp_hand_size]))
        rest = list(unseen[obs.opp_hand_size :])
    else:
        opp_hand = belief.hand(obs, unseen, rng)
        rest = list((Counter(unseen) - Counter(opp_hand)).elements())
        rng.shuffle(rest)
    pile = (*rest, *tail)
    hands, initial, draws = [None, None], [None, None], [None, None]
    hands[me], hands[opp] = obs.hand, opp_hand
    initial[me], initial[opp] = obs.initial_hand, opp_hand  # fabricated for the opponent: consistent with the hand, not with history
    draws[me], draws[opp] = obs.own_draws, ()
    return State(
        ruleset=pile_ruleset(rs),
        seed=rng.getrandbits(62),
        hands=(hands[0], hands[1]),
        draw_pile=pile,
        discard=discard_from_public(obs.public),
        current_color=obs.current_color,
        player=obs.player,
        phase=obs.phase,
        drawn=obs.drawn,
        public=obs.public,
        own_draws=(draws[0], draws[1]),
        initial_hands=(initial[0], initial[1]),
        n_reshuffles=sum(1 for ev in obs.public if ev[0] == "reshuffle"),
        turn=obs.turn,
        end=None,
        winner=None,
    )


# -- depth-limited perfect-information search ---------------------------------------------------

Leaf = Callable[[State], float]  # seat-0 return estimate of a non-terminal state
Leaves = Leaf | tuple[Leaf, Leaf]  # one leaf, or one per seat of the *observer* (step 9.5e)


def leaf_for(leaf: Leaves, seat: int) -> Leaf:
    """The leaf value function an agent sitting at `seat` should use.

    Step 9.5c fits one value network per seat, and a single agent object plays both seatings of
    a deal (`uno.experiments.aivat_uno.evaluate_matchup` builds `(row, col)` once and
    `_sweep_chunk` also plays `pair[::-1]`), so the seat is not known at construction and the
    choice has to be made per observation. Step 9.5d's `uno` and `sweep` sections handed the
    agent `nets[0]` at *both* seats: both nets return seat-0 values, so it was not a sign error,
    but at seat 1 the search evaluated its leaves with a network fitted on a different state
    distribution, and every seat-1 number of that step is mildly pessimistic because of it.
    """
    return leaf[seat] if isinstance(leaf, tuple) else leaf


def search_value(
    s: State,
    seat: int,
    depth: int,
    leaf: Leaf,
    exhaustion: str = "hand",
    parent: State | None = None,
    action: int = -1,
    alpha: float = -math.inf,
    beta: float = math.inf,
) -> float:
    """Minimax value of `s` to `seat`, in return units, cut off at `depth` decisions and evaluated
    there by `leaf`. A determinized world has no hidden information and no chance left (the pile
    order is fixed), so this is an ordinary alpha-beta search; with two players a skip or reverse
    means the same player moves again, so the maximising side is read off `s.player` rather than
    from the parity of the depth."""
    if s.terminal:
        u = returns(s)[0] if parent is None else terminal_return(parent, action, s, exhaustion)
        return u if seat == 0 else -u
    if depth <= 0:
        v = leaf(s)
        return v if seat == 0 else -v
    maximising = s.player == seat
    best = -math.inf if maximising else math.inf
    for a in legal_actions(s):
        v = search_value(step(s, a, check=False), seat, depth - 1, leaf, exhaustion, s, a, alpha, beta)
        if maximising:
            best = max(best, v)
            alpha = max(alpha, best)
        else:
            best = min(best, v)
            beta = min(beta, best)
        if beta <= alpha:
            break
    return best


def response_value(
    s: State,
    seat: int,
    depth: int,
    leaf: Leaf,
    opponent,
    exhaustion: str = "hand",
    parent: State | None = None,
    action: int = -1,
) -> float:
    """The `"response"` continuation (step 9.5e): the world is solved as a perfect-information
    best response to the opponent's *actual fixed strategy*, instead of two-sided minimax.

    `search_value` assumes the opponent also sees everything and plays the minimising move. That
    is classic PIMC, and `uno.compiled.pimc` measured it exactly on the toys as the **worse** of
    the two rules by up to 8.8 score points, worst on free play -- which full UNO is. Here the
    opponent's nodes take the expectation under the policy it really follows, which is the
    sampled analogue of `uno.compiled.solvers.best_response_perfect` (max at the responder's
    nodes, the opponent's own weights at its nodes). Legitimate wherever the column's policy is
    known, which is every ladder matchup.

    **The opponent must be memoryless**, or this is only an approximation of its real strategy.
    The policy is evaluated on the opponent's L1 observation *inside the determinized world*,
    whose `initial_hands` / `own_draws` for that seat are fabricated (see the module docstring),
    so a policy reading its own history would be handed a history that never happened. The
    baseline rulebots read only the current hand and board, so for them this is exact; a
    memoryful column (a Nash strategy, an RL checkpoint with history features) needs the real
    information state threaded through instead, which this does not do.

    **No alpha-beta.** Pruning bounds do not propagate through an expectation node, so this
    walks the whole depth-limited tree. At the depth 2 that step 9.5d found best that is cheap;
    it is why the pruning path above is kept for the minimax continuation rather than unified.
    """
    if s.terminal:
        u = returns(s)[0] if parent is None else terminal_return(parent, action, s, exhaustion)
        return u if seat == 0 else -u
    if depth <= 0:
        v = leaf(s)
        return v if seat == 0 else -v
    legal = legal_actions(s)
    if s.player == seat:
        return max(response_value(step(s, a, check=False), seat, depth - 1, leaf, opponent, exhaustion, s, a) for a in legal)
    probs = opponent.policy(observe(s, s.player, Level.L1), legal)
    return sum(p * response_value(step(s, a, check=False), seat, depth - 1, leaf, opponent, exhaustion, s, a) for a, p in probs.items() if p > 0)


def search_q(s: State, seat: int, depth: int, leaf: Leaf, exhaustion: str = "hand", opponent=None) -> dict[int, float]:
    """The searched value of every legal action at `s`, to `seat`. No pruning across the root:
    every action's value is wanted, not just the best (step 9.5d part 4 reads them all).

    With `opponent`, each world is solved as a best response to it (`response_value`) instead of
    two-sided (step 9.5e)."""
    if opponent is not None:
        return {a: response_value(step(s, a, check=False), seat, depth - 1, leaf, opponent, exhaustion, s, a) for a in legal_actions(s)}
    return {a: search_value(step(s, a, check=False), seat, depth - 1, leaf, exhaustion, s, a) for a in legal_actions(s)}


# -- the same search, with every leaf of every world in one batch ---------------------------------
#
# `search_q` above calls `leaf` once per leaf state. When the leaf is `uno.eval.value_net.NetLeaf`
# that is a batch-of-1 forward through a small network, and torch's per-call dispatch dominates its
# arithmetic: 78.5 us per row at batch 1 against 7.0 us at batch 32-256 on `FULL_UNO`. A PIMC
# decision has a whole *sample* of worlds to search, every one of them independent, so all of their
# leaves can go through the network together -- measured 2.9-3.0x end to end on the step 9.5f
# belief ladder, at identical action sequences.
#
# Batching and alpha-beta are incompatible: pruning needs the values it is about to skip. Under the
# `response` continuation that costs nothing, since `response_value` already walks unpruned --
# bounds do not propagate through an expectation node. Under `minimax` it is a real trade, and it
# was measured rather than assumed (w = 32 on `FULL_UNO`, against Random): batching wins 3.75x at
# depth 2 and 1.91x at depth 4, and ties at depth 3, so there is no depth at which pruning pays for
# the batch-of-1 forwards it costs. Both continuations therefore take this path, and `search_value`
# stays as the pruning reference the tests check the values against.

Plan = tuple  # ("v", x) a known value | ("l", i) leaf row i | ("max" | "min", kids) | ("exp", [(p, kid)])


def _plan(s: State, seat: int, depth: int, rows: list[State], opponent, exhaustion: str, parent: State | None = None, action: int = -1) -> Plan:
    """`response_value`'s walk with the leaf calls deferred: every leaf state is appended to `rows`
    and stands in the plan as its index, so the caller can value them all at once.

    Mirrors `response_value` node for node -- same terminal handling, same `probs.items()` order at
    an opponent node, the same actions skipped at probability zero -- so that `_fold` reproduces its
    arithmetic and not merely its value."""
    if s.terminal:
        u = returns(s)[0] if parent is None else terminal_return(parent, action, s, exhaustion)
        return ("v", u if seat == 0 else -u)
    if depth <= 0:
        rows.append(s)
        return ("l", len(rows) - 1)
    legal = legal_actions(s)
    if s.player == seat:
        return ("max", [_plan(step(s, a, check=False), seat, depth - 1, rows, opponent, exhaustion, s, a) for a in legal])
    if opponent is None:
        return ("min", [_plan(step(s, a, check=False), seat, depth - 1, rows, opponent, exhaustion, s, a) for a in legal])
    probs = opponent.policy(observe(s, s.player, Level.L1), legal)
    return ("exp", [(p, _plan(step(s, a, check=False), seat, depth - 1, rows, opponent, exhaustion, s, a)) for a, p in probs.items() if p > 0])


def _fold(plan: Plan, values: list[float]) -> float:
    """The plan's value once `values[i]` holds the (sign-adjusted) leaf value of row `i`."""
    tag = plan[0]
    if tag == "v":
        return plan[1]
    if tag == "l":
        return values[plan[1]]
    if tag == "max":
        return max(_fold(k, values) for k in plan[1])
    if tag == "min":
        return min(_fold(k, values) for k in plan[1])
    return sum(p * _fold(k, values) for p, k in plan[1])


def batch_leaf(leaf: Leaf):
    """`leaf` over many states at once: its own `batch` method when it has one (`NetLeaf`), else
    one call per state, so any plain callable still works as a leaf."""
    fn = getattr(leaf, "batch", None)
    return fn if fn is not None else (lambda states: [leaf(s) for s in states])


def batched_search_q(states, seat: int, depth: int, leaf: Leaf, exhaustion: str = "hand", opponent=None) -> list[dict[int, float]]:
    """`[search_q(s, ...) for s in states]`, with every leaf of every state evaluated in one call.

    Equal to `search_q` to the last bit, not merely to a tolerance: `_plan` visits the same nodes
    in the same order and `_fold` combines them with the same operations, so the only thing that
    changes is when the leaf is consulted. `tests/eval/test_search.py` pins that on all four
    rulesets, for both continuations -- including against `search_q`'s alpha-beta, which this drops
    (see the note above for why that is still a win at every depth measured).
    """
    rows: list[State] = []
    plans = [{a: _plan(step(s, a, check=False), seat, depth - 1, rows, opponent, exhaustion, s, a) for a in legal_actions(s)} for s in states]
    raw = batch_leaf(leaf)(rows) if rows else ()
    values = [float(v) for v in raw] if seat == 0 else [-float(v) for v in raw]
    return [{a: _fold(plan, values) for a, plan in per_state.items()} for per_state in plans]


# -- the agents ----------------------------------------------------------------------------------


class PimcSearch:
    """PIMC proper: sample `worlds` determinizations of the L1 observation, solve each with a
    depth-limited perfect-information search evaluated by `leaf`, and play the action with the
    best average value. This is the agent whose bias the exact toys measure (`uno.compiled.pimc`):
    it assumes it will see everything from the next decision on, which is strategy fusion.

    Deterministic given the information state (see the module docstring), so the reported policy
    is honest and AIVAT stays unbiased."""

    def __init__(
        self,
        rs: Ruleset,
        leaf: Leaves,
        worlds: int = 8,
        depth: int = 2,
        name: str = "pimc",
        seed: int = 0,
        exhaustion: str = "hand",
        opponent=None,
        belief=None,
    ) -> None:
        self.rs, self.leaf, self.worlds, self.depth = rs, leaf, worlds, depth
        self.name, self.seed, self.exhaustion, self.opponent = name, seed, exhaustion, opponent
        self.belief = belief  # None = the uniform `"prior"` split (step 9.5f)

    @property
    def continuation(self) -> str:
        """`"response"` when each world is solved against the known opponent, `"minimax"` when
        two-sided: the axis `uno.compiled.pimc` measures exactly on the toys."""
        return "minimax" if self.opponent is None else "response"

    def _rng(self, obs: Observation) -> random.Random:
        return random.Random(stable_hash((self.seed, self.name, obs.info_state)))

    def _q(self, obs: Observation, rng: random.Random) -> dict[int, float]:
        """One world's searched action values. `_worlds_q` is what the agent itself uses; this is
        kept as the single-world spelling of it."""
        return self._worlds_q(obs, rng, 1)[0]

    def _worlds_q(self, obs: Observation, rng: random.Random, n: int) -> list[dict[int, float]]:
        """`n` sampled worlds' searched action values, every leaf of every world in one batch.

        Drawing the worlds first and searching them afterwards consumes `rng` in exactly the
        order the interleaved version did -- `determinize_l1` is the only consumer, and the
        search takes none -- so the worlds are the same worlds, which is what lets the batched
        agent replay the unbatched one's games rather than merely resample them.
        """
        worlds = [determinize_l1(self.rs, obs, rng, self.belief) for _ in range(n)]
        return batched_search_q(worlds, obs.seat, self.depth, leaf_for(self.leaf, obs.seat), self.exhaustion, self.opponent)

    def _scores(self, obs: Observation, legal: tuple[int, ...]) -> dict[int, float]:
        total = dict.fromkeys(legal, 0.0)
        for q in self._worlds_q(obs, self._rng(obs), self.worlds):
            for a in legal:
                total[a] += q[a]
        return {a: v / self.worlds for a, v in total.items()}

    def policy(self, obs: Observation, legal: tuple[int, ...]) -> dict[int, float]:
        if len(legal) == 1:
            return {legal[0]: 1.0}
        scores = self._scores(obs, legal)
        return {max(legal, key=lambda a: (scores[a], -a)): 1.0}  # ties: the lowest action id

    def scores_split(self, obs: Observation, legal: tuple[int, ...]) -> tuple[dict[int, float], dict[int, float]]:
        """The world sample cut into two independent halves: one to *choose* an action with, the
        other to *value* it with.

        Choosing and valuing on the same sample is the winner's curse: `max_a Qhat(a)` overshoots
        `max_a Q(a)` by the expected maximum of the estimation noise, at every decision. That bias
        is positive and does not cancel, so summing it over a game's ~22 decisions accumulates into
        tens of score points -- enough, when this was first written without the split, to report a
        rulebot as 34-40 points worse per game than a search that in fact scores no better than it.
        With the halves separated the valuation is independent of the selection, the per-decision
        term may legitimately come out negative, and the estimator has no systematic inflation.
        """
        if self.worlds < 2:
            raise ValueError("scores_split needs at least 2 worlds")
        k = self.worlds // 2
        halves: tuple[dict[int, float], dict[int, float]] = (dict.fromkeys(legal, 0.0), dict.fromkeys(legal, 0.0))
        for i, q in enumerate(self._worlds_q(obs, self._rng(obs), 2 * k)):
            half = halves[i >= k]
            for a in legal:
                half[a] += q[a] / k
        return halves


class PimcRollout:
    """The cheap comparator: same determinizations, but each world is valued by playing it out
    with `agents` instead of searching it. The rollout players act on their own L1 observations
    inside the world, so there is **no strategy fusion** -- this is a one-step policy improvement
    over `agents` under the belief, a different object from `PimcSearch`, and the toys' PIMC bias
    numbers do not describe it."""

    def __init__(self, rs: Ruleset, agents, worlds: int = 8, rollouts: int = 1, name: str = "pimc_rollout", seed: int = 0, exhaustion: str = "hand") -> None:
        self.rs, self.agents, self.worlds, self.rollouts = rs, agents, worlds, rollouts
        self.name, self.seed, self.exhaustion = name, seed, exhaustion

    def policy(self, obs: Observation, legal: tuple[int, ...]) -> dict[int, float]:
        if len(legal) == 1:
            return {legal[0]: 1.0}
        rng = random.Random(stable_hash((self.seed, self.name, obs.info_state)))
        sign = 1.0 if obs.seat == 0 else -1.0
        total = dict.fromkeys(legal, 0.0)
        for _ in range(self.worlds):
            s = determinize_l1(self.rs, obs, rng)
            for a in legal:
                child = step(s, a, check=False)
                if child.terminal:
                    total[a] += self.rollouts * sign * terminal_return(s, a, child, self.exhaustion)
                else:
                    total[a] += sum(sign * rollout(child, self.agents, rng, self.exhaustion) for _ in range(self.rollouts))
        return {max(legal, key=lambda a: (total[a], -a)): 1.0}
