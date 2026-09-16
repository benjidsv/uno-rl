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

The opponent's `initial_hands` / `own_draws` are fabricated to match the hand we just dealt them:
nothing downstream reads them (`uno.eval.value_net.state_features` and the baseline agents work
off `hands`, `discard`, `top`, `current_color`, `phase`, `drawn` and the pile length), but they
must not be *inconsistent with the hand*, or a later `observe` of that seat would disagree with
itself. They are not a reconstruction of what really happened, and no L1/L2 key of the opponent's
seat should be taken from a determinized state.

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
from uno.game.observe import Observation, discard_from_public
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


def determinize_l1(rs: Ruleset, obs: Observation, rng: random.Random) -> State:
    """A state consistent with `obs`: the unseen cards split uniformly into the opponent's hand
    and the pile. See the module docstring on the fabricated opponent memory."""
    unseen, tail = unseen_cards(rs, obs)
    rng.shuffle(unseen)
    me, opp = obs.seat, obs.seat ^ 1
    opp_hand = tuple(sorted(unseen[: obs.opp_hand_size]))
    pile = (*unseen[obs.opp_hand_size :], *tail)
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


def search_q(s: State, seat: int, depth: int, leaf: Leaf, exhaustion: str = "hand") -> dict[int, float]:
    """The searched value of every legal action at `s`, to `seat`. No pruning across the root:
    every action's value is wanted, not just the best (step 9.5d part 4 reads them all)."""
    return {a: search_value(step(s, a, check=False), seat, depth - 1, leaf, exhaustion, s, a) for a in legal_actions(s)}


# -- the agents ----------------------------------------------------------------------------------


class PimcSearch:
    """PIMC proper: sample `worlds` determinizations of the L1 observation, solve each with a
    depth-limited perfect-information search evaluated by `leaf`, and play the action with the
    best average value. This is the agent whose bias the exact toys measure (`uno.compiled.pimc`):
    it assumes it will see everything from the next decision on, which is strategy fusion.

    Deterministic given the information state (see the module docstring), so the reported policy
    is honest and AIVAT stays unbiased."""

    def __init__(self, rs: Ruleset, leaf: Leaf, worlds: int = 8, depth: int = 2, name: str = "pimc", seed: int = 0, exhaustion: str = "hand") -> None:
        self.rs, self.leaf, self.worlds, self.depth = rs, leaf, worlds, depth
        self.name, self.seed, self.exhaustion = name, seed, exhaustion

    def _rng(self, obs: Observation) -> random.Random:
        return random.Random(stable_hash((self.seed, self.name, obs.info_state)))

    def _scores(self, obs: Observation, legal: tuple[int, ...]) -> dict[int, float]:
        rng = self._rng(obs)
        total = dict.fromkeys(legal, 0.0)
        for _ in range(self.worlds):
            q = search_q(determinize_l1(self.rs, obs, rng), obs.seat, self.depth, self.leaf, self.exhaustion)
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
        rng = self._rng(obs)
        halves: tuple[dict[int, float], dict[int, float]] = (dict.fromkeys(legal, 0.0), dict.fromkeys(legal, 0.0))
        for i in range(2 * k):
            q = search_q(determinize_l1(self.rs, obs, rng), obs.seat, self.depth, self.leaf, self.exhaustion)
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
