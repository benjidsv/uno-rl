"""The opponent-hand belief for the PIMC agent (step 9.5f, docs/spec.md section 26).

`uno.eval.search.determinize_l1` splits the unseen cards **uniformly**, which is the `"prior"`
weighting `uno.compiled.pimc` measures exactly on the toys: PIMC with no inference at all. This
module narrows that split towards the `"posterior"`, using what the opponent's own actions said.

**What was tried first, and why it does not work.** docs/plan.md's original design is a particle
filter over the deal: sample the opponent's initial hand, replay the public history, weight each
particle by the opponent's action probabilities. Measured before this module was written, that
collapses outright -- a particle survives the opponent's first play with probability
`hand_size / unseen` ~ 7/98, and the survival of 64 particles goes 0.156 -> 0.018 -> 0.002 ->
0.0004 -> 0 over their first five plays. No particle count fixes a geometric decay, and a filter
that resets on collapse is worse than the prior it replaced (Brier 0.167 against 0.085, with the
bin predicting 1.000 observed at 0.112). **The lesson is to condition, never to weight:** a world
must be *built* consistent with what was observed, not sampled and then tested against it.

So both rungs here sample the opponent's hand at the **current** split -- where consistency with
the hand sizes is automatic -- and use the history only to constrain or to score that sample.

**Rung `"voids"`: what a voluntary draw proves.** `uno.game.engine.voluntary_draw` is a *choice*,
and no play-first policy takes it holding a playable card. So at each voluntary draw by the
opponent, against the board `(top, colour)` then in force, their hand held nothing playable. If
they have drawn `k` cards since, at most `k` cards of their hand *now* can be playable against
that board -- a necessary condition, exact, and checkable against a current hand with no
propagation at all.

On a **forced-play** ruleset the statement needs no policy assumption whatever: `legal_actions`
offers `Draw` only when nothing is playable, so a draw *proves* the void by the rules rather than
merely implying it. The rung therefore has teeth on every toy, not only the free-play ones --
decisions carrying at least one constraint: `TOY` 16.5 %, `TOY2` 23.3 %, `TOY2_SKIP` 18.3 %,
`TOY_FULL` 16.5 %, `FULL_UNO` 73.4 %. (`TOY` and `TOY_FULL` agree to the digit because neither
rulebot ever draws voluntarily, so the two rulesets play out identically under this profile.)

It needs **no opponent policy**, so unlike the `response` continuation it does
not make the row opponent-specific; it is the half that survives into a general player. Measured:
it alone rules out **79.9 %** of `determinize_l1`'s worlds on full UNO (median surviving share
15.6 %, so w = 64 is effectively w = 10), with the true hand violating it 0 times in 2,487.

**Rung `"window"`: the policy likelihood, exactly, where it is computable.** Over the opponent's
**trailing run of plays with no draw in between**, their hand at each of those decisions is
recoverable from the hand they hold now: it is that hand plus the cards they have played since.
So a sampled hand can be rolled backwards through that window and scored by the opponent's real
action probabilities -- the genuine posterior restricted to the window, not an approximation of
it. The window stops at the first draw going back, because a draw is exactly the point where the
hand stops being recoverable, which is what defeated the filter above.

Both rungs fall back to the uniform prior when no consistent hand is found, and both **count** it
(`Belief.fallbacks`): a report that hides that number is hiding whether the belief was available.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass

from uno.agents.baseline import Agent
from uno.game.engine import Phase, is_playable
from uno.game.observe import Level, Observation
from uno.game.rules import Kind, Play, Ruleset

BELIEFS = ("prior", "voids", "window")


@dataclass(slots=True)
class Void:
    """At a voluntary draw the opponent held nothing playable against `(top, colour)`; `since` is
    how many cards they have drawn since, so at most `since` cards of their hand now may be
    playable against it."""

    top: int
    colour: int
    since: int


def _is_penalty(rs: Ruleset, public: tuple, i: int) -> bool:
    """A `("draw", p, n)` forced by the *other* player's +2 / +4 rather than chosen."""
    prev = public[i - 1] if i else None
    if not (prev and prev[0] == "play" and prev[1] != public[i][1]):
        return False
    return rs.card_types[prev[2]].kind in (Kind.DRAW_TWO, Kind.WILD_DRAW_FOUR)


def voids(rs: Ruleset, public: tuple, opp: int) -> list[Void]:
    """Every constraint the opponent's voluntary draws put on the hand they hold now.

    A voluntary draw is a decision no play-first policy takes with a playable card in hand, so it
    proves a void against the board then in force. Cards drawn since could be playable, hence
    `Void.since`; cards played since only leave the hand, so they cannot loosen the count. A
    reshuffle does not affect the statement (it changes the pile, not the hand)."""
    types = rs.card_types
    top, colour = None, None
    out: list[Void] = []
    for i, ev in enumerate(public):
        match ev[0]:
            case "setup":
                top, colour = ev[1], types[ev[1]].color
            case "play":
                top, colour = ev[2], ev[3]
            case "draw":
                if ev[1] != opp:
                    continue
                for v in out:
                    v.since += ev[2]
                if ev[2] == 1 and not _is_penalty(rs, public, i) and top is not None:
                    out.append(Void(top=top, colour=colour, since=0))
    return out


def satisfies(rs: Ruleset, hand, vs: list[Void]) -> bool:
    for v in vs:
        if sum(1 for c in hand if is_playable(rs, c, v.top, v.colour)) > v.since:
            return False
    return True


def trailing_window(rs: Ruleset, public: tuple, opp: int) -> list[tuple[int, int, int, int]]:
    """The opponent's trailing run of plays with no draw of theirs in between, oldest first, as
    `(action's card, chosen colour, board top, board colour)`.

    Their hand at each of these decisions is the hand they hold now plus the cards played since,
    which is what makes the likelihood exactly computable there. Walking back stops at their first
    draw: that is where the hand stops being recoverable."""
    types = rs.card_types
    boards, top, colour = [], None, None
    for ev in public:
        if ev[0] == "setup":
            top, colour = ev[1], types[ev[1]].color
        boards.append((top, colour))
        if ev[0] == "play":
            top, colour = ev[2], ev[3]
    out = []
    for i in range(len(public) - 1, -1, -1):
        ev = public[i]
        if ev[0] == "draw" and ev[1] == opp:
            break
        if ev[0] == "reshuffle":
            continue
        if ev[0] == "play" and ev[1] == opp:
            bt, bc = boards[i]
            if bt is None:
                break
            out.append((ev[2], ev[3], bt, bc))
    out.reverse()
    return out


class Belief:
    """Samples the opponent's hand from the current split, narrowed by `voids` and (for
    `"window"`) scored by the opponent's real policy over the trailing no-draw run.

    Stateless across decisions by construction -- there is nothing to carry, because nothing is
    propagated -- so the policy stays a deterministic function of the information state, which
    AIVAT's row correction needs (docs/spec.md section 25), with no cache to invalidate."""

    def __init__(self, rs: Ruleset, kind: str = "window", opponent: Agent | None = None, tries: int = 32) -> None:
        if kind not in BELIEFS:
            raise ValueError(f"unknown belief {kind!r}: one of {BELIEFS}")
        if kind == "window" and opponent is None:
            raise ValueError("the 'window' belief needs the opponent's policy; 'voids' does not")
        self.rs, self.kind, self.opponent, self.tries = rs, kind, opponent, tries
        self.fallbacks = 0  # decisions where no consistent hand was found and the prior was used
        self.draws = 0

    def hand(self, obs: Observation, unseen: list[int], rng: random.Random) -> tuple[int, ...]:
        """One opponent hand of `obs.opp_hand_size` cards from `unseen`.

        `"voids"` rejects hands the history rules out; `"window"` additionally accepts in
        proportion to how likely the opponent's trailing plays were from that hand. Both fall back
        to a uniform draw after `tries` attempts, so the caller always gets a usable world."""
        self.draws += 1
        k = obs.opp_hand_size
        pool = list(unseen)
        if self.kind == "prior" or k == 0:
            return tuple(sorted(rng.sample(pool, k)))

        opp = obs.seat ^ 1
        vs = voids(self.rs, obs.public, opp)
        window = trailing_window(self.rs, obs.public, opp) if self.kind == "window" else []
        cands: list[tuple[int, ...]] = []
        weights: list[float] = []
        for _ in range(self.tries):
            # `sample` is O(k); shuffling the whole unseen pool to take 7 of ~98 cards cost
            # 105 ms/game against the search's own 128.7, i.e. most of the belief's price
            h = tuple(sorted(rng.sample(pool, k)))
            if vs and not satisfies(self.rs, h, vs):
                continue
            if not window:
                return h  # the hard constraints are all there is, so the first survivor is already a uniform draw from them
            w = self._window_likelihood(obs, h, window)
            if w > 0.0:
                cands.append(h)
                weights.append(w)
        if cands:
            # Sampling-importance-resampling: the proposal is uniform over void-consistent hands,
            # so drawing one proportionally to its likelihood targets the windowed posterior.
            # Returning the argmax instead would collapse every world onto one explanation, which
            # is the same mistake as reading a reference value off the worlds that chose the action
            # (the winner's curse of docs/spec.md section 25), one level over.
            total = sum(weights)
            r = rng.random() * total
            acc = 0.0
            for h, w in zip(cands, weights):
                acc += w
                if r < acc:
                    return h
            return cands[-1]
        self.fallbacks += 1
        return tuple(sorted(rng.sample(pool, k)))

    def _window_likelihood(self, obs: Observation, hand: tuple[int, ...], window) -> float:
        """P(the opponent played exactly that trailing run) if they hold `hand` now.

        Rolled backwards: at the oldest decision of the window their hand was `hand` plus every
        card played since, and each play removes one card as we walk forward again."""
        rs = self.rs
        h = Counter(hand)
        for card, _colour, _t, _c in window:
            h[card] += 1
        p = 1.0
        mine = obs.opp_hand_size
        for card, colour, top, colour_now in window:
            legal = self._legal(tuple(sorted(h.elements())), top, colour_now)
            if not legal:
                return 0.0
            o = Observation(
                level=Level.L1,
                seat=obs.seat ^ 1,
                player=obs.seat ^ 1,
                phase=Phase.TURN_START,
                current_color=colour_now,
                top=top,
                opp_hand_size=len(obs.hand),
                pile_size=obs.pile_size,
                public=obs.public,
                turn=obs.turn,
                end=None,
                winner=None,
                initial_hand=tuple(sorted(h.elements())),
                hand=tuple(sorted(h.elements())),
                own_draws=(),
                drawn=None,
            )
            probs = self.opponent.policy(o, legal)
            p *= float(probs.get(self._action_of(card, colour), 0.0))
            if p <= 0.0:
                return 0.0
            h[card] -= 1
            if not h[card]:
                del h[card]
        return p

    def _action_of(self, card: int, colour: int) -> int:
        for a in self.rs.play_actions[card]:
            act = self.rs.actions[a]
            if not isinstance(act, Play) or not self.rs.card_types[card].is_wild or act.color == colour:
                return a
        return self.rs.play_actions[card][0]

    def _legal(self, hand, top: int, colour: int) -> tuple[int, ...]:
        rs = self.rs
        acts: list[int] = []
        for t in sorted(set(hand)):
            if is_playable(rs, t, top, colour):
                acts.extend(rs.play_actions[t])
        if not rs.forced_play or not acts:
            acts.append(rs.draw_action)
        return tuple(acts)
