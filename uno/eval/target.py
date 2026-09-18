"""The training target: the luck-corrected return step 11 optimises (step 10a.2, item A6).
See docs/foundations/spec.md section 28.

**One declaration, no new estimator.** 9.5b built AIVAT as per-game *records* precisely so
that any subset of its corrections is one subtraction (`uno.eval.aivat.GameTerms.estimate`),
so the training target is a *subset selection* over machinery that is already validated
against the exact toys, not a second estimator to close all over again:

    TRAINING = chance + opponent + deal,  never the learner's own decisions

- **chance** (draws, reshuffles): luck the agent cannot influence. Free variance reduction.
- **opponent**: in a league its policy is ours exactly, and in self-play the opponent *is*
  us, so the correction costs no modelling.
- **deal**: this *is* D5's shaping term. With no discount and `Phi(terminal) = 0` a deal
  potential contributes exactly `-Phi(s0)` to every return, so the control variate and the
  potential are one object (`uno.eval.shaping`). It reads hidden cards but **not the
  agent's action**, which is what keeps it unbiased -- the centralised critic the step 9
  review flagged as standard (Baisero & Amato).
- **the row (the agent's own decisions): no, and this is the trap.** With an exact V the row
  term *is* `Q(s, a) - V(s)`, so subtracting it makes the corrected return's conditional
  expectation independent of the action taken and the policy gradient's expectation
  **zero**: the credit-assignment signal is exactly what the correction cancels. An
  evaluator needs an unbiased *mean*, an objective needs an unbiased *gradient*, and the
  row correction is right for the first and fatal for the second. (For the deterministic
  search rows of 9.5d-9.5e it is identically zero, so nothing in those results warns of
  this.)

`uno.compiled.gradient` measures all of that exactly on a compiled tree, and
`uv run uno target` closes the sampled gradient below against those exact values.

**The sampled gradient.** `sampled_gradient` is the ordinary score-function estimator of
the logit gradient, `(1{a taken} - sigma(a|I)) * G` summed over the learner's decisions,
with `G` the corrected return. UNO pays one reward, at the terminal, so the return-to-go of
every decision of a game is that game's return and `Decisions.signal` carries it per
decision; a ruleset with intermediate rewards would fill the field differently and nothing
else here would change.

Accumulation is vectorised over *all* games at once rather than per game: the estimator is
two `bincount`s over the flat decision table, which is also why the per-game projections
used for its standard error are a third. A training loop that accumulated one game at a
time in Python would spend more time in the accumulation than in the corrections it is
measuring.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uno.compiled.lookup import NodeIndex
from uno.compiled.tree import Keying
from uno.eval.aivat import Corrections, GameTerms
from uno.eval.play import Trace

# The declared subset (the module docstring). `row=False` is the load-bearing entry.
TRAINING = Corrections(chance=True, row=False, column=True, deal=True)

# The same subset where no deal mean exists (full UNO has no tree, so no exact
# `E_deals nu`; 9.5c estimates one by Monte Carlo, and without it the deal term is off).
TRAINING_NO_DEAL = Corrections(chance=True, row=False, column=True, deal=False)


def corrected_return(terms: GameTerms, seat: int, corrections: Corrections = TRAINING) -> float:
    """The luck-corrected return of one game **in the learner's own units**, for a learner
    at `seat`: the single function step 11 calls.

    `GameTerms.estimate` is in seat-0 units (every AIVAT quantity is), so the sign flips at
    seat 1 -- the same conversion `uno.eval.aivat.seat_scores` makes before turning a return
    into a score. Raises if `corrections` includes the row term, because that is the one
    choice this module exists to forbid; pass it explicitly through
    `GameTerms.estimate` if you are deliberately measuring the trap."""
    if corrections.row:
        raise ValueError("the training target must not correct the learner's own decisions (uno.eval.target, docs/foundations/spec.md section 28): use GameTerms.estimate directly to measure that case")
    sign = 1.0 if seat == 0 else -1.0
    return sign * terms.estimate(corrections, seat)


@dataclass(frozen=True)
class Decisions:
    """The learner's own decisions over a batch of games, as flat arrays.

    One row per decision the learner made, in play order: which game it belongs to, the
    information set and the set-action taken (both in the keying's own numbering, so they
    are already canonical on a quotient tree), and the learning signal of that decision."""

    game: np.ndarray  # (d,) int64, index into the batch of games
    info: np.ndarray  # (d,) int64, information set
    ia: np.ndarray  # (d,) int64, the set-action taken
    signal: np.ndarray  # (d,) float64, the corrected return of the decision's game
    games: int

    def __len__(self) -> int:
        return len(self.game)


def own_decisions(idx: NodeIndex, keying: Keying, trace: Trace, seat: int) -> tuple[list[int], list[int]]:
    """`(information sets, set-actions)` of the decisions `seat` made in a played game.

    Read through the same `uno.compiled.lookup.NodeIndex` AIVAT's value model uses, so a
    played state and its action land on the tree's own numbering: the index is built on the
    L2 keying (the finest), and an L1 set is a function of an L2 set, so the L1 set of the
    representative node is the L1 set of the state. The set-action comes from
    `keying.arc_ia`, never from the arc's position under its node -- the members of a set
    are canonicalised by their own group elements, which is the trap
    `uno.compiled.lookup.NodeIndex.arc` documents."""
    infos: list[int] = []
    ias: list[int] = []
    for st in trace.steps:
        if st.state.player != seat:
            continue
        node, arcs = idx.arcs(st.state, (st.action,))
        infos.append(int(keying.node_info[node]))
        ias.append(int(keying.arc_ia[arcs[st.action]]))
    return infos, ias


def sampled_gradient(k: Keying, d: Decisions, sigma: np.ndarray) -> np.ndarray:
    """(n_ia,) the score-function estimate of the logit policy gradient, per game played.

    `mean over games of sum over the game's own decisions of (1{a taken} - sigma(a|I)) * G`.
    Both terms are one `bincount`: the indicator scatters the signal onto the action taken,
    and the baseline spreads each set's accumulated signal over its actions in proportion
    to `sigma`. No Python loop over games, and none over a set's actions."""
    taken = np.bincount(d.ia, weights=d.signal, minlength=k.n_ia)
    per_set = np.bincount(d.info, weights=d.signal, minlength=k.n_info)
    return (taken - sigma * per_set[k.ia_info]) / d.games


def per_game(k: Keying, d: Decisions, sigma: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """(games,) the estimator's contribution to `<gradient, direction>` from each game.

    The mean of this is exactly `sampled_gradient(...) @ direction` (two spellings of one
    sum, pinned in `tests/eval/test_target.py`), and its standard deviation over games is
    what gives that projection a standard error -- which is how a gradient's *bias* gets a
    z against the exact value without testing thousands of components at once."""
    weight = direction[d.ia] - k.info_sum(sigma * direction)[d.info]
    return np.bincount(d.game, weights=weight * d.signal, minlength=d.games)
