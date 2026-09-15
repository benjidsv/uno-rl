# Step 9: review before the sampled methods (2026-09-15)

> On top of commit `fca5589`. Numbers marked *scratch* come from a script run outside the repo on the cached `TOY` / `TOY_FULL` trees (utility swap, max / min backward passes, CFR variants); step 9 turns the ones we keep into experiments with a rerun command.

A second review after step 8.9, on methodology: interpretation, statistics
and missing tests, not code. Each item carries its status.

**Framing agreed in this review:** the toys exist to validate the sampled
methods planned for full UNO (AIVAT, PIMC, MCTS) against exact values. How
well toy conclusions transfer to full UNO is secondary, and the toys are not to
be tuned for realism.

**Build order after this review:** 9 = these follow-ups, 9.5 = AIVAT and the
sampled methods (PIMC / MCTS) validated on the toys, 10 = formalise the toy
results (figures and tables that RL relies on), 11 = RL.

## A. Readings that were wrong or overstated

### A1. The split share is not the share of games decided by the deal (docs corrected)

Step 7 and `context.md` read the split share (pairs scoring 1/2) as "3 games in
5 on `TOY` are decided before anyone moves". Seat asymmetry alone makes pairs
split. *Scratch*, exact on `TOY` (step 8.6 Heuristic), with ties:

| matchup | split, exact | if the two games were independent | excess |
|---|---|---|---|
| heuristic vs random | 0.596 | 0.468 | +0.128 |
| BR L1(random) vs random | 0.546 | 0.387 | +0.159 |
| BR L1(heuristic) vs heuristic | 0.778 | 0.498 | +0.281 |
| BR L1(random) vs heuristic | 0.793 | 0.499 | +0.294 |

Only 4.3 % of `TOY` worlds (deal x pile order) have a result that no play by
either side can change (0 % on `TOY_FULL`). Status: corrected in `context.md`
section 5.4, step 7 reading 2 and the `estimate.py` comment. The correlation is
the luck readout, or the split share next to its independence baseline.

### A2. The free-play results depend on the exhaustion rule (open)

*Scratch*: same compiled tree, exhaustion scored as a draw instead of on hand
size, everything re-solved.

| | R vs R seat gap | BR L1 edge over Random (seat 0 / 1) | Nash seat gap |
|---|---|---|---|
| `TOY`, standard | 29.9 | +14.4 / +20.1 | 23.9 |
| `TOY`, exhaustion = draw | 27.2 | +15.5 / +21.5 | 22.5 |
| `TOY_FULL`, standard | 29.9 | +15.2 / +23.8 | 17.3 |
| `TOY_FULL`, exhaustion = draw | 27.2 | +19.1 / +32.9 | 28.6 |

Forced-play readings move 1-2 points. On `TOY_FULL` the voluntary draw's effect
on the Nash seat gap changes sign (-6.6 points under the rule, +6.1 under
exhaustion = draw). Exhaustion = draw is not "the right rule" (it gives the
trailing player a claim to a tie, also toy-only). But every voluntary draw on a
2-3-card pile changes pile parity, so "fetch a card" and "manipulate who meets
the empty pile" are not separable, and step 8.7 reading 2 / `context.md` 6.5
("a real alternative to a bad play") state an untested mechanism. Step 8.8's
claim test removed only the empty-pile claim.

Proposal: keep the rule; scope the voluntary-draw and seat-gap readings to it;
use a second terminal utility as a free robustness axis when validating the
sampled methods (same tree, utility swap).

### A3. "Draw with a play in hand = 60 % of full-UNO decisions" counts availability (to fix)

`transfer.py` counts a decision under that kind whenever a card plays and
`Draw` is legal, i.e. almost every full-UNO decision with a play. How often the
draw is right there is unknown. Untested hypothesis for the earlier PPO runs: a
near-uniform initial policy draws voluntarily at a large share of decisions.

### A4. The game budgets have 50 % power (docs corrected)

`games_needed(edge, var, z=3)` is the n at which the edge equals 3 standard
errors, where a 3-sigma test detects a real edge half the time. 80 % power
needs (3 + 0.84)^2 / 9 = 1.64x that, 95 % needs 2.4x. The intro's ~2,400
becomes ~3,900; full UNO's ~9,000 ~15,000. Status: noted in `context.md`,
`plan.md`, steps 7 and 8.5; the code is unchanged.

### A5. "Pairing does not help on full UNO" was measured on dissimilar policies (to measure)

Step 8.5 measured Heuristic against Random, which diverge on the first move.
Checkpoint against checkpoint is untested on full UNO. A draw coupling that
survives diverging draw counts is also untested (per-draw tickets: every card
gets an independent random ticket for "seat s's k-th draw", the lowest ticket
in the pile is drawn; same distribution as the top of a shuffled pile, and it
also gives full UNO a reshuffle chance model for AIVAT).

### A6. Two-sided L2 / L3 games have values (to add)

"Optimal at L2 / L3 only exists as a best response" holds for one-sided
information. When both players see both hands (L2) or everything (L3) the game
has a minimax value, one max / min backward pass. *Scratch* seat-0 scores:

| | Nash L1 | L2 both see (expectiminimax) | L3 both see (minimax) |
|---|---|---|---|
| `TOY` | 0.6196 | 0.6075 | 0.6099 |
| `TOY_FULL` | 0.5866 | 0.5885 | 0.6337 |

On `TOY` hidden information favours seat 0.

## B. Assumed, not verified

- **B1. Choice of equilibrium** (to do on `TOY2` / `TOY2_FULL`). Nash-dependent
  cells for three CFR variants, *scratch*: on `TOY` (1000 its) everything within
  0.1 point. On `TOY_FULL` (400 its) the Nash row moves up to 0.4 point (nash vs
  random seat 1 0.5123 / 0.5167 / 0.5129 for CFR+ / DCFR / PCFR+), BR L2 vs Nash
  0.1 point, and the NashConv of BR L1(random) filled with that Nash is 5.50 /
  6.13 / 5.59 points. `TOY2_FULL` (NashConv 0.005, 150 its) unchecked.
- **B2. Fill-in** (to do). Minimal-exploitability fill-in = CFR on the unreached
  sets with the reached sets frozen, which gives a lower bound instead of one
  arbitrary fill; also count exact ties at reached sets (lowest action id).
- **B3. Heuristic threshold** (no action). `low_threshold=2` only matters for
  action cards, i.e. `TOY2_SKIP`. A rulebot tuned on the toy (genetic) is
  optional, not a goal.
- **B4. On-seeds checks cannot see the deal model** (to do). `scores_on_seeds`
  starts from each seed's real state, so a wrong deal probability passes every
  validation (the step 8.9 bug did). Standalone check: the mean of on-seeds exact
  values over many seeds against the global exact value, z on the deal variance.
  AIVAT against the global value will be the first sampled check sensitive to it.
- **B5. The thesis is tested on proxies** (step 9.5). Full-UNO ceiling and
  sampled methods come with 9.5.
- **B6. Stale numbers in `context.md`** (to fix): 6.8 quotes the step 4
  Heuristic's exhaustion figures (0.44, 5-6 points); 6.4 "BR to Random transfers
  within 0.8-1.4" is the step 7 Heuristic; "Nash gives up ~1-2 vs the Heuristic"
  is 0.5 / 2.2.

## C. For step 9.5 (AIVAT and sampled methods)

- Validate AIVAT with imperfect value functions, not only the exact one:
  another profile's values, L1 set values, exact + noise, a card-count
  heuristic, a critic fitted on dev seeds. The variance-reduction curve against V
  quality is what predicts the full-UNO gain.
- The AIVAT estimator's exact variance is computable on the toy (one more
  backward pass): an M4-style check instead of "ratio > 1".
- Draw corrections use the lazy chance model (uniform over the unknown pile)
  although the sampled game has a fixed permutation; say why in the spec.
- PIMC as a "BR L1" must weight sampled worlds by the opponent policy's reach,
  otherwise it does no inference. Strategy-fusion bias grows with game length,
  so the toy bias is not an upper bound for full UNO.

## D. Before RL (to discuss once A and B are settled)

The two artifacts step 10 should produce (Benji):

1. **A performance range with several ceilings**, so an RL checkpoint can be
   placed on it: Random, Heuristic, a tuned Heuristic, Nash, BR L1 against the
   training opponent, BR L2 / L3, the two-sided L2 / L3 values of A6.
2. **An edge-by-decision tool**: for a checkpoint against a reference strategy
   (exact ceiling on the toy, a sampled ceiling such as PIMC on full UNO), where
   it loses its points, by decision kind and by situation. The items below are
   the pieces of it.

Also proposed: tune the parameterised Heuristic on the toy (exact evaluation,
genetic or grid search) and check which parameters transfer to full UNO (a
toy-to-full transfer test that needs no exact full-UNO values).

- Where the skill is: the Heuristic's gap to the best response by decision kind
  (swap in the best response's action only at colour / wild hold / wild colour /
  draw / skip decisions), and the reach-weighted distribution of the best - worst
  action gap per decision (how much each decision matters; credit assignment).
- Memory: BR L1 conditions on the full history. A backward pass on a coarse,
  history-free key gives a valid policy, a lower bound on the memoryless ceiling:
  does the RL observation need history?
- Metric for agents trained against a fixed opponent: the gap to BR L1 against
  that opponent (exact, luck-free); exploitability punishes unvisited sets
  (the step 8.8 fill-in problem again), keep it for self-play.
- Shaping: "an L2 / L3 potential only adds noise" is too strong; critics with
  hidden information are standard and the history + state critic (Baisero &
  Amato, AAMAS 2022) is unbiased. To test, not to assert.
- An ablation separating exploration, variance and representation: tabular
  learner, PPO with the exact L1 critic, plain PPO, scored exactly per checkpoint
  on `TOY`.
- Mechanisms flagged as hypotheses, now cheap: pile size vs ranks (4c(1,1)+2W
  forced, pile 3, two ranks, 670k histories); "Nash hides its hand" as the
  mutual information between the opponent's hand and the public history.
