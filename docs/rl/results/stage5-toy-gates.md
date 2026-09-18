# Stage 5: exact-toy gates for the PPO learner

**Status:** results pending the run recorded in this file's "How it was run" section.
Placeholders marked `TBD` are filled from `logs/rl_toy/`.

## Question

Does the learner implemented for [pipeline-plan.md](../pipeline-plan.md) stages 1-4
compute the gradient it claims to compute, and does its sampled optimizer reach the
exact-toy targets it is supposed to reach?

Three separate claims, deliberately not merged:

1. **Gradient correctness.** Do the learner's raw sampled gradients agree with the
   exact expected gradient of the *same parameters*, within game-level uncertainty?
2. **Optimizer capability (the retained tabular gate).** With the representation
   question removed -- one logit per action orbit -- do five training seeds reach
   within 0.5 score points of the same-seat fixed-opponent best response at both
   seats, within a declared cap?
3. **The architecture's achieved level.** How does sampled optimization of the
   feed-forward arm-B network compare with exact-gradient optimization of the same
   architecture, and what is the BR gap of each?

## Predeclared decision rules

Frozen before the five-seed runs, after the bounded diagnostic in
"Calibration" below:

| Rule | Value | Where |
|---|---|---|
| Gradient agreement | worst \|z\| <= 3.0 on projections with a game-level SE | `GRADIENT_Z` |
| PPO/score-function identity | relative error < 1e-5 at the collection policy | `gradient_check` |
| Tabular gate tolerance | 0.5 score points below the same-seat BR | `TABULAR_TOLERANCE` |
| Tabular gate cap | 400,000 learner decisions per seed per seat | `TABULAR_CAP` |
| Neural cap | 100,000 learner decisions per seed per seat | `NEURAL_CAP` |
| Exact comparator | 250 Adam steps at lr 3e-3 on the exact gradient | `EXACT_STEPS`, `EXACT_LR` |

The neural arm has **no** pass/fail threshold: the plan asks for a comparison and
a reported BR gap, not a claim that the architecture is sufficient.

## Configuration and seeds

- Ruleset `TOY`; opponents `PlayFirstRandom` ("random") and `Heuristic`, one fixed
  opponent per run, each seat measured separately.
- Arm B, width 64, 2 relational blocks, FP32, temperature 1. The critic is the
  separate structured network on the declared critic contract.
- PPO as proposed in pipeline plan section 6, except the tabular arm's actor
  learning rate (0.05, a tabular softmax needs a larger step than a network) and
  `critic_epochs = 2`.
- Training deals from `TRAIN_DEALS` = `[0, 800000)` through the seeded permutation
  of `uno.rl.manifest.DealSchedule`; training seeds 0-4.
- Exact values come from the cached `TOY` L1 quotient tree; they carry **no**
  trajectory-sampling noise. Sampled scores carry a stated 95% CI.

## Calibration (bounded diagnostic, seed 0, before the gate)

| Cell | 100,000 decisions | 400,000 decisions |
|---|---|---|
| random, seat 0 | gap +0.62 pts | -- |
| random, seat 1 | gap +1.02 pts | -- |
| heuristic, seat 0 | gap +0.15 pts | -- |
| heuristic, seat 1 | gap +1.72 pts | gap +0.10 pts |

The hardest cell (heuristic, seat 1) plateaued at about 0.4344-0.4351 from roughly
150,000 decisions on. The cap was frozen at 400,000 for margin, and the tolerance
kept at the plan's 0.5 points.

## Results

### 1. Gradient agreement

TBD

### 2. Tabular five-seed gate

TBD

### 3. Neural arm: sampled against exact-gradient optimization

TBD

## Interpretation

TBD

## Incidental finding in existing code (not changed)

`uno.eval.belief._is_penalty` decides whether a public draw event was a +2/+4
penalty by looking only at `public[i - 1]`. The engine appends `("reshuffle",)`
*between* the play and the penalty draw when the pile runs out mid-penalty
(`uno.game.engine._Work.draw`), so in that case the penalty is read as a voluntary
draw. `uno.game.observe._voluntary` already skips reshuffle events, so the
engine-side helper and the belief-side helper disagree about the same question.

**The misclassification is real; its reach through `voids` is not.** Measured on
`FULL_UNO`:

| Quantity | Measured |
|---|---|
| draw events classified differently (400 uniform-random games) | 488 / 125,216 = **0.39%** |
| of those that `voids` can act on (it requires a 1-card draw event) | **0** |
| spurious voids, 400 uniform-random games (mean 661 turns) | **0** |
| spurious voids, 2,400 frozen-league vs Heuristic games (mean 40 turns) | **0** |
| frozen-league decisions whose distribution changes | **0 / 42,090** per agent |
| frozen-league paired score change vs the Heuristic, 300 deals x 2 seats | **+0.00 pts** for all four |

`voids` inserts only on `ev[2] == 1`, and a penalty that triggered a reshuffle goes
on to deliver its full 2 or 4 cards, so "reshuffle then 1-card penalty" needs the
pile to empty twice inside one penalty draw. A pinned history of that shape does
produce a phantom void (`tests` in `uno.rl.evidence`), but no sampled game reached
it. The only consumer that reads `_is_penalty` directly and counts *events* rather
than voids is `uno.eval.distill.arm_b_features`, which the readiness review already
supersedes.

So the frozen league, the PIMC belief and every published measurement are
unaffected.

**Fixed**, since the blast radius was measured to be empty:
`uno.eval.belief._is_penalty` now skips reshuffle events, matching
`uno.game.observe._voluntary` and the contract that spec section 26.2 already
stated ("penalty draws from a +2 / +4 are excluded: they are not decisions").
Verification before accepting the change:

- pinned regression cases in `tests/eval/test_belief.py`, shown non-vacuous by
  restoring the one-step scan and watching them fail;
- the four frozen rulebots produce **identical action sequences** -- not merely
  identical scores -- over the same 2,400 games against the Heuristic, so no
  tabulation cache is invalidated and `CACHE_VERSION` does not move;
- the full suite passes.

The remaining live error was in `uno.eval.distill.arm_b_features`, which counts
penalty *events* rather than voids and therefore did see the 0.39%; that encoder is
superseded by `uno/rl/features.py` either way.

## Limitations

- `TOY` only. Nothing here measures `TOY_FULL`, `TOY2_SKIP`, `TOY2_FULL` or full
  UNO, and nothing here is evidence about the league objective.
- The exact-gradient comparator is an **achieved** point at one declared learning
  rate and step count, not a certified optimum of the architecture. It saturates
  into a near-deterministic policy, which is a property of undiscounted
  policy-gradient ascent without entropy, not a bound on what the architecture can
  represent.
- The tabular arm shows what the *optimizer and the surrounding pipeline* can do.
  It says nothing about whether the feed-forward representation is sufficient.
- BR is optimality against one fixed known opponent. Reaching it says nothing about
  exploitability, about the hidden-opponent league objective, or about full UNO.
- These are DEV measurements. No EVAL block was claimed or spent.

## Next decision

TBD
