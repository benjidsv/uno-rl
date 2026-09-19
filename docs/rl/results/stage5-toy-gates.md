# Stage 5: exact-toy gates for the PPO learner

**Status, corrected 2026-09-19:** gradient agreement passed. The retained tabular
gate passed against Heuristic but failed against PlayFirstRandom: three of five
training seeds met the two-seat rule. Neural comparisons were measured without a
predeclared numerical tolerance, so they are diagnostics rather than a passed
learning gate. Canonical artifact: `logs/rl_toy/rl_toy_all_1789764834.json`.

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

No neural threshold was frozen before the multi-seed run. That departed from
`pipeline-plan.md` sections 3 and 13, which required a justified tolerance or a
decision to remain in diagnostics. The measurements below must therefore not be
retroactively turned into a pass/fail result.

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

All four fixed-opponent/seat checks passed:

| Opponent | Seat | Worst projection \|z\| | PPO/reference relative error |
|---|---:|---:|---:|
| PlayFirstRandom | 0 | 1.388 | 3.30e-6 |
| PlayFirstRandom | 1 | 1.046 | 3.25e-6 |
| Heuristic | 0 | 1.688 | 3.66e-6 |
| Heuristic | 1 | 2.162 | 3.53e-6 |

The worst projection remained below 3.0 and every PPO identity error remained
below 1e-5. Each sampled score also lay within its declared 95% check around the
exact score. This validates the raw sampled gradient at collection weights; it
does not validate the outcome of repeated clipped PPO updates.

### 2. Tabular five-seed gate

Gaps below are exact `BR - learned score`, in score points. The gate required
every seed to be within 0.50 points at both seats.

| Opponent | Seed | Seat 0 gap | Seat 1 gap | Two-seat result |
|---|---:|---:|---:|---|
| PlayFirstRandom | 0 | 0.144 | 0.624 | fail |
| PlayFirstRandom | 1 | 0.207 | 0.890 | fail |
| PlayFirstRandom | 2 | 0.237 | 0.352 | pass |
| PlayFirstRandom | 3 | 0.195 | 0.363 | pass |
| PlayFirstRandom | 4 | 0.109 | 0.110 | pass |
| Heuristic | 0 | 0.022 | 0.099 | pass |
| Heuristic | 1 | 0.033 | 0.101 | pass |
| Heuristic | 2 | 0.026 | 0.107 | pass |
| Heuristic | 3 | 0.019 | 0.023 | pass |
| Heuristic | 4 | 0.046 | 0.022 | pass |

Thus Heuristic passed 5/5 seeds, while PlayFirstRandom passed 3/5. The retained
overall gate failed. These are exact values, so trajectory-sampling uncertainty
cannot explain the two misses.

### 3. Neural arm: sampled against exact-gradient optimization

At 100,000 sampled learner decisions per seed/seat, sampled neural BR gaps ranged
from 2.61–6.16 points against PlayFirstRandom and 1.77–6.42 against Heuristic.
The 250-step exact-gradient comparator's gaps ranged from 2.65–11.56 and
4.17–10.36 points respectively. Sampled PPO often exceeded this achieved exact
comparator, but not in every cell. That says the chosen exact Adam run is a weak
optimization reference in several cells; it is neither an architecture optimum
nor evidence that the sampled learner reached its attainable limit.

## Interpretation

The production gradient and score-function plumbing passed its declared checks.
The broader claim that the sampled optimizer passed Stage 5 does not follow: the
retained tabular control missed in two PlayFirstRandom seat-1 runs, and the neural
study had no valid acceptance threshold. The study also stopped at `TOY`; the
planned `TOY_FULL` and `TOY2_SKIP` learning progression was not measured.

Full-UNO work subsequently established useful diagnostic learning, but it did not
retroactively close this gate. The residual toy failures should remain visible
while their practical relationship to full-UNO learning is diagnosed.

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

Treat Stage 5 as partially passed: retain the successful gradient result, preserve
the failed learning outcomes, and do not claim the phase exit gate. Before a
frozen full-UNO report, either explain the PlayFirstRandom failures under a
reviewed rule or run a newly predeclared follow-up. Any neural pass rule must be
declared before new multi-seed data. Exercise voluntary draws and action cards on
`TOY_FULL` / `TOY2_SKIP` rather than treating `TOY` as complete coverage.
