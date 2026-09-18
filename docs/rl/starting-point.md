# RL starting point: the foundation handoff

**Status:** foundation handoff, with learner direction updated 2026-09-18. This document
summarizes the facts that constrain RL work without reproducing the foundation
history or its numerical tables. For exact values, follow the links to the
[foundation results index](../foundations/results/index.md).

## 1. What exists

UNO v5 has a deterministic immutable two-player engine, trusted reference walkers,
compiled exact trees, exact evaluation and best response, CFR, symmetry reduction,
sampled full-UNO evaluators, AIVAT components, PIMC/search, belief variants, value
networks, training-target corrections, shaping potentials, a calibrated approximate
response attack, behavioral profiling, and a frozen four-policy rulebot league.

The toys make important parts of the learning stack falsifiable. Exact policy
values, best responses, Nash strategies, gradients, and several sampled estimators
can be checked against known answers there. Full UNO remains sampled and its Nash
value and true strategy ceiling are unknown.

The implementation and contract map is in the
[foundation specification](../foundations/spec.md). Search the relevant numbered
section rather than reading it in full.

## 2. Learner interface

The [observation contract](observation-contract.md) defines three comparison arms:

1. K1-style feed-forward position/discard ablation (A), with an explicitly
   different clock from the historical coarse key.
2. A plus structured public draw/play evidence (B), including board color and
   rank/kind relationships; the first practical actor.
3. A plus recurrent public/private event history (C), retaining the same discard
   input rather than requiring learned bookkeeping.

All arms share the action space, legality mask, seat convention, action-cap clock,
reshuffle handling, setup-tail treatment, and color/rank equivariance requirements.
Actor, privileged critic, and attacker inputs are separate contracts. The revised
contract specifies intended behavior; the existing distillation encoder and
test-only extractors do not yet certify it. See the [readiness review](readiness-review.md).

K1 sufficiency is **not established**. Correcting the old coarse-key symmetry leak
made the toy memory measurements trustworthy, but those local-search scores are not
certified memory ceilings and do not establish full-UNO sufficiency. K1 therefore
remains an ablation against the structured and recurrent alternatives.

Policies return action distributions, not sampled actions. Their reported
probabilities must be the probabilities actually played. Learner observations and
actions must be equivariant under the game's color/rank symmetries; canonicalizing
an observation without treating stabilizer action orbits correctly is insufficient.

## 3. Training opponents and objectives

Step 10C froze four cheap, position-only policies as an equal-weight league after
exact-toy selection, full-UNO DEV selection, behavioral profiling, and held-out
confirmation. The league intentionally includes distinct memory, tactical, and
draw/pass behavior. Its exact membership and measurements are in the
[10C result](../foundations/results/10-readiness/10c-results.md).

The toy-selected tactical policy did not transfer into a clear full-UNO gain. The
minimal-memory variant had the best held-out aggregate, but no new rulebot
established improvement over the Heuristic across both seats and opponent columns.
The league is useful training coverage, not evidence that its members are optimal or
individually unexploitable. The agreed initial objective is its equal-weight score,
with balanced seats, one fixed opponent per episode, and hidden opponent identity.
Robustness is a later objective. The average of opponent-specific BR scores is an
upper bound, not generally the attainable value against that hidden mixture.

A learner maximizing score against the league is solving a different objective
from minimizing NashConv. Likewise, an opponent-specific best response is not a
general policy. Reports must keep league score, held-out generalization, distance to
known toy benchmarks, PIMC comparison, and attack lower bounds distinct.

## 4. Exact-toy validation

Phase 1 should use `TOY`, `TOY_FULL`, and `TOY2_SKIP` for fast diagnosis, with
`TOY2_FULL` reserved for milestones because its cached trees and strategies are
large. The exact tools provide:

- same-seat policy values against fixed opponents;
- BR L1 ceilings and exact Nash measurements;
- exact expected policy gradients for supported tabular comparisons;
- known effects of voluntary draw, action cards, memory abstractions, and
  deliberately grafted strategy flaws;
- exact checks of sampled evaluators and attack quality.

Exact game values remove trajectory-sampling noise, but learned policies still vary
with initialization, minibatches, opponent sampling, and other training randomness.
Use multiple training seeds and predeclare tolerances. A learner should first show
that its encoding and actual sampled gradient estimator approach the appropriate
known target against one fixed opponent before league results are interpreted.

Foundation toy values and their caveats live in the
[results index](../foundations/results/index.md); detailed provenance is linked from
each section.

## 5. Full-UNO evaluation

Full UNO needs paired, per-seat evaluation on common deals and, where appropriate,
common opponent/action streams. The relevant unit, sample count, uncertainty, and
whether an interval is an SE or CI must be explicit. Near-identical policies can
benefit substantially from coupling; dissimilar matchups benefit less. Choose
budgets from DEV variance, not from the most favorable historical case.

DEV seeds are used for fitting, tuning, checkpoint selection, and pilots. Persistent
claims require a fresh block registered in `uno/eval/seeds.py`. Do not repeatedly
inspect or select checkpoints on a confirmatory block.

The report-card and evaluation infrastructure can measure raw and corrected values,
decision categories, search benchmarks, and attacks. Stream summaries: thousands of
full traces are unnecessarily expensive to retain. Measure end-to-end wall time and
memory, including simulation, observation construction, inference, and updates.

## 6. Search benchmark and attack

The column-free PIMC row is a strong achievable full-UNO benchmark and a possible
training opponent. It is not an upper bound on learnable strength. Its current
working budget is 32 worlds at depth 2. That budget and its belief assumptions were
calibrated for the existing leaves and opponents; they must be rechecked when those
components change.

The attack instrument plays an approximate response against a frozen target. If its
scores as responder at the two seats are `b0` and `b1`, then
`NashConv >= b0 + b1 - 1` in score units, with paired covariance carried into the
uncertainty. This is a **lower** bound. A negative or small result says only that the
chosen attack found little.

The shipped response continuation can query position-only targets inside sampled
worlds. `determinize_l1` fabricates opponent private history, so recurrent or other
history-dependent targets require consistent latent-history reconstruction or a
different attack interface. Establish compatibility before reporting LBR numbers on
such a checkpoint. Target support must be measured rather than assumed: in the
foundation calibration, a tabulated target silently returned its first legal action
at 79.5% of fabricated-history queries instead of raising, making the attack largely
insensitive to it. The calibration and limitations are in the
[10a.3C result](../foundations/results/10-readiness/10a3c-results.md) and
[spec section 32](../foundations/spec.md#32-the-attack-instrument-unoevalattackpy-unocompiledflawspy).

## 7. Training returns and variance reduction

Evaluation corrections and training corrections are not interchangeable. The
shipped `TRAINING` target corrects chance, opponent actions, and deal effects, never
the learner's own sampled actions. The exact toy gradient check validates that
specific construction, not every future PPO, GAE, batching, or recurrent-state
choice.

Deal shaping/correction can be relatively cheap. Full sibling corrections were
expensive even after batched inference because constructing sibling states dominates
part of the cost. Start with ordinary returns plus a baseline and add corrections as
measured ablations. Compare variance reduction per wall time, preserve terminal
potential zero, and avoid applying the same correction twice.

Details and measurements are in the foundation index sections on
[training targets](../foundations/results/index.md#the-training-target-and-the-exact-policy-gradient-step-10a2),
[shaping](../foundations/results/index.md#shaping-potentials-and-the-deal-floor-by-level-step-10a2),
and the associated 10a.2 result.

## 8. Beliefs and known dead ends

The current `voids` belief assumes play-first behavior and immediate play of a
playable drawn card. `soft(0.1)` relaxes that assumption, but its toy robustness is
not universal calibration. The failed deal-time particle sampler collapsed because
sampled hands rarely contained each observed play; it does not disprove all particle
filters or recurrent latent-state methods.

Do not require a new particle filter before learning. Reconsider stronger inference
only if an RL experiment shows that inference quality or consistent history
reconstruction is limiting performance.

## 9. Open questions handed to RL

- Can the sampled learner reproduce appropriate exact-toy targets across training
  seeds, rather than merely improve a noisy curve?
- How much do the structured and recurrent observation arms buy over K1 at matched
  data and matched wall time?
- Does league training generalize to held-out policies and stronger search rows?
- Which limitation dominates after the first full-UNO run: representation,
  optimization, exploration, variance, opponent coverage, or evaluation strength?
- Can attacks support every target architecture strongly enough to make their lower
  bounds informative?

Those are experiments, not assumptions. The [RL plan](plan.md) defines their order:
PPO from scratch and exact-toy validation first, B then A/C comparisons on full UNO.
The secondary distillation question is whether PIMC can be compressed; warm starts
remain optional if scratch learning stalls. Workload scenarios and unmeasured
budgets are explicit in [compute-budget.md](compute-budget.md).
