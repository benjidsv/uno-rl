# RL research plan

**Agreed direction, 2026-09-18.** Build PPO from scratch, first validate the actual
learning path on exact toys, then maximize score against the frozen rulebot league
on full UNO. Arm B is the first practical agent, A its memory ablation, and C the
subsequent recurrent comparison. No RL learner is implemented yet.

Adaptability and robustness are a distant second objective, after league learning
works. The numbered phases below are execution stages of this first objective;
Phase 2 full-UNO training does not mean robustness has become the training goal.
PIMC compression is a secondary experiment, not a prerequisite or the default
initialization. A warm start is an option if learning from scratch disappoints.

## 1. Objective and protocol

Optimize expected terminal score, `(return + 1) / 2`, averaged equally across
learner seats and the four frozen opponents. Use terminal return in `[-1, 1]` for
training if convenient; record the units and factor-of-two conversion explicitly.
The actor does not receive opponent identity. Sample one opponent per episode and
keep it fixed; do not replace this with an independently resampled policy at each
opponent decision. Balance seats and opponent allocations by completed games.

The frozen members, each with weight 0.25, are:

| Member | Coverage |
|---|---|
| `heuristic_memory` | conservative play with public evidence |
| `rulebot_transfer_toy_v1` | toy-selected tactical behavior |
| `rulebot_full_dev_v1` | separately DEV-selected tactical behavior |
| `rulebot_unpredictable_t0.15_m0.2_d0.02_k0.04_kw0.1` | stochastic play, voluntary draws, keeping playable draws |

Membership comes from `uno.agents.rulebot`; evidence and limitations are in the
[10C result](../foundations/results/10-readiness/10c-results.md). These are related
policies, not four independent strategic families. Keep the league fixed for the
first experiment; self-play, opponent reweighting, and league expansion are later
interventions if measured failures justify them.

- Use DEV seeds for fitting, hyperparameters, pilots, and checkpoint selection.
  Persistent claims require a fresh registered EVAL block after freezing policies,
  opponents, evaluator, and attack configuration.
- Use multiple training seeds. Separate game-sampling uncertainty from variation
  between trained policies. Pair comparisons on deals and appropriate streams.
- Report per-opponent, per-seat cells as well as the primary equal-weight mean.
  A league-mean gain may trade off individual cells; report such regressions rather
  than silently requiring every cell to improve or hiding them in an aggregate.
- Keep league score, generalization, fixed-opponent best response (BR), PIMC gaps,
  and exploitability measurements distinct.
- Predeclare decision rules and resource caps before a long run. Pilot settings
  below are starting choices, not convergence predictions or measured optima.

## 2. Shared implementation

### Policy and architecture

Use one interface for RL, distillation, tabulation, and evaluation:

`actor observation -> structured encoder -> logits -> legal mask -> distribution`

Collection, old/new log probabilities, losses, evaluation, and checkpoint replay
must use the same distribution, including temperature. Policies report complete
probabilities for their played distribution; legality comes from the engine, never
from the nonzero support of a behavior policy.

Follow the revised [observation contract](observation-contract.md). Preserve card
multiplicities and color/rank relationships. Use shared color/rank processing and
shared action scoring with pooled context; give Draw, Pass, and wild chosen-color
actions explicit descriptors. Pool ranks only within interchangeable copy classes.
An invariant value encoder with a different final head is not automatically a
sufficient or equivariant policy encoder.

Start with width 128 as a bounded pilot choice. A and B share the same position
encoder; B adds structured public evidence. C retains all of A, including discard
counts, and adds history. Prefer a structured GRU first, with shared updates that
preserve color/rank identities; a generic dense recurrent cell is not automatically
equivariant. Exact layer counts and evidence bins must be recorded in the first
implementation experiment, before interpreting comparisons.

Use a separate critic initially. It may consume L2 features but never pile order
or seeds. Include clock/phase and actor perspective; do not assume the old value
network's feature layout already satisfies the new contract. Detach the baseline
in the actor loss. Current private hands alone need not summarize a memoryful
opponent's future behavior; revisit critic history before relying on bootstrapping.

### Learning and collection

Initial reference: PPO with an accompanying Monte Carlo policy-gradient mode in
the same collector and policy code. Use `gamma = 1`, terminal rewards, complete
returns plus a detached baseline, and legal-action exploration. A small entropy
regularizer is allowed but logged separately from game return. No shaping, deal
correction, chance/opponent correction, or GAE initially.

Start with Adam; learning rate, entropy coefficient, clipping threshold, minibatch
size, and update epochs remain pilot parameters. Candidate batch/update sizes and
explicit cost arithmetic are in [compute-budget.md](compute-budget.md). Record the
chosen values and freeze the validation configuration before a gate run. Do not
launch a broad hyperparameter or architecture sweep as a first step.

Collection must handle consecutive learner turns after skip/reverse, draw followed
by play/pass, and variable numbers of intervening opponent actions. Store learner
seat, terminal outcome, legal mask, old log probability, and enough actor-visible
input to reproduce each decision. Freeze the behavior policy for a rollout batch.

Distinguish true terminal games (including the rule-defined action cap) from a
collector cutoff. Start with complete episodes; do not assign zero terminal value
to an unfinished episode. Declare actor-loss normalization: the reference averages
summed decision contributions over games; a flat decision average has a different
scale. Keep this choice consistent in the exact-gradient comparison. Disable
advantage normalization and entropy when testing the unregularized gradient, then
test any enabled normalization/regularization as its own transformation.

Batch live games, observation construction, and inference. Keep immutable engine
semantics and reference paths; profile before accelerating simulation. Store compact
rollouts, not thousands of full state traces or context duplicated for every action.
For C, preserve sequence boundaries, replay/burn-in semantics, and hidden-state
reset rules. Cached hidden states from old parameters are not automatically valid
for new-policy PPO probabilities. Validate replay against a full-history oracle.

## 3. Phase 1: exact-toy validation

### 3.1 Production-path correctness

Repair the issues in the [readiness review](readiness-review.md), and test actual
production extractors and distributions. Required checks include nontrivial masks,
color and rank transformations, stabilizer orbits, hidden-state independence of
actor inputs, serialization, and batched/reference agreement. Test-only field
extractors do not certify a production encoder.

### 3.2 Gradient and tabular controls

Begin against `PlayFirstRandom` on `TOY`, then Heuristic. Compare:

1. A tabular L1 learner, for which the fixed-opponent BR is the appropriate target.
2. The chosen neural architecture optimized with exact expected gradients where
   practical.
3. The same architecture using the sampled path intended for full UNO.

Compare network-parameter gradients, not just unrelated tabular logits. Use
well-visited, multi-action decisions and game-level uncertainty for sampled gradient
projections. Check the PPO surrogate gradient at the collection policy; subsequent
clipped/repeated PPO updates are not claimed to be unbiased gradients of that same
objective. Keep score/return units and game weighting identical across paths.

**Initial tabular gate:** five training seeds, each within 0.5 score points of the
same-seat BR at both seats on `TOY` against the fixed opponent. Predeclare the data
cap using the pilot. Failure triggers diagnosis, not a post-hoc relaxed threshold.
This is a chosen acceptance target, not evidence it is achievable within a given
number of updates.

For the neural learner, report both the BR gap and the gap to exact-gradient
optimization of the same architecture. Do not require a compressed network to reach
an unattainable L1 BR or call the exact-gradient run a certified architecture optimum.
The quantitative neural tolerance remains open: set it from the bounded diagnostic
before the multi-seed gate run, recording the rationale and training budget.

### 3.3 Broader mechanics and league

Extend to `TOY_FULL` and `TOY2_SKIP` to exercise voluntary draws and action cards.
Reserve `TOY2_FULL` for milestones because of tree size. Verify sampled evaluation
against exact evaluation with predeclared uncertainty checks. Only after the fixed
opponent path works, train against the frozen episode-level league.

For hidden opponent identity, the average of opponent-specific BR values is an
upper bound, not generally an achievable league target. Exact evaluation must retain
the fixed episode-level opponent mixture; averaging action distributions at each
node defines a different opponent.

**Exit gate:** production correctness and gradient checks pass; five-seed tabular
and declared neural gates are met, or any residual discrepancy has an explicit
reviewed explanation. The sampled learner must be trustworthy before full-UNO
learning. A bounded throughput smoke run may occur earlier but provides no learning
or strength claim.

## 4. Phase 2: full UNO, still optimizing league score

First measure the complete system in a bounded pilot: simulation, features,
inference, learner updates, peak memory, checkpointing, and evaluation. Fill in the
open compute budget and set data/time caps before a larger run.

Train B from scratch first, then A with the shared encoder and evidence removed.
Add C after the feed-forward pipeline works. Compare at equal environment data and
at equal wall time; count learner decisions, total engine decisions, and completed
games separately. Use matched training-seed protocols and report all seed outcomes,
not just the best run. PIMC labels and teacher compute count as extra resources in
any later warm-start comparison.

### Evaluation and selection

Use a fixed DEV selection schedule and primary equal-weight league score. Record
checkpoint cadence and tie-breaking before training; use exact toy evaluation
where feasible. Freeze the chosen checkpoint/configuration before fresh EVAL.

The report card contains:

- scores and paired baseline differences for every opponent and seat;
- training-seed variation and game-sampling uncertainty separately;
- Heuristic and Random references plus behaviorally different holdouts, such as a
  higher keep-drawn rate, and a frozen PIMC row;
- PIMC performance against the same columns under the same evaluation protocol;
- draw/keep, wild retention, action-card use, entropy, length, and action-cap rates;
- compatible attack scores and NashConv lower bounds, with covariance and budgets;
- total data, training/evaluation time, inference cost, and peak memory.

Heuristic alone is not strong evidence of generalization because it resembles a
league member. Holding out policies measures coverage; improving robustness is not
yet the optimization target. Choose full-UNO evaluation budgets from DEV variance.

Validate target support before using response search. A/B can use the position and
public-history interface; C needs validated latent-history treatment or a different
attacker. Real-game matches against frozen valid attackers remain possible. A weak
attack is not safety evidence. Sophisticated recurrent-target search is not a
prerequisite for implementing C.

**Exit gate:** reproducible league improvement, a measured strength/cost comparison
of representations, and an identified main limitation. Set the practical minimum
league gain and evaluation precision from the pilot before confirmatory runs.

## 5. Secondary track: can PIMC be compressed?

Question: can a cheap feed-forward student preserve much of the frozen column-free
PIMC teacher's playing strength? Distillation is not initially a warm start or a
new training opponent requirement.

Repair legality, equivariance, masking/temperature consistency, feature-contract
mismatches, and validation splitting before interpreting compression results.
Reuse the shared actor encoder/distribution. Bootstrap from teacher-reached states,
then query the frozen teacher on student-reached states (dataset aggregation).
One aggregation round is a bounded start, not a guarantee against distribution
shift. Split by deal groups across seats/columns, fit preprocessing on training data
only, and retain a disjoint DEV game-score diagnostic.

Report held-out soft-target loss, disagreement, teacher-Q regret, paired game-score
gap by seat/column, label-generation time, student inference cost, and memory. Q
regret uses the teacher's searched values and is not true-game regret. Teacher world
sampling and action-ID tie-breaking can make individual labels nonsymmetric; test
and document this rather than forcing exact reproduction of label noise.

The approximate seven-point smoke gap is not an established compression limit.
Choose an acceptable retained-strength/cost trade-off after the corrected pilot,
before a larger compression run. Register fresh EVAL only for a frozen claim.
If from-scratch RL later stalls, compare distilled initialization against scratch
with teacher compute/data explicitly included; do not silently replace the primary
experiment.

## 6. Later decisions and open items

Immediate open items:

- Hardware, pilot throughput/memory, and the final per-run resource cap.
- Exact encoder depth, structured-evidence bins, and recurrent update layout.
- PPO hyperparameters, neural toy tolerance, DEV checkpoint schedule, and minimum
  practically useful full-UNO gain.
- Corrected distillation budget and acceptable strength/cost trade-off.

Resolve these through bounded diagnostics, not an unbounded search. The accepted
choices above are the starting direction and do not need to be re-litigated each
session.

After league learning works, select interventions from measured failures:
representation, optimization, exploration, variance per wall time, or throughput.
Deal correction/shaping and GAE are separate ablations; chance/opponent corrections
carry substantial state-construction costs. Keep terminal potential zero and avoid
double correction.

Only later consider the trade-off between league exploitation and adaptability or
robustness: broader opponents, self-play, adaptive leagues, or stronger attacks.
Revisit particle filtering/history reconstruction only when its absence is a
measured constraint. PIMC remains an achievable benchmark, never an upper bound.

## 7. Result lifecycle

New experiments go under `docs/rl/results/` and begin with question, predeclared
rule, base commit, configuration, training and evaluation seeds, result with units
and uncertainty, interpretation, limitations, and next decision. Exact values have
one canonical home. Update the [results index](results/index.md) and replace the
[README](README.md)'s current state rather than appending history. A plan or review
is not a learner result.
