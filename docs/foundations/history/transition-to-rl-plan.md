# Steps 10–11: completed transition plan

> **Historical transition plan.** The Step 10 work described here is complete. The
> active learning phases have been distilled into
> [`../../rl/plan.md`](../../rl/plan.md); current status lives in
> [`../../rl/README.md`](../../rl/README.md).

Agreed direction, 2026-09-17; **10a.3, 10B and 10C complete as of 2026-09-18**.
10C froze its four-agent equal-weight league and confirmed it on fresh EVAL
block 03 after exact toy and full-UNO DEV selection. Based on the
[Step 10 review](../results/10-readiness/10-review.md) and Benji's request to retain the existing
10B → 10C → 11 sequence.

## Placement and scope

**10a.3A (done) → 10a.3B (done) → 10a.3C (done) → 10B (done) → 10C (done) → 11.**

This preserves milestone order and purpose. It adds work before 10B, so it is
not a promise of unchanged elapsed time. Keep 10a.3 bounded: no RL training,
new general particle filter, MCTS, or engine rewrite. The learner architecture
comparison belongs to Step 11. All implementation should favor batched/vectorized
paths, with measured throughput and bounded memory.

| Step | Deliverable | Effect of the review |
|---|---|---|
| 10a.3 | Corrected memory evidence, observation contract, calibrated approximate-response evaluation | New preparation step |
| 10B | Reproducible figures and generated result tables | Same deliverable; use corrected claims and add attack results |
| 10C | Cheap stronger rulebot, transfer experiment, frozen training league | Same deliverable; separate transfer evidence from full-UNO league tuning |
| 11 phase 1 | Validated toy learner with exact measurements | Implement and debug learning against known benchmarks |
| 11 phase 2 | Full-UNO training with sampled measurements | Compare memory choices and measure attacks on checkpoints |
| 11 phase 3 | Further iterations guided by measured failures | Improve whichever component limits learning |

## 10a.3A: repair the memory measurement (done)

Completed on 2026-09-17; see [the results](../results/10-readiness/10a3a-results.md). The
implementation canonicalizes coarse fields and actions together, pins the
review's positive-reach collision, checks the same lifted policies across full
and quotient `SMALL` trees, and refreshes all five planned rulesets. The
remaining 10a.3 work starts at B below.

1. Turn the review's positive-reach K1 counterexample into a bounded regression:
   identical raw K1 observations must induce identical action distributions.
2. Repair coarse canonicalization and action mapping together. Use the coarse
   observation's own symmetry and stabilizer, without choosing coordinates from
   discarded history. Keep the existing full-L1 solvers unchanged.
3. Establish an independent full-tree check on a small ruleset. Verify equivalent
   coarse partitions and the values of the same lifted policies across full and
   quotient representations. Do not require separate local optimization runs to
   find the same optimum; the iterative coarse solver does not certify optimality.
4. Rerun K0/K1/K2 against Random and Heuristic on `TOY`, `TOY_FULL`, and
   `TOY2_SKIP`, then refresh `TOY2` and `TOY2_FULL` once the small checks pass.
   Reuse caches where valid; report resource limits rather than silently retaining
   an old row as a corrected result.
5. Publish achieved scores and absolute gaps to BR L1, with optimization caveats.
   Use the same-seat Random baseline for any normalized skill measure. Preserve
   old results as historical and mark their K1 interpretation as superseded.

**Exit:** deployed coarse policies pass raw-input measurability checks; their
exact values replay correctly; refreshed rows and remaining limitations are
documented. This establishes trustworthy toy evidence, not full-UNO sufficiency.

## 10a.3B: define the observation contract (done)

Completed on 2026-09-17; the contract is defined in [observation-contract.md](../../rl/observation-contract.md). Tests verify the extraction fields and invariance properties.

Specify three Step 11 comparison arms without building the learner yet:

- Audited K1 feed-forward baseline, with an explicit field list.
- K1 plus structured public draw/play evidence, including the board, subsequent
  draws, penalty distinction, and evidence age. This is a new representation,
  not a renaming of the current weak K2 hint.
- Current position plus recurrent public/private event history.

Specify legality masks, publicly known setup-tail information, action-cap clock,
reshuffles, seat conventions, and color/rank equivariance. Treat inferred voids
as uncertain evidence for unknown opponents. Preserve the distinctions between
actor observations, privileged training critics, and attacker's sampled worlds.

Define how each arm's policy can be queried by the evaluator. In particular,
document which opponent inputs the existing determinizer can reconstruct. A
recurrent target cannot be queried using fabricated private history as though
that were its real conditional policy.

**Exit:** explicit inputs, invariance tests, and evaluator compatibility requirements
are written down. Full-UNO sufficiency remains a Step 11 experiment. A redesigned
particle filter is a possible later arm, not a dependency of 10B or 10C.

## 10a.3C: build and calibrate LBR (done)

Completed on 2026-09-18; see [the results](../results/10-readiness/10a3c-results.md) and
[spec section 32](../spec.md). All six items below were built: the interface with an
explicitly enforced (and *measured*) rejection of unsupported private-history
queries, response-PIMC plus a rollout comparator, exact toy calibration against
Random, the Heuristic, `passer0.5`, Nash, a near-Nash graft and deliberately
localized flaws, scoring by actual games validated against `scores_on_seeds`, a
bounded full-UNO DEV cost/sensitivity pilot with the configuration frozen before
EVAL block 02 was claimed, and the attack block in the report card. The exit
criterion's caveat held in both directions: the instrument recovers 77-97 % of a
known toy NashConv, and against the column-free PIMC row on full UNO it finds
nothing — reported as a weak bound, not as safety. Recurrent-target support
remains a Step 11 integration gate, as scoped.

1. Define an approximate-response interface taking a frozen target, responder seat,
   belief/continuation configuration, and computation budget. Record supported
   target representations; reject unsupported private-history queries explicitly.
2. Reuse response-PIMC as the first attacker. Add a local rollout comparator using
   an L1 continuation policy if needed to expose response-PIMC's blind spots; a
   broad search-algorithm comparison is outside this step.
3. On manageable toys, compare achieved attack scores with exact BR against Random,
   Heuristic, a near-Nash policy, and deliberately flawed policies. Report the
   exact exploitability missed, not only the lower bound. Include a case where the
   attack must detect a known flaw so a vacuous detector cannot pass unnoticed.
4. Score the resulting attacker by actual games, separately from its search Q.
   Validate sampled uncertainty against exact toy values and test information
   boundaries, returned probabilities, reproducibility, and batching equivalence.
5. Run a bounded full-UNO DEV cost/sensitivity pilot against supported baselines.
   Freeze the configurations and measurement budget before claiming a fresh EVAL
   block for the baseline report. A noisy or weak bound is a reported result;
   do not keep expanding the experiment until it produces a positive finding.
6. Add the result to the report interface: per-seat attack scores, compute budget,
   uncertainty, and a two-seat NashConv lower bound. With attacker scores `b0`
   and `b1`, use `NashConv >= b0 + b1 - 1` in score units, accounting for paired
   covariance. Never substitute 0.5 for an unknown seat-specific Nash value.

**Exit:** an L1-valid attack instrument is calibrated on known toy weaknesses,
timed on full UNO, and produces a reproducible baseline report with explicit
limitations. It need not approximate every BR closely. Recurrent-target support
is a Step 11 integration gate before reporting attacks on recurrent checkpoints;
it must not silently become a prerequisite to finishing this baseline instrument.

Deliver the three parts as reviewable changes, with commands and results. Benji
makes the commits. Update the spec, results index, context, and status as the
corresponding work is completed; this plan alone does not mark anything done.

## 10B: retain the presentation step (done)

Completed on 2026-09-18; see [the results](../results/10-readiness/10b-results.md). `uv run uno figures`
writes the five figures the brainstorm named, recomputing every cheaply-exact number live
from `.cache/` and checking it against `index.md` to the digit rather than copying it
(0 failed checks); full-UNO-sampled numbers are read from a cited table instead of
re-measured. Two scope cuts, named in the results file: `TOY2_FULL`'s BR L2/L3 stay pinned
(their L1+L3 tree was never cached and recompiling it is out of this step's bound), and
`index.md`'s hand-written tables are not spliced over by the script (its prose and
footnotes are interleaved with the numbers throughout) -- the digit-check is the concrete
form of the staleness concern below, without a full markdown-generation pipeline.

Keep the figure/table generation proposed in `results/10-readiness/10-brainstorm.md` section 6.
Use corrected memory results and label full-UNO PIMC as an achievable benchmark,
not a certified ceiling. Include LBR's lower-bound status and attacker budget.
Historical search gains from different configurations are not an additive causal
decomposition; retain their experiment context in the figure captions.

Generated tables should reproduce unchanged measurements to the printed digit.
Explicitly corrected rows should reproduce the new results, with provenance,
rather than being forced to match superseded values. The new learner is not a
dependency of this step. Later checkpoints and 10C's bot can populate the same
generated figures without rebuilding the presentation machinery.

## 10C: retain the rulebot and league step

Keep the symmetric parameterized rulebot, cheap memory, toy-to-UNO transfer study,
and a frozen league containing distinct draw/pass behaviors. Reuse the evidence
contract from 10a.3B where useful; no neural learner or particle filter is required.

Separate two experiments:

1. **Transfer:** tune on exact toys, freeze, and measure on full UNO. A null or
   negative transfer result is valid evidence and stays in the report.
2. **League quality:** if transfer does not yield a sufficiently strong cheap bot,
   tune a separately named variant on full-UNO DEV data and evaluate it on a fresh
   block. Its gain must not be described as toy transfer.

Do not reject a potentially useful full-UNO memory feature solely because short
toys fail to reward it. Freeze bot parameters, league membership/weights, and
held-out behavioral opponents before the main RL comparison. Use 10a.3's attack
instrument where supported; opponent diversity does not require low exploitability
of every individual training bot.

## Step 11: retain the three learning phases

**Phase 1 — exact toy evaluation.** Start against a single known opponent; compare
tabular L1 learning, a network with exact expected gradients, and sampled learning.
Use `TOY`, `TOY_FULL`, and `TOY2_SKIP` for fast diagnosis, `TOY2_FULL` at milestones.
Predeclare tolerances and use multiple training seeds. Validate the actual sampled
gradient estimator and policy encoding before interpreting learning curves. Then
test the frozen league, distinguishing league reward from exploitability and
declaring whether opponent identity is visible. Start with ordinary returns and
a baseline; measure deal correction and other corrections as separate ablations.

**Phase 2 — sampled full UNO.** Compare the three observation arms at equal data
and equal wall time, with batched simulation/inference. Establish attack compatibility
for each target architecture before using its LBR numbers. Measure paired score
improvements, held-out opponents, PIMC benchmark gaps, attack lower bounds, and
throughput. Choose budgets from DEV variances and keep checkpoint selection off
confirmatory EVAL blocks. A small DEV pilot may inform implementation before the
main training comparison, but does not replace Phase 1 validation.

**Phase 3 — targeted iterations.** Use the observed failure to choose work on
memory, optimization, exploration, variance reduction, opponent coverage, or
stronger evaluation attacks. Reconsider a particle filter if inference quality
or consistent latent-history reconstruction is a measured obstacle. Matching
PIMC is not evidence that the true ceiling has been reached.
