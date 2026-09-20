# RL pipeline execution plan

**Drafted 2026-09-18, against base commit `247f2b4`. Planning only; no learner or
performance improvement is implemented or validated by this document.**

This makes the [research plan](plan.md) operational. That document remains the
authority for objectives and phase gates; the [observation contract](observation-contract.md)
remains the authority for actor/critic information and symmetry. Read this plan
alongside the [readiness review](readiness-review.md) when implementing. Existing
distillation code is not a compliant learner to reuse unchanged.

## 1. Decisions and scope

### Confirmed in discussion

- Synchronous PPO first: parallel collection with frozen weights, followed by
  optimization. No collection overlaps learner updates.
- Local MLflow tracking. Cross-machine resume is desirable if it stays modest in
  complexity; a shared tracking server is not required.
- Target both machines below. A comfortable full exploratory run is 30–60 minutes;
  multi-hour overnight runs are available when justified.
- Write a concrete plan before implementation. No commits or publication.

| Machine | CPU and memory | Accelerator |
|---|---|---|
| Mac | M1 Pro, 16 GB unified memory | MPS |
| Desktop | Ryzen 7700X, 32 GB system RAM | RTX 4070 Ti, 12 GB VRAM |

### Inherited research decisions

Train from scratch to maximize the equal-weight frozen four-opponent league score,
balanced over both seats. Opponent identity is hidden from the actor and fixed per
episode. Validate the actual learning path on exact toys first. B is the first
full-UNO actor, A its ablation, C the later recurrent comparison. Use terminal
returns and a separate detached baseline initially: no GAE, reward shaping,
training-return corrections, self-play, or adaptive league.

### Proposed defaults, not measured optima

The numerical settings and implementation choices below are bounded starting
proposals. They can change during the diagnostic pilot with a recorded reason;
freeze them before multi-seed gates and comparisons. A proposed default does not
mean Benji separately endorsed every hyperparameter.

Interpret the routine budget as **per training seed**, including initialization,
collection, updates, scheduled DEV evaluation, logging, and checkpointing. A
multi-seed study takes multiple such runs. Final confirmatory evaluation, PIMC
benchmarks, and attacks have separately declared budgets.

## 2. Execution topology

```text
resolved configuration + seed manifest
                  |
             coordinator ------------- local MLflow + local artifacts
                  |
        frozen actor version k
                  |
        persistent CPU worker processes
        [batched live games in each worker]
                  |
        complete, compact episode rollouts
                  |
        fixed returns and old-critic advantages
                  |
        synchronous actor and critic optimization
                  |
        checkpoint / scheduled DEV evaluation
                  |
             actor version k+1
```

The coordinator owns episode assignments, policy versions, rollout assembly,
optimizers, evaluation scheduling, checkpoints, and MLflow writes. Workers own
engine states, frozen opponents, incremental public summaries, action sampling,
and a read-only CPU actor replica. They do not update models or write MLflow runs.

Use persistent **processes**, with an explicit `spawn` context, rather than rely
on Python threads for engine parallelism. Start with one PyTorch/BLAS compute
thread per collector process; profile coordinator thread counts separately during
optimization. Numerical-library threads multiplied by worker count must not
silently oversubscribe the machine. Keep process startup out of per-update work.

Each worker advances several live games, batches ready learner observations, runs
one actor call, samples actions, and advances games through opponent decisions to
the next learner decision or terminal state. Consecutive learner turns and
after-draw choices require no alternating-seat assumption. Construct observations
and features in batches where practical; retain the immutable engine as oracle.

Start with worker-local CPU inference. It avoids a request/response exchange at
every learner decision. Transfer compact rollout chunks to the coordinator,
bounded by backpressure; do not send full immutable state/history objects for
every step. Simple contiguous array transfers are the first implementation;
shared-memory buffer pools are a later optimization if measured transfer cost
justifies them.

Centralized batched inference is a bounded competing design, not an initial
requirement. Test it if local actor inference becomes a material bottleneck or
worker replication limits memory. Its benchmark must include queueing, feature
transfer, inference latency, and stalled simulation. GPU utilization alone is not
a success criterion. Never change PPO batch size merely to make an accelerator
look faster without treating that as an algorithmic change.

Choose CPU/MPS/CUDA for optimization independently from collection. Keep one
accelerator-owning process initially. Cross-device collection and recomputation
must pass frozen-policy probability checks before accepting that pairing.

## 3. One synchronous iteration

1. Freeze actor version `k`, critic version `k`, and all preprocessing. Create the
   complete episode manifest for the iteration before dispatching work.
2. Assign equal completed-game quotas to the eight opponent × learner-seat cells.
   For fixed-opponent toys, balance the two seats. Queue small bundles dynamically
   to avoid binding a slow opponent cell to one worker.
3. Collect every assigned game to true terminal state under actor `k`. All games
   retain their sampled opponent throughout the episode. Policy version mismatch
   is an error. Complete the manifest regardless of game completion order.
4. Reassemble episodes in logical ID order, validate counts and probabilities,
   and attach the learner-perspective terminal return to its decisions.
5. Evaluate critic `k` on pre-action critic inputs, then freeze advantages for all
   PPO epochs. This can happen in large batches after collection; collector
   workers need not carry a critic replica.
6. Optimize actor and critic on this batch; do not recompute advantages with the
   newly fitted critic. Discard rollout training data after its configured epochs.
7. Advance counters, evaluate if due, checkpoint if due, and publish actor `k+1`.

### Episode quotas and variable batch sizes

Use a fixed **game count**, not a race to the first N completed decisions. The
initial throughput smoke uses 64 games per cell: 512 league games per iteration.
This is a sizing proposal, not a requirement that games average any given length.

After the pilot, target approximately 8,192 learner decisions per iteration by
choosing and freezing a game quota:

`games_per_cell = max(1, round(8192 / (number_of_cells * mean_learner_decisions_per_game)))`.

Use the mean under equally weighted cells, measured before the main run. Record
the resulting quota and retain variable decision counts as policy behavior
changes. Do not adapt quotas within a comparison based on its emerging scores.
For tiny toys, a different predeclared quota is expected.

Selecting only the fastest completed games would favor short trajectories. A
predetermined manifest avoids that selection effect and makes worker count an
execution setting rather than a data-allocation setting. Terminal-on-deal games,
if supported by a ruleset, count as games even if they have zero learner decisions.

The ruleset action cap is a true terminal result. A process interruption or
wall-time cutoff is not a draw or zero-return terminal. Finish the iteration on a
graceful stop; after a hard failure, replay it from the preceding saved boundary.
Never silently discard a long game and retain the rest of its iteration.

## 4. Rollout and randomness contracts

Store per episode: logical ID, deal seed, action-stream IDs, training seed,
opponent version, learner seat, behavior-policy version, terminal return/reason,
total engine decisions, learner decisions, and row offsets. Training metadata
must not become actor input.

Store per learner decision: exact declared actor features, separate pre-action
critic features, engine legal mask, chosen action, old log probability, and the
episode/sequence reference. Materialize returns and advantages once per update.
Keep complete old distributions only in bounded diagnostics when needed; ordinary
PPO needs the selected-action probability. The public agent API still returns the
complete distribution actually played.

Use compact typed arrays, fixed schema versions, and finite-value checks. Estimate
bytes per decision, rollout, worker, and optimizer before allocating a large batch.
Do not preserve thousands of full game traces. Keep a small explicitly bounded
failure/replay sample outside the normal training buffer.

Randomness is keyed by logical work, not process ID or completion order. Separate
model initialization, episode/deal scheduling, learner action sampling, opponent
action sampling, and minibatch permutations. Version the derivation scheme.
Worker count changes must preserve the assigned episodes and logical streams;
floating-point batch/device changes may still affect sampled trajectories.

The new collector can use explicit per-episode, per-role action streams, but that
differs from the current evaluator's information-state-hashed sampler. Document
that change and test distributions/game semantics rather than claiming identical
historical traces. Coupled evaluations use the existing explicit shared-stream
contract; checkpoint names remain distinct from their coupling stream IDs.

Proposed DEV partition for new learner work:

| Deal seeds | Use |
|---|---|
| `[0, 800000)` | Training deals |
| `[800000, 900000)` | Fixed scheduled selection panels |
| `[900000, 1000000)` | Pilot evaluation sizing and diagnostic/holdout panels |

These are DEV partitions, not fresh confirmatory data. Before implementation,
record concrete panels and audit overlap with any new learner datasets. Cycle
training deals through a seeded permutation when the finite pool is exhausted;
use new episode/action streams, log cycle counts, and never drift into EVAL.
Fresh registered EVAL blocks are selected only for frozen claims.

## 5. Feed-forward actor and evidence proposal

Implement one production path for feature extraction, encoding, masking,
probabilities, sampling, recomputation, and evaluation. Temperature starts at 1
everywhere. Evaluation samples that same policy; argmax deployment would be a
different policy and requires a separate result.

### Encoder

- Width 128, FP32, no dropout, batch normalization, or learned input-statistics
  normalization initially. Use contract-defined deterministic count scaling.
- Preserve colored card/rank tokens, separate semantic action-card kinds, wild
  tokens, and indexed evidence relations. No learned absolute color IDs or
  distinguishable embeddings for interchangeable ranks.
- Initial shared token projection, followed by two shared relational processing
  blocks. Each block uses a two-layer ReLU MLP with a width-128 output and residual
  update. Supply local token features, color context, rank context across colors,
  and invariant global context. Pool interchangeable ranks within copy classes;
  never merge distinct semantic kinds as if they were interchangeable ranks.
- Shared action-scoring MLP with one width-128 hidden layer. A colored play uses
  its card/color/rank context; a wild play includes its chosen-color context.
  Draw and Pass have explicit descriptors. Apply engine masks before log-softmax.
- Global pooling is supplementary: action logits must retain their indexed local
  relations. Relabeling color/rank axes must relabel output probabilities, including
  equality inside stabilizer action orbits.

Evidence whose active color differs from the top card's color needs a relation
between two color indices. Encode that relation with shared processing and
index-respecting aggregation; flattening ordered color pairs into an unrestricted
MLP would break the contract. Final tensor shapes, pooling operators, and parameter
counts must be recorded and tested before training. These blocks are a concrete
architecture proposal, not a proof of sufficient expressiveness.

### B evidence

Use exact card-draw ages `0, 1, 2, 3, 4` plus a `5+` overflow bin. Exact bins keep
their joint board identity. The overflow token/color channels keep the most recent
qualifying board and a separate global counter retains the number of all overflow
events; older board identities are an explicit compression loss. Index voluntary-
draw evidence by active color and full top identity, including rank/kind/color, as
required by the contract. At insertion, exclude cards drawn at that event from
that event's age; subsequent opponent draws age it by actual card count. Plays do
not clear history.

Maintain exact integer counts internally so incremental bin updates agree with
full-history reconstruction. Scale retained board indicators by the declared
multiplicity cap of 8; scale global per-bin event counts as `min(count, 32)/32`.
Expose separate penalty-event and penalty-card counters, each capped at 32 and
divided by 32, and draws since the last opponent play capped/scaled the same way.
These caps are explicit compression choices to diagnose if B stalls. Include
recent play/chosen color and setup fields without silently changing their contract.
No hard void mask is inferred from voluntary draws.

A uses the same position encoder with the B evidence branch absent. Use the same
position width/depth; report the resulting parameter-count difference rather than
claiming identical capacity. Both full-history reconstruction and incremental B
summaries must produce identical declared features on pinned and sampled histories.

### Separate critic

Start with a separate width-128 structured invariant network, two processing
blocks and a scalar linear value head. Input: B's declared public/own features plus
opponent current hand multiplicities, actor-relative seat, clock, and phase.
Never include seed, pile order, or opponent identity. This is an L2 subset, not a
claim to summarize all memory relevant to future play.

Use this same critic input contract for A/B comparisons to isolate actor changes;
the critic may see public evidence omitted by A. It shares no trainable parameters,
normalization, or recurrent state with the actor. Actor export excludes it.
Fit undiscounted terminal returns with MSE. Approximation affects baseline quality;
it does not authorize TD/GAE bootstrapping without separate validation.

## 6. PPO objective and starting configuration

For game `g` and learner decision `t`, use learner return `R_g` in `[-1, 1]` and
`A_gt = R_g - V_old(x_gt)`. Freeze the critic before fitting the current batch and
detach `A_gt`. Define ratio `r_gt = exp(logp_new - logp_old)`.

The actor maximizes the **game-average sum**:

`(1/G) * sum_g sum_t min(r_gt * A_gt, clip(r_gt, 1-eps, 1+eps) * A_gt)`.

With uniformly shuffled decision minibatches, multiply their mean surrogate by
`T/G`, where `T` and `G` are the complete rollout's decision and game counts. This
preserves the full-batch objective's scale in expectation. Do not divide each
game's contribution by its length, normalize separately by minibatch game counts,
or silently replace this with a flat decision-mean loss. Test minibatch/full-batch
gradient agreement, including unequal game lengths and final partial minibatches.

Critic loss is a decision-mean MSE, optimized separately. Report raw entropy in
nats together with mean legal-action count, the local ceiling `log(|A(s)|)`,
per-state normalized entropy, and effective actions `exp(entropy)`; raw entropy
alone is confounded by the number of legal actions. Initial entropy bonus is zero.
If later enabled, use a declared
decision-mean entropy bonus with its own coefficient and validate it separately
from the unregularized return gradient. Advantage normalization starts disabled.

| Setting | Proposed first diagnostic |
|---|---|
| Discount | `gamma = 1` |
| Returns | Complete terminal Monte Carlo returns |
| Actor / critic optimizer | Separate Adam optimizers |
| Actor / critic learning rate | `3e-4` / `1e-3`, fixed initially |
| Adam betas / epsilon / weight decay | `(0.9, 0.999)` / `1e-8` / `0` |
| PPO clip | `0.2` |
| Actor / critic epochs | 4 / 4 |
| Decision minibatch | 512, or the entire rollout if smaller |
| Advantage normalization | Off initially; if enabled, normalize once globally over the complete rollout before shuffling into minibatches |
| Entropy coefficient | `0` initially; `0.001` is a conditional exploration diagnostic |
| Actor / critic gradient norm cap | `0.5` / `0.5`, separately |
| Precision / temperature | FP32 / 1 |
| Value clipping / LR schedule | None initially |
| Actor early stop | Minibatch approximate KL above `0.02`, checked before each optimizer step |

For the KL diagnostic use sampled old-policy actions and the nonnegative estimate
`mean((r - 1) - log(r))`; also log clip fraction and maximum absolute log-ratio.
After each actor epoch, compute surrogate, entropy, KL, ratio-clip fraction and
maximum absolute log-ratio over the complete rollout under that epoch's final
policy. Use the current optimizer minibatch's approximate KL for the early-stop
decision before its optimizer step, reusing the loss forward pass. Thus a
high-variance minibatch stops the actor before the next update without an extra
forward pass. Do not report the last optimizer minibatch as an update aggregate.
This is a sampled diagnostic, not an exact KL guarantee. Complete critic fitting
even if actor epochs stop early. Log actual optimizer steps and epochs used.

The game-summed actor objective may have larger gradients than common
decision-averaged PPO code. Measure unclipped norm and fraction of clipped updates;
do this separately for actor and critic. Report critic explained variance both for
the frozen pre-update critic on the fresh rollout and for the fitted critic on that
same rollout; the latter measures fit, not next-rollout generalization. Do not
interpret the proposed learning rate as transferred calibration. For the
exact-gradient test disable entropy, normalization, clipping of gradient norms,
and optimizer effects; compare raw parameter gradients at the collection policy.

If a diagnostic fails, first determine whether the issue is correctness, baseline
fit, exploration collapse, or update magnitude. Change one relevant setting with
a logged rationale. No automatic broad hyperparameter search or unrecorded tuning
on the five-seed gate. NaNs, illegal actions, or information leaks stop the run.

## 7. Throughput and memory pilot

The supplied pilot measured approximately 13–13.5k learner decisions/s by halving
total engine decisions, versus much higher large-batch synthetic MLP throughput.
This supports investigating parallel collection. It does not establish real
feature/actor throughput, parallel scaling, or a training duration. In particular,
the batch-512 update proxy favored CPU over MPS, while larger batches favored MPS.
See [compute budget](compute-budget.md) for the original measurements and caveats.

After production features/distributions work, run a bounded systems diagnostic on
each machine with the same episode manifest and frozen initial policy:

1. Establish one-worker end-to-end collection, update, and memory costs.
2. At 64 live games/worker, try 1, 2, 4 workers, then 6 and 8 if previous results
   and memory justify them. Do not assume the M1 Pro's core configuration.
3. For the best one or two worker counts, compare 32, 64, and 128 live games per
   worker. Tune execution settings using throughput, not league score.
4. Compare CPU with MPS on Mac and CPU with CUDA on desktop for the real actor plus
   critic update at unchanged PPO minibatch/epochs. Warm up and synchronize device
   timing. Include weight copies back to collectors.
5. Recheck the chosen setting on a longer roughly five-minute integrated run,
   including evaluation, logging, and checkpointing. Inspect straggler time and
   memory trend, not only a short peak-rate measurement.

Use 20–30-second timed samples after warmup for candidates and stop exploring once
the next worker count gains less than 10% throughput or causes memory pressure.
The threshold is a practical search rule, not a universal scaling claim. Start
with a maximum of eight collectors; consider more only with evidence. Cap this
systems tuning at about 15 minutes per machine, excluding first-time setup.

Proposed application memory budgets: 10 GiB on the Mac, 24 GiB system RAM and
10 GiB VRAM on desktop. These leave headroom and are adjustable operating limits,
not measured requirements. Count all workers, coordinator, rollout queues, model
and optimizer copies, and accelerator allocations. On unified memory do not add
overlapping CPU/GPU accounting as if they were separate physical pools. Report
aggregate RSS with shared-page caveats and watch system memory pressure/swap.

Report total learner decisions/s over the full iteration, games/s, total engine
decisions/s, stage times, queue wait, final-game drain time, inference batch-size
distribution, transfer volume, and memory. Worker stage times are overlapping
work, so their sum is not elapsed wall time. No correction, search, or teacher
label generation is included silently in a learner rate.

## 8. Time and data budgets

- **Diagnostic:** about five minutes for integrated health, followed by exact toy
  diagnostics as needed. This is not a convergence test.
- **Routine:** choose 30 or 60 minutes per seed before launching. Use 30 minutes
  first; a 60-minute extension is a separately declared continuation, not a
  post-hoc change to a finished equal-budget comparison.
- **Overnight:** choose an explicit hour/data cap for a multi-seed study or a
  longer-data experiment after the short run passes its gates.

Reserve roughly 10% of routine time for scheduled DEV evaluation initially; measure
and revise before freezing a study. Checkpoints and MLflow time count too. Use
measured complete-iteration rate to choose the data cap; do not multiply the proxy
single-thread rate by core count as a promised budget.

Time caps are soft at iteration boundaries. Do not start another iteration when
the estimated time remaining cannot cover collection, update, required evaluation,
and saving. Log actual overrun if the final complete iteration takes longer than
expected. A hard interrupt rolls back to the last valid saved boundary.

Use two explicitly labeled comparisons: equal learner-decision budget (with actual
complete-iteration overshoot reported) and equal wall-time budget on the same
machine. Keep common game quotas within an A/B study. Report total engine work and
games as well: equal learner decisions need not mean equal simulation cost.
Cross-hardware runs support portability; they are not interchangeable wall-time
measurements. Do not select the best training seed and present it as typical.

## 9. Evaluation and selection

Use exact toy evaluation where feasible. On full UNO, evaluate the stochastic
played policy with the declared temperature and masks, initially with raw terminal
scores. AIVAT evaluation is an optional separately validated cost/variance choice,
not required to start the learner.

Proposed routine cadence: initial policy, every 10% of the declared learner-data
cap, and the final completed iteration. For a wall-time comparison, freeze ten
corresponding wall-time milestones instead. Evaluate at the first completed update
after a threshold and record actual counters. Avoid extra evaluations triggered by
promising training returns. Run evaluation synchronously initially so its resource
cost and checkpoint association are unambiguous.

Use a fixed disjoint DEV panel across checkpoints and variants, balanced over
opponents/seats and coupled to cached baseline evaluations under the same protocol.
Start sizing at 256 deals per cell in the pilot. Estimate variance of the actual
paired league-mean difference, grouping all seats/opponents sharing a deal;
choose the main-run panel from precision and time cost. A sizing estimate is
`n_required = ceil(n_pilot * (pilot_95pct_halfwidth / target_halfwidth)^2)`.
Validate the resulting precision on the chosen panel rather than treating this
estimate as a guarantee. Do not change panel size because a checkpoint looks good.

Select the checkpoint with the largest DEV equal-weight mean score; ties within
`1e-6` go to the earlier checkpoint. Always report the final checkpoint too. The
selected checkpoint's DEV interval is descriptive, not a post-selection
confirmatory interval. Training return is a diagnostic, not the selection metric.

For every evaluation retain cell means, paired baseline differences, deal counts,
standard errors/95% CIs with labels, and covariance needed for aggregates. Report
game draws, game lengths, action-cap rate, forced draws, optional-draw opportunity
rate, `P(Draw | Draw and a Play are both legal)`, keeps, wild/action-card use, and
entropy. Aggregate the eight cell scores equally, never by decision count.

Holdout rulebots, Random/Heuristic references, and frozen PIMC comparisons belong in
the final report card. Give expensive PIMC and attack measurements their own caps.
PIMC is achievable strength, not an upper bound. Attacks produce lower bounds on
NashConv, not safety certificates, and must support the target's query semantics.

Before confirmatory full-UNO evaluation, freeze policies for every training seed,
baseline, panel protocol, and attack settings; register a fresh EVAL block. Report
training-seed variation separately from deal-sampling uncertainty. When pooling
seeds, account for shared evaluation deals rather than treating every seed × deal
entry as independent. Five training seeds are the initial study proposal.

## 10. Local MLflow and artifacts

Use a local MLflow tracking URI on each machine, configured explicitly rather than
relying on version-dependent defaults. Proposed fresh setup: local SQLite metadata
plus a local artifact directory, both outside version control. Reuse Benji's local
setup if already present. Pin the chosen MLflow version when implementing; this
plan does not add the dependency or create a database.

Use one experiment such as `uno-v5-rl`, one run per training seed/configuration,
and tags for `study_id`, arm, phase, seed, machine, and run kind. A resumed run on
another machine becomes a new run linked to the same logical training lineage.
Separate final evaluation jobs get their own linked runs. No model registry,
serving deployment, or centralized MLflow server is required.

Only the coordinator logs. Batch metrics once per update, with evaluation metrics
at evaluation boundaries. Use cumulative learner decisions as the MLflow step and
also log update index, games, engine decisions, optimizer steps, and cumulative
active runtime. Preserve a compact local tabular metrics file so results can be
plotted against both data and time without relying on the UI's step convention.
In MLflow, use exactly one slash per metric: the prefix selects a single semantic
folder and the remainder is a flat descriptive name. The folders are `primary`,
`progress`, `training`, `actor`, `critic`, `batch`, `timing`, `throughput`,
`collection`, `resources`, and `evaluation`; MLflow does not recursively render
nested slash components. Keep only DEV/train score, whole-iteration data
throughput, entropy, approximate KL, and pre-update critic explained variance in
`primary`.
The local CSV keeps its compact flat analysis schema.

| Group | Required content |
|---|---|
| Provenance | Base commit, dirty patch artifact if any, resolved config/full hash, algorithm hash, dependency lock, Python/Torch/MLflow versions, OS/device, schema/rules/league hashes |
| Seeds | Training seed, DEV partitions and exact panels, logical episode counters, stream derivation version, deal reuse counts |
| Training | Per-cell scores/counts, rollout-wide final-policy actor surrogate/entropy/KL/ratio-clip diagnostics, critic MSE and pre/post-fit explained variance, return/advantage moments, gradient norms and clipping rates |
| Progress | Learner and engine decisions, completed games, update/optimizer counts, actual epochs, cumulative time |
| Performance | Stage wall times, throughput, inference batches, queue/drain time, transfer bytes, all-process memory and device memory |
| Evaluation | Checkpoint ID/hash, per-cell scores/differences, primary mean, SE/CI and sample counts, behavioral summaries, baseline/panel versions |
| Recovery | Parent source run/checkpoint, logical lineage, original versus current device/workers, replayed work, stop reason |

Log undefined statistics, such as explained variance with zero return variance,
as explicitly unavailable rather than silently inventing a value. Keep metric
definitions and units stable across runs. Avoid logging every game or decision to
MLflow and never include credentials in configuration artifacts.

Artifacts: resolved configuration, seed/panel manifest, run manifest, portable
checkpoints, metrics/evaluation tables, bounded failure examples, and concise
summary. The readable scientific interpretation still belongs under
`docs/rl/results/`; link its MLflow run IDs and artifact hashes. MLflow is not a
replacement for result provenance and limitations.

If tracking becomes unavailable, retain local metrics/checkpoints and mark pending
uploads for retry; do not lose a training run because logging failed. Failure to
write a recoverable checkpoint is a reason to stop gracefully. Track upload status
so an apparently successful run is not mistaken for a complete artifact archive.

## 11. Checkpoints and cross-machine resume

Save only at completed-update boundaries, using a temporary file followed by an
atomic replacement and a checksum manifest. Proposed cadence: every scheduled
evaluation, graceful shutdown, and at most two minutes between boundary saves.
Retain latest, previous latest, DEV best, and final; cap other milestone retention.

A portable training checkpoint contains:

- Actor and critic parameters, optimizer states, and any scheduler state.
- Resolved full configuration and its provenance hash; a separate immutable
  algorithm hash; feature/action/rules/league schemas and hashes; preprocessing;
  model architecture; format version.
- Next update/episode IDs, deal schedule position, RNG states for initialization/
  sampling/minibatches and available accelerator generators, and stream scheme.
- Cumulative counters/time, next evaluation threshold, selected-best score and
  checkpoint ID, panel manifest, lineage and source MLflow run ID.

Store tensors portably on CPU, remapping optimizer state on load. No worker pool,
live game, open queue, or unfinished rollout is serialized. Keep a small independent
actor-only export for evaluation and deployment.

To continue elsewhere, copy the checkpoint and manifest, validate compatibility,
choose device/worker settings, and create a new local MLflow run tagged with parent
run ID and checkpoint hash. Continue logical counters; record segment time and
cumulative time separately. There is no need to migrate the original MLflow DB.
Preserve prior selected-best artifacts when moving if subsequent final selection
can still choose them.

Device, worker/live-game execution width, evaluation budget/cadence, run caps, and
artifact paths may change. Batch construction, training seed, model, observation
schema, league, optimizer hyperparameters, and objective are immutable; changing
one makes an explicit experiment branch, not an invisible resume. The full config
hash records the exact segment while the algorithm hash enforces this boundary.
Cross-device continuation preserves progress and algorithm configuration, not
bitwise numerical trajectories. Exact continuation on one device is a tested
property only if deterministic kernels/order support it; saving RNG states alone
is not proof. A worker failure retries its assigned logical work;
duplicate episode results are rejected. If recovery is uncertain, reload the last
boundary and replay the whole iteration.

Validate CPU save/load, same-machine interrupted-versus-uninterrupted continuation,
and a short Mac ↔ desktop continuation before relying on portability. Cross-device
tests require finite consistent probabilities within declared tolerance, valid
optimizer state, counters, and no repeated/missing episode IDs, not identical
future learning curves.

## 12. Recurrent C extension

Build C only after the feed-forward path and toy gates work. Reuse A position and
discard features and add full contract-defined ordered public/private events and
setup fields. Do not substitute B summaries for C history.

Proposed first architecture: width-128 structured GRU state over color/rank/card
indices with shared update weights, invariant global context, and the same shared
action scorer. Initial state is symmetry-fixed. No unrestricted flattened GRU over
ordered color blocks. Exact module shapes need their own recorded design and
equivariance validation before implementation comparisons.

Store each episode event sequence once, with learner-decision offsets and aligned
private draws. Collection uses incremental hidden states reset per episode/seat.
Training starts with full-episode replay from the initial state under current
parameters for each minibatch; group episodes by length and mask padding. Sum
decision surrogates and average over games as in the reference objective. Do not
treat collected old hidden states as current-policy states.

Start with full backpropagation through each episode. If memory/time makes this
impractical, stop and declare a truncated-BPTT/burn-in design as a separate choice;
validate forward replay and gradient consequences. Do not silently truncate history
while retaining a full-history claim. C's practical batch size may differ and must
be reported in equal-data/time comparisons.

Use the same fixed critic contract initially. Ordinary real-game evaluation and
exact toy replay are available after testing. Counterfactual AIVAT/search against
C requires validated history reconstruction and branch-local state; it is not a
prerequisite for the initial recurrent training comparison.

## 13. Implementation sequence and acceptance gates

Each stage is future work. Complete bounded relevant checks before moving on;
none of the following tests has been run as part of drafting this plan.

| Stage | Deliverable | Acceptance evidence |
|---|---|---|
| 1. Production contract | A/B features, legal distribution, separate critic, serialization | Production masks equal engine; nontrivial color/rank and stabilizer tests; hidden-state independence; full/incremental evidence agreement; scalar/batch agreement |
| 2. Reference learner | Single-worker collector, MC policy-gradient mode, PPO loss, game accounting | Frozen old/new probabilities agree; known terminal outcomes and draw/pass/skip/reverse cases; unequal-length weighting; detached baseline; replayable seeds |
| 3. Parallel collector | Persistent workers, batched games, bounded transfers | Same logical manifests/streams at 1 versus multiple workers; no missing/duplicate games; no completion-order selection; bounded memory |
| 4. Operational support | MLflow, boundary checkpoints, failure handling | Save/load and interrupted continuation; logging outage recovery; worker retry; cross-device short continuation |
| 5. Exact toy gates | Tabular and neural sampled/exact paths | Raw network gradients agree with exact expectations within predeclared game-level uncertainty; five-seed learning gates below |
| 6. Systems pilot | Per-machine worker/device configuration and real rate | Integrated throughput/memory report and frozen run budgets; can precede completed learning gates without strength claims |
| 7. Full-UNO B | From-scratch league learning | DEV-selected/final checkpoints and reproducible improvement across seeds under a frozen evaluation rule |
| 8. A comparison | Shared-position ablation | Equal-data and equal-time reports, same critic/evaluation protocol, all seed outcomes |
| 9. C comparison | Structured recurrent learning | Replay/equivariance/sequence tests followed by matched resource comparisons |
| 10. Frozen report | Registered fresh EVAL and bounded report card | Per-cell uncertainty, seed variation, baseline comparisons, costs, limitations, next decision |

Toy order: `TOY` against PlayFirstRandom, then Heuristic; extend to `TOY_FULL` and
`TOY2_SKIP`; reserve `TOY2_FULL` for milestones. Test the same neural parameters
against exact expected gradients, not unrelated tabular logits. At collection
weights the PPO gradient should match the unregularized sampled policy-gradient
reference; repeated clipped updates need not match the original objective gradient.

Retain the existing tabular gate: five training seeds, each within **0.5 score
points** of its same-seat fixed-opponent BR at both seats, within a declared cap.
The neural gate compares sampled optimization to exact-gradient optimization of
the same architecture and also reports the BR gap. Exact optimization is an
achieved comparator, not a certified architecture optimum. Freeze its numerical
tolerance and cap after the bounded toy diagnostic, before the five-seed gate;
if it cannot be justified, remain in diagnostics rather than waive the gate.

Before the full-UNO multi-seed study, record a practical minimum gain and a target
CI half-width. Initial discussion candidates are a one-score-point gain over the
Heuristic reference on the same league and a half-point aggregate paired 95% CI;
they are not accepted feasibility claims. Use DEV variance and cost to decide
whether these are sensible, then freeze the rule. A weak result must not cause a
post-hoc smaller threshold or favorable baseline substitution.

Likely future ownership: `uno/rl/` for learner-specific features/models/buffers/
collection/optimization/checkpointing; `uno/experiments/` for bounded commands;
`uno/eval/` for reusable evaluation adapters; mirrored tests under `tests/rl/`.
Keep engine rules in `uno/game/` and exact oracles in their existing modules.
Choose actual command names at implementation; no command in this plan implies
an existing executable learner.

## 14. Remaining decisions and document lifecycle

No further user preference is required to begin implementing a separately
authorized first stage. The following are deliberately measurement-dependent:

| Decision | When resolved | Evidence required |
|---|---|---|
| Worker count, live games, update device | Systems pilot on each machine | End-to-end throughput, memory, probability agreement |
| Game quota and routine data cap | After real collector pilot | Actual learner decisions/game and full-iteration rate |
| Final PPO values | Bounded toy diagnostics, before gate runs | Gradient correctness, stability, learning within declared cap |
| Neural tolerance | Before multi-seed toy gate | Same-architecture exact-gradient comparison and explicit rationale |
| Evaluation panel and practical gain threshold | Before full-UNO study | DEV paired variance and evaluation cost |
| C sequence layout/budget | Before C implementation comparison | Structured recurrence/replay design and memory pilot |
| Overnight duration | After routine run report | Identified reason more data or seeds could answer the next question |

Record diagnostic changes and frozen configurations under `docs/rl/results/`.
Preserve failures and limitations. Link this document from the active RL navigation
when adopting it; update stale open items in the research plan then, without
rewriting historical results. This draft creates only this new plan file.

Technical references used during planning: [Python multiprocessing](https://docs.python.org/3/library/multiprocessing.html)
for process/start-method behavior and [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking)
for parameters, metrics, runs, and artifacts. Implementation must check APIs against
the versions actually pinned in the environment.
