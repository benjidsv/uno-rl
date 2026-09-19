# Stage 7 preflight — PPO minibatch calibration

**Status, preregistered 2026-09-19:** pending execution. This study may change
the Stage 7 minibatch and update device, but it is not a full-UNO strength result
and does not choose Stage 7's eventual production data budget. Run Gate A first;
run Gate B only if a candidate passes Gate A.

## Question

Can a larger PPO decision minibatch improve total collection-plus-update
throughput for the implemented structured actor/critic without materially
reducing short-run learning per fresh decision or per wall time?

“Update throughput” has two recorded meanings:

- fresh rollout decisions divided by total PPO update time, which prices one
  synchronous iteration; and
- actor and critic sample presentations per update-stage second, which measures
  repeated optimization work.

The second count is rollout rows times completed epochs. It deliberately excludes
the additional no-gradient KL pass and critic baseline evaluation, so it is a
stable workload proxy rather than a count of every model forward. End-to-end
learner decisions/s remains the systems selection metric.

## Gate A — frozen-policy compute sweep

Hold fixed `FULL_UNO`, B, seed 0, the four-member rulebot league, 32 games per
opponent/seat cell (256 games), one CPU collector, 64 live games, one collection
Torch thread, four actor and four critic epochs, and two retained samples after
one excluded warm-up. Sweep minibatches 512, 1,024 and 2,048 on CPU-4 and
synchronized MPS. Each command starts from the same initial policy and logical
episode manifest; no score is interpreted.

```sh
uv run uno rl systems FULL_UNO B 1 64 cpu 32 2 512 1 4
uv run uno rl systems FULL_UNO B 1 64 cpu 32 2 1024 1 4
uv run uno rl systems FULL_UNO B 1 64 cpu 32 2 2048 1 4
uv run uno rl systems FULL_UNO B 1 64 mps 32 2 512 1 4
uv run uno rl systems FULL_UNO B 1 64 mps 32 2 1024 1 4
uv run uno rl systems FULL_UNO B 1 64 mps 32 2 2048 1 4
```

A larger minibatch is a compute candidate only if all retained samples complete
all four actor and four critic epochs, contain no non-finite metrics, keep sampled
aggregate RSS below 10 GiB, grow whole-system swap by no more than 256 MiB, and
improve mean end-to-end learner decisions/s by at least 15% over minibatch 512 on
the same device. If several pass, retain only the fastest. Select MPS over CPU at
the retained minibatch only if it is at least 10% faster end to end; otherwise the
tie goes to CPU. If nothing passes, retain CPU-4/minibatch-512 and do not run Gate
B.

The memory thresholds are operating limits, not causal diagnoses. Aggregate RSS
can double-count shared pages, and on MPS it is not total physical unified-memory
usage; system free percentage and swap are therefore reported alongside it.

## Gate B — matched short learning diagnostic

If Gate A retains a candidate, compare it with CPU-4/minibatch-512 on training
seeds 0–4. Hold fixed `FULL_UNO`, B, the league, 64 games per cell, 64 DEV deals
per cell, the one-process/64-live-game collector, one collection Torch thread,
four actor and four critic epochs, a 100,000-decision soft data cap, and a
600-second soft wall cap. The fixed panel is DEV and is shared across variants;
it is not a fresh EVAL claim.

Baseline commands:

```sh
for seed in 0 1 2 3 4; do
  uv run uno rl train FULL_UNO B "$seed" 100000 1 64 64 cpu 1 4 512 600
done
```

Candidate commands, replacing `DEVICE` and `MINIBATCH` with Gate A's one retained
pair:

```sh
for seed in 0 1 2 3 4; do
  uv run uno rl train FULL_UNO B "$seed" 100000 1 64 64 DEVICE 1 4 MINIBATCH 600
done
```

Time and data caps are soft at complete-iteration boundaries. Preserve every
printed artifact directory. Use `metrics.csv` timestamps and learner-decision
steps to make two paired comparisons:

1. equal wall time: for each seed pair, use the latest scheduled DEV point from
   each run not later than their common completed wall time; and
2. equal data: use the latest scheduled DEV point from each run not beyond their
   common learner-decision count.

Report actual wall and active seconds, fresh learner decisions, games, total
engine decisions, completed epochs, early stops, and the DEV league mean for all
ten runs. For each comparison, form the five paired candidate-minus-baseline DEV
differences and a two-sided 95% *t* interval over training seeds.

Adopt the compute candidate for the subsequent Stage 7 diagnostic only if its
paired 95% CI lower bound is at least -0.02 score in both the equal-wall and
equal-data comparisons, no run has a non-finite metric or systematic premature
KL stopping, and its realized systems advantage remains at least 15% at matched
work. Otherwise retain minibatch 512. An interval crossing -0.02 is
inconclusive, not evidence of equivalence, and defaults to 512.

## Interpretation limits and next decision

This is deliberately a bounded implementation-calibration study: five short DEV
runs can reject a clearly harmful minibatch, but cannot establish convergence,
final policy strength, or a globally optimal PPO setting. Do not tune the -0.02,
15%, or device thresholds after seeing these runs. Once the decision is recorded
here, use the selected configuration to choose Stage 7's explicit production
data/time budget and separately freeze its practical strength rule and DEV panel
precision.
