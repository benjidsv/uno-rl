# Compute budget — open planning estimates

**Status, 2026-09-18:** a simulation/inference hardware pilot is measured on the
Mac M1 Pro (below); no PPO update, real feature extraction, or the actual
structured/GRU encoder exists yet, so there is still no end-to-end learner rate.
These remain workload calculations, not promised training times or required
epochs. Useful data volume and all budget caps remain open. Measure a bounded
pilot on any other candidate machine the same way before committing to the
multi-seed runs in the [plan](plan.md).

## Units and candidate PPO workload

An update collects fresh on-policy data. An epoch is one optimization pass over
that update's data, not another pass through the environment. More PPO epochs do
not substitute for fresh games. Use completed episodes initially; the nominal
batch target may be slightly exceeded to finish them.

Illustrative configuration: 8,192 learner decisions/update, minibatches of 512,
and 4 PPO epochs/update: 16 minibatches/epoch and 64 optimizer steps/update.
This is a sizing candidate, not a frozen hyperparameter choice. Actor/critic work
must both be included in measured update time.

| Learner decisions/run | Approx. updates | Optimizer steps | Sample presentations (4 epochs) |
|---|---:|---:|---:|
| 100,000 (diagnostic) | 13 | 832 | 425,984 |
| 1,000,000 (initial budget scenario) | 123 | 7,872 | 4,030,464 |
| 10,000,000 (extension scenario) | 1,221 | 78,144 | 40,009,728 |

Rows round updates upward. They do not predict convergence. Exact-gradient toy
optimization has a different cost model: time one complete tree evaluation,
backpropagation, tabulation, and memory footprint separately.

For a rough conversion only, suppose a game contains 20 learner decisions and 40
total decisions. Then 1M learner decisions is about 50,000 games / 2M total
decisions; 10M is about 500,000 games. The older full-UNO correction experiment
reported 41.5 total decisions/game on its profile, which motivates this scale but
is not a measurement of the learner's games. Draw behavior, seat, and opponent
change this ratio. See [10a.2, section 1.3](../foundations/results/10-readiness/10a2-results.md#13-cost-the-brainstorms-hope-is-wrong-and-the-reason-is-the-engine)
for the measured profile and detailed cost discussion.

## Conditional wall-time scenarios

Let `R` be end-to-end learner decisions/second, including collection, features,
inference, and four optimization epochs. Approximate training time is `N / R`.
Evaluation, startup, exact-tree work, and label generation are additional.

| Assumed R (not measured) | 1M decisions | 10M decisions | Five seeds at 10M |
|---|---:|---:|---:|
| 100/s | 2.78 h | 27.78 h | 138.89 h |
| 1,000/s | 16.7 min | 2.78 h | 13.89 h |
| 5,000/s | 3.3 min | 33.3 min | 2.78 h |

These scenarios deliberately span a wide range; none is a forecast for this
machine. Three arms x five seeds multiplies the single-run cost by 15, before
pilots or evaluation. At 1,000/s and 10M decisions/run, that would be 41.67 h
if all arms had the same throughput. Recurrent throughput must be measured
separately; do not assume a fixed 2–5x multiplier.

Measure collection and update time separately as well. For example, if collection
runs at `Rc` learner decisions/s and an update takes `Tu` seconds, then
`time = N/Rc + ceil(N/8192)*Tu`, plus evaluation and startup. Do not double-count
update cost when using an already end-to-end rate.

## Measured pilot: Mac M1 Pro, 2026-09-18

`uv run uno pilot_throughput [games]` (source: `uno/experiments/pilot_throughput.py`).
Single process, single thread, no multiprocessing or vectorized engine, `FULL_UNO`,
1,000 deals/cell x 4 frozen opponents x 2 seats, DEV seeds. Two things are measured,
neither of which is the real learner rate, since no feature extractor or actor
encoder exists yet:

1. **Simulation ceiling** — engine step, observation construction, and one Python
   policy call per decision, no network involved. `PlayFirstRandom` and `Heuristic`
   stand in for the learner's actor (Heuristic finishes games faster, so its
   games/s is higher despite doing more per-decision work — game length, not just
   raw call cost, drives this number).
2. **Synthetic MLP proxy** — a plain 2-hidden-layer, width-128 net (the plan's
   pilot width; not the real structured/GRU encoder), input dim 512, timed
   forward-only and forward+backward+Adam-step, on CPU and MPS, at the candidate
   batch sizes from the workload table above.

| Measurement | Rate |
|---|---:|
| Simulation, engine-only stand-in, total decisions/s (both seats) | ~25,900/s |
| Simulation, engine-only stand-in, learner-seat decisions/s | ~13,000/s |
| Simulation, Heuristic stand-in, total decisions/s (both seats) | ~27,000/s |
| Simulation, Heuristic stand-in, learner-seat decisions/s | ~13,500/s |
| MLP proxy, CPU, batch 8,192, forward-only | ~2,600,000/s |
| MLP proxy, CPU, batch 8,192, forward+backward+step | ~830,000/s |
| MLP proxy, MPS, batch 8,192, forward-only | ~4,440,000/s |
| MLP proxy, MPS, batch 8,192, forward+backward+step | ~1,890,000/s |
| MLP proxy, MPS, batch 512, forward+backward+step | ~68,000/s (CPU: ~342,000/s) |
| Peak RSS over the whole run (sim + both device benches) | ~540 MiB |

**Reading these together:** at width 128, the network math is not the bottleneck
on this machine by roughly two orders of magnitude — single-threaded Python
simulation caps out around 13,000 learner decisions/s, while even a full PPO-style
forward+backward+optimizer step at batch 8,192 clears 800,000/s on CPU alone.
MPS wins decisively at batch 8,192 but loses to CPU at batch 512 (higher per-launch
overhead), so small-batch update steps may prefer CPU on this hardware.

Using the simulation ceiling as a provisional, pre-feature-construction `R` for
the workload table: 1,000,000 learner decisions ≈ 77 s; 10,000,000 ≈ 12.8 min;
five seeds at 10,000,000 (one arm) ≈ 64 min. This omits real feature-tensor
construction (not yet implemented — the current policies score positions
directly, not via a network) and any real encoder cost, and it is a single
Python worker with no batching or multiprocessing across games. **Treat it as
optimistic for collection and irrelevant to the true update cost** until the
actor encoder, feature pipeline, and PPO loop are implemented and timed end to
end; re-run this pilot once they exist.

## Evaluation and distillation can dominate

Four opponents x two seats x 4,000 deals/cell = 32,000 games/checkpoint. At a
hypothetical 2 ms/game this is 64 s; at 20 ms/game it is 10.7 min. These are cost
scenarios, not a recommendation to evaluate that frequently or a precision claim.
Use a smaller fixed DEV diagnostic during training and select the confirmatory
budget from paired-difference variance. Training-seed replication adds another
factor when every trained policy is evaluated.

Historical column-free PIMC took about 20–28 ms/game on its measured profile
([10a.2, section 4](../foundations/results/10-readiness/10a2-results.md#4-a5a-a-column-free-pimc-row-on-full-uno)).
At that rate, 32,000 games would take about 10.7–14.9 min. This excludes any new
opponent/architecture effect and cannot price an attack that repeatedly queries
PIMC inside search.

For distillation, let `L` be the number of labeled decisions, `q` teacher queries/s,
and `d` student decisions/s during collection. Label/collection cost is roughly
`L/q + L/d` when stages are separate, plus fitting. For 100,000 labels, 100
queries/s means 16.7 min of labeling; 1,000/s means 1.7 min. These query rates
are hypothetical. Teacher rollout generation and subsequent relabeling may repeat
search work, so instrument both. Batched teacher leaf evaluation is not the same
as fully batched state construction.

Candidate student fitting budget: at most 40–60 epochs with held-out early
stopping, matching the scale of the old smoke command. This is an experiment cap,
not an expected number needed for successful compression. Use several training
seeds only after the corrected single-run diagnostic is informative.

## Memory and pilot measurements

At float32, a compact rollout of `N` decisions and `F` stored feature scalars costs
`4*N*F` bytes before masks, actions, probabilities, critic inputs, and activations.
For 8,192 decisions and 512 scalars this is 16 MiB. Duplicating those features
across 50 actions would make just that array 800 MiB. These are illustrative
shapes; store shared context once. Recurrent sequence activations and exact-tree
caches require separate measurements.

Run a short warm-up and several bounded collection/update batches. Record:

- CPU/GPU model, backend, thread/worker counts, precision, batch dimensions;
- learner and total decisions/s, games/s, game lengths by opponent/seat;
- simulation, feature, inference, update, and evaluation time separately;
- host/device peak memory, checkpoint size and loading time;
- recurrent replay/burn-in cost and teacher queries/s when applicable.

Then replace scenarios with measured estimates, choose a diagnostic data cap, and
predeclare extension/stop rules. Plateauing DEV score triggers diagnosis before
allocating ten times more data. There is currently no defensible estimate of
"epochs until strong UNO": data requirements are one of the first measurements.

The pilot above covers simulation and a synthetic network proxy on one machine.
Still open: real feature-tensor construction cost and memory, the actual
structured/GRU encoder's throughput (versus the plain-MLP proxy), true update
time inside a PPO loop, checkpoint size/load time, recurrent replay/burn-in
cost, teacher queries/s, and the same measurements on any other candidate
machine.
