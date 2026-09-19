# Stage 6 — M1 systems pilot

**Status, corrected and completed 2026-09-19:** the original parallel-collector
conclusion was invalid for the reasons documented below. A replacement sweep
with phase-specific Torch thread budgets, synchronized MPS timing, stage timing,
and memory sampling selected the single-process path. Stage 7's execution setting
is frozen below; its data/time budget and every learner-strength claim remain open.

**Question.** What CPU worker/device configuration and full-UNO rate can the
implemented feed-forward B PPO pipeline sustain on this Mac before freezing the
first routine-run budget?

**Decision rule.** Retain a setting only if it has a valid CPU-collection path,
stays comfortably below the 10 GiB Mac application limit, and does not lose
end-to-end throughput to the next worker count. Confirm the choice with a
roughly one-minute fixed-policy iteration and a roughly ten-minute run including
the ordinary DEV evaluation, tracking, and checkpoint boundaries. This is a
systems measurement, not a learning-strength gate.

## Configuration and provenance

Measured 2026-09-18 on macOS arm64 (the user identified the device as an Apple
M1). `FULL_UNO`, B actor (899,713 parameters), fixed four-member `rulebot`
league, FP32, 4 actor plus 4 critic PPO epochs, minibatch 512, and 64 live games
per CPU collector were held fixed. Collection used the frozen CPU actor replica;
CPU was the update device unless stated otherwise. The one-minute pilot used 32
games/cell (256 games). Raw artifacts are under the ignored `logs/rl/` paths
printed by the commands.

## Result

| Measurement | CPU, 1 worker | CPU, 2 workers | MPS, 1 worker |
|---|---:|---:|---:|
| Short fixed-manifest end-to-end learner decisions/s | 144.4 | 141.0 | 140.4 |
| Short collection learner decisions/s | 1,459.9 | 2,118.3 | 1,487.9 |

Two workers reduced collection time but did not improve total rate because PPO
update dominated; it therefore fails the plan's 10% scaling rule. MPS was also
slower than CPU at the unchanged 512-decision minibatch. Do not extrapolate the
synthetic large-batch MPS proxy to this actual workload.

The retained one-worker CPU point processed 7,970 learner decisions and 256
games in 56.05 s: **142.2 end-to-end learner decisions/s**, 4.57 games/s, and
289.2 total engine decisions/s. Collection took 6.93 s (1,160.7 learner
decisions/s); update took 49.12 s. Mean inference batch was 30.7, drain was 2.09
s, compact rollout storage was 88.9 MiB, and coordinator peak RSS was 1.92 GiB.
RSS is a boundary sample; it is not a total physical-unified-memory measurement.

The integrated run was:

```sh
uv run uno rl train FULL_UNO B 0 80000 1 64 16 cpu
```

It completed five whole updates, 80,485 learner decisions, 2,560 games, and
162,163 total engine decisions in 527.6 active seconds. Per update collection
was 7.1–8.4 s and update was 88.8–97.7 s; scheduled 16-deals/cell DEV evaluations
and checkpoint boundaries completed normally. The final DEV value is deliberately
not interpreted: this was one DEV training seed and a systems-health run.

## Original interpretation (superseded)

The original interpretation selected **one CPU collector, 64 live games, CPU
optimization**, and 64 games/cell (the existing training default) for the first B diagnostic. The 32-games/cell
one-minute point produced approximately the plan's 8,192-decision target (7,970),
so no quota adjustment is justified from this one policy alone. For planning,
use 140 end-to-end learner decisions/s before evaluation; the integrated run's
rate is 152.6 active decisions/s but is too short and evaluation-panel-specific
to replace the conservative figure. This would price 100k decisions at about 12
minutes of active training and 1M at about 2.0 hours, before a production-sized
evaluation budget. The one-process timing remains a useful provisional planning
point; the selection decision is withdrawn.

## 2026-09-19 audit and next decision

The worker count was real and each worker was capped at one Torch thread. During
PPO, however, only the coordinator works and it used six Torch threads on this
machine. A 1,218-decision diagnostic took 7.56 s to update with six threads versus
11.90 s with one. The production-shaped runs spent roughly 7–12 s collecting and
89–102 s updating, so a single multi-core coordinator process is the expected
steady-state Activity Monitor view. This supports PPO/model computation—not the
engine or coordinator merge—as the main end-to-end bottleneck.

The parallel implementation queued eight episodes per task. Because each task is
collected independently, that also capped a worker's actor batch at eight despite
the nominal 64-live-game setting. Raising the bundle to 64 is not an architectural
fix: 512 games produce exactly eight tasks for eight workers, and a 128-game
systems point produces only two tasks. Those cases remove dynamic balancing or
leave most workers idle. More generally, a manifest with fewer than `workers ×
live_games` episodes cannot sustain the requested per-worker width; the corrected
pass must report realized widths rather than assume them.

The result queue was unbounded and coordinator ingestion retained per-decision
views into payload arrays. Those are memory-design defects. By contrast, a local
measurement of 4,010 full-UNO decisions put coordinator merge plus final assembly
at 0.021 s versus 2.43 s collection. The claim that per-decision merging caused
the observed stall is therefore unsupported. High memory pressure from multiple
Python/Torch processes, rollout payloads, serialization buffers, final assembly,
and PPO allocations is plausible, especially if another run remained alive, but
it was not measured as the root cause.

The next bounded pass must use the instrumented simulation, feature, actor
inference, serialization, result-queue wait/backpressure, coordinator assembly,
critic-value, actor-update, and critic-update timings. It must also retain the
bounded result queue, record transfer bytes and sampled aggregate RSS/macOS memory
pressure, and report the realized inference-batch distribution. A redesigned
worker that decouples live-game width from bounded result chunks remains an open
implementation choice; do not substitute a bundle-64 run for it. The runtime
bundle knob was removed from the hashed experiment configuration, restoring the
completed run's `564a372b249cbd0a` hash and strict-resume compatibility.

## Corrected execution sweep

The corrected sweep used the same frozen initial B policy and logical manifest at
every point: `FULL_UNO`, 16 games per opponent/seat cell (128 games, 3,673 learner
decisions), 64 live games, minibatch 512, four actor and four critic epochs, one
warm-up plus two retained samples. Collection was fixed to one Torch thread while
the CPU PPO thread count varied. MPS timing was explicitly synchronized before
the clock stopped. Thread counts are runtime provenance, not hashed algorithm
configuration.

| Update device / threads | Iteration s | Collection s | Update s | Learner decisions/s |
|---|---:|---:|---:|---:|
| CPU / 1 | 40.69 | 2.55 | 38.15 | 90.3 |
| CPU / 2 | 30.32 | 2.36 | 27.97 | 121.1 |
| CPU / 4 | 25.65 | 2.45 | 23.21 | 143.2 |
| CPU / 6 | 24.44 | 2.41 | 22.03 | 150.3 |
| MPS / synchronized (4 CPU-side threads) | 25.42 | 2.31 | 23.10 | 144.5 |

Four CPU threads improved end-to-end rate by 18.2% over two. Six improved only
5.0% over four, below the predeclared 10% continuation rule, so the CPU ladder
stopped before eight threads. Synchronized MPS was 0.9% faster than CPU-4 and
3.9% slower than CPU-6: all three are operationally tied at minibatch 512. The tie
goes to CPU-4 because it uses fewer CPU threads than CPU-6 and MPS showed no
material end-to-end advantage. This does not claim CPU is universally better;
larger minibatches would be a separate PPO/learning choice, not a systems-only
selection.

Sampled CPU process RSS was about 1.38–1.62 GiB across the sweep, far below the
10 GiB application budget. MPS process RSS was 0.52–0.64 GiB, but that is not a
comparable unified-memory total; its system-free snapshot was 32% versus 62–64%
for the CPU points. No MPS or CPU-1/2/6 sample grew swap. The two CPU-4 samples
each observed 8 MiB of whole-system swap movement against roughly 3 GiB already
allocated; this is too small and too system-wide to attribute to the run.

Commands (the last two arguments are collection and update Torch threads):

```sh
uv run uno rl systems FULL_UNO B 1 64 cpu 16 2 512 1 1
uv run uno rl systems FULL_UNO B 1 64 cpu 16 2 512 1 2
uv run uno rl systems FULL_UNO B 1 64 cpu 16 2 512 1 4
uv run uno rl systems FULL_UNO B 1 64 cpu 16 2 512 1 6
uv run uno rl systems FULL_UNO B 1 64 mps 16 2 512 1 4
```

Raw ignored artifacts are the corresponding timestamped `logs/rl/systems_*`
directories ending in `1789773567`, `1789773695`, `1789773792`, `1789773876`,
and `1789773961`.

## Integrated confirmation and final systems decision

The selected CPU-4 execution setting completed the ordinary tracked training
path, including DEV evaluations and best/latest/final checkpoint boundaries:

```sh
uv run uno rl train FULL_UNO B 0 40000 1 64 16 cpu 1 4
```

It completed three whole updates, 51,246 learner decisions, 1,536 games and
103,419 engine decisions in 322.25 active collection/update seconds. Per-update
collection was 9.22, 9.93 and 10.10 s; PPO was 93.71, 103.31 and 95.98 s. The
means were 9.75 s collection and 97.66 s update. Even the impossible ideal of
zero-cost collection would improve mean iteration throughput by only 9.98%, just
below the plan's 10% threshold. A real parallel collector must retain nonzero
collection, scheduling, serialization and assembly costs, so further worker-count
tuning is not justified by these measured mean timings. This is a stopping-rule
decision for the present workload, not a universal bound for later policies or
different batch sizes.

Use **one single-process CPU collector, 64 live games, one collection Torch
thread, and four CPU PPO Torch threads**, with minibatch 512 unchanged. Retain
**140 learner decisions/s** as the conservative pre-evaluation planning rate; the
integrated active rate was about 159/s but is only one short, changing-policy run.
The run's DEV-selected update was 2 and final DEV score was 0.1719, but the panel
had only 16 deals/cell and the run was a systems health check, so neither value is
interpreted as learning evidence. The raw ignored artifact is
`logs/rl/FULL_UNO_B_seed0_1789774153/` (config hash `fc08fa5b4eba9f71`).

The bounded parallel queue and transfer instrumentation remain useful maintenance,
but the eight-game-task parallel collector is not validated or selected for Stage
7. Redesign it only if a future workload makes collection large enough that the
theoretical end-to-end gain can clear the scaling threshold.

## Limitations and next decision

The sweep covers one M1, one feed-forward architecture and minibatch 512. It does
not establish learning improvement, suitable PPO hyperparameters, the production
DEV panel size, a full-run data cap, recurrent throughput, or another machine's
setting. Proceed to Stage 7 only after explicitly choosing its data/time budget;
do not reinterpret this systems selection as a learning result.
