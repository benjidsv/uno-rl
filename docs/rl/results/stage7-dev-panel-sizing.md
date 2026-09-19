# Stage 7 preflight — DEV panel sizing

**Status, trained-checkpoint validation 2026-09-19:** 256 deals/cell remains the
frozen coarse monitoring panel. The initialized actor suggested about 5,760
deals/cell for a paired half-width of 0.005, but the trained final actor had higher
paired variance: 8,192 deals/cell achieved 0.00550 and estimated 9,926. The
predeclared rounding rule therefore requests one final 9,984-deal validation.
These are DEV diagnostic seeds, not a frozen EVAL claim.

## Question and decision rule

How many deals per opponent/seat cell are needed for a paired 95% CI half-width
of at most **0.005 score** when comparing the B policy with the same-seat
Heuristic reference against the frozen four-member rulebot league?

Evaluate one nested maximum panel on the DEV diagnostic partition beginning at
deal 900,000, then read its first 64, 128 and 256 deals per cell. Group all eight
opponent/seat outcomes sharing one deal before computing the paired standard
error. Choose the smallest tested size whose observed paired half-width is at
most 0.005. If 256 does not meet the target, compute

`ceil(256 * (observed_halfwidth / 0.005)^2)`,

round that estimate up to the next 64 deals, and validate it in a second bounded
diagnostic run before freezing the panel. Do not shrink the choice later because
a trained checkpoint happens to have a favorable interval.

## Configuration and command

Use `FULL_UNO`, feed-forward B initialized with training seed 0, the frozen
rulebot league, Heuristic as the learner-row reference, balanced seats, raw
terminal score, temperature 1, 64 live actor-evaluation games, and one Torch
thread. The diagnostic block is disjoint from training and from the fixed
scheduled-selection panel.

```sh
uv run uno rl devsize FULL_UNO B 0 256 0.005 1
```

The command evaluates the requested maximum only once for each policy and derives
70% and 85% nested-prefix rows from the same samples. `tqdm` reports completed
actor and Heuristic games. It saves a semantic JSON result and the
per-deal/per-cell actor and reference scores under its printed `logs/rl/devsize_*`
directory.

## Required report and limitation

Report actor, Heuristic and total runtime; score and paired difference; standard
error, 95% half-width and CI; required-deal estimate; every per-cell mean and
paired difference; deal range; policy provenance; and behavior summaries.

The initialized policy is available before the learning pilot but may not have
the trained policy's variance. After the one-hour pilot, rerun the same command
with its actor-only export as the final positional argument at the already chosen
panel size. Expand the panel if that validation misses 0.005; do not interpret
either sizing run as evidence of learning.

## Result

The final nested run used 8,192 diagnostic deals/cell (65,536 games for each
policy). The initialized actor scored 0.0321, Heuristic scored 0.5010, and their
paired difference was -0.4689. That strength gap is expected from an untrained
actor and is not a learning result. The historical 37.7% Draw-action field cannot
be interpreted as voluntary drawing because it combined forced and optional Draw
actions.

| Deals/cell | Paired 95% half-width | Estimated deals for 0.005 |
|---:|---:|---:|
| 256 | 0.0237 | 5,766 |
| 512 | 0.0168 | 5,778 |
| 1,024 | 0.0119 | 5,779 |
| 2,048 | 0.0083 | 5,599 |
| 4,096 | 0.0059 | 5,697 |
| 5,760 | 0.0050 | 5,737 |
| 8,192 | 0.0042 | 5,719 |

The stable required-size estimates and near-`1/sqrt(n)` decay show ordinary
sampling variance, not an evaluator failure. The 8,192-deal run took 1,268.4 s
for the actor and 54.0 s for Heuristic. At that rate, 5,760 deals/cell costs
about 15 minutes for one paired evaluation, while 256 deals/cell costs about 40
seconds for the actor.

## Decision

The original 0.005 target remains appropriate for a frozen one-off policy
estimate but fails the routine-monitoring cost constraint: using it at eleven
checkpoint boundaries would spend roughly 2.7 hours per seed on actor evaluation
alone. For the first pilot, use the same fixed 256-deal scheduled panel at the
initial policy, quarter-data milestones, and final policy. Five evaluations cost
about 3.3 minutes and are intended to detect a coarse improving, flat, unstable,
or collapsing curve—not resolve a one-point gain.

The frozen pilot preset is:

```sh
uv run uno rl pilot 0
```

It uses a 500,000-decision soft cap and 3,600-second soft wall cap, so whichever
limit is reached first ends the run. It creates a tagged local MLflow run in the
`uno-v5-rl` experiment. One Ctrl-C requests a graceful stop after the current
complete update, followed by final evaluation, checkpointing, summary writing and
tracking closure; a second Ctrl-C aborts immediately. The final boundary reuses a
scheduled evaluation rather than replaying the same panel. If the pilot is
promising, validate its frozen actor on the separate diagnostic panel before
setting the multi-seed strength protocol.

During incremental sizing, changing the requested maximum changed logical
episode IDs and therefore stochastic policy streams. The final 8,192-deal run is
internally nested, so the table and decision above remain valid. Future diagnostic
panel IDs are now width-independent, preserving streams when a panel is extended.

## Trained-checkpoint validation

The completed pilot's final actor (update 38, 501,367 learner decisions, checksum
`c3fc7243e0e1f97b83e0747607fa656b`) was evaluated on 8,192 diagnostic deals per
cell, paired against the same-seat Heuristic reference. The run covered deals
900,000–908,191 and saved its semantic result and per-deal scores under
`logs/rl/devsize_FULL_UNO_B_seed0_1789827026/`.

| Deals/cell | Actor score | Heuristic score | Paired difference | Paired 95% half-width | Estimated deals |
|---:|---:|---:|---:|---:|---:|
| 5,734 | 0.4439 | 0.4999 | -0.0561 | 0.00655 | 9,840 |
| 6,963 | 0.4422 | 0.4997 | -0.0575 | 0.00596 | 9,887 |
| 8,192 | 0.4420 | 0.5013 | -0.0593 | 0.00550 | 9,926 |

At the maximum prefix, the actor's own nominal 95% interval was
`[0.4381, 0.4458]`; the Heuristic interval was `[0.4973, 0.5052]`, and the paired
actor-minus-Heuristic interval was `[-0.0648, -0.0538]`. Thus the strict paired
precision target was narrowly missed, but the substantive DEV conclusion is not
ambiguous: this checkpoint is about 5.9 score points below the reference. Every
opponent/seat cell favored Heuristic, with cell differences from -0.0500 to
-0.0706. This is descriptive DEV evidence, not a multi-seed or EVAL claim.

The trained actor took 824.2 seconds and Heuristic 55.6 seconds. The required-size
estimates were stable across the three prefixes, so round 9,926 upward to the next
64-deal boundary, 9,984, and validate that nested maximum:

```sh
uv run uno rl devsize FULL_UNO B 0 9984 0.005 1 \
  logs/rl/FULL_UNO_B_seed0_mb512_cpu_1789786184/checkpoints/final_actor.pt
```

This reruns the panel and should take about 18 minutes at the observed rate. It is
needed only to satisfy the predeclared 0.005 sizing rule; the current interval is
already sufficient for the qualitative strength decision.

The corrected behavior counters also reject the proposed Draw-penalty diagnosis.
Forced draws occupied 29.55% of learner decisions. Optional draws were available
on 59.87% of decisions, but the actor chose Draw on only 0.0056% of those
opportunities, versus 0% for Heuristic. The earlier approximately 30% field had
mixed forced and optional draws; this trained actor is not losing six score points
by routinely declining playable cards.
