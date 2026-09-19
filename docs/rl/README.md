# RL project

**Status, 2026-09-19:** the learner exists. Pipeline-plan stages 1-5 are
implemented in `uno/rl/` and their acceptance checks pass; the exact-toy gates are
measured in [stage5-toy-gates.md](results/stage5-toy-gates.md). The first
[full-UNO B pilot](results/pilot-notes.md) improved from 0.0313 to 0.4287 DEV
score over 501,367 learner decisions, establishing basic from-scratch learning but
not final strength. Adaptability and robustness remain distant follow-up
objectives. B is the first practical actor, A its memory ablation, and C the
subsequent recurrent comparison.

## Next work

Stage 6 is complete for the M1: use the single-process collector with 64 live
games and one collection Torch thread, then CPU PPO with four Torch threads at
the unchanged 512-decision minibatch. The corrected sweep found CPU-4, CPU-6,
and synchronized MPS within 5% end to end; the tie goes to CPU-4. On the longer
integrated run, even free collection could not meet the plan's 10% parallel-
scaling threshold. See the [Stage 6 result](results/stage6-m1-systems.md). The
[DEV-panel sizing](results/stage7-dev-panel-sizing.md) selected 256 deals/cell
for coarse quarter-budget monitoring and reserved 5,760 for one-off precision.
The one-hour B pilot is complete; next fix its diagnostic logging defects and
validate the final actor once on the 5,760-deal panel before choosing the next
training intervention. The
[minibatch calibration](results/stage7-minibatch-calibration.md) retained
CPU-4/minibatch-512 without a learning follow-up. Stages 8-10 (the A ablation,
recurrent C comparison and frozen report card) follow Stage 7.
PIMC compression is secondary, with known correctness issues to repair before
interpreting its gap. A distilled warm start is an option only if from-scratch RL
later disappoints.

Commands: `uv run uno rl shapes|smoke|systems|devsize|pilot|train|resume` for the learner,
`uv run uno rl_toy gradient|tabular|neural|all` for the exact-toy gates.

## Read next

| Task | Document |
|---|---|
| agreed objective, pipeline, phase gates, and open choices | [plan.md](plan.md) |
| execution plan the learner implements: topology, iteration, PPO, gates | [pipeline-plan.md](pipeline-plan.md) |
| actor fields, symmetry, memory, critic/attacker boundaries | [observation-contract.md](observation-contract.md) |
| rough updates/epochs, conditional time estimates, pilot measurements | [compute-budget.md](compute-budget.md) |
| reproduced readiness defects and review checks | [readiness-review.md](readiness-review.md) |
| foundation handoff and established evidence | [starting-point.md](starting-point.md) |
| learner measurements (none yet) | [results/index.md](results/index.md) |
| foundation maintenance | [foundation README](../foundations/README.md) |

## Active boundaries

- Equal-weight frozen league, balanced seats, opponent fixed per episode and hidden
  from the actor; per-member results accompany the primary aggregate.
- Ordinary terminal returns plus a baseline first. GAE, shaping, corrections, and
  adaptive opponents need separate evidence.
- Compare representations at equal data and wall time. K1-style A is an ablation,
  not an established sufficient representation or a certified ceiling.
- DEV for tuning/selection, fresh registered EVAL for frozen claims. PIMC is an
  achievable benchmark; weak attack lower bounds do not establish safety.

Keep this page as current working memory. Record measurements under `results/`,
link them from its index, and replace completed active-state prose rather than
appending project history. Plans and review findings are not learner results.
