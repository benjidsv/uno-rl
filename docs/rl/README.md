# RL project

**Status, 2026-09-19:** the learner exists. Pipeline-plan stages 1-4 are
implemented in `uno/rl/`. The [Stage 5 exact-toy study](results/stage5-toy-gates.md)
validated raw gradient agreement, but its retained tabular gate passed only 3/5
seeds against PlayFirstRandom and no neural tolerance was predeclared; the Stage 5
exit gate is therefore open. The first
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
for coarse quarter-budget monitoring. The trained final actor's 8,192-deal
validation clearly established a 0.0593 paired deficit to Heuristic but narrowly
missed the strict 0.005 half-width target; the sizing rule requests a final nested
extension to 9,984 deals/cell. The one-hour B pilot and its diagnostic
logging/resume follow-up are complete. The 9,984-deal extension is optional
precision closure rather than an architecture gate. In the frozen
[Stage 8 A/B ablation](results/stage8-a-ablation.md), A seed 0 was 0.01276 ahead
of B on the direct paired diagnostic panel, but its interval crossed the
predeclared one-point practical boundary. The [extended B diagnostic](results/stage8-b-extended-diagnostic.md)
reduced the pre-fix Heuristic deficit to about 0.012 at 9,984 deals/cell, but did
not reach parity; it remains a one-seed schema-2 DEV result. Restart matched A/B
seeds 0–4 under schema 3 before choosing an arm. The direct comparison panel is
frozen at 5,248 deals/cell. The
[minibatch calibration](results/stage7-minibatch-calibration.md) retained
CPU-4/minibatch-512 without a learning follow-up. Stages 8-10 (the A ablation,
recurrent C comparison and frozen report card) follow Stage 7.
PIMC compression is secondary, with known correctness issues to repair before
interpreting its gap. A distilled warm start is an option only if from-scratch RL
later disappoints.

Commands: `uv run uno rl shapes|smoke|systems|devsize|pilot|ablation|train|resume` for the learner,
`uv run uno rl_toy gradient|tabular|neural|all` for the exact-toy gates.

## Read next

| Task | Document |
|---|---|
| agreed objective, pipeline, phase gates, and open choices | [plan.md](plan.md) |
| execution plan the learner implements: topology, iteration, PPO, gates | [pipeline-plan.md](pipeline-plan.md) |
| actor fields, symmetry, memory, critic/attacker boundaries | [observation-contract.md](observation-contract.md) |
| metric names, units, weighting, aggregation, and uncertainty | [metrics_contract.md](metrics_contract.md) |
| rough updates/epochs, conditional time estimates, pilot measurements | [compute-budget.md](compute-budget.md) |
| reproduced readiness defects and review checks | [readiness-review.md](readiness-review.md) |
| foundation handoff and established evidence | [starting-point.md](starting-point.md) |
| learner measurements | [results/index.md](results/index.md) |
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
