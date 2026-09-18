# RL project

**Status, 2026-09-18:** the learner exists. Pipeline-plan stages 1-5 are
implemented in `uno/rl/` and their acceptance checks pass; the exact-toy gates are
measured in [stage5-toy-gates.md](results/stage5-toy-gates.md). No full-UNO
learning result exists yet. First maximize frozen-league score with PPO from
scratch. Adaptability and robustness are distant follow-up objectives. B is the
first practical actor, A its memory ablation, and C the subsequent recurrent
comparison.

## Next work

Stage 6, the per-machine systems pilot of [pipeline-plan.md](pipeline-plan.md)
section 7: worker count, live games per worker and update device, measured end to
end, before a full-UNO run's budget is frozen. Stages 7-10 (full-UNO B, the A
ablation, the recurrent C comparison and the frozen report card) follow it.
PIMC compression is secondary, with known correctness issues to repair before
interpreting its gap. A distilled warm start is an option only if from-scratch RL
later disappoints.

Commands: `uv run uno rl shapes|smoke|train|resume` for the learner,
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
