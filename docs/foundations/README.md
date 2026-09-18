# Pre-RL foundations: Steps 1-10

**Status:** completed and handed off to the RL project on 2026-09-18. These
foundations remain authoritative where cited, but this directory is not the default
entrypoint for RL work. Start active work in [`../rl/README.md`](../rl/README.md).

This project built and validated the UNO engine, exact toy games, best responses,
CFR, symmetry reduction, low-variance evaluation, PIMC/search, belief diagnostics,
training-target components, attack instrumentation, and a frozen rulebot league.

## Route by question

| Question | Read |
|---|---|
| What should the engine or an algorithm do? | the matching numbered section of [`spec.md`](spec.md) |
| What are the current Steps 1-10 numbers? | the matching section of [`results/index.md`](results/index.md) |
| How was a number produced? | the result file linked from the index |
| Why was a design chosen or rejected? | the matching section of [`rationale.md`](rationale.md) |
| What was originally planned? | [`history/`](history/) only when historical intent matters |
| Which foundation facts constrain learning? | the curated RL handoff in [`../rl/starting-point.md`](../rl/starting-point.md) |

Do not read the specification, rationale, or complete results index by default.
Find headings or terms first:

```sh
rg -n '^#{1,4} ' docs/foundations/spec.md \
  docs/foundations/rationale.md docs/foundations/results/index.md
rg -n -i '<topic>' docs/foundations
```

## Specification map

- Sections 1-13: game parameters, rules, information levels, invariants, named
  rulesets, and baseline agents.
- Sections 14-22: exact solvers, estimators, compiled trees, CFR, symmetry,
  sizing, coarse keys, and edge attribution.
- Sections 23-26: AIVAT, critics, PIMC/value networks, search, and beliefs.
- Sections 27-30: seed/report contracts, training targets, shaping, belief rungs,
  and the column-free row.
- Section 31 delegates learner observations to the RL observation contract.
- Sections 32-33: attack instrumentation and the Step 10C rulebot package.

## Evidence map

- `results/03-08-foundations/`: engine and agent checks, exact ladders, CFR,
  symmetry, larger toys, and foundational reviews.
- `results/09-evaluation/`: paired evaluation, memory diagnostics, AIVAT, PIMC,
  continuation search, beliefs, and budget studies.
- `results/10-readiness/`: the Step 10 review and repairs, report card, training
  targets, attack calibration, figures, and frozen league.
- `results/index.md`: current headline measurements and corrections across all
  three groups.

Foundation result files are historical evidence. Preserve their original question,
configuration, and limitations; record later corrections explicitly instead of
rewriting the original experiment.
