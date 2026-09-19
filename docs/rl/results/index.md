# RL results index

**Status, 2026-09-19:** exact-toy gates and the corrected M1 systems pilot are
complete. There is no interpreted full-UNO learner-strength result yet.

This file is the canonical headline ledger for learned-policy measurements. The
Steps 1-10 baselines and foundation measurements remain in the
[foundation results index](../../foundations/results/index.md); do not copy their
large tables here.

## Reporting conventions

- Score is `(return + 1) / 2`, with draws worth 1/2.
- Report both seats and compare against the same-seat baseline.
- Separate exact game values, trajectory-sampling uncertainty, and variation across
  independently trained policies.
- Name the training seeds, DEV selection data, EVAL block, opponents, checkpoint
  rule, units, and whether uncertainty is an SE or CI.
- League score, held-out score, BR gap, PIMC gap, and attack lower bound are distinct
  quantities.
- A NashConv attack result is a lower bound. A weak bound is not evidence of safety.

Each entry should state the question, headline result, decision, and link to the
semantic experiment file containing configuration and provenance.

## Systems

- [Stage 6 M1 systems pilot](stage6-m1-systems.md) — selected one-process,
  64-live-game/one-thread collection followed by four-thread CPU PPO; conservative
  planning rate 140 learner decisions/s, not a strength claim.

## Preregistered diagnostics

- [Stage 7 PPO minibatch calibration](stage7-minibatch-calibration.md) — pending
  frozen-policy compute sweep and, only for a qualifying candidate, a matched
  five-seed short learning check at equal wall time and equal fresh data.
