# RL results index

**Status, 2026-09-19:** exact-toy gates, the corrected M1 systems pilot, and the
first full-UNO B learning pilot are complete. The pilot is a successful one-seed
DEV learning diagnostic, not a frozen learner-strength or EVAL result.

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
- [Stage 7 PPO minibatch calibration](stage7-minibatch-calibration.md) — retained
  CPU-4/minibatch-512; CPU-1,024 improved end-to-end throughput by only 4.8%,
  below the preregistered 15% gate, so no learning-equivalence run was made.

## Diagnostics

- [Stage 7 DEV panel sizing](stage7-dev-panel-sizing.md) — about 5,760 deals/cell
  are needed for a paired 95% CI half-width of 0.005; the first pilot instead
  uses 256 deals/cell at five coarse monitoring points because full precision at
  every checkpoint is computationally infeasible.
- [Stage 7 full-UNO B pilot](pilot-notes.md) — seed 0 reached 501,367 learner
  decisions in 3,606.5 seconds and improved monotonically from 0.0313 to 0.4287
  DEV score. Full-rollout KL stayed controlled despite nearly universal actor
  gradient-norm clipping. The result also records weak critic fit, a misnamed
  draw-action metric, and a final-minibatch-only PPO logging defect.
