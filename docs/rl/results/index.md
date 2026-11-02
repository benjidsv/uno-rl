# RL results index

**Status, 2026-09-19:** exact-toy measurements, the corrected M1 systems pilot,
and the first full-UNO B learning pilot are complete. The exact gradient checks
passed, but the retained toy learning gate did not. The full-UNO pilot is a
successful one-seed DEV learning diagnostic, not a frozen strength or EVAL result.

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

- [Critic calibration as a win probability](critic-calibration.md) — rel-C's and B's
  critics are miscalibrated as `(v + 1) / 2` against each league bot (worst bin gap
  0.105-0.218, late game worst); per-opponent Platt fails on held-out data, so the
  play demo uses rollout values.

- [Stage 5 exact-toy study](stage5-toy-gates.md) — all four raw-gradient checks
  passed, including the PPO/score-function identity. The retained tabular gate
  passed 5/5 seeds against Heuristic but only 3/5 against PlayFirstRandom because
  seeds 0 and 1 missed the seat-1 tolerance. Neural comparisons had no predeclared
  threshold and remain diagnostic; Stage 5's exit gate is open.

- [Stage 7 DEV panel sizing](stage7-dev-panel-sizing.md) — 256 deals/cell remains
  the coarse monitoring panel. The initialized policy suggested about 5,760
  deals/cell for a paired 95% CI half-width of 0.005, while the trained final
  actor's 8,192-deal validation estimated 9,926; the predeclared rounding rule
  requests a final nested extension to 9,984 deals/cell.
- [Stage 7 full-UNO B pilot](pilot-notes.md) — seed 0 reached 501,367 learner
  decisions in 3,606.5 seconds and improved monotonically from 0.0313 to 0.4287
  coarse-panel DEV score. On the larger diagnostic panel its final actor scored
  0.4420 against 0.5013 for Heuristic, a paired difference of -0.0593 with a
  nominal 95% CI of `[-0.0648, -0.0538]`. Full-rollout KL stayed controlled
  despite nearly universal actor gradient-norm clipping. Corrected counters show
  the former approximately 30% Draw metric was almost entirely forced drawing,
  not avoidable optional draws.
- [Stage 8 A/B ablation](stage8-a-ablation.md) — A seed 0 completed 506,946
  decisions and scored 0.45473 against B's 0.44197 on 8,192 directly paired
  diagnostic deals/cell: `A - B = +0.01276`, nominal 95% CI
  `[+0.00879, +0.01672]`. This excludes zero but crosses the predeclared
  one-point practical boundary and contains no training-seed uncertainty. The
  direct A/B panel is frozen at 5,248 deals/cell; matched seeds 1–4 remain.
- [Stage 8 extended B diagnostic](stage8-b-extended-diagnostic.md) — the corrected
  longer B run reduced the Heuristic deficit from `-0.0593` to `-0.0116` at
  9,984 deals/cell, but the interval still excludes parity and the score curve
  shows diminishing returns.
