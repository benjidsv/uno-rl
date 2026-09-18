# RL results index

**Status, 2026-09-18:** no UNO v5 learner results yet. Phase 1 exact-toy validation
is next.

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
