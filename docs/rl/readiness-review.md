# RL readiness review — 2026-09-18

**Scope:** read-only review at `4572c5d`, before the agreed plan/contract revision.
This is an implementation handoff, not a learner result. No training run or new
algorithm was implemented. The decisions are now in [plan.md](plan.md); the
[observation contract](observation-contract.md) records intended corrected behavior.

## Findings to resolve before interpreting learning

| Area | Finding at reviewed commit | Required follow-up |
|---|---|---|
| Distillation legality | `DistillBuffer.extend_trace` passes `tuple(step.probs)` as legality. PIMC returns singleton support. | Query and store the engine legal set, not behavior support. |
| Distillation symmetry | Ordered color blocks are flattened into an unrestricted MLP. `ArmBPolicy.symmetric()` only renames. | Shared equivariant processing, symmetry-preserving preprocessing, production probability tests. |
| Loss/distribution | Fitting uses unmasked softmax at temperature 1; deployment masks and divides by 0.05. | One declared distribution/mask/temperature convention in fitting and deployment. |
| Feature fidelity | Histograms/action summaries omit some full card relationships. Tail size does not clear after reshuffle; uncapped-toy clock divides by 1 rather than becoming zero. | Implement revised fields; name any additional compression as an ablation. |
| Validation | Preprocessing uses all rows; ordered game-ID split is grouped by opponent/round. | Fit preprocessing on training only; split grouped deals across seats/columns, with disjoint game-score diagnostics. |
| Test coverage | Contract tests use separate extractors; legality checks field presence. Distillation symmetry test uses a rank permutation and a singleton-action initial state. | Exercise production code and non-vacuous failure cases. |
| Contract | Recurrent symmetry, fabricated-history consistency, generic wrapping, and evaluation/training corrections were overstated or conflated. | Revised contract governs new implementation; validate integrations explicitly. |

The old distillation command collects teacher then student trajectories, fits two
students, and evaluates only 20–100 DEV deals/cell. The commit message's approximate
seven-point teacher gap is a smoke observation with unresolved implementation
confounds, not a compression limit. The old checkpoint/feature format should not
be silently reused under a corrected architecture; version new artifacts.

## Checks actually performed in the review

Using the existing `.venv/bin/python` (plain `uv run` could not access its external
cache in the sandbox):

- `-m pytest -q tests/eval/test_distill.py`: **5 passed**.
- `-m pytest -q tests/game/test_observation_contract.py tests/eval/test_target.py tests/eval/test_attack.py tests/agents/test_rulebot.py`: **60 passed**.
- Bounded color/rank probe: `TOY`, first 20 deal seeds, all 48 symmetries,
  `PolicyNet` width 16 initialized with torch seed 4, identity feature scaling,
  policy temperature 1. Maximum mapped probability error **0.00374611140605402**
  at deal seed 9, group element 44 (color map `[3, 2, 0, 1]`). This refutes
  equivariance by construction; it is not a measured trained-policy strength loss.
- Teacher-support probe: `TOY`, seed 0, PIMC with two worlds, depth 1, prior belief,
  leaf `len(hand1) - len(hand0)`, against Heuristic. Actor decisions with **2 and
  5 engine-legal actions** were stored as **1 legal action**, with one-hot labels.

These probes were one-off read-only scripts, not added regression tests. Their
regression equivalents belong in the implementation work. No full suite, expensive
sweep, or confirmatory evaluation was run. Passing the existing suites did not
exclude the reproduced defects.

## Evidence carried into the plan

- [Corrected memory measurements](../foundations/results/10-readiness/10a3a-results.md):
  achieved coarse-policy values, not certified representation optima.
- [Training targets and PIMC](../foundations/results/10-readiness/10a2-results.md):
  specific corrected-gradient checks, substantial correction construction cost,
  and an achievable strong column-free full-UNO policy.
- [Frozen league](../foundations/results/10-readiness/10c-results.md): behavioral
  differences but related policies and no universal superiority claim.
- [Attack calibration](../foundations/results/10-readiness/10a3c-results.md): target
  support and lower-bound interpretation remain essential.

Historical result files remain unchanged. This review does not invalidate the
engine/solver evidence; it identifies where the new learner cannot inherit a
stronger guarantee than was actually tested.
