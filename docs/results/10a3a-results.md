# Step 10a.3A: repaired coarse-memory measurement (2026-09-17)

> On top of commit `3a4c782`. Reruns: `uv run uno memory <ruleset> 20` for
> `TOY`, `TOY_FULL`, `TOY2`, `TOY2_SKIP`, and `TOY2_FULL`. Tests and resource
> measurements are listed below. This is 10a.3A only; the observation contract
> and LBR remain 10a.3B/C.

## What was repaired

The old quotient construction extracted K0 / K1 / K2 after the full L1 history
had selected canonical color/rank coordinates. Two observations with identical
raw K1 fields could therefore receive different action distributions because
their discarded initial-hand/draw histories selected different frames. The old
score was exact for the resulting L1 policy, but it was not evidence for a policy
measurable from the claimed coarse input.

`coarsen` now canonicalizes the coarse key under its own symmetry and maps legal
actions and tree arcs through the same group element. Action orbits come from the
coarse key's stabilizer. `CoarseAgent` deploys that policy directly from a raw
observation, and `lift_to_l1` maps it into an ordinary compiled L1 strategy for
exact replay.

The experiment now builds and releases K0, K1, and K2 sequentially. This avoids
retaining three node/arc keyings together on the 28-million-node `TOY2_FULL`
tree. It prints scores and absolute score-point gaps to BR L1; the historical
09.5a table printed returns.

## Regression and independent oracle

The review's `TOY` collision remains at the same positive on-policy reach,
`0.0017460317460317458`. Before the repair, equal raw K1 observations returned
`{1: 1}` and `{3: 1}`. Both the direct coarse agent and the lifted compiled agent
now return `{1: 0.5, 3: 0.5}` for both observations.

On the independent full and quotient trees for `SMALL`:

- quotient K0 / K1 / K2 groups are exactly the symmetry orbits of the raw
  full-tree groups;
- one shared deployed policy lifts without misses; and
- its exact value agrees across the two trees at both seats to `1e-12`.

This compares the same policy. It deliberately does not require two separate
imperfect-recall local searches to choose the same iterate.

## Corrected achieved scores

Score `(return + 1) / 2`, seat 0 / seat 1. Every cell uses 20 local-response
iterations starting from the Heuristic and passed `heuristic <= achieved <= BR
L1`. These are achieved scores, not certified optima of the K0 / K1 / K2 policy
classes.

| ruleset | opponent | Heuristic | K0 | K1 | K2 | BR L1 |
|---|---|---:|---:|---:|---:|---:|
| `TOY` | Random | 0.7416 / 0.4147 | 0.7674 / 0.5188 | 0.7927 / 0.5490 | 0.7927 / 0.5498 | 0.7934 / 0.5513 |
| | Heuristic | 0.6934 / 0.3066 | 0.7295 / 0.4098 | 0.7378 / 0.4313 | 0.7378 / 0.4313 | 0.7378 / 0.4361 |
| `TOY_FULL` | Random | 0.7416 / 0.4147 | 0.7708 / 0.5241 | 0.8017 / 0.5768 | 0.8017 / 0.5768 | 0.8017 / 0.5882 |
| | Heuristic | 0.6934 / 0.3066 | 0.7362 / 0.4211 | 0.7446 / 0.4613 | 0.7446 / 0.4613 | 0.7446 / 0.4732 |
| `TOY2` | Random | 0.6730 / 0.4474 | 0.7027 / 0.5023 | 0.7289 / 0.5650 | 0.7292 / 0.5668 | 0.7305 / 0.5694 |
| | Heuristic | 0.6385 / 0.3615 | 0.6714 / 0.4242 | 0.6928 / 0.4917 | 0.6929 / 0.4918 | 0.6959 / 0.4937 |
| `TOY2_SKIP` | Random | 0.6414 / 0.4806 | 0.7188 / 0.5504 | 0.7276 / 0.5753 | 0.7276 / 0.5755 | 0.7425 / 0.5919 |
| | Heuristic | 0.5767 / 0.4233 | 0.6678 / 0.4962 | 0.6776 / 0.5193 | 0.6776 / 0.5193 | 0.6941 / 0.5356 |
| `TOY2_FULL` | Random | 0.6730 / 0.4474 | 0.7115 / 0.5237 | 0.7440 / 0.6118 | 0.7448 / 0.6129 | 0.7479 / 0.6164 |
| | Heuristic | 0.6385 / 0.3615 | 0.6791 / 0.4444 | 0.7051 / 0.5287 | 0.7052 / 0.5297 | 0.7118 / 0.5371 |

Absolute gaps below BR L1, in score points:

| ruleset | opponent | K0 gap | K1 gap | K2 gap |
|---|---|---:|---:|---:|
| `TOY` | Random | 2.5973 / 3.2508 | 0.0704 / 0.2300 | 0.0704 / 0.1437 |
| | Heuristic | 0.8333 / 2.6323 | 0.0000 / 0.4762 | 0.0000 / 0.4762 |
| `TOY_FULL` | Random | 3.0873 / 6.4193 | 0.0000 / 1.1460 | 0.0000 / 1.1460 |
| | Heuristic | 0.8333 / 5.2116 | 0.0000 / 1.1905 | 0.0000 / 1.1905 |
| `TOY2` | Random | 2.7749 / 6.7093 | 0.1605 / 0.4344 | 0.1272 / 0.2516 |
| | Heuristic | 2.4442 / 6.9531 | 0.3080 / 0.1964 | 0.2991 / 0.1875 |
| `TOY2_SKIP` | Random | 2.3681 / 4.1419 | 1.4893 / 1.6566 | 1.4893 / 1.6339 |
| | Heuristic | 2.6286 / 3.9391 | 1.6447 / 1.6298 | 1.6447 / 1.6298 |
| `TOY2_FULL` | Random | 3.6374 / 9.2609 | 0.3913 / 0.4576 | 0.3113 / 0.3443 |
| | Heuristic | 3.2746 / 9.2768 | 0.6719 / 0.8415 | 0.6629 / 0.7433 |

Corrected coarse set counts (share of quotient L1 sets):

| ruleset | K0 | K1 | K2 |
|---|---:|---:|---:|
| `TOY` | 557 (4.6 %) | 2,119 (17.6 %) | 2,411 (20.0 %) |
| `TOY_FULL` | 1,562 (1.1 %) | 5,851 (4.2 %) | 7,570 (5.5 %) |
| `TOY2` | 1,652 (1.3 %) | 10,026 (7.7 %) | 12,628 (9.8 %) |
| `TOY2_SKIP` | 5,229 (1.2 %) | 27,226 (6.2 %) | 34,452 (7.8 %) |
| `TOY2_FULL` | 5,737 (0.2 %) | 31,967 (0.9 %) | 49,010 (1.3 %) |

The smaller counts are expected: symmetric copies previously kept apart by the
discarded L1 coordinate frame now share one coarse set. K0 was affected most;
some refreshed K0 cells fall by several score points. K1 changes range from
negligible to about 0.6 points. Heuristic and BR L1 controls are unchanged.

## Interpretation and limits

K1 remains a useful cheap baseline on these toys, and K2 still adds little under
this particular encoding. The corrected evidence does not establish that K1 is
sufficient on full UNO. A gap between the achieved iterate and BR L1 also does
not isolate representational loss: `memoryless_response` is an iterative local
search and does not certify the K-class optimum.

No normalized-skill claim is made here. If one is added later, its zero must be
the same-seat Random baseline rather than a return of zero.

## Verification and resources

- `uv run uno memory <ruleset> 20` on all five rulesets: 0 failed checks in
  every run.
- `tests/compiled/test_coarse.py`: 9 passed, including the positive-reach
  collision, corrected pinned values, and the full/quotient oracle.
- Full suite: 294 passed in 468.35 s.
- `uv run uno report TOY heuristic` and `uv run uno report TOY random`: 0
  failed checks; all published report-card values reproduced unchanged.
- `TOY2_FULL` end-to-end wall time: 424.2 s on the cached 28,057,517-node
  tree. Peak RSS is recorded separately below.
