# Stage 8 — extended B diagnostic

## Question

Did the corrected, longer B run close the deficit seen in the pre-fix 500k B
pilot, and did it reach the Heuristic league reference?

## Decision rule and configuration

This is a one-seed DEV diagnostic, not a frozen strength claim. The actor is the
final checkpoint of `FULL_UNO_B_seed0_mb512_cpu_1789839695`, trained for 1,010,278
learner decisions with the schema-2 process that was already running when the
schema-3 correction was made. It was evaluated against the fixed eight-cell
Heuristic panel on the nested diagnostic block beginning at deal 900,000. The
predeclared target was a paired 95% half-width of 0.005.

## Result

| Deals/cell | B score | Heuristic score | B − Heuristic | 95% half-width | Estimated deals for 0.005 |
|---:|---:|---:|---:|---:|---:|
| 6,989 | 0.48973 | 0.49991 | -0.01018 | 0.00607 | 10,295 |
| 8,486 | 0.48919 | 0.50128 | -0.01209 | 0.00553 | 10,383 |
| 9,984 | 0.48953 | 0.50116 | -0.01163 | 0.00507 | 10,263 |

At the largest panel the nominal 95% interval for B minus Heuristic is
`[-0.01670, -0.00656]`. It excludes parity, and the precision target is narrowly
missed. The nested estimates are stable: increasing the panel does not reveal a
material change in the point estimate.

Relative to the pre-fix B result (`-0.0593` at 8,192 deals/cell), the final
deficit is smaller by about `0.0477` score points, an approximately 80% reduction
in the gap. This is strong evidence that the longer/corrected run helped. It is
not a causal estimate of the evidence fix alone because representation schema,
training duration, and the running process are confounded.

## Training trajectory

The coarse scheduled DEV scores were:

| Learner decisions | DEV score | Raw entropy |
|---:|---:|---:|
| 0 | 0.0313 | 0.984 |
| 257,037 | 0.3804 | 0.479 |
| 501,591 | 0.4253 | 0.331 |
| 757,997 | 0.4458 | 0.244 |
| 1,010,278 | 0.4663 | 0.220 |

The curve is clearly sublinear, but it has not literally flattened: the last
quarter still gained about 0.0205 on the coarse panel. The high-precision final
panel gives a stronger estimate of roughly 0.4895 because the 256-deal coarse
panel has a score standard error around 0.011. The evidence supports diminishing
returns and a likely near-term plateau, not a proof that the policy is exactly
stationary.

## Interpretation

The remaining deficit is concentrated in the harder opponents: at 9,984 deals the
gaps were about -0.0291 and -0.0164 against the two full-dev seats, and -0.0290
and -0.0135 against the unpredictable opponent. The two heuristic-memory cells
were effectively tied. This looks like incomplete tactical/generalization
strength rather than a uniform score-calibration problem.

At the largest panel B had mean legal-action count 2.63, raw entropy 0.221 nats,
normalized entropy 0.158, and 1.36 effective actions. Heuristic had 3.04 legal
actions, 0.156 nats, normalized entropy 0.108, and 1.25 effective actions. B is
therefore not simply random or collapsed; it is somewhat more diffuse than the
reference on the states it reaches. Its optional-draw and keep rates were zero,
so this run provides no evidence that the learner has learned the voluntary-draw
behavioral branch.

## Limitations and next decision

This result is one seed, one final checkpoint, and a DEV panel. It cannot decide
whether B is better than A across training seeds, nor whether the remaining gap is
caused by representation, optimization, or the shared terminal-return signal.
The schema-2 provenance must remain attached to the artifact; do not mix this
actor into a schema-3 matched A/B comparison.

The next useful comparison is a fresh, schema-3 matched A/B study at equal data
with multiple training seeds. If both arms still plateau similarly, investigate
the shared optimizer/critic and tactical behavior before adding more B history or
simply extending the budget.
