# Stage 8 — A/B feed-forward ablation

**Status, 2026-09-19:** A seed 0 completed successfully and favored A over B by
0.01276 at the original 500k budget. Before running matched seeds 1–4, B's 5+
evidence bin was found to have an ambiguous marginal projection. Schema 2 attempted
to resolve it by retaining one board, but a follow-up review found that it selected
the oldest board and accidentally lost the promised overflow count. Its ongoing
1,000,000-decision seed-0 run remains a diagnostic, not part of the final matched
comparison.

**Post-launch schema note:** that corrected B process loaded schema 2 and continues
with its in-memory code. The follow-up review replaced the ambiguous overflow rule
with schema 3 (most recent 5+ board plus an all-event count). Current loaders reject
unknown schemas rather than silently changing a policy; actor exports with the
known schema-2 hash use an explicit read-only compatibility extractor. Training
resume still requires schema 3. Run future matched A/B seeds under one frozen
schema. The revised schema-3 comparison uses the doubled
1,000,000-decision budget to test whether B needs a longer warmup.

## Question

At the 1,000,000 learner-decision budget, does B's structured public-evidence
branch improve full-UNO league learning enough to justify retaining it over the
simpler A position/discard representation?

A and B use the same position encoder, action scorer, critic, optimizer, league,
training manifests and evaluation panels. They differ only in B's structured
public-evidence inputs. This is a representation ablation, not a test of whether
either arm has converged or surpassed Heuristic.

## Frozen equal-data protocol

- Arms: A and B.
- Training seeds: 0, 1, 2, 3 and 4, paired by seed.
- Primary checkpoint: the final complete update at or beyond 1,000,000 learner
  decisions. DEV-selected checkpoints are secondary because selection adds noise.
- Collection: one process, 64 live games, 64 games per opponent/seat cell and one
  collection Torch thread.
- PPO: CPU, four Torch threads, minibatch 512, and the existing frozen optimizer.
- Scheduled DEV: the same 256 deals/cell at initialization, 25%, 50%, 75% and the
  final boundary.
- Safety wall cap: 7,500 seconds (2 hours 5 minutes). This is not the comparison
  budget. A run stopped by time before reaching the data cap is incomplete rather
  than equal-data evidence.
- Tracking study/kind: `stage8-a-b-ablation` / `ablation-data`.

Both actor and critic gradients are already norm-clipped separately at 0.5 on
every optimizer step. Record their unclipped norms and clipping fractions. PPO
value-function clipping remains disabled for both arms. Do not retune the critic,
gradient caps, learning rates or entropy separately by arm.

Run one arm/seed with:

```sh
uv run uno rl ablation A 0
```

Replace A and the seed as required.

## Decision rule

The primary estimand is the paired-seed mean `B - A` final-checkpoint league score
at 1,000,000 learner decisions. Report every seed, the mean, training-seed spread,
per-cell results, actual decision overshoot, games, engine decisions, wall time
and memory. Keep game-sampling uncertainty separate from training-seed variation.

Use 0.01 score as the predeclared practical-effect threshold:

- evidence entirely above +0.01 favors B's evidence branch;
- evidence entirely below -0.01 favors A on strength as well as simplicity;
- an interval wholly inside `[-0.01, +0.01]` treats the arms as practically
  equivalent and favors A's simpler representation;
- otherwise the result is inconclusive at this budget.

The 256-deal scheduled panel is a coarse learning and selection diagnostic. After
A seed 0 finishes, measure the paired A-minus-B per-deal variance on a separate
nested DEV panel and freeze the final comparison-panel size before inspecting
seeds 1–4. Do not substitute the B-versus-Heuristic variance estimate: a direct
A/B difference can have different covariance.

## Equal-wall follow-up

Equal wall time is a separate secondary comparison, not something inferred by
dividing the equal-data scores by runtime. Freeze its cap and evaluation schedule
after the equal-data seed-0 pair measures A's actual rate. Run both arms under the
same current revision.

## Stop and interpretation rules

Stop and diagnose rather than continuing the study if A has a numerical failure,
policy collapse, invalid probability replay, or a material unexplained throughput
or memory regression. Weak critic explained variance, critic gradient clipping,
or failure to beat Heuristic alone are not stopping conditions: the critic and
training budget are shared controls, and the first B curve had not converged.

The optional 9,984-deal B-versus-Heuristic extension belongs to the Stage 7 panel
sizing protocol. It is an estimated size for reaching a paired 95% half-width of
at most 0.005, not a prerequisite for this ablation and not a guarantee of an
exactly 0.005 realized interval.

## Seed-0 early result (500k)

*Note: This was a preliminary result before the B representation bug was fixed and
the budget doubled.*

A stopped on the data cap after 37 complete updates, 506,946 learner decisions,
18,944 games and 1,016,078 engine decisions. It took 3,473.3 active seconds and
3,624.1 wall seconds. Its fixed coarse-panel DEV curve was:

| Learner decisions | A DEV score | B DEV score at nearest boundary |
|---:|---:|---:|
| 0 | 0.0205 | 0.0313 |
| 135,079 / 127,889 | 0.3008 | 0.3438 |
| 256,887 / 256,841 | 0.3862 | 0.3979 |
| 382,160 / 379,856 | 0.4087 | 0.4087 |
| 506,946 / 501,367 | 0.4482 | 0.4287 |

The final actor was then evaluated against same-seat Heuristic over 9,984 deals
per opponent/seat cell from diagnostic deals 900,000–909,983. A scored 0.45435,
Heuristic scored 0.50116, and the paired difference was -0.04681 with nominal 95%
CI `[-0.05174, -0.04189]`. The half-width was 0.00493, meeting the 0.005 target.
All eight cell differences still favored Heuristic. This is DEV evidence from one
training seed, not a frozen strength claim.

### Direct paired A/B result (500k)

The A and B diagnostic files share identical Heuristic outcomes over their first
8,192 deals, confirming the intended common deal/opponent streams. Pairing the
two actors' equal-weight eight-cell mean score for each deal gives:

| Deals/cell | A score | B score | A - B | 95% half-width | 95% CI |
|---:|---:|---:|---:|---:|---:|
| 8,192 | 0.45473 | 0.44197 | +0.01276 | 0.00397 | `[+0.00879, +0.01672]` |

Every opponent/seat cell favored A; the cell advantages ranged from +0.00464 to
0.02014. Thus seed 0 provides clear game-sampling evidence that A is stronger at
this budget. It does not settle the predeclared practical decision: in the
protocol's `B - A` orientation the interval is `[-0.01672, -0.00879]`, which
crosses the -0.01 boundary, and it contains no training-seed uncertainty. The A
run also overshot the soft data cap by 5,579 more decisions than B; this is small
and declared, but remains part of the equal-data limitation.

### Frozen direct-comparison panel

The paired A/B half-width estimates were stable around 0.0040–0.0047 from 5,734
through 8,192 deals/cell. The implied requirement was about 5,200 deals/cell. A
direct nested check at 5,184 narrowly missed the target with half-width 0.00502;
the next 64-deal boundary, 5,248, achieved 0.00498. Freeze **5,248 deals/cell**
for final A/B diagnostic comparisons on seeds 1–4. This sizing is specific to the
direct A/B difference and is separate from either actor's comparison to Heuristic.

### Optimization and behavior

A's mean pre-update critic explained variance was 0.0539 and its mean post-fit
same-rollout value was 0.1260; final values were 0.0933 and 0.1623. Critic gradient
clipping occurred on 88.0% of optimizer steps and 99.0% in the final update. This
confirms that the critic fits some current-rollout structure while the shared 0.5
gradient cap is frequently active. Do not change it within the A/B study.

At high precision A and B had almost identical policy entropy, 0.3557 versus
0.3540. A's mean game length was modestly shorter, 45.81 versus 46.58 turns. Its
optional-draw rate was 0.0297% of opportunities, still effectively zero for the
earlier behavioral hypothesis.

## Next decision

Run matched A/B seeds 0–4 under the extended 1,000,000-decision frozen equal-data
preset. Evaluate each final actor on the first 5,248 diagnostic deals/cell and
retain per-deal scores. Report the five paired training-seed differences before
deciding whether A's simplicity remains superior. Do not tune the critic or
extend training within only one arm during this comparison.
