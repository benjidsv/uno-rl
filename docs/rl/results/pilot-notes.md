# Stage 7 full-UNO B pilot — result and live notes

**Status, completed 2026-09-19:** the first full-UNO learning pilot reached its
500,000-decision soft cap and improved monotonically across all five coarse DEV
checkpoints, from 0.0313 to 0.4287 score. This passes the intended learning-health
diagnostic, but it is one training seed selected and measured on DEV—not a frozen
strength or EVAL claim. The timestamped Probe 1 remains below as an unmodified
record of the mid-run decision.

## Question and stopping rule

Does the first from-scratch B policy show a credible improving learning curve under
the frozen Stage 7 pilot configuration, without numerical failure, policy collapse,
or uncontrolled PPO movement?

The run keeps its declared 500,000-decision and 3,600-second soft caps. A live
probe may justify an emergency stop only for non-finite values, repeated severe KL
overshoot or clear collapse. A surprising diagnostic by itself is investigated but
does not change the run configuration midstream.

## Configuration and provenance

- command: `uv run uno rl pilot 0`
- ruleset/actor/training seed: `FULL_UNO` / B / 0
- frozen four-member rulebot league, balanced seats
- 512 games per update: 64 games per opponent/seat cell, 64 live games
- one collection Torch thread; CPU PPO with four Torch threads
- minibatch 512; four actor epochs and four critic epochs
- actor/critic learning rates: `3e-4` / `1e-3`
- PPO ratio clip: 0.2; actor/critic gradient-norm caps: 0.5 / 0.5
- sampled-KL actor stopping threshold: 0.02
- fixed DEV panel: 256 deals per opponent/seat cell at 25% data milestones
- config hash: `b54d8ae8d7c05b74`
- MLflow run ID: `1790d39371bf41d69d6e7c5e4be399b3`
- local run directory: `logs/rl/FULL_UNO_B_seed0_mb512_cpu_1789786184`
- after completion, 95 supported single-level MLflow metric aliases were
  backfilled with original values, steps and timestamps; each has exactly one
  slash selecting its semantic folder
- the 88 flat source keys and a superseded first attempt at nested aliases remain
  intact because MLflow exposes no metric-key deletion API
- pre-migration database backups:
  `logs/rl/mlflow.db.backup-20260919-pilot-backfill` and
  `logs/rl/mlflow.db.backup-20260919-flat-folders`

## Metric glossary

| Metric | Brief definition | How to read it |
|---|---|---|
| DEV score | Equal-weight mean score over the eight fixed opponent/seat cells; score is `(return + 1) / 2` | Primary coarse learning signal. This is DEV, not fresh confirmatory EVAL. |
| Training score | Equal-weight cell score on the just-collected on-policy rollout | Fast but noisy and distribution-dependent; use as a health diagnostic, not checkpoint-selection evidence. |
| Logged voluntary draw rate | Number of Draw actions divided by all learner decisions | Misnamed in this run: it includes mandatory draws when no play exists as well as optional draws when a play exists. It is not an avoidable-error rate. |
| DEV entropy | Mean learner action entropy over the complete evaluation panel | Falling entropy means a more decisive policy; near-zero entropy plus worsening score would suggest collapse. |
| Actor entropy | Entropy reported by the PPO actor update | Intended as an optimization diagnostic, but affected by the aggregation defect below in this run. |
| Approximate KL | Full-rollout sampled estimate `mean((r - 1) - log r)` after an actor epoch | Reliable policy-movement diagnostic here. Actor epochs stop when it exceeds 0.02. |
| PPO ratio clip fraction | Fraction of sampled actions with `abs(r - 1) > 0.2` | Distinct from gradient clipping. In this run the logged value is only the final minibatch, due to the defect below. |
| Actor gradient norm | Mean global actor gradient L2 norm *before* norm clipping, over actor optimizer steps | Compare with the 0.5 cap. It measures raw scale, not the applied post-clipping norm. |
| Actor clipped fraction | Fraction of actor optimizer steps whose pre-clipping norm exceeded 0.5 | A value of 1 means every minibatch gradient was norm-clipped; it does **not** mean every action hit the PPO ratio clip. |
| Critic MSE | Decision-mean return-prediction MSE over the final critic epoch | Scale changes with the return distribution; interpret with explained variance. |
| Explained variance | Fit of the frozen pre-update critic values to the rollout returns | Zero means no improvement over a constant predictor; negative is worse; one is perfect. |
| Early stopped | Whether the actor stopped before all planned epochs after a full-rollout KL check | The critic still completes all four epochs. Occasional stopping is protective; repeated stopping indicates oversized actor updates. |
| Iteration throughput | Fresh learner decisions divided by collection-plus-PPO iteration seconds | Whole training-iteration data throughput; scheduled DEV and checkpoint time are outside this individual-iteration rate. |

## Probe 1 — update 18, 256,841 learner decisions

Read-only snapshot at 1,849.9 seconds of run wall time. The run was still marked
`RUNNING`; no code, configuration, checkpoint or tracking state was changed.

### Learning curve

| Learner decisions | Update | DEV score | SE | 95% half-width | DEV entropy |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0.0313 | 0.0040 | 0.0079 | 0.9837 |
| 127,889 | 8 | 0.3438 | 0.0101 | 0.0198 | 0.5350 |
| 256,841 | 18 | 0.3979 | 0.0109 | 0.0214 | 0.4395 |

The latest on-policy training score was 0.3770; observed update scores ranged from
0.0332 to 0.4004. The DEV rise is much larger than its trajectory-sampling error
and is consistent across the training and fixed-panel views. The simultaneous
entropy decline shows the policy becoming more selective, while a DEV entropy of
0.44 is not evidence of deterministic collapse.

### PPO and critic health

| Diagnostic | Observed through update 18 | Interpretation at this probe |
|---|---:|---|
| Actor pre-clipping gradient norm | 1.19–4.51; latest 2.23 | Above the 0.5 cap but bounded, with no upward explosion. |
| Actor clipped fraction | 0.879 once, 0.970 once, 1.000 on 16/18 updates | Gradient clipping is effectively always active. |
| Full-rollout approximate KL | 0.00087–0.02252; latest 0.00293 | Generally conservative relative to the 0.02 threshold. |
| Actor early stops | one, at update 6 with KL 0.02252 | Isolated protective stop, not systematic instability. |
| Logged PPO ratio clip fraction | 0–0.120; latest 0.0395 | Apparently modest, but only a final-minibatch snapshot in this run. |
| Critic explained variance | -0.001–0.087; latest 0.045 | Positive but weak critic fit; worth watching, not an emergency failure. |
| Critic MSE | latest 0.827 | Not directly comparable to the early 0.163 without accounting for the changing return distribution. |
| Latest collection / PPO / iteration time | 7.56 / 79.14 / 86.70 s | PPO remains dominant; about 145 fresh learner decisions/s end to end for this iteration. |

`actor_clipped_fraction = 1` is therefore not evidence that every PPO probability
ratio is saturated. It records a different mechanism: every minibatch's raw global
gradient norm exceeded 0.5. The actor objective deliberately preserves the
game-average sum by scaling minibatch means by `T/G`; through this probe an update
had roughly 24.5–34.9 learner decisions per game. That objective scale makes a
conventional 0.5 norm cap likely to be active. Persistent clipping may make the cap
an important optimizer control, but the improving score, bounded raw norms and
mostly small full-rollout KL do not support stopping or retuning this run.

### Diagnostic logging defect discovered

In `uno/rl/ppo.py`, `stats.surrogate`, `stats.entropy`, and
`stats.clip_fraction` are overwritten on every actor minibatch. Their logged values
therefore describe only the final minibatch of the final completed actor epoch.
They are not rollout-wide or update-wide aggregates. `max_abs_log_ratio` is a true
maximum across processed minibatches, `actor_grad_norm` and
`actor_clipped_fraction` are optimizer-step averages, and `approx_kl` is recomputed
over the full rollout, so those diagnostics do not share this defect.

This is a reporting defect, not evidence that the PPO update itself is wrong: the
loss used for every minibatch and the KL early-stop computation are unchanged. Fix
it in a separate session by choosing and documenting a consistent aggregation
contract, adding a test that uses unequal final minibatch sizes, and naming the
result so it is clear whether it summarizes the final policy, final epoch, or all
optimization steps. Do not retrofit invented aggregate values into this run.

## Probe 1 decision and limitations

Continue unchanged. The measured policy is improving substantially, full-rollout
KL is controlled, gradients are finite and explicitly bounded, and there is no
observed collapse. Do not change the gradient cap during this run: doing so would
make one pilot internally heterogeneous.

This is one seed on a reused DEV panel, observed before the declared stopping
boundary. It does not establish final strength, convergence, training-seed
variation, superiority to the same-seat Heuristic baseline, or any EVAL claim.
The persistent gradient clipping and weak critic remain follow-up questions after
the completed curve is reviewed.

## Final result — update 38, 501,367 learner decisions

The run stopped cleanly on the data cap after 38 complete updates. The final
scheduled checkpoint crossed 500,000 decisions, so its DEV result was reused as
the final evaluation rather than replaying the same panel.

| Learner decisions | DEV score | SE | 95% half-width | DEV entropy | Logged draw-action rate |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.0313 | 0.0040 | 0.0079 | 0.9837 | 0.3758 |
| 127,889 | 0.3438 | 0.0101 | 0.0198 | 0.5350 | 0.3103 |
| 256,841 | 0.3979 | 0.0109 | 0.0214 | 0.4395 | 0.3061 |
| 379,856 | 0.4087 | 0.0105 | 0.0206 | 0.4001 | 0.2975 |
| 501,367 | 0.4287 | 0.0112 | 0.0220 | 0.3580 | 0.2939 |

The final nominal 95% interval is `[0.4067, 0.4507]`. The fixed panel produced a
monotone curve, with a large initial gain followed by slower improvement. Because
the routine panel retained only aggregates, not paired per-deal checkpoint
differences, it cannot resolve whether either late increment is individually
different from sampling noise. The curve as a whole establishes that the
from-scratch implementation learns; it does not establish convergence.

Final per-cell scores:

| Opponent | Seat 0 | Seat 1 |
|---|---:|---:|
| `heuristic_memory` | 0.4258 | 0.3906 |
| `rulebot_transfer_toy_v1` | 0.4414 | 0.4297 |
| `rulebot_full_dev_v1` | 0.4219 | 0.4297 |
| `rulebot_unpredictable_t0.15_m0.2_d0.02_k0.04_kw0.1` | 0.5000 | 0.3906 |

### Final optimization and systems summary

| Quantity | Final or aggregate value |
|---|---:|
| Stop reason / cap overrun | data cap / 6.5 s |
| Updates / games | 38 / 19,456 |
| Learner / engine decisions | 501,367 / 1,005,525 |
| Active / wall time | 3,438.7 / 3,606.5 s |
| Whole-run learner throughput | 139.0 decisions/s wall; 145.8 decisions/s active |
| Final on-policy training score | 0.4277 |
| Full-rollout approximate KL | mean 0.00275; max 0.02252; final 0.00215 |
| Actor KL early stops | 1 of 38 updates |
| Actor pre-clipping gradient norm | mean 2.33; range 1.19–4.51; final 2.17 |
| Actor clipped fraction | mean 0.996; exactly 1.0 on 36/38 updates |
| Critic explained variance | mean 0.060; max 0.126; final 0.060 |
| Final critic MSE | 0.851 |
| Final collection / PPO / iteration time | 7.63 / 76.55 / 84.18 s |

PPO remained stable by its reliable full-rollout KL diagnostic: only update 6
crossed the 0.02 threshold, and the final KL was 0.00215. Persistent actor
gradient clipping is real but did not coincide with exploding norms, repeated KL
stops or policy collapse. It remains a hyperparameter/control question for a
separate experiment, not a post-hoc reason to reject this pilot.

The critic remained weak rather than steadily improving. Its explained variance
is measured before fitting each fresh rollout, so roughly 0.06 means the incoming
critic removed only about 6% of terminal-return variance on average and at the
final update. This is plausible with undiscounted terminal `+1/-1` targets,
unobserved future chance and opponent actions, and a changing policy, but it is
not strong value prediction. Before tuning it, record post-update same-rollout EV,
next-rollout EV, and critic gradient-clipped fraction to distinguish optimization
underfit from limited generalization or irreducible target noise.

The logged draw-action rate declined from 0.3758 to 0.2939, but it counts every
Draw action. It therefore does not show that 29% of decisions rejected a playable
card and cannot justify a Draw penalty. Split future logging into forced draws,
optional-draw opportunities, and `P(Draw | Draw and a Play are both legal)` before
considering an explicitly separate, annealed shaping experiment.

## Decision and next work

The pilot is successful as a learning and systems-health diagnostic. Preserve its
final/best actor at `checkpoints/final_actor.pt` (the final update was also the DEV
selection winner). Before choosing another training intervention:

1. fix the final-minibatch-only surrogate/entropy/ratio-clip diagnostics recorded
   above, without inventing replacements for this historical run;
2. split the misnamed draw-action metric and add post-fit critic diagnostics;
3. validate the frozen final actor once on the preregistered 5,760-deal diagnostic
   panel against the same-seat Heuristic reference; and
4. only then choose a declared follow-up among more data, critic optimization,
   gradient-cap calibration, or an optional-draw curriculum.

The high-precision DEV command is:

```sh
uv run uno rl devsize FULL_UNO B 0 5760 0.005 1 \
  logs/rl/FULL_UNO_B_seed0_mb512_cpu_1789786184/checkpoints/final_actor.pt
```

Do not spend a fresh registered EVAL block on this single-seed development choice.
