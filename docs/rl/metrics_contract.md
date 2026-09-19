# RL metrics contract

**Scope, 2026-09-19:** this is the naming, unit, weighting, and uncertainty
contract for `uno rl train|resume|pilot|ablation`, plus the derived systems and
panel-sizing diagnostics. A code change that changes a definition must update
this file in the same change. Configuration, seeds, hashes, and lineage are
provenance, not metrics.

## Common conventions

Let `G` be games and `T` learner decisions in one training rollout, `R_g` the
learner's terminal return in `[-1, 1]`, and `S_g = (R_g + 1) / 2` the score in
`[0, 1]`; draws score `0.5`. A *cell* is one opponent/learner-seat pair. Full UNO
strength aggregates have eight equally weighted cells. A learner decision is an
actor choice; an engine decision is an action by either player.

The local `metrics.csv` names below are canonical. MLflow uses namespaced aliases
from `uno/rl/train.py`; `primary/*` entries duplicate selected metrics rather than
define new ones. `step` is cumulative learner decisions. `wall_time` is the Unix
timestamp at logging; `run_wall_seconds` is elapsed monotonic time in the current
process segment. Blank/`None` means unavailable, never zero. Training rows occur
once per completed update; evaluation rows occur at the initial, scheduled, and
final boundaries, so unrelated columns are blank on each row type.

## Strength and uncertainty

| Metric | Definition |
|---|---|
| `train_score_<opponent>_s<seat>` | Mean `S_g` in that cell in the current on-policy rollout. |
| `train_score` | Equal-weight mean of the current rollout's cell scores. It is a noisy training diagnostic, not a selection metric, and has no reported interval. |
| `dev_<opponent>_s<seat>` | Mean score in that cell on the fixed DEV panel. |
| `dev_league_mean` | Equal-weight mean of the DEV cell means. Equivalently, first average all cells sharing each deal, then average deals. |
| `dev_se` | Sample SD of those per-deal league means divided by `sqrt(dev_deals)`. The descriptive 95% CI is `dev_league_mean +/- 1.96 * dev_se`. |
| `dev_deals` | Distinct deals per cell, not total games; total games are `cells * dev_deals`. |
| selected DEV checkpoint | Largest `dev_league_mean`; ties within `1e-6` go to the earlier update. Its DEV interval is post-selection and descriptive. The final checkpoint is always reported separately. |
| paired difference `A - B` | On shared panels, form each policy's equal-cell per-deal mean, subtract within deal, then take the mean and `SD/sqrt(number of deals)`. Never treat cells or seed-by-deal rows as independent. |

Game-sampling uncertainty is the deal-level SE/CI above. Training-seed variation
is reported separately from final scores across seeds; it is never folded into
`dev_se`. The Stage 8 primary estimand is the paired-seed mean of final-checkpoint
`B - A` at the equal learner-decision budget.

## Optimization diagnostics

All actor distribution diagnostics are recomputed after the update over every
decision in that rollout, under one coherent final policy. Write
`log r_t = log pi_new(a_t|x_t) - log pi_old(a_t|x_t)` and
`A_t = R_g - V_old(x_t)`.

| Metric | Definition |
|---|---|
| `surrogate` | `(T/G) * mean_t[min(r_t A_t, clip(r_t, 1-eps, 1+eps) A_t)]`; game-average-sum units. |
| `entropy` | Mean legal-action policy entropy per learner decision, in nats. |
| `approx_kl` | Sample mean `((r_t - 1) - log r_t)`; nonnegative sampled diagnostic, not exact KL. |
| `clip_fraction` | Fraction of rollout decisions with `abs(r_t - 1) > eps`. |
| `max_abs_log_ratio` | Maximum `abs(log r_t)` over the rollout. |
| `actor_grad_norm`, `critic_grad_norm` | Mean pre-clipping global gradient norm over that update's optimizer steps. |
| `actor_clipped_fraction`, `critic_clipped_fraction` | Fraction of optimizer steps whose pre-clipping norm exceeded that model's norm cap. |
| `actor_epochs`, `critic_epochs`; `actor_steps`, `critic_steps` | Epochs and minibatch optimizer steps completed in this update. Actor epochs may stop on KL; critic epochs still complete. |
| `early_stopped` | `1` iff actor epochs stopped because rollout-wide `approx_kl > target_kl`; else `0`. |
| `critic_mse` | Decision-mean MSE against undiscounted terminal `R_g`, averaged over the final critic epoch. |
| `critic_pre_update_ev` | `1 - Var(R-V_old)/Var(R)` over decisions before fitting. |
| `critic_post_update_same_rollout_ev` | The same statistic after fitting, on the same rollout; a fit diagnostic, not held-out prediction. |
| `return_mean`, `return_std` | Population mean/SD of terminal returns broadcast to decisions; long games receive more weight. |
| `advantage_mean`, `advantage_std` | Population mean/SD of frozen `R_g - V_old`; also decision-weighted. |

## Evaluation behavior

`dev_<name>` applies the following whole-panel summary to the evaluated actor.
The same names in panel-sizing output are also recorded for the reference policy.

| `<name>` | Definition |
|---|---|
| `game_draw_rate` | Fraction of games with learner return `0`. |
| `action_cap_rate` | Fraction ending at the ruleset action cap. |
| `mean_game_length` | Mean engine `state.turn` per game; `turn` counts plays and voluntary Draw actions, not Pass or penalty cards. |
| `forced_draw_rate` | Forced Draw actions divided by learner decisions. |
| `optional_draw_opportunity_rate` | Learner decisions where Draw and at least one Play were legal, divided by learner decisions. |
| `optional_draw_rate` | Draws taken divided by optional-draw opportunities: `P(Draw | Draw and Play legal)`. |
| `keep_rate` | Pass after drawing a playable card, divided by learner decisions. |
| `wild_play_rate` | Wild plays divided by learner decisions. |
| `action_card_rate` | Non-wild action-card plays divided by learner decisions. |
| `entropy` | Mean legal-action entropy per evaluated learner decision, in nats. |

Zero denominators yield `0` for these behavioral rates. Game rates weight games;
action rates and entropy weight learner decisions, not cells.

## Work, time, throughput, and storage

`update`, `games`, `engine_decisions`, and `optimizer_steps` are cumulative over
training only; DEV evaluation is excluded, and optimizer steps sum actor plus
critic steps. The `step`/learner-decision counter is likewise training-only.
`active_seconds` is cumulative completed-iteration time and survives resume; it
excludes evaluation, startup, checkpointing, and tracking. `actor_presentations`
and `critic_presentations` are **per update**, despite their MLflow `progress/`
namespace: `T * completed epochs`. `end_win`, `end_exhaustion`, and `end_cap` are
current-rollout terminal counts. The run summary also records the soft-cap
`stop_reason`, actual decision overshoot through `learner_decisions`, wall-time
`overrun_s`, evaluation count, and whether the final evaluation was reused.

| Family | Definition and unit |
|---|---|
| `collect_s`, `update_s`, `iteration_s` | Current-iteration elapsed wall seconds; iteration is collection plus update. |
| `critic_values_s`, `actor_update_s`, `critic_update_s` | Synchronized update-stage seconds. |
| `worker_s`; `simulation_s`, `feature_s`, `actor_inference_s`, `worker_assembly_s` | Sums over collector workers. Workers overlap, so these are work-seconds and must not be added to elapsed wall time. |
| `serialization_s`, `result_queue_put_s`, `result_queue_wait_s`, `deserialization_s`, `coordinator_assembly_s` | Parallel transfer-stage seconds. Queue wait is coordinator blocked time and overlaps worker work. |
| `drain_s` | Collection time after the final game was started: the straggler tail. |
| `iteration_fresh_decisions_per_s` | `T / iteration_s`. This is the primary end-to-end training rate before scheduled evaluation/checkpoint overhead. |
| `collect_decisions_per_s`, `update_fresh_decisions_per_s` | `T / collect_s` and `T / update_s`. |
| `actor_presentations_per_s`, `critic_presentations_per_s` | Presentations divided by the corresponding model-update seconds; excludes diagnostic forward passes. |
| `mean_inference_batch`; `inference_batch_calls|min|p50|p90|max` | Realized learner rows per collection actor call and its call-level distribution. |
| `rollout_bytes` | Bytes in the compact in-memory rollout arrays, not process memory. |
| `transfer_bytes` | Serialized worker-result payload bytes transferred this update. |
| `retried_bundles` | Worker bundles recollected after failure in this update. |
| `collection_torch_threads`, `update_torch_threads` | Actual Torch thread counts used; execution settings, not outcomes. |

The separate `systems` command additionally reports games/s, engine decisions/s,
per-process and sampled aggregate RSS in MiB, minimum macOS free-memory percent,
and swap range/growth. Aggregate RSS can double-count shared pages and is not a
physical unified-memory total. The ordinary training run currently records
`rollout_bytes` but does **not** sample process RSS.

The `devsize` command uses the same deal-level estimator for actor, reference, and
paired difference. Its `estimated_deals_for_target` is
`ceil(n * (observed_halfwidth / target_halfwidth)^2)`, bounded below by two; it is
a planning estimate, not a promised realized interval. Actor/reference/total
seconds are elapsed evaluation times.
