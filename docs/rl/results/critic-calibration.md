# Critic calibration as a win probability (play demo, phase 0)

## Question

Is `(v + 1) / 2` from a full checkpoint's critic a calibrated win probability
against each rulebot-league member, and at which points in the game? The play demo
(`docs/play/plan.md`) wanted to show the critic as a live win-chance curve.

## What the critic is trained to predict

`uno/rl/algo/ppo.py`: with the shipped settings (`gamma = 1`, `gae_lambda = 0.95`)
the critic is fitted to lambda-returns built from `V_old` and the raw terminal
return (no AIVAT correction). Their fixed point is the expected terminal return of
the learner's policy from that state, against the league *average*: the critic sees
L2 inputs of its own seat and not the opponent's identity. Draws are ~0 (0.2% at
most in these runs), so `(v + 1) / 2` reads as a win probability, biased per opponent
by construction.

## Decision rule (declared before running)

Show the raw critic if every per-opponent bin (10 equal-width bins) with `n >= 200`
is within 0.05 of its empirical rate. Otherwise fit a per-opponent Platt correction
`sigmoid(a v + b)` on block 0 and check it with the same rule on block 1. If even
corrected it misses, show rollout values only.

## Configuration

`uv run python -m uno.play.calibration CKPT --games 1000 --workers 4`
(`uno/play/calibration.py`). Each checkpoint's own actor plays each of the four
frozen league bots from both seats on two coupled blocks of the DEV diagnostic
partition, deals `[950000, 951000)` (block 0) and `[951000, 952000)` (block 1),
1,000 games per cell per block: 16,000 games per checkpoint. Streams are the
evaluator's (`EVAL` key, seed 0); the critic is evaluated on the collector's
critic features at every learner decision. Units: score, uncertainty is a
per-bin binomial SE (in the JSON). Records and full reports:
`.cache/play/calibration_{arm_C_relative,B_seed0}.{npz,json}`.

| Label | Checkpoint | Actor |
|---|---|---|
| rel-C | `arm_C_relative/checkpoints/best.pt` (run `1b115b9c`, update 879, 9.0M decisions) | arm C, relational |
| B | `logs/FULL_UNO_B_seed0_nmb32_bf16_cuda_1790104526/checkpoints/best.pt` | arm B, relational |

## Result

Worst gap is over bins with `n >= 200`; phase gaps are for turn `< 10 / 10-29 / >= 30`;
Brier is against a constant per-opponent rate on the same decisions.

**rel-C** (344,343 decisions)

| Opponent | Win rate | Mean pred | Worst gap | By phase | Brier / const | EV | Platt held-out gap |
|---|---:|---:|---:|---|---:|---:|---:|
| heuristic_memory | 0.675 | 0.740 | 0.161 | 0.156 / 0.098 / 0.271 | 0.1877 / 0.2195 | 0.162 | 0.202 |
| rulebot_transfer_toy_v1 | 0.714 | 0.750 | 0.144 | 0.051 / 0.081 / 0.183 | 0.1790 / 0.2040 | 0.127 | 0.143 |
| rulebot_full_dev_v1 | 0.684 | 0.747 | 0.143 | 0.107 / 0.124 / 0.211 | 0.1857 / 0.2161 | 0.157 | 0.078 |
| rulebot_unpredictable | 0.712 | 0.748 | 0.105 | 0.072 / 0.078 / 0.152 | 0.1771 / 0.2051 | 0.141 | 0.066 |

**B** (333,537 decisions)

| Opponent | Win rate | Mean pred | Worst gap | By phase | Brier / const | EV | Platt held-out gap |
|---|---:|---:|---:|---|---:|---:|---:|
| heuristic_memory | 0.650 | 0.615 | 0.135 | 0.092 / 0.119 / 0.195 | 0.1924 / 0.2275 | 0.158 | 0.059 |
| rulebot_transfer_toy_v1 | 0.679 | 0.624 | 0.216 | 0.163 / 0.140 / 0.253 | 0.1976 / 0.2179 | 0.104 | 0.071 |
| rulebot_full_dev_v1 | 0.655 | 0.626 | 0.173 | 0.079 / 0.100 / 0.247 | 0.1934 / 0.2259 | 0.146 | 0.043 |
| rulebot_unpredictable | 0.685 | 0.628 | 0.218 | 0.112 / 0.134 / 0.291 | 0.1947 / 0.2158 | 0.111 | 0.059 |

Reliability against heuristic_memory, `(bin, n, mean prediction, empirical rate)`:

| Bin | rel-C n | pred | rate | B n | pred | rate |
|---|---:|---:|---:|---:|---:|---:|
| 0.0 | 872 | 0.033 | 0.194 | 1,988 | 0.033 | 0.168 |
| 0.1 | 857 | 0.153 | 0.187 | 2,280 | 0.154 | 0.268 |
| 0.5 | 7,215 | 0.554 | 0.482 | 11,151 | 0.552 | 0.581 |
| 0.9 | 19,708 | 0.957 | 0.915 | 6,981 | 0.953 | 0.954 |

**Decision: rollout values only** for both checkpoints. Neither raw critic passes
(worst gaps 0.105-0.218). Per-opponent Platt fails on held-out data for all four
rel-C opponents (0.066-0.202) and for three of four B opponents (0.043-0.071).

## Interpretation

- Both critics carry real signal: Brier beats the constant rate for every opponent
  and phase, and EV is 0.10-0.16. Low EV is expected in a luck-heavy game.
- The failures differ in kind. rel-C's critic is **optimistic overall** (mean
  prediction 0.74-0.75 against win rates 0.68-0.71) and overconfident at both ends.
  B's is **pessimistic overall** (0.62-0.63 against 0.65-0.69) but sharper at the top
  bin. Both are worst **late in the game** (turn >= 30), and over the whole game
  predictions in the lowest bin (mean 0.03) still win 17-19% of the time.
- A single-slope Platt fit cannot fix a phase-dependent error, which is probably
  why it fails on held-out data. A correction conditioned on phase might pass. It was not
  in the declared rule and was not tried here.
- Hypotheses, not tested: the offset may come from the gap between the training
  policy the critic tracked and the `best` actor evaluated here, or from lambda
  bootstrapping toward `V_old`. The late-game overconfidence may come from few
  long-game training samples.

## Limitations

One checkpoint per arm and one training seed each; DEV data; only the `best`
checkpoints; bins are equal-width over predictions, so the extreme bins are small
(still >= 200). This measures the critic as a win-probability readout. It says
nothing directly about its quality as a PPO baseline, where only the advantage's
variance matters.

## Next decision

The demo shows rollout values (`uno/play/analysis.py`). For RL: whether the late-game
critic error matters for advantage quality is open; the saved per-decision records
(`v`, turn, final return) are enough to start on it without new games.
