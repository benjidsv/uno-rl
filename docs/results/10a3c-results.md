# Step 10a.3C: the LBR attack instrument, built and calibrated (2026-09-18)

> On top of commit `b00c2f7`. Reruns:
> `uv run uno lbr exact TOY`, `uv run uno lbr games TOY 200 4`,
> `uv run uno lbr pilot 200 32 2 "" 100`, `uv run uno lbr eval 4000`,
> `uv run uno report TOY heuristic|random|nash`, `uv run pytest tests/eval/test_attack.py`.
> New code: `uno/eval/attack.py`, `uno/compiled/flaws.py`, `uno/experiments/lbr.py`,
> `tests/eval/test_attack.py`; the attack block in `uno/experiments/report.py`; EVAL
> block 02 in `uno/eval/seeds.py`. Contract: [spec section 32](../spec.md).

## What this is for

An attack's **achieved score in real games** lower-bounds the best-response score against a frozen
target, so even a weak or biased planner gives a *valid* bound. The failure mode is not invalidity
but **insensitivity**: a blunt instrument reports a small number, and a small number looks exactly
like a safe target. The toys are where that can be settled, because the exact best response is
computable there. So the bulk of this step is calibration against known answers, not the attack.

## 1. The interface, and the one restriction that is measured rather than declared

`Budget(continuation, worlds, depth, belief, seed)` + `attack.build(rs, target, budget, leaf, ...)`,
with four families. Only `response` reads the target — it solves each determinized world as a best
response to the target's real policy — and only it is therefore restricted:

| continuation | reads the target? | supported targets |
|---|---|---|
| `response` | yes, inside every world | position-only targets only |
| `surrogate` | no: a fixed stand-in opponent | any |
| `minimax` | no: assumes a clairvoyant opponent | any |
| `rollout` | no: worlds played out by fixed L1 policies | any |

**Why the restriction exists, measured.** Inside a determinized world the opponent's
`initial_hands` / `own_draws` are fabricated, so a target that reads its own private history is
asked about information sets that never happened. This does not raise. Queried at opponent nodes
inside determinized worlds on `TOY`, the tabulated Nash strategy **missed the keying at 233 of 293
queries (79.5 %), and every single miss returned a point mass on the first legal action**. A
`response` attack on such a target would therefore be best-responding to "plays its first legal
action" — a silently insensitive attack, which is the exact thing this step exists to rule out.

`attack.supports` probes the target (same position, private history altered) and `check_supported`
raises `UnsupportedTarget`. Measured on `TOY`: the four rulebots change at **0 of 105** probes,
every tabulated strategy at **81–105 of 105**.

**A distinction I initially got wrong.** `supports` is *not* an L1-validity test. An L1 policy is
entitled to read its own initial hand and own draws — and since those plus the public plays
*determine* its current hand, any policy that rebuilds its position from the L1 key answers
differently when they are perturbed, for a legitimate reason. My own attacker failed the check when
I first used it this way. L1 validity is `attack.hidden_state_probe`, which substitutes a different
hidden world behind an identical L1 observation; the attacker passes it at 0 of 54 decisions moved.

Worth stating: the *exact* instrument (`pimc_n`) takes the target as a strategy over the keying,
correctly keyed, so it **can** respond to a memoryful target. The refusal is a property of
**sampling**, not of best response.

## 2. Exact calibration on the toys

An attacker is a deterministic function of its information state, so it can be tabulated and scored
by one `evaluate`: the attack score, the BR L1 ceiling and the gap are then **all exact**, with no
game sampling. Cost 0.1–0.8 ms per information set (~2–9 s for all of `TOY`; 92–676 s for
`TOY_FULL`, which is why the grid below is `TOY` only — see the limitations).

`uno.compiled.flaws` grafts **one** symmetry-closed action class onto a Nash strategy, so the
exploitability that exists is concentrated there and cannot be collected without finding that class.
Random and the Heuristic are weak everywhere and cannot distinguish a sharp attack from a blunt one.
Exact NashConv of the grafts, score points:

| flaw | `TOY` | `TOY_FULL` | `TOY2_SKIP` |
|---|---|---|---|
| `never_pass` | 1.60 | 1.63 | 1.90 |
| `always_pass` | 12.07 | 24.51 | 26.20 |
| `always_wild_play` | 13.98 | 12.50 | 10.81 |
| `never_wild_play` | 27.47 | 80.10 | 23.33 |
| `never_action_card` | — | — | 36.01 |

`never_pass` is the **near-Nash** case: declining a playable drawn card is nearly free, so a correct
attack must report a *small* number there. A no-op graft raises rather than returning an unflawed
"flawed" target.

**The attacks** (`TOY`, w = 16, `prior` belief, depth 40 so every world is solved exactly).
`missed` = BR L1 − achieved; `found` = share of the range from the same-seat Random baseline to
BR L1 (the report card's normalised scale, never selected on); the bound is `b0 + b1 − 1`:

| target | family | achieved s0 / s1 | missed s0 / s1 | found | NashConv ≥ | of true |
|---|---|---|---|---|---|---|
| random (true 34.47) | response | 0.7894 / 0.5453 | 0.40 / 0.60 | 0.97 / 0.97 | 33.47 | **97.1 %** |
| | surrogate | 0.7882 / 0.5427 | 0.52 / 0.85 | 0.96 / 0.96 | 33.09 | 96.0 % |
| | minimax | 0.7890 / 0.5389 | 0.44 / 1.24 | 0.97 / 0.94 | 32.79 | 95.1 % |
| | rollout | 0.7915 / 0.5228 | 0.19 / 2.85 | 0.99 / 0.86 | 31.43 | 91.2 % |
| heuristic (17.39) | response | 0.7339 / 0.4183 | 0.40 / 1.79 | 0.97 / 0.90 | 15.21 | 87.5 % |
| | surrogate | 0.7351 / 0.4183 | 0.28 / 1.79 | 0.98 / 0.90 | 15.33 | 88.1 % |
| | minimax | 0.7319 / 0.4075 | 0.60 / 2.86 | 0.96 / 0.84 | 13.94 | 80.2 % |
| | rollout | 0.7339 / 0.4008 | 0.40 / 3.53 | 0.97 / 0.80 | 13.47 | 77.4 % |
| passer0.5 (40.08) | response | 0.8189 / 0.5691 | 0.12 / 1.16 | 0.99 / 0.94 | 38.80 | 96.8 % |
| | surrogate | 0.8105 / 0.5569 | 0.96 / 2.38 | 0.93 / 0.87 | 36.74 | 91.7 % |
| | minimax | 0.8140 / 0.5614 | 0.61 / 1.94 | 0.96 / 0.90 | 37.53 | 93.7 % |
| | rollout | 0.8106 / 0.5610 | 0.95 / 1.97 | 0.93 / 0.90 | 37.16 | 92.7 % |
| **nash (0.00)** | response | *refused* | | | | |
| | surrogate | 0.6154 / 0.3739 | 0.42 / 0.65 | 0.97 / 0.96 | **−1.06** | — |
| | minimax | 0.6180 / 0.3768 | 0.16 / 0.36 | 0.99 / 0.98 | **−0.51** | — |
| | rollout | 0.5932 / 0.3580 | 2.64 / 2.24 | 0.82 / 0.87 | **−4.88** | — |
| nash_never_pass (1.60) | surrogate | 0.6178 / 0.3862 | 0.30 / 0.91 | 0.98 / 0.95 | 0.40 | 24.9 % |
| | minimax | 0.6192 / 0.3839 | 0.16 / 1.13 | 0.99 / 0.93 | 0.31 | 19.4 % |
| | rollout | 0.5950 / 0.3664 | 2.58 / 2.88 | 0.83 / 0.83 | −3.86 | — |
| nash_always_pass (12.07) | surrogate | 0.6543 / 0.4338 | 1.40 / 1.86 | 0.90 / 0.88 | 8.81 | 73.0 % |
| | minimax | 0.6616 / 0.4386 | 0.67 / 1.39 | 0.95 / 0.91 | 10.02 | 83.0 % |
| | rollout | 0.6264 / 0.4163 | 4.19 / 3.62 | 0.71 / 0.77 | 4.27 | 35.4 % |
| nash_never_wild_play (27.47) | surrogate | 0.7928 / 0.4520 | 1.82 / 1.18 | 0.89 / 0.94 | 24.48 | 89.1 % |
| | minimax | 0.8071 / 0.4536 | 0.39 / 1.01 | 0.98 / 0.94 | 26.07 | **94.9 %** |
| | rollout | 0.7807 / 0.4352 | 3.02 / 2.86 | 0.82 / 0.84 | 21.59 | 78.6 % |

Readings:

- **The instrument recovers 77–97 % of each target's exact NashConv**, and misses 0.12–4.19 score
  points of BR L1 per seat. It is a real lower bound, not a tight one, and it is never reported as
  exploitability.
- **Against Nash every family returns a negative bound** (−0.51 to −4.88), which clamps to the
  trivial 0. That is the correct behaviour for a lower bound on 0 and the sanity check that the
  instrument does not manufacture exploitability.
- **The rollout family is consistently the weakest** (77.4–92.7 % where the searches reach
  91–97 %), and it is the only family that reports a *negative* bound against the genuinely flawed
  `nash_never_pass`. Having it in the set is what prevents one family's blind spot from being read
  as target safety — and it earns its place by being visibly worse, not by agreeing.
- `minimax` beats `surrogate` on the grafted targets (94.9 % vs 89.1 % on `never_wild_play`) but
  loses on the rulebots. Reading nothing about the opponent is not uniformly worse here.

**The world ladder**, pre-declared on the **two-seat sum** (the reported quantity), not per seat:

| rung | w = 1 | w = 4 | w = 16 |
|---|---|---|---|
| heuristic / response, missed sum | 5.16 | 3.73 | **2.18** |
| nash_never_wild_play / surrogate, missed sum | 8.05 | 4.43 | **2.99** |

Per-seat `missed` is *not* monotone (heuristic seat 0 went 0.87 → 0.99 → 0.40), and a first draft of
this experiment checked exactly that and failed. The attacker's own seed moves a single seat by
0.6–0.8 points at these budgets, so a per-seat wobble of a few tenths is below the instrument's own
resolution: checking it would be checking noise.

**Non-vacuity.** The attack separates the flawed target from the near-Nash one by **24.08 points**
(24.48 against 0.40). A vacuous attacker passes every validity check above and fails this one.

**The attacker's own seed is a reported uncertainty.** A built attacker is *one realization* of its
keyed world sampling, not the averaged behavioural policy that `pimc_n` computes. On `TOY` vs Random:

| | seeds | spread (sd) | mean | `pimc_n` |
|---|---|---|---|---|
| w = 4, seat 0 / 1 | 6 | 0.84 / 0.58 pts | 0.7855 / 0.5287 | 0.7852 / 0.5275 |
| w = 16, seat 0 / 1 | 3 | 0.12 / 0.72 pts | 0.7902 / 0.5426 | 0.7899 / 0.5396 |

So an attack score carries two independent uncertainties — the deals and the attacker's seed — and
the seed mean lands on the exact behavioural average, which is the check that the sampled attacker
is the object the exact theory describes.

## 3. The sampled path, checked against the exact one

`uv run uno lbr games TOY 200 4`, playing the *same* `Symmetrized` object that was tabulated:

| seat | played (200 deals) | exact on those deals | z | whole-game exact |
|---|---|---|---|---|
| 0 | 0.7575 | 0.7684 | **−0.52** | 0.7881 (1.96 pts of deal luck) |
| 1 | 0.5350 | 0.5295 | **+0.26** | 0.5194 (1.01 pts) |

The comparison is against `scores_on_seeds`, the exact value **on the same deals** — the calibrated
scale of spec section 14. Against the whole-game value it would be off by the 1–2 points of pure
deal luck shown in the last column, which is not an error in the attacker.

**Search Q is not the achieved score, and the comparison needs care.** At the attacker's first
decision (the only Q comparable to a whole-game score), over the 52 games with a real root choice:
Q 0.7614 against an achieved 0.7885, a paired difference of **−2.70 ± 5.06 points (z −0.53)**. Not
resolved, so **its sign carries no finding**; the point is only that the two are different objects,
and only the achieved score is a bound. Two drafts of this line were wrong before this one: the
first averaged Q over decisions at every depth and compared it to a root score, and the second
paired 52 Q values against the scores of all 60 games.

Also checked: the attacker is L1-valid (0 of 54 decisions moved under a hidden-world swap), reports
a deterministic and repeatable policy (AIVAT's row correction needs this), and an attacker built
twice replays its games exactly.

## 4. The report card (item 6)

`uv run uno report` now prints per-seat attack scores, the budget, the seed spread and the NashConv
lower bound, with the pre-declared check that the bound stays **below** the exact NashConv:

| row | attack s0 / s1 | NashConv ≥ | exact NashConv | recovered | family |
|---|---|---|---|---|---|
| heuristic | 0.7321 / 0.4224 | 15.45 | 17.39 | 89 % | response, w = 8 |
| random | 0.7828 / 0.5279 | 32.51 | 34.47 | 94 % | response, w = 8 |
| nash | 0.6031 / 0.3691 | 0.00 (raw −2.78) | 0.00 | n/a | surrogate (response refused) |

All pinned card values still reproduce to the digit; 0 failed checks on all three. The block runs by
default only where an attacker tabulation is cheap (≤ 50,000 information sets).

## 5. Full UNO: the bounded DEV pilot

`uv run uno lbr pilot 200 32 2 "" 100`, DEV seeds only, d = 2, `voids` belief. Single-core cost
measured separately: **0.26 s/game** against a rulebot target and **8.11 s/game** against a target
that is itself a search (~31×, since every attacker node re-runs the target's own search).

| target | family | w | n | b0 | b1 | NashConv ≥ |
|---|---|---|---|---|---|---|
| random | response | 32 | 200 | 0.8300 ± 0.0266 | 0.7900 ± 0.0289 | +62.00 ± 3.72 |
| heuristic | response | 8 | 200 | 0.6100 ± 0.0346 | 0.6250 ± 0.0343 | +23.50 ± 4.37 |
| heuristic | response | 32 | 200 | 0.6650 ± 0.0335 | 0.6400 ± 0.0340 | +30.50 ± 4.02 |
| heuristic | surrogate | 32 | 200 | 0.6300 ± 0.0342 | 0.6200 ± 0.0344 | +25.00 ± 4.24 |
| column-free PIMC row | surrogate | 32 | 100 | 0.5200 ± 0.0502 | 0.4900 ± 0.0502 | **+1.00 ± 6.11** |

- **Sensitivity:** the Heuristic bound climbs +7.0 points from w = 8 to w = 32, so the bound is
  budget-limited, not target-limited, at this scale.
- Reading the column is worth ~5.5 points of bound here (`response` 30.50 vs `surrogate` 25.00).
- **Against the column-free PIMC row the attack finds nothing** (+1.00 ± 6.11, clamped to 0). That
  is a reported result, not a finding about that row: the interval is far too wide to bound
  anything, and a weak bound is never evidence of safety. Resolving a 5-point bound there would
  need ~1,340 deals (~45 min), which this step does not spend — the plan's instruction was not to
  keep expanding an experiment until it turns positive.
- A **seat-aliasing guard** is now permanent in both full-UNO sections: the two seatings of a deal
  must not produce identical per-deal scores. It was added because a 120-deal run printed b0 and b1
  identical to four decimals with identical standard errors; checked directly, the seatings are
  genuinely separate (13 of 40 deals disagree) and that was a coincidence, but it is exactly the
  kind of coincidence that should not have to be re-diagnosed by hand.

## 6. Full UNO: the frozen confirmatory block

Frozen *before* EVAL block 02 was claimed, from the pilot's variances: `response` continuation
against position-only targets, w = 32, d = 2, `voids` belief, attacker seed 0, 4,000 deals, both
seatings. The pilot put the bound's standard error at ~4.0 points on 200 deals, so one block takes
it to ~0.9 — ample for the 20–60 point bounds the rulebot targets produce, and enough to resolve
even a 5-point bound at z ≈ 5.6. The column-free PIMC target is deliberately **not** on this block.

`uv run uno lbr eval 4000`, **EVAL block 02** (1,008,000–1,011,999), 0 failed checks:

| target | n | b0 | b1 | NashConv ≥ | reported |
|---|---|---|---|---|---|
| random | 4,000 | 0.7730 ± 0.0066 | 0.7692 ± 0.0067 | **+54.22 ± 0.90** | 52.46 |
| heuristic | 4,000 | 0.6390 ± 0.0076 | 0.6268 ± 0.0076 | **+26.57 ± 0.96** | 24.69 |

These are the step's headline full-UNO numbers: the first exploitability-style measurements this
project has on full UNO, and lower bounds on NashConv rather than exploitability, since the
seat-specific Nash values are unknown and 0.5 was never substituted for one.

**The DEV pilot was optimistic, which is the whole reason for the block.** It put the same two bounds
at +62.00 ± 3.72 and +30.50 ± 4.02 — about 7.8 and 3.9 points above the confirmatory values, i.e.
roughly 2σ and 1σ of the pilot's own standard error. Nothing was selected on block 02, so these are
the reportable figures and the pilot's are DEV readings.

**Cross-block check.** This attacker *is* the published working-set PIMC row (response continuation,
w = 32, `voids`), so its score against Random can be read against two earlier blocks: 0.7730 ± 0.0066
/ 0.7692 ± 0.0067 here, against 0.7860 ± 0.0065 / 0.7685 ± 0.0067 on block 01 (10a.2's instrument
row) and 0.7813 / 0.7603 on the now-spent block 00. Consistent at ~1–1.5σ across three disjoint
blocks, so the row has not drifted and the attack harness reproduces the established agent.

Cost figures for this step come from the dedicated single-core probe (0.26 / 8.11 s/game) and the
uncontended pilot, **not** from this run: the machine was running an unrelated job throughout, so the
0.04–0.05 wall s/game printed here is contended throughput and not a cost measurement.

## 7. Limitations

- **`TOY` only for the exact grid.** `TOY_FULL` costs 92–676 s per attacker tabulation and
  `TOY2_SKIP` 64–333 s, so the full target × family grid was run on `TOY` alone. The graft
  exploitability table covers all three, but the *attack* numbers do not. Free play is where the
  continuation rule mattered most historically (step 9.5e), so a trimmed `TOY_FULL` grid is the
  obvious next extension.
- **The bound is loose by construction.** 77–97 % of a known NashConv on a toy says nothing about
  what fraction it would recover against a trained checkpoint, whose weaknesses need not resemble a
  rulebot's or a grafted one's.
- **Recurrent targets are not supported** by the `response` continuation, and the surrogate families
  attack them only as position-free opponents. Per the plan this is a Step 11 integration gate, not
  a prerequisite finished here.
- **No AIVAT on the attack rows.** The full-UNO numbers are raw paired scores; AIVAT would reduce
  their variance and was not wired in.
- The attacker is a PIMC search, so a larger bound needs a *stronger attack*, not more deals.
