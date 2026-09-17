# Results index: the current numbers

The headline numbers of every toy as the code produces them today, with the
step that measured them. Each step's file keeps its numbers as measured at the
time; where a later step changed an agent or a method, that file says so and
this page has the current value.

## Conventions

- **score**: P(win) of the named player with draws worth 1/2, i.e. `(return + 1) / 2`.
- **seat 0 / seat 1**: where the named player sits; seat 0 moves first. Every
  matchup is reported per seat ([step 3](03-census.md), rule 1).
- **seat gap**: the score of a profile at seat 0 minus its score at seat 1, in
  points. **Edge**: the seat-0 score minus 1/2 (half the gap for self-play).
- **BR Lx**: the score of a best responder at information level x against the
  named opponent, per responder seat.
- **exploitability** of a row agent at seat s: the BR L1 score against it minus
  the Nash score of the responder's seat, in points.
- **BR L1(random) as a row** is undetermined at the information sets Random
  never reaches. Its exploitability is shown as solved (first action orbit
  there), in brackets with those sets filled with the Nash strategy
  ([step 8.8](08.8-review.md)), and in square brackets with the least
  exploitable fill-in (CFR with the reached sets fixed, [step 9](09-results.md);
  an upper bound, tied reached sets are not freed).
- Seat gaps, exploitability and voluntary-draw numbers are under the exhaustion
  rule (fewer cards wins). Scoring exhaustion as a draw or as a loss for the stuck
  player moves them by up to 13 points on the forced toys and distorts free play
  ([step 9](09-results.md)).
- **Nash**: the CFR average strategy on the quotient tree; its NashConv (score
  points) goes with every number that uses it.
- The **Heuristic** is the symmetric one of [step 8.6](08.6-symmetry.md). Steps
  3-8.5 used an earlier version (ties on the lowest type id).

## Sizes

| ruleset | deck / pile | forced play | quotient histories | L1 sets | full tree | step |
|---------|-------------|-------------|--------------------|---------|-----------|------|
| `TOY` | 9 / 2 | yes | 33,646 | 12,036 | 1,252,040 histories, 558,656 L1 sets | [3](03-census.md), [8.6](08.6-symmetry.md) |
| `TOY_FULL` | 9 / 2 | no | 774,007 | 138,876 | | [8.6](08.6-symmetry.md) |
| `TOY2` | 10 / 3 | yes | 311,819 | 129,428 | | [8.7](08.7-toy2.md) |
| `TOY2_SKIP` | 10 / 3 | yes | 922,744 | 439,268 | | [8.7](08.7-toy2.md) |
| `TOY2_FULL` | 10 / 3 | no | 28,057,516 | 3,753,530 | | [8.7](08.7-toy2.md) |

## Nash value and seat gaps

| ruleset | Nash seat-0 score | solver, NashConv | seat gap: random vs random | heuristic vs heuristic | BR L1(random) vs random | Nash vs Nash |
|---------|-------------------|------------------|------|------|------|------|
| `TOY` | 0.6196 | CFR+ 1000, 0.001 pts | 29.9 | 38.7 | 24.2 | 23.9 |
| `TOY_FULL` | 0.5866 | CFR+ 1000, < 0.001 pts | 29.9 | 38.7 | 21.4 | 17.3 |
| `TOY2` | 0.5784 | CFR+ 1000, 0.000 pts | 17.5 | 27.7 | 16.1 | 15.7 |
| `TOY2_SKIP` | 0.5748 | CFR+ 1000, 0.000 pts | 16.8 | 15.3 | 15.1 | 15.0 |
| `TOY2_FULL` | 0.5564 | DCFR 150, 0.005 pts | 17.5 | 27.7 | 13.2 | 11.3 |
| `FULL_UNO` (sampled) | | | 0.0 ± 3.1 | 0.9 ± 3.1 | | |

`FULL_UNO`: 4000 held-out pairs per matchup (`uv run uno transfer`, [step 8.5](08.5-transfer.md); re-run in [step 8.9](08.9-fixes.md) with the step 8.6 Heuristic and action rng); random self-play seat 0 scores 0.5002 ± 0.0155, Heuristic self-play 0.5045 ± 0.0155.

## Best-response columns (score of the responder)

| ruleset | opponent | seat | BR L1 | BR L2 | BR L3 | L2 - L1 | L3 - L2 |
|---------|----------|------|-------|-------|-------|---------|---------|
| `TOY` | random | 0 | 0.7934 | 0.8188 | 0.8280 | +2.5 | +0.9 |
| | random | 1 | 0.5513 | 0.5937 | 0.5989 | +4.2 | +0.5 |
| | heuristic | 0 | 0.7378 | 0.7730 | 0.7798 | +3.5 | +0.7 |
| | heuristic | 1 | 0.4361 | 0.4774 | 0.4786 | +4.1 | +0.1 |
| `TOY_FULL` | random | 0 | 0.8017 | 0.8443 | 0.8590 | +4.3 | +1.5 |
| | random | 1 | 0.5882 | 0.6419 | 0.6941 | +5.4 | +5.2 |
| | heuristic | 0 | 0.7446 | 0.8060 | 0.8187 | +6.1 | +1.3 |
| | heuristic | 1 | 0.4732 | 0.5131 | 0.5437 | +4.0 | +3.1 |
| `TOY2` | random | 0 | 0.7305 | 0.7976 | 0.8191 | +6.7 | +2.2 |
| | random | 1 | 0.5694 | 0.6586 | 0.6804 | +8.9 | +2.2 |
| | heuristic | 0 | 0.6959 | 0.7728 | 0.7907 | +7.7 | +1.8 |
| | heuristic | 1 | 0.4937 | 0.5421 | 0.5587 | +4.8 | +1.7 |
| `TOY2_SKIP` | random | 0 | 0.7425 | 0.7865 | 0.8063 | +4.4 | +2.0 |
| | random | 1 | 0.5919 | 0.6277 | 0.6503 | +3.6 | +2.3 |
| | heuristic | 0 | 0.6941 | 0.7538 | 0.7738 | +6.0 | +2.0 |
| | heuristic | 1 | 0.5356 | 0.5638 | 0.5816 | +2.8 | +1.8 |
| `TOY2_FULL` | random | 0 | 0.7479 | 0.8443 | 0.8875 | +9.6 | +4.3 |
| | random | 1 | 0.6164 | 0.7508 | 0.8208 | +13.4 | +7.0 |
| | heuristic | 0 | 0.7118 | 0.8248 | 0.8620 | +11.3 | +3.7 |
| | heuristic | 1 | 0.5371 | 0.6080 | 0.6600 | +7.1 | +5.2 |
| | nash | 0 | 0.5565 | 0.6952 | 0.7640 | +13.9 | +6.9 |
| | nash | 1 | 0.4436 | 0.5528 | 0.6271 | +10.9 | +7.4 |

Against the Nash agent on `TOY` (the quotient CFR+ strategy, `uv run uno cfr TOY`): 0.6196 / 0.6624 / 0.6761 from seat 0, 0.3804 / 0.4239 / 0.4311 from seat 1. The full-tree strategy of [step 8](08-cfr.md) is another Nash strategy with the same L1 value and slightly different L2 / L3 columns (0.6630 / 0.6764, 0.4240 / 0.4311).

## Exploitability of the row agents (points, row at seat 0 / seat 1)

| ruleset | random | heuristic | BR L1(random): solved (Nash-filled) [least exploitable] |
|---------|--------|-----------|---------------------------------------------------------|
| `TOY` | 17.1 / 17.4 | 5.6 / 11.8 | 1.6 / 0.4 (1.4 / 0.3) [1.2 / 0.2] |
| `TOY_FULL` | 17.5 / 21.5 | 6.0 / 15.8 | 4.6 / 1.3 (3.1 / 2.4) [2.8 / 1.1] |
| `TOY2` | 14.8 / 15.2 | 7.2 / **11.7** | 1.7 / 2.2 (1.4 / 1.7) [1.4 / 1.6] |
| `TOY2_SKIP` | 16.7 / 16.8 | 11.0 / 11.9 | 1.3 / 1.7 (1.1 / 1.1) [1.0 / 1.0] |
| `TOY2_FULL` | 17.3 / 19.1 | 9.4 / 15.5 | 6.3 / 4.9 (3.4 / 3.3) [3.1 / 2.6] |

`uv run uno symmetry`, `uno toy2` and `uno validate` print both rows ([step 8.8](08.8-review.md)).

**Corrected in [step 10a](10a-results.md):** the Heuristic's `TOY2` seat-1 exploitability read
**11.8** here until the report card recomputed it as **11.7480**. The old figure was the difference
of two *rounded* values (BR L1 0.6959 minus Nash 0.5784 = 0.1175 exactly, which rounds up), where
the underlying doubles give 0.69587054 - 0.57839081. Only this cell was affected -- it is the one
that lands on a rounding boundary. [08.7](08.7-toy2.md) keeps its number as measured then, per the
convention above.

## Nash row and column (exact, score of the first-named, seat 0 / seat 1)

| ruleset | nash vs random | nash vs heuristic | random vs nash | heuristic vs nash | BR L1(random), as solved, vs nash |
|---------|----------------|-------------------|----------------|-------------------|-----------------------|
| `TOY` | 0.7871 / 0.5235 | 0.7328 / 0.4139 | 0.4765 / 0.2129 | 0.5861 / 0.2672 | 0.6149 / 0.3792 |
| `TOY_FULL` | 0.7595 / 0.5123 | 0.7065 / 0.4372 | 0.4877 / 0.2405 | 0.5628 / 0.2935 | 0.5560 / 0.4060 |
| `TOY2` | 0.7184 / 0.5330 | 0.6784 / 0.4578 | 0.4670 / 0.2816 | 0.5422 / 0.3216 | 0.5697 / 0.4099 |
| `TOY2_SKIP` | 0.7268 / 0.5545 | 0.6712 / 0.5059 | 0.4455 / 0.2732 | 0.4941 / 0.3288 | 0.5670 / 0.4160 |
| `TOY2_FULL` | 0.7048 / 0.5478 | 0.6671 / 0.4843 | 0.4522 / 0.2952 | 0.5157 / 0.3329 | 0.5407 / 0.4238 |

## The voluntary draw (free-play toys)

Draws with a play in hand, expected per game (seat 0 / seat 1 of the profile);
in brackets the part that is a *claim* (a draw on an empty pile, which ends the
game by exhaustion).

| ruleset | BR L1(random) vs random, random vs BR L1(random) | Nash vs Nash | BR L1 value of the draw vs random (seat 0 / seat 1) |
|---------|--------------------------------------------------|--------------|------------------------------------------------------|
| `TOY_FULL` | 0.029 / 0.184 (no claims) | 0.228 / 0.372 (claims 0.039 / 0.048) | +0.8 / +3.7 pts over `TOY`, none of it from claims |
| `TOY2_FULL` | 0.129 / 0.329 (no claims) | 0.248 / 0.495 (claims 0.055 / 0.079) | +1.7 / +4.7 pts over `TOY2`, none of it from claims |

## Evaluator variance (Heuristic vs Random, 4000 held-out pairs)

| ruleset | corr(seat 0, seat 1) | var ratio mirrored / raw | paired Heuristic - Random, var ratio | step |
|---------|----------------------|--------------------------|---------------------------------------|------|
| `TOY` | -0.30 exact | 0.70 exact | 0.47 / 0.49 | [7](07-mirror.md), re-sampled in [8.9](08.9-fixes.md) |
| `TOY2` | -0.31 exact | 0.69 exact | 0.42 / 0.45 | [8.7](08.7-toy2.md) |
| `FULL_UNO` | -0.10 sampled | 0.90 sampled | 0.89 / 0.89 | [8.5](08.5-transfer.md), re-sampled in [8.9](08.9-fixes.md) |

Near-identical policies on `FULL_UNO` ([step 9](09-results.md), `uv run uno pairing`, 4000 held-out seeds): the Heuristic minus a variant on the Heuristic's action-rng stream (5 % Random mixed in, or the opponent "low" at 1 card), against the same column, keeps 15-23 % of the unpaired variance with pile draws (1,250-1,800 deals for 3 points at 3 sigma with 80 % power, against ~8,000 unpaired) and 12-20 % with ticket draws; the same mixture on its own stream keeps 49-72 %.

## AIVAT ([step 9.5b](09.5b-results.md), [step 9.5c](09.5c-results.md))

Variance ratio AIVAT / raw on the same games, score of the row, seat 0 / seat 1.

| ruleset | matchup | value function, corrections | ratio | deals for 3 pts at 3σ, 80 % power (raw → AIVAT) |
|---|---|---|---|---|
| `TOY` | heuristic vs random | exact values, chance + both players + deal | 0 / 0 (every game gives the value) | |
| `TOY` | heuristic vs random | exact values, chance + both players, no deal | 0.42 / 0.49 | |
| `TOY` | heuristic vs random | fitted critic, no deal | 0.73 / 0.88 | |
| `MID_12` | heuristic vs random | fitted critic, no deal | 0.76 / 0.75 | |
| `MID_25` | heuristic vs random | fitted critic, no deal | 0.73 / 0.72 | |
| `FULL_UNO` | heuristic vs random | fitted critic (R² 0.11), no deal | 0.71 / 0.76 | 3,860 → 2,757 / 3,956 → 2,996 |
| `FULL_UNO` | heuristic vs heuristic_col | fitted critic (R² 0.14), no deal | 0.75 / 0.72 | 4,096 → 3,090 / 4,096 → 2,941 |
| `TOY` | heuristic vs random | fitted network (9.5c), no deal | 0.45 / 0.51 | |
| `TOY` | heuristic vs random | fitted network, chance + both players + deal | 0.04 / 0.06 | |
| `MID_12` | heuristic vs random | fitted network, no deal | 0.53 / 0.54 | |
| `MID_25` | heuristic vs random | fitted network, no deal | 0.51 / 0.50 | |
| `FULL_UNO` | heuristic vs random | fitted network (R² 0.15), no deal | **0.46 / 0.47** | 3,860 → 1,776 / 3,956 → 1,854 |
| `FULL_UNO` | heuristic vs random | fitted network, + deal (Monte Carlo deal mean) | 0.38 / 0.39 | 3,860 → 1,483 / 3,956 → 1,558 |
| `FULL_UNO` | heuristic vs heuristic_col | fitted network (R² 0.19), no deal | 0.51 / 0.52 | |

Deal model through AIVAT's deal term, 5,000,000 seeds per profile: every pooled |z| < 1 on `TOY`,
`TOY2`, `TOY2_SKIP` and `DUPS` (3σ ≈ 0.04-0.05 score points).

The **deal floor** ([step 9.5c](09.5c-results.md), `uv run uno pimc_value floor`): the share of the
outcome variance fixed by the dealt hands, which is the ratio AIVAT reaches with the exact L2 values
and every correction but the deal's. Heuristic vs Random, 2,000 EVAL deals x 32 pile determinizations,
seat 0 / seat 1: `TOY` 0.42 / 0.47, `TOY2` 0.38 / 0.40, `TOY2_SKIP` 0.46 / 0.40, `MID_12` 0.25 / 0.26,
`MID_25` 0.17 / 0.16, `FULL_UNO` **0.086 / 0.090**; against heuristic_col it is about twice as high
(`FULL_UNO` 0.17). The toy values match the exact ratios. So on full UNO an exact L2 value function
would need ~350 deals where raw needs ~3,900, and the fitted network reaches ~1,800.

## PIMC as an agent ([step 9.5d](09.5d-results.md))

**Exact PIMC-infinity on the toy world trees** (`uv run uno pimc exact`), gap to BR L1 in score
points, over 64 solves with 0 failed checks. The continuation rule matters far more than the belief:
solving each world as a best response to the fixed opponent (`posterior/response`) stays within
0.04-0.70 points of BR L1 everywhere, while classic two-sided PIMC (`minimax`) loses 0.15-3.21 on
the forced toys and **8.4-8.8 on `TOY_FULL`**, where the voluntary draw makes assuming a clairvoyant
opponent expensive. Modelling the opponent's reach is worth 0.0-0.6 points against Random but up to
2.6 against the Heuristic, and it *costs* exploitability (2.91 / 4.03 against 0.81 / 0.34 on `TOY`).

**PIMC-n** (`pimcn`): sampling n worlds instead of the whole belief costs 2-7 points at n = 1 and is
essentially paid off by n = 8-16 (gap 0.02-0.45 at n = 64). This is a statement about *toy* belief
sizes -- 2-23 consistent worlds at the decisions actually faced -- and does not transfer.

**Full UNO** (`sweep`, `uno`), raw score against Random, 8 cores:

| row | seat 0 | seat 1 |
|---|---|---|
| PIMC search, 32 worlds, depth 2 | **0.7033** | 0.5633 |
| PIMC rollout, 32 worlds | 0.5400 | 0.5000 |
| Heuristic | 0.5767 | 0.6433 |

More worlds help a great deal at first (0.415 -> 0.595 -> 0.695 at seat 0 for 2 -> 8 -> 32 worlds)
and then stop: past 32 the ladder is flat, with w = 128 only +3.0 / +6.0 points over w = 32 for four
times the cost, neither significant. Search beats rollouts at equal worlds, and a *small* world
count is actively harmful -- at 2 worlds, or as a rollout player at 8, PIMC scores below
`PlayFirstRandom`, because the argmax of a few noisy per-action estimates is biased towards whichever
action's noise came out high. Extra *depth* does not help either: paired at w = 32, depth 4 comes out
7.5 / 6.0 points below depth 2 (z -1.7 / -1.3, so the harm is not proven at 3 sigma) for 8.3x the
cost, and the point estimates fall monotonically with depth, which disfavours an odd-even leaf
artefact. So the usable setting is around **32 worlds at depth 2**: below it the agent is
noise-driven, above it neither more worlds nor more depth buys anything measurable, and where the two
compete at equal compute worlds win (w = 128 d = 2 beats w = 32 d = 3 at both seats).

**AIVAT on a search row** (`uno`, 300 EVAL deals, V = 9.5c's network, off-profile): variance ratio
0.54-0.62 against the Heuristic row's 0.41-0.52. For a deterministic search agent the row correction
is identically zero, so the gain comes from chance and the opponent only. This is the regime AIVAT
pays in: per game on one core a w = 32 search costs ~722 ms against the Heuristic's ~1.3 ms, so the
corrections are 4-11 % of the cost rather than several times it.

**The continuation rule** ([step 9.5e](09.5e-results.md), `uv run uno pimc sweep 3000 "" "s32x2r,s32x2,heu,s32x3r,s32x4r"`):
solving each determinized world as a best response to the column's known policy, instead of
two-sided minimax, is worth **+9.73 +- 1.04 points at seat 0 (z +9.3) and +4.57 +- 1.06 at seat 1
(z +4.3)** against Random at 32 worlds and depth 2, over 3,000 EVAL deals -- established at both
seats, the largest lever found on full UNO, and the one the exact toys pointed at. It is also
*cheaper* (131 against 179 wall-clock ms/game on 8 cores), because expectimax expands only the
opponent's support. Raw scores at 3,000 deals, with per-seat leaf networks (step 9.5d used the
seat-0 one at both seats):

| row | vs Random, seat 0 / seat 1 | vs Heuristic, seat 0 / seat 1 |
|---|---|---|
| PIMC search w=128 d=2, **response** | **0.7677 / 0.7403** | |
| PIMC search w=32 d=2, **response** | **0.7567 / 0.7167** | **0.6180 / 0.6010** (1,000 deals) |
| PIMC search w=32 d=2, two-sided | 0.6593 / 0.6710 | |
| Heuristic | 0.6163 / 0.6037 | |

So the PIMC row now beats the rulebot at **both** seats, against Random and head to head, where
step 9.5d's row lost seat 1. The seat asymmetry of that step is gone. AIVAT on the new row: variance
ratio 0.60-0.68 against the Heuristic row's 0.44-0.49, with the row correction identically zero
(a deterministic agent), and 0 failed checks. The row is **opponent-specific** -- it uses the
column's exact policy, so it is a ladder instrument, not a general player. Inference (a posterior
belief rather than a uniform determinization) was built in [step 9.5f](09.5f-results.md) -- though
**not** as the particle filter of the original design, which collapses; see the belief ladder
below, worth +2.83 / +4.40 points at both seats.

**The budget ladders** (step 9.5e, 3,000 EVAL deals per config, response continuation, paired):
**more worlds stop paying past 32 and depth never pays.** Worlds at depth 2 climb +5.57 / +3.33
points from w = 8 to w = 32 and then flatten -- w = 128 is ~+1.1 over w = 32 at z ~ 1.0, and w = 256
scores *below* w = 128 at both seats. Depth at w = 32 moves nothing (every difference within 1.5
sigma of zero) for 3.7x the cost at d = 3 and 13.5x at d = 4, which also disposes of 9.5d's
odd-even-artefact worry. At equal compute worlds beat depth: w = 128 d = 2 (530 ms/game) scores
0.7677 / 0.7403 against w = 32 d = 4 (1766 ms/game) at 0.7427 / 0.7170. Because this held after the
continuation rule and the seat networks were both fixed, the flattening is **not** an artefact of
those approximations, as 9.5d supposed. Step 9.5e concluded the binding constraint is therefore the
off-profile leaf value network (outcome R^2 0.07-0.25 on these games); [step 9.5f](09.5f-results.md)
supplied a **competing explanation not ruled out by that grid** -- 79.9 % of the worlds being
sampled were impossible, so w = 32 was effectively ~6 real worlds, well inside the steep part of
this very ladder.

**That ladder has since been re-run with `voids` on** ([09.5f](09.5f-results.md) section 7.2, 3,000
EVAL deals, unpaired), and the answer is neither of the two things the question expected. The
belief is a **level shift of ~+3.5 points at every world count, not a change of slope**: both
ladders climb hard to w = 32 and flatten after (`voids` 32 -> 64 is +1.20 / +0.77, z ~ 1). So
flatness never could have separated the explanations -- it is what both predict. What separates
them is a comparison the question did not ask for: **`voids` at w = 32 (0.7813 / 0.7603) outscores
`prior` at w = 128 (0.7677 / 0.7403) and at w = 256 (0.7660 / 0.7347)**, for a quarter and an eighth
of the cost. 9.5e's plateau was therefore **not a ceiling imposed by the leaf network** -- better
worlds went straight through it -- and the leaf network is demoted from "the residual explanation"
to one constraint among several.

**The usable setting is now w = 32 with `voids`**, superseding 9.5e's "64-128 worlds at depth 2".
For measuring *differences between variants* w = 32 is also the better instrument: w = 64 is 1.97x
the cost for an unestablished +1.20 / +0.77, and at a fixed wall clock w = 32 buys twice the deals,
so a 3-point effect reaches z ~ 3.2 where w = 64 reaches z ~ 2.2. It also sits on the flat part of
the ladder, where a comparison is insensitive to the exact world count. Standing caveat: this
ladder has been reopened twice (9.5e by the continuation rule, 9.5f by the belief), both times
because something changed what a *world is* -- re-check the shape whenever world quality changes.

| row | seat 0 | seat 1 | ms/game |
|---|---|---|---|
| PIMC response w=8 d=2 | 0.7010 | 0.6833 | 35.4 |
| PIMC response w=32 d=2 | 0.7567 | 0.7167 | 130.5 |
| PIMC response w=64 d=2 | 0.7573 | 0.7343 | 271.0 |
| **PIMC response w=128 d=2** | **0.7677** | **0.7403** | 530.0 |
| PIMC response w=256 d=2 | 0.7660 | 0.7347 | 1060.2 |

**D2 with a search Q** (`d2`, 150 EVAL deals): where the Heuristic's choices differ most from a
w = 32 search -- **action-card timing 68-73 % and wild handling 25-29 %**, i.e. ~97 % between them,
nearly seat-symmetric. The control row (the search itself) sits at -3.3 / -2.5 against the
Heuristic's +21, which is what licenses reading the split. Not a value decomposition: the shares are
the finding, not the totals.

**The opponent-hand belief** ([step 9.5f](09.5f-results.md), `uv run uno pimc belief 3000 32 2`):
3,000 EVAL deals against Random, w = 32, d = 2, response continuation, paired on the same deals.
Until this step the agent split the unseen cards **uniformly** (`uno.compiled.pimc`'s `prior`), i.e.
it did no opponent modelling at all.

| row | seat 0 | seat 1 | ms/game |
|---|---|---|---|
| PIMC response w=32 d=2, uniform (`prior`) | 0.7530 | 0.7163 | 126.5 |
| **PIMC response w=32 d=2, `voids`** | **0.7813** | **0.7603** | 139.6 |
| PIMC response w=32 d=2, `window` | 0.7907 | 0.7517 | 162.9 |

Paired against the prior: `voids` **+2.83 +- 0.95 (z +3.0) / +4.40 +- 0.99 (z +4.4)**, `window`
**+3.77 +- 0.94 (z +4.0) / +3.53 +- 1.02 (z +3.5)**. Inference is established at **both** seats and
is the second-largest full-UNO lever after the continuation rule. But `window - voids` is
+0.93 / -0.87 (z +1.0 / -0.9), **within noise and of opposite sign at the two seats**: against this
column the soft policy likelihood adds nothing measurable over the hard constraints. `voids` is the
setting -- it costs +10 % against `window`'s +29 %, and it reads no policy *probabilities*, unlike
`window`. It is **not** policy-free, though: see the scope correction below. (The `prior` row here,
0.7530 / 0.7163, reproduces step 9.5e's w = 32 figures of 0.7567 / 0.7167 within noise: the two runs
share a configuration, which is the cross-check that the rewiring left that path alone.)

Two things behind the number: the **particle filter over the deal** that `plan.md` specified does
not work (64 particles survive the opponent's first five plays 0.156 -> 0 -- a particle lives only
if it holds the played card, ~7/98 -- and the built version scored *worse than the prior*, Brier
0.167 against 0.085); and the void constraint alone rules out **79.9 %** of the uniform split's
worlds, so **w = 64 was effectively w = 10**, with the opponent's true hand violating it 0 times in
2,487 decisions.

**The belief against the Heuristic column** ([09.5f](09.5f-results.md) section 7.3,
`uv run uno pimc belief 4000 32 2 "" heuristic`): 4,000 EVAL deals, w = 32, d = 2, paired. Against
Random the column is nearly uniform, so its *play* choices say little and only its draws inform;
the exact toys put the belief axis at 0.0-0.6 points there but up to **2.6 against the Heuristic**,
which is the column that tests the soft rung.

| row | seat 0 | seat 1 | ms/game * |
|---|---|---|---|
| PIMC response w=32 d=2, uniform (`prior`) | 0.6175 | 0.5897 | 16.0 |
| **PIMC response w=32 d=2, `voids`** | **0.6402** | **0.6268** | 19.7 |
| PIMC response w=32 d=2, `window` | **0.6663** | **0.6332** | 47.3 |

\* post-optimisation (context.md 5.21); not comparable with the vs-Random table above.

Paired: `voids - prior` **+2.27 (z +2.7) / +3.70 (z +4.2)**, `window - prior` **+4.88 (z +5.6) /
+4.35 (z +4.9)**, `window - voids` **+2.60 (z +3.1) / +0.65 (z +0.8)**. Two readings:

- **`voids` carries to a second column** at much the size it had against Random, at both seats --
  which is what a rung reading no policy *probabilities* should do. Both columns satisfy the
  behavioural assumption it does make (below). Argued in 9.5f, measured here.
- **The column was indeed why `window` looked worthless**, and the toys' "up to 2.6 against the
  Heuristic" is matched at seat 0 (+2.60). But it is **one seat** (seat 1 is +0.65, z +0.8), so by
  this project's both-seats standard `window` over `voids` is **indicated, not established** -- and
  it cannot be settled on held-out data, since all 4,000 `EVAL_SEEDS` are spent. Meanwhile batching
  repriced it: the leaf batch sped `prior` / `voids` by 4.4-4.5x and `window` by only 2.6x (its
  likelihood is Python-side), so `window` now costs **2.4x** `voids` where it used to cost 1.29x.

**`voids` at w = 32 is the working set.** `window` is the better player against a known rulebot
column and stays available as a ladder instrument, but it is unestablished at seat 1, costs 2.4x,
and reads the opponent's policy -- so like the `response` continuation it makes the row
opponent-specific, which rules it out of a general player whatever the score says.

**The belief closed against the exact toys** ([09.5f](09.5f-results.md) sections 8.1 and 9.2,
`uv run uno pimc closure TOY 20 4000 3 [rulebot|passer]`): total variation to the exact posterior
on the full (non-symmetric) `TOY` world tree, 20 constrained sets per seat, 4,000 draws, seat 0 /
seat 1. The **column is the opponent whose posterior is the target**, and it decides the ordering,
because `voids` encodes an assumption about that opponent -- that a playable drawn card is played
at once ([09.5f](09.5f-results.md) section 8.2). `passer0.5` is `uno.agents.baseline.KeepsDrawn`,
a `PlayFirstRandom` that keeps a playable drawn card half the time.

| belief | vs `rulebot` (assumption holds) | vs `passer0.5` (assumption false) |
|---|---|---|
| uniform split (`prior`) | 0.4325 / 0.4496 | 0.2502 / 0.2495 |
| `voids` | **0.0502 / 0.0407** (88 / 91 % closed) | 0.2044 / 0.2099 (19 / 16 %) |
| `soft`, decay 0.1 | 0.1838 / 0.1528 (58 / 66 %) | **0.0815 / 0.1208** (67 / 52 %) |
| distance to close | 0.2939 / 0.3254 | 0.1275 / 0.1473 |
| sampling noise | 0.0070 / 0.0041 | 0.0082 / 0.0052 |

`voids` is the sharper belief against an honest column and near-worthless against one that keeps a
playable drawn card; `soft` -- the same constraint weighted by `decay ** excess` instead of
enforced -- is the better belief there and never puts zero on the truth. **`voids` stays the
ladder default** (every ladder column is honest, and 8.3 shows it never contradicts them);
**`soft` is the rung for a column that may not be.**

**How much the two rungs actually differ** ([09.5f](09.5f-results.md) section 9.1,
`uv run uno pimc divergence 300 32 2`): at w = 32 over 300 DEV deals `soft(0.1)` chooses a
different action from `voids` at **19.5 %** of decisions against Random and **21.5 %** against the
Heuristic -- against a **control** of 18.4 % / 20.6 %, which is what re-seeding the world sampler
alone produces (a second `voids` under a different name, since `PimcSearch._rng` is keyed on it).
The belief moves about one decision in a hundred, which is why no affordable score run separates
the rungs against an honest column: the 2,400-deal DEV sweep of 8.8 was not under-powered, there
is almost nothing there to separate.

**The decay balance point** ([09.5f](09.5f-results.md) section 9.4, `uv run uno pimc decay TOY 20
4000 3`): `soft`'s decay swept against **both** columns' exact posteriors, mean over seats, as the
share of the uniform split's distance closed. `decay = 0` is `voids` and `decay = 1` is the prior,
and the sweep reproduces both endpoints (a check of the section).

| rung | vs `rulebot` | vs `passer0.5` | worst column |
|---|---|---|---|
| `voids` (= `soft` 0) | **89.7 %** | 17.1 % | 17.1 % |
| `soft` 0.05 | 73.4 % | 41.8 % | 41.8 % |
| **`soft` 0.1** | 61.2 % | 59.2 % | **59.2 %** |
| `soft` 0.2 | 44.9 % | **60.0 %** | 44.9 % |
| `soft` 0.5 | 18.3 % | 27.6 % | 18.3 % |
| `soft` 1 (= prior) | -0.2 % | -0.3 % | -0.3 % |

**0.1 is where the two columns cross**, so the default 8.8 chose as a robustness tie-break is also
the maximin rung over the league. A row that knows its column should switch rungs instead of
compromising (`voids` at 89.7 % against an honest one); 0.1 is the answer to "one rung, opponent
unknown", which is what step 11 needs when an RL agent may or may not have learned to hold a
playable drawn card.

## Corrected coarse-memory evidence ([step 10a.3A](10a3a-results.md))

The step 9.5a quotient policy retained discarded L1 history through its color/rank coordinate
frame. The repaired construction canonicalizes each coarse key and its actions together. The
review's positive-reach collision now returns `{1: 0.5, 3: 0.5}` for both equal raw K1 inputs,
through both the direct coarse agent and its lifted L1 strategy. A full-tree `SMALL` oracle checks
the K0 / K1 / K2 partitions and exact same-policy values against the quotient.

Corrected K1 achieved score, seat 0 / seat 1, followed by its absolute gap to BR L1 in score
points. These are the best of 20 local-response iterations, not certified K1 optima:

| ruleset | vs Random: score (gap) | vs Heuristic: score (gap) |
|---|---:|---:|
| `TOY` | 0.7927 / 0.5490 (0.07 / 0.23) | 0.7378 / 0.4313 (0.00 / 0.48) |
| `TOY_FULL` | 0.8017 / 0.5768 (0.00 / 1.15) | 0.7446 / 0.4613 (0.00 / 1.19) |
| `TOY2` | 0.7289 / 0.5650 (0.16 / 0.43) | 0.6928 / 0.4917 (0.31 / 0.20) |
| `TOY2_SKIP` | 0.7276 / 0.5753 (1.49 / 1.66) | 0.6776 / 0.5193 (1.64 / 1.63) |
| `TOY2_FULL` | 0.7440 / 0.6118 (0.39 / 0.46) | 0.7051 / 0.5287 (0.67 / 0.84) |

K1 moved by at most about 0.6 points; K0 moved much more, up to about 5.3 points. K2 still adds
little over K1. The Heuristic and BR L1 controls are unchanged. This is trustworthy toy evidence,
not evidence that K1 is sufficient on full UNO or that the local solver found the K1 optimum.

## Learner observation contract (step 10a.3B)

Defined in [`docs/observation-contract.md`](observation-contract.md). Specifies the
three comparison arms that Step 11 will implement and compare (K1 feed-forward,
K1+structured evidence, and recurrent public/private history), with explicit
field lists, legality masks, equivariance requirements, and evaluator
compatibility invariants. Tests verify the contract's field extractions.

## The report card and the seed registry ([step 10a](10a-results.md))

`uv run uno report [ruleset] [agent]` prints one agent's card -- score, normalised skill and
`br_gap` per column and seat, exploitability and NashConv, edge by decision -- and **checks every
number on this page to the digit**. `report TOY heuristic` and `report TOY random` both run with
0 failed checks, and the card's `br_gap` reproduces two numbers it is not pinned against (the
Heuristic leaves 5.18 / 13.66 points against Random, the table above says 5.2 / 13.7; Random's gap
to BR L1 is 14.41 / 20.06 against +14.4 / +20.1).

Normalised skill on `TOY`, 0 = Random and 1 = BR L1 at the same seat against the same column:
the Heuristic is **0.64 / 0.32** against Random and **0.71 / 0.27** against itself. It is an affine
rescaling per cell, so those are not score points and nothing is ever selected on them.

**`EVAL_SEEDS` (1,000,000-1,003,999) is spent**: steps 9.5d-9.5f chose the full-UNO working set on
it, so it is block 00 of `uno.eval.seeds`, marked spent, and a new reported number claims a fresh
block from `EVAL_POOL` `[10^6, 2 * 10^6)` instead. Every number on this page predates that and
stands. **Block 01** (1,004,000-1,007,999) is claimed by [step 10a.2](10a2-results.md) for the
column-free PIMC row.

## The training target and the exact policy gradient ([step 10a.2](10a2-results.md))

`uno.eval.target.TRAINING` = **chance + opponent + deal, never the agent's own decisions**. The
exclusion is measured, not argued: on a compiled tree the exact policy gradient and the exact
mean of the sampled estimator are both available (`uno.compiled.gradient`).

| claim | result |
|---|---|
| exact gradient vs a central finite difference of `evaluate` | max relative error 4.8e-11 |
| every subset sparing the row, every value function: bias | max deviation 1e-17 (pre-decision corrections of 4.5e-2 against a gradient norm of 4.7e-2, cancelling in the centred form) |
| the row correction with an exact V | the gradient is **exactly zero** |
| the row correction with an approximate V | leaves 0.86 (`l1_set`, R² 0.55) or 0.085 (noise 0.1, R² 0.74) of it on `TOY` -- **not** ordered by R² |
| sampled, 2,000 games, `TRAINING` | unbiased (z +0.56) at **0.370** of the raw gradient variance |
| sampled, + the row correction | -0.000139 ± 0.000822 against an exact +0.047026 (z -112) |

**Cost on `FULL_UNO`** (200 dev deals, 41.5 decisions/game): distinct value rows per game are
**1** for the deal, 784 for chance, 995 for `TRAINING`, 1,015 for every correction, against the
41.5 the step's own `V(s)` calls make. Timed per component: the step's V 0.57 ms/game against the
shipped corrections' 41.9, of which **22.8 is the engine building sibling states**, 12.5
row-at-a-time features (4.1 vectorised) and 4.5 the forward pass. So the corrections are **55x**
the step's own value calls with the features vectorised, and 73 % of that floor is not the
network. The deal correction alone is one row a game.

## Shaping potentials and the deal floor by level ([step 10a.2](10a2-results.md))

`Phi(s0) - E Phi` variance as a share of the raw outcome variance, heuristic vs random, seat 0 /
seat 1. With no discount and `Phi(terminal) = 0` a game's shaping terms telescope to `-Phi(s0)`,
so this *is* AIVAT's deal correction -- array-equal per game, not merely close.

| ruleset | L1 (own hand) | L2 (both hands) | L3 (+ pile order) |
|---|---|---|---|
| `TOY` | 0.276 / 0.180 | **0.424 / 0.487** | 0.466 / 0.527 |
| `TOY_FULL` | 0.276 / 0.180 | 0.424 / 0.487 | 0.466 / 0.527 |
| `TOY2` | 0.209 / 0.126 | 0.387 / 0.413 | 0.500 / 0.501 |
| `TOY2_SKIP` | 0.229 / 0.219 | 0.461 / 0.423 | 0.626 / 0.565 |

The L2 column reproduces the exact no-deal ratio published in the AIVAT table above (`TOY` 0.42 /
0.49). The ordering L1 <= L2 <= L3 is a theorem (each level is a conditional expectation of the
next). Under **Nash self-play** the L2 floor is 0.67-0.71 and the L3 floor **0.86-0.96**: between
equals almost the whole outcome is the deal. `TOY_FULL` matches `TOY` because neither rulebot
ever draws voluntarily, so the two rulesets play out identically under that profile.

## Exploitability of the belief rungs ([step 10a.2](10a2-results.md))

`TOY`, full world tree, response continuation, exploitability in score points (row at seat 0 /
seat 1). On the full tree PIMC takes a *pure* per-set action rather than an orbit played
uniformly, so these rows are comparable with each other and **not** with the quotient rows above.

| rung | vs random, score | exploitability | vs heuristic, score | exploitability | NashConv (vs heu) |
|---|---|---|---|---|---|
| `prior` | 0.7917 / 0.5440 | 1.36 / 1.35 | 0.7361 / 0.4238 | 1.35 / 1.35 | 2.71 |
| `voids` | 0.7917 / 0.5445 | 1.36 / 1.50 | 0.7361 / 0.4238 | 1.35 / 1.50 | 2.86 |
| `soft(0.1)` | 0.7917 / 0.5445 | 1.36 / 1.35 | 0.7361 / 0.4238 | 1.35 / 1.35 | 2.71 |
| `posterior` | 0.7919 / 0.5491 | 1.61 / 0.83 | 0.7373 / 0.4329 | **3.12 / 7.82** | **10.94** |

The rungs that ship read no policy *probabilities* and are nearly free in exploitability (one
seat moves 1.35 -> 1.50); **full opponent-fitting costs 8.2 points of NashConv** against the
Heuristic column. The rungs change the choice at 1,396-3,276 information sets while moving the
score by at most 0.05 points.

## A column-free full-UNO row ([step 10a.2](10a2-results.md))

The first full-UNO row whose continuation and belief read **nothing** of the column's real
policy: the continuation best-responds to a *fixed surrogate* (the Heuristic) and the belief is
`voids`. Handed its own column as the surrogate it reproduces the `response` row's games
**array-equal**. **EVAL block 01** (1,004,000-1,007,999), 4,000 deals, both seatings, w = 32,
d = 2.

| row | vs random, seat 0 / seat 1 | ms/game | vs heuristic, seat 0 / seat 1 | ms/game |
|---|---|---|---|---|
| **column-free: assume heuristic, `voids`** | **0.7690 / 0.7510** | 27.9 | **0.6425 / 0.6122** | 20.1 |
| column-free: `minimax`, `voids` | 0.7285 / 0.7310 | 49.8 | 0.5787 / 0.5853 | 35.2 |
| *instrument:* `response(column)`, `voids` | 0.7860 / 0.7685 | 29.5 | 0.6452 / 0.6208 | 19.1 |
| the rulebot on the same deals | 0.6212 / 0.5845 | 0.5 | 0.4155 / 0.3787 | 0.5 |

All ± 0.0065-0.0078. Paired on the deal: the column-free row **beats the Heuristic by 14.8 /
16.7 points against Random and 22.7 / 23.4 against the Heuristic** (z > 15 at both seats), so
it is a strong general player and the first full-UNO rung that could be *trained against*.
Reading the column is worth only **+1.70 / +1.75 (z +2.1)** against Random and **+0.27 / +0.85
(z +0.3 / +1.0)** against the Heuristic -- so 9.5e's +9.73 / +4.57 for the `response`
continuation is mostly the value of best-responding to *something competent*, not to *this*
opponent. Assuming a clairvoyant opponent instead costs 3.6-6.7 points (z 3.9-7.3), and
`minimax` is also ~1.75x more expensive, since expectimax expands only the surrogate's support.

**Cross-block check:** the instrument is the published working-set row, selected on the now-spent
block 00 where it scored 0.7813 / 0.7603 against Random. On the fresh block 01 it gives 0.7860 ±
0.0065 / 0.7685 ± 0.0067 -- consistent, so that selection did not materially inflate it.
