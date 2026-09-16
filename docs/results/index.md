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
| `TOY2` | 14.8 / 15.2 | 7.2 / 11.8 | 1.7 / 2.2 (1.4 / 1.7) [1.4 / 1.6] |
| `TOY2_SKIP` | 16.7 / 16.8 | 11.0 / 11.9 | 1.3 / 1.7 (1.1 / 1.1) [1.0 / 1.0] |
| `TOY2_FULL` | 17.3 / 19.1 | 9.4 / 15.5 | 6.3 / 4.9 (3.4 / 3.3) [3.1 / 2.6] |

`uv run uno symmetry`, `uno toy2` and `uno validate` print both rows ([step 8.8](08.8-review.md)).

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

**D2 with a search Q** (`d2`, 150 EVAL deals): where the Heuristic's choices differ most from a
w = 32 search -- **action-card timing 68-73 % and wild handling 25-29 %**, i.e. ~97 % between them,
nearly seat-symmetric. The control row (the search itself) sits at -3.3 / -2.5 against the
Heuristic's +21, which is what licenses reading the split. Not a value decomposition: the shares are
the finding, not the totals.
