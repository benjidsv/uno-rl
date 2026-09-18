# Step 9 results: robustness of the toy results, pairing on full UNO (2026-09-15)

> On top of commit `fca5589` (uncommitted). Reruns: `uv run uno robustness <ruleset>` (`TOY2_FULL`: `150 light`, 71 min before the tabulation cache), `uv run uno pairing` (14 min). Logs: `logs/step9/`. Tests: `uv run pytest` (114 passed, 4 min 22 s). Findings and decisions: [09-review](09-review.md).

## What was built

- `CompiledTree.rescored(exhaustion)`: the same tree with exhaustion scored
  `"hand"` (the rule), `"draw"` or `"stuck"` (the player who must draw from the
  empty pile loses). Every solver runs unchanged on it.
- `two_sided(tree)`: the value when both players see both hands (lazy tree,
  expectiminimax) or everything (world tree, minimax), plus the mass of worlds no
  play changes. Pinned against a brute-force walk of the engine on a 7-card
  ruleset with skips and the voluntary draw, under all three scorings.
- `best_response`: `unreached_sets` (mask) and `ties` (reached sets whose best
  orbit is tied).
- `cfr(fixed=(mask, strategy))`: CFR with some sets fixed, so the least
  exploitable completion of a best response's unreached sets is computable.
- `OnSeeds.deal_var` and `z_global`: the deal-model check (B4).
- `Ruleset.draws = "tickets"` (spec section 8) and `stream` on agents
  (`uno.eval.play.stream_name`, spec section 13), with `Mixture`.
- Speed-ups for the big trees: `transfer` moves a strategy between the lazy and
  world trees by key (no agent asked again, Nash included), and
  `store.load_or_tabulate` caches the baseline agents' tabulations (~50 s per
  agent per tree on `TOY2_FULL`). Used by `cfr`, `toy2`, `toy2_columns`,
  `validate`, `robustness` and `common`.
- Experiments `robustness` and `pairing`. The ticket test caught a bug in the
  first ticket implementation (it could draw the setup's known tail early,
  4.7 sigma off the exact value on `SMALL`); fixed before any number below.

## 1. The split share (A1)

Exact, world tree, first-named agent. "Independent" is the split share two
independent games with the same per-seat outcome distributions would have.

| ruleset | matchup | corr | splits | independent | excess |
|---|---|---|---|---|---|
| `TOY` | heuristic vs random | -0.300 | 0.596 | 0.468 | +0.128 |
| | br_l1(random) vs random | -0.408 | 0.546 | 0.387 | +0.159 |
| | br_l1(heuristic) vs heuristic | -0.608 | 0.778 | 0.498 | +0.281 |
| | nash vs heuristic | -0.616 | 0.784 | 0.501 | +0.283 |
| `TOY_FULL` | heuristic vs random | -0.300 | 0.596 | 0.468 | +0.128 |
| | br_l1(heuristic) vs heuristic | -0.539 | 0.716 | 0.469 | +0.247 |
| `TOY2` | heuristic vs random | -0.309 | 0.632 | 0.489 | +0.143 |
| | br_l1(heuristic) vs heuristic | -0.510 | 0.717 | 0.481 | +0.236 |
| `TOY2_SKIP` | heuristic vs random | -0.391 | 0.663 | 0.478 | +0.185 |
| | br_l1(heuristic) vs heuristic | -0.548 | 0.713 | 0.466 | +0.247 |

The deal adds 12-29 points of splits over independent games. The earlier "3
games in 5 are decided before anyone moves" is corrected in step 7 and
`context.md`.

## 2. Exhaustion scoring (A2, A6)

Points unless a score. Edges are over Random at the same seat under the same
scoring; "two-sided" is the seat-0 score when both see both hands / both see
everything.

| ruleset | scoring | R vs R gap | Nash gap | BR L1 edge vs random | Heuristic edge vs random | Heuristic exploitability | two-sided | worlds no play changes |
|---|---|---|---|---|---|---|---|---|
| `TOY` | hand | 29.9 | 23.9 | +14.4 / +20.1 | +9.2 / +6.4 | 5.6 / 11.8 | 0.6075 / 0.6099 | 0.043 |
| | draw | 27.2 | 22.5 | +15.5 / +21.5 | +11.1 / +7.3 | 5.5 / 12.5 | 0.6040 / 0.6028 | 0.035 |
| | stuck | 20.3 | 11.2 | +18.9 / +25.5 | +11.7 / +6.2 | 3.8 / 17.3 | 0.5611 / 0.5778 | 0.045 |
| `TOY_FULL` | hand | 29.9 | 17.3 | +15.2 / +23.8 | +9.2 / +6.4 | 6.0 / 15.8 | 0.5885 / 0.6337 | 0 |
| | draw | 27.2 | 28.6 | +19.1 / +32.9 | +11.1 / +7.3 | 20.3 / 14.4 | 0.6429 / 0.6429 | 0 |
| | stuck | 20.3 | 0.2 | +36.6 / +51.4 | +11.7 / +6.2 | 18.7 / 30.4 | 0.5151 / 0.5889 | 0 |
| `TOY2` | hand | 17.5 | 15.7 | +14.3 / +15.7 | +8.5 / +3.5 | 7.2 / 11.7 | 0.5903 / 0.5857 | 0.072 |
| | draw | 17.8 | 18.0 | +15.0 / +15.8 | +8.8 / +4.1 | 7.4 / 10.7 | 0.6026 / 0.5987 | 0.072 |
| | stuck | 20.2 | 21.3 | +16.8 / +17.4 | +8.7 / +4.8 | 9.2 / 10.9 | 0.6162 / 0.6114 | 0.072 |
| `TOY2_SKIP` | hand | 16.8 | 15.0 | +15.8 / +17.6 | +5.7 / +6.5 | 11.0 / 11.9 | 0.5972 / 0.5963 | 0.104 |
| | draw | 16.9 | 15.6 | +15.6 / +17.1 | +6.1 / +6.8 | 10.5 / 11.1 | 0.6026 / 0.6029 | 0.101 |
| | stuck | 16.5 | 15.9 | +17.7 / +19.5 | +6.1 / +6.9 | 12.2 / 11.9 | 0.6065 / 0.6075 | 0.103 |
| `TOY2_FULL` | hand | 17.5 | 11.3 | +16.0 / +20.4 | +8.5 / +3.5 | 9.4 / 15.5 | 0.5596 / 0.5866 | 0 |
| | draw | 17.8 | 12.8 | +19.5 / +26.2 | +8.8 / +4.1 | 14.0 / 17.4 | 0.5843 / 0.5745 | 0 |
| | stuck | 20.2 | 20.7 | +35.7 / +47.0 | +8.7 / +4.8 | 27.1 / 21.3 | 0.6129 / 0.6354 | 0 |

Readings:

1. **No scoring of exhaustion is neutral.** Even on the forced-play toys the
   Nash seat gap moves up to 13 points with it (`TOY` 23.9 / 22.5 / 11.2, `TOY2`
   15.7 / 18.0 / 21.3); only `TOY2_SKIP` barely moves (15.0 / 15.6 / 15.9). Seat gaps and exploitability are properties of the toy *under
   its scoring*.
2. **The alternatives distort free play.** Under "draw" the trailing player can
   draw the pile empty and claim: on `TOY_FULL` the Nash value, both two-sided
   values and minimax play all equal 0.6429 (seat 0 forces a win in 29 % of
   worlds, the rest are drawn). On `TOY2_FULL` (3-card pile) the equilibrium
   barely moves (Nash gap 11.3 → 12.8), although perfect-information play stalls
   67 % of worlds into draws. Under "stuck" the endgame becomes drawing the pile
   down so the opponent is stuck, on both free toys: Random, which never draws
   voluntarily, is 41 / 47 points exploitable on `TOY_FULL` and 47 / 36 on
   `TOY2_FULL`.
3. **The voluntary draw's effect on the Nash seat gap depends on the pile.** With
   a 2-card pile its sign changes with the scoring (`TOY_FULL` minus `TOY`: -6.6
   points under the rule, +6.1 under "draw", -11.0 under "stuck"). With a 3-card
   pile it shrinks the gap or leaves it under all three (`TOY2_FULL` minus
   `TOY2`: -4.4, -5.2, -0.6). Step 8.7's reading holds on the three-rank family,
   though every voluntary draw on these piles changes pile parity, so "fetch a
   card" is still not separated from parity play.
4. **More stable on forced play:** the Heuristic leaves 5.2 / 13.7 points against
   Random on `TOY` (4.4 / 14.2 draw, 7.2 / 19.3 stuck) and 5.8 / 12.2 on `TOY2`
   (6.2 / 11.7, 8.1 / 12.6). "The rulebot is decent from seat 0 and weak from
   seat 1" survives every scoring. On `TOY2_SKIP`, where its skip rule is a
   guess, it leaves 10.1 / 11.1 (9.5 / 10.3 draw, 11.6 / 12.6 stuck), weak from
   both seats under every scoring.
5. **Two-sided values (A6):** seeing everything is not always worth something to
   seat 0. Under the rule, the Nash L1 score of seat 0 is above both two-sided
   values on `TOY` (0.6196 against 0.6075 / 0.6099), about equal to the L2 one on
   the free toys (`TOY_FULL` 0.5866 / 0.5885, L3 0.6337; `TOY2_FULL` 0.5564 /
   0.5596, L3 0.5866), and below both on `TOY2` (0.5784
   against 0.5903 / 0.5857) and `TOY2_SKIP` (0.5748 against 0.5972 / 0.5963). *The review's "on `TOY` hidden information favours
   seat 0" holds for `TOY` only.* Worlds no play changes: 4-10 % on the forced toys,
   none on free play.

For validating the sampled methods (step 9.5), the scorings are three different
exact games on the same tree, at no cost: a method must match the exact value
under each. The forced toys are the ones where "draw" and "stuck" are both
reasonable games; on free play only "draw" on `TOY2_FULL` is.

## 3. Equilibrium choice (B1)

Largest spread over CFR+ / DCFR / PCFR+ (1000 iterations, NashConv ≤ 0.0006
points) of the Nash-dependent cells, points, seat 0 / seat 1:

| ruleset | nash vs random | nash vs heuristic | BR L2 vs nash | BR L3 vs nash | BR L1(random) Nash-filled exploitability |
|---|---|---|---|---|---|
| `TOY` | 0.04 / 0.01 | 0.02 / 0.00 | 0.08 / 0.05 | 0.05 / 0.04 | 0.00 / 0.00 |
| `TOY_FULL` | 0.20 / 0.44 | 0.22 / 0.13 | 0.18 / 0.02 | 0.15 / 0.12 | 0.01 / 0.62 |
| `TOY2` | 0.01 / 0.03 | 0.01 / 0.08 | 0.07 / 0.01 | 0.02 / 0.02 | 0.00 / 0.02 |
| `TOY2_SKIP` | 0.04 / 0.05 | 0.07 / 0.07 | 0.02 / 0.01 | 0.02 / 0.00 | 0.01 / 0.03 |
| `TOY2_FULL` (DCFR / PCFR+, 150) | 0.07 / 0.23 | 0.07 / 0.02 | (not in light mode) | 0.25 / 0.01 | 0.04 / 0.10 |

On forced play the equilibrium choice is invisible at the printed digits. On
the free toys Nash-dependent cells carry up to ±0.5 point (`TOY_FULL`) and ±0.25
(`TOY2_FULL`, at NashConv 0.005-0.007, so part of that spread is the solver's
remaining error).

## 4. Filling BR L1(random)'s unreached sets (B2)

Exploitability of the best response to Random as a row agent, points, seat 0 /
seat 1. "Least" is CFR with the reached sets fixed (converged: identical at 250,
500 and 1000 iterations). "Tied" counts reached sets whose best action orbit is
tied, where the choice is a tie-break too.

| ruleset | unreached sets | tied reached sets | as solved | Nash-filled | least exploitable |
|---|---|---|---|---|---|
| `TOY` | 42 % / 47 % | 510 / 594 (14 % / 20 % of reached) | 1.56 / 0.36 | 1.38 / 0.29 | **1.21 / 0.24** |
| `TOY_FULL` | 76 % / 84 % | 9,403 / 4,749 (48 % / 52 %) | 4.55 / 1.26 | 3.06 / 2.44 | **2.83 / 1.14** |
| `TOY2` | 52 % / 60 % | 4,378 / 3,632 (14 % / 14 %) | 1.68 / 2.17 | 1.41 / 1.70 | **1.37 / 1.64** |
| `TOY2_SKIP` | 49 % / 59 % | 13,104 / 11,322 (12 % / 13 %) | 1.34 / 1.74 | 1.05 / 1.09 | **1.01 / 0.99** |
| `TOY2_FULL` | 85 % / 91 % | 176,767 / 77,907 (58 % / 51 %) | 6.25 / 4.89 | 3.40 / 3.30 | **3.11 / 2.59** |

Readings:

1. On `TOY_FULL` the Nash fill-in was *worse* than the solver's arbitrary one at
   seat 1 (2.44 against 1.26). The least exploitable completion is 2.83 / 1.14
   (NashConv 3.97, against 5.50 Nash-filled): the "free-play toys separate
   exploiting Random from being unexploitable" reading of step 8.8 shrinks again,
   and is mostly a seat-0 effect on `TOY_FULL`.
2. **`TOY2_FULL` keeps the separation**: least exploitable 3.11 / 2.59 (NashConv
   5.70), against 1.0-1.6 per seat on the forced toys. It is the toy where playing
   well against Random and being unexploitable differ most (DCFR, identical at
   37, 75 and 150 iterations).
3. On the forced toys the least exploitable fill barely improves on Nash (0.04-0.17
   point): near-Nash holds.
4. Half or more of the free toys' *reached* sets are tied too (48-52 % on
   `TOY_FULL`, 51-58 % on `TOY2_FULL`, 12-20 % on the forced toys), so the least
   exploitable numbers are upper bounds: freeing the ties within their tied
   orbits could lower them further. Not done.

## 5. The deal model (B4)

Mean over 20,000 seeds (2,000,000 onwards) of the exact score on each real deal
against the whole-game exact score:

| ruleset | random vs random: z | heuristic vs random: z |
|---|---|---|
| `TOY` (and `TOY_FULL`, same games) | -0.48 | +0.09 |
| `TOY2` | -0.57 | -0.49 |
| `TOY2_SKIP` (19,904 valid deals) | -0.25 | -0.16 |
| `TOY2_FULL` (same games as `TOY2`) | -0.57 | -0.49 |

With ticket draws on a 10-card ruleset with duplicate cards, a +2 per color and
a wild +4 (`DUPS` in `pairing`, 777k histories, the paths the named toys never
take), 20,000 seeds (`uv run uno pairing`): random vs random z +1.73 (pile) / +1.16
(tickets), heuristic vs random +0.77 / +0.37; the simulations on 4000 of them
|z| ≤ 0.71.

## 6. Pairing on full UNO, pile against tickets (A5)

`uv run uno pairing`, 4000 held-out seeds. Variants of the Heuristic:
`heuristic_t1` (opponent "low" at 1 card), `heuristic_eps05` (5 % Random mixed in)
on the Heuristic's rng stream, `heuristic_eps05_own` on its own stream.

**Tickets are the same game.** On `TOY` and `TOY_FULL` (Random, Heuristic,
UniformLegal and Nash, the last two drawing voluntarily), the exact on-seeds
means over 20,000 seeds match the pile rules' exact value in both draw modes (all
|z| ≤ 1.0), and the simulations match their on-seeds values (|z| ≤ 1.7). On full
UNO the per-seat means agree in 19 of 20 cells (z on the per-seed difference). The
one miss, the log's one failed check, is `heuristic_eps05_own` vs Random, seat 1:
0.5693 (pile) against 0.6035 (tickets), z -3.26. Examined like the step 8.8 miss:
the same policy on the Heuristic's stream (`heuristic_eps05`) gives z -1.39 on the
same seeds, and the same rows on 20,000 fresh seeds (5,000,000 onwards, scratch
script, paired z) give |z| ≤ 0.93 in all four cells, that cell +0.93, the other
direction. With the exact toy checks passing, judged chance; the threshold stays
at 3. A seed's pile and ticket scores correlate +0.09 to +0.11, so the paired SE
is only ~5 % below the independent one.

**Paired comparisons** (Heuristic minus a variant, same column), variance ratio
paired / unpaired, seat 0 / seat 1, pile → tickets, and deals for 3 points at 3
sigma with 80 % power (unpaired: 7,700-8,200):

| column | comparison | ratio, pile | ratio, tickets | deals, pile | deals, tickets |
|---|---|---|---|---|---|
| random | heuristic - heuristic_t1 | 0.229 / 0.227 | 0.202 / 0.202 | 1,778 / 1,796 | 1,565 / 1,590 |
| | heuristic - heuristic_eps05 | 0.197 / 0.194 | 0.168 / 0.195 | 1,534 / 1,546 | 1,298 / 1,539 |
| | heuristic - heuristic_eps05_own | 0.717 / 0.713 | 0.646 / 0.673 | 5,587 / 5,685 | 5,028 / 5,277 |
| | heuristic - random | 0.890 / 0.893 | 0.831 / 0.827 | 7,088 / 7,190 | 6,607 / 6,627 |
| heuristic | heuristic - heuristic_t1 | 0.174 / 0.187 | 0.166 / 0.168 | 1,425 / 1,529 | 1,365 / 1,377 |
| | heuristic - heuristic_eps05 | 0.154 / 0.152 | 0.124 / 0.136 | 1,260 / 1,246 | 1,020 / 1,112 |
| | heuristic - heuristic_eps05_own | 0.541 / 0.573 | 0.485 / 0.510 | 4,438 / 4,691 | 3,976 / 4,177 |
| | heuristic - random | 0.809 / 0.837 | 0.759 / 0.754 | 6,524 / 6,675 | 6,091 / 6,011 |

**Mirrored** variance ratio mirrored / raw, pile → tickets: heuristic vs random
0.901 → 0.843, heuristic_t1 vs heuristic 0.644 → 0.563, heuristic_eps05 vs
heuristic 0.671 → 0.618, random vs heuristic 0.928 → 0.844.

Readings:

1. **"Pairing removes only ~10 % on full UNO" holds for dissimilar policies
   only.** For near-identical variants on one rng stream it removes 77-85 %:
   1,250-1,800 deals for a 3-point difference instead of ~8,000. Checkpoint
   comparisons on full UNO are affordable without AIVAT. The variants are proxies:
   two real checkpoints may differ more.
2. **The shared rng stream is the big lever**: the same 5 % mixture on its own
   stream keeps 54-72 % of the variance (49-67 % with tickets), on the
   Heuristic's stream 15-20 % (12-20 %).
   Checkpoints of one policy should share a stream.
3. **Tickets help consistently but modestly**: 3-20 % less variance in the paired
   comparisons, mirrored Heuristic vs Random from 10 % to 16 % removed, never
   worse in any cell. Worth ~10-20 % fewer games; not a game changer next to
   stream sharing.
