# Step 7 results: mirrored evaluator (2026-09-14, `uno/eval/estimate.py`, `uno mirror`)

> Commit `0ab8e8d` (2026-09-14). Rerun: `uv run uno mirror TOY`. The sampled tables were drawn before the step 8.6 fix of the action rng (agent name in the key) and with the step 4 Heuristic; the exact numbers of the Random column do not move. *Step 8.8: on-seeds and moment checks, see [step 8.8](08.8-review.md).*

Design: play once, estimate many ways. `play_pairs` plays every seed from
both seats (same deal, same pile order) and keeps the traces; raw, mirrored
and paired-comparison estimates are pure functions of the records, and AIVAT
(step 9) will be a fourth one over the same games. Everything below is exact
where it can be: `exact_mirrored` walks the 10,080 worlds of `TOY` (deal x
pile order) and gets the per-seat means, variances, the covariance between
the two seats' scores on the same deal and the share of split pairs; the
sampled numbers must match them. [spec section 15](../spec.md).

What was built:

- `uno/eval/estimate.py`: `play_pairs`, `estimate_mirrored`, `compare` (paired
  difference of two row agents against the same column, same deals),
  `exact_mirrored`, `games_needed`.
- `uno/reference/solve.py`: `outcome_distribution`, the exact win / draw / loss mass
  under two fixed agents; `evaluate` is now its mean. This is what makes the
  exact variance available for stochastic agents, not just deterministic ones.
- `uno/agents/baseline.py`: `BySeat`, so a per-seat best-response table can play both
  seats of a pair (the BR rows of the ladder).
- `uno/eval/play.py`: `DEV_SEEDS` / `EVAL_SEEDS` (below). `uno ladder`
  reports on the held-out range from now on.
- `uno mirror` prints the tables below; `tests/eval/test_estimate.py` (8
  tests) holds the checks listed in [spec section 15](../spec.md). Whole suite: 66
  tests, ~95 s.
- In passing: `test_chance_outcomes_on_real_play` (tree tests) tripped
  hypothesis' filter-too-much health check on one run (the `deck_size <= 16`
  assume discards most generated rulesets); the check is now suppressed for
  that test. Flaky, not wrong.

Cost on `TOY`: 4000 pairs of two L1 agents play in ~1 s; `exact_mirrored`
takes ~4 s for a stochastic matchup (two `outcome_distribution` walks per
world), well under a second for deterministic ones.

**Seed hygiene**: `DEV_SEEDS` = 0-3999 (build, test, later tune) and
`EVAL_SEEDS` = 1,000,000-1,003,999, held out, used only for reported numbers;
`uno ladder` now reports on it too. The old range's 1.5-point seat-0
tilt was harmless while every number was compared on the same seeds, and
becomes a bias the moment an agent, a value function or a hyperparameter is
picked because it scored well there. The held-out range has its own tilt
(seat 1 sits 0.8 points *under* exact for Heuristic vs Random), which is the
point: "on seeds" is the test, the global exact value is the reference.

## Matchup values, 4000 held-out pairs (score of the first-named agent)

`sim` is the raw per-seat estimate; `mean` is the seat average under the raw
CI (2n games as if independent) and under the mirrored CI (n pair scores).
Every row passes the on-seeds check; the deterministic rows match to the digit.

| matchup                       | seat | exact  | on seeds | sim ± 95 %      | mean exact | mean raw ±  | mean mirrored ± |
|-------------------------------|------|--------|----------|-----------------|------------|-------------|-----------------|
| heuristic vs random           | 0    | 0.7390 | 0.7408   | 0.7346 ± 0.0130 | 0.5734     | ± 0.0098    | ± 0.0081        |
|                               | 1    | 0.4077 | 0.3995   | 0.4004 ± 0.0148 |            |             |                 |
| BR L1(random) vs random       | 0    | 0.7934 | 0.7953   | 0.7887 ± 0.0116 | 0.6723     | ± 0.0094    | ± 0.0073        |
|                               | 1    | 0.5513 | 0.5508   | 0.5467 ± 0.0148 |            |             |                 |
| BR L1(heuristic) vs heuristic | 0    | 0.7447 | 0.7508   | 0.7508 ± 0.0132 | 0.5986     | ± 0.0101    | ± 0.0064        |
|                               | 1    | 0.4525 | 0.4529   | 0.4529 ± 0.0152 |            |             |                 |
| BR L1(random) vs heuristic    | 0    | 0.7366 | 0.7416   | 0.7416 ± 0.0134 | 0.5877     | ± 0.0101    | ± 0.0061        |
|                               | 1    | 0.4389 | 0.4445   | 0.4445 ± 0.0151 |            |             |                 |

## What mirroring buys (exact / sampled)

| matchup                       | corr(seat 0, seat 1) | splits        | var ratio mirrored / raw |
|-------------------------------|----------------------|---------------|--------------------------|
| heuristic vs random           | -0.298 / -0.320      | 0.610 / 0.623 | 0.704 / 0.682            |
| BR L1(random) vs random       | -0.407 / -0.413      | 0.546 / 0.554 | 0.605 / 0.598            |
| BR L1(heuristic) vs heuristic | -0.611 / -0.606      | 0.772 / 0.771 | 0.394 / 0.400            |
| BR L1(random) vs heuristic    | -0.647 / -0.639      | 0.795 / 0.790 | 0.357 / 0.365            |

Games for a 3-point edge at 3 sigma (exact variance):

| matchup                       | one seat, raw | seat mean, raw | seat mean, mirrored |
|-------------------------------|---------------|----------------|---------------------|
| heuristic vs random           | 1713 / 2273   | 1993           | 1404                |
| BR L1(random) vs random       | 1370 / 2260   | 1815           | 1098                |
| BR L1(heuristic) vs heuristic | 1828 / 2373   | 2100           | 828                 |
| BR L1(random) vs heuristic    | 1875 / 2350   | 2112           | 753                 |

## Paired comparisons (first minus second row agent, same column, same deals)

| comparison                                     | seat | paired diff ± 95 % | unpaired ± | corr   | var ratio | games, unpaired / paired |
|------------------------------------------------|------|--------------------|------------|--------|-----------|--------------------------|
| heuristic - random, vs random                  | 0    | +0.0866 ± 0.0128   | ± 0.0189   | +0.543 | 0.458     | 7444 / 3410              |
|                                                | 1    | +0.0484 ± 0.0136   | ± 0.0202   | +0.548 | 0.453     | 8504 / 3855              |
| BR L1(random) - heuristic, vs random           | 0    | +0.0541 ± 0.0077   | ± 0.0174   | +0.807 | 0.198     | 6312 / 1250              |
|                                                | 1    | +0.1464 ± 0.0117   | ± 0.0209   | +0.688 | 0.312     | 9094 / 2841              |
| BR L1(heuristic) - heuristic, vs heuristic     | 0    | +0.0586 ± 0.0088   | ± 0.0193   | +0.797 | 0.205     | 7780 / 1595              |
|                                                | 1    | +0.1450 ± 0.0111   | ± 0.0207   | +0.716 | 0.286     | 8932 / 2555              |
| BR L1(heuristic) - BR L1(random), vs heuristic | 0    | +0.0091 ± 0.0052   | ± 0.0188   | +0.922 | 0.078     | 7360 / 572               |
|                                                | 1    | +0.0084 ± 0.0042   | ± 0.0214   | +0.961 | 0.039     | 9525 / 370               |

Reading:

1. **The intro's arithmetic holds and is now exact**: a 3-point edge at one
   seat, raw, needs 1700-2400 games at 3 sigma (2273 from seat 1 for the
   Heuristic, the intro said ~2400). That is the budget iterations 1-4 never
   set.
2. **Mirroring is worth 30-65 %** of the variance on the seat mean, and more
   the more the deal decides the game: 61 % of Heuristic-vs-Random pairs
   split, 77-80 % once a best responder is involved. Splits are a luck-share
   readout: on `TOY`, three games in five between two decent players are
   decided before anyone moves.
3. **Pairing two agents on the same deals is the big win** (common random
   numbers): the Heuristic's +8.7 / +4.8 over Random is measured with 45 % of
   the unpaired variance, the best response's edge over the Heuristic with
   20-30 %, and two near-identical policies (the two BR tables against the
   Heuristic differ by 0.9 points) with 4-8 %: 370-570 games instead of
   7000-9500 to see 3 points at 3 sigma. For steps 10-11 the question "is
   this checkpoint better than the rulebot" is a paired comparison, never two
   raw winrates.
4. **Self-play pairs are degenerate** under common random numbers (the two
   games of an A-vs-A pair are the same game, every pair is 1/2); a mirrored
   A-vs-A cell reports nothing, and comparing two agents through self-play
   cells needs a third, fixed column.
5. **Per-seat numbers stay ordinary**: pairing helps the seat mean and the
   paired differences only. The seat-1 estimate is the loosest (sd 0.48 vs
   0.41-0.43 at seat 0) because seat 1's score sits nearer 1/2.
6. Side result: **BR L1 to Random transfers**: against the Heuristic it scores
   0.7366 / 0.4389 versus 0.7447 / 0.4525 for the best response computed
   against the Heuristic itself, 0.8-1.4 points short, so a best response to
   Random is a decent general L1 policy on `TOY`.
7. Caveat for AIVAT: the exact variances assume the two games of a pair are
   independent given the world, which holds because the rng streams differ by
   seat, and the sampled numbers confirm it within 2 %. For `compare` the
   column agent's streams are *shared* on purpose, so its exact paired
   variance is not a world-by-world sum; the paired numbers above are sampled
   only (exact for deterministic columns, checked in the tests).
