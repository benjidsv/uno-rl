# Step 8 results: compiled tree and Nash L1 by CFR+ (2026-09-14, `uno/compiled/`, `uno/compiled/cfr.py`, `uno cfr`)

> Commit `eafb7e7` (2026-09-14). Rerun: `uv run uno cfr TOY 1000` (on the quotient tree since step 8.6; the full-tree run of this step took 203 s). The Heuristic rows are the step 4 Heuristic; current values in [index](../index.md).

## What was built

The step was going to be "CFR", and the blocker was speed: the pure-Python
walkers cost 7-16 s per pass over `TOY` and CFR needs hundreds of passes. So
the tree is now walked **once** into flat numpy arrays (`compile_tree`, [spec section 16](../../spec.md)) and every exact computation became a vectorized pass over them,
CFR included: evaluate, outcome distributions, best response at L1 / L2 / L3,
exploitability, exact mirrored moments. The old walkers in `uno/reference/solve.py` and
`uno/eval/estimate.py` stay as the independent reference; `tests/compiled/test_compiled.py`
pins the compiled versions to them on small rulesets and `uno cfr`
reproduces the twelve step 5-6 values on `TOY`.

Two things had to exist for that:

- **A level that is constant across an information set** so the backward pass
  can decide a whole set at once: `2 * len(public) + [phase == TURN_START]`.
  Depth by action count is *not* constant across L1 sets (the opponent's
  hidden pass is one action that leaves no public event). The build asserts
  the property, at every level.
- **Observations from keys** (`observation_from_key`): tabulating an agent over
  559k sets without keeping States means rebuilding what it would see from
  the information-set key alone. Property-tested against `observe` at every
  decision of random games, all three levels (`tests/game/test_observe_keys.py`).

Sizes and speed on `TOY` (M1-class laptop, one core):

| tree                    | nodes     | decisions | arcs      | edges     | keyings                          | compile |
|-------------------------|-----------|-----------|-----------|-----------|----------------------------------|---------|
| lazy (deal roots, 5600) | 1,252,041 | 876,848   | 1,208,489 | 1,252,040 | L1 558,656 sets, L2 876,848 sets | 10 s    |
| world (10,080 roots)    | 1,384,297 | 986,496   | 1,374,217 | 1,384,296 | L1 558,656 sets, L3 986,496 sets | 10 s    |

The world tree is only 10 % bigger than the lazy one (the pile holds 2 cards),
so L3 is as cheap as L1 now. Tabulating an agent (Random, Heuristic) over a
keying costs 12 s per tree, a Python loop over the sets; that is the slowest
piece left and it is done once per agent. Memory: ~1.2 GB per tree with both
keyings, mostly the key dictionaries.

| computation                   | compiled | reference | speedup |
|-------------------------------|----------|-----------|---------|
| BR L1 seat 1 vs random        | 0.10 s   | 13.2 s    | 130x    |
| BR L2 seat 0 vs heuristic     | 0.10 s   | 6.7 s     | 67x     |
| BR L3 seat 0 vs random        | 0.12 s   | 15.8 s    | 130x    |
| evaluate heuristic vs random  | 0.04 s   | 1.8 s     | 45x     |
| one CFR+ iteration (2 passes) | 0.20 s   | -         |         |

A best response to *any* tabulated L1 policy now costs 0.1 s per seat, which
is what the RL rows of steps 10-11 need per checkpoint.

**A bug the small tests did not catch.** The first version of the compiled
best response passed every hypothesis check on tiny rulesets and matched all
L2 / L3 values on `TOY`, yet was 0.2-0.6 points *below* the reference at L1
(0.5455 vs 0.5513 for seat 1 vs Random). The two seat arguments of the
forward pass were swapped, so the counterfactual reach held chance only and
dropped the opponent's probabilities. L2 and L3 sets are (nearly) singletons,
where reach weights cannot matter, and on tiny decks the weights happened not
to change any argmax. What caught it was replaying the reference's table on
the compiled tree (0.5513, so the reference was right) and the twelve known
`TOY` values. Those eight L1 / L2 values are now a regression test
(`test_toy_reproduces_steps_5_6`); the hypothesis tests are capped at 5,000
histories per example so the file stays under a minute.

## CFR+ on `TOY`

`cfr(tree)`: regret matching+, alternating updates, linear averaging.
NashConv in score points = score(BR seat 0) + score(BR seat 1) - 1; the
bracket is where the Nash seat-0 score must lie. Vanilla CFR (plain regret
matching, simultaneous updates, uniform averaging) for comparison.

| iteration | CFR+ NashConv (pts) | bracket of the Nash seat-0 score | seconds | vanilla NashConv (pts) |
|-----------|---------------------|----------------------------------|---------|------------------------|
| 1         | 38.30               | [0.4290, 0.8121]                 | 0.2     | 38.30                  |
| 5         | 4.61                | [0.5985, 0.6445]                 | 1.2     | 9.82                   |
| 10        | 1.41                | [0.6128, 0.6270]                 | 2.1     | 5.13                   |
| 20        | 0.42                | [0.6175, 0.6217]                 | 3.8     | 2.70                   |
| 50        | 0.079               | [0.6192, 0.6200]                 | 9.2     | 1.14                   |
| 100       | 0.023               | [0.6195, 0.6197]                 | 18.5    | 0.61                   |
| 200       | 0.008               | [0.6196, 0.6196]                 | 36.8    | 0.33                   |
| 500       | 0.002               |                                  | 94.5    |                        |
| 1000      | 0.001               |                                  | 203     |                        |

**Nash L1 value of `TOY`: seat 0 scores 0.6196, seat 1 0.3804** (NashConv
0.001 points after 1000 CFR+ iterations, 203 s; the strategy is cached in
`.cache/cfr_TOY.npz`; on the quotient tree since step 8.6, `cfr_TOY_L1_sym.npz`). CFR+ is 40x ahead of vanilla at equal iterations. The vanilla curve falls roughly as 1 / T here (5.13 at t = 10, 0.61 at t = 100), faster than its O(1 / sqrt(T)) bound. *(Corrected in step 8.8: this read "the textbook 1 / sqrt(T)".)* Every cell below that uses the
Nash agent means "CFR+ average strategy, 1000 iterations, NashConv 0.001 pts".

## Exploitability of the row agents (now absolute)

Per seat: Nash value at that seat minus what the row keeps against its L1
best responder. "BR to it" is the responder's score, "Nash" the responder's
Nash score at the responder's seat.

| row            | seat | keeps vs BR L1 | BR to it | Nash (responder) | exploitability (pts) |
|----------------|------|----------------|----------|------------------|----------------------|
| random         | 0    | 0.4487         | 0.5513   | 0.3804           | 17.1                 |
| random         | 1    | 0.2066         | 0.7934   | 0.6196           | 17.4                 |
| heuristic      | 0    | 0.5475         | 0.4525   | 0.3804           | 7.2                  |
| heuristic      | 1    | 0.2553         | 0.7447   | 0.6196           | 12.5                 |
| BR L1 (random) | 0    | 0.6007         | 0.3993   | 0.3804           | 1.9                  |
| BR L1 (random) | 1    | 0.3745         | 0.6255   | 0.6196           | 0.6                  |
| Nash L1        | 0    | 0.6196         | 0.3804   | 0.3804           | 0.0                  |
| Nash L1        | 1    | 0.3804         | 0.6196   | 0.6196           | 0.0                  |

## Nash row and column (exact, score of the first-named, seat 0 / seat 1)

| matchup                | seat 0 | seat 1 | compare                                   |
|------------------------|--------|--------|-------------------------------------------|
| nash vs random         | 0.7871 | 0.5235 | BR L1 to random: 0.7934 / 0.5513          |
| nash vs heuristic      | 0.7357 | 0.4314 | BR L1 to heuristic: 0.7447 / 0.4525       |
| nash vs nash           | 0.6196 | 0.3804 | the Nash value                            |
| random vs nash         | 0.4765 | 0.2129 |                                           |
| heuristic vs nash      | 0.5686 | 0.2643 |                                           |
| BR L1 (random) vs nash | 0.6123 | 0.3797 | within 0.7 / 0.1 points of the Nash value |

Nash row on the 4000 held-out pairs (`sim` vs the exact value on those seeds,
same layout as step 7):

| matchup           | seat | exact  | on seeds | sim ± 95 %      | mean exact | raw ±    | mirrored ± | corr (exact / sim) | splits        | var ratio     |
|-------------------|------|--------|----------|-----------------|------------|----------|------------|--------------------|---------------|---------------|
| nash vs random    | 0    | 0.7871 | 0.7902   | 0.7815 ± 0.0117 | 0.6553     | ± 0.0095 | ± 0.0069   | -0.429 / -0.482    | 0.572 / 0.606 | 0.585 / 0.532 |
|                   | 1    | 0.5235 | 0.5199   | 0.5130 ± 0.0149 |            |          |            |                    |               |               |
| nash vs heuristic | 0    | 0.7357 | 0.7436   | 0.7411 ± 0.0134 | 0.5835     | ± 0.0101 | ± 0.0066   | -0.584 / -0.572    | 0.769 / 0.766 | 0.420 / 0.431 |
|                   | 1    | 0.4314 | 0.4338   | 0.4300 ± 0.0150 |            |          |            |                    |               |               |

Both rows pass the on-seeds check; the residual is the Nash agent's own
action noise (it is a mixed strategy), 0.3-0.9 points, inside the CI.

## Best responses to the Nash agent by information level

| seat | BR L1  | BR L2  | BR L3  | L2 - L1 | L3 - L2 | against Random (step 6) |
|------|--------|--------|--------|---------|---------|-------------------------|
| 0    | 0.6196 | 0.6630 | 0.6764 | +4.3    | +1.3    | +2.5 / +0.9             |
| 1    | 0.3804 | 0.4240 | 0.4311 | +4.4    | +0.7    | +4.2 / +0.5             |

## Reading

1. **The seat gap under optimal play is 23.9 points** (0.6196 vs 0.3804, an edge of 12 over 1/2), against 29.9 under random play (0.649 vs 0.351) and 37.8 in Heuristic self-play (0.689 vs 0.311). Optimal play narrows the gap but does not remove it; the ladder stays per seat. *(Corrected in step 8.8: this compared an edge, 12, with two gaps, 30 and 38.)*
2. **The exploitability column of the ladder is now absolute.** Random gives
   up 17 points per seat, the Heuristic 7 from seat 0 and 12.5 from seat 1
   (its seat-1 play is the weak side, consistent with step 6), a best response
   to Random only 1.9 / 0.6. That last number is the surprise: **on `TOY`
   the best response to Random is a near-Nash strategy**, and against Nash it
   scores 0.6123 / 0.3797, within 0.7 / 0.1 points of the Nash value. Playing
   well against a random opponent and playing unexploitably are almost the
   same thing on this toy; the two will separate on games with more room for deception. *Step 8.8 caveat: the best response to Random is undetermined at the information sets Random never reaches (42-47 % of the L1 sets of `TOY`), and its exploitability depends on what is played there ([step 8.8](08.8-review.md)).*
3. **Nash gives up little against weak opponents**: 0.6 / 2.8 points against
   Random and 0.9 / 2.1 against the Heuristic compared with the dedicated best
   responses. The cost of safety is small here.
4. **Information is worth more against a strong opponent**: seeing Nash's hand
   is worth 4.3 / 4.4 points, against 2.5 / 4.2 for Random's; the pile order
   adds 1.3 / 0.7. An L2 / L3 oracle column against the RL rows will read the
   same way.
5. For steps 10-11 the quality measure of a checkpoint is now: tabulate it
   over the L1 keying (`tabulate`), `exploitability` (0.2 s), and the per-seat
   numbers in the table above are its ceiling and its exploitability, with no
   luck anywhere. The Nash L1 set values also come out of the same pass and
   are the potential for reward shaping.
