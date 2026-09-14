# TOY2: restoring seat parity (step 8.7, 2026-09-14)

Companion to notes.md. `TOY` is solved (and `TOY_FULL` since step 8.6), so
the toy can grow. What it should grow *for*, from step 8.5: seat 0 is worth
15 points on `TOY` and nothing on full UNO, and the two most frequent
full-UNO decisions (draw with a play in hand, action-card timing) are absent
or rare on it. The step 8.6 candidate census stopped at a 6M-history cap on
every 13-card candidate, so their true sizes were unknown. Two items were
left open there:

1. measure what the quotient + cache buy on the step 10-11 loop (a best
   response per checkpoint), then decide on numba kernels / a faster walker /
   discounted CFR;
2. size the 13-card candidates past the cap and pick `TOY2`.

Order of work: the sizing tool first (it decides everything else), the loop
measurement alongside, then compile and solve the chosen candidate with the
same pipeline as `TOY_FULL` in step 8.6.


## 1. Sizing without walking: the position DP (`uno/sizing.py`, RULES.md section 19)

The census walks every history (~140k/s in pure Python); a 50M-history
candidate is minutes to count and one of 500M looks the same under a cap.
But the subtree below a history depends only on its **position** (hands,
unknown pile multiset + known tail, top, color, player, phase, drawn card);
the public history and draw records feed the information-set keys, not the
transitions. So histories, decision nodes, terminals, depth and the exact
end distribution under any history-free policy are all memoised over
positions, of which there are far fewer.

`size(rs, symmetric, world, position_cap)` returns those counts plus the
exact end mass and game length under UniformLegal (the census's policy,
used as the cross-check) and PlayFirstRandom (the ladder's Random row). Its
`random_p0_score` is the exact random-vs-random seat-0 score, which the
earlier candidate tables sampled on 2000-3000 seeds. What it cannot count is
L1 / L2 information sets (history-dependent); those come from `compile_tree`
once a candidate is chosen.

Cross-check, every number to the history:

| tree                    | histories | census    | positions | DP s | census s |
|-------------------------|-----------|-----------|-----------|------|----------|
| TOY full lazy           | 1,252,040 | 1,252,040 | 174,952   | 2.7  | ~8       |
| TOY quotient lazy       | 33,646    | 33,646    | 12,737    | 0.6  | 0.5      |
| TOY quotient world      | 33,997    | 33,997    | 13,469    | 0.4  |          |
| TOY full world          | 1,384,296 | 1,384,296 | 219,504   | 2.4  |          |
| TOY_FULL quotient lazy  | 774,007   | 774,007   | 49,177    | 1.4  | ~17      |
| TOY_FULL quotient world | 730,598   | 730,598   | 47,888    | 0.8  |          |

Random-vs-random seat 0 comes out 0.6493 on every `TOY` tree, the step 5-6
exact value. Positions per history shrink as the tree grows (2.6x on the
`TOY` quotient, 16x on `TOY_FULL`), which is what makes the 13-card decks
countable. Memory is ~0.5 KB per position (2 GB at the 4M cap used below).

`tests/test_sizing.py` (5 tests, 19 s): counts and UniformLegal end mass
equal the census's on small rulesets, full and quotient; the PlayFirstRandom
mass reproduces the compiled tree's exact random-vs-random value and its
node count; the world tree agrees; the cap stops the walk; `TOY` /
`TOY_FULL` sizes pinned.


## 2. What one checkpoint of the step 10-11 loop costs (`scripts/loop_cost.py`)

Per checkpoint the loop needs: load the cached tree, tabulate the agent over
the L1 keying (the one Python loop left, one policy call per canonical set),
its exploitability (two best responses). CFR+ runs once per ruleset for the
Nash value. Heuristic standing in for the checkpoint (an RL policy is one
batched forward pass instead of a Python call per set, so tabulate is an
upper bound). Two runs (one beside the sizing batches, one alone) agree
within 10 %:

| ruleset  | nodes   | L1 sets | load (cache) | tabulate | per set | BR (one seat) | exploitability | evaluate | CFR+ iteration |
|----------|---------|---------|--------------|----------|---------|---------------|----------------|----------|----------------|
| TOY      | 33,647  | 12,036  | 0.04 s       | 0.16 s   | 13 us   | 3 ms          | 7 ms           | 1 ms     | 8 ms           |
| TOY_FULL | 774,008 | 138,876 | 0.05 s       | 1.86 s   | 13 us   | 62 ms         | 145 ms         | 20 ms    | 182 ms         |

Compile walk (no cache): 28k nodes/s on `TOY` with L1+L2 keys, 75k nodes/s
on `TOY_FULL` with L1 only. The CFR+ passes run at 4.2-4.4M nodes/s per
iteration (two passes each; step 8.6 logged 110 ms per iteration on
`TOY_FULL`, that run was inside the cached-tree script with a warm cache of
its own, this one is the cold figure).

Reading: on `TOY` and `TOY_FULL` the exact side of a checkpoint is
milliseconds; the loop's cost will be the training, not the measurement.
The only slow pieces are one-off: the compile walk (Python engine +
canonicalisation, cached afterwards) and CFR+ to a tight NashConv. So the
step 8.6 question answers itself per candidate size: numba kernels buy
nothing the loop needs; a faster walker and discounted CFR only matter if
`TOY2` is big enough for its compile or its CFR+ run to hurt, and section 3
says how big that is: `TOY2` (312k nodes) is `TOY_FULL`-sized, so the
answer stays "nothing needed"; `TOY2_FULL` (28M nodes) is where a
6-second CFR+ iteration and a 10-minute compile show up (section 5).


## 3. Sizing the candidates (`scripts/sizing.py`, quotient trees, 4M-position cap)

All candidates: one copy per (color, rank), no reverse, `draw_then_may_play`,
no reshuffle, hand 3 unless noted. "+" marks a capped count (lower bound).
`R vs R p0` is the exact PlayFirstRandom self-play seat-0 score (0.6493 on
`TOY`, 0.50 on full UNO); `exh` / `ties` the exhaustion and tie shares under
it; `dec/game` the expected number of decisions.

Every candidate sized (three batches, ~25 min of DP in total; a capped
candidate costs 80-340 s and 2 GB before giving up):

| candidate                     | deck | pile | orbits  | positions | quotient histories | R vs R p0 | exh   | ties  | dec/game |
|-------------------------------|------|------|---------|-----------|--------------------|-----------|-------|-------|----------|
| TOY 4c(1,1)+W forced          | 9    | 2    | 144     | 12,737    | 33,646             | 0.6493    | 0.188 | 0.121 | 6.5      |
| TOY_FULL 4c(1,1)+W free       | 9    | 2    | 144     | 49,177    | 774,007            | 0.6493    | 0.188 | 0.121 | 6.5      |
| 4c(1,1)+W forced, no play after draw | 9 | 2 | 144   | 7,848     | 19,220             | 0.6716    | 0.208 | 0.149 | 6.2      |
| 3c(2,1)+W forced              | 10   | 3    | 736     | 63,217    | 282,509            | 0.6602    | 0.046 | 0.024 | 6.5      |
| **3c(1,1,1)+W forced = TOY2** | 10   | 3    | 584     | 88,758    | 311,819            | 0.5875    | 0.091 | 0.045 | 7.1      |
| **3c(1,1,1)+W free = TOY2_FULL** | 10 | 3   | 584     | 492,523   | 28,057,516         | 0.5875    | 0.091 | 0.045 | 7.1      |
| 3c(1,1,1)+W h2 forced         | 10   | 5    | 318     | 199,761   | 1,115,076          | 0.6110    | 0.025 | 0.014 | 5.3      |
| 3c(1,1,1)+W h4 forced         | 10   | 1    | 190     | 16,295    | 49,625             | 0.5821    | 0.379 | 0.154 | 7.7      |
| 3c(1,1,1)+W h4 free           | 10   | 1    | 190     | 37,389    | 332,642            | 0.5821    | 0.379 | 0.154 | 7.7      |
| **3c(1,1)+S+W forced = TOY2_SKIP** | 10 | 3  | 3,070   | 234,191   | 922,744            | 0.5841    | 0.092 | 0.038 | 6.8      |
| 3c(1,1)+S+W free              | 10   | 3    | 3,070   | 957,986   | 76,830,321         | 0.5841    | 0.092 | 0.038 | 6.8      |
| 3c(1,1)+S+W h2 forced         | 10   | 5    | 3,948   | 523,285   | 3,146,816          | 0.6166    | 0.028 | 0.012 | 5.2      |
| 3c(1,1)+S+W h4 forced         | 10   | 1    | 470     | 36,241    | 115,119            | 0.5826    | 0.356 | 0.113 | 7.1      |
| 3c(1,1)+S+W h4 free           | 10   | 1    | 470     | 70,308    | 775,646            | 0.5826    | 0.356 | 0.113 | 7.1      |
| 3c(1,1)+D2+W forced           | 10   | 3    | 3,070   | 100,831   | 524,879            | 0.5790    | 0.585 | 0.039 | 5.1      |
| 4c(1,1)+2W forced             | 10   | 3    | 360     | 93,303    | 670,282            | 0.5992    | 0.131 | 0.067 | 7.2      |
| 4c(1,1)+2W free               | 10   | 3    | 360     | 477,561   | 73,455,626         | 0.5992    | 0.131 | 0.067 | 7.2      |
| 3c(1,1,1)+2W forced           | 11   | 4    | 1,282   | 523,425   | 5,686,085          | 0.5913    | 0.068 | 0.034 | 7.5      |
| 3c(1,1)+S+2W forced           | 11   | 4    | 9,590   | 1,515,087 | 16,567,553         | 0.5775    | 0.066 | 0.025 | 7.2      |
| 3c(1,1)+S+2W free             | 11   | 4    | 9,590   | 4M (cap)  | 3,311,244,309+     |           |       |       |          |
| 4c(1,1)+3W forced             | 11   | 4    | 640     | 379,034   | 11,688,709         | 0.5952    | 0.093 | 0.048 | 7.6      |
| 5c(1,1)+W forced              | 11   | 4    | 366     | 331,792   | 2,063,824          | 0.6119    | 0.082 | 0.042 | 7.5      |
| 5c(1,1)+W free                | 11   | 4    | 366     | 2,531,442 | 867,784,881        | 0.6119    | 0.082 | 0.042 | 7.5      |
| 5c(1,1)+W h4 forced           | 11   | 2    | 260     | 84,432    | 391,180            | 0.6165    | 0.231 | 0.114 | 8.6      |
| 5c(1,1)+2W forced             | 12   | 5    | 757     | 2,406,092 | 76,355,312         | 0.5874    | 0.077 | 0.034 | 8.1      |
| 4c(1,1,1)+W forced            | 13   | 6    | 2,544   | 4M (cap)  | 71,815,792+        |           |       |       |          |
| 4c(1,1,1)+W free              | 13   | 6    | 2,544   | 4M (cap)  | 2,801,848,781+     |           |       |       |          |
| 4c(1,1,1)+W h4 forced         | 13   | 4    | 4,184   | 4M (cap)  | 57,031,562+        |           |       |       |          |
| 4c(1,1,1)+W h2 forced         | 13   | 8    | 504     | 4M (cap)  | 131,307,274+       |           |       |       |          |
| 4c(2,1)+W forced              | 13   | 6    | 2,402   | 4M (cap)  | 186,376,446+       |           |       |       |          |
| 3c(1,1,1,1)+W forced          | 13   | 6    | 2,544   | 4M (cap)  | 73,549,568+        |           |       |       |          |
| 3c(2,2)+W forced              | 13   | 6    | 1,676   | 1,227,752 | 33,070,540         | 0.6667    | 0.005 | 0.003 | 6.7      |
| 3c(1,1,1)+S+W forced          | 13   | 6    | 40,958  | 4M (cap)  | 49,831,676+        |           |       |       |          |
| 3c(1,1)+S+D2+W forced         | 13   | 6    | 768,448 | 4M (cap)  | 57,717,548+        |           |       |       |          |
| 4c(1,1)+S+W forced            | 13   | 6    | 56,628  | 4M (cap)  | 47,304,539+        |           |       |       |          |
| 4c(1,1)+S+W free              | 13   | 6    | 56,628  | 4M (cap)  | 3,777,431,457+     |           |       |       |          |
| 4c(1,1)+S+W h4 forced         | 13   | 4    | 34,718  | 4M (cap)  | 34,448,264+        |           |       |       |          |
| 4c(1,1)+S+W h4 free           | 13   | 4    | 34,718  | 4M (cap)  | 1,339,699,114+     |           |       |       |          |
| 4c(1,1)+D2+W forced           | 13   | 6    | 56,628  | 4M (cap)  | 60,556,539+        |           |       |       |          |
| 6c(1,1)+W forced              | 13   | 6    | 506     | 4M (cap)  | 84,642,596+        |           |       |       |          |
| 6c(1,1)+W h4 forced           | 13   | 4    | 772     | 2,397,696 | 36,200,261         | 0.6003    | 0.105 | 0.051 | 9.7      |
| 4c(1,1,1)+2W forced           | 14   | 7    | 4,714   | 4M (cap)  | 159,242,774+       |           |       |       |          |
| 4c(1,1)+S+2W forced           | 14   | 7    | 180,950 | 4M (cap)  | 107,649,847+       |           |       |       |          |
| 5c(1,1,1)+W forced            | 16   | 9    | 4,402   | 4M (cap)  | (cap in first root)|           |       |       |          |
| 4c(1,1,1,1)+W forced          | 17   | 10   | 8,266   | 4M (cap)  | (cap in first root)|           |       |       |          |

What the table says:

- **The 13-card third-rank family is out of reach.** With forced play it is
  57-131M quotient histories (hand 4 / hand 2), free play is over 2.8
  billion. The step 8.6 budget was 20-50M histories in memory; even the
  forced variants would need a different representation, not a faster
  walker.
- **Free play multiplies the tree by 20-400x** (23x on 9 cards, 90x on
  3c(1,1,1), 110x on 4c+2W, 420x on 5c) because every voluntary draw
  branches over the pile at every decision. A `TOY2` that keeps the
  voluntary draw exactly solvable has to be small in forced form: only
  3c(1,1,1)+W (28M free) and 4c(1,1)+2W (73M free) are near the budget.
- **Seat parity moves with the pile and the deck's playability, cheaply.**
  A third rank on three colors (10 cards) takes random-vs-random from 0.649
  to 0.588 at 312k histories, 9x `TOY` and smaller than `TOY_FULL`; a skip
  per color on three colors gets 0.584 at 923k. A second wild on four colors
  gets 0.599 at 670k, five colors 0.612 at 2.1M. The largest candidates buy
  nothing more: 5c+2W is 76M histories for 0.587, 6c h4 36M for 0.600.
- **Duplicates and short piles go the wrong way.** Two copies of a rank
  (3c(2,1), 3c(2,2)) make nearly every card playable and the race returns
  (0.660-0.667); so does hand 2 on a longer pile (0.611-0.617). Hand 4 on a
  1-card pile (0.582) is exhaustion-dominated (36-38 % of games). Forbidding
  the play after a draw (`draw_then_may_play=False`) makes `TOY` *worse*
  (0.672). +2 cards at 10 cards end 58 % of games by exhaustion.
- **Ten cards is the size.** Every 13-card candidate is at least 34M
  histories with forced play and over a billion free, except 3c(2,2)+W (33M,
  and a pure race). The only free-play trees under 50M are the 9- and
  10-card decks without skips (774k, 28M) and 4c+2W (73M, just over).
- The exact random-vs-random scores agree with the step 8.6 sampled ones
  within their noise (5c: 0.634 sampled on 2000 seeds vs 0.612 exact is the
  largest miss, 2 sigma; the rest are within a point).


## 4. TOY2 and TOY2_SKIP solved (`scripts/toy2.py`)

Named rulesets (`uno/rules.py`): `TOY2` = 3 colors x ranks (1, 1, 1) + 1
wild, hand 3, forced play (10 cards, pile 3); `TOY2_FULL` = the same without
forced play; `TOY2_SKIP` = 3 colors x ranks (1, 1) + one skip per color + 1
wild, hand 3, forced (10 cards, pile 3). Group orders 36 / 36 / 12. Same
pipeline as step 8.6 section 5, CFR+ 1000 iterations.

### Sizes and costs

| ruleset   | tree            | roots | nodes     | L1 sets | other keying | compile | MB    |
|-----------|-----------------|-------|-----------|---------|--------------|---------|-------|
| TOY2      | lazy L1+L2      | 584   | 311,820   | 129,428 | L2 215,502   | 9.6 s   | 45    |
| TOY2      | world L1+L3     | 2,800 | 373,936   | 129,428 | L3 277,298   | 10.5 s  | 54    |
| TOY2_SKIP | lazy L1+L2      | 3,070 | 922,745   | 439,268 | L2 663,976   | 23.1 s  | 140   |
| TOY2_SKIP | world L1+L3     | 8,360 | 1,065,143 | 439,268 | L3 788,518   | 24.7 s  | 158   |

Best response 0.1 s per seat on `TOY2`, 0.3 s on `TOY2_SKIP`; CFR+ 44 ms /
142 ms per iteration, NashConv 0.000 pts after 1000 (44 s / 142 s). Note
the L1-set share: 41-48 % of the nodes, against 18 % on `TOY_FULL`; three
colors leave less symmetry to merge with.

### Best-response columns (score of the responder)

| ruleset   | opponent  | seat | BR L1  | BR L2  | BR L3  | L2-L1 | L3-L2 | TOY: L1 / L2-L1 / L3-L2 |
|-----------|-----------|------|--------|--------|--------|-------|-------|-------------------------|
| TOY2      | random    | 0    | 0.7305 | 0.7976 | 0.8191 | +6.7  | +2.2  | 0.7934 / +2.5 / +0.9    |
| TOY2      | random    | 1    | 0.5694 | 0.6586 | 0.6804 | +8.9  | +2.2  | 0.5513 / +4.2 / +0.5    |
| TOY2      | heuristic | 0    | 0.6959 | 0.7728 | 0.7907 | +7.7  | +1.8  | 0.7378 / +3.5 / +0.7    |
| TOY2      | heuristic | 1    | 0.4937 | 0.5421 | 0.5587 | +4.8  | +1.7  | 0.4361 / +4.1 / +0.1    |
| TOY2_SKIP | random    | 0    | 0.7425 | 0.7865 | 0.8063 | +4.4  | +2.0  |                         |
| TOY2_SKIP | random    | 1    | 0.5919 | 0.6277 | 0.6503 | +3.6  | +2.3  |                         |
| TOY2_SKIP | heuristic | 0    | 0.6941 | 0.7538 | 0.7738 | +6.0  | +2.0  |                         |
| TOY2_SKIP | heuristic | 1    | 0.5356 | 0.5638 | 0.5816 | +2.8  | +1.8  |                         |

Row matchups (first-named, seat 0 / seat 1): `TOY2` random vs random 0.5875
/ 0.4125, heuristic vs random 0.6730 / 0.4474, heuristic vs heuristic 0.6385
/ 0.3615; `TOY2_SKIP` 0.5841 / 0.4159, 0.6414 / 0.4806, 0.5767 / 0.4233.

### Nash L1 and exploitability

**Nash value: `TOY2` seat 0 scores 0.5784, `TOY2_SKIP` 0.5748** (NashConv
0.000 pts). Exploitability of the rows (points, seat 0 / seat 1):

| row            | TOY2        | TOY2_SKIP   | TOY (step 8.6) |
|----------------|-------------|-------------|----------------|
| random         | 14.8 / 15.2 | 16.7 / 16.8 | 17.1 / 17.4    |
| heuristic      | 7.2 / 11.8  | 11.0 / 11.9 | 5.6 / 11.8     |
| br_l1(random)  | 1.7 / 2.2   | 1.3 / 1.7   | 1.6 / 0.4      |

Nash row and column: `TOY2` nash vs random 0.7184 / 0.5330, nash vs
heuristic 0.6784 / 0.4578, random vs nash 0.4670 / 0.2816, heuristic vs nash
0.5422 / 0.3216, br_l1(random) vs nash 0.5697 / 0.4099; `TOY2_SKIP` nash vs
random 0.7268 / 0.5545, nash vs heuristic 0.6712 / 0.5059, random vs nash
0.4455 / 0.2732, heuristic vs nash 0.4941 / 0.3288, br_l1(random) vs nash
0.5670 / 0.4160.

### Seat gap (seat-0 minus seat-1 score of the same profile, points)

| profile                  | TOY  | TOY_FULL | TOY2 | TOY2_SKIP |
|--------------------------|------|----------|------|-----------|
| random vs random         | 29.9 | 29.9     | 17.5 | 16.8      |
| heuristic vs heuristic   | 38.7 | 38.7     | 27.7 | 15.3      |
| nash vs nash             | 23.9 | 17.3     | 15.7 | 15.0      |
| br_l1(random) vs random  | 24.2 | 21.4     | 16.1 | 15.1      |

Reading:

1. **Seat parity is restored by a third (about 8-9 points of the 24 under
   optimal play)**, on both candidates alike: the Nash gap goes from 23.9 to
   15.7 / 15.0, the random-vs-random gap from 29.9 to 17.5 / 16.8. On `TOY`
   the voluntary draw alone took the Nash gap to 17.3; `TOY2_FULL` (section
   5) says whether the two effects add.
2. **Hidden information is worth more on `TOY2`**: the opponent's hand is
   worth 4.8-8.9 points to a best responder (2.5-4.2 on `TOY`), and the pile
   order 1.7-2.2 (0.1-0.9). Three ranks make the opponent's holding harder
   to infer from the plays, which is the room an inference-capable L1 agent
   needs. The skip variant has less of it (2.8-6.0 / 1.8-2.3).
3. **The Heuristic is the odd one out.** Its self-play gap stays at 27.7 on
   `TOY2` but drops to 15.3 on `TOY2_SKIP`, and it is more exploitable on the
   skip variant (11.0 / 11.9 vs 7.2 / 11.8): its action-card rule (spend
   skips when the opponent is low) is a guess, and the ladder now has a
   place to measure it.
4. **A best response to Random is again near-Nash** (1.3-2.2 points), as on
   `TOY`; `TOY_FULL` was the one where it drifted (4.6 points), so that
   property seems to come from the voluntary draw, not the deck.

Pick: **`TOY2` = 3c(1,1,1)+W h3 forced, `TOY2_FULL` = the same free**, for
the reasons in section 3 (the only candidate whose free version, 28M
histories, fits the 20-50M budget; the skip variant is 77M free). `TOY2_SKIP`
stays a named ruleset: it is solved, adds the action-card decision at 923k
histories, and costs nothing to carry in the ladder as a third column of
toys. `tests/test_toy2.py` pins the `TOY2` sizes, the four BR L1 values, the
row matchups and the Nash value.


## 5. Faster CFR: discounting and prediction (`uno/cfr.py`, `scripts/cfr_variants.py`)

`TOY2_FULL` compiled to 28,057,517 nodes and 3,753,530 L1 sets in 356 s
(1.8 GB cached, ~5 GB peak during the walk). At the measured 4.3M nodes/s
per CFR+ iteration that is 6.6 s per iteration, so 1000 iterations would be
close to two hours. Two accelerations were added to `cfr` on the same
passes, RULES.md section 17: **DCFR** (Brown & Sandholm 2019: positive
regrets discounted by t^1.5 / (t^1.5 + 1), negatives by 1/2, quadratic
averaging) and **PCFR+** (Farina, Kroer & Sandholm 2021: regret matching+ on
accumulated + last instantaneous regret, quadratic averaging), plus their
combination. All variants keep the orbit averaging, so every strategy stays
symmetric; every run is checked exactly (two best responses per logged
iteration), so a stalled variant would be seen, not trusted.

NashConv in score points at iteration t on the cached quotient trees, 300
iterations each; all variants bracket the same Nash value on every tree and
all strategies are symmetric:

| tree     | variant | t=10  | t=20  | t=30  | t=50  | t=100 | t=200 | t=300 | ms/it |
|----------|---------|-------|-------|-------|-------|-------|-------|-------|-------|
| TOY      | vanilla | 4.954 | 2.576 | 1.727 | 1.045 | 0.553 | 0.288 | 0.202 | 3.0   |
| TOY      | cfr+    | 1.322 | 0.382 | 0.186 | 0.071 | 0.017 | 0.007 | 0.003 | 4.9   |
| TOY      | dcfr    | 0.438 | 0.082 | 0.033 | 0.013 | 0.006 | 0.002 | 0.002 | 4.7   |
| TOY      | pcfr+   | 0.345 | 0.088 | 0.027 | 0.006 | 0.001 | 0.000 | 0.000 | 4.6   |
| TOY      | dpcfr+  | 0.276 | 0.052 | 0.017 | 0.005 | 0.001 | 0.000 | 0.000 | 4.8   |
| TOY2     | vanilla | 4.828 | 2.486 | 1.682 | 1.057 | 0.552 | 0.296 | 0.207 | 30.7  |
| TOY2     | cfr+    | 1.397 | 0.407 | 0.191 | 0.073 | 0.020 | 0.005 | 0.002 | 45.9  |
| TOY2     | dcfr    | 0.567 | 0.102 | 0.036 | 0.011 | 0.003 | 0.001 | 0.000 | 45.3  |
| TOY2     | pcfr+   | 0.540 | 0.126 | 0.052 | 0.017 | 0.007 | 0.001 | 0.000 | 43.4  |
| TOY2     | dpcfr+  | 0.577 | 0.123 | 0.047 | 0.017 | 0.005 | 0.001 | 0.000 | 43.9  |
| TOY_FULL | vanilla | 9.173 | 4.622 | 3.085 | 1.867 | 0.939 | 0.473 | 0.317 | 74.2  |
| TOY_FULL | cfr+    | 2.931 | 0.781 | 0.354 | 0.129 | 0.034 | 0.008 | 0.004 | 111.9 |
| TOY_FULL | dcfr    | 0.809 | 0.113 | 0.039 | 0.009 | 0.002 | 0.001 | 0.000 | 114.8 |
| TOY_FULL | pcfr+   | 0.923 | 0.130 | 0.039 | 0.009 | 0.001 | 0.000 | 0.000 | 120.6 |
| TOY_FULL | dpcfr+  | 0.913 | 0.127 | 0.041 | 0.009 | 0.001 | 0.000 | 0.000 | 117.7 |

Reading:

1. **Both accelerations are worth 4-7x in iterations at equal cost per
   iteration.** CFR+ needs 150-200 iterations for 0.01 points; DCFR and
   PCFR+ get there at 30-50 on every tree. The per-iteration cost is
   unchanged (the passes dominate; vanilla is cheaper only because it does
   one pass per iteration instead of two).
2. **DCFR and PCFR+ are tied within noise**; PCFR+ pulls ahead late on `TOY`
   and `TOY_FULL` (0.000 vs 0.001-0.002 at 200-300), DCFR is ahead early on
   `TOY2` (0.011 vs 0.017 at 50). Neither stalls anywhere. The combination
   adds nothing over either.
3. For `TOY2_FULL` that turns two hours into ~17 minutes: DCFR, 150
   iterations, NashConv expected ~0.001 points, better than the 1000-iteration
   CFR+ target of the earlier toys. DCFR is used because it led on `TOY2`
   (the closest relative) and its guarantee does not depend on the game
   being well-behaved.

`tests/test_cfr.py`: on the quotient of `TOY` every variant's NashConv
falls, every average strategy is symmetric, all brackets intersect at 0.6196,
vanilla > CFR+ > DCFR and CFR+ > PCFR+ after 60 iterations, and the explicit
flags reproduce the preset.


## 6. TOY2_FULL solved (`scripts/toy2.py TOY2_FULL 150 light dcfr`)

L1 keying only (the L2 / L3 columns need a second keying on 28M nodes, which
does not fit next to the first in 16 GB; they stay open). Cached tree loads
in 1.7 s. Best response 2.5-2.8 s per seat. DCFR 150 iterations in 671 s
(4.5 s per iteration, exploitability logs excluded), **NashConv 0.005 pts**;
the whole run is 15 minutes against the two hours CFR+ would have needed.

DCFR curve (pts): 75.8 at t=1, 9.5 at 5, 2.19 at 10, 0.38 at 20, 0.156 at
30, 0.045 at 50, 0.022 at 70, 0.011 at 100, 0.005 at 150. Same shape as on
the small trees, so the race in section 5 transferred.

### Values

| opponent  | seat | BR L1  | TOY2 (forced) | worth of the voluntary draw (pts) | TOY_FULL - TOY |
|-----------|------|--------|---------------|-----------------------------------|----------------|
| random    | 0    | 0.7479 | 0.7305        | +1.7                              | +0.8           |
| random    | 1    | 0.6164 | 0.5694        | +4.7                              | +3.7           |
| heuristic | 0    | 0.7118 | 0.6959        | +1.6                              | +0.7           |
| heuristic | 1    | 0.5371 | 0.4937        | +4.3                              | +0.7           |

Row matchups identical to `TOY2` (Random and the Heuristic never draw
voluntarily). **Nash L1 value: seat 0 scores 0.5564, seat 1 0.4436.**

Exploitability of the rows (pts, seat 0 / seat 1): random 17.3 / 19.1,
heuristic 9.4 / 15.5, br_l1(random) 6.3 / 4.9 (was 1.7 / 2.2 on `TOY2`, 4.6
/ 1.3 on `TOY_FULL`). Nash row / column: nash vs random 0.7048 / 0.5478,
nash vs heuristic 0.6671 / 0.4843, random vs nash 0.4522 / 0.2952, heuristic
vs nash 0.5157 / 0.3329, br_l1(random) vs nash 0.5407 / 0.4238.

Voluntary draws with a play in hand, expected per game (seat 0 / seat 1 of
the profile): br_l1(random) vs random 0.129 / 0.329, br_l1(heuristic) vs
heuristic 0.133 / 0.327, nash vs nash 0.248 / 0.495, nash vs random 0.251 /
0.567. (`TOY_FULL`: 0.03 / 0.18 for the best responders, 0.23 / 0.37 at
equilibrium.)

### Seat gap, all four toys (seat-0 minus seat-1 score, points)

| profile                 | TOY  | TOY_FULL | TOY2 | TOY2_SKIP | TOY2_FULL |
|-------------------------|------|----------|------|-----------|-----------|
| random vs random        | 29.9 | 29.9     | 17.5 | 16.8      | 17.5      |
| heuristic vs heuristic  | 38.7 | 38.7     | 27.7 | 15.3      | 27.7      |
| br_l1(random) vs random | 24.2 | 21.4     | 16.1 | 15.1      | 13.2      |
| nash vs nash            | 23.9 | 17.3     | 15.7 | 15.0      | 11.3      |

Reading:

1. **The two effects add.** The third rank takes the Nash gap from 23.9 to
   15.7, the voluntary draw from 15.7 to 11.3; on `TOY` the draw alone was
   worth 6.6 points of gap, here 4.4. Under optimal play seat 0 now scores
   0.556 against 0.620 on `TOY`: the first-mover edge is halved, at 28M
   histories, 15 minutes to solve, and a tree the checkpoint loop handles in
   3 seconds per best response.
2. **The voluntary draw is used far more on `TOY2_FULL`**: a best responder
   draws with a play in hand 0.13 times per game from seat 0 (0.03 on
   `TOY_FULL`) and 0.33 from seat 1 (0.18); at equilibrium 0.25 / 0.50. With
   three ranks a drawn card is more often the one that fixes a color
   problem, so the draw is a real alternative to a bad play, not just a
   tempo tool of the trailing seat.
3. **Playing well against Random is no longer near-Nash.** A best response
   to Random gives up 6.3 / 4.9 points to its own best responder (1.7 / 2.2
   on `TOY2`): the game has enough room for deception that exploiting a weak
   opponent and being unexploitable have separated, which is what the
   exploitability column of the ladder needs to be informative for RL rows.
4. **The Heuristic is weakest where the draw matters**: 15.5 points
   exploitable from seat 1 (11.8 on `TOY2`), and it never draws voluntarily.
   Seat 1's draw is worth 4.3-4.7 points to a best responder; the rulebot
   leaves all of it on the table.

### What is next

- L2 / L3 columns for `TOY2_FULL`: either a second compile with only the L2
  keying (another 1.8 GB file, then two trees loaded for the comparison) or a
  keying computed on the loaded tree without keeping the keys. Open.
- `scripts/ladder.py` / `scripts/mirror.py` still report on `TOY`; the
  sampled estimators (mirrored, and AIVAT in step 9) should be validated on
  `TOY2` next, where the exact numbers above are the reference.
- `TOY2_SKIP` carries the action-card decision at 923k histories; its free
  version (77M) is out of reach, so it is the forced-play ruleset for
  skip-timing questions only.
- Discounted CFR is the default for anything above ~1M nodes; PCFR+ is a
  tie and stays as a preset.
