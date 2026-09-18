# Step 3 results: census (2026-09-14, `uno census`)

> Commit `fbca563` (2026-09-14). Rerun: `uv run uno census [node_cap] [games] [letters]`. Code: `uno/reference/census.py`, `uno/game/chance.py`. The Heuristic of this step broke ties on the lowest type id; current values in [index](../index.md).

Method: the **lazy-chance tree** (`uno/game/chance.py`), not permutations. A draw is a
chance node over the distinct types left in the unknown part of the pile; the
deal is a chance node over (hand multisets, setup prefix). Cross-checked
against brute-force permutation enumeration on tiny decks (`tests/reference/test_census.py`).
Pure Python walks ~145k histories/s.

## Round 1: size

| candidate (colors, ranks x copies, extras, hand)   | deck | histories   | L1 info sets | complete | notes                                   |
|-----------------------------------------------------|------|-------------|--------------|----------|-----------------------------------------|
| A `TOY_CANDIDATE` 2c (2,2,2) +W h2                  | 13   | > 4e7       | > 1.2e6      | no       | original candidate, rejected            |
| B 2c (2,2) +W h2                                    | 9    | > 4e7       | > 6e6        | no       | pile 4                                  |
| C 2c (2,2) +W h3                                    | 9    | 3,653,688   | 858,428      | yes      | pile 2, 24 s per pass                   |
| D 2c (1,1,1) +W h2                                  | 7    | 365,664     | 88,800       | yes      | pile 2, 60 % exhaustion under uniform   |
| R 3c (1,1) +W h2                                    | 7    | 413,232     | 100,836      | yes      | like D with 3 colors                    |
| O = C + forced play                                 | 9    | 56,288      | 26,668       | yes      | 0.3 s per pass                          |
| H = B + forced play                                 | 9    | 59,020      | 29,580       | yes      |                                         |
| N 2c (2,2) +W +S h2 forced                          | 11   | 7,185,024   | 3,438,712    | yes      | one skip per color                      |
| E / F / L / M / Q (skips, +2, no after-draw, ...)   | 9-11 | > 2e7       |              | no       |                                         |
| K, P (hand 3-4 on 7-9 cards)                        | 7-9  | tiny        |              | yes      | degenerate: > 95 % exhaustion           |

What drives size: the pile. Every voluntary draw branches over ~5 types and
uniform-legal play can draw at any decision, so histories grow like
(types x branching)^pile. Forced play removes the "draw with a play in hand"
branch and shrinks C by 65x (3.65e6 -> 5.6e4) without touching anything else;
under play-first random and the heuristic the match numbers on C and O are
identical to the last game (neither agent ever draws voluntarily).

## Round 2: tempo

Round 1 picked O, but seat 0 won 72 % of random-vs-random games on every
candidate. Diagnosis: on O, **59 % of games contain no draw at all and seat 0
wins 100 % of those**; in games with a draw seat 0 wins 33 %. With 2 colors x
2 ranks nearly every card plays on every top, nobody gets stuck, and the game
is a race the first mover wins by construction (one card ahead all game). Not
the exhaustion rule (5 % of games). **Per-card playability** is the lever.

Forced play, hand 3, 3000 random-vs-random games. "P0 win" = seat-0 wins / all games, ties counted as non-wins, which is why S reads 0.59 here and 0.65 as a score with ties worth 1/2 in step 4 (re-run on the current engine 2026-09-14: 1775 wins, 362 ties, 863 losses):

| candidate               | deck | histories | P0 win | no-draw games | decisions with a choice |
|-------------------------|------|-----------|--------|---------------|-------------------------|
| O 2c (2,2)+W            | 9    | 56k       | 0.72   | 0.59          | 0.41                    |
| **S 4c (1,1)+W**        | 9    | 1.25M     | 0.59   | 0.31          | 0.35                    |
| T 3c (1,1,1)+W          | 10   | > 5M      | 0.56   | 0.25          | 0.29                    |
| 3c (1,1,1,1)+W, 4c (1,1,1)+W | 13 | > 5M   | 0.56   | 0.17          | 0.24                    |
| 4c (1,1)+W hand 4       | 9    | 135k      | 0.50   | 0.29          | 0.53 (pile 0: 71 % exhaustion, 11 % invalid deals) |

Decision: **`TOY` = S** (4 colors, ranks (1,1) one copy each, 1 wild, hand 3,
forced play, 9 cards, pile 2). Exact: 1,252,040 histories, 558,656 L1
information sets, 7 s per pass in pure Python. Best response is cheap; CFR
needs an array-compiled tree from the start: vanilla CFR needs far more than
1000 passes for a tight exploitability on 558k sets, so plan CFR+ (or MCCFR)
on a compiled tree and report exploitability against iteration count.
**`TOY_FULL`** = S without forced play: > 2e7 histories (stopped at the cap),
so it is a name for later, not a working size. Its best-response gap to `TOY`
would be the worth of the voluntary draw.

Exact-vs-sampled check: the census gives the exact UniformLegal vs UniformLegal
seat-0 score; `match` on 4000 seeds lands inside the 95 % CI (`TOY`: exact
0.651, sampled 0.660 [0.647, 0.674]; same on C, D, O).

Rules for the ladder that fell out of this:

1. Seat 0 is still worth 0.66 on `TOY` (score, draws = 1/2). Every matchup
   value must be reported per seat or seat-paired (mirrored evaluator, step 7),
   never pooled.
2. Skill checks compare against the random baseline **at the same seat**, not
   against 0.5.
3. The toy was picked after looking at outcomes: round 2 added the tempo
   criterion once seat 0's 72 % showed up, and "comfortably above 50 %" was
   never a pre-registered threshold. Fine for design; it means the steps 5-6
   numbers are the first computed on a ruleset fixed *before* they were run.
4. ~19 % of toy games end by exhaustion, so part of the toy's skill is the
   pile-parity / card-count tactic of [spec section 9](../../spec.md), which full UNO does
   not have. Conclusions about what PPO could or could not learn transfer only
   for the decisions the toy shares with full UNO (color management, holding
   the wild, the hold-after-draw choice).
