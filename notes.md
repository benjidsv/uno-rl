Project: UNO RL V5

This is my fifth iteration on experimenting with UNO in RL. The main problem: skill vs luck. UNO variance is enormous and the reward is sparse: during training it's hard to find good plays and take advantage of them. Most of the outcome is luck of the draw.

Previous wins (UNO 1v1, raw winrate vs random):
- rulebots: 60%
- genetic parameterized rulebot: 70%

Failures:
- no meaningful improvement from PPO agents, even with self-play etc.

Suspected causes (to be tested, not assumed):
- the evaluator was too noisy to see small improvements (going from 60% to 63% needs thousands of games for one sigma)
- rlcard's rules are inconsistent with its legal-move function, so the game itself may have had a low or buggy skill ceiling
- we never knew the ceiling, so "no improvement" could mean "already optimal"

Consequence: this iteration is about **measurement first, learning second**. No RL until the ladder below is filled with exact numbers on the toy game.


## Engine (own implementation)

Why not rlcard: rules are unreliable, and we need clone/replay for search, best response and AIVAT.
Why not pyspiel as the core: the oracle levels (see below) are mostly *not* pyspiel-shaped objects. pyspiel is useful for one thing only (CFR / best response / exploitability on the L1 game), so it will be a thin adapter over our core, added later if we want it.

Design rules:
- **One parametrized ruleset**, toy and full UNO are just different parameters: colors, ranks, copies per card, number of wilds, action cards (skip / reverse / +2 / +4), hand size, draw-then-may-play or not, forced play or not, end condition, reshuffle rule.
- **Deterministic given the deck permutation.** The whole game is a pure function of (permutation, action sequence). All chance is in the initial permutation. Reshuffle is deterministic too (or forbidden in the toy: draw pile empty = game over, fewest cards wins). This makes cloning, replay, mirrored evaluation and L3 search trivial.
- **Immutable state** (frozen dataclass / tuples). `clone` = copy. `step(state, action) -> state`.
- `observe(state, player, level)` returns the L1 / L2 / L3 observation. The information level is a property of the observer, not of the game.
- **Rules spec written before code** (RULES.md), with property tests: card conservation, every legal action applies without error, every applied action was legal, determinism, clone equivalence, game always terminates.

Which decisions carry skill in UNO (the toy must keep them, otherwise there is nothing to measure):
- which color to play down (color management: keep the color you hold most)
- holding a wild vs spending it
- wild color choice based on inferred opponent voids (they drew on red => probably no red)
- draw vs play when both are legal
- timing of action cards


## Toy UNO ("Kuhn UNO") — must be defined and sized

Not yet defined. The size decides whether exact solving is feasible: exact L1 solving enumerates deck permutations consistent with what the player saw, and permutations explode fast.

Candidate: 2 colors x 3 ranks x 2 copies + 1 wild, hand size 2, no reshuffle. (to be checked)
**Resolved by the step 3 census below: the candidate is > 4e7 histories; the toy is
4 colors x 2 ranks x 1 copy + 1 wild, hand 3, forced play (`TOY`, 1.25M histories).**

Sizing procedure:
1. implement the parametrized engine
2. for candidate parameters, run a census: number of distinct deals (use duplicate cards, they shrink the count a lot), number of histories, number of L1 information sets
3. target: full tree <= ~10^6 histories so best response and CFR run in seconds
4. **skill check**: best-response L1 vs random must be comfortably above 50%. If the ceiling is near a coin flip the toy is useless as a testbed (and that result alone would explain iterations 1-4).


## The oracle levels — what "optimal" means at each level

Three information levels:
- **L1 player**: own hand + public info (what the trained agent sees)
- **L2 medium**: L1 + opponent's hand
- **L3 oracle**: L2 + deck order (perfect information, unrealistic)

Key clarification (this was the source of my confusion): the three levels are **not the same kind of object**.

- **L1 is the only level where "optimal" can mean a Nash equilibrium**, because both players have hidden info. Nash L1 is computed with CFR (Counterfactual Regret Minimization: iterate over the game tree, accumulate regret per information set, average strategy converges to Nash in 2p zero-sum games). Only L1 is symmetric, so only L1 has an "OPT vs OPT" cell.
- **L2 and L3 "optimal" only exist as a best response to a fixed, known opponent policy.** Once the opponent's policy is known and their hand is visible there is nothing hidden:
  - L3 = plain minimax/expectimax over a deterministic game (the opponent's randomness is just weights in the search).
  - L2 = expectimax where the only unknown is the draw order.
- The "vs Random" and "vs Heuristic" cells at L1 are also best responses, not Nash. "Best response to random" and "Nash" give different numbers. So the columns of the ladder are "best response to the row agent at level Lx", plus a separate Nash L1 row.

Vocabulary: what I called "exact POMDP" is called **best response** in extensive-form games. Algorithm: one pass over the tree, each information set picks the action maximizing the sum over its histories of (chance reach prob) x (opponent reach prob) x (continuation value). Polynomial in tree size, exact. pyspiel has it (`best_response`, `exploitability`); it is also ~100 lines to write against our core.

Gap decomposition (free diagnostics):
- L3 - L2 = value of knowing the draw order (pure luck component)
- L2 - L1 = value of the hidden information (opponent's hand)
- L1 best response = the actual ceiling for the trained agent in that matchup

Full UNO: same definitions, but sampled instead of exact. L3 via search with a depth cutoff, L1/L2 via PIMC (Perfect Information Monte Carlo: sample worlds consistent with the observation, solve each, vote) or particle filter + expectimax.

Checks:
- L3 search value vs random on the toy equals an independent tabular solve
- best-response L1 vs random equals the value found by enumerating all deals
- CFR exploitability on the toy goes to ~0 with iterations


## A better evaluator (mirrored decks)

We seed and generate a deck permutation and play it from both seats: the model plays seat A with the baseline in seat B, then swapped, same permutation. If seat 1 gets a +4 both agents benefit from it. Score per deck pair:
- won both (2-0): 1
- split (1-1): 0.5
- lost both (0-0): 0

Report a confidence interval over deck pairs.

What this actually is: the score is just the mean of two paired win indicators, so it estimates the **same expectation as raw winrate**. All the variance reduction comes from pairing (duplicate bridge / duplicate poker). So the check is: CI width **per game played** (a pair costs 2 games) must be smaller than raw winrate's.

Caveats:
- the two games diverge as soon as different cards get drawn, so pairing is not a perfect control. It still reduces variance; the residual is what AIVAT is for.
- stochastic agents: fix their rng with seed + hash of the **information state** (not the full state). Same decision in the same situation. This is "common random numbers": it mostly helps when comparing two agents or two checkpoints against the same opponent.
- **this is an evaluation metric, not a training reward**. Don't feed pairs into PPO.

Checks:
- on the toy, the estimate of exact values (OPT vs random etc.) must be within CI and converge as games increase
- mirrored CI narrower than raw winrate CI at equal games played


## AIVAT
https://arxiv.org/abs/1612.06915

Summary: a provably unbiased, low-variance value estimator. It uses an arbitrary heuristic value estimate V(state) and the explicit strategy of the evaluated agent to subtract out the variance from both chance events (draws) and the agent's own randomized choices. In poker it reduces the number of hands needed by >10x.

How it works in one line: at each chance node or evaluated-agent decision, add the control variate (V of the state actually reached) - (expected V over what could have been reached). The expectation of each term is zero, so the estimate stays unbiased, but if V is decent the terms cancel most of the luck.

Engine requirements: at every draw we need V over **all possible drawn cards** (clone the state, apply each outcome), and at every agent decision we need the agent's action probabilities. Another reason for clone + deterministic core.

Choice of V: on the toy, the exact L2 value. On full UNO, a PIMC estimate or a learned critic. AIVAT stays unbiased whatever V is.

Checks:
- V = constant reproduces the raw winrate exactly
- V = exact value on the toy drives variance to ~0
- empirical variance reduction > 1 on held-out test seeds (V must not be fitted on the eval seeds)


## The ladder

Rows are agents, columns are opponents. Each cell is a matchup value (P(win) for the row agent), reported by three estimators: raw, mirrored, AIVAT. On the toy the exact value is also known.

Column semantics: "BR Lx" = best response to the row agent, computed with information level x. Those are per-row ceilings (how badly this agent can be exploited by an opponent that sees x). "Nash L1" is a single fixed policy.

| Toy UNO             | Random | Heuristic | Nash L1 | BR L1 | BR L2 | BR L3 |
|---------------------|--------|-----------|---------|-------|-------|-------|
| Random              |        |           |         |       |       |       |
| Heuristic           |        |           |         |       |       |       |
| Nash L1             |        |           |         |       |       |       |
| BR L1 to random     |        |           |         |       |       |       |
| Naive RL (PPO)      |        |           |         |       |       |       |
| RL + shaping        |        |           |         |       |       |       |

Same table for full UNO with sampled (PIMC) versions of the BR columns.

The most useful cell for an RL agent: **BR L1 to it**. Tabulate the network's policy over all toy information sets and compute an exact best response. That is its exploitability, the honest measure of its quality, independent of luck.


## Training side (only after the ladder is filled)

- If BR L1 vs random on the toy is close to the old rulebot numbers, PPO had nothing left to learn and the fix is the game, not the algorithm.
- Sparse reward fix: potential-based reward shaping with the exact L2/L3 value on the toy (unbiased: does not change the optimal policy). On full UNO use the PIMC value or a critic.
- Self-play evaluation: exploitability via best response on the toy, not just winrate vs random.


## Step 3 results: census (2026-09-14, `scripts/census.py`)

Method: the **lazy-chance tree** (`uno/tree.py`), not permutations. A draw is a
chance node over the distinct types left in the unknown part of the pile; the
deal is a chance node over (hand multisets, setup prefix). Cross-checked
against brute-force permutation enumeration on tiny decks (`tests/test_tree.py`).
Pure Python walks ~145k histories/s.

### Round 1: size

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

### Round 2: tempo

Round 1 picked O, but seat 0 won 72 % of random-vs-random games on every
candidate. Diagnosis: on O, **59 % of games contain no draw at all and seat 0
wins 100 % of those**; in games with a draw seat 0 wins 33 %. With 2 colors x
2 ranks nearly every card plays on every top, nobody gets stuck, and the game
is a race the first mover wins by construction (one card ahead all game). Not
the exhaustion rule (5 % of games). **Per-card playability** is the lever.

Forced play, hand 3, 3000 random-vs-random games (win rate, draws excluded):

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
needs an array-compiled tree or MCCFR (1000 passes x 7 s is 2 h).
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

## Step 4 results: agents and runner

`uno/agents.py`: `UniformLegal` (sanity), `PlayFirstRandom` (the ladder's
Random row), `Heuristic` (deterministic). Agents return distributions.
`uno/play.py`: `play_game` records (state, obs, probs, action) per decision;
action rng seeded from (seed, L1 info state) = common random numbers;
`match` gives score with normal CI, win rate with Wilson CI, end kinds.

`TOY`, 4000 seeds, score for the first-named agent (draws = 1/2):

| matchup                     | seat 0 | seat 1 |
|-----------------------------|--------|--------|
| random vs random            | 0.656  | 0.344  |
| heuristic vs random         | 0.738  | 0.399  |
| heuristic vs heuristic      | 0.698  | 0.302  |

Heuristic edge over random: +8 points from seat 0, +5.5 from seat 1. (On the
rejected 2-color toy it was +10 / -3: the heuristic lost to random when moving
second, another symptom of the pure race.) Exhaustion endings: 19 % of games
under play-first random, 12 % of them ties.


## Build order
1. RULES.md: parametrized ruleset, toy and full parameters
2. core engine + property tests
3. tree-size census on candidate toy sizes, pick the toy
4. random / heuristic agents
5. L3 minimax vs random and vs heuristic
6. best response L1 and L2 (exact, own implementation)
7. mirrored evaluator, validated against exact values
8. CFR for Nash L1 (own implementation first, pyspiel adapter if we want exploitability tooling)
9. AIVAT
10. skill check on the toy: is there something to learn?
11. RL
