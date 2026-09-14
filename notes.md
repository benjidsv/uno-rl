Project: UNO RL V5

This is my fifth iteration on experimenting with UNO in RL. The main problem: skill vs luck. UNO variance is enormous and the reward is sparse: during training it's hard to find good plays and take advantage of them. Most of the outcome is luck of the draw.

Previous wins (UNO 1v1, raw winrate vs random):
- rulebots: 60%
- genetic parameterized rulebot: 70%

Failures:
- no meaningful improvement from PPO agents, even with self-play etc.

Suspected causes (to be tested, not assumed):
- the evaluator was too noisy to see small improvements (a 3-point edge near 60 % is one sigma at ~270 games and three sigma at ~2400; we never ran a fixed budget)
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

Gap decomposition (free diagnostics). These are *values of information* to a
best responder (how far a ceiling moves when the responder sees more), not
variance components: do not compare them with sampling noise as if they were
the same unit.
- L3 - L2 = value of knowing the draw order
- L2 - L1 = value of knowing the opponent's hand **beyond what can be inferred**.
  BR L1 already performs exact Bayesian inference over the opponent's hand from
  the public history and the opponent's known policy, so this gap is the part
  that no opponent modelling can recover. The headroom for opponent modelling /
  inference in an L1 agent is BR L1 minus that agent's own value, not L2 - L1.
- L1 best response = the actual ceiling for the trained agent in that matchup

Full UNO: same definitions, but sampled instead of exact. L3 via search with a depth cutoff, L1/L2 via PIMC (Perfect Information Monte Carlo: sample worlds consistent with the observation, solve each, vote) or particle filter + expectimax.

Information sets and recall. The L1 information state is (seat, initial hand,
full public history, own draws in order), RULES.md section 10. It has perfect
recall, which CFR's convergence guarantee needs, and it is at least as fine as
anything an RL agent will observe, so BR L1 is the ceiling for any L1 agent,
with or without memory. The L2 key used in step 6 (L1 + the opponent's
*current* hand) does **not** have perfect recall: from the current hand the
responder cannot recover which card the opponent drew, so it cannot reconstruct
its own earlier L2 information states. Checked 2026-09-14: 13-23 % of the
reachable L2 sets on `TOY` merge histories with different opponent memories,
yet the value equals the perfect-recall key (L1 + opponent's initial hand +
opponent's draws in order) to 4 decimals against Random and Heuristic, because
neither policy depends on its memory. Against a memoryful opponent (CFR, an RL
agent with history features) that key could under-report L2. **Fixed the same
day**: L2 is now keyed (and observed) as L1 + opponent's initial hand +
opponent's draws in order, which is what "seeing the opponent's hand at every
decision" means for an observer that remembers, and the solver asserts that
every history of an information set has the same legal actions.

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

What this actually is: the score is just the mean of two paired win indicators, so it estimates the **same expectation as seat-averaged raw winrate**. Seat is worth ~15 points on `TOY` (step 3), so the pair score on its own pools seats, which rule 1 of step 3 forbids. Report three numbers per matchup: the seat-0 estimate and the seat-1 estimate (the two games of every pair, so they share the deal) and their mean as the summary. Paired differences between two agents are taken per seat. All the variance reduction comes from pairing (duplicate bridge / duplicate poker). So the check is: CI width **per game played** (a pair costs 2 games) must be smaller than raw winrate's.

Caveats:
- the two games diverge as soon as different cards get drawn, so pairing is not a perfect control. It still reduces variance; the residual is what AIVAT is for.
- stochastic agents: fix their rng with seed + hash of the **information state** (not the full state). Same decision in the same situation. This is "common random numbers": it mostly helps when comparing two agents or two checkpoints against the same opponent.
- **this is an evaluation metric, not a training reward**. Don't feed pairs into PPO.

Checks:
- on the toy, the estimate must match the exact value **on the same deals** (`evaluate_on_seeds`), per seat, to within the CI; a plain "exact inside the CI" test fails one time in twenty by construction
- mirrored CI narrower than raw winrate CI at equal games played


## AIVAT
https://arxiv.org/abs/1612.06915

Summary: a provably unbiased, low-variance value estimator. It uses an arbitrary heuristic value estimate V(state) and the explicit strategy of the evaluated agent to subtract out the variance from both chance events (draws) and the agent's own randomized choices. In poker it reduces the number of hands needed by >10x.

How it works in one line: at each chance node or evaluated-agent decision, add the control variate (V of the state actually reached) - (expected V over what could have been reached). The expectation of each term is zero, so the estimate stays unbiased, but if V is decent the terms cancel most of the luck.

Engine requirements: at every draw we need V over **all possible drawn cards** (clone the state, apply each outcome), and at every agent decision we need the agent's action probabilities. Another reason for clone + deterministic core.

Choice of V: on the toy, the exact value of the state **under the two policies actually being evaluated** (`evaluate` from that state as root; for an RL row this needs its policy tabulated). A best-response value is also unbiased but does not zero the variance. On full UNO, a PIMC estimate or a learned critic. AIVAT stays unbiased whatever V is.

Checks:
- V = constant reproduces the raw winrate exactly
- V = exact value of the matchup on the toy drives variance to ~0, provided correction terms are applied at every stochastic node whose distribution is known: chance, the evaluated agent, and the opponent when its policy is known (always, on the toy). With corrections only at chance and evaluated-agent nodes, a stochastic opponent's action noise stays in
- empirical variance reduction > 1 on held-out test seeds (V must not be fitted on the eval seeds)


## The ladder

Rows are agents, columns are opponents. Each cell is a matchup value (P(win) for the row agent), reported by three estimators: raw, mirrored, AIVAT. On the toy the exact value is also known.

Column semantics: "BR Lx" = best response to the row agent, computed with information level x. Those are per-row ceilings (how badly this agent can be exploited by an opponent that sees x). "Nash L1" is one fixed policy but not a canonical one: the Nash *value* is unique, the strategy is not, and different equilibria score differently against Random and Heuristic. So the row and the column are "the CFR average strategy after N iterations with exploitability ε", and N and ε go next to every cell that uses it. Every cell holds two numbers, seat 0 / seat 1 (rule 1 of step 3).

| Toy UNO             | Random | Heuristic | Nash L1 | BR L1 | BR L2 | BR L3 |
|---------------------|--------|-----------|---------|-------|-------|-------|
| Random              |        |           |         |       |       |       |
| Heuristic           |        |           |         |       |       |       |
| Nash L1             |        |           |         |       |       |       |
| BR L1 to random     |        |           |         |       |       |       |
| Naive RL (PPO)      |        |           |         |       |       |       |
| RL + shaping        |        |           |         |       |       |       |

Same table for full UNO with sampled (PIMC) versions of the BR columns.

The most useful cell for an RL agent: **BR L1 to it**. Tabulate the network's policy over all toy information sets and compute an exact best response. Exploitability proper needs the Nash value: per seat, exploitability = (BR L1 value against the agent) - (Nash L1 value at that seat), and NashConv is the sum over the two seats. Until step 8 delivers the Nash value, the BR L1 column is exploitability up to an unknown per-seat constant: good for *ranking* agents, not as an absolute. Either way it is independent of luck.


## Training side (only after the ladder is filled)

- If BR L1 vs random on the toy is close to the old rulebot numbers, PPO had nothing left to learn and the fix is the game, not the algorithm.
- Sparse reward fix: potential-based reward shaping. Policy-invariant for any potential with Φ(terminal) = 0, but the potential should be a function of the **agent's own information set** (an L1 value: BR L1 or the Nash L1 value). An L2/L3 potential depends on cards the agent cannot see, so the shaping term adds noise the agent cannot condition on and the critic still has to average it out. On full UNO use the PIMC value or a critic, again at L1.
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
   pile-parity / card-count tactic of RULES.md section 9, which full UNO does
   not have. Conclusions about what PPO could or could not learn transfer only
   for the decisions the toy shares with full UNO (color management, holding
   the wild, the hold-after-draw choice).

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
under play-first random; 12 % of all games are ties (about 60 % of the
exhaustion endings).


## Step 4.5: fixes (2026-09-14)

- `step(state, action, check=False)` skips the legality check; the tree walk
  uses it (it just computed `legal_actions`). Census pass 9.5 s -> ~8 s.
- `match` only treats a `ValueError` from the *deal* as an invalid seed; an
  error inside a game now propagates instead of being counted as invalid.
- The stable-hash test asserted `x == constant or True`; pinned the constant.
- Observation "leak" at non-decision nodes: not fixable and not a problem. The
  leak is the timing (opponent still to move after their own draw event =>
  the drawn card was playable), not a field. Documented in RULES.md section 10:
  L1/L2 observations are only meaningful at the observer's own decisions.
- Pass after a playable draw stays legal under forced play (intended).

## Steps 5-6 results: exact best responses on `TOY` (2026-09-14, `uno/solve.py`, `scripts/ladder.py`)

One best-response routine, three information-set keys (L1 `info_state`, L2 =
L1 + opponent's initial hand + opponent's draws in order, L3 = full state with
pile order). L1/L2 walk the lazy tree; L3 enumerates *worlds* (deal x pile
order) and walks the bare engine. (The first version keyed L2 on the
opponent's *current* hand to keep its memory out of L2; that partition has
imperfect recall, see "Information sets and recall" above. Same values on
Random and Heuristic to 4 decimals.)

Cost on `TOY`: 12-16 s and ~1 GB per L1/L2 solve against Random (0.94M
histories reachable under Random's support, out of 1.25M), 4-7 s against the
deterministic Heuristic. No array-compiled tree yet: that is for CFR (step 8).

Checks (all in `tests/test_solve.py`): value == max over every pure strategy
on tiny rulesets (independent brute force); L3 worlds == distinct deck
permutations with their frequencies; table replayed through `evaluate` ==
solver value; L1 <= L2 <= L3; BR dominates Random / Heuristic / UniformLegal
as responder; the three levels coincide on a pile-0 ruleset; simulation
matches the exact value on the same deals.

**Deal luck is visible at 4000 seeds.** The first simulation check (exact
value inside the plain CI) failed for seat 1: seeds 0-4000 favour seat 0 by
~1.5 points, and every measurement on that seed range shifts together (the
step 4 table too: random vs random sampled 0.656 / 0.344, exact 0.649 /
0.351). `evaluate_on_seeds` computes the exact value on the dealt seeds with
their real pile order (bare-engine expansion from the real initial state), and
the simulation matches that to within action-sampling noise only; for two
deterministic agents the two are equal. A
1.5-point shift at 4000 games is 2 sigma, the miss a plain 95 % CI test
produces one time in twenty by construction; the on-seeds comparison is the
test, and it now covers L3 too. This
is the mirrored evaluator's and AIVAT's job in numbers: at 4000 games the
sampling noise of the deal alone (1.5 points) is larger than the L3 - L2 gap
(a value of information, 0.2-0.9 points): different units, but the noise
swamps the signal all the same.

Exact values of the row agents (score of the first-named, seat 0 / seat 1;
step 4 sampled values in brackets):

| matchup                | seat 0          | seat 1          |
|------------------------|-----------------|-----------------|
| random vs random       | 0.6493 [0.656]  | 0.3507 [0.344]  |
| heuristic vs random    | 0.7390 [0.738]  | 0.4077 [0.399]  |
| heuristic vs heuristic | 0.6891 [0.698]  | 0.3109 [0.302]  |

Best-response columns (exact score of the responder, per seat):

| opponent  | seat | BR L1  | BR L2  | BR L3  | L2-L1 | L3-L2 | row agent itself |
|-----------|------|--------|--------|--------|-------|-------|------------------|
| random    | 0    | 0.7934 | 0.8188 | 0.8280 | +2.5  | +0.9  | 0.6493 (random)  |
| random    | 1    | 0.5513 | 0.5937 | 0.5989 | +4.2  | +0.5  | 0.3507 (random)  |
| heuristic | 0    | 0.7447 | 0.7770 | 0.7837 | +3.2  | +0.7  | 0.6891 (heur.)   |
| heuristic | 1    | 0.4525 | 0.4806 | 0.4825 | +2.8  | +0.2  | 0.3109 (heur.)   |

Re-run 2026-09-14 after the L2-key / on-seeds fixes: all 12 best-response
values unchanged to 4 decimals; every row passes the on-seeds check, L3 now
included (`evaluate` builds its default roots lazily, so the per-seed L3 value
costs 0.8 s for 4000 seeds instead of 15 min). Against the deterministic
Heuristic the simulated score equals the on-seeds exact value to the last
digit, as it must; against Random the residual is 0.2-1.1 points of action
noise, inside the CI. The seat-1 deal luck of seeds 0-4000 is visible as the
on-seeds column sitting ~1.1 points under the exact column on every seat-1 row.

SIMCHECK

Reading:

1. **There is skill to learn at L1.** Against Random, the L1 ceiling is +14.4
   points over Random itself from seat 0 and +20.1 from seat 1; the Heuristic
   only takes +9.0 / +5.7 of that. So the Heuristic leaves 5.4 / 14.4 points
   on the table against Random, most of it from seat 1. The old "60-70 %
   rulebot, PPO no better" story is testable now: on this toy an L1 learner
   has ~5-14 points of room above the rulebot.
2. **Hidden information is worth 2.5-4.2 points, the pile order 0.2-0.9.** The
   L3 - L2 gap is small because the pile holds 2 cards; the L2 - L1 gap (the
   opponent's hand) is the bigger prize. BR L1 already infers the opponent's
   hand as well as the public history allows, so the L2 - L1 gap is the part
   that *no* inference can recover; the room for opponent modelling in an L1
   agent is BR L1 minus the agent's own value (5.4 / 14.4 points for the
   Heuristic against Random).
3. **Seat dominates everything.** BR L1 vs Random: 0.79 from seat 0, 0.55 from
   seat 1. Even the L3 oracle moving second against Random only reaches 0.60.
   All ladder cells stay per seat (rule from step 3).
4. **The Heuristic is exploitable but not badly**: BR L1 to it gets 0.745 /
   0.453, less than BR L1 to Random (0.793 / 0.551), so it is the stronger
   opponent, as expected. Its exploitability at L1 is BR L1 against it minus
   the Nash L1 value at that seat (step 8); that difference, not the BR score
   itself, is the quality measure for step 10 onwards.


## Step 7 results: mirrored evaluator (2026-09-14, `uno/estimate.py`, `scripts/mirror.py`)

Design: play once, estimate many ways. `play_pairs` plays every seed from
both seats (same deal, same pile order) and keeps the traces; raw, mirrored
and paired-comparison estimates are pure functions of the records, and AIVAT
(step 9) will be a fourth one over the same games. Everything below is exact
where it can be: `exact_mirrored` walks the 10,080 worlds of `TOY` (deal x
pile order) and gets the per-seat means, variances, the covariance between
the two seats' scores on the same deal and the share of split pairs; the
sampled numbers must match them. RULES.md section 15.

What was built:

- `uno/estimate.py`: `play_pairs`, `estimate_mirrored`, `compare` (paired
  difference of two row agents against the same column, same deals),
  `exact_mirrored`, `games_needed`.
- `uno/solve.py`: `outcome_distribution`, the exact win / draw / loss mass
  under two fixed agents; `evaluate` is now its mean. This is what makes the
  exact variance available for stochastic agents, not just deterministic ones.
- `uno/agents.py`: `BySeat`, so a per-seat best-response table can play both
  seats of a pair (the BR rows of the ladder).
- `uno/play.py`: `DEV_SEEDS` / `EVAL_SEEDS` (below). `scripts/ladder.py`
  reports on the held-out range from now on.
- `scripts/mirror.py` prints the tables below; `tests/test_estimate.py` (8
  tests) holds the checks listed in RULES.md section 15. Whole suite: 66
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
`scripts/ladder.py` now reports on it too. The old range's 1.5-point seat-0
tilt was harmless while every number was compared on the same seeds, and
becomes a bias the moment an agent, a value function or a hyperparameter is
picked because it scored well there. The held-out range has its own tilt
(seat 1 sits 0.8 points *under* exact for Heuristic vs Random), which is the
point: "on seeds" is the test, the global exact value is the reference.

### Matchup values, 4000 held-out pairs (score of the first-named agent)

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

### What mirroring buys (exact / sampled)

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

### Paired comparisons (first minus second row agent, same column, same deals)

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


## Step 8 results: compiled tree and Nash L1 by CFR+ (2026-09-14, `uno/compiled.py`, `uno/cfr.py`, `scripts/cfr.py`)

### What was built

The step was going to be "CFR", and the blocker was speed: the pure-Python
walkers cost 7-16 s per pass over `TOY` and CFR needs hundreds of passes. So
the tree is now walked **once** into flat numpy arrays (`compile_tree`, RULES.md
section 16) and every exact computation became a vectorized pass over them,
CFR included: evaluate, outcome distributions, best response at L1 / L2 / L3,
exploitability, exact mirrored moments. The old walkers in `uno/solve.py` and
`uno/estimate.py` stay as the independent reference; `tests/test_compiled.py`
pins the compiled versions to them on small rulesets and `scripts/cfr.py`
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
  decision of random games, all three levels (`tests/test_observe_keys.py`).

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

### CFR+ on `TOY`

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
`.cache/cfr_TOY.npz`). CFR+ is 40x ahead of vanilla at equal iterations and
the vanilla curve is the textbook 1 / sqrt(T). Every cell below that uses the
Nash agent means "CFR+ average strategy, 1000 iterations, NashConv 0.001 pts".

### Exploitability of the row agents (now absolute)

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

### Nash row and column (exact, score of the first-named, seat 0 / seat 1)

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

### Best responses to the Nash agent by information level

| seat | BR L1  | BR L2  | BR L3  | L2 - L1 | L3 - L2 | against Random (step 6) |
|------|--------|--------|--------|---------|---------|-------------------------|
| 0    | 0.6196 | 0.6630 | 0.6764 | +4.3    | +1.3    | +2.5 / +0.9             |
| 1    | 0.3804 | 0.4240 | 0.4311 | +4.4    | +0.7    | +4.2 / +0.5             |

### Reading

1. **The first-mover edge under optimal play is 12 points** (0.6196 vs 0.3804),
   against 30 under random play (0.649) and 38 in Heuristic self-play
   (0.689). Skill narrows the seat effect but does not remove it; the ladder
   stays per seat.
2. **The exploitability column of the ladder is now absolute.** Random gives
   up 17 points per seat, the Heuristic 7 from seat 0 and 12.5 from seat 1
   (its seat-1 play is the weak side, consistent with step 6), a best response
   to Random only 1.9 / 0.6. That last number is the surprise: **on `TOY`
   the best response to Random is a near-Nash strategy**, and against Nash it
   scores 0.6123 / 0.3797, within 0.7 / 0.1 points of the Nash value. Playing
   well against a random opponent and playing unexploitably are almost the
   same thing on this toy; the two will separate on games with more room for
   deception.
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


## Step 8.5 results: transfer check, `TOY` against `FULL_UNO` (2026-09-14, `scripts/transfer.py`)

Question: step 8 made exact computation 45-130x faster; should the toy grow
because it "probably won't translate to real UNO"? Before choosing how, measure
where `TOY` and `FULL_UNO` differ. Part 1 is sampled, the same code on both
rulesets, 4000 held-out pairs. Part 2 is exact on `TOY`. For part 2,
`CompiledTree` now records how each terminal ended (`end`, `end_columns`,
RULES.md section 16), tested against the census end mass.

### 1a. Matchup values (score of the first-named, 95 % CI)

| ruleset  | matchup                | seat 0          | seat 1          | actions / game | exhaustion endings | ties  |
|----------|------------------------|-----------------|-----------------|----------------|--------------------|-------|
| TOY      | random vs random       | 0.6480 ± 0.0138 | 0.3520 ± 0.0138 | 6.5            | 0.185              | 0.122 |
| TOY      | heuristic vs random    | 0.7346 ± 0.0130 | 0.4004 ± 0.0148 | 6.2            | 0.099              | 0.066 |
| TOY      | heuristic vs heuristic | 0.6921 ± 0.0141 | 0.3079 ± 0.0141 | 5.9            | 0.026              | 0.023 |
| FULL_UNO | random vs random       | 0.5020 ± 0.0155 | 0.4980 ± 0.0155 | 55.1           | 0                  | 0     |
| FULL_UNO | heuristic vs random    | 0.6162 ± 0.0151 | 0.5905 ± 0.0152 | 44.5           | 0                  | 0     |
| FULL_UNO | heuristic vs heuristic | 0.5078 ± 0.0155 | 0.4923 ± 0.0155 | 37.4           | 0                  | 0     |

No `FULL_UNO` game hit the 1000-action cap.

### 1b. Luck: mirroring Heuristic vs Random; games for a 3-point edge at 3 sigma

| ruleset  | corr(seat 0, seat 1) | splits | var ratio | seat 0 alone | seat 1 alone | seat mean raw | mirrored |
|----------|----------------------|--------|-----------|--------------|--------------|---------------|----------|
| TOY      | -0.320               | 0.623  | 0.682     | 1747         | 2277         | 2012          | 1373     |
| FULL_UNO | -0.092               | 0.523  | 0.908     | 2365         | 2419         | 2392          | 2173     |

### 1c. Skill proxy: Heuristic minus Random, both against Random, same deals

| ruleset  | seat | paired diff      | unpaired ± | corr   | var ratio | games for 3 pts, unpaired / paired |
|----------|------|------------------|------------|--------|-----------|------------------------------------|
| TOY      | 0    | +0.0866 ± 0.0128 | ± 0.0189   | +0.543 | 0.458     | 7444 / 3410                        |
| TOY      | 1    | +0.0484 ± 0.0136 | ± 0.0202   | +0.548 | 0.453     | 8504 / 3855                        |
| FULL_UNO | 0    | +0.1143 ± 0.0206 | ± 0.0216   | +0.095 | 0.905     | 9732 / 8806                        |
| FULL_UNO | 1    | +0.0925 ± 0.0204 | ± 0.0217   | +0.119 | 0.881     | 9839 / 8671                        |

### 1d. Decisions by kind (Heuristic vs Random games)

A decision counts under every kind it matches. Counts are per game, both
players together; shares are of all decisions.

| decision kind                          | TOY / game | share | FULL_UNO / game | share |
|----------------------------------------|------------|-------|-----------------|-------|
| decisions                              | 6.18       | 1     | 44.46           | 1     |
| any choice (2+ legal actions)          | 2.87       | 0.464 | 31.50           | 0.709 |
| which color: plays in 2+ colors        | 1.36       | 0.220 | 4.93            | 0.111 |
| wild: hold or spend                    | 0.75       | 0.122 | 3.58            | 0.081 |
| wild: pick a color                     | 1.14       | 0.184 | 4.53            | 0.102 |
| action card among other plays          | 0          | 0     | 9.03            | 0.203 |
| draw with a play in hand               | 0          | 0     | 26.80           | 0.603 |
| play or keep the drawn card            | 0.51       | 0.082 | 4.70            | 0.106 |

### 2. `TOY`, exact: how games end and where the points come from

The row agent sits at the given seat. "From X" is the score collected in games
that end by X. "Within" is the score given that the game ended by exhaustion.

| row vs column          | seat | score  | P(exhaustion) | from emptied hand | from exhaustion | within exhaustion |
|------------------------|------|--------|---------------|-------------------|-----------------|-------------------|
| random vs random       | 0    | 0.6493 | 0.188         | 0.5420            | 0.1073          | 0.570             |
|                        | 1    | 0.3507 | 0.188         | 0.2696            | 0.0811          | 0.430             |
| heuristic vs random    | 0    | 0.7390 | 0.108         | 0.6913            | 0.0477          | 0.442             |
|                        | 1    | 0.4077 | 0.108         | 0.3739            | 0.0338          | 0.315             |
| BR L1(random) vs random| 0    | 0.7934 | 0.190         | 0.6697            | 0.1237          | 0.653             |
|                        | 1    | 0.5513 | 0.150         | 0.4866            | 0.0647          | 0.431             |
| nash vs random         | 0    | 0.7871 | 0.210         | 0.6492            | 0.1379          | 0.656             |
|                        | 1    | 0.5235 | 0.145         | 0.4696            | 0.0539          | 0.372             |
| heuristic vs heuristic | 0    | 0.6891 | 0.033         | 0.6722            | 0.0169          | 0.518             |
|                        | 1    | 0.3109 | 0.033         | 0.2952            | 0.0157          | 0.482             |
| nash vs heuristic      | 0    | 0.7357 | 0.107         | 0.6540            | 0.0816          | 0.760             |
|                        | 1    | 0.4314 | 0.070         | 0.3944            | 0.0370          | 0.533             |
| nash vs nash           | 0    | 0.6196 | 0.147         | 0.5194            | 0.1002          | 0.682             |
|                        | 1    | 0.3804 | 0.147         | 0.3337            | 0.0467          | 0.318             |

Edge over Random at the same seat, both against Random, split by the ending
the points come through (score points):

| row           | seat | edge   | via emptied hand | via exhaustion |
|---------------|------|--------|------------------|----------------|
| heuristic     | 0    | +8.97  | +14.93           | -5.96          |
| heuristic     | 1    | +5.70  | +10.43           | -4.73          |
| BR L1(random) | 0    | +14.41 | +12.77           | +1.64          |
| BR L1(random) | 1    | +20.06 | +21.70           | -1.64          |
| nash          | 0    | +13.78 | +10.72           | +3.06          |
| nash          | 1    | +17.28 | +20.00           | -2.72          |

This is an accounting identity, not a causal split. An agent that avoids
losing by exhaustion moves probability mass from "exhaustion" to "emptied
hand". So "via exhaustion ≈ 0" does not prove the pile-parity tactic is
worthless. The causal test would be a variant ruleset that scores exhaustion
differently.

### Reading

1. **The seat effect is a toy artefact.** Seat 0 is worth 15 points on `TOY`
   and nothing measurable on `FULL_UNO`: random self-play gives 0.502 ± 0.016,
   Heuristic self-play 0.508. Reporting per seat stays harmless on full UNO but
   buys nothing there.
2. **Heuristic vs Random on full UNO is the old 60 % rulebot**: seat mean
   0.603. It beats Random by +11.4 / +9.3 points, more than on `TOY`
   (+8.7 / +4.8). So full UNO has at least as much skill against Random as the
   rulebot shows. Its ceiling is still unknown.
3. **The luck has a different structure.** On `TOY` the luck is in the deal:
   mirroring removes 32 % of the variance and pairing two agents on the same
   deals removes 55 %. On `FULL_UNO` the luck is in the draws during a
   44-action game. The two games of a pair decouple almost immediately, so
   mirroring removes 9 % and pairing about 10 %. The share of split pairs
   (0.52) is near the 0.48 that two independent games at these per-seat
   values would give; on `TOY` it is 0.62 against 0.55. **The step 7
   conclusions about pairing do not transfer.** On full UNO a 3-point
   comparison between two agents costs ~9000 games either way, and AIVAT,
   which corrects at every draw, is the estimator that can help.
4. **The two most frequent full-UNO decisions do not exist on `TOY`.** "Draw
   with a play in hand" is 60 % of all decisions (26.8 per game) and
   action-card timing is 20 %, both zero on `TOY`. Nobody knows yet whether
   voluntary drawing is ever right. If it almost never is, it is a large but
   nearly empty decision. Color choice and the wild's hold / spend / color
   decisions exist on both, at comparable shares of decisions (8-11 % full,
   12-22 % toy), with 3-5x more of them per game on full UNO.
5. **Exhaustion is less of a distortion than feared.** Strong play does not
   drift into exhaustion: 15 % of Nash-vs-Nash games, against 19 % under
   Random, and RULES.md section 9 expected more. The best responder's and
   Nash's edges over Random arrive mostly through emptied hands (exhaustion
   contributes -2.7 to +3.1 points). The Heuristic *loses* exhaustion endings
   (0.44 within them at seat 0, Random gets 0.57): it gives back 5-6 points
   there (holding the wild is a guess, not checked) and wins them back by
   emptying its hand.

Consequence for "should the toy grow": the gap is not the size of `TOY` but
**the decisions it lacks** (voluntary draw, action cards) and **the length of
its games** (in-game draw luck, which mirroring cannot pair away). A `TOY2`
should add the missing decisions, not just cards. Measuring the value of the
voluntary draw first (`TOY_FULL` against `TOY`, both best responses to Random)
would say whether 60 % of full UNO's decisions carry any skill. Long-game
luck cannot be made exact; that part belongs to sampled estimators (AIVAT),
validated on the toys.


## Step 8.6 results: symmetry, compact keys, tree cache (2026-09-14, `uno/symmetry.py`, `scripts/symmetry.py`)

### Why

Two goals after step 8.5: a bigger toy (the one we have lacks the voluntary
draw and action cards, and seat 0 is worth 15 points) and a faster exact
stack, since every ladder cell of steps 10-11 is an exact computation per
checkpoint. Before building, three things were measured on `TOY`:

- where the memory goes: the compiled tree with an L1 keying is 76 MB of
  numpy arrays and ~580 MB of Python key dictionaries (nested tuples); with
  L1 + L2 it is 1.1 GB, 90 % keys. The numpy passes were never the problem
  (reach 0.04 s, values 0.02 s, best response 0.11 s per 1.25M nodes, i.e.
  30-60M nodes/s), so float32 would have saved ~30 MB of transient arrays
  and nothing else; it was dropped;
- the walk: 140k histories/s of pure-Python engine, 9-14 s per tree;
- the **symmetry**: the rules never name a color or a rank, so renaming
  colors (4! = 24 ways) and ranks (2! = 2 ways, ranks with equal copies) maps
  the game onto itself. One subtree per *orbit* of deals instead of one per
  deal, one information set per orbit of keys. Measured on `TOY`: 5,600 deals
  -> 144 orbits, 1,252,040 -> 33,646 histories (37x), 558,656 -> 12,036 L1
  sets (46x). The group grows factorially with the deck (144 for three
  one-copy ranks, 240 for five colors, 576 for four ranks), so this is the
  lever that makes a bigger toy affordable, while memory and walk speed only
  make the same toy cheaper.

Decisions: build the symmetric quotient (default for every compiled tree),
store keys as bytes, cache compiled trees and CFR results in `.cache/`. numba
(installs fine on this Python 3.14), a faster walker and discounted /
predictive CFR wait until the gain from these three is measured. Seat parity
is not addressed here: the candidate census below says it comes from game
length and playability, not size, and the mirrored evaluator handles it.

### What was built

- `uno/symmetry.py` (RULES.md section 18): the group as lookup tables; its
  action on hands, public histories, keys of every level, states and
  policies; `canonical(level, key)` = lexicographic minimum over the group
  with the element reaching it (search pruned on the observer's hand, the
  candidates cached per hand); `stabilizer` and `action_orbits`;
  `canonical_roots` for deals (unknown pile as a multiset) and worlds (pile
  order kept).
- `initial_deals(rs, symmetric)`, `worlds(rs, symmetric)`, `census(rs,
  symmetric)` (counts canonical L1 sets).
- `compile_tree(rs, levels, symmetric=True, cache=None)`: canonical roots
  with orbit mass, canonical keys encoded as bytes (`encode_key` /
  `decode_key` in `uno/solve.py`), each arc's set-action found through its
  node's canonicalising element, per-set **action orbits** under the
  stabilizer of the canonical key; `CompiledTree.save` / `load`, `nbytes`,
  `cache_name`. `CompiledAgent` canonicalises the observation and maps the
  action back with the inverse element; `tabulate` refuses an agent that is
  not symmetric; `tabulate_table` spreads a pure choice over its orbit.
- `best_response` chooses the best action *orbit* and plays it uniformly;
  `cfr` averages regrets and average-strategy mass within orbits.
- `Heuristic` made symmetric (RULES.md section 13): the old "ties: lowest
  type id" is replaced by "keep the current color, then a number card, then
  uniform". Random and UniformLegal were symmetric already.
- `uno/play.py`: the action rng is keyed on (seed, **agent name**, information
  state), see "a coupling found by the tests" below.
- `tests/test_symmetry.py` (8 tests, 13 s): the group acts on the engine
  (`g . step(s, a) == step(g . s, g . a)`, legal actions, returns, keys of
  every level, inverses), key encoding round-trips, canonical forms are
  invariant / reached / fixed by their stabilizer / minimal, the three
  baseline agents commute with the group, quotient census == full census
  (end mass, UniformLegal score), quotient tree == full tree (evaluate for
  every pair of baselines, best response at L1 / L2 / L3 against each,
  symmetric strategies replaying to their value through the engine, the
  exploitability of the uniform profile), `tabulate` rejects an asymmetric
  agent on `TOY`, save / load / cache round trip, and the twelve `TOY`
  values plus the Nash value on the quotient. `test_compiled.py` keeps the
  full-tree checks explicitly (`symmetric=False`) with the new Heuristic
  values pinned. Suite: 82 tests, ~2.7 min.

### A bug the small rulesets did not catch, again

The first symmetric best response reproduced every L2 and L3 value of steps
5-6 and was 1.3-1.9 points *above* the L1 ones (0.8065 vs 0.7934 for seat 0
against Random). A best response above the true maximum is not a strategy.
The cause: at a canonical set whose key has a non-trivial stabilizer (hand
`{red 1, blue 1, W}` on a green top is fixed by swapping red and blue), a
pure choice of "red 1" lifted through each node's own canonicalising element
plays red 1 at some histories of the real information set and blue 1 at
others, i.e. it conditions on which side of the symmetry the history came
from, which correlates with the opponent's hidden hand. L2 and L3 sets are
(nearly) singletons, where no stabilizer moves anything. The fix is the
orbit structure above: a symmetric strategy must be uniform within every
orbit of a set's actions under its stabilizer, and a symmetric best response
(which always exists against a symmetric opponent) picks the best orbit. 284
of the 12,036 canonical L1 sets of `TOY` have such an orbit. The same
constraint applies to CFR (regret averaging within orbits), otherwise its
"strategies" would not be strategies either.

### A coupling found by the tests

With the Heuristic now stochastic on ties (10.2 % of its decisions against
Random on `TOY` are played uniformly over 2+ cards), the mirrored check of
step 7 failed on the correlation between the two seats' scores: sampled
-0.364 on 4000 pairs against the exact -0.300 (4.5 sigma). The action rng
was keyed on (seed, information state), so in a mirrored pair the Heuristic
in game 2 and Random in game 1 sit at the same seat, meet the same
information states, and draw the *same* random numbers: when the Heuristic's
tie set equals Random's play set the two games make the same move. That
couples the two games beyond the shared deal, and `exact_mirrored` assumes
independence given the world. Deterministic agents never triggered it. The
key now includes the agent's name: different agents never share a stream,
and the same agent in the same situation still makes the same choice across
matchups (what `compare` needs). Sampled tables of steps 7-8.5 were drawn
with the old streams; the exact numbers do not move.

### Sizes and costs on `TOY` (M1 Pro, one core)

| tree | roots | nodes | L1 sets | other keying | compile | bytes (arrays + keys) | load from cache |
|------|-------|-------|---------|--------------|---------|-----------------------|-----------------|
| full lazy L1+L2 | 5,600 | 1,252,041 | 558,656 | L2 876,848 | 14.0 s | 179 MB | 0.34 s |
| quotient lazy L1+L2 | 144 | 33,647 | 12,036 | L2 18,620 | 1.4 s | 4.2 MB | 0.01 s |
| full world L1+L3 | 10,080 | 1,384,297 | 558,656 | L3 986,496 | 13.3 s | 196 MB | 0.38 s |
| quotient world L1+L3 | 242 | 33,998 | 12,036 | L3 20,882 | 1.4 s | 4.4 MB | 0.01 s |

The full trees are 6x smaller than before from the bytes keys alone (1.1 GB
-> 179 MB with L1 + L2); the quotient is another 40x. The 5-6 s tabulation
loop is gone with it (46x fewer sets).

| computation on `TOY` | full tree | quotient | speedup |
|----------------------|-----------|----------|---------|
| best response, any level and seat | 0.35-0.38 s | 0.01 s | ~35x |
| one CFR+ iteration | 200 ms (step 8) | 5.2 ms | 38x |
| CFR+ to NashConv 0.001 pts (1000 iterations) | 203 s | 5.2 s | 39x |
| exploitability of a tabulated agent | ~0.2 s | ~0.02 s | |

### The twelve values, both trees, and the new Heuristic

Random column unchanged to 4 decimals on both trees. The Heuristic's values
moved with its tie-break (pinned with the pure-Python reference solver of
`uno/solve.py` on the full game, then reproduced on the full and the quotient
compiled trees):

| opponent | seat | BR L1 | BR L2 | BR L3 | L2-L1 | L3-L2 | old (step 6) L1 / L2 / L3 |
|----------|------|-------|-------|-------|-------|-------|---------------------------|
| random | 0 | 0.7934 | 0.8188 | 0.8280 | +2.5 | +0.9 | same |
| random | 1 | 0.5513 | 0.5937 | 0.5989 | +4.2 | +0.5 | same |
| heuristic | 0 | 0.7378 | 0.7730 | 0.7798 | +3.5 | +0.7 | 0.7447 / 0.7770 / 0.7837 |
| heuristic | 1 | 0.4361 | 0.4774 | 0.4786 | +4.1 | +0.1 | 0.4525 / 0.4806 / 0.4825 |

Row matchups: random vs random 0.6493 / 0.3507 (same), heuristic vs random
0.7416 / 0.4147 (was 0.7390 / 0.4077), heuristic vs heuristic 0.6934 / 0.3066
(was 0.6891 / 0.3109). Mixing on ties makes the Heuristic both a little
stronger against Random and less exploitable: 5.6 / 11.8 points (was 7.2 /
12.5). The best response to Random is now the *symmetric* one (uniform over
tied orbits), which scores the same against Random and is less exploitable
than the lowest-id pure one of step 8: 1.6 / 0.4 points (was 1.9 / 0.6), an
instance of "symmetrising never raises exploitability".

Nash L1 on the quotient: seat 0 scores **0.6196** (NashConv 0.001 pts after
1000 CFR+ iterations, 5.2 s), the step 8 value. Nash row / column with the
new Heuristic: nash vs heuristic 0.7328 / 0.4139, heuristic vs nash 0.5861 /
0.2672; nash vs random 0.7871 / 0.5235 and random vs nash 0.4765 / 0.2129
unchanged; br_l1(random) vs nash 0.6149 / 0.3792.

### `TOY_FULL` solved exactly (the worth of the voluntary draw)

`TOY_FULL` (`TOY` without forced play) was "> 2e7 histories, a name for
later" in step 3. Its quotient: 774,008 nodes, 138,876 L1 sets, 303,492 L2
sets, 86 MB, 17 s to compile (world tree 730,599 nodes, 305,754 L3 sets,
13 s); best response 0.2 s, CFR+ 110 ms per iteration.

| opponent | seat | BR L1 | BR L2 | BR L3 | L2-L1 | L3-L2 | minus `TOY`, L1 / L2 / L3 (pts) |
|----------|------|-------|-------|-------|-------|-------|----------------------------------|
| random | 0 | 0.8017 | 0.8443 | 0.8590 | +4.3 | +1.5 | +0.8 / +2.6 / +3.1 |
| random | 1 | 0.5882 | 0.6419 | 0.6941 | +5.4 | +5.2 | +3.7 / +4.8 / +9.5 |
| heuristic | 0 | 0.7446 | 0.8060 | 0.8187 | +6.1 | +1.3 | +0.7 / +3.3 / +3.9 |
| heuristic | 1 | 0.4732 | 0.5131 | 0.5437 | +4.0 | +3.1 | +0.7 / +3.6 / +6.5 |

Row matchups are identical to `TOY` (neither Random nor the Heuristic ever
draws voluntarily). **Nash L1 of `TOY_FULL`: seat 0 scores 0.5866, seat 1
0.4134** (NashConv < 0.001 pts, 1000 iterations, 110 s). Exploitability of
the rows: random 17.5 / 21.5, heuristic 6.0 / 15.8, br_l1(random) 4.6 / 1.3.
Nash vs random 0.7595 / 0.5123, nash vs heuristic 0.7065 / 0.4372.

Voluntary draws with a play in hand, expected per game (seat 0 / seat 1 of
the named profile): br_l1(random) vs random 0.029 / 0.184, br_l1(heuristic)
vs heuristic 0.020 / 0.178, nash vs nash 0.228 / 0.372, nash vs random 0.244
/ 0.433.

Reading:

1. **The voluntary draw is a real decision, mostly for seat 1.** At L1 it is
   worth 0.7-0.8 points to seat 0 and 3.7 points to seat 1 against both
   opponents; at equilibrium it moves 3.3 points from seat 0 to seat 1 (the
   seat gap under optimal play shrinks from 12 to 8.7 points). Seat 1 draws
   with a play in hand once every 5-6 games against a fixed opponent and
   once every 2-3 games at equilibrium; seat 0 almost never against a fixed
   opponent (0.02-0.03 per game) but 0.23 per game at equilibrium. So on
   this toy "draw with a play in hand" is a tempo tool of the trailing seat
   (the pile-parity tactic of RULES.md section 9), and its worth grows with
   information: 3-5 points at L2, up to 9.5 at L3 for seat 1, since with the
   pile known a draw fetches a chosen card.
2. **Exploitability rises with the option.** Random and the Heuristic give
   up 15.8-21.5 points at seat 1 on `TOY_FULL` against 11.8-17.4 on `TOY`;
   a best response to Random is no longer near-Nash (4.6 points from seat 0,
   against 1.6). More room for deception, as step 8 predicted.
3. Step 8.5 asked whether 60 % of full UNO's decisions ("draw with a play in
   hand") carry any skill. On the 9-card toy they carry ~1-4 points at L1 and
   are used in a fifth to a third of games. That is not "nearly empty".

### Sizing bigger candidates (scratch measurements, before this step)

Quotient census with a 6M-history cap; "R vs R" is the random-vs-random
seat-0 score on 2000 seeds (the seat-parity readout of step 3).

| candidate | deck | pile | group | deals -> orbits | quotient histories | R vs R seat 0 | actions / game |
|-----------|------|------|-------|-----------------|--------------------|---------------|----------------|
| `TOY` 4c(1,1)+W h3 forced | 9 | 2 | 48 | 5,600 -> 144 | 33,646 | 0.651 | 6.5 |
| `TOY_FULL` | 9 | 2 | 48 | 5,600 -> 144 | 774,007 | 0.651 | 6.5 |
| 5c(1,1)+W h3 forced | 11 | 4 | 240 | 58,800 -> 366 | 2,063,824 | 0.634 | 7.5 |
| 4c(1,1,1)+W h3 forced | 13 | 6 | 144 | 332,640 -> 2,544 | > 6M | 0.568 | 8.3 |
| 4c(1,1,1)+W h4 forced | 13 | 4 | 144 | 554,400 -> 4,184 | > 6M | 0.593 | 9.9 |
| 4c(1,1,1)+W h3 free | 13 | 6 | 144 | 332,640 -> 2,544 | > 6M | 0.568 | 8.3 |
| 4c(1,1)+skip+W h3 forced | 13 | 6 | 48 | 2,667,840 -> 56,628 | > 6M | 0.556 | 8.2 |
| 4c(1,1)+draw2+W h3 forced | 13 | 6 | 48 | 2,667,840 -> 56,628 | > 6M | 0.544 (37 % exhaustion) | 7.2 |
| 4c(1,1,1)+2W h3 forced | 14 | 7 | 144 | 609,840 -> 4,714 | > 6M | 0.548 | 9.0 |

Seat parity follows game length and per-card playability (a third rank,
action cards), not deck size (five colors: 0.634); those candidates all pass
6M quotient histories, true sizes unknown. With keys gone, 20-50M histories
fit in memory (~110 bytes per history in the compiled arrays), so some are
probably in reach; sizing them needs a faster census walk (numba, or a
DFS with undo), which is the next measurement.

### What is next

- Measure what the quotient + cache buy on the step 10-11 loop (a best
  response per checkpoint is now 0.01 s on `TOY`, 0.2 s on `TOY_FULL`),
  then decide on numba kernels / a faster walker / discounted CFR.
- Size the 13-card candidates past the 6M cap and pick `TOY2` (a third rank
  or skips, the two that move seat parity and add the missing decisions).
- The RL agent (step 11) is made symmetric by construction (canonical
  observation in, action mapped back), so every exact cell stays on the
  quotient; a raw asymmetric checkpoint would need `symmetric=False`.


## Build order
1. RULES.md: parametrized ruleset, toy and full parameters
2. core engine + property tests
3. tree-size census on candidate toy sizes, pick the toy
4. random / heuristic agents
5. L3 minimax vs random and vs heuristic
6. best response L1 and L2 (exact, own implementation)
7. mirrored evaluator, validated against exact values
8. CFR for Nash L1 (own implementation on a compiled tree; the pyspiel adapter is not needed, exploitability comes from our own best response)
9. AIVAT
10. skill check on the toy: is there something to learn?
11. RL
