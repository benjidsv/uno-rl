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
