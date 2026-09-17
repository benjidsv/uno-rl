# UNO RL v5: plan

Motivation, design, the information levels, the evaluators and the ladder. The
measured results are in [results/](results/index.md), one file per step, and the
rules in [spec.md](spec.md).

## Build order and status

| step | what | status | results |
|------|------|--------|---------|
| 1 | spec: parametrized ruleset, toy and full parameters | done | [spec](spec.md) |
| 2 | core engine + property tests | done | [spec sections 1-11](spec.md) |
| 3 | tree-size census on candidate toy sizes, pick the toy | done | [03](results/03-census.md) |
| 4 | random / heuristic agents (4.5: fixes) | done | [04](results/04-agents.md) |
| 5-6 | best response L3, L1, L2 (exact, own implementation) | done | [05-06](results/05-06-best-response.md) |
| 7 | mirrored evaluator, validated against exact values | done | [07](results/07-mirror.md) |
| 8 | CFR for Nash L1 on a compiled tree; exploitability from our own best response | done | [08](results/08-cfr.md) |
| 8.5 | transfer check, `TOY` against `FULL_UNO` | done | [08.5](results/08.5-transfer.md) |
| 8.6 | symmetric quotient, compact keys, tree cache; `TOY_FULL` solved | done | [08.6](results/08.6-symmetry.md) |
| 8.7 | `TOY2` family sized and solved, DCFR / PCFR+, validation | done | [08.7](results/08.7-toy2.md) |
| 8.8 | review, restructure, methodology fixes | done | [08.8](results/08.8-review.md), [re-runs](results/08.8-results.md) |
| 8.9 | fixes before AIVAT: deal probabilities with duplicate cards, rng names, cache version | done | [08.9](results/08.9-fixes.md) |
| 9 | review follow-ups: split share, test power, exhaustion scorings, two-sided values, equilibrium choice, fill-in, deal model, pairing on full UNO, ticket draws, speed-ups | done | [09 review](results/09-review.md), [results](results/09-results.md) |
| 9.5a | exact: nested memory keys (K0/K1/K2, `uno.compiled.coarse`), edge-by-decision decomposition and fixed-opponent metric (`uno.compiled.edge`), deal model at 100k seeds, divergence statistic, node lookup (`uno.compiled.lookup`) and the sampled D2 cross-check | done | [09.5 brainstorm](results/09.5-brainstorm.md), [09.5a results](results/09.5a-results.md) |
| 9.5b | AIVAT: exact validation on the toys (exact moments, three scorings, tickets, imperfect-V curve), chance model with reshuffle, fitted L2-state critic, intermediate rulesets `MID_12` / `MID_25`, full UNO, transfer curve | done | [09.5b results](results/09.5b-results.md) |
| 9.5c | PIMC values as a teacher: determinized rollouts validated against the exact toys, the deal floor (AIVAT's residual with exact L2 values), a symmetric value network fitted on rollout targets as AIVAT's V on `MID_12` / `MID_25` / full UNO, deal correction with a Monte Carlo deal mean, cost of rollouts used directly | done | [09.5c results](results/09.5c-results.md) |
| 9.5d | PIMC agent: exact PIMC-∞ on the toy world trees against BR L1 / L2 (with and without opponent-reach weighting), PIMC-n, the full-UNO PIMC **search** row evaluated with the 9.5c value network (the rollout player turned out weaker at equal worlds and is reported as a comparator); full-UNO D2 with a search Q; the worlds-versus-depth budget study (MCTS deferred) | done | [09.5d](results/09.5d-results.md) |
| 9.5e | PIMC's approximations: the `response` continuation (each determinized world solved as a best response to the known column, +10.2 / +5.2 points on full UNO and *cheaper*), per-seat leaf networks (9.5d used the seat-0 one at both seats), AIVAT on the new row; inference deferred | done | [09.5e](results/09.5e-results.md) |
| 9.5f | **PIMC inference**, which **refuted this plan's particle-filter design**: sampling the opponent's hand at the deal and weighting it by their actions collapses (survival 0.156 -> 0 over their first five plays, since a particle lives only if it holds the played card, ~7/98), and the built version scored worse than the uniform prior. Replaced by rungs that *condition* instead: `voids` (a draw proves the drawer held nothing playable -- no policy *probabilities* needed, though it does assume a playable drawn card is played at once) and `window` (plus their real probabilities over the trailing run of plays). Worth **+2.83 / +4.40** points against Random at both seats, where the toys predicted 0.0-0.6. Closed against the exact toy posterior afterwards, which found `voids` to be a **policy assumption** with no sound alternative on free play, and turned the rungs into a small **league**: `voids` against honest columns (88-91 % of the distance to the posterior closed), `soft` against one that keeps a playable drawn card (52-67 %, where `voids` manages 16-19 %) | done | [09.5f](results/09.5f-results.md) |
| 10a | **the contract**: the seed registry (`uno.eval.seeds`, which records that `EVAL_SEEDS` was *spent* on 9.5d-9.5f selection), the batched tabulation (`batch_l1_observations` + `tabulate_batched`: a network over `TOY2_FULL`'s 3.75M sets goes ~200 s -> ~21 s, array-equal to the row-at-a-time path) and the report card (`uv run uno report`, reproducing the published values to the digit). The measurement changed the middle item: `observation_from_key` is 57-62 % of `tabulate` and batching only the forward pass would have left it | done | [10a](results/10a-results.md) |
| 10a.2 | the rest of the contract. **A6 the training target**: chance + opponent + deal, never the row -- and the exclusion is now measured, since the exact policy gradient and the exact mean of the sampled estimator are both computable on a tree (with an exact V the row correction drives the gradient to *identically* zero; `TRAINING` is unbiased at 0.37 of the raw gradient variance). Its **cost** refuted the brainstorm's own premise: the siblings' values batch but the sibling *states* must be built, so the corrections are 55x the step's own `V(s)` calls and 73 % of that is the engine, while the deal correction alone is one row a game. **A4 shaping**: the potentials emitted per level, `Phi(terminal) = 0`, and the L2 deal potential is AIVAT's deal correction *array-equal* per game; the deal floor is 0.42-0.49 at L2 and 0.86-0.96 under Nash self-play. **A5b**: the shipped belief rungs cost almost no exploitability where full opponent-fitting costs 8.2 points of NashConv. **A5a**: a column-free full-UNO row (fixed surrogate continuation + `voids`) that reproduces the `response` row *game for game* when handed its own column, and on fresh EVAL block 01 beats the rulebot by 14.8 / 16.7 points against Random and 22.7 / 23.4 against the Heuristic at z > 15 -- the first full-UNO rung that could be trained against. It also repriced 9.5e's headline lever: reading the column is worth only +1.70 / +1.75 (z +2.1) against Random and nothing against the Heuristic, so the `response` continuation's value is mostly "assume a competent opponent", not "assume *this* one". **LBR cut** and deferred to step 11 | done | [10a.2](results/10a2-results.md) |
| 10a.3 | repair coarse-key memory evidence; declare observation arms; build and calibrate LBR on toys and supported full-UNO baselines | planned next | [execution plan](10-to-11-plan.md), [review](results/10-review.md) |
| 10b | figures and generated tables, using corrected memory evidence and explicit benchmark/lower-bound labels | planned | [execution plan](10-to-11-plan.md) |
| 10c | the distilled rulebot, transfer test, and frozen league; distinguish toy transfer from full-UNO DEV tuning | planned | [execution plan](10-to-11-plan.md) |
| 11 | RL: exact toy validation, sampled full-UNO training, then iterations guided by measured gaps | planned | [execution plan](10-to-11-plan.md) |

Current headline numbers per toy: [results/index.md](results/index.md).

**2026-09-17 sequencing update:** insert 10a.3 before 10B, then keep 10B → 10C → 11.
This adds preparation work without replacing those milestones. It does not promise
unchanged elapsed time. Full-UNO learner memory comparisons and recurrent-target
attack integration belong to Step 11; a redesigned particle filter is not a
prerequisite. [The execution plan](10-to-11-plan.md) supersedes the earlier LBR
deferral and narrows the memory/ceiling claims identified in the review.

## Why this iteration

This is my fifth iteration on experimenting with UNO in RL. The main problem: skill vs luck. UNO variance is enormous and the reward is sparse: during training it's hard to find good plays and take advantage of them. Most of the outcome is luck of the draw.

Previous wins (UNO 1v1, raw winrate vs random):
- rulebots: 60%
- genetic parameterized rulebot: 70%

Failures:
- no meaningful improvement from PPO agents, even with self-play etc.

Suspected causes (to be tested, not assumed):
- the evaluator was too noisy to see small improvements (a 3-point edge near 60 % is one sigma at ~270 games and three sigma at ~2400, where a 3-sigma test still misses a real 3-point edge half the time, ~3,900 for 80 % power; we never ran a fixed budget)
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
- **Rules spec written before code** ([the spec](spec.md)), with property tests: card conservation, every legal action applies without error, every applied action was legal, determinism, clone equivalence, game always terminates.

Which decisions carry skill in UNO (the toy must keep them, otherwise there is nothing to measure):
- which color to play down (color management: keep the color you hold most)
- holding a wild vs spending it
- wild color choice based on inferred opponent voids (they drew on red => probably no red)
- draw vs play when both are legal
- timing of action cards


## Toy UNO ("Kuhn UNO"): defining and sizing it

The size decides whether exact solving is feasible: exact L1 solving enumerates deck permutations consistent with what the player saw, and permutations explode fast.

Candidate: 2 colors x 3 ranks x 2 copies + 1 wild, hand size 2, no reshuffle. (to be checked)
Resolved by the [step 3 census](results/03-census.md): the candidate is > 4e7 histories; the toy became `TOY`, 4 colors x 2 ranks x 1 copy + 1 wild, hand 3, forced play (1.25M histories). Every toy since then is in [spec section 12](spec.md#12-named-rulesets).

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
- **One-sided L2 and L3 "optimal" only exist as a best response to a fixed, known opponent policy** (step 9: the *two-sided* games, where both players see both hands or everything, do have minimax values, `two_sided`). Once the opponent's policy is known and their hand is visible there is nothing hidden:
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

Status of that last clause: PIMC was built in [9.5d](results/09.5d-results.md) and improved in [9.5e](results/09.5e-results.md), both with a **uniform** determinization, i.e. no inference. Inference itself was built in [9.5f](results/09.5f-results.md) -- but **not as a particle filter**, because that design does not work. Sampling the opponent's hand at the deal and weighting it by their observed actions collapses: a particle survives an observed play only if it happens to hold the played card (`hand_size / unseen` ~ 7/98 on full UNO), so 64 particles decay 0.156 -> 0.018 -> 0.002 -> 0.0004 -> 0 over the opponent's first five plays, and no particle count fixes a geometric decay. The version built to that spec scored *worse than the uniform prior it replaced*. What works is to **condition rather than weight** -- sample the hand at the current split, where consistency is automatic, and use the history only to constrain it (`voids`: a draw proves the drawer held nothing playable) or to score it (`window`: the opponent's real probabilities over their trailing no-draw run of plays). Worth +2.83 / +4.40 points against Random at both seats, where the toys predicted 0.0-0.6. See [spec.md](spec.md) section 26.

Information sets and recall. The L1 information state is (seat, initial hand,
full public history, own draws in order), [spec section 10](spec.md). It has perfect
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

The most useful cell for an RL agent: **BR L1 to it**. Tabulate the network's policy over all toy information sets and compute an exact best response. Exploitability proper needs the Nash value: per seat, exploitability = (BR L1 value against the agent) - (Nash L1 value at that seat), and NashConv is the sum over the two seats. Since [step 8](results/08-cfr.md) the Nash value is known, so the BR L1 column gives absolute exploitability; either way it is independent of luck.


## Training side (only after the ladder is filled)

- If BR L1 vs random on the toy is close to the old rulebot numbers, PPO had nothing left to learn and the fix is the game, not the algorithm.
- Sparse reward fix: potential-based reward shaping. Policy-invariant for any potential with Φ(terminal) = 0, but the potential should be a function of the **agent's own information set** (an L1 value: BR L1 or the Nash L1 value). An L2/L3 potential depends on cards the agent cannot see, so the shaping term adds noise the agent cannot condition on and the critic still has to average it out. On full UNO use the PIMC value or a critic, again at L1.
- Self-play evaluation: exploitability via best response on the toy, not just winrate vs random.
