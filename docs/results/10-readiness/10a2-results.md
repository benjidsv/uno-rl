# Step 10a.2: the rest of the contract (2026-09-17)

> On top of commit `cddb19b`. Plan: [plan.md](../../plan.md) row 10a.2, brainstorm:
> [10-brainstorm.md](10-brainstorm.md) section 5 (items A4-A6) and section 8 (LBR).
> Re-run: `uv run uno target TOY`, `uv run uno target TOY_FULL 1000 1000 0`,
> `uv run uno shaping TOY|TOY_FULL|TOY2|TOY2_SKIP`,
> `uv run uno rungs TOY`, `uv run uno colfree`; tests
> `uv run pytest tests/eval/test_target.py tests/eval/test_shaping.py tests/eval/test_rungs.py`.

10a fixed what gets measured about a checkpoint, on which seeds. This finishes the contract:
**A6** the training target and the one item that could refute a design decision, **A4** the
shaping artifacts, **A5** the two missing ladder rows. LBR is **cut** -- see section 6.

The headline: the brainstorm's argument about the row correction is exactly right and is now a
measurement rather than an argument; its hope about the corrections' *cost* is wrong by a
factor of fifty, and the reason is not the network.

## 1. A6: the training target, and the trap measured

`uno.eval.target.TRAINING` = **chance + opponent + deal, never the row**.
`corrected_return(terms, seat)` is the single function step 11 calls, and it *raises* if handed
a `Corrections` with `row=True`: the one choice this module exists to forbid is the one the
type system can forbid. No new estimator -- 9.5b built AIVAT as per-game records so that any
subset is one subtraction, and the target is a subset selection over machinery already closed
against the exact toys.

### 1.1 The exact side, and why the parameterisation is the whole argument

On a compiled tree the exact policy gradient of a tabulated policy is available, and so is the
**exact mean of the sampled estimator** under any subset of corrections
(`uno.compiled.gradient`). So the bias is a number, not a limit.

The gradient with respect to the *probabilities* is the reach-weighted arc value -- CFR's
instantaneous regret with the learner's own reach left in. The score-function estimator
`(1{a taken} - sigma(a)) * G` estimates the **logit** gradient, the centred one. The two differ
by a per-set constant, and a corrected return puts into that constant everything the
corrections collected *before* the decision. That is not a technicality: on `TOY` those
pre-decision corrections reach **4.5e-2** against a gradient norm of **4.7e-2**, i.e. they are
the same size as the thing being estimated, and they cancel in the centred form to **8.7e-18**.

| what | result |
|---|---|
| exact gradient vs a central finite difference of `evaluate` | max relative error **4.8e-11** |
| the logit gradient sums to zero over every set | max \|per-set sum\| 8.7e-18 |
| every subset sparing the row, every value function | max deviation from the exact gradient **1e-17** |
| the row correction with an exact V | `|mean| / |exact|` = **0.000000** |

The last line is the brainstorm's prediction, confirmed with no sampling noise at all: with an
exact V the row term is `Q(s, a) - V(s)`, so the corrected return's conditional expectation
does not depend on the action and the gradient is *identically* zero.

**What is new, and matters for step 11:** with an *approximate* V the row correction does not
destroy everything, and how much it destroys is **not predicted by outcome R²**.

| value function | R² (`TOY`) | gradient left | R² (`TOY_FULL`) | gradient left |
|---|---|---|---|---|
| exact values | 0.746 | **0.000** | 0.516 | **0.000** |
| L1 set values | 0.550 | 0.860 | 0.467 | 0.408 |
| exact + noise 0.1 | 0.736 | 0.085 | 0.500 | 0.140 |

A *coarser* V (R² 0.55) leaves 86 % of the gradient while a *better* one (R² 0.74) leaves 8.5 %.
This is 9.5b's "R² does not carry" one level up, and it means "our critic is only approximate,
so the row correction is probably safe" is not an argument available to step 11.

### 1.2 The sampled side

2,000 dev games per matchup and seat, the corrected signal formed per game, the sampled
gradient compared against the exact one through two pre-declared projections (onto the exact
gradient, and onto one orbit's own component). `uniform_legal` vs `heuristic`, seat 0:

| corrections | projection onto the exact gradient | z | variance ratio |
|---|---|---|---|
| none (raw return) | +0.048214 ± 0.006802 | +0.34 | 1.000 |
| chance | +0.049942 ± 0.006273 | +0.91 | 0.851 |
| opponent | +0.047073 ± 0.006532 | +0.01 | 0.922 |
| deal | +0.047613 ± 0.005288 | +0.22 | 0.604 |
| **TRAINING** | +0.048200 ± 0.004139 | **+0.56** | **0.370** |
| + row (the trap) | -0.000139 ± 0.000822 | -112.42 | 0.015 |

against an exact +0.047026. The kept subset is unbiased and removes **63 %** of the gradient
variance, most of it from the deal term alone; the row correction drives the estimate to zero
to within its own standard error. 0 failed checks on `TOY` and on the free-play `TOY_FULL`.

**A z needs a central limit, and this section nearly shipped without one.** The projection onto
a single orbit is zero in every game that does not visit that set. Picked by largest gradient
alone, the orbit was visited **once in 200 games** and reported a confident z of **-5.86** for a
provably unbiased subset. At 2,000 games it has 23 visits and z -1.18; at 20,000, 232 visits and
z -1.13. There was no bias -- the sample standard error of a statistic that is non-zero in one
game means nothing. The instrument now picks the projection among sets the profile actually
reaches and prints `..` instead of a check below 30 contributing games.

### 1.3 Cost: the brainstorm's hope is wrong, and the reason is the engine

The brainstorm said not to inherit 9.5c's "AIVAT costs more CPU than it saves", because a
training loop already computes `V(s)` and "the siblings the corrections need **batch**". The
siblings' *values* do batch. The siblings themselves have to be **built**, and that is most of
the cost. On `FULL_UNO`, 200 dev deals, 41.5 decisions/game:

| distinct value rows per game | |
|---|---|
| deal only | **1.0** |
| chance only | 783.8 |
| TRAINING (chance + opponent + deal) | 994.6 |
| every correction (+ own actions) | 1015.0 |
| what the step's own `V(s)` calls make | 41.5 |

| timed per component (ms/game) | |
|---|---|
| a training step: `V(s)` at its own states, batched | **0.57** |
| the shipped corrections (`CriticValues.records`) | **41.9** |
| of which: building the sibling states | 22.8 (engine; does not batch) |
| of which: features, row at a time | 12.5 |
| of which: the forward pass | 4.5 |
| the same features, vectorised | 4.1 (3.0x) |

So the corrections cost **73x** the step's own value calls as shipped and **55x** with the
features vectorised, and **73 % of that floor is the engine building 1,015 states per game**,
not the network. Every component is timed on its own rather than obtained by differencing the
total, and the parts are checked against the end-to-end time -- the first version of this
section timed only the shipped path and the forward pass, and attributing the remainder would
have credited it to the network.

**The consequence for step 11, and it is actionable:** the *deal* correction is **one row a
game**, it is nearly free, and it is precisely the one that removes the deal floor (section 2).
Chance and opponent are the expensive pair, and on full UNO they are not affordable per step
without either a cheaper chance model or a value function evaluated on observations rather than
on rebuilt states.

## 2. A4: the shaping potentials, and the identity that makes them one object

With no discount and `Phi(terminal) = 0`, a game's shaping terms telescope to `-Phi(s0)`, which
is what AIVAT's deal correction subtracts. So the control variate and the shaping potential are
**one object**, and the check is an equality, not a resemblance:

> read off played games, the L2 deal potential's per-game term is AIVAT's deal correction
> **array-equal** -- on every ruleset, at both seats.

`Phi(terminal) = 0` is the load-bearing convention and the only place a potential differs from
the value function under it (`potential` zeroes the terminals; `values`, for the exact-critic
arm of D6, keeps the terminal returns; the two are array-equal everywhere else).

**The deal floor by level of information** (`Phi(s0) - E Phi` variance as a share of the raw
outcome variance), heuristic vs random, seat 0 / seat 1:

| ruleset | L1 (own hand) | L2 (both hands) | L3 (+ pile order) |
|---|---|---|---|
| `TOY` | 0.276 / 0.180 | **0.424 / 0.487** | 0.466 / 0.527 |
| `TOY_FULL` | 0.276 / 0.180 | 0.424 / 0.487 | 0.466 / 0.527 |
| `TOY2` | 0.209 / 0.126 | 0.387 / 0.413 | 0.500 / 0.501 |
| `TOY2_SKIP` | 0.229 / 0.219 | 0.461 / 0.423 | 0.626 / 0.565 |

Three things to read off it:

- **the L2 column is 9.5c's deal floor**, and it reproduces the published exact figure on `TOY`
  (0.42 / 0.49) to the digit, plus `aivat_moments`' own "every correction but the deal"
  variance exactly. It was not fitted to either: the potential is built from the profile's
  values and the agreement is the cross-check.
- **the ordering L1 <= L2 <= L3 is a theorem, not an observation.** Each level is a conditional
  expectation of the next, so the tower property orders the variances; a violation would mean a
  potential is not the conditional expectation it claims to be. It is asserted in the tests.
- **`plan.md`'s recommendation costs half the correction.** It argues for an L1 potential
  because "an L2/L3 potential depends on cards the agent cannot see". For a *per-step* shaping
  reward that argument holds. For the **deal** term it does not -- `-Phi(s0)` is subtracted from
  every return and does not depend on the action at all -- and the L1 potential removes 0.28 /
  0.18 where the L2 one removes 0.42 / 0.49 and the L3 one 0.47 / 0.53.

Under **Nash self-play** the L2 floor is 0.67-0.71 and the L3 floor **0.86-0.96**: between
equals, almost the entire outcome is the deal. That is the number to remember when step 11
reports a self-play curve.

`TOY_FULL` reproduces `TOY`'s row to the digit because neither rulebot ever draws voluntarily,
so the two rulesets play out identically under that profile -- the same coincidence
`uno.eval.belief`'s docstring records for the void constraint.

**Emitted** (`.cache/shaping/`): the per-information-set L1 table for the Nash, BR L1 and
rulebot profiles, per seat, round-trip checked. The per-*node* vectors are deliberately not
stored -- on `TOY2_FULL` one is 28M doubles, and it is two passes over a cached tree away.

## 3. A5b: what the belief rungs cost in exploitability

The rungs that ship on full UNO (`voids`, `soft`) are neither of the beliefs the exact toys
measured (`prior`, `posterior`). `uno.compiled.rungs.rung_belief` builds a rung's belief as a
weight per world out of `uno.eval.belief`'s **own** `voids` / `satisfies` / `void_weight`, and
`pimc_response(..., b=...)` takes it, so the rungs join the existing axis rather than forming a
parallel one. It runs on the **full** world tree, because on a quotient a member world's hand
is in that member's own canonical coordinates.

`TOY`, response continuation, exploitability in score points (row at seat 0 / seat 1):

| rung | vs random | exploitability | vs heuristic | exploitability | NashConv (vs heu) |
|---|---|---|---|---|---|
| `prior` | 0.7917 / 0.5440 | 1.36 / 1.35 | 0.7361 / 0.4238 | 1.35 / 1.35 | 2.71 |
| `voids` | 0.7917 / 0.5445 | 1.36 / 1.50 | 0.7361 / 0.4238 | 1.35 / 1.50 | 2.86 |
| `soft(0.1)` | 0.7917 / 0.5445 | 1.36 / 1.35 | 0.7361 / 0.4238 | 1.35 / 1.35 | 2.71 |
| `posterior` | 0.7919 / 0.5491 | 1.61 / 0.83 | 0.7373 / 0.4329 | **3.12 / 7.82** | **10.94** |

- **The shipped rungs are nearly free.** One seat moves from 1.35 to 1.50 points; the other does
  not move at all. They read no policy *probabilities*, and that is what keeps them cheap here.
- **Full opponent-fitting is not free.** Against the Heuristic column `posterior` costs **8.2
  points of NashConv** over `prior` -- the published quotient warning (2.91 / 4.03 against
  0.81 / 0.34), reproduced on the full tree. Step 11 trains against a league; a learner that
  fits it this hard buys score against the league and pays for it in exploitability, and only
  the toy can see that.
- **The rungs move the policy and not the score.** `voids` chooses differently from the prior at
  1,396-3,276 information sets and the score moves by at most 0.05 points. `soft(0.1)` changes
  no choice at all at seat 0. That is 9.5f's "toy games are too short to accumulate the draw
  evidence", measured as a policy difference instead of inferred from a flat score.

The family reproduces its own endpoints -- `soft(0)` *is* `voids` and `soft(1)` *is* the prior,
array-equal as beliefs **and** as policies -- which is the check that the exact rung and the
sampled one are the same object.

## 4. A5a: a column-free PIMC row on full UNO

Both levers behind the ladder's PIMC points read the column's real policy, so that row is an
instrument, not a player. A **column-free** row fixes its assumptions in advance: the
continuation best-responds to a *fixed surrogate* (the Heuristic: "assume a competent
play-first opponent") or assumes a clairvoyant one (`minimax`), and the belief is `voids`,
which reads no policy *probabilities*. `window` is excluded by construction whatever it scores.

**The identity first**, because it is what makes the rest a measurement rather than a new
agent: handed its own column as the surrogate, the column-free row *is* the `response` row --
one policy, so one name, so one world-sampling stream -- and the per-game scores are
**array-equal** at both columns over 200 deals.

**EVAL block 01** (1,004,000-1,007,999), 4,000 deals, both seatings, w = 32, d = 2:

| row | vs random, seat 0 / seat 1 | ms/game | vs heuristic, seat 0 / seat 1 | ms/game |
|---|---|---|---|---|
| **column-free: assume heuristic, `voids`** | **0.7690 ± 0.0067 / 0.7510 ± 0.0068** | 27.9 | **0.6425 ± 0.0076 / 0.6122 ± 0.0077** | 20.1 |
| column-free: `minimax`, `voids` | 0.7285 ± 0.0070 / 0.7310 ± 0.0070 | 49.8 | 0.5787 ± 0.0078 / 0.5853 ± 0.0078 | 35.2 |
| *instrument:* `response(column)`, `voids` | 0.7860 ± 0.0065 / 0.7685 ± 0.0067 | 29.5 | 0.6452 ± 0.0076 / 0.6208 ± 0.0077 | 19.1 |
| the rulebot on the same deals | 0.6212 ± 0.0077 / 0.5845 ± 0.0078 | 0.5 | 0.4155 ± 0.0078 / 0.3787 ± 0.0077 | 0.5 |

Paired on the deal (score points):

| comparison | vs random | vs heuristic |
|---|---|---|
| instrument - (assume heuristic) | +1.70 ± 0.83 (z +2.1) / +1.75 ± 0.83 (z +2.1) | +0.27 ± 0.84 (z +0.3) / +0.85 ± 0.86 (z +1.0) |
| instrument - (`minimax`) | +5.75 ± 0.88 (z +6.6) / +3.75 ± 0.86 (z +4.4) | +6.65 ± 0.91 (z +7.3) / +3.55 ± 0.91 (z +3.9) |
| (assume heuristic) - the rulebot | +14.77 ± 0.95 (z +15.6) / +16.65 ± 0.96 (z +17.4) | +22.70 ± 0.98 (z +23.2) / +23.35 ± 0.98 (z +23.8) |

Four readings:

- **There is a strong general full-UNO player, and this is it.** The column-free row beats the
  Heuristic by **14.8 / 16.7** points against Random and **22.7 / 23.4** against the Heuristic
  itself, at both seats, z > 15 everywhere -- established by this project's both-seats standard
  with room to spare. That is the "honest strong rung" the brainstorm wanted and the first
  full-UNO row that could legitimately be *trained against*.
- **Reading the column is worth much less than the ladder implied.** The instrument's advantage
  over the surrogate row is **+1.70 / +1.75 (z +2.1)** against Random -- *indicated, not
  established* at 3 sigma -- and **+0.27 / +0.85 (z +0.3 / +1.0)** against the Heuristic, which
  is nothing at all. 9.5e measured the `response` continuation as worth +9.73 / +4.57 over
  two-sided minimax and read that as the value of best-responding *to the column*. It is mostly
  the value of best-responding to **something competent**: a fixed surrogate collects almost
  all of it without ever looking at the opponent.
- **The assumption is what matters, and `minimax` is the wrong one.** Assuming a clairvoyant
  opponent costs **3.6-6.7 points** against assuming a competent play-first one, established at
  both seats and both columns (z 3.9-7.3). Same information, different prior about the
  opponent; the exact toys said the same thing (section 25) and this is its full-UNO size.
- **The column-free row is also cheaper than the classic one**: 27.9 against 49.8 ms/game vs
  Random, 20.1 against 35.2 vs the Heuristic, because expectimax expands only the surrogate's
  support where minimax expands everything -- 9.5e's cost finding reproduced on a row that does
  not read its opponent.

**A cross-block check nobody asked for.** The instrument here is the published working-set row,
and block 00 -- where that row was *selected* -- gave 0.7813 / 0.7603 against Random. On the
fresh block 01 it gives **0.7860 ± 0.0065 / 0.7685 ± 0.0067**: consistent at both seats, so the
selection on the spent block did not materially inflate it. This is the first time that row has
been scored on seeds nothing was chosen on.

**Two caveats to carry.** `voids` is not assumption-free -- it assumes a playable drawn card is
played at once (spec section 26.2), which is a fixed property of the row rather than a reading
of the column, but it is an assumption an RL opponent could violate on purpose (that is what
`soft` is for, and what 9.5f's `KeepsDrawn` column measures). And the leaf network was fitted
on rulebot profiles, so it is off-profile here exactly as in 9.5d-9.5f.

## 5. Checks

| what | result |
|---|---|
| `tests/eval/test_target.py` | 9 passed, 1.2 s |
| `tests/eval/test_shaping.py` | 7 passed |
| `tests/eval/test_rungs.py` | 7 passed, 15.6 s |
| `uv run uno target TOY` / `TOY_FULL` | **0 failed checks** each |
| `uv run uno shaping TOY` / `TOY_FULL` / `TOY2` / `TOY2_SKIP` | **0 failed checks** each |
| `uv run uno rungs TOY` (both columns) | **0 failed checks** |
| `uv run uno colfree` (the identity, then the block-01 ladder) | **0 failed checks** |

## 6. LBR is cut

The brainstorm scoped local best response into 10a and said it would be "the first to be cut if
10a grows", and the user kept it on that basis. 10a grew: A6 turned into three measurements
(exact, sampled, cost) because the cost answer refuted the brainstorm's own premise, and A5
turned into two experiments and a new exact-belief module. LBR is therefore **cut from 10a.2
and deferred to step 11**, where it belongs next to the first self-play curve -- it is the only
full-UNO quality measure that is not "score against a fixed column", which matters exactly when
there is a checkpoint to point it at. Nothing else in this step depends on it.

What that leaves unmeasured, stated plainly: on full UNO there is still **no exploitability
number of any kind**, so the only luck-free quality measure step 11 has is on the toys, and the
full-UNO side of a report card remains "score against named columns" plus the ladder position.
