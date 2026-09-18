# Step 10 brainstorm: formalising the results for RL (2026-09-17)

> On top of commit `29bccad`. **Nothing built**: this is the brainstorm before step 10, per the
> workflow in `context.md` section 10. Numbers quoted are from [index.md](../index.md) and the
> 9.5a-9.5f results files. Plan: [plan.md](../../history/original-plan.md) row 10.
> Planned as **three commits**: 10a (the contract), 10b (figures and generated tables),
> 10c (the distilled rulebot and the transfer test).

## 0. What this step is for

Step 11 will produce checkpoints. Step 10 decides, **before any of them exist**, what gets measured
about one, on which seeds, at what cost, and what counts as better. The project's whole thesis is
that iterations 1-4 failed because that was never fixed in advance; the failure mode to avoid here
is a step that draws pictures of numbers we already have while the things step 11 actually trips on
go unbuilt.

So: 10a is the contract and the plumbing under it, and it comes first because it is the part that
can fail. 10b is the presentation. 10c is the one measurement in this step that is a genuine
experiment rather than an assembly job.

## 1. What the charter asked, and what 9.5 already delivered

The step 9 review ([09-review](../09-evaluation/09-review.md) section D) chartered two artifacts: **a performance
range with several ceilings**, and **an edge-by-decision tool**. Since then 9.5a-9.5f built most of
the machinery for the second and answered two of its supporting items outright:

| section D item | status today |
|---|---|
| D1 range with ceilings | not built; it is this step |
| D2 edge by decision | **built** (`uno.compiled.edge`: mirror + forward terms, `stakes`, `wild_split`, sampled cross-check; `uno.eval.decisions` labels; full-UNO version with a search Q in `pimc d2`) |
| D3 memory | **answered**: K1 (hand, top, discard multiset -- no play-by-play) recovers 96-99.8 % of BR L1's edge on the forced toys |
| D4 fixed-opponent metric | **built** (`br_gap`), not yet wired into anything a checkpoint passes through |
| D5 shaping | designed, nothing emitted; the potentials themselves do not exist as artifacts |
| D6 ablation | step 11 |
| D7 mechanisms | skipped, still skipped |

What is missing is not instruments. It is **the thing that takes a checkpoint and runs all of them**,
and the declarations around it.

## 2. The framing question: what the toys are for in step 11

Asked and answered in discussion (Benji): the toys exist to validate the sampled methods, which is
also the step 9 review's own framing -- *"the toys exist to validate the sampled methods of full UNO;
transfer of toy conclusions is secondary, and the toys are not tuned for realism."* They did that
job: AIVAT, the value network, PIMC and the belief rungs were all closed against exact toy values
before being trusted, and twice the toy check caught something the score alone would have read as
progress (9.5e's vacuous continuation test, 9.5f's collapsing particle filter).

**So RL is judged on full UNO. The toy is the learner's test harness.** Concretely, in step 11 the
toy is the only place where:

- **exploitability is exact** -- the single luck-free quality measure, and the only way to tell
  "learned to exploit Random" from "learned to play well". On full UNO there is no exact analogue
  (see section 8 on LBR);
- **the shaping test (D5) is decidable** -- the L1 / L2 / L3 potentials exist exactly there and
  nowhere else;
- **the ablation (D6) is cheap** -- tabular learner vs PPO with the exact L1 critic vs plain PPO,
  scored exactly per checkpoint, which separates exploration from variance from representation;
- **the plumbing is debugged** -- symmetrisation, tabulation, the report card itself -- before
  full-UNO compute is spent on it;
- **overfitting to the training league is detectable.** Settled in discussion: step 11 trains mainly
  against a **league of rulebots**, with self-play as a secondary experiment. The characteristic
  failure of that design is a policy that beats its league and nothing else, and score against the
  league cannot see it -- exact exploitability can, and only on the toy.

And the standing limit, in the same voice as 9.5d's "toy belief sizes do not transfer": a toy game
is **6.2 decisions** against full UNO's **44.5**. Toy exploitability is a property of the learner on
the toy. It is evidence the learner works, not a number that carries.

This corrects the recommendation in the first draft of this brainstorm (section 9).

## 3. The range, redefined: PIMC is the full-UNO ceiling proxy

The range was chartered as Random / Heuristic / tuned Heuristic / Nash / BR L1 / BR L2 / BR L3, with
the two-sided values off the range (on `TOY` they sit *below* the Nash L1 value, so they are not
ceilings for an L1 agent). That is the toy range and it is already computed -- 10b just draws it.

The full-UNO range has no exact rungs at all, and this is where 9.5's work pays: **PIMC with the
`response` continuation is a measured proxy for BR L1**. On the toys, exact PIMC-infinity under
`response` lands within **0.04-0.70 score points of BR L1** on every toy, seat and opponent, where
classic two-sided `minimax` loses 0.15-3.21 on the forced toys and **8.4-8.8 on `TOY_FULL`**. That
is the licence: the instrument was closed against the exact value on the toys, which is exactly what
the toys are for.

Proposed full-UNO range against a named column, per seat:

| rung | vs Random, seat 0 / seat 1 | what it is |
|---|---|---|
| `PlayFirstRandom` | 0.5 by construction | the floor |
| `Heuristic` | 0.616 / 0.604 | the current rulebot |
| **distilled rulebot** | ? | commit 10c; the historical ~70 % bot, reconstructed |
| PIMC `prior` w=32 d=2 response | 0.7530 / 0.7163 | search, no inference |
| PIMC `voids` w=32 d=2 response | **0.7813 / 0.7603** | the working set |
| *(proxy ceiling)* | the `response` row itself | BR-like: it reads the column's policy |

Two honest caveats to print with it every time:

- **The proxy is a lower bound of a ceiling, and its error grows with game length.** Strategy fusion
  and non-locality are what PIMC pays, the toys bound that payment at 0.04-0.70 points over **6**
  decisions, and full UNO has 44. The toy number bounds the *instrument*, not full UNO's bias.
- **`response` reads the column's policy**, so it is a ladder instrument, not a player. The same is
  true of `window`. A rung that a step-11 agent could actually be *trained against* must be
  column-free, and no such PIMC has ever been measured (section 4d).

On top of the raw ranges I would add one derived scale so checkpoints are comparable across panels:
**normalised skill, 0 = Random at that seat against that column, 1 = the ceiling rung of that
panel** (BR L1 on the toys, PIMC `response` on full UNO). The Heuristic then lands at ~0.4 from seat
0 and ~0.2 from seat 1 across every toy, which is the project's most robust qualitative finding, and
a checkpoint is one dot. It is an affine rescaling per cell, so a difference in scale units is
**not** score points: print both, always, and never *select* on the normalised number.

For a self-play row the range is the wrong instrument entirely -- report exploitability / NashConv
instead. That is D4's split, and it is why the report card below carries two numbers rather than one.

## 4. Five holes that would break step 11

**(a) The full-UNO eval seeds are spent, and they were spent on selection.** 9.5f states it outright
for `window` at seat 1. The deeper problem: the working set itself -- `response`, w = 32, `voids` --
was *chosen* by 3,000-4,000-deal runs on `EVAL_SEEDS` across 9.5d-9.5f. By this project's own rule
("nothing is ever chosen on them") that range is now a dev range, and step 11 has no held-out
full-UNO seeds. Seeds were never the scarce resource; the declaration was.

**(b) Tabulating a *network* over the L1 keying does not exist.** Every exact cell for a checkpoint
needs the policy tabulated and orbit-uniform, or `tabulate` refuses. Today that is a Python call per
set: 0.16 s on `TOY`, 1.9 s on `TOY_FULL`, **~50 s on `TOY2_FULL`'s 3.75M sets**. `context.md` 6.9's
"the exact side of a checkpoint is milliseconds to seconds" is a statement about *tabular* rows; for
a network it is false as the code stands.

**(c) The observation and action encoding for an L1 learner is undecided.** 9.5c's value network is
colour-symmetric by construction (per-colour blocks, DeepSets pooling), which is why `Symmetrized`
costs it nothing. A *policy* network needs the stronger property -- equivariance over actions, not
invariance of a scalar -- and if it has it, symmetrisation is free and the |G|-fold data-efficiency
gain is automatic. If it does not, every checkpoint pays `Symmetrized`'s orbit averaging at every
forward pass, or the exact cells need the full tree instead of the quotient. This is a step-10
decision with a step-11 price tag, written out in section 5.1.

**(d) There is no general full-UNO opponent.** Both levers carrying the PIMC row's points read the
column (`response`, `window`), and 9.5f's own open list records that a column-free PIMC has never
been measured. Related and also open from that list: **the exploitability of the belief rungs is
unmeasured**, and the toys warn that fitting a belief to one opponent *costs* exploitability
(`TOY2_SKIP`: `posterior/response` 3.52 / 6.43 against `prior/response`'s 1.78 / 2.29). Both are
"fill in the ladder's own rows before putting an RL row next to them".

Cost note that matters for step 11: post-optimisation the search is **17.7-21.0 ms/game** (`prior` /
`voids`, 400 EVAL deals against the Heuristic). That is fine as an *evaluation* opponent and
impossible as a *training* one -- 10^6 games of self-play against it is ~5.5 hours of opponent alone.
Which is one of the arguments for 10c.

**Repriced by the league answer.** Since the training opponents are rulebots whose policies we own,
the ladder may legitimately use `response` and `window` against them -- a best response is *supposed*
to read its row. So the column-free PIMC drops from **blocker to nice-to-have**: it is wanted as an
honest "strong general player" rung on the range, not as something step 11 waits on. The
exploitability of the belief rungs, by contrast, goes **up** in priority: the league *is* the 9.5f
league, and its known exploit is the one an RL agent can find (below).

**(e) The shaping potentials are a plan, not a file.** D5's design is settled, including the neat
part -- with no discount and Phi(terminal) = 0, an L3 deal potential contributes exactly -Phi(s0) to
every return, which *is* AIVAT's deal correction: the control variate and the shaping term are one
object. Step 10 can emit the arrays and let step 11 plug them in.

## 5. Commit 10a: the contract

| item | what | check |
|---|---|---|
| **A1 seed registry** | widen the ranges (DEV `[0, 10^6)`, EVAL `[10^6, 2*10^6)`), partition EVAL into disjoint blocks **claimed per step in a registry**, and record 1,000,000-1,003,999 as spent on 9.5d-9.5f selection | an assertion that no two claimed blocks overlap; a test that a block cannot be claimed twice |
| **A2 batched tabulation** | `observation_from_key` over all sets -> batched features -> one forward pass -> orbit-average, for a network policy | **array-equal** to the Python-loop tabulation on `TOY` and `TOY2`, and the resulting exact cells identical to the digit |
| **A3 the report card** | one command, one checkpoint, fixed format (below) | run it on `Heuristic` and `PlayFirstRandom` and require the pinned values (`TOY` Heuristic exploitability 5.6 / 11.8, etc.) -- the 9.5f "reproduces to the digit" device |
| **A4 shaping artifacts** | per ruleset: Phi over L1 sets from Nash and from BR L1; the L2 / L3 variants for D6; exact per-state values under a profile for the exact-critic arm | Phi(terminal) = 0; the L3 deal potential equals AIVAT's deal correction term on the same seeds |
| **A5 the two missing ladder rows** | column-free PIMC on full UNO; exploitability of `prior` / `voids` / `soft(0.1)` on the toys | the column-free row must reproduce the `response` row when handed its own column; exploitability run on the existing exact trees, 0 failed checks |
| **A6 the training target** | the luck-corrected return as a **declared subset** of 9.5b's AIVAT records -- chance + opponent + deal, never the row -- emitted as the one function step 11 calls | the exact-gradient check of section 5.2: zero bias for the kept subset, and the row correction predicted to drive the sampled gradient to **zero** |

**The report card**, sketched so the format is argued before it is built:

```
checkpoint: ppo-e40            ruleset: TOY2_FULL      seeds: EVAL block 03
exact
  vs random      0.7012 / 0.5433   normalised 0.72 / 0.61   (BR L1 0.7479 / 0.6164)
  vs heuristic   0.6604 / 0.4981   normalised 0.65 / 0.58
  br_gap vs its training column               2.81 / 4.22 pts
  exploitability                              5.90 / 6.71 pts   NashConv 12.61
  edge by decision (mirror / forward agree)   wild colour 41 %, draw 27 %, colour 19 %, other 13 %
sampled (FULL_UNO, 1,800 deals, AIVAT + shared-stream pairing)
  vs random      0.6820 +- 0.0104 / 0.6655 +- 0.0107
  vs heuristic   0.5610 +- 0.0101 / 0.5488 +- 0.0104
  vs PIMC voids  0.3910 +- 0.0099 / 0.3744 +- 0.0102
  paired vs previous checkpoint               +2.10 (z +2.4) / +1.85 (z +2.1)
```

Two numbers carry it: **position on the range against the training column**, and **exploitability**.
Everything else is diagnosis.

Pre-declared budgets, from arithmetic the project already has (3 points, 3 sigma, 80 % power):

| claim | estimator | deals |
|---|---|---|
| checkpoint vs previous checkpoint, full UNO | shared-stream pairing | 1,250-1,800 |
| checkpoint vs a fixed column, full UNO | AIVAT, network V | ~1,780 (3,860 raw) |
| checkpoint vs a fixed column, full UNO | AIVAT + deal correction | ~1,480 |
| anything on a toy | exact | 0 |

### 5.1 The encoding decision

**The group is bigger than the toys suggest.** `symmetry.py` measures |G| = 48 on `TOY` (4! colour
permutations x 2! interchangeable ranks), 36 on `TOY2` (3! x 3!), 12 on `TOY2_SKIP`. The same
construction on `FULL_UNO` -- never run, this is arithmetic and not a repo number -- gives
**4! x 9! = 8,709,120**: the four colours, and the nine 2-copy number ranks, which are
interchangeable because rank identity enters the rules only through equality (0 has one copy and the
action cards have distinct effects, so both are fixed).

That number decides the trade-off, because it is far too large to canonicalise by enumeration or to
collect by augmentation.

**Option A, equivariant by construction** (per-colour weight-shared blocks with a second set-level
pooling over the interchangeable ranks; colour-free actions from the global block; the wild's
(card, colour) joint actions indexed from the colour block).

- Symmetrisation is **free at inference**: no canonicalisation, no orbit averaging, on every
  training step and on every one of the 129k-3.75M sets a report card tabulates.
- The data-efficiency factor is **obtained in full and automatically**, including full UNO's ~8.7M
  relabelings, which no augmentation scheme can enumerate. Every game teaches all of them at once.
- `tabulate` accepts it directly, so the exact cells run on the **quotient** tree (37-46x smaller on
  `TOY`) rather than the full one.
- It reuses a proven component -- 9.5c's value network is already colour-symmetric and reaches
  R^2 0.83-0.84 against the true values -- and a shared encoder for policy and value is what PPO
  wants anyway.
- It makes the section 5.8 bug class **impossible**: with no canonicalising element there is no way
  for a policy to condition on which element `canonical` happened to return.
- Against it: **equivariance is strictly harder than invariance and fails silently.** The value net
  only had to produce an invariant scalar; a policy must permute its logits the same way the
  relabeling permutes the actions. A subtly wrong head trains fine and is quietly asymmetric.
- Against it: the rank level constrains what the net can express about *specific* ranks. That is
  correct here (7 is not special), but it is an assumption the architecture bakes in, and if a
  ruleset ever broke it there would be no diagnostic saying so.
- Against it: real work before the first learning curve.

**Option B, plain encoder wrapped in `Symmetrized`.**

- Simplest path to a first curve; `Symmetrized` exists, is tested, and its correctness argument is
  already settled (the A3 correction: canonical observation *and* orbit averaging).
- Architecture-agnostic; the net can change without re-deriving anything.
- It yields one diagnostic option A cannot: the distance between the raw net and its symmetrised
  image **measures how much capacity and data the learner spent on relabelings**.
- Against it: the cost is paid per forward pass, which is hole (b) all over again -- the report
  card's "milliseconds to seconds" depends on not paying it 3.75M times.
- Against it: **it fixes the output, not the learning.** The net still learns red and blue
  separately. Recovering the factor needs augmentation, and on full UNO 8.7M images can only be
  sampled, at the cost of the extra forwards.
- Against it, and this is the sharp one: unless `Symmetrized` is also in the *training* loop,
  **the policy that was measured is not the policy that was trained**. Putting it in the loop means
  paying its cost there.
- A catch that rules out the obvious half-measure: the toys' G includes the **rank** permutations,
  so a colour-only equivariant net is *not* orbit-uniform and `tabulate` refuses it. Colour-only is
  a valid milestone only if the toy cells go through `Symmetrized` meanwhile, with "measured is not
  trained" recorded while that lasts.

**Recommendation: A, with B kept as its test oracle.** A correctly equivariant network must be a
fixed point of `Symmetrized` to numerical tolerance -- a free, strong check that turns option B's
machinery into option A's regression test rather than a fork in the road. Build colour and rank
levels together, since on the toys both are needed for `tabulate` and on full UNO the rank level is
where 362,880 of the 8.7M lives.

### 5.2 The training target: which AIVAT corrections belong in it

Asked in discussion: should the training objective use AIVAT? **Yes, but only three of its four
correction sets, and the one it must drop is the one AIVAT was built for.**

The split is exogenous against endogenous. AIVAT corrects at every stochastic node whose
distribution is known, and those nodes are not alike from a learner's point of view:

| corrections | in the training target? | why |
|---|---|---|
| **chance** (draws, reshuffles) | **yes** | pure luck the agent cannot influence; removing it is free variance reduction on the return |
| **opponent** | **yes** | the league's policies are ours exactly, and in self-play the opponent *is* us, so the correction is available at no modelling cost |
| **deal** | **yes** | it *is* D5's shaping term: with no discount and Phi(terminal) = 0, a deal potential contributes exactly -Phi(s0) to every return, so the control variate and the potential are one object. It depends on hidden cards but **not on the agent's action**, which is what keeps it unbiased -- the centralised / history+state critic the step 9 review already flagged as standard (Baisero & Amato) |
| **row** (the agent's own decisions) | **no, and this is the trap** | with an exact V the row term *is* Q(s, a) - V(s), the advantage. Subtract it from the return and the corrected return's conditional expectation no longer depends on the action, so `E[grad log pi * corrected return] = 0`: the credit-assignment signal is exactly what got cancelled |

**The distinction to hold on to: an evaluator needs an unbiased mean, an objective needs an unbiased
gradient.** Those are different requirements, which is why the row correction is right for
AIVAT-as-evaluator -- it is most of the variance it removes -- and fatal here. (For the deterministic
search rows of 9.5d-9.5e the row correction is identically zero, so nothing in those results warns
about this.)

**No new estimator is needed.** 9.5b built the estimator as *records* precisely so that "per-game
sums give any subset of corrections (chance, row, column, deal) by subtraction". The training target
is a subset selection over machinery that already exists and is already validated.

**The check that makes this a toy question.** On the compiled tree the **exact policy gradient** of a
tabulated policy is available -- counterfactual reach x Q per information set, the same quantity CFR
already computes. So: play N games with the learner, form the corrected signal, compute the sampled
gradient, and compare it against the exact one. Bias must be zero for {chance, opponent, deal}; the
variance ratio is the number that says whether the corrections are worth their cost; and the
prediction for the row correction is that adding it drives the sampled gradient **to zero**, which
turns the argument above into a measurement. This is the project's own method -- close the sampled
thing against the exact toy value before trusting it -- applied to the training objective rather
than to an evaluator.

**Cost is open and must be measured, not assumed.** 9.5c's verdict that "AIVAT costs more CPU than
it saves" was measured for *evaluation* with cheap rulebots and one leaf call per state: the
corrections were 30-90 ms against a 2.6 ms rulebot pair, so extra deals beat cheaper deals. A
training loop changes both sides of that arithmetic -- PPO already computes V(s) at every state, and
the siblings the corrections need (every drawn card, every action of the support) **batch**, which
5.21 measured at 78.5 us a row at batch 1 against 7.0 us at batch 32-256. Whether the corrections
are then a few per cent or a multiple of the step cost is unknown until it is run. Pre-declare the
measurement; do not inherit 9.5c's conclusion into a different regime.

## 6. Commit 10b: figures and generated tables

A plain script in the `uv run uno <experiment>` convention (`uv run uno figures`) writing SVG into
`docs/generated/report-card/figures/`, so a figure is reproducible like every other number; matplotlib as a new
dependency. A notebook is welcome on top for exploration but is not the artifact -- the script is,
because it can be rerun by `scripts/rerun_all.sh` and diffed.

Five figures, each of which has to show a mechanism rather than decorate a table:

1. **The range**, per ruleset and seat, rungs as ticks, normalised scale on the second axis. The
   headline artifact of the step.
2. **Values of information** (L1 -> L2 -> L3 across the toys): hidden information grows with the toy,
   2.5-4.4 points on `TOY` to 7-14 on `TOY2_FULL`.
3. **The budget picture**: deals for a 3-point claim by estimator, log scale, toys and full UNO
   side by side. This is the project's thesis in one image.
4. **Edge by decision**, stacked bars: the Heuristic against BR L1 on the toys, against the search Q
   on full UNO. Doubles as the report card's template.
5. **The full-UNO lever ladder**: where 9.5's points came from -- continuation +9.73 / +4.57, belief
   +2.83 / +4.40, worlds 8 -> 32 +5.57 / +3.33, depth nothing.

Plus: **generate `index.md`'s tables rather than hand-maintaining them.** Staleness has bitten
before (review item B6) and step 11 will produce rows faster than hand-editing survives. The
regression that licenses the switch is the 9.5f device: the generated tables must reproduce the
current hand-written ones **to the digit** before replacing them.

## 7. Commit 10c: the distilled rulebot and the transfer test

**Why it is worth a commit** (this is the one item in step 10 that measures something new):

- The historical baseline was a **70 %+ genetic rulebot with memory and inference**; ours is a ~60 %
  one-parameter bot. If step 11 beats *that*, the comparison with iterations 1-4 is not like for
  like. Benji's note in discussion: those bots' inference was **similar to `voids` / `soft`** -- so
  the project has independently rediscovered, as its second-largest full-UNO lever, the thing the
  old rulebots already did.
- The distillation target is already identified and measured, not guessed: D2 puts ~97 % of the
  Heuristic's disagreement with the search in **action-card timing (68-73 %)** and **wild handling
  (25-29 %)**, and 9.5a puts the Heuristic's `TOY` seat-1 weakness at **71 % wild-colour decisions**.
- A rulebot is **microseconds** where PIMC is ~20 ms/game, so it is the only strong opponent RL can
  actually *train* against.
- It is the cheapest test of the project's central premise -- "toy skill is UNO skill" -- and the
  only one that needs no exact full-UNO value.

**Design.** A parameterised, symmetric (orbit-uniform, so the exact toy cells still work) rulebot
with a void memory: colour preference weights, wild hold/spend threshold, wild colour from tracked
voids, action-card timing against opponent hand size, draw-vs-play on free play. Tune **on the toys
by exact evaluation on DEV** (grid, then genetic if the grid is not enough), then measure on full
UNO with shared-stream pairing on a **fresh EVAL block**.

**The league.** With step 11 training against rulebots, 10c is not producing one baseline but the
**training distribution**, and its diversity has to be measured rather than assumed. 9.5f already
established the shape: `voids` assumes a playable drawn card is played at once, `KeepsDrawn` breaks
that assumption, and against it the rungs swap places (`voids` closes 16-19 % of the distance to the
exact posterior where `soft(0.1)` closes 52-67 %; 0.1 is the maximin rung over the pair). So the
league needs at least one member that **keeps playable drawn cards**, or the agent will learn the
one line every member's belief rules out, and the exploit will only show up as exploitability on the
toy. The rungs are a league with a known exploit and a known answer to it; use both.

**The two claims must be reported separately**, because only the second is interesting:

1. *the tuned bot is stronger* -- measured directly on full UNO, paired, pre-declared as "kept only
   if the paired difference is >= 3 points at **both** seats at 3 sigma with 80 % power";
2. *toy tuning predicted the full-UNO gain* -- the transfer statistic: toy-predicted gain against
   full-UNO measured gain, per rule. This is the number the step exists for, and it is allowed to
   come out at zero.

**Risks, named in advance.** Tuning on exact toy values can fit toy artefacts -- pile parity,
exhaustion scoring, forced play. Mitigation: require a rule to help on **both** `TOY2_FULL` (free
play) and `TOY2_SKIP` (action cards) before it is carried to full UNO, and report any rule that
helps on a toy and does nothing on full UNO rather than dropping it, since that is the transfer
result. Second risk: a tuned rulebot is a moving baseline -- pin its parameters and give it its own
agent name the moment it is fixed, or every cell that used the old one silently changes meaning
(`context.md` 5.5: a name must identify a policy).

## 8. Deferred, and the one item I would cut first

- **Local best response (LBR)** as a cheap full-UNO exploitability lower bound (Lisy & Bowling 2017)
  was listed for step 11 in the 9.5 brainstorm. Most of its pieces now exist (`determinize_l1`, the
  belief rungs, the value network). It is the *only* candidate quality measure on full UNO that is
  not "score against a fixed column", which matters a lot if step 11 does self-play -- iterations
  1-4's diagnosis was precisely "no ceiling known". I would scope it into 10a **and cut it first**
  if 10a grows; it is the one item here that could quietly become its own step.
- **MCTS / IS-MCTS**: still deferred, unchanged since 9.5d.
- **D7 mechanisms**: still skipped.
- **Genetic tuning of the rulebot beyond what a grid gives**: only if the grid plateaus below ~70 %.

## 9. Corrections to this brainstorm's own first draft

Kept per `context.md` section 7's habit.

| as first proposed (in discussion) | correction |
|---|---|
| `TOY2_FULL` should be the headline RL toy, since it is the only one where exploiting Random and being unexploitable separate (3.1 / 2.6 points) | the separation is real but it is not what the toys are *for*: they validate the sampled methods, and that job is done. RL is judged on full UNO; the toy is the learner's test harness and the only place exploitability is exact (section 2) |
| a stronger rulebot rung is needed so step 11's claim is comparable with the historical 70 % bot | still true, and the rung already half exists: PIMC at 0.7813 / 0.7603 is the strong row, and the historical bots' "memory + inference" is this repo's `voids` / `soft`. 10c reconstructs the *cheap* version of it, which is what training needs |
| one interactive page for placing checkpoints on the range | not wanted; a script (and optionally a notebook) writing SVGs is the artifact, because it reruns and diffs |

## 10. Open questions

1. **Observation/action encoding (hole c)**: build the policy network colour-equivariant by
   construction (symmetrisation free, |G|-fold data efficiency), or a plain encoder wrapped in
   `Symmetrized` (simpler, pays orbit averaging at every forward pass)? This is a 10a decision
   because A2's batched tabulation is written against whichever it is.
2. **LBR in 10a or deferred to 11?** My recommendation is "in, first to be cut".
3. **Who is in the league, and is it fixed before step 11 starts?** Settled: fixed-opponent
   training against a league of rulebots, self-play secondary -- so `br_gap` is the load-bearing
   half of the card and exploitability is the overfitting detector. Still open: the membership
   (Heuristic, distilled bot, a `KeepsDrawn`-style member, PIMC as an evaluation-only rung?), and
   whether checkpoints are **selected** on league score -- if they are, the league's seeds are dev
   seeds and the held-out block has to be reserved for the report card alone.
4. **Is `TOY2_FULL` still worth keeping in the loop at all**, given section 2 -- or do the toy cells
   in the report card run on `TOY` and `TOY2_SKIP` only, with `TOY2_FULL` reserved for milestones
   (its trees are 1.9-3.0 GB and ~50 s to tabulate per agent)?
