# Step 10 second opinion: memory, LBR, and readiness for RL

> **Follow-up:** 10a.3A implemented the memory repair and refreshed the evidence;
> see [10a.3A results](10a3a-results.md). The diagnosis and reproduction below
> remain the historical motivation. Observation-contract and LBR recommendations
> remain pending as 10a.3B/C.

Review on 2026-09-17 at `f12ffaf`. No learner or algorithm changes made.
Sources: `context.md`, the results index, Step 9.5a–10a.2 results, relevant spec
sections, and the coarse-key, belief, search, target, and evaluator code.
Recommendations below are proposed work, not accepted implementation decisions.

## Assessment

Reopen the observation decision and add an approximate-response evaluation before
serious full-UNO training. Keep K1 as a baseline. The existing infrastructure is
enough to start a carefully staged learner; a perfect belief filter and every
planned presentation artifact need not precede the first toy learning curve.

The strongest parts of the repo are its reference/compiled cross-checks, explicit
information boundaries, per-seat accounting, and willingness to measure failed
ideas. The recurring weakness is promoting a local observation into a general
conclusion: toy sufficiency into full-game sufficiency, one failed sampler into
rejection of particle filtering, or a flat search-budget curve into a ceiling.

Actual status: 10a and 10a.2 are complete. LBR was explicitly deferred in
`10a2-results.md` section 6; figures and distilled rulebots are still planned.
The opening/status sections of `context.md` lag its own later additions.

## 1. The K1 evidence needs repair even before transfer

Three distinct questions have become conflated:

1. What score did the constructed coarse policy achieve? Exactly measured.
2. What is the optimal score over all strictly K1 policies? Not solved.
3. Is K1 sufficient on full UNO? Not tested.

`memoryless_response` performs iterative greedy updates and keeps the best
iterate. Exact evaluation of that iterate does not make the optimization exact.
An achieved gap to BR L1 can bound representational loss from above **if the
policy really belongs to the claimed representation**; a residual gap does not
prove the representation caused it.

That membership condition currently fails on the quotient. `coarsen` documents
that it extracts K1 after canonicalizing the full L1 history. This can retain
history through the choice of coordinate frame. Its action-orbit joining enforces
L1 validity, not all of the invariances a strictly coarse observation requires.

### Reproduced counterexample

Built the default 20-iteration quotient K1 response to Random at seat 1 on `TOY`,
then lifted its probabilities to an ordinary L1 `CompiledAgent`.

- Score: **0.5494623015873016**, matching the historical K1 row.
- Canonical L1 set 423 has positive on-policy reach **0.0017460317460317458**.
- Its observation and symmetry image 12 have **identical raw K1 keys**.
- The actual `CompiledAgent.policy` outputs are **`{1: 1.0}` and `{3: 1.0}`**.

The two L1 information states are:

```text
seat 1, initial hand (0, 1, 2), own draws (3,)
seat 1, initial hand (0, 2, 3), own draws (1,)
public history in both:
  setup 4; player 0 plays wild 8 choosing color 2;
  player 1 draws one; player 0 plays 5 choosing color 2
```

Both now hold `(0, 1, 2, 3)` and agree on every K1 field. Remembering which card
was drawn lets the lifted policy distinguish them. This is a valid L1 policy,
but not a strictly K1-measurable one. It does **not** invalidate ordinary L1 BR,
CFR, or the score of this policy. It invalidates using that score as a demonstrated
K1-only achievement. The numerical effect of repairing it has not been measured.

Reproduction recipe: `load_or_compile(TOY)`; `coarsen(tree, k1)`;
`memoryless_response(..., opponent=(L1, tabulate(Random)), seat=1,
start_l1_sigma=tabulate(Heuristic), iterations=20)`; copy each coarse set's
action probabilities to its constituent L1 sets; construct `CompiledAgent`;
compare `observation_from_key(TOY, L1, k.key(423))` with the observation from
`tree.symmetry.key(L1, 12, k.key(423))`. Set numbering is for this revision.

Before reusing the memory evidence, require that equal raw encoder inputs give
equal action distributions. Establish a full-tree baseline on a manageable toy,
then repair quotient key/action canonicalization around the coarse fields,
including the coarse stabilizer. Validate the actual deployed encoder the same way.

The headline percentages also need care. In 9.5a, `0.0989 / 0.1026 = 96%`
divides returns relative to a 50% score, not skill gains relative to the same-seat
Random baseline. Absolute score gaps and the report card's explicit normalization
are clearer. The table already has nontrivial gaps: K1 trails BR L1 by about
**1.37 / 1.58 score points on TOY2_SKIP against Random**, subject to the membership
and optimization qualifications above.

## 2. What the full-UNO belief results establish

`voids` improves the particular prior-PIMC player by **2.83 / 4.40 points against
Random** and **2.27 / 3.70 against Heuristic**. That is strong practical evidence
for retaining historical evidence in the learner. It does not compare against
the optimum of the K1 policy class. Prior-PIMC is not that optimum, and even its
world-sampling RNG is keyed on full `obs.info_state`.

So the right conclusion is: **history helps the current full-UNO method, and
there is no justification for imposing K1 as the learner's final information
bottleneck.** A formal separation of the optimal policy classes remains open.

Existing K2 is not the obvious remedy. It records draw colors since the
opponent's last play, clears on that play, and counts draw events. The later
`voids` machinery retains board-specific constraints across plays, distinguishes
penalties, and weakens evidence as more cards are drawn. Testing K2 did not test
the representation used by the successful inference experiment.

Proposed representation comparison, with a shared equivariant action head:

| Input | Purpose |
|---|---|
| Audited K1 feed-forward encoder | Cheap baseline and explicit memory ablation |
| K1 plus structured draw/play evidence | Cheap use of the measured history signal |
| Current position plus recurrent public/private event history | Learn evidence beyond the hand-written summary |

Treat void evidence as features or soft assumptions for an unknown opponent,
not hard truth. Cover voluntary draws, kept playable draws, penalty draws,
reshuffles, and evidence age. Include publicly known setup-tail information and
the action-cap clock when defining full-UNO inputs. Preserve color/rank relationships
in the memory representation; pooling events into invariant scalars can discard
which color or rank the evidence concerns.

Run a small full-UNO DEV pilot alongside the toy validation. Compare these arms
at equal training data and separately at equal wall time, over multiple training
seeds and held-out opponent behaviors. Prediction of the hidden hand can diagnose
memory, but decision quality and game score must decide its usefulness.

## 3. Particle filtering was not disproved

The reported collapse disproves the practical deal-time proposal/filter that was
tried. It does not establish that sequential Monte Carlo cannot work. Better
observation-conditioned proposals and resample-move methods are standard responses
to particle degeneracy; they still require correct importance weights and may
remain expensive. See [Doucet and Johansen, sections 4.3–4.5](https://www.cs.ubc.ca/~arnaud/doucet_johansen_tutorialPF.pdf).

For UNO, a candidate approach would propose latent hand/draw histories consistent
with observed plays, maintain weights for the target opponent model, and rejuvenate
compatible histories when diversity collapses. Resampling copies alone cannot
recover support after every particle has died. Arbitrarily inserting a played
card also does not implement a posterior without the appropriate proposal correction.

The experiment to justify a second attempt: exact toy posterior comparison,
effective sample size and diversity, fallback/support failures, behavior under
deliberate draws/passes, and downstream action quality per unit compute. A fixed
64-particle deal sampler failing is evidence to redesign the proposal, not simply
raise the particle count. This remains a candidate inference project, not a
prerequisite for learning from raw history.

## 4. Add LBR as an attack instrument

The original [Lisý–Bowling LBR paper](https://arxiv.org/abs/1612.07547) proposes an
approximate lower bound on best-response value. An UNO adaptation needs its own
validation; the poker-specific local continuation does not transfer automatically.

Start with a frozen target checkpoint and an L1-valid attacker. Reuse the current
response-search implementation as a baseline attacker, then compare an attack that
evaluates legal root actions using sampled worlds and a fixed L1 continuation
policy. Such rollouts avoid clairvoyant choices in the continuation; their variance
and cost still need measurement. Independent final games, not internal search Q
values, determine the attacker's achieved value.

**A weak or biased planner can still produce a valid lower bound through its
actual played score.** It only needs to be an admissible L1 policy; it need not
have an exact belief or exact Q. The danger is an insensitive attack, not automatic
invalidity of the bound. A strong attack reveals weakness; a failed attack does
not demonstrate safety.

For full UNO, the seat-specific Nash value is unknown. Do not subtract 0.5 from
each seat and label the result exploitability. If `b0` is the score of an attacker
at seat 0 against the target's seat-1 policy, and `b1` is the corresponding
seat-1 attack score, then in this constant-sum score convention:

```text
NashConv(target) >= b0 + b1 - 1
```

The unknown Nash seat values cancel in the sum. Report a confidence lower bound
for that sum, multiplied by 100 for score points; zero is always a trivial lower
bound. Preserve the paired-deal covariance when estimating uncertainty. Freeze
attackers selected on DEV before held-out scoring, or account for multiple selection.

On the toys, compare attack scores with exact BR for Random, Heuristic, Nash,
and deliberately flawed policies. Report how much known exploitability each
attacker detects, its misses, and cost. Including several attacker families helps
avoid confusing an attack's blind spot with strength of the learner.

There is a concrete integration obstacle for memoryful checkpoints:
`determinize_l1` fabricates the opponent's initial hand and own-draw history.
Current history-free rulebots tolerate it; a recurrent opponent may receive an
impossible history. Reconstruct consistent latent histories/memory or explicitly
restrict the attacker version to opponents whose input can be reconstructed.

## 5. Step 11: staged learning with diagnostic gates

### Phase 1: toys, exact evaluation

First train against one fixed known opponent so the objective has an unambiguous
BR target. Compare a tabular L1 learner, a network using exact expected policy
gradients, and sampled learning. This separates representation/optimization from
sampling and estimator issues. A small supervised fit to an exact policy or
action-value target is also a useful capacity/encoding check, not an RL success.

Use `TOY` for fast plumbing, `TOY_FULL` for voluntary draws/passes, and `TOY2_SKIP`
for action timing. Reserve `TOY2_FULL` for milestones. Define tolerances before
runs, and report exact score gaps across several training seeds. An arbitrary
"95% of the ceiling" can hide the seat and normalization problems above.

Then train on a league and evaluate unseen behavioral variants as well as exact
exploitability. Declare whether opponent identity is visible and whether the
mixture is drawn once per episode. For a hidden episode-level opponent identity,
the optimal learner may infer the identity from history. Per-column BR values
are upper benchmarks, not necessarily simultaneously attainable by one policy.
Low exploitability should be a separate objective/experiment if robustness is
required; optimizing league reward alone does not promise it.

Start the sampled learner with ordinary returns and a baseline; compare the cheap
deal control variate next. Keep chance/opponent corrections as measured ablations.
The 10a.2 gradient result validates the stated score-function estimator. It does
not automatically validate PPO clipping, bootstrapping, truncated recurrent
backpropagation, advantage normalization, or GAE. Test the actual estimator against
the exact gradient with a frozen rollout policy, detached targets, and a frozen
opponent. An exact L1 critic is not an exact critic for a representation that merges
those histories, and an exact state-only critic may miss a memoryful opponent's state.

### Phase 2: full UNO, sampled evaluation

Carry forward the verified learner and benchmark K1, evidence features, and
recurrent memory. Log trajectories/decisions per second, time to a target score,
learning-seed spread, paired improvement, per-opponent/per-seat scores, and LBR
lower bounds. Keep a diverse cheap training league including deliberate draws
and kept playable cards; freeze held-out behavioral opponents too, not just seeds.

The column-free PIMC is a strong existing reference (block 01: **0.7690 / 0.7510
against Random**, **0.6425 / 0.6122 against Heuristic**). Label it a benchmark.
Beating it is success; failing to beat it is demonstrated attainable headroom
in that matchup. Matching it does not demonstrate proximity to the true optimum.

Full-UNO AIVAT and LBR remain sampled. Choose evaluation budgets from actual DEV
variances for the new policies. Existing 1,250–1,800-deal estimates for nearby
rulebot variants are not universal checkpoint budgets. Batch policy inference
across active games and events, not just leaves inside one search; measure the
engine/state-construction floor before choosing costly corrections.

### Phase 3: iterations driven by the remaining failure

- Tabular succeeds, exact-gradient network fails: encoder, capacity, symmetry,
  or optimizer problem.
- Exact-gradient network succeeds, sampled learner fails: sampling, credit
  assignment, estimator, or exploration problem.
- History-aware learner beats K1: representation/memory is an empirical lever.
- League score rises while attacks strengthen: opponent coverage or robustness
  objective needs work.
- Learner matches PIMC: strengthen the benchmark/attacks before declaring a ceiling.

## 6. Other conclusions I would soften

- `soft(0.1)` is a useful measured compromise on two toy opponents; it is not
  proven calibrated or minimax over full-UNO opponents.
- A 19.5% belief-change disagreement rate minus an 18.4% reseeding disagreement
  rate does not identify that history matters at exactly 1.1% of decisions.
  These are different joint comparisons. Use repeated paired sampler seeds and
  decision-value measurements before converting the difference into that claim.
- Symmetry enforces consistency across relabelings; group size is not a measured
  multiplicative gain in sample efficiency. An equivariant network must also
  retain enough relational information to express the desired policy.
- Toy Nash deal-variance fractions do not establish the full-UNO training-gradient
  benefit of deal correction. Measure the gradient variance and wall time directly.
- Distilled rulebots are useful, but tuning only on toys again risks throwing away
  full-game memory effects. Separate a toy-to-UNO transfer study from building the
  strongest affordable full-UNO training league. Neither figures nor a historical
  70% target should block the first validated learning experiment.

## Checks performed for this review

- `.venv/bin/uno report TOY heuristic`: **0 failed checks**; reproduced score,
  BR-gap, and exploitability headline values.
- Two bounded in-memory K1 diagnostics: found the raw-input collision, then
  verified it through `CompiledAgent.policy` and checked positive on-policy reach.
- No training, full-UNO sweep, evaluation-block claim, solver fix, or commit.
