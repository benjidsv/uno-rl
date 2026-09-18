# Observation contract for RL learner arms

**Revised 2026-09-18 following the agreed RL review.** This supersedes the original
Step 10a.3B learner design (base `aee9384`), not historical foundation measurements.
It specifies intended production behavior; existing test-only extractors and the
PIMC distillation encoder do not yet implement/validate this revision. Track the
implementation gaps in [readiness-review.md](readiness-review.md).

B is the first practical agent, A the position/memory ablation, and C the subsequent
recurrent comparison. All actors are L1, with the same action space, legality,
seat convention, and color/rank symmetry. No arm is established sufficient for
full UNO. Training choices and gates belong in [plan.md](plan.md).

## 1. Shared interface and boundaries

### 1.1 Actions, masks, and probabilities

Use the ruleset's static `Ruleset.actions`: card plays, joint wild/card-color
plays, Draw, and Pass. At an actor decision, legality depends on own current hand,
top discard, active color, phase, pending drawn card, and static rules flags.
Use `uno.game.engine.legal_actions` as the reference. At `AFTER_DRAW`, only the
pending card's plays and Pass are available. At turn start, Draw is suppressed
when the rules require playing and a play exists.

The network outputs logits; the policy masks illegal actions before normalization.
The same mask and temperature apply during collection, learning, evaluation, and
checkpoint replay. The reported distribution is the one actually sampled.
Nonzero policy support is not the legal set: a deterministic teacher may report
one action while many are legal. Distillation must label all legal actions.

### 1.2 Actor-relative seat convention

Actor features use self/opponent indices: own hand is self, public player indices
are mapped to self/opponent, and outcomes/baselines use the learner's perspective.
Keep absolute seat as an explicit scalar to represent deal/starting-seat effects.
Raw engine states and existing value-network functions retain their documented
indexing; use an explicit adapter rather than assuming those functions already
swap seats. Never negate value merely because an action granted another own turn.

### 1.3 Clock, reshuffles, and setup

All arms include `obs.turn / max_turns` when a cap exists, otherwise zero.
`turn` counts plays and voluntary draw actions, not Pass or penalty cards.
The full-UNO rule-defined cap is terminal, unlike a collector truncation.
All arms include `min(n_reshuffles, 3) / 3`.

B and C also expose `has_reshuffled` and `setup_tail_size`, the number of public
setup-skipped cards still at the pile bottom. Before the first reshuffle, bound
this by the remaining pile size as the tail is consumed; after reshuffle it is
zero. Use the engine/observation reconstruction as the oracle for exhaustion and
partial-tail cases. Full ordered setup information remains available to C.

### 1.4 Symmetry without discarding relationships

Require `policy(g . obs)(g . a) = policy(obs)(a)` for color permutations and
rank permutations within equal-copy classes (foundation spec section 18).

The production preference is equivariance by construction: keep card/color/rank
indexed features, shared processing, and shared action scoring. Preserve rank
identity across colors and color identity across ranks/events. Invariant global
context can be pooled, but an unrestricted MLP over flattened ordered color blocks
breaks equivariance. Standardization must share statistics across permutable
coordinates and fit only on training data.

The existing `ValueNet` is an invariant value encoder with compressed inputs; a
policy head alone does not establish action equivariance or preserve the complete
information listed below. Histogram summaries are additional compression, not
an interchangeable encoding of full card multiplicities.

For recurrence, require `F(g h, g e) = g F(h, e)` and a symmetry-fixed initial
state. A generic dense GRU/LSTM does not satisfy this simply because `h0 = 0`.
Keep structured color/rank memory with shared recurrent updates. Pooling away the
color of every event before recurrence loses its relation to later cards.

Canonicalization plus stabilizer-orbit averaging is an alternative correctness
oracle on small toys. Canonicalize only the arm's declared inputs: a full-history
coordinate choice can leak discarded history into A/B. The existing generic
`Symmetrized` wrapper uses full information keys and is not automatically a valid
A/B wrapper. Do not enumerate full UNO's color/rank group in the training loop or
inherit an unsupported microsecond cost claim for that path.

A quotient-tree tabulation must satisfy both global equivariance and uniformity
within each information set's stabilizer action orbits. Test transformations of
actual probabilities, not just input shapes or a single rank swap.

## 2. Actor arms

### 2.1 A: position plus discard (K1-style ablation)

| Field | Required information |
|---|---|
| `hand` | full own multiplicity per card type |
| `discard` | full current discard-pile multiplicity per card type, from public history |
| `top` | full card identity, including color/rank/kind relationships |
| `current_color` | active color |
| `drawn` | pending card identity or none |
| `phase` | turn start / after draw |
| `opp_hand_size` | current size, scaled by initial hand size without clipping |
| `pile_size` | current size, scaled by deck size |
| `turn` | clock from section 1.3 |
| `reshuffles` | capped count from section 1.3 |
| `seat` | absolute seat, with other fields in the actor-relative frame |

A omits ordered history, initial hand, ordered private draws, and setup-tail
features. It has no mutable episode memory. It may reconstruct its declared fields
from history, but no other historical information may affect its output.

This is **K1-style**, not literally the historical coarse key. That key includes a
tree-level clock, whereas A uses an action-cap clock and reshuffle features.
Historical K1 achieved values do not certify this encoder's attainable score.
The [corrected memory results](../foundations/results/10-readiness/10a3a-results.md)
report achieved local-search values, not K1 optima; gaps to full L1 BR do not
isolate representation loss. Neural runs measure achieved performance, not a
representation ceiling.

### 2.2 B: A plus structured public evidence

B includes all of A, unchanged, plus a fixed-width public-history summary:

| Evidence | Required distinction |
|---|---|
| Opponent voluntary draws | board at the draw: active color and top card's rank/kind/color |
| Evidence age | cards drawn by the opponent since the evidence, with declared bounded bins/counts |
| Penalty draws | separate event count and card count; not evidence of lacking playable cards |
| Draws since last opponent play | card count, not a count of draw events |
| Most recent opponent play | full card identity and chosen color, or none |
| Setup state | remaining setup-tail size and has-reshuffled flag |

Use fixed tensor indices/bins keyed by the relevant color and rank/kind, so the
relations survive and permute with the cards. B is not a variable-length event
model. The precise age bins, count caps, and whether old evidence is retained in an
overflow bin must be recorded before implementation comparisons; age threshold 4
is a pilot candidate, not a measured universal default. Define separate event age
if used; elapsed decisions and cards drawn since evidence are different quantities.

Voluntary draws are **soft evidence**. A policy may draw with a play available or
keep a playable drawn card. These features neither restrict legality nor assert a
hard void. At a voluntary draw, `since` excludes the cards drawn at that event,
matching the existing belief signal; the learner must be able to discount that
assumption. Plays do not erase all earlier evidence. Parse penalty/voluntary status
correctly through reshuffle events and partial/exhausted draws, using the engine
history contract and explicit regression cases.

Retaining top rank/kind as well as active color is intentional: the existing
`voids` signal is about playable cards against a board, not only a color. B extends
the old K2 hint, which counted draw events and cleared on play; it is not K2 under
a new name. Recent-play identities let the encoder relate evidence to current
cards without adding opponent private information.

### 2.3 C: A plus recurrent public/private history

C receives **all of A, including the discard multiset**, the setup fields from
section 1.3, initial own-hand multiplicities, full public history in order, and own
private draws in order. It need not first learn discard bookkeeping to benefit
from sequence information. C does not require B's engineered evidence summaries;
its recurrent encoder learns a summary from history.

Preserve the engine event schema:

| Event | Encoding requirements |
|---|---|
| `setup(top, skipped)` | full top identity and ordered skipped cards; do not silently replace the known tail order with a multiset |
| `play(player, type, color)` | self/opponent flag, full card identity, chosen color |
| `draw(player, n)` | self/opponent flag and actual card count; align own private cards in their recorded order |
| `reshuffle` | event token and its position in the sequence |

Associate own private draws with the corresponding public draw event using its
actual count. Do not append all private draws as an unaligned second sequence or
invent an opponent draw identity. Setup and initial hand are processed exactly
once. Pass emits no public event: do not manufacture a publicly observed opponent
Pass or leak whether an opponent's drawn card was playable. The current phase and
pending card are explicit actor inputs.

The reference query replays the actor's actual L1 information state from scratch.
An incremental cache must reproduce that distribution, processing only new events
and resetting per game/seat. Cover after-draw decisions, repeated queries,
consecutive own turns, and branched evaluator queries. A cache is an optimization,
not an extra information source or a mutable state shared across hypothetical
branches. Document PPO sequence truncation/burn-in and recompute hidden states as
required when weights change.

Full L1 history is available before encoding. This gives C access to information
A/B omit; it does not prove a finite recurrent network can realize every
L1-measurable policy. Architecture capacity and practical training cost are measured
questions, with no fixed relative-cost multiplier assumed.

## 3. Actor, critic, training targets, and evaluation

| Role | Boundary |
|---|---|
| Actor | declared A/B/C inputs only; L1; no opponent private state, seed, pile order, or opponent ID |
| Critic | separate network initially; may use L2, actor frame, clock, phase; never pile order or seed |
| Initial training signal | terminal return minus a detached baseline; no AIVAT corrections |
| Optional corrected training signal | shipped `TRAINING`: chance + opponent + deal, never learner actions; separately validate any PPO/GAE integration |
| AIVAT evaluator | can correct either player's actions with valid probabilities/value queries; report configuration |
| Attacker | an L1 policy; hypothetical queries must respect its target-support contract |

Pile multiset information may be derived from L2-visible card counts; it does not
authorize pile order. A privileged action-independent baseline can weight the
actor's Monte Carlo gradient without giving private inputs to the deployed actor.
Detach it; critic inputs must not enter actor activations, recurrent memory, or
normalization. An approximate baseline and bootstrapped targets have different
requirements: current position need not be a sufficient state for a history-aware
policy, and GAE/TD bias needs separate validation.

The learner-action correction is excluded from **training returns** because it can
remove the credit-assignment signal. That does not prohibit learner-action
correction in **evaluation**. Deterministic played distributions give zero action
sampling correction at their own decisions; stochastic actors must report their
actual distribution. AIVAT validity also requires valid conditional expectations
and history-consistent queries, not merely a deterministic network function.

## 4. Evaluation and hypothetical-state compatibility

A/B do not use private initial hands or ordered private draws, so fabricated
opponent private histories in `determinize_l1` do not directly affect their policy.
Their current position and public summary can be queried in sampled continuations.
Still validate production target support, normalization, legal actions, and all
query modes; “compatible inputs” is not proof of integration correctness.

C works in ordinary games using real history. Exact toy tabulation can replay
`observation_from_key` at each L1 information set, with cost proportional to the
history replay work. AIVAT integration must replay valid actor histories for every
actual or counterfactual policy query and isolate branch-local hidden states.

The determinizer retains the attacker's real history but sets the sampled
opponent's initial hand to its current sampled hand and its private draws to empty.
That is generally **inconsistent with the public history**, and cannot stand in
for C's conditional policy. Restricting the attacker to its own seat does not
magically supply the target's hidden history at continuation decisions.

Response search against C needs a validated model of consistent latent histories
with declared conditional weighting, or a different attack that avoids those
queries. Consistency alone is not an exact posterior. Never reveal the target's
real private history to an L1 attacker as a shortcut. Real-game scores against
frozen L1 attackers are still possible and yield attack lower bounds; they do not
establish optimal responses or safety. Sophisticated recurrent-target search is
not a prerequisite for building C.

## 5. Required production tests and current status

The original `tests/game/test_observation_contract.py` uses test-only extractors.
Its legality test checks field presence and nonempty engine actions, not a mask
reconstructed from production inputs. Those passing tests do not certify the new
learner. Required implementation checks are:

1. **Legality:** production masks equal the engine on multiple rulesets, including
   multi-action states, voluntary draws, playable drawn wilds, and Pass.
2. **Information:** changing unseen cards/seed/private histories while keeping the
   declared actor input fixed leaves features and probabilities unchanged. A/B also
   ignore historical distinctions outside their summaries.
3. **Symmetry:** transform production features and probabilities under independent
   color and allowed rank permutations, with nontrivial action choices and
   stabilizers. Include preprocessing, serialization, and batched inference.
4. **Evidence:** pinned histories for board rank/color, penalty draws through a
   reshuffle, actual card counts, evidence ages, recent plays, and consumed setup
   tails. Tests must detect an intentionally incorrect extractor.
5. **Queries:** A/B agreement on compatible determinized inputs; C full replay,
   incremental replay, L1-key reconstruction, branch isolation, and seat resets.
6. **Learning interface:** collection/recomputed probabilities agree with frozen
   weights; masks and temperatures match every loss and deployment path; batched
   paths agree with bounded reference implementations.

No claim of completed tests, learner implementation, full-UNO sufficiency, or
recurrent-target attack support follows from this contract. The next implementation
must supply the evidence and record remaining unsupported paths explicitly.
