# Observation contract for Step 11 learner arms

Step 10a.3B. Specifies the three comparison arms that Step 11 will implement and
compare, without building the learner. This document is the source of truth for
what each arm sees, how it is queried, and what invariants it must satisfy.

**Base:** commit `aee9384`, on top of 10a.3A and the 10→11 plan.

The three arms share an action space, a legality mask, an equivariance
requirement, and a seat convention. They differ in how much of the public and
private history the actor's policy network sees. Full-UNO sufficiency of any arm
is a Step 11 experiment, not a claim of this document.

---

## 1. Shared structure

### 1.1 Action space and legality mask

The action space is the ruleset's static `Ruleset.actions` (spec §5): one id per
(card-type, chosen-color) combination for plays, one `Draw` id, one `Pass` id.
The same space for both players and every state.

The **legality mask** is a boolean vector over that space. It is a function of
five quantities only:

| field | meaning |
|-------|---------|
| `hand` | the actor's current hand (sorted tuple of type ids) |
| `top` | the top discard type id |
| `current_color` | the color currently in force |
| `phase` | `TURN_START` or `AFTER_DRAW` |
| `drawn` | the pending drawn card (only when `phase == AFTER_DRAW`) |

plus the ruleset's static flags (`forced_play`, `draw_then_may_play`). All three
arms include these fields. The mask is computed identically across arms by
`uno.game.engine.legal_actions`; no arm's architecture may bypass it.

At `TURN_START`: `Play(c)` / `Play(c, color)` for every playable card in hand
(each distinct type once), plus `Draw` unless `forced_play` and something is
playable.

At `AFTER_DRAW`: `Play(drawn)` / `Play(drawn, color)` (the drawn card only),
plus `Pass`.

### 1.2 Color/rank equivariance

All three arms must produce policies invariant under the color/rank symmetry
group G of spec §18. A symmetric policy satisfies
`policy(g . obs)(g . a) = policy(obs)(a)` for every group element g.

Two enforcement paths, equivalent in outcome:

1. **Equivariant by construction.** The architecture commutes with the group
   action on its inputs and outputs. This is the preferred path. The existing
   `ValueNet` achieves it via DeepSets over colour blocks: one shared MLP per
   colour block, sum/max pooling. A policy network following the same pattern
   is equivariant without any explicit symmetry call.

2. **Canonicalize + orbit-average.** `Symmetrized` wrapping: canonicalise the
   observation, forward-pass once, map actions back through the inverse element,
   average within action orbits under the canonical key's stabilizer. This
   requires a `canonical()` call per forward pass (~6.5 us on `FULL_UNO`).

For Arm C (recurrent), equivariance means the encoder's hidden state must also
commute with the group. Given equivariant event embeddings (section 3.3) and a
standard recurrent cell, this holds if the initial hidden state is a fixed point
of the group (e.g. zero).

Consequences for tabulation: `tabulate` and `tabulate_batched` refuse a policy
that is not orbit-uniform on a quotient tree (spec §18). An arm that is not
equivariant by construction must be wrapped in `Symmetrized` before tabulation.

### 1.3 Seat convention

Inputs are in the **seat-0 frame**: the actor's hand is hand 0 and the
opponent's hand size is hand 1's size. When the actor sits at seat 1, the
frame is swapped: hand 0 becomes the actor's hand, the mover flag is flipped,
and the public history's player indices are swapped. This is the same frame
`position_features` (spec §24) and the training critic use.

Output: the actor's own action distribution over the full action space, with
zeroes outside the legality mask.

### 1.4 Turn clock and action-cap

On `FULL_UNO` the game ends at `max_turns = 1000` play+draw actions (spec §9).
The **turn count** (`obs.turn`, = number of play + voluntary-draw events in the
public history) is included in all arms as a normalized scalar `turn / max_turns`
(or `turn / 1000` on `FULL_UNO`; 0 on toys with `max_turns = None`).

### 1.5 Reshuffle state

All arms include the **reshuffle count** capped at 3: `min(n_reshuffles, 3) / 3`.
This is already a feature of `position_features`. After a reshuffle the setup
tail is no longer at the known bottom of the pile and the unseen-card pool
changes, which affects the game through the draw distribution.

### 1.6 Setup-tail information

Until the first reshuffle, the cards skipped at setup sit at the known bottom
of the pile. They are public (the setup event lists them) and they affect the
draw distribution: those cards are drawn last. All arms include `pile_size`.
Arms B and C also include:

- `setup_tail_size`: number of setup-skipped cards still at the pile bottom;
  equals `len(skipped)` before the first reshuffle, 0 after.
- `has_reshuffled`: boolean, whether any reshuffle has occurred.

These are derivable from the public history. On the toys (no reshuffle, no
skipped cards in most cases) both are trivial.

---

## 2. The three arms

### 2.1 Arm A: K1 feed-forward baseline

**Purpose:** the explicit memory ablation. A policy that sees the current
position and the discard-pile contents but no sequential history. The cheapest
arm to train and evaluate.

**Inputs**, all derivable from the K1 coarse key (spec §20):

| field | encoding | source |
|-------|----------|--------|
| `hand` | `(num_types,)` multiplicity vector | `obs.hand` counted per type |
| `top` | integer type id, one-hot kind (6 dims) | `obs.top`, `card_types[top].kind` |
| `current_color` | integer, one-hot (colors dims) | `obs.current_color` |
| `opp_hand_size` | integer / `hand_size` | `obs.opp_hand_size` |
| `pile_size` | integer / `deck_size` | `obs.pile_size` |
| `phase` | boolean: `AFTER_DRAW` flag | `obs.phase` |
| `drawn` | type id (-1 for none), one-hot kind (6 dims) | `obs.drawn` |
| `discard` | `(num_types,)` multiset of discard pile | `discard_from_public(obs.public)` counted |
| `turn` | normalized turn clock | `obs.turn / max_turns` |
| `reshuffles` | `min(n_reshuffles, 3) / 3` | derivable from public history |
| `seat` | integer (0 or 1) | `obs.seat` |

**Not included:** the public history as a sequence, `own_draws` in order,
`initial_hand`, setup-tail size or reshuffle flag (beyond the count). Arm A is
strictly position-plus-discard.

**Architecture hint:** DeepSets over colour blocks (one block per colour for
each of: own hand, discard, drawn card, top card; a global block for scalars),
sum + max pool, MLP. This is `ValueNet`'s architecture with a policy head instead
of a value head. Equivariant by construction.

**What it cannot represent:** any policy that conditions on the order of past
events, the order of own draws, or the specific cards the opponent has drawn.
The K1 gap to BR L1 on the toys is 0.0–1.7 score points (10a.3A results), so
the representation's cost is measurable but bounded on the toys.

### 2.2 Arm B: K1 + structured evidence (feed-forward)

**Purpose:** extend the position with the measured historical signal —
specifically the draw/play evidence that `voids` captures (spec §26.2–26.7) —
in a form the network can learn to weight.

**Inputs:** everything in Arm A, plus:

| field | encoding | source |
|-------|----------|--------|
| **void evidence** (per colour) | per colour: count of opponent voluntary draws where that colour was current and `since ≤ S` for a threshold S (e.g. 4); and `min(since, S)` for the most recent such void. | parsed from public history using `_voluntary` |
| **total void count** | integer: number of active void constraints | count of opponent voluntary draws |
| **penalty draw count** | integer: opponent penalty draws (from +2 / +4) | parsed from public history |
| **opponent draws since last play** | integer | the K2 `_void_hint` length |
| **most recent opponent play** | type id + chosen color (or "none") | last `("play", opp, ...)` event |
| **setup_tail_size** | integer | derivable from public history |
| **has_reshuffled** | boolean | whether reshuffle count > 0 |

**Design choices:**

1. **Fixed-size summary, not variable-length.** Arm B is a feed-forward arm.
   The evidence is encoded as a fixed-width vector appended to Arm A's input.
   This keeps it cleanly separated from Arm C's "learn your own summary" role.

2. **Soft evidence, not hard constraints.** Against unknown opponents the
   `voids` assumption (that a playable drawn card is played at once) may not
   hold. Arm B's features encode the constraint's *inputs* (how many voids,
   how many cards drawn since, at which board) so the network can learn how
   much weight to give them. It does **not** filter legal actions or mask the
   world-sampling distribution; that is the belief module's job during
   evaluation.

3. **Per-colour void counts preserve equivariance.** A void constraint against
   colour c is evidence about colour-c cards in the opponent's hand. Encoding
   it in the colour-c block maintains the DeepSets-over-colours structure.
   The `since` count and `age` go in the same block.

4. **Penalty vs voluntary distinction.** Penalty draws (+2 / +4) carry no
   inference about the opponent's hand. Arm B separates voluntary draws (which
   generate void evidence) from penalty draws (which do not). This uses the
   existing `_voluntary(rs, public, i)` function.

5. **This is NOT K2.** K2 records "colours current at each opponent draw since
   their last play" — it clears on play, counts events not cards, and does not
   distinguish penalty from voluntary draws. Arm B extends this with: per-colour
   structure, the `since` count, the penalty/voluntary distinction, and the
   setup-tail and reshuffle state. The review called this "a new representation,
   not a renaming of the current weak K2 hint."

**Architecture hint:** same DeepSets-over-colours with the void evidence fields
appended to each colour block, and the scalar evidence features appended to the
global block. Equivariant by construction.

### 2.3 Arm C: recurrent public/private event history

**Purpose:** let the network learn its own summary of the game history, rather
than choosing the summary for it. The arm that can, in principle, recover any
L1-measurable policy.

**Inputs:**

| component | content | encoding |
|-----------|---------|----------|
| **current position** | Arm A's position fields (hand, top, color, opp_hand_size, pile_size, phase, drawn, turn, reshuffles, seat) | same as Arm A, but without the discard multiset (which the recurrent state can reconstruct) |
| **public event sequence** | the full `obs.public` tuple, in order | per-event structured embedding (section 2.3.1) |
| **private draw sequence** | `obs.own_draws` in order | per-draw type embedding |
| **initial hand** | `obs.initial_hand` | `(num_types,)` multiset (order-invariant) |

The current position is concatenated with the recurrent encoder's final hidden
state and fed to a policy head.

**2.3.1 Event embedding.** Each public event is embedded as a fixed-width vector:

| event type | fields | embedding |
|------------|--------|-----------|
| `("setup", top, skipped)` | top card type, skipped card types | top one-hot kind + colour block; skipped as a multiset |
| `("play", player, type, color)` | who played, card type, chosen color | player flag + card one-hot kind + colour block for the card + colour block for the chosen color |
| `("draw", player, n)` | who drew, how many cards | player flag + draw count (1, 2, or 4) |
| `("reshuffle",)` | — | a single reshuffle token |

The event embeddings go through a shared equivariant encoder: each event's
colour-related fields live in colour blocks, pooled by the same DeepSets
pattern, so a colour permutation permutes the blocks and the overall embedding
is invariant. Private draws use the same card-type embedding.

**Recurrent cell.** The contract does not prescribe GRU vs LSTM vs Transformer.
Step 11 chooses. The contract requires:

- the cell processes events **in order**, from setup through the current
  decision;
- the initial hidden state is a fixed point of the symmetry group (e.g. zero);
- the cell's output, combined with the current position, yields equivariant
  action logits (either by construction or by `Symmetrized` wrapping).

**What it can represent that Arms A/B cannot:** any L1-measurable policy, since
the L1 information state (seat, initial hand, public history, own draws in order)
is fully available. In particular, it can learn temporal patterns in the
opponent's play, long-range colour/rank correlations, and pile-composition
tracking across reshuffles.

**What it costs:** longer forward passes, backpropagation through time, and the
evaluator integration constraint of section 4.

---

## 3. Actor vs critic vs attacker

The three arms define the **actor** (the learner's own policy). The other roles
in the system have different information:

| role | sees | information level | specification |
|------|------|-------------------|---------------|
| **Actor** | arm inputs | L1 | this document |
| **Training critic** | both hands, pile-as-multiset, top, color, phase, drawn, pile size, reshuffles, mover. Never the pile order. | L2 | spec §23–24; `position_features` / `ValueNet` |
| **Training target** | `corrected_return(terms, seat)` = chance + opponent + deal, never the row | — | spec §28, `TRAINING` |
| **Evaluator (AIVAT)** | L2 values as `nu`; corrects chance + opponent + deal (row correction is identically zero for a keyed PIMC, and destructive for a learner; spec §28) | L2 | spec §23 |
| **Attacker (LBR, 10a.3C)** | plays sampled L1 worlds from `determinize_l1` | L1 | spec §25; the attacker is an L1 policy itself |

An actor observation is always **strictly L1**: it never reads the opponent's
private state (`opp_initial_hand`, `opp_draws`, `opp_hand`), never reads the
pile order, and never reads the seed. Checked by the invariance test of
section 6: permuting the unseen cards leaves the extracted fields unchanged.

A training critic may use L2 but must not leak it into the actor. PPO's
advantage `A = R - V(s)` uses a critic `V` fitted on L2 features; the actor's
policy gradient uses the advantage as a scalar weight, not as an input, so no
L2 information enters the actor's parameters except through its correlation
with the return.

---

## 4. Evaluator compatibility

### 4.1 Arms A and B (feed-forward)

Given a `State` or an `Observation`:

1. Extract the L1 observation.
2. Extract the arm's fields from it.
3. Forward-pass the policy network.
4. Mask to legal actions and normalize.

**`determinize_l1` compatibility.** The determinizer fabricates the opponent's
`initial_hands` and `own_draws` to be consistent with the dealt hand, and sets
a fresh seed. Arms A and B do not read any of these fields (Arm A reads the
discard, which is real; Arm B reads the public history, which is real). So
these arms are fully compatible with `determinize_l1`, and `PimcSearch` /
`PimcRollout` / `response_value` / `batched_search_q` all work unchanged.

**Exact-tree tabulation.** `tabulate(tree, Level.L1, agent)` and
`tabulate_batched` both work: they call the agent at every L1 set using
`observation_from_key`, which provides all the fields Arms A and B need.

**AIVAT.** The row correction for a keyed PIMC agent is identically zero
(spec §25). For a learned policy with a fixed network, the policy is a
deterministic function of the L1 observation, so the row correction's conditional
mean is zero and AIVAT is unbiased. No change to the AIVAT machinery is needed.

### 4.2 Arm C (recurrent)

**The problem.** A recurrent policy's output depends on the *sequence* of
events, not just the current position. `determinize_l1` fabricates the
opponent's private history. A recurrent policy that reads opponent-seat draws or
the opponent's initial hand from a determinized state would receive history that
never happened. **Therefore Arm C's input is defined to include only the
actor's own L1 information state** (section 2.3), which is always real.

**Consequence for the evaluator:**

1. **Ordinary game evaluation** (`play_pairs`, `estimate_mirrored`, `compare`):
   unchanged. The evaluator calls `policy(obs, legal)` at each of the actor's
   own decisions, where the observation carries the real history.

2. **AIVAT:** unchanged in structure. The actor reports `{a: 1.0}` if
   deterministic, or the true stochastic distribution if sampling; the row
   correction uses those probabilities. A recurrent actor that maintains hidden
   state across decisions within a game is still a deterministic function of the
   L1 information state (since the state is reconstructable from it by perfect
   recall), so AIVAT is unbiased.

3. **PIMC search / LBR attack against a recurrent target:** the attacker
   determinizes at L1 and solves each world. When the **opponent** in that world
   is Arm C, the attacker must evaluate the opponent's policy at opponent
   decisions inside the determinized world. The opponent's *public* history is
   real (shared between the real game and the determinized world). The
   opponent's *private* history (initial hand, own draws) is fabricated.

   - If Arm C reads only public events and own draws, the attacker can evaluate
     it **provided the "own draws" it reads are the opponent's fabricated
     draws**. This is the same fabricated history that `determinize_l1` already
     provides. The danger is that those draws are inconsistent with the
     fabricated initial hand, which would produce impossible recurrent states.
     `determinize_l1` fabricates `initial_hands[opp] = opp_hand` and
     `own_draws[opp] = ()`, so the recurrent input would be: initial hand =
     whatever was dealt, no draws, plus the real public history. This is
     *consistent* (the opponent starts with that hand and has drawn nothing),
     but it is not the **real** conditional policy — it is the policy the
     opponent **would** follow if they had started with that hand and drawn
     nothing. For a memoryful opponent, that conditional policy differs from
     the real one.

   - **The constraint:** a recurrent target cannot be queried using fabricated
     private history as though it were the target's real conditional policy.
     An attacker of a recurrent target must either: (a) reconstruct consistent
     private history from the determinized world (possible for draws: the drawn
     cards are known; the initial hand is the fabricated one); or (b) restrict
     the attack to the actor seat, attacking the target only from the
     responder's side where the target's own decisions use the real history.

   This is a Step 11 integration gate (per the plan, line 106-107):
   recurrent-target support must be validated before reporting attacks on
   recurrent checkpoints. It must not silently become a prerequisite for
   finishing the LBR baseline instrument (10a.3C).

4. **Exact-tree tabulation:** `tabulate` replays each L1 key's public history
   and own draws through the recurrent encoder. `observation_from_key` provides
   the full `public` tuple and `own_draws`, so the recurrent input is available.
   The cost is proportional to the average history length times the number of L1
   sets; on `TOY2_FULL` (3.75M sets, ~6 events average) this may be expensive
   but is correct.

---

## 5. What is explicitly NOT in scope

- **No new particle filter.** A particle filter is a possible later arm, not a
  dependency of this step or of 10B/10C.
- **No learner implementation.** Step 11 builds the learner. This document
  defines the inputs and invariants it must satisfy.
- **No full-UNO sufficiency claim.** Whether any arm is sufficient on full UNO
  is a Step 11 experiment.
- **K2 is not a comparison arm.** K2 is retained as a historical coarse-memory
  measurement. The review's observation that K2 "did not test the representation
  used by the successful inference experiment" stands. Arm B is a new
  representation extending the measured `voids` signal with structured evidence.
- **Architecture selection.** This document does not prescribe hidden sizes,
  learning rates, recurrent cell types, or training procedures. Those are
  Step 11 decisions, informed by exact toy experiments.

---

## 6. Invariance tests

The following tests verify the contract's requirements on field extraction. They
do not test a learner; they test that the inputs are well-defined and satisfy
the stated invariants.

1. **Legality consistency.** For each arm, the legality mask computed from the
   arm's fields equals `legal_actions(state)` on random games over multiple
   rulesets. This is already tested for K0/K1/K2 by `test_coarse.py`; the test
   extends it to the specific field sets of Arms A, B, C.

2. **L1 validity.** No arm's field extraction reads information beyond L1.
   Tested by permuting the unseen cards (opponent hand + pile) and checking
   that the extracted fields are unchanged. This parallels spec §11 property 9
   and is already tested for the `Observation` itself; the test extends it to
   each arm's specific field extraction.

3. **Color/rank equivariance.** Applying a symmetry-group element to the state
   and extracting arm fields yields the same result (after applying the group
   element to the fields) as extracting first. For Arms A/B, `position_features`
   equivariance is already tested. For the event-embedding structure of Arm C,
   the test checks that applying g to each event's fields and then embedding
   yields the same vector as embedding first and then permuting the colour
   blocks.

4. **Determinizer compatibility (Arms A/B).** Extracting arm fields from a
   `determinize_l1` state (taking the actor-seat observation) yields the same
   values as extracting from the original observation, for every field the arm
   uses.

5. **Recurrent input availability (Arm C).** On a small toy,
   `observation_from_key(rs, Level.L1, key)` provides the full `public` tuple
   and `own_draws` needed to run the recurrent encoder, and these match the
   corresponding fields from `observe(state, seat)` at the same decision.

---

## Summary table

| property | Arm A (K1 ff) | Arm B (K1 + evidence) | Arm C (recurrent) |
|----------|---------------|----------------------|-------------------|
| History | none (position + discard) | fixed-size evidence summary | full event sequence |
| Architecture | feed-forward | feed-forward | recurrent + position |
| Equivariance | by construction (DeepSets) | by construction (DeepSets) | by construction or `Symmetrized` |
| `determinize_l1` compatible | yes | yes | actor seat only; opponent seat needs consistent history |
| `tabulate` on toys | yes | yes | yes (replay history per key) |
| PIMC search as target | no constraint | no constraint | attacker must not use fabricated private history as real conditional policy |
| AIVAT | unchanged | unchanged | unchanged |
| Cost to train (relative) | 1x | ~1.1x | 2–5x (BPTT) |
| What it measures | K1 representation ceiling | value of structured draw/play evidence | value of learned sequential memory |
