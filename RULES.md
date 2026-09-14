# UNO V5 rules specification

One parametrized ruleset. The toy game and full UNO are the same rules with
different parameters. Two players, always. Everything below is what the engine
in `uno/` implements and what `tests/` checks; if code and this file disagree,
this file wins and the code is wrong.

## 1. Parameters

| field                  | meaning                                                                 | toy (`TOY`) | full (`FULL_UNO`)          |
|------------------------|-------------------------------------------------------------------------|-------------|----------------------------|
| `colors`               | number of colors, identified by ints `0..colors-1`                      | 4           | 4                          |
| `rank_copies`          | copies of each number rank per color; rank `r` has `rank_copies[r]`     | `(1, 1)`    | `(1, 2, 2, 2, 2, 2, 2, 2, 2, 2)` |
| `skip_per_color`       | skip cards per color                                                    | 0           | 2                          |
| `reverse_per_color`    | reverse cards per color                                                 | 0           | 2                          |
| `draw_two_per_color`   | +2 cards per color                                                      | 0           | 2                          |
| `wild_count`           | plain wilds in the deck                                                 | 1           | 4                          |
| `wild_draw_four_count` | wild +4 in the deck                                                     | 0           | 4                          |
| `hand_size`            | cards dealt to each player                                              | 3           | 7                          |
| `draw_then_may_play`   | after a voluntary draw, the drawn card may be played at once            | True        | True                       |
| `forced_play`          | if a playable card is in hand, drawing is illegal                       | True        | False                      |
| `reshuffle`            | `"none"` or `"seeded"`, what happens when the draw pile is empty        | `"none"`    | `"seeded"`                 |
| `max_turns`            | cap on play + draw actions; reaching it ends the game as a draw. `None` = no cap | `None`      | 1000                       |

Deck size: toy `4 * (1+1) + 1 = 9`, full `108`.

A ruleset is valid only if the deck has at least `2 * hand_size + 1` cards.

## 2. Cards and deck

A **card type** is `(kind, color, rank)` with
`kind ∈ {NUMBER, SKIP, REVERSE, DRAW_TWO, WILD, WILD_DRAW_FOUR}`,
`color ∈ 0..colors-1` for colored kinds and `None` for wilds,
`rank ∈ 0..len(rank_copies)-1` for `NUMBER` and `None` otherwise.

The deck is a **multiset** of card types. Two copies of red 3 are the same
type and are never distinguished anywhere: hands, piles, permutations and
observations are all written in type ids. This is what makes duplicate deals
collapse in the tree census.

Type ids are the index into `Ruleset.card_types`, ordered:
numbers by `(color, rank)`, then skips by color, reverses by color, +2 by color,
then `WILD`, then `WILD_DRAW_FOUR`. Only types with a positive count exist.

## 3. Chance

The only chance event is the **seed**.

- `deal(ruleset, seed)` returns a permutation of the deck multiset (a tuple of
  type ids, index 0 = top of the pile) produced by `random.Random(seed).shuffle`
  of the sorted deck.
- Reshuffle number `k` (1-based) of a game uses `random.Random(f"{seed}:{k}")`.
- Nothing else is random. Given `(permutation, seed, action sequence)` the game
  is a pure function. `initial_state(ruleset, permutation, seed)` accepts any
  permutation of the deck multiset, so a census can enumerate deals directly.
  `seed` is required when `reshuffle == "seeded"`.

## 4. Setup

1. Block deal from the top: player 0 takes cards `0..hand_size-1`, player 1
   takes `hand_size..2*hand_size-1`. Hands are kept as sorted tuples of type ids.
2. The rest is the draw pile, in permutation order.
3. First discard: pop cards from the top of the pile until a `NUMBER` card is
   found. It becomes the discard pile and `current_color` is its color. Every
   card popped before it is placed at the **bottom** of the pile, in the order
   popped. If the pile holds no number card the deal is invalid and
   `initial_state` raises `ValueError`.
4. Player 0 moves first. Phase is `TURN_START`.

The setup is public: the public history starts with the event
`("setup", top_type, skipped_types)`.

## 5. Actions

Actions are integer ids, fixed per ruleset (`Ruleset.actions`), in this order:

1. `Play(type)` for every non-wild type, in type order
2. `Play(type, color)` for every wild type and every color, type-major
3. `Draw`
4. `Pass`

So a wild play is a joint action (card, chosen color). The action space is the
same for both players and does not depend on the state; `legal_actions`
returns the subset that is legal now.

## 6. Turn structure

The player to move is `state.player`. There are two phases.

**Playable** card `c` given the top discard `t` and `current_color`:
`c` is a wild, or `c.color == current_color`, or (`c` and `t` are numbers with
the same rank), or (`c` and `t` are the same non-number colored kind, e.g. skip
on skip).

**`TURN_START`**

- legal: `Play(c)` / `Play(c, color)` for every playable `c` in hand (each
  distinct type once), plus `Draw` unless `forced_play` is set and some card in
  hand is playable.
- `Draw`: draw one card (section 8). If `draw_then_may_play` and the drawn
  card is playable, the phase becomes `AFTER_DRAW` with `drawn` set to it.
  Otherwise the turn passes to the opponent.

**`AFTER_DRAW`**

- legal: `Play(drawn)` / `Play(drawn, color)` only (that exact type, even if
  another copy of the same type or another playable card is in hand this is
  the same action id, so no distinction is needed), plus `Pass`.
- `Pass`: turn passes to the opponent. `Pass` writes **no** public event: as in
  real UNO, the opponent cannot tell "drew, could not play" from "drew, chose
  not to play". Both look like `("draw", p, 1)` followed by the opponent's turn.

## 7. Playing a card (2 players)

Remove the card from the hand, put it on top of the discard, set
`current_color` (the card's color, or the chosen color for a wild), record
`("play", player, type, current_color)`.

If the hand is now empty the player **wins immediately**, whatever the card.

Otherwise apply the effect:

| kind             | effect                                                                 |
|------------------|------------------------------------------------------------------------|
| `NUMBER`, `WILD` | turn passes to the opponent                                            |
| `SKIP`           | same player moves again                                                |
| `REVERSE`        | same player moves again (reverse acts as skip with two players)        |
| `DRAW_TWO`       | opponent draws 2 (section 8), then the same player moves again          |
| `WILD_DRAW_FOUR` | opponent draws 4, then the same player moves again                      |

No stacking, no challenge, `WILD_DRAW_FOUR` is always playable. A penalty
draw gives the drawing player no option to play.

## 8. Drawing

Cards are drawn one at a time from the top of the pile (`draw_pile[0]`).
Drawn cards are appended to the player's `own_draws` and inserted into the
sorted hand. After the whole draw (1, 2 or 4 cards) the public history gets
`("draw", player, n_actually_drawn)`.

If the pile is empty when a card must be drawn:

- `reshuffle == "seeded"` and the discard holds more than one card: the top
  discard stays, the rest is shuffled with `Random(f"{seed}:{k}")`
  (`k` = number of reshuffles so far + 1) and becomes the new pile.
  `("reshuffle",)` is recorded, then the draw continues.
- otherwise (`"none"`, or nothing to reshuffle): the game ends by
  **exhaustion** (section 9). Cards already drawn in this draw stay drawn.

## 9. End of game and returns

The game is terminal when one of:

- **win**: a player's hand is empty. That player wins.
- **exhaustion**: a draw was required and impossible. Fewer cards in hand
  wins; equal counts is a draw.
- **cap**: after a step, the number of play and draw actions taken so far
  equals `max_turns` and the game is not otherwise over. Draw. Passes are not
  counted, so the cap is a function of the public history.

`returns(state)` is zero-sum: winner gets `+1`, loser `-1`, draw `(0, 0)`.
Win probability for a player is `(r + 1) / 2`.

**Why exhaustion is decided on hand size (toy, `reshuffle == "none"`).**
`Draw` stays legal on an empty pile: it draws nothing and ends the game, so the
player to move can *claim* the game whenever they hold fewer cards, and force a
draw on equal counts. Under `forced_play` (the `TOY`) the claim only exists
when the mover has nothing playable, so the pile running out just means
"first player stuck loses on cards". This is deliberate. Official UNO scores the winner by the
cards left in the loser's hand, so "keep the hand low, do not get stuck" is
already the objective the full game rewards; the rule turns that objective into
the terminal condition once the pile runs out, without reshuffles or infinite
pass loops. It exercises the same skills as full UNO: play over draw, color
management so as not to get stuck, holding the wild as insurance.

Known distortion: the claim only exists on an *empty* pile, so whoever draws
the last card hands the claim to the opponent. Near the end, a player who
cannot play with one card left in the pile loses on the spot if they were tied
or behind. This pile-parity tactic is real, computable from public information
and therefore counts as skill on the toy, but it is toy-specific: full UNO
reshuffles, so it does not transfer. The toy exists for measurement, not for
transfer, so we accept it. Expect a large share of exhaustion endings under
strong play and read the L3 / L2 / L1 gaps with that in mind.

## 10. Information levels

`observe(state, player, level)` returns a hashable observation.

- **L1** (what a real player sees): seat, initial hand, current hand (sorted),
  pending `drawn` card (or `None`), `current_color`, top discard type,
  opponent hand size, pile size, the full public history, and `own_draws` in
  order.
  The information state proper is `(seat, initial hand, public history,
  own_draws)`; everything else is derived from it. Own draws are kept in
  order, not as a multiset, so the game has **perfect recall** (a player can
  reconstruct every past information state from the current one). CFR needs
  that.
- **L2** (medium): L1 plus the opponent's hand, carried as the opponent's
  initial hand and draws in order (the current hand is derived from them and
  the public history). This is "the opponent's hand at every one of my
  decisions" for an observer with memory, and it keeps perfect recall: keying
  on the current hand alone would not, since the observer could no longer tell
  which card the opponent drew.
- **L3** (oracle): L2 plus the draw pile in order and the seed. This
  determines the state exactly (`reconstruct(ruleset, obs) == state`).

An L1 or L2 observation is only meaningful at the observer's **own decision
nodes**. Observing a state where the opponent is to move mid-turn reveals that
their drawn card was playable (otherwise the turn would already have passed),
and no field masking can hide that: it is the timing of the observation that
leaks, not a field. Agents observe at their own decisions, solvers key on
`info_state`, and the evaluator's value functions may use the full state, so
nothing needs the other case.

Public events, in the order they happen:
`("setup", top, skipped)`, `("play", player, type, color)`,
`("draw", player, n)`, `("reshuffle",)`.

## 11. Invariants (property tests)

Checked at every step of random games over random small rulesets:

1. **Conservation**: multiset of hands + pile + discard equals the deck.
2. **Legality**: every legal action applies without error; every illegal
   action id raises `IllegalAction`.
3. **Determinism**: replaying `(ruleset, permutation, seed, actions)` gives an
   equal state at every step.
4. **Immutability**: `step` never changes its input state.
5. **Termination**: with `reshuffle == "none"` the game ends within
   `2 * (2 * |pile0| + |hands0|) + 1` steps (every non-pass step lowers
   `2 * |pile| + |hands|` by at least 1 except the final draw from an empty
   pile, and passes never outnumber draws);
   with `"seeded"` it ends within `2 * max_turns + 1` steps (at most one pass
   per draw).
6. **Terminal correctness**: `win` implies the winner's hand is empty and the
   other is not; `exhaustion` implies the pile is empty and the winner has
   strictly fewer cards (or `None` on equal counts); `cap` implies the play +
   draw count equals `max_turns`.
7. **Progress**: a non-terminal state has at least one legal action.
8. **Color**: `current_color` equals the top discard's color when the top is
   not a wild.
9. **Information**: the L1 observation of a player is unchanged when the
   unseen cards (opponent hand + pile) are permuted among themselves; the L2
   observation is unchanged when the pile is permuted; the L3 observation
   determines the state.
10. **Perfect recall**: initial hand + own draws - own plays equals the current
    hand, as multisets.

## 12. Named rulesets

- `TOY`: 4 colors, ranks `(1, 1)` (one copy each), no action cards, 1 wild,
  hand 3, `draw_then_may_play`, **forced play**, no reshuffle, no cap. 9 cards,
  draw pile of 2 after the deal. Chosen for low per-card playability: with 2
  colors x 2 ranks x 2 copies nobody ever got stuck and seat 0 won 72 % of
  random games on pure tempo; here seat 0 wins 59 % of random games outright
  (score 0.65 with ties worth 1/2). Sizes in notes.md (step 3).
- `TOY_FULL`: same as `TOY` without forced play, so "draw with a play in hand"
  is a decision. More than 2e7 histories (census stopped at the cap), so it is
  a name for later, not a working size. The value gap between `TOY_FULL` and
  `TOY` best responses would be the worth of that decision.
- `TOY_CANDIDATE`: the original 13-card candidate (2 colors, ranks `(2, 2, 2)`,
  1 wild, hand 2, no forced play). Rejected by the census: more than 4e7
  histories. Kept as a name only.
- `FULL_UNO`: 4 colors, ranks `(1, 2, 2, 2, 2, 2, 2, 2, 2, 2)`, 2 skip / 2
  reverse / 2 draw-two per color, 4 wild, 4 wild draw four, hand 7,
  `draw_then_may_play`, no forced play, seeded reshuffle, cap 1000. 108 cards.

## 13. Baseline agents (`uno/agents.py`)

An agent is a function `policy(observation, legal_actions) -> {action: prob}`
whose support is inside the legal actions and whose probabilities sum to 1.
Agents return distributions, never sampled actions, because best response
tabulates the policy and AIVAT needs the action probabilities. Sampling is done
by the runner (`uno/play.py`) with an rng seeded from `(match seed, agent name,
information state)`, so the same agent makes the same choice in the same
situation across games and opponents (common random numbers). The agent name
is part of the key so that two different stochastic agents never share a
stream: in a mirrored pair (section 15) they sit at the same seat in the two
games and would otherwise make the same random choices in the same
information states, coupling the games beyond the shared deal.

- `UniformLegal`: uniform over every legal action, including `Draw` with a play
  in hand and `Pass` after a playable draw. Sanity opponent only.
- `PlayFirstRandom`: uniform over the legal plays; `Draw` only when nothing
  plays; after a draw always plays the drawn card, uniform over the colors for a
  wild. **This is the ladder's "Random" row.**
- `Heuristic`: a rulebot, **symmetric** under color and rank renaming
  (section 18) since step 8.6. Play over draw. Among playable non-wild cards:
  action cards first when the opponent holds at most 2 cards, then the color
  the player holds most, then a card of the current color (the color stays),
  then a number card; what still ties is played uniformly at random. Wilds are
  held unless no other card plays; a wild takes the color held most (the
  current color on ties, then uniform). After a draw, always play the drawn
  card. Up to step 8.5 ties were broken on the lowest type id, which is not
  symmetric; the old values are in notes.md steps 4-8.5, the new ones from
  step 8.6 on.

## 14. Exact solvers (`uno/solve.py`)

Only for `reshuffle == "none"` rulesets (they walk the lazy-chance tree of
`uno/tree.py`). Values are the responder's expected return in `[-1, 1]`;
`score = (value + 1) / 2` is P(win) with draws worth 1/2, the same quantity
`uno.play.match` estimates.

- `evaluate(rs, agents, levels)`: exact expected returns of two fixed agents
  over the whole game. `evaluate_on_seeds(rs, agents, seeds, levels)`: the
  same on the real deal and real pile order of every seed, i.e. exactly what a
  simulation on those seeds estimates up to action sampling. **A fixed seed
  range carries its own deal luck**: seeds 0-4000 on `TOY` favour seat 0 by
  about 1.5 points, so a simulation is checked against the exact value on the
  same games, not against the global value.
- `best_response(rs, opponent, seat, level)`: the responder's best pure
  strategy against a fixed opponent (seen at L1), at information level:
  - **L1**: information set = `info_state`.
  - **L2**: `info_state` plus the opponent's initial hand and draws in order
    (section 10). Perfect recall, like L1.
  - **L3**: the full state, pile order included; every set is one history.
    L3 roots are *worlds* (every deal x every order of the unknown pile) and the
    tree below a world is the bare engine, with no chance node.

  Two passes: forward, collecting the counterfactual reach of every history
  (deal x chance x opponent, never the responder) and grouping the responder's
  histories by information set; backward, memoised values where each
  information set picks the action maximising the reach-weighted sum of its
  histories' continuation values (ties: lowest action id). This is a best
  response because every level has perfect recall (section 10); the recursion
  is well founded because the public history grows between two decisions of
  the same player. The forward pass asserts that all histories of a set have
  the same legal actions. Only sets reachable under the opponent's support are
  solved and tabulated.
- `TabularAgent(rs, br)`: plays the table through `uno.play` at `br.level`.
  Unreachable sets fall back to the first legal action and are counted.

Checks (`tests/test_solve.py`): on tiny rulesets the value equals the maximum
over every pure strategy of the responder, and the L3 worlds equal the set of
distinct deck permutations with their frequencies; on small rulesets the table
replayed through `evaluate` gives the solver's value, values are monotone in
information (L1 <= L2 <= L3), dominate Random / Heuristic / UniformLegal as
responder, coincide across levels on a ruleset with no draw pile, and the
simulated score matches the exact value on the same deals.

## 15. Estimators (`uno/estimate.py`)

Play once, estimate many ways. `play_pairs(rs, agents, seeds, levels)` plays
every seed from both seats (the first-named agent at seat 0, then at seat 1,
same deal and pile order) and keeps the traces. Every estimator is a pure
function of those records, so AIVAT (step 9) reads the same games. The score
of a game is the first-named agent's `(return + 1) / 2`.

- `estimate_mirrored(pairs)`: the seat-0 and seat-1 means (n games each,
  normal 95 % CI); the seat mean with the **raw** CI (the 2n games treated as
  independent, variance `(var0 + var1) / 4` per pair) and with the
  **mirrored** CI (variance of the n pair scores `(s0 + s1) / 2`, i.e.
  `(var0 + var1 + 2 cov) / 4`); the correlation between the two seats' scores
  on the same deal, the share of split pairs and the variance ratio
  mirrored / raw. The whole gain is the negative covariance: a deal that
  favours seat 0 does so whoever sits there. Per-seat estimates gain nothing
  from pairing and are always reported: the pair score alone pools seats
  (step 3 rule 1).
- `compare(a, b)`: paired difference per seat, over deals, of two row agents
  that played the same seeds against the same column, with the unpaired CI
  alongside. **Common random numbers**: the runner keys the action rng on
  (seed, information state), so the column agent makes the same choices in
  both matchups until the public histories diverge.
- Self-play pairs are degenerate: under common random numbers the two games
  of an A-vs-A pair are the same game, so every pair scores exactly 1/2.
- `exact_mirrored(rs, agents, levels)`: exact means, variances, covariance
  and split share over the worlds of `uno.solve.worlds` (deal x pile order),
  from `outcome_distribution` on each world. Exact for deterministic agents;
  for stochastic ones it takes the two games of a pair as independent given
  the world, which holds because the rng streams are keyed on the agent name
  and the information state (section 13): no two agents, and no agent at two
  seats, share a stream.
- `games_needed(edge, var_per_game, z)` = `z² var / edge²`, with the per-game
  variance of the design (a pair is two games).
- `uno.solve.outcome_distribution`: exact distribution of seat 0's return
  under two fixed agents, `{+1, 0, -1} -> probability`; `evaluate` is its mean.
- Seed ranges (`uno.play`): `DEV_SEEDS` = 0..3999 for building, testing and
  tuning; `EVAL_SEEDS` = 1,000,000..1,003,999 held out for the ladder's
  reported numbers, nothing is ever chosen on it.

Checks (`tests/test_estimate.py`): pair traces replay; naming the agents the
other way round plays the same two games scored from the other side; self-play
pairs are identical games; `outcome_distribution` sums to 1 with the
`evaluate` mean; `exact_mirrored` means equal `evaluate` (worlds + bare engine
against the lazy tree); on `TOY` (Heuristic vs Random, 4000 seeds) and on the
7-card test ruleset (BR L1 vs Heuristic, deterministic) the sampled per-seat
means match the exact on-seeds values (to the last digit for deterministic
pairs, within the CI otherwise) and the sampled variances, correlation and
split share match the exact ones (10 %, 0.06, 0.03); exact and sampled
variance ratios are below 1; paired differences of deterministic agents equal
the difference of the on-seeds exact values and never exceed the unpaired
variance.

## 16. Compiled tree (`uno/compiled.py`)

The lazy-chance tree of section 14 (or, with L3, the world tree: one root per
deal x unknown-pile order, bare engine below) walked **once** into flat numpy
arrays. Every exact computation of sections 14-15 is then a vectorized pass,
and CFR (section 17) becomes affordable. The walkers of `uno/solve.py` and
`uno/estimate.py` stay as the reference implementation the compiled versions
are tested against.

By default (`symmetric=True`, step 8.6) the tree is the **quotient** under
the color / rank symmetry of section 18: one root per orbit of deals (or of
worlds) carrying the orbit's probability, and one information set per orbit
of keys. `symmetric=False` builds the full tree. `cache=<directory>` loads
the tree from `tree_<ruleset>_<levels>_<sym|full>.npz` if present and saves
it there otherwise (`CompiledTree.save` / `load`; the file holds every array
and the encoded keys, the `Symmetry` is rebuilt).

Layout:

- **nodes** = histories plus a virtual chance root (node 0) whose edges lead
  to the deals (or the worlds). Per node: player (0 / 1, terminal, chance),
  level, seat-0 return at terminals, end kind at terminals (`end`: win,
  exhaustion, cap; `end_columns` gives the indicators for `values`).
- **arcs** = (decision node, legal action), one per legal action, in action
  order. **edges** = (arc, chance outcome): one per child with its
  probability (a draw has one edge per distinct card type left in the pile).
- Nodes are sorted by level (terminals last within a level), arcs by node,
  edges by arc, so each level is a contiguous slice of the three tables.
- **level** of a decision node = `2 * len(public) + [phase == TURN_START]`.
  It increases strictly along every edge (play and draw append a public
  event; a pass moves from `AFTER_DRAW` to `TURN_START`) and all histories of
  an information set share it, at every information level, because the
  phase at a player's own decision is a function of what they know. Depth by
  action count would not do: the opponent's "drew a playable card and passed"
  is one action longer than "drew an unplayable card" and looks the same.
  Terminals get their parent's level plus one. The build asserts the
  same-level and same-legal-actions properties of every set.
- A **keying** is one information-set partition (L1, L2 or L3 key of section
  14) over the tree: set id per node, set-action ("ia") id per arc, and the
  key -> id dictionary for playback and tabulation. Keys are stored
  **encoded as bytes** (`uno.solve.encode_key` / `decode_key`, lossless, a
  few dozen bytes each) instead of nested tuples: on the full `TOY` tree the
  key dictionaries were ~90 % of 1.1 GB. Sets are numbered in order of first
  appearance, hence contiguous per level; a set's ia are contiguous. A
  strategy is a vector over ia; a tree can hold several keyings (L1 + L2 on
  the lazy tree, L1 + L3 on the world tree). On a symmetric tree the key of a
  set is the canonical key, its actions are the legal actions of the
  canonical representative, and each arc's ia is found through the element
  `g` that canonicalises its node's key (`g . action`); the keying also holds
  the **action orbits** of every set under the stabilizer of its canonical
  key (`ia_orbit`, `orbit_size`, section 18).

Kernels:

- `reach(w0, w1)`: forward by level; per node the product along the path of
  player 0's weights, player 1's weights and chance, `(n, 3)`.
- `values(w, util)`: backward by level; per node the expectation of `util`
  (any number of terminal columns) under per-arc weights `w`, plus the
  per-arc values. Both are a loop over the levels of a few `reduceat` /
  fancy-indexing operations.

On top of them: `evaluate` and `outcome_distribution` (the profile's weights
and the seat-0 return, or the three outcome indicators); `best_response(tree,
level, opponent, seat)` (counterfactual reach = chance x opponent from
`reach`, then backward from the deepest level choosing per set the action
maximising the reach-weighted sum of its histories' arc values; ties to the
lowest action id, as section 14; on a symmetric tree the choice is an action
*orbit*, played uniformly, section 18); `exploitability(tree, level, sigma)` = both
best responses to a profile, NashConv in return units and in score points
(`score(BR seat 0) + score(BR seat 1) - 1`), the per-seat gains and the
bracket `[1 - score(BR seat 1), score(BR seat 0)]` that contains the Nash
seat-0 score; `exact_mirrored` (section 15) from the per-world outcome
distributions of the two seat-swapped profiles on a world tree.

Strategies from agents: `tabulate(tree, level, agent)` asks the agent at
every set of the keying, both seats, with the observation rebuilt from the
set's key by `uno.observe.observation_from_key` (the key is by definition
everything the observer knows, section 10; the L3 key is the state). On a
symmetric tree it raises if the agent is not symmetric (its policy is not
uniform within some action orbit). `tabulate_table` maps a
`BestResponse.table` onto a keying (on a symmetric tree the chosen action is
spread over its orbit). `CompiledAgent` plays any strategy over any keying
through `uno.play`; on a symmetric tree it canonicalises the observation's
key, looks the canonical set up and maps the actions back with the inverse
element.

Checks (`tests/test_compiled.py`, `tests/test_observe_keys.py`): the
observation rebuilt from an L1 / L2 / L3 key equals `observe` at every
decision of random games over random rulesets; node, terminal, decision and
set counts equal the census; the end-kind mass under the uniform profile equals
the census end mass; levels increase along edges; the uniform
profile's value equals the census UniformLegal score; on small rulesets the
compiled best response at every level equals the reference value to 1e-9 for
UniformLegal, Random and Heuristic opponents, its pure strategy replays to
that value both on the compiled tree and through the engine, the reference
table mapped onto the keying has the reference value, and L1 on the world
tree equals L1 on the lazy tree; evaluate, outcome distributions and exact
mirrored moments match the reference on a 7-card ruleset. On `TOY` the
twelve step 5-6 values are reproduced (`scripts/cfr.py`). The symmetric
tree's checks are in section 18.

## 17. CFR (`uno/cfr.py`)

Nash L1: counterfactual regret minimisation on the L1 keying of the compiled
tree, both seats at once. Iteration `t`, for player `i`:

1. `reach` and `values` under the current profile;
2. counterfactual regret per set-action: sum over the set's histories of
   (chance x opponent reach) x (arc value - history value), signed for the
   player;
3. regret matching: current strategy proportional to positive accumulated
   regrets, uniform where none. **CFR+** clips the accumulated regrets at
   zero after each update;
4. average strategy: accumulate own reach x current strategy, weighted by `t`
   (linear averaging, CFR+) or 1 (vanilla).

On a symmetric tree (section 18) the regret and the average-strategy mass
are averaged within every action orbit at each update, so every strategy
stays symmetric: this is CFR on the game whose actions are the orbits, and
its equilibrium is a symmetric Nash equilibrium of the full game.

`alternating` (CFR+) updates player 0, recomputes the passes, then updates
player 1; vanilla updates both from one pass.

**Variants** (step 8.7, notes_toy2.md), same passes, `cfr(variant=...)`:

- discounting (DCFR, Brown & Sandholm 2019): after iteration `t` the
  accumulated positive regrets are multiplied by `t^a / (t^a + 1)`, the
  negative ones by `t^b / (t^b + 1)`, and the contribution of iteration `t`
  to the average strategy carries weight `t^g`. CFR+ is `a = inf` (no
  discount), negatives clipped, `g = 1`; DCFR is `(1.5, 0, 2)`;
- prediction (PCFR+, Farina, Kroer & Sandholm 2021): the current strategy is
  regret matching on the accumulated regrets *plus the last instantaneous
  regret* (the optimistic step), with regret matching+ and `g = 2`.

Presets in `uno.cfr.VARIANTS`: `vanilla`, `cfr+`, `dcfr`, `pcfr+`, and
`dpcfr+` (prediction with DCFR's positive-regret discount). The orbit
averaging applies to every variant, so all of them stay symmetric.

The result is the normalised
average strategy (sets never reached by their owner are uniform), the
regrets, and a log of exploitability (section 16) at chosen iterations, whose
last entry (iteration count and NashConv) goes next to every ladder cell that
uses the strategy. `CFRResult.save` / `load` persist it (`.cache/`, not
committed).

Checks (`tests/test_cfr.py`): on the 7-card ruleset CFR+ reaches NashConv
under 0.2 score points in 60 iterations and the log decreases; the profile's
value sits inside its bracket at every logged iteration; the reference best
response to the average strategy played as an agent equals the compiled
exploitability; the profile's value through the engine equals the compiled
one; every row agent's best-response ceiling sits above the bracket (its
exploitability is non-negative); the simulation of the Nash agent on seeds
matches the exact value on those seeds; save / load round-trips; vanilla CFR
converges more slowly and brackets the same value; on the quotient of `TOY`
every variant's NashConv falls, every average strategy is symmetric, all
brackets intersect at the Nash value 0.6196, vanilla > CFR+ > DCFR and CFR+
> PCFR+ in NashConv after 60 iterations, and the explicit flags reproduce
the preset.

## 18. Symmetry (`uno/symmetry.py`)

The rules never name a particular color or rank. Renaming the colors (any
permutation) and the ranks (any permutation among ranks with the same
number of copies) maps deals to equally likely deals, legal actions to legal
actions and returns to themselves; wilds are fixed. The game is invariant
under the group G = S_colors x (rank permutations within copy classes):
|G| = 48 on `TOY`, 144 with three one-copy ranks, 240 with five colors.

`Symmetry(rs)` holds the group as lookup tables (`types[g, t]`, `colors[g,
c]`, `actions[g, a]`, `inverse[g]`; element 0 is the identity) and its
action on hands, public histories, keys of every level, states and policies.
`canonical(level, key)` returns the lexicographic minimum of the key over G
and an element `g` reaching it (`g . key == canonical`); the search is
pruned on the observer's initial hand(s), the first key component after the
seat, with the candidate elements cached per hand. `stabilizer(level,
canonical_key)` lists the elements fixing a canonical key and
`action_orbits` partitions a set's actions under it.

**Quotient.** `initial_deals(rs, symmetric=True)` / `worlds(rs,
symmetric=True)` keep one representative per orbit (the canonical state
itself) with the orbit's total probability; for deals the unknown part of
the pile is a multiset and is ignored by the orbit (sorted), for worlds the
pile order counts. `census(rs, symmetric=True)` walks that quotient tree and
counts canonical L1 sets. The compiled tree (section 16) does the same and
keys its sets on canonical keys. On `TOY`: 5,600 deals -> 144 orbits,
1,252,040 -> 33,646 histories (37x), 558,656 -> 12,036 L1 sets (46x).

**Exactness.** A policy is symmetric when it commutes with the group,
`policy(g . obs)(g . a) = policy(obs)(a)`. Values computed on the quotient
equal the full game's values whenever every policy involved is symmetric:
the subtree below a deal and below its image are isomorphic, so one
representative weighted by the orbit's mass gives the same expectation, and
a quotient node weighted by its root's orbit mass stands for exactly the
full-game histories in its orbit. `UniformLegal`, `PlayFirstRandom` and the
`Heuristic` (section 13) are symmetric. An RL agent is made symmetric by
canonicalising its observation and mapping its action back with the inverse
element (`canonical`, `map_policy`), which is also a |G|-fold gain in data
efficiency; a raw, asymmetric agent needs the full tree (`symmetric=False`),
and symmetrising a policy (averaging its images under G) never raises its
exploitability, because the best-response value is convex in the opponent's
strategy and every image has the same exploitability.

**Orbits.** A symmetric strategy is, at a canonical set whose key has a
non-trivial stabilizer, uniform within every orbit of the set's actions
under that stabilizer (a hand `{red 1, blue 1, W}` on a green top is fixed by
swapping red and blue, so "play red 1" and "play blue 1" must get equal
probability). This matters for anything that *chooses* on the quotient: a
pure best response picking one action of an orbit would, once lifted through
each node's own `g`, play different real actions at histories of the same
real information set and score above the true best response (0.8065 against
0.7934 for BR L1 seat 0 vs Random on `TOY`, found while building this). So
`best_response` chooses the best orbit (mean of its actions' reach-weighted
values) and plays it uniformly, which attains the full game's best-response
value against a symmetric opponent (a symmetric best response always exists:
average the images of any best response), and CFR averages regrets and
average-strategy mass within orbits (section 17). `tabulate` refuses an
agent whose tabulated policy is not orbit-uniform.

Checks (`tests/test_symmetry.py`): on random games over random rulesets,
`g . step(s, a) == step(g . s, g . a)`, legal actions, returns and every
level's keys commute with the action, inverses invert, keys round-trip
through `encode_key` / `decode_key`, canonical forms are invariant
(`canonical(g . k) == canonical(k)`), reached by their element, fixed by
their stabilizer and minimal; the three baseline agents commute with the
group; on small rulesets the quotient census has the full census's end mass
and UniformLegal score; the quotient tree gives the full tree's `evaluate`
for every pair of baseline agents, the full tree's best-response values at
L1 / L2 / L3 against each of them (with symmetric strategies that replay to
their value on the quotient and through the engine in the real game, no
misses) and the same exploitability of the uniform profile; `tabulate`
rejects an asymmetric agent on `TOY`; a tree survives save / load and the
cache; on the quotient of `TOY` the twelve step 5-6 / 8.6 best-response
values and the Nash value 0.6196 are reproduced.

## 19. Sizing by position (`uno/sizing.py`)

`census` (section 16 and `uno/tree.py`) visits every history, so counting a
candidate toy costs as much as walking it and a cap hides everything above
it. But the subtree below a history is a function of its **position** only:
the two hands, the pile (the unknown multiset and the known tail), the top
card and current color, the player, the phase and the pending drawn card
(plus the turn count when `max_turns` is set). The public history, the
initial hands and the draw records feed the information-set keys, not the
transitions. So every count that never mentions an information set is a
function of the position and the tree is sized by a dynamic program
memoised over positions: `size(rs, symmetric, world, position_cap)` returns
the number of roots (deals or worlds, or their orbits), positions, histories,
decision nodes, terminals, the maximal depth, and, under two history-free
policies (UniformLegal, the census's policy, and PlayFirstRandom, the ladder's
Random row), the exact end distribution and the expected number of decisions
per game. `random_p0_score` is the exact random-vs-random seat-0 score, the
seat-parity readout of the census rounds. `world=True` sizes the L3 world
tree (the bare engine over ordered piles). `position_cap` stops the walk
(~0.5 KB per memoised position) and marks the result incomplete; the
histories are then a lower bound and the masses are dropped.

What it cannot count: L1 / L2 information sets, which depend on the history;
they come out of `compile_tree` once a candidate is chosen. On `TOY` the DP
visits 12,737 positions for 33,646 quotient histories (2.6x) and 174,952 for
the 1,252,040 full histories (7x); on `TOY_FULL` 49,177 for 774,007 (16x).
The ratio grows with the tree, which is what makes the 13-card candidates
countable.

Checks (`tests/test_sizing.py`): on small rulesets, full and quotient, every
count and the UniformLegal end mass equal the census's; the PlayFirstRandom
end mass reproduces the compiled tree's exact random-vs-random value and the
compiled node count; the world tree has the same values and depth; the cap
stops the walk; `TOY` and `TOY_FULL` sizes and values of step 8.6 are pinned.
