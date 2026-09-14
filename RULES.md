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
by the runner (`uno/play.py`) with an rng seeded from `(match seed, information
state)`, so the same agent makes the same choice in the same situation across
games and opponents (common random numbers).

- `UniformLegal`: uniform over every legal action, including `Draw` with a play
  in hand and `Pass` after a playable draw. Sanity opponent only.
- `PlayFirstRandom`: uniform over the legal plays; `Draw` only when nothing
  plays; after a draw always plays the drawn card, uniform over the colors for a
  wild. **This is the ladder's "Random" row.**
- `Heuristic`: deterministic. Play over draw. Among playable non-wild cards,
  prefer the color the player holds most (ties: lowest type id); action cards
  first when the opponent holds at most 2 cards. Wilds are held unless no other
  card plays; a wild is played on the color held most. After a draw, always
  play the drawn card.

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
