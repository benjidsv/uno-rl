# UNO V5 rules specification

One parametrized ruleset. The toy game and full UNO are the same rules with
different parameters. Two players, always. Everything below is what the engine
in `uno/` implements and what `tests/` checks; if code and this file disagree,
this file wins and the code is wrong.

## 1. Parameters

| field                  | meaning                                                                 | toy (`TOY_CANDIDATE`) | full (`FULL_UNO`)          |
|------------------------|-------------------------------------------------------------------------|-----------------------|----------------------------|
| `colors`               | number of colors, identified by ints `0..colors-1`                      | 2                     | 4                          |
| `rank_copies`          | copies of each number rank per color; rank `r` has `rank_copies[r]`     | `(2, 2, 2)`           | `(1, 2, 2, 2, 2, 2, 2, 2, 2, 2)` |
| `skip_per_color`       | skip cards per color                                                    | 0                     | 2                          |
| `reverse_per_color`    | reverse cards per color                                                 | 0                     | 2                          |
| `draw_two_per_color`   | +2 cards per color                                                      | 0                     | 2                          |
| `wild_count`           | plain wilds in the deck                                                 | 1                     | 4                          |
| `wild_draw_four_count` | wild +4 in the deck                                                     | 0                     | 4                          |
| `hand_size`            | cards dealt to each player                                              | 2                     | 7                          |
| `draw_then_may_play`   | after a voluntary draw, the drawn card may be played at once            | True                  | True                       |
| `forced_play`          | if a playable card is in hand, drawing is illegal                       | False                 | False                      |
| `reshuffle`            | `"none"` or `"seeded"`, what happens when the draw pile is empty        | `"none"`              | `"seeded"`                 |
| `max_turns`            | cap on play + draw actions; reaching it ends the game as a draw. `None` = no cap | `None`                | 1000                       |

Deck size: toy `2 * (2+2+2) + 1 = 13`, full `108`.

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
- **L2** (medium): L1 plus the opponent's hand (sorted).
- **L3** (oracle): L2 plus the draw pile in order, the seed, and the
  opponent's initial hand and draw order. This determines the state exactly
  (`reconstruct(ruleset, obs) == state`).

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

- `TOY_CANDIDATE`: 2 colors, ranks `(2, 2, 2)`, no action cards, 1 wild,
  hand 2, `draw_then_may_play`, no forced play, no reshuffle, no cap. 13 cards.
  A candidate only; step 3 (census) decides the final toy.
- `FULL_UNO`: 4 colors, ranks `(1, 2, 2, 2, 2, 2, 2, 2, 2, 2)`, 2 skip / 2
  reverse / 2 draw-two per color, 4 wild, 4 wild draw four, hand 7,
  `draw_then_may_play`, no forced play, seeded reshuffle, cap 1000. 108 cards.
