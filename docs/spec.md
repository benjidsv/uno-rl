# Specification

One parametrized ruleset. The toy game and full UNO are the same rules with
different parameters. Two players, always. Everything below is what the engine
in `uno/` implements and what `tests/` checks; if code and this file disagree,
this file wins and the code is wrong. Measured numbers are not part of the spec: they are in [results/](results/index.md), one file per step.

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

One more field, not a rule of the game: `draws` (`"pile"` by default, or
`"tickets"`) says how a drawn card is chosen (section 8). Both give the same game;
tickets only change which card a seed hands out. It is left out of the ruleset's
repr, which names cached trees, and the exact trees refuse `"tickets"`.

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

The exact walkers (`uno.game.chance`) do not enumerate permutations: the deal
is a chance node over (hand multisets, setup prefix), and n cards dealt from a
multiset with counts `c_t` (N cards) form the multiset `m` with probability
`prod_t C(c_t, m_t) / C(N, n)` (multivariate hypergeometric). Checked against
every permutation of 7-card decks with duplicate cards, at any hand size
(`tests/reference/test_census.py`). Before step 8.9 this was wrong for hands of
3 or more cards from a deck with duplicates; no named toy has duplicates.

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

**Ticket draws** (`draws == "tickets"`, step 9, for evaluation). The card is not
the top of the pile. For player p's k-th draw of the game (k = cards p has drawn
so far), every physical card of the deck gets an independent uniform ticket from
`Random(f"{seed}:t{p}:{k}")` (the i-th copy of a type holds that type's i-th
ticket), and the card of the pile's unknown part with the lowest ticket is drawn
(`uno.game.engine.ticket_index`). The unknown part excludes the cards skipped at
setup, which stay at the bottom in their public order and are drawn last, as with
the pile; after a reshuffle the whole pile is unknown. The lowest ticket among
the cards present is equally likely to be any of them, and a player's k-th
tickets are independent of everything before, so every draw is uniform over the
unknown part given the history: the same game as the top of a shuffled pile, at
L1 and L2. What changes is the coupling across games on one seed: two games give
a player the same k-th card unless their piles differ in the card holding the
lowest ticket, however their draw counts parted earlier, where the pile shifts
every later draw by one card per extra draw. There is no pile order, so the L3
worlds (pile order known) are not defined: the lazy tree and `worlds` raise.
A game with tickets is still a deterministic function of (permutation, seed,
actions), so the on-seeds exact values of section 14 work unchanged.

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

`uno.game.rules.NAMED` holds every named ruleset; the cache (`uno.store`) names
its files after them. Sizes are the quotient tree of section 18 (the full tree
in brackets where it was compiled); why each was chosen is in the results.

| name | cards | hand | forced play | deck / pile | histories | L1 sets | status |
|------|-------|------|-------------|-------------|-----------|---------|--------|
| `TOY` | 4 colors x ranks `(1, 1)`, 1 wild | 3 | yes | 9 / 2 | 33,646 (1,252,040) | 12,036 (558,656) | solved, [step 3](results/03-census.md) |
| `TOY_FULL` | as `TOY` | 3 | no | 9 / 2 | 774,007 | 138,876 | solved, [step 8.6](results/08.6-symmetry.md) |
| `TOY2` | 3 colors x ranks `(1, 1, 1)`, 1 wild | 3 | yes | 10 / 3 | 311,819 | 129,428 | solved, [step 8.7](results/08.7-toy2.md) |
| `TOY2_FULL` | as `TOY2` | 3 | no | 10 / 3 | 28,057,516 | 3,753,530 | solved (Nash by DCFR), step 8.7 |
| `TOY2_SKIP` | 3 colors x ranks `(1, 1)`, 1 skip per color, 1 wild | 3 | yes | 10 / 3 | 922,744 | 439,268 | solved, step 8.7 |
| `TOY_CANDIDATE` | 2 colors x ranks `(2, 2, 2)`, 1 wild | 2 | no | 13 / 8 | > 4e7 (full, census capped) | | rejected, step 3 |
| `FULL_UNO` | 4 colors x ranks `(1, 2, 2, 2, 2, 2, 2, 2, 2, 2)`, 2 skip / 2 reverse / 2 draw-two per color, 4 wild, 4 wild draw four | 7 | no | 108 | | | sampled only |

Every toy has no reverse or draw-two cards, `draw_then_may_play`, no reshuffle
and no turn cap. `FULL_UNO` has `draw_then_may_play`, a seeded reshuffle and a
cap of 1000.

## 13. Baseline agents (`uno/agents/baseline.py`)

An agent is a function `policy(observation, legal_actions) -> {action: prob}`
whose support is inside the legal actions and whose probabilities sum to 1.
Agents return distributions, never sampled actions, because best response
tabulates the policy and AIVAT needs the action probabilities. Sampling is done
by the runner (`uno/eval/play.py`) with an rng seeded from `(match seed, agent name,
information state)`, so the same agent makes the same choice in the same
situation across games and opponents (common random numbers). The agent name
is part of the key so that two different stochastic agents never share a
stream: in a mirrored pair (section 15) they sit at the same seat in the two
games and would otherwise make the same random choices in the same
information states, coupling the games beyond the shared deal. So a name must
identify a policy: `play_pairs` raises when two different agent objects share a
name, and self-play passes the same object twice (step 8.9). The stream is keyed
on the name rather than on the object so that runs reproduce across processes
and the same policy built twice (two experiments, two loads of a checkpoint)
still gets common random numbers. An agent may set `stream` to key its rng on
another name (`uno.eval.play.stream_name`): two variants of one policy (two
checkpoints, `Mixture(heuristic, random, 0.05, stream="heuristic")`) then draw
the same random numbers in the same information state, which `compare` wants;
`play_pairs` also raises when the two agents of a pair share a stream (step 9).

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
  card. (Before [step 8.6](results/08.6-symmetry.md) ties went to the lowest type id, which is not symmetric.)

## 14. Exact solvers (`uno/reference/solve.py`)

Only for `reshuffle == "none"` rulesets (they walk the lazy-chance tree of
`uno/game/chance.py`). Values are the responder's expected return in `[-1, 1]`;
`score = (value + 1) / 2` is P(win) with draws worth 1/2, the same quantity
`uno.eval.play.match` estimates.

- `evaluate(rs, agents, levels)`: exact expected returns of two fixed agents
  over the whole game. `evaluate_on_seeds(rs, agents, seeds, levels)`: the
  same on the real deal and real pile order of every seed, i.e. exactly what a
  simulation on those seeds estimates up to action sampling. **A fixed seed
  range carries its own deal luck**: seeds 0-4000 on `TOY` favour seat 0 by
  about 1.5 points, so a simulation is checked against the exact value on the same games, not against the global value. `scores_on_seeds(rs, agents, seeds, levels)` returns the per-seat score means on those games and the mean over the seeds of Var(score | seed), the action noise of such a simulation; the check is z = (sim - mean) / sqrt(cond_var / n) with |z| <= 3 (0.27 % false alarms per check), and exact equality for two deterministic agents (zero conditional variance). A CI of the simulated score is the wrong scale for this: it also counts the deal and pile variance that the seeds fix, so a check against it is conservative and a miss under it more significant than it looks (step 8.8). The on-seeds values start from real deals and never use the chance model's deal probabilities, so no simulation check sees a wrong one (the step 8.9 bug passed them all); `OnSeeds.deal_var` (the variance over the seeds of the per-seed exact score) and `z_global(exact, seat)` = (mean over the seeds - whole-game exact score) / sqrt(deal_var / n) check the chance model against real deals (step 9).
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
- `TabularAgent(rs, br)`: plays the table through `uno.eval.play` at `br.level`.
  Unreachable sets fall back to the first legal action and are counted.

Checks (`tests/reference/test_solve.py`): on tiny rulesets the value equals the maximum
over every pure strategy of the responder, and the L3 worlds equal the set of
distinct deck permutations with their frequencies; on small rulesets the table
replayed through `evaluate` gives the solver's value, values are monotone in
information (L1 <= L2 <= L3), dominate Random / Heuristic / UniformLegal as
responder, coincide across levels on a ruleset with no draw pile, and the simulated score matches the exact value on the same deals (|z| <= 3 against the action noise); `scores_on_seeds` has zero conditional variance for two deterministic agents, their simulation matches it to the digit, and its conditional variance is below the total variance of a stochastic matchup.

## 15. Estimators (`uno/eval/estimate.py`)

Play once, estimate many ways. `play_pairs(rs, agents, seeds, levels)` plays
every seed from both seats (the first-named agent at seat 0, then at seat 1,
same deal and pile order) and keeps the traces (two different agents with the same name raise, section 13). Every estimator is a pure
function of those records, so AIVAT (step 9.5) reads the same games. The score
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
- `compare(a, b, divergence=True)`: paired difference per seat, over deals, of
  two row agents that played the same seeds against the same column, with the
  unpaired CI alongside. **Common random numbers**: the runner keys the action
  rng on (seed, information state), so the column agent makes the same choices
  in both matchups until the public histories diverge. `divergence` (step
  9.5a): per seat, the share of deals whose two rows' own games play the
  identical action sequence start to finish, and, among the rest, the mean
  index (a count, and a share of the shorter game's length) of the first
  differing action -- what "pairing removes 77-85 % of the variance between
  near-identical variants on one rng stream but only ~10 % between dissimilar
  policies" (section 6.6 of `context.md`) actually measures: how much of a
  pair's length the two games spend walking the exact same path
  (`uno.experiments.pairing`'s rank correlation of this against the variance
  ratio). Needs full traces: raises with a clear message if either `Pairs` was
  slimmed (`trace=None`, e.g. `uno.experiments.transfer`'s `Played.pairs`,
  which passes `divergence=False` instead).
- `bootstrap_moments(pairs, reps, level)`: seeded percentile bootstrap CIs over the pairs (resampling seeds; 99.8 % per moment by default, a family-wise 99 % for the five moments of a matchup) for the per-seat variances, the correlation, the split share and the variance ratio, with the definitions of `estimate_mirrored`; `covers(exact)` says which exact moments lie inside.
- Self-play pairs are degenerate: under common random numbers the two games
  of an A-vs-A pair are the same game, so every pair scores exactly 1/2.
- `exact_mirrored(rs, agents, levels)`: exact means, variances, covariance
  and split share over the worlds of `uno.game.chance.worlds` (deal x pile order),
  from `outcome_distribution` on each world. Exact for deterministic agents;
  for stochastic ones it takes the two games of a pair as independent given
  the world, which holds because the rng streams are keyed on the agent name
  and the information state (section 13): no two agents, and no agent at two
  seats, share a stream.
- `games_needed(edge, var_per_game, z)` = `z² var / edge²`, with the per-game
  variance of the design (a pair is two games).
- `uno.reference.solve.outcome_distribution`: exact distribution of seat 0's return
  under two fixed agents, `{+1, 0, -1} -> probability`; `evaluate` is its mean.
- Seed ranges (`uno.eval.play`): `DEV_SEEDS` = 0..3999 for building, testing and
  tuning; `EVAL_SEEDS` = 1,000,000..1,003,999 held out for the ladder's
  reported numbers, nothing is ever chosen on it.

Checks (`tests/eval/test_estimate.py`): pair traces replay; naming the agents the
other way round plays the same two games scored from the other side; self-play
pairs are identical games; `outcome_distribution` sums to 1 with the
`evaluate` mean; `exact_mirrored` means equal `evaluate` (worlds + bare engine
against the lazy tree); on `TOY` (Heuristic vs Random, 4000 seeds) and on the
7-card test ruleset (BR L1 vs Heuristic, deterministic) the sampled per-seat
means match the exact on-seeds values (to the last digit for deterministic pairs, |z| <= 3 against the action noise otherwise) and the exact variances, correlation, split share and variance ratio lie inside the bootstrap CIs of the sampled ones (99.8 % per moment, a family-wise 99 % for the five); the bootstrap CIs contain the sample moments, are reproducible and narrow with more pairs; exact and sampled
variance ratios are below 1; paired differences of deterministic agents equal
the difference of the on-seeds exact values and never exceed the unpaired
variance.

## 16. Compiled tree (`uno/compiled/`)

The lazy-chance tree of section 14 (or, with L3, the world tree: one root per
deal x unknown-pile order, bare engine below) walked **once** into flat numpy
arrays. Every exact computation of sections 14-15 is then a vectorized pass,
and CFR (section 17) becomes affordable. The walkers of `uno/reference/solve.py` and
`uno/eval/estimate.py` stay as the reference implementation the compiled versions
are tested against.

By default (`symmetric=True`, step 8.6) the tree is the **quotient** under
the color / rank symmetry of section 18: one root per orbit of deals (or of
worlds) carrying the orbit's probability, and one information set per orbit
of keys. `symmetric=False` builds the full tree. `cache=<directory>` loads
the tree from `tree_<ruleset>_<levels>_<sym|full>.npz` if present and written under the current `uno.store.CACHE_VERSION`, and
compiles and saves it there otherwise, overwriting a stale file; bump the
version whenever the engine, the chance model, the keys, the symmetry or the
layout change what a compile produces (step 8.9) (`CompiledTree.save` / `load`; the file holds every array
and the encoded keys, the `Symmetry` is rebuilt).

**One keying at a time** (step 8.7, for trees whose keys do not all fit in
memory, `TOY2_FULL` at 28M nodes): two compiles of the same (ruleset,
symmetric, world roots) walk the same nodes in the same order whatever the
keyings, so a keying loaded from one file can be attached to a tree loaded
from another (`tree.keyings[lv] = other.keyings[lv]`) and every solver then
works as if both had been compiled together. `world=True` compiles the world
tree without an L3 keying (cache suffix `W`), and `best_response_perfect`
computes the L3 best response on it with no keying at all: on a world tree
every history is its own information set, so the responder takes the
maximal arc at each of its nodes; on a symmetric tree the actions of an
orbit under a state's stabilizer have isomorphic subtrees and equal values,
so the single best action scores the same as the uniform orbit and the value
equals `best_response(tree, Level.L3, ...)`. Its strategy is per arc.

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
  **encoded as bytes** (`uno.game.keys.encode_key` / `decode_key`, lossless, a
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

**Any keying, not only a tree's own** (step 9.5a): `CompiledTree.keying_of(level)`
returns `tree.keyings[level]` for a `Level`, or `level` itself unchanged when it
is already a `Keying` -- `weights`, `profile_weights` and `best_response` all
accept either, so a `Strategy` is `(Level | Keying, sigma)` throughout. This is
what lets a **coarse** keying (section 20, not one of `tree.keyings`, built by
`uno.compiled.coarse.coarsen`) be used with the same exact passes as L1 / L2 / L3.

On top of them: `evaluate` and `outcome_distribution` (the profile's weights
and the seat-0 return, or the three outcome indicators); `best_response(tree,
level, opponent, seat)` (counterfactual reach = chance x opponent from
`reach`, then backward from the deepest level choosing per set the action
maximising the reach-weighted sum of its histories' arc values; ties to the
lowest action id, as section 14; on a symmetric tree the choice is an action
*orbit*, played uniformly, section 18; responder sets with zero counterfactual reach, which the opponent never lets the game reach, do not affect the value and take their first orbit, or `completion`, a strategy over the same keying, when given, and `BestResponse.unreached` counts them. As a row agent a best response also meets other opponents, who can steer the game into those sets, so its exploitability depends on the fill-in: the ladder reports the best response to Random both as solved and with its unreached sets filled with the Nash strategy, `br_row(tree, opponent, completion)`, step 8.8); `own` (step 9.5a, section 20) folds the responder's own per-arc weights into the reach used to combine a set's histories -- a no-op on a perfect-recall keying (L1 / L2 / L3), where it is a positive constant within every set and therefore cannot change which action maximises it (every history of such a set shares the same public history, hence the same own-reach exactly), but meaningful on an imperfect-recall *coarse* keying, where it weights each merged history by how much of a fixed reference policy's own reach it carries. Wherever a set's own-reach under `own` is entirely zero (no constituent history is currently visited under it, though chance x opponent alone can still reach the set), the weighting falls back to chance x opponent alone for that set, never to a degenerate all-zero tie: an earlier version folded `own` into the `unreached` mask too and starved such sets of a real choice, silently losing value (caught because it made `own`-weighted `best_response` disagree with its own `own=None` value on a perfect-recall keying, where they must be identical); `exploitability(tree, level, sigma)` = both
best responses to a profile, NashConv in return units and in score points
(`score(BR seat 0) + score(BR seat 1) - 1`), the per-seat gains and the
bracket `[1 - score(BR seat 1), score(BR seat 0)]` that contains the Nash
seat-0 score; `exact_mirrored` (section 15) from the per-world outcome
distributions of the two seat-swapped profiles on a world tree.

Step 9 additions. `BestResponse.unreached_sets` is the mask behind `unreached`,
and `ties` counts the reached responder sets where two or more action orbits
share the best value (relative 1e-9), whose choice is a tie-break.
`CompiledTree.rescored(exhaustion)` is the same tree with exhaustion endings
scored another way (arrays and keyings shared, `util` replaced), so every solver
runs unchanged: `"hand"` is the rule of section 9, `"draw"` scores every
exhaustion ending as a draw (degenerate on free-play rulesets: the trailing
player can always draw the pile empty and claim, so optimal play stalls into
draws), `"stuck"` makes the player who had to draw from the empty pile lose (the
drawer is the Draw arc's player, or the opponent of a play whose penalty draw
exhausted the pile). `two_sided(tree)` is the value of the
game in which both players see what a node holds, from a max / min backward pass
(`extremal_values`): on the lazy tree both hands with the pile as chance
(expectiminimax, the two-sided L2 game), on a world tree everything (minimax per
world, L3), with the mass of the worlds whose result is the same under every pair
of strategies and the outcome shares under minimax play. Both games are invariant
under the symmetry group, so the quotient gives the full game's values. Unlike
the one-sided L2 / L3 columns (best responses to a fixed opponent) these are
games in which both sides play optimally, and they sit between the one-sided best
responses of the two seats to any fixed opponent.

Strategies from agents: `tabulate(tree, level, agent)` asks the agent at
every set of the keying, both seats, with the observation rebuilt from the
set's key by `uno.game.observe.observation_from_key` (the key is by definition
everything the observer knows, section 10; the L3 key is the state). On a
symmetric tree it raises if the agent is not symmetric (its policy is not
uniform within some action orbit). `tabulate_table` maps a
`BestResponse.table` onto a keying (on a symmetric tree the chosen action is
spread over its orbit). `CompiledAgent` plays any strategy over any keying
through `uno.eval.play`; on a symmetric tree it canonicalises the observation's
key, looks the canonical set up and maps the actions back with the inverse
element. `transfer(src, dst, level, sigma)` (step 9) rewrites a strategy over one
tree's keying as a strategy over another tree's keying of the same level by
matching set keys (raising unless every destination set is in the source with the
same actions): the lazy and world trees of a ruleset share their L1 sets, numbered
differently, so a tabulated or solved strategy moves between them without asking
an agent again. `uno.store.load_or_tabulate` caches `tabulate` in
`tab_<name>_<D|W>_<sym|full>_<level>_<agent>.npz`, keyed on the agent's name (a
name identifies a policy, section 13) and a fingerprint of the keying; it is meant
for the fixed baseline agents, and `CACHE_VERSION` is bumped when one of their
policies changes.

Checks (`tests/compiled/test_compiled.py`, `tests/game/test_observe_keys.py`): the
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
mirrored moments match the reference on a 7-card ruleset; the node order of
a compile does not depend on its keyings (lazy and world trees), a keying
attached from another compile gives the same L2 best response, and
`best_response_perfect` equals the keyed L3 best response and replays to its value; the unreached sets of a best response are exactly the responder sets the reference walker does not reach under the opponent's support, a completion leaves the value unchanged, and `br_row` assembles both seats into a row that replays to each seat's value. On `TOY` the twelve step 5-6 values are reproduced (`uno cfr`).
The symmetric tree's checks are in section 18.

## 17. CFR (`uno/compiled/cfr.py`)

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

**Variants** (step 8.7, [step 8.7](results/08.7-toy2.md)), same passes, `cfr(variant=...)`:

- discounting (DCFR, Brown & Sandholm 2019): after iteration `t` the
  accumulated positive regrets are multiplied by `t^a / (t^a + 1)`, the
  negative ones by `t^b / (t^b + 1)`, and the contribution of iteration `t`
  to the average strategy carries weight `t^g`. CFR+ is `a = inf` (no
  discount), negatives clipped, `g = 1`; DCFR is `(1.5, 0, 2)`;
- prediction (PCFR+, Farina, Kroer & Sandholm 2021): the current strategy is
  regret matching on the accumulated regrets *plus the last instantaneous
  regret* (the optimistic step), with regret matching+ and `g = 2`.

Presets in `uno.compiled.cfr.VARIANTS`: `vanilla`, `cfr+`, `dcfr`, `pcfr+`, and
`dpcfr+` (prediction with DCFR's positive-regret discount). The orbit
averaging applies to every variant, so all of them stay symmetric.

`cfr(fixed=(mask, strategy))` (step 9) plays `strategy` at the set-actions of
`mask` (whole sets) at every iteration and in the average: CFR on the game in
which those sets are fixed. With a best response's reached sets fixed, the free
responder sets converge to the least exploitable completion of its unreached
sets.

The result is the normalised
average strategy (sets never reached by their owner are uniform), the
regrets, and a log of exploitability (section 16) at chosen iterations, whose
last entry (iteration count and NashConv) goes next to every ladder cell that
uses the strategy. `CFRResult.save` / `load` persist it (`.cache/`, not
committed), stamped with `CACHE_VERSION` like the trees; `uno.store.load_nash`
ignores a stale result.

Checks (`tests/compiled/test_cfr.py`): on the 7-card ruleset CFR+ reaches NashConv
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

## 18. Symmetry (`uno/game/symmetry.py`)

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
`Heuristic` (section 13) are symmetric. A learned agent is made symmetric in two steps: canonicalise its observation and map its action back with the inverse element (`canonical`, `map_policy`), which is also a |G|-fold gain in data efficiency; and at a canonical key with a non-trivial stabilizer, average its output within every action orbit of that stabilizer (see **Orbits** below). `uno.agents.symmetrized.Symmetrized` does both. The first step alone does not give a symmetric policy: `canonical` returns one of several elements that reach the canonical key, and the action played would depend on which. A raw, asymmetric agent needs the full tree (`symmetric=False`). Symmetrising a policy by *mixing* its images under G (one image drawn uniformly for the whole game, equivalently the reach-weighted average of their behaviour strategies) never raises its exploitability, because the best-response value is convex in the opponent's mixed strategy and every image has the same exploitability; averaging the behaviour probabilities set by set is a different policy and has no such guarantee.

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
value against a symmetric opponent (a symmetric best response always exists: mix the images of any best response), and CFR averages regrets and
average-strategy mass within orbits (section 17). `tabulate` refuses an
agent whose tabulated policy is not orbit-uniform.

Checks (`tests/game/test_symmetry.py`): on random games over random rulesets,
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
misses) and the same exploitability of the uniform profile; `tabulate` rejects an asymmetric agent on `TOY`; `Symmetrized` commutes with the group on random games and leaves the Heuristic unchanged, and a wrapped asymmetric agent tabulates on the quotient with the value it has on the full tree and through the engine; a tree survives save / load and the
cache, and a cached file of another `CACHE_VERSION` is recompiled; on the quotient of `TOY` the twelve step 5-6 / 8.6 best-response
values and the Nash value 0.6196 are reproduced.

## 19. Sizing by position (`uno/reference/sizing.py`)

`census` (`uno/reference/census.py`) visits every history, so counting a
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

Checks (`tests/reference/test_sizing.py`): on small rulesets, full and quotient, every
count and the UniformLegal end mass equal the census's; the PlayFirstRandom
end mass reproduces the compiled tree's exact random-vs-random value and the
compiled node count; the world tree has the same values and depth; the cap
stops the walk; `TOY` and `TOY_FULL` sizes and values of step 8.6 are pinned, and so are those of the
duplicate-card candidate 3c(2,1)+W (step 8.9).

## 20. Coarse keys and the memoryless response (`uno/compiled/coarse.py`)

Step 9.5a, D3 of the step 9 review's section D ("Memory: BR L1 conditions on the
full history. A backward pass on a coarse, history-free key gives a valid
policy, a lower bound on the memoryless ceiling: does the RL observation need
history?"). Three nested keys, each a deterministic function of the L1
observation rebuilt from a canonical L1 set's key
(`uno.game.observe.observation_from_key`):

- **K0**: (seat, tree level, hand, top, current color, opponent hand size, pile
  size, phase, drawn card). The tree level (`2 * len(public) + [phase ==
  TURN_START]`, section 16) is included so a coarse set never spans two tree
  levels, which the per-level backward pass of `best_response` needs: a set
  whose arcs span two levels would have its choice finalised, wrongly, at the
  first level it is seen. The level also acts as a weak clock, a little memory
  of how long the game has run.
- **K1**: K0 + the discard pile as a multiset (order dropped, contents kept).
- **K2**: K1 + a void hint -- the colors that were current at each of the
  opponent's draws since their last play (the list clears whenever they play),
  a coarse proxy for "what they might be avoiding". Counts every draw event
  once, not once per card of a multi-card penalty draw.

Because K0's fields (hand, top, color, phase, drawn) are exactly what
determines legality (given the ruleset's static flags: `opp_hand_size` and
`pile_size` do not affect which of *this* player's actions are legal), two L1
sets sharing a K0 value are guaranteed to share the same legal actions; K1 and
K2 only add fields, so the guarantee carries over. `coarsen` asserts it anyway,
as `compile_tree` does for its own keyings, and raises if it ever fails.

**`coarsen(tree, keyfn, level=Level.L1)`** groups `tree.keyings[level]`'s
information sets by `keyfn` applied to each set's own canonical observation,
into a new `Keying` (section 16) over the tree's own nodes and arcs -- usable
directly with `best_response`, `tree.weights` / `tree.reach` / `tree.values`
and `evaluate` (`CompiledTree.keying_of`), but with no `.keys` (no
byte-encoding exists for K0 / K1 / K2 shapes, unlike the L1 / L2 / L3 keys of
`uno.game.keys`): those functions never touch a keying's `.keys`, so this is
enough for everything here, but `tabulate` / `CompiledAgent` do not work on a
coarse keying. Sets are numbered by `(tree level, key)`, not by key alone: a
plain sort by key value need not group same-level sets together (K0 / K1 / K2
sort by hand before level, and `identity_key`, section 21's tests, does not
carry a level field at all), and `best_response`'s per-level loop silently
assumes sets are numbered contiguously per level, as `compile_tree`'s own
keyings are by construction ("in order of first appearance"). Numbering by key
value alone passed every small-ruleset unit test (which never happened to
interleave two sets from different levels under the wrong id order) and still
produced a self-consistent-looking `BestResponse` (same value from
`best_response` and from replaying its own `sigma` through `evaluate`) on
`SMALL`'s *non*-symmetric tree, and was only caught by comparing an
`own`-weighted best response's value against its `own=None` value.

**Orbits.** On a symmetric tree, several L1 sets can merge into one coarse
set, each carrying its own action-orbit partition (section 18) -- possibly a
*different* one, since a set's orbit structure depends on the stabilizer of
its whole canonical key (history included), not only on K0 / K1 / K2. A coarse
policy is one distribution per coarse set, so for it to be valid for every
merged L1 set at once, it must be uniform within the **transitive join** of
their orbit partitions (union two actions whenever *any* merged set's own
orbits group them together): the join is never finer than any single
constituent's partition (so it never asks for *less* uniformity than a
constituent independently requires), which is what keeps the resulting
strategy achievable and never an over-estimate (section 18, "a best response
above the maximum is not a strategy"). It can force a little *more* uniformity
than the coarse-optimal strategy strictly needs -- `coarsen` picks the safe
join rather than search for the tightest valid partition, which only matters
for `memoryless_response`'s value as a lower bound, never for its validity.
`project_onto_coarse` (a starting L1-level strategy, e.g. a K0-measurable
agent's tabulation, restricted to a coarse keying via one representative L1
set per coarse set) must itself be averaged within these joined orbits before
use (`Keying.orbit_mean`), for the same reason: skipping it reproduces the
section 18 bug on the coarse keying.

**Known limitation on a symmetric tree**: K0 / K1 / K2 read their fields off
whichever canonical L1 representative `compile_tree` happened to pick for each
L1 set (minimising the *whole* key, history included) -- not a representative
re-minimised for K0 / K1 / K2's own, smaller field set. The same abstract
coarse position can therefore land in different coarse groups depending on
which L1 representative reached it, so the quotient tree's K0 / K1 / K2
partition can be a *finer* (less merged) one than the "ideal" quotient of the
abstract K0 / K1 / K2 partition, and `coarsen(quotient_tree, k0)`'s value is
not expected to numerically match `coarsen(full_tree, k0)`'s (unlike
`identity_key`, where every group has one member and no such gap can arise:
`tests/compiled/test_coarse.py` checks this exact contrast). Properly
re-canonicalising K0 / K1 / K2 under the group would need every constituent's
action list remapped through its own canonicalising element at the arc level,
not just the key; not done (this was step 9.5a's plan, which expected the two
to agree -- a correction logged in `docs/results/09.5a-results.md`). The value
computed here is still a valid, achievable lower bound on *that tree's* own BR
L1 either way; only the full/quotient comparison is affected.

**`memoryless_response(tree, coarsening, opponent, seat, start_l1_sigma,
iterations)`**: an iterated greedy best response. From a starting policy
projected onto the coarse keying, repeatedly compute the best response to
`opponent` with `own` fixed at the *current* iterate's weights
(`best_response(..., own=tree.weights(keying, sigma, seat))`, section 21),
then evaluate the new policy exactly and keep going. Every iterate is a real
(history-free) policy, so the best value seen is a valid lower bound on the
best achievable memoryless policy against `opponent`; `best_response(tree,
Level.L1, opponent, seat)` (full history) is the upper bound. The iteration
need not be monotone (a local best response under imperfect recall is not
guaranteed to improve every step), so the best `(value, sigma)` pair across
the whole run is kept, not just the last; it stops early when the policy
repeats an earlier iterate exactly.

Checks (`tests/compiled/test_coarse.py`): the identity coarsening (`identity_key`
= the full L1 key) reproduces `best_response` L1 exactly, on both the full and
the quotient tree, and the pinned `TOY` BR-to-random values (0.7934 / 0.5513);
Random and the Heuristic tabulations are constant within every K0 set; a coarse
response replays to its value through `evaluate` and never exceeds BR L1, on
both trees independently; brute force over every pure K0 policy (at most 4,096
of them, else the example is skipped) bounds `memoryless_response`'s result;
one `best_response` pass with `own` fixed agrees between the full and the
quotient tree on `identity_key` (not on K0 / K1 / K2, the known limitation
above).

## 21. Edge attribution and the sampled cross-check (`uno/compiled/edge.py`, `uno/compiled/lookup.py`)

Step 9.5a, D2 and D4 of the step 9 review's section D ("An edge-by-decision
tool: for a checkpoint against a reference strategy... where it loses its
points, by decision kind and by situation"; "Metric for agents trained against
a fixed opponent: the gap to BR L1 against that opponent").

**The identity** (performance-difference lemma, Kakade & Langford 2002,
adapted to a fixed-opponent extensive-form game). For two row strategies A
("row"), B ("ref") against the same fixed opponent O, both playing `seat`:

```
V^B - V^A = sum_h rho^A(h) * sum_a sigma_A(a|h) * [V^B(h) - Q^B(h,a)]     (mirror)
          = sum_h rho^B(h) * sum_a sigma_B(a|h) * [Q^A(h,a) - V^A(h)]     (forward)
```

summed over A's (resp. B's) own decision nodes h. `rho^X(h)` is the **full**
reach probability of h under (X, O, chance) -- unlike a best response's
counterfactual reach, it includes the row player's own action probabilities
along the path, since it is a state-visitation frequency, not a best-response
weight (`uno.compiled.edge._reach`). `V^X`, `Q^X` are node and arc values under
(X, O): well-defined at *every* node of the tree, not only the ones X actually
reaches, because `CompiledTree.values` is a backward pass over the whole tree
regardless of which weights reach a node. Both identities are exact and,
summed over all of A's (B's) own arcs, equal `V^B - V^A` to numerical
precision; the two need not agree term-by-term, since they group the same
total differently (by A's decisions or by B's).

When B is a best response to O, **every mirror term is non-negative**: B
maximises the reach-weighted sum of continuation values at every information
set (section 16), and A's own reach is a positive common factor within a
perfect-recall set (section 14), so grouping mirror terms by set never changes
sign (`tests/compiled/test_edge.py`).

**`br_gap(tree, row, opponent, seat)`**: BR L1's return minus `row`'s, at
`seat` (the D4 metric: the exact, luck-free gap to the ceiling against a
*fixed* training opponent). **`mirror_terms`** / **`forward_terms`** return
the per-arc decomposition, labelled by `uno.eval.decisions.node_kind` /
`action_class` / `situation` (`uno.store.load_or_label`, section 22) for
grouping.

**Stakes and the wild split** are separate, always-defined diagnostics, not
terms of the identity above:

- `stakes(tree, row, ref, opponent, seat)`: `max_a Q^ref(h,a) - min_a Q^ref(h,a)`
  at every `seat`-owned decision node, weighted by `rho^row(h)` -- how much a
  decision could possibly be worth under the reference's own values, always
  >= 0 by construction (a max minus a min).
- `wild_split(tree, row, ref, opponent, seat, kinds)`: at wild-kind decisions
  only, splits the reference's values into a "spend" part (the best action
  overall minus the best wild) and a "color" part (the best wild minus the
  worst wild), both >= 0 by construction. Fully vectorized (a small
  `(num_actions,)` lookup broadcast through `k.ia_action`, then
  `np.maximum.at` / `np.minimum.at` scatters per node), so it stays cheap even
  on `TOY2_FULL`'s ~8M set-actions.

**`uno.compiled.lookup.NodeIndex`** maps a *played* game's states to a compiled
tree's own nodes and arcs, via the **L2** keying, the finest one available
where "on the full `TOY` tree the L2 keying has exactly one node per set...
the L2 key of a played state *is* its node" (section 16); on a symmetric tree,
up to `|G|` nodes share one canonical L2 set, with equal values for a
symmetric profile, and `NodeIndex` picks the tree's own lowest-numbered node
of that set as the representative. `node(state)` finds it; `arc(state,
action)` finds the arc there for `action`. Because `tree.arc_action` is in
*each node's own* frame (`compile_tree` canonicalises a node's *key*, not its
underlying `State`, so a symmetric tree's nodes below a canonical deal are
real, un-relabelled states), matching `action` (mapped through the node's own
canonicalising element) against `tree.arc_action` directly is only correct at
the one node whose search happened to produce that element -- not in general
at `NodeIndex`'s chosen representative, which may be a *different* node of the
same set with its own, different canonicalising element. What stays consistent
across every node of a set is `k.arc_ia` / `k.ia_action`, `compile_tree`'s own
per-node-canonicalised action ids (used to build the keying in the first
place): `arc` matches on those instead. `build(tree, level=Level.L2)` needs an
L2 keying on the tree: cached for `TOY`, `TOY_FULL`, `TOY2` and `TOY2_SKIP`;
skipped for `TOY2_FULL`, whose L2 keying alone is ~3 GB.

**The sampled D2 check** (`uno.compiled.edge.sampled_mirror`): plays `row`
against `opponent` and, at each of `row`'s own decisions, sums `sigma_row(a|h)
* [V^ref(h) - Q^ref(h,a)]` from the tree's own node / arc values under (ref,
opponent), read off via `NodeIndex` -- exactly the per-node quantity
`mirror_terms` sums exactly. A played game samples decision nodes with
probability exactly `rho^row`, so the per-game total's sample mean over many
games is an unbiased estimator of the exact mirror total; `uno.experiments.edge`
checks it (and each `node_kind` sub-total) against the exact value with a
z-score on the sampled standard error, |z| <= 3.

Checks (`tests/compiled/test_edge.py`, `test_lookup.py`): both identities equal
the exact value gap on random profiles and vanish (at the per-node level, not
per arc: individual arc terms of a self-comparison need not be zero, only each
node's own terms summed, since `values`' backward pass makes a node's value
the weighted average of its arcs') when A = B; `br_gap` matches BR L1 minus a
row's value; mirror terms are non-negative when the reference is a best
response; grouped totals by `node_kind` agree between the full and the
quotient tree for symmetric agents; `stakes` and `wild_split` are non-negative
on a ruleset with a wild. `NodeIndex` finds every state along random games,
with matching legal actions; the looked-up node's value matches an
independent reference (`uno.reference.solve.evaluate` from that state with
lazy chance); the looked-up arc's value matches the reference Q-value
including chance-branching actions (a draw with several outcomes); on the
quotient, every node of one L2 set shares its value to 1e-12.

## 22. Decision labels (`uno/eval/decisions.py`, `uno.store.load_or_label`)

Step 9.5a. Classifiers of a decision, as functions of an observation and legal
actions (not a `State`), so they apply equally to played games (`observe`) and
to a compiled tree's information sets (`uno.game.observe.observation_from_key`):

- `node_kind(rs, obs, legal)`: one label per decision **node** (a partition,
  unlike `classify` below): `"single"` (one legal action), `"drawn"` (after a
  voluntary draw), `"wild"` (a wild among the legal plays), `"action"` (an
  action card among 2+ plays), `"color"` (plays in 2+ colors), else
  `"other"` (a single-color choice, or a draw against one kind of play).
- `action_class(rs, obs, legal, a)`: one label per (node, action): `"draw with
  a play in hand"`, `"forced draw"`, `"pass"`, `"play drawn card"`, `"wild
  with a colored play available"`, `"wild when only wilds play"`, `"action
  card"`, `"number"`.
- `situation(obs)`: `(seat, opponent hand size bucket, pile size bucket)`.
- `classify(rs, obs, legal)`: the step 8.5 decision-kind census of section 6.7
  of `context.md` (moved here unchanged from `uno.experiments.transfer`; its
  categories overlap, unlike `node_kind`'s partition).

`set_labels(tree, level=Level.L1)` computes `node_kind` / `situation` /
`action_class` for every set of a keying, one `observation_from_key` call per
set (the same cost as `tabulate`, section 16); `uno.store.load_or_label`
caches it the same way `load_or_tabulate` caches an agent's tabulation, keyed
on a fingerprint of the keying (not an agent's name: labels are a property of
the tree alone).

## 23. AIVAT (`uno/eval/aivat.py`, `uno/compiled/aivat.py`, `uno/eval/critic.py`)

Step 9.5b. Burch et al. 2016 (arXiv:1612.06915): a fourth estimator on the same
games as raw, mirrored and paired.

**The estimator.** A value model gives, in seat-0 return units, `nu(h)` at a
state and, for an action, `alpha(h, a)` = the expectation of `nu` over the
action's chance outcomes. For one game, with `u` the final return:

```
X = u - sum_decisions [ chance: nu(child) - alpha(h, a_taken) ]
      - sum_decisions of corrected players [ alpha(h, a_taken) - sum_a sigma(a|h) alpha(h, a) ]
      - [ deal: nu(first state) - E_deals nu ]
```

Each bracket has conditional mean zero given what precedes it, when `alpha` uses
the true chance model and `sigma` is the policy actually played. So X is
unbiased for any `nu` and any subset of corrections (`Corrections(chance, row,
column, deal)`, relative to the first-named agent), provided the subset is chosen
before the event it corrects. With `nu` the exact values of the profile being
played and every correction on, the sum telescopes and X equals the root value on
every game.

**What V may see.** Anything fixed before the chance event it integrates over:
both hands, the public history, the pile as a multiset. Never the pile order: a V
that reads it makes every draw deterministic given V's input, which leaves only
the deal and reshuffle permutations as chance events, with intractable
expectations. The value function is therefore an **L2-state** value. On a toy
that is exactly the lazy tree's node value, whose node is an L2 set. (The step 9.5
brainstorm and context called it an "L3 critic"; corrected here.)

**Records.** A value model (`ValueModel`: `exhaustion`, `deal_mean`,
`records(trace)`) emits one `StepRecord` per decision: player, probs, action,
`node`, `alpha` for every action of the support, `child`, and the chance `event`
(voluntary, penalty or reshuffle draw). `game_terms` folds a game's records into
`GameTerms` (return, deal, chance, action per seat, chance split by event,
skipped decisions), from which every subset of corrections is one subtraction.
`pair_terms(pairs, models)` works over `play_pairs` records; `stream_pairs` plays
and folds pair by pair and keeps no trace, as full UNO needs (step 9.5a, bug 4).
`estimate_aivat(terms, c)` returns per-seat AIVAT and raw estimates, the per-game
difference AIVAT - raw, and the pair (mirrored) AIVAT estimate.
`rescored_return` / `terminal_return` score a played terminal under the three
exhaustion scorings of `CompiledTree.rescored`.
`mirror_regret(records, seat)` = `sum node - sum_a sigma alpha` over a seat's
decisions: section 21's sampled edge decomposition, which `sampled_mirror` now
reads off the same records.

**Exact moments** (`uno.compiled.aivat.aivat_moments(tree, seat0, seat1, nu, c,
seat)`). One backward pass carries the mean and second moment of `F(h)` = the
estimator from node h on. Along a chance edge, `F = F(child) - (nu(child) -
alpha(arc))` (the root arc: the deal correction). At a decision, `F = F(arc) -
(alpha(arc) - sum_arcs w alpha)` for a corrected mover. The mean equals the
profile's value for any `nu` and any subset. With no correction the variance is
the outcome variance; with exact `nu` and every correction it is zero.

**Value functions** (all group-invariant, so quotient moments equal the full
game's):

- `exact_values`: the profile's own values.
- `noisy(sigma)`: exact plus N(0, sigma²) drawn once per L2 set.
- `l1_set_values`: the reach-weighted mean over the mover's L1 set.
- other profiles' values.
- `critic_values`: a fitted critic evaluated from each node's L2 key.

`outcome_r2` = `1 - E[(u - nu)²] / E[(u - mean u)²]` over visited decisions,
reach-weighted. The critic's sampled R² over visited states estimates the same
quantity. `TreeValues(idx, nu, exhaustion)` reads `nu` / `alpha` for played games
through `NodeIndex.arcs`, and works for ticket-draw games.

**Chance model with reshuffle** (`uno.game.chance.draw_outcomes`). Children of an
action with probabilities, for any ruleset. The drawn cards are uniform over the
unknown part of the pile: the pile minus the setup tail before the first
reshuffle, the whole pile after it. This is the same distribution for pile and
ticket draws (section 8).

- A voluntary draw on an empty pile with a seeded reshuffle draws from the
  discard minus its top. The reshuffle is written out as the engine does it:
  discard down to its top, `n_reshuffles + 1`, `("reshuffle",)` before the draw
  event.
- A draw that empties the pile part-way and then reshuffles returns None: no
  chance model, and the corrections that need it are skipped. Skipping is decided
  before the event, so it stays unbiased.
- More than `enum_cap` ordered draws are sampled `samples` times from an rng
  independent of the game (`f"{seed}:aivat:{decision}"`).
- On `reshuffle == "none"` the result equals `chance_outcomes`.

Children are stepped with `draws = "pile"`.

**Critic** (`uno.eval.critic`). Features are colour- and rank-symmetric
functions of the position (`position_features`):

- per hand in the seat-0 frame: size (raw and one-hot), counts by kind, colours
  held, the largest colour count, playable cards, cards of the current colour;
- global: the mover, `AFTER_DRAW`, pile size and emptiness, the hand-size
  difference;
- the hand blocks multiplied by "seat 0 moves".

The same features come from a `State` or an L2 observation. The model is a
fractional logistic on the score, fitted by chunked IRLS with ridge
(`fit_logistic`). The ridge strength is picked by outcome R² on the last fifth of
the fitting games (`fit_critic`). There is one critic per profile (row, column,
seat), fitted on **`FIT_SEEDS` = 10,000,000..10,099,999** (`uno.eval.play`: a
dev-class range, nothing reported on it).

`CriticValues(critic, exhaustion, samples, deal_mean)` is the value model: `alpha`
by `draw_outcomes`, terminal children by `terminal_return`. `deal_mean` comes from
the tree on a toy and is None on full UNO, which has no deal correction.

**Intermediate rulesets** (`uno.game.rules.MID_12`, `MID_25`), sampled only,
chosen by `uv run uno aivat_uno sweep` with a pre-declared rule. Heuristic vs
Random, 1,000 dev pairs: actions per game within ±20 % of 12 or 25. The shares of
"draw with a play in hand", "action card among other plays", "wild: hold or
spend" and "which color" must each be within 0.5-2x of FULL_UNO's. Among the
qualifying candidates, the smallest largest |log share ratio| wins.

Checks (`tests/eval/test_aivat.py`, `test_critic.py`,
`tests/game/test_draw_outcomes.py`), on full and quotient trees:

- constant V reproduces raw on every game;
- exact V with every correction gives the root value on every game, pile and
  tickets;
- exact means equal `evaluate` for every subset, and the raw variance equals the
  outcome variance;
- a symmetric V gives the same moments on the full and quotient trees;
- sampled moments match exact ones with an uncorrected stochastic column;
- `rescored_return` matches the rescored tree;
- `mirror_regret` reproduces `sampled_mirror`;
- `draw_outcomes` equals `chance_outcomes` without reshuffle, contains the real
  child and sums to 1 with reshuffles and tickets, and its chance corrections of an
  arbitrary fixed V have mean zero, reshuffle draws included;
- IRLS recovers coefficients, features are symmetric and agree between state and
  L2 observation, and the critic's played path (`CriticValues`) equals the tree
  path (`critic_values` + `TreeValues`) record by record.

Experiments: `uv run uno aivat <ruleset>` on the toys, `uv run uno aivat_uno` on
the intermediates and full UNO.

## 24. PIMC values and the value network (`uno/eval/pimc.py`, `uno/eval/value_net.py`)

Step 9.5c. A better V for AIVAT on rulesets with no tree: Monte Carlo rollouts on determinized
states (PIMC at L2) as a teacher, a network fitted on their values as the V.

**Determinization.** Given the L2 state (both hands, public history, pile as a multiset) the order of
the unknown part of the pile is uniform, the setup tail at the bottom until the first reshuffle: the
chance model of section 23. `determinize(state, rng)` shuffles that part from an rng of its own, sets
`draws = "pile"` (a uniform order drawn from the top is distributed like tickets, section 8) and gives
the state a fresh seed, so the engine's reshuffles and the rollout's action choices are independent of
the real game. The result depends on the state only up to its pile order and seed. Penalty draws across
a reshuffle are played by the engine: nothing is skipped.

**Rollout values.** `rollout_value(state, agents, n, rng)` is the mean seat-0 return of `n` games from
`n` determinizations, played by the profile's own policies with actions sampled from `rng`: unbiased
for the profile's L2 value, variance about `(1 - V²) / n`. `rollout_q` takes an action first: unbiased
for `alpha(h, a)`. Rngs are keyed on the game's seed and a label naming the value
(`rollout_rng(seed, label)`), never on anything the game itself uses.

**Why not rollouts as AIVAT's V.** Any V whose noise is independent of the game keeps AIVAT unbiased
(`RolloutValues`), but with rollout values the corrections telescope, at best, to "n more games from the
same hands", costing those games; independent noise per value adds about `2 (1 - V²) / n` per
decision. Rollouts are therefore the **teacher**. `uno pimc_value mc` measures the cost.

**The deal floor.** With the exact L2 values and corrections at chance and both players, AIVAT's
estimate is `nu(first state)`, so its variance is the variance over deals of the deal's value.
`deal_floor(rs, agents, seeds, replays)` replays every dealt hand under `replays` determinizations:
between-deal variance = Var(deal means) - mean within-deal variance / replays (unbiased), share =
between / (between + within), 95 % bootstrap CI over deals. On a toy it must equal the exact ratio of
`aivat_moments(exact_values, chance+both)`.

**Teacher targets** (`teacher_targets`). One game per FIT seed; every decision state plus two siblings
AIVAT also values: the child of one non-taken action of the support and another chance outcome of the
taken action when it draws (both through a determinization, so from the true chance model). Each gets
`n` rollouts: target, noise variance, the game's return. Processes over seed chunks; the targets do not
depend on the number of workers.

**Features** (`position_features`), from a `State` or an L2 observation, in the seat-0 frame:

- one block per colour: counts of number / skip / reverse / +2 cards of the colour in each hand and in
  the discard, playable cards of the colour per hand, the colour is current, the top card has it, the
  drawn card has it;
- a global block: per hand, size, wilds, +4s, playable cards, number cards matching the top card's rank
  in another colour, action cards matching its kind in another colour, ranks held in 2+ colours, colours
  held, largest colour count, size 1 and size 2; mover, `AFTER_DRAW`, pile and discard sizes, empty pile,
  reshuffles (capped at 3); top card's kind; the drawn card's kind; wilds and +4s in the discard; size
  difference.

A colour renaming permutes the blocks and a rank renaming within a copy class changes no feature.

**Network** (`ValueNet`). DeepSets over colours: a shared MLP on each colour block concatenated with the
global block, sum and max pooling, an MLP on the pooled vectors and the global block, tanh: the seat-0
return. Invariant under the symmetry group (section 18) up to float rounding, and independent of the
number of colours. Inputs standardized with colour-pooled statistics. Fitted by Adam on the squared error
to the targets (float32), early stopping on the last fifth of the games, best epoch kept. Inference in
**float64** on CPU, so V is a batch-independent function of the state. It implements the value-model
interface `CriticValues` and `critic_values` read (`features`, `observation_features`, `values`), as
`Critic` does.

**Deal correction without a tree.** `E_deals V(first state)` is estimated once over `DEAL_SEEDS`
(40,000,000+, 250,000 deals), independent of the evaluated games. The deal term then has mean
`true mean - estimate`: a fixed error, printed as its standard error, that does not shrink with more
evaluated games.

Checks (`tests/eval/test_pimc.py`, `tests/eval/test_value_net.py`):

- determinization keeps everything but the pile order and the setup tail's place; rollout values and
  rollout Q are identical when the unknown pile is permuted or the seed changes (pile and ticket draws,
  reshuffles, full UNO);
- the child of a drawing action over many determinizations is distributed as `draw_outcomes`,
  reshuffles and empty-pile draws included;
- rollout values and Q against the exact node values and arc expectations; the deal floor against the
  exact ratio;
- features invariant under the group (as a multiset of colour blocks) and equal from a state and its
  L2 observation; the network's value invariant; the fit learns; float64 inference batch-independent;
  save / load; the played path (`CriticValues`) equals the tree path (`critic_values` + `TreeValues`).

Experiment: `uv run uno pimc_value check | floor | pilot | aivat | mc` (module docstring).

## 25. The PIMC agent (`uno/compiled/pimc.py`)

Step 9.5d. PIMC (perfect-information Monte Carlo) samples worlds consistent with what the
player knows, solves each with perfect information, and plays the action that scores best on
average. Section 24's rollouts use determinization to *value* a state under a fixed profile;
here determinization drives the *choice*, which is where PIMC's own bias (strategy fusion and
non-locality) lives.

**PIMC-infinity, exactly.** On the toys the `n -> infinity` limit is computable: average over
the whole belief instead of a sample of it. At an L1 set `I` the choice is
`argmax_a sum_{h in I} b(h) * V_L3(h, a)`, and on a **world tree** both factors are already
there. The roots are the worlds and everything below is the bare engine, so every non-root arc
has exactly one edge of probability 1 and `reach(...)[:, 2]` at a node *is* that world's prior.
Hence:

- `V_L3(h, a)` is a **gather** off a node-value vector (`world_values`), not a pass of its own;
- neither `b` nor `V_L3` depends on PIMC's own policy (own reach is constant within an L1 set,
  which has perfect recall), so the policy is fixed by **one global grouped argmax** -- there is
  no per-level backward loop, unlike `best_response`, whose levels exist precisely because a
  set's value there depends on what the responder does deeper down.

Two axes, four variants:

- **belief**: `"posterior"` weights `b(h) ~ prior(h) * opponent_reach(h)`, the true Bayesian
  belief given the opponent's known strategy (PIMC *with* inference); `"prior"` weights
  `b(h) ~ prior(h)`, i.e. no inference -- what a real PIMC does when it deals the unseen cards
  uniformly, and the only one available where the opponent's strategy is unknown;
- **continuation**: `"minimax"` solves each world two-sided (classic PIMC: the opponent is
  assumed clairvoyant too); `"response"` solves it as a perfect-information best response to the
  opponent's actual fixed strategy (`best_response_perfect`, whose `sigma` replays through
  `tree.values` to give exactly these per-arc values).

`"prior"` + `"minimax"` reads nothing about the opponent at all, so that policy is literally
opponent-independent (and its exploitability is identical whichever opponent it was built
against: a check, not a coincidence).

**Orbits.** The choice is an *orbit* of actions played uniformly, as in `best_response`: a pure
choice at a set with a non-trivial stabilizer is not a symmetric strategy (section 18).

**Unreached sets.** Under `"posterior"` the sets the opponent never allows carry no mass and get
a fill-in, not a choice, exactly as a best response does (section 16); under `"prior"` every set
holds at least one world, so the mass is always positive and no fill-in ever arises.

**Tie-breaking is value-relevant here, unlike a best response.** For a best response a tie
between orbits means equal continuation value, so any tie-break scores the same; a PIMC tie is a
tie in the belief-weighted *perfect-information* score, and two actions can look identical to
that optimistic model while being worth different amounts in the real game. "PIMC" is therefore
a family indexed by its tie-break: the quotient's orbit-uniform choice and a full tree's pure
per-set choice are different policies (0.01-0.44 score points apart on `TOY`). The orbit-uniform
one is canonical because it is the symmetric one.

`value` is the policy's **actual** return from an ordinary `evaluate` pass, never the argmax's
own optimistic sum. `believed` is the root of the same perfect-information solve: BR L3's value
under `"response"`, the two-sided value under `"minimax"`. `fusion_gap = believed - value` is an
upper reference only under `"response"`; under `"minimax"` it routinely goes negative, because
the two-sided value assumes a clairvoyant *opponent* as well and a weak fixed opponent gives the
policy more than that.

**PIMC-n** (`pimc_n`). With `n` sampled worlds instead of the whole belief, and perfect recall
giving at most one visit to an information set per game, what the player follows is the
*behavioural* policy `pi_n(a | I) = P(the argmax over n worlds drawn from the belief is a)`.
`pi_n` is estimated by resampling each set, then scored by one exact `evaluate`, so the number
carries no game-playing noise -- only `pi_n`'s own Monte Carlo error.

It converges to PIMC-infinity, and on the toys almost always from below (n = 1 costs 0.9-7.4
score points), but **not necessarily**: PIMC-infinity maximises the belief-weighted
*perfect-information* score, which is not the true value, so the deterministic limit need not be
the best member of its own family and a randomised `pi_n` can be worth slightly more (measured:
`TOY` vs the Heuristic at seat 1, -0.08 at n = 32 and -0.11 at n = 64). The pre-declared check is
therefore that the gap *shrinks* with n, never that it stays positive.

**A world's arcs are not listed in its information set's order.** The members of an L1 set are
different worlds, each canonicalised by its own group element, so arc `j` of a member's node is
not set-action `j` (3.9-23.8 % of arcs on the toys, worse at seat 1). Anything averaging
per-action values across a set's members must map arcs through `k.arc_ia` -- the trap section 21
documents for `tree.arc_action`, one level over. Taking the positional shortcut cost 21 score
points at `TOY` seat 1 and, being systematic, grew worse with more sampled worlds.

Checks (`tests/compiled/test_pimc.py`, `uv run uno pimc exact | pimcn`):

- `value <= best_response(tree, L1, opponent, seat).value`: PIMC-infinity is an L1 policy and
  BR L1 is the best of those (section 18's lesson one level over);
- `believed` equals BR L3 / the two-sided value, per continuation;
- `"prior"` leaves no set without mass, `"posterior"` does;
- every policy is orbit-uniform, and on a small ruleset the quotient's value equals that same
  policy lifted onto the full tree with `CompiledAgent` -- the check that separates the
  tie-break property above from a bug.

### The PIMC agent where no tree exists (`uno/eval/search.py`)

**Determinization at L1.** A PIMC *player* cannot see the opponent's hand either, so
`determinize_l1(rs, obs, rng)` samples it as well: unseen = deck - own hand - discard - the setup's
skipped cards while they are still at the bottom of the pile (after a reshuffle nothing of the pile
is known, as in section 24), split uniformly into `opp_hand_size` cards and the rest of the pile,
with `draws="pile"` and a fresh seed. It is driven from an **observation**, not a state, because
that is all an agent receives. The opponent's `initial_hands` / `own_draws` are fabricated to match
the hand dealt: nothing downstream reads them, but no key of the opponent's seat may be taken from
a determinized state.

**Depth-limited search.** A determinized world has no hidden information and no chance left, so
`search_value` is ordinary alpha-beta cut off at `depth` decisions and evaluated there by a leaf
value function (step 9.5c's network). With two players a skip or reverse means the same player
moves again, so the maximising side is read from `s.player`, never from the parity of the depth.

**The continuation rule (step 9.5e).** `search_value` assumes the opponent is clairvoyant too,
which is classic PIMC and the `"minimax"` continuation above. Given an `opponent`, `PimcSearch`
instead solves each world with `response_value`: max at its own nodes, the **expectation under the
policy the opponent really follows** at the opponent's nodes. That is the sampled analogue of
`best_response_perfect`, i.e. the `"response"` continuation, which the exact toys measure as the
better rule by up to 8.8 score points and most on free play -- which full UNO is. Every ladder
matchup knows its column's policy, so it needs no training. Three consequences worth stating:

- **The row is opponent-specific.** A response agent built against Random is a different policy
  from one built against the Heuristic, so it takes a different name (section 13: a name
  identifies a policy, and two agents sharing one share an action rng stream). On the toys this
  belief-fitting also *costs* exploitability, so the row is labelled with its column.
- **The opponent must be memoryless.** Its policy is evaluated on its L1 observation *inside the
  determinized world*, whose `initial_hands` / `own_draws` for that seat are fabricated, so a
  policy reading its own history would be handed a history that never happened. The baseline
  rulebots read only the current hand and board, so for them this is exact; a memoryful column (a
  Nash strategy, an RL checkpoint with history features) needs the real information state threaded
  through, which `response_value` does not do.
- **No alpha-beta.** Pruning bounds do not propagate through an expectation node, so the response
  path walks the whole depth-limited tree; the pruned minimax path is kept beside it rather than
  unified. At the depth 2 step 9.5d found best, that is cheap.

**Leaves are per seat (step 9.5e).** Step 9.5c fits one value network per seat, and a single agent
object plays both seatings of a deal (`evaluate_matchup` builds the pair once; `sweep` also plays
it reversed), so the seat is unknown at construction and `leaf` may be a `(seat 0, seat 1)` pair
resolved per observation (`leaf_for`). Step 9.5d's `uno` and `sweep` sections passed `nets[0]` at
both seats: both nets return seat-0 values, so it was not a sign error, but at seat 1 the search
evaluated its leaves with a network fitted on a different state distribution, which is why every
seat-1 number of that step is mildly pessimistic.

**Two agents.** `PimcSearch` solves each sampled world (PIMC proper: it assumes it will see
everything from the next decision on -- the strategy fusion the exact toys measure).
`PimcRollout` instead plays each world out with fixed policies, whose rollout players act on their
own L1 observations, so it carries **no fusion**: it is a one-step policy improvement, a different
object, and the toys' bias numbers do not describe it.

**The sampling must be keyed, or AIVAT breaks.** AIVAT subtracts `sum_a sigma(a|h) alpha(h, a)`
using the probabilities the agent reports. An agent that sampled worlds afresh and returned a point
mass would report a deterministic policy while following a stochastic one, and the row correction
would stop being zero-mean. Keying the sampling on the information state makes the policy genuinely
deterministic, so `{a: 1.0}` is the truth. A consequence worth knowing: for such an agent the row
correction is *identically zero*, so AIVAT's gain comes only from chance and the opponent.

**Selection and valuation must not share a sample.** Reading a reference value off the same worlds
used to pick the action is the winner's curse: `max_a Qhat` overshoots `max_a Q` by the expected
maximum of the noise, at every decision, and a sum over a game's decisions accumulates it into tens
of score points. `PimcSearch.scores_split` cuts the agent's own worlds into disjoint halves, one to
choose with and one to value with. The same trap has a second face: a *control* row that is itself
the search must draw its worlds from a different seed than the estimator, or its action is picked
using the half it is then valued on and the total is driven strongly negative. Measured, both
directions cost ~30 score points per game before being controlled.

This is also why a small world count is not merely imprecise but harmful: choosing the argmax of a
few noisy per-action estimates is biased towards whichever action's noise came out high, which is
how a rollout agent at 8 worlds scores *below* `PlayFirstRandom` (`uv run uno pimc sweep`).

**D2 with a search Q.** Section 21's exact decomposition uses the performance-difference lemma, so
its per-decision terms sum to the value gap. A search Q is not a fixed policy's value function and
carries Monte Carlo noise, so nothing telescopes: `uv run uno pimc d2` reports where a row's choices
differ most from the search's preference, by `node_kind`, and its totals are **not** a value
decomposition. Terms may be negative, so the kinds need not partition a positive whole.

Experiment: `uv run uno pimc exact | pimcn | uno | d2 | sweep | belief` (module docstring). The
`belief` section is step 9.5f's opponent-hand inference, section 26.

## 26. The opponent-hand belief (`uno/eval/belief.py`)

Step 9.5f. `determinize_l1` (section 25) splits the unseen cards **uniformly**, which is exactly
`uno.compiled.pimc`'s `"prior"` weighting: PIMC with no inference at all. This section is the
sampled analogue of its `"posterior"`, where a world is weighted by how likely the opponent's
*observed actions* were had they held that hand. On the toys the exact gap between the two is
0.0-0.6 score points against Random and up to 2.6 against the Heuristic, and it only opens under
the `"response"` continuation (under `"minimax"` it is ~0 even on `TOY_FULL`).

### 26.1 The design that does not work, and why

docs/plan.md's original sampled-oracle design is a **particle filter over the deal**: sample the
opponent's initial hand, replay the public history, weight each particle by the opponent's action
probabilities, resample per step. Measured before the rungs below were written, it collapses:

| opponent's play | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| share of 64 particles still alive | 0.156 | 0.018 | 0.002 | 0.0004 | 0 |

A particle survives one observed play only if it happens to hold the played card, which is
`hand_size / unseen` ~ 7/98 on full UNO, so survival decays geometrically and **no particle count
fixes it**: by the fourth play one would need ~10^5 particles, and it keeps decaying. A filter
that resets on collapse (62 % of decisions) scored **worse than the prior it replaced** -- Brier
0.167 against the uniform split's 0.085, its "probability 1.000" bin observed at 0.112.

**The lesson is to condition, never to weight.** A world must be *built* consistent with what was
observed, not sampled and then tested against it. Both rungs below therefore sample the opponent's
hand at the **current** split, where consistency with the hand sizes is automatic, and use the
history only to constrain or to score that sample. Neither propagates anything, so neither can
degenerate.

### 26.2 `"voids"`: what a voluntary draw proves

`voluntary_draw` (section 7) is a *choice*, and no play-first policy takes it holding a playable
card. So at each voluntary draw by the opponent, against the board `(top, colour)` then in force,
their hand held nothing playable. If they have drawn `k` cards since, at most `k` cards of their
hand **now** may be playable against that board -- a necessary condition, exact, and checkable
against a current hand with no propagation at all (`Void(top, colour, since)`, `voids`,
`satisfies`). Penalty draws from a +2 / +4 are excluded: they are not decisions.

On a **forced-play** ruleset the statement needs no policy assumption at all: `legal_actions`
(section 6) offers `Draw` only when nothing is playable, so a draw *proves* the void by the rules.
The rung therefore has teeth on every toy, not only the free-play ones. Share of decisions
carrying at least one constraint:

| ruleset | `TOY` | `TOY2` | `TOY2_SKIP` | `TOY_FULL` | `FULL_UNO` |
|---|---|---|---|---|---|
| forced play | yes | yes | yes | no | no |
| decisions with a constraint | 16.5 % | 23.3 % | 18.3 % | 16.5 % | 73.4 % |

`TOY` and `TOY_FULL` agree to the digit because neither rulebot ever draws voluntarily, so under
that profile the free-play toy plays out exactly as the forced one.

It reads **no opponent policy**, so unlike the `response` continuation it does not make the row
opponent-specific; it is the half that would survive into a general player. Measured on full UNO:
it alone rules out **79.9 %** of `determinize_l1`'s worlds (median surviving share 15.6 %, so
w = 64 is effectively w = 10), and the opponent's real hand violated it **0 times in 2,487**
decisions.

### 26.3 `"window"`: the policy likelihood, exactly, where it is computable

Over the opponent's **trailing run of plays with no draw of theirs in between**, their hand at
each of those decisions is recoverable from the hand they hold now: it is that hand plus the cards
they have played since. So a sampled hand is rolled backwards through that window and scored by
the opponent's real action probabilities (`trailing_window`, `Belief._window_likelihood`) -- the
genuine posterior restricted to the window, not an approximation of it. The window stops at the
first draw going back, because a draw is precisely where the hand stops being recoverable, which
is what defeated 26.1.

Candidates are drawn by **sampling-importance-resampling**: propose uniformly among
void-consistent hands, then draw one proportionally to its window likelihood. Taking the argmax
instead would collapse every world onto a single explanation -- the winner's curse of section 25,
one level over.

### 26.4 Fallbacks, determinism and cost

No consistent hand is found within `tries` attempts at a small share of decisions (4.8 % measured
on full UNO), where the rung falls back to the uniform prior. `Belief.fallbacks` **counts** it: a
report that hides that number hides whether the belief was available at all.

The rungs carry no state across decisions -- there is nothing to propagate -- so the agent stays a
deterministic function of the information state with no cache to invalidate, which is what keeps
AIVAT's row correction identically zero (section 25). Sampling uses `rng.sample` (O(k)) rather
than shuffling the whole unseen pool to take 7 of ~98 cards, which was most of the rung's price.

Cost at w = 32, d = 2, wall clock per game against Random: prior 132.8 ms, `voids` 140.5 ms,
`window` 181.4 ms. The belief is consulted once per world at the agent's own decisions only, so it
is a **+6 % / +37 %** addition and not the multiple a per-decision measurement suggests.

### 26.5 Checks (`tests/eval/test_belief.py`)

The dangerous failure of any inference is **over-pruning**: a belief that excludes the world the
opponent is actually in looks confident and is wrong, and nothing downstream would notice. So the
load-bearing tests assert the truth survives, one per rung -- `voids` never excludes the hand the
opponent really holds (on `FULL_UNO` and `TOY_FULL`), and the `window` likelihood is strictly
positive for it, since the opponent demonstrably made those plays from that hand. Beside them sits
a non-vacuity test (the void constraint must actually exclude a measurable share of uniform
worlds), because a test of the first kind would pass unchanged with `voids` returning `[]` --
the step 9.5e lesson. A determinized world is still checked consistent with the observation, and
the agent is still checked to be a function of its information state with a belief wired in.

Experiments: `uv run uno pimc belief [deals] [worlds] [depth] [workers]` scores the ladder, and
`uv run uno pimc calibration [deals] [samples]` checks that a belief is *correct* rather than
merely confident -- the instrument that refuted 26.1, kept in the repo so a confidently wrong
belief cannot pass unnoticed again. It carries two pre-declared checks: no rung may be materially
worse than the exact prior, and no populated reliability bin may miss by more than 0.25 (the
refuted filter missed by 0.89). Measured: Brier 0.0846 / 0.0828 / 0.0825 for prior / `voids` /
`window`, worst bin gap 0.088 / 0.093 / 0.135. `window` is the least calibrated of the three while
being marginally the sharpest, as a *windowed* posterior should be.
