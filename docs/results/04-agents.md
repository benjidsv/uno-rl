# Step 4 results: agents and runner

> Commits `fbca563` (step 4) and `30a1e1a` (step 4.5), 2026-09-14. Sampled with `uno.eval.play.match` (the `census` experiment prints the same matchups). The Heuristic of this step broke ties on the lowest type id; since [step 8.6](08.6-symmetry.md) it is symmetric and its numbers changed, current values in [index](index.md).

`uno/agents/baseline.py`: `UniformLegal` (sanity), `PlayFirstRandom` (the ladder's
Random row), `Heuristic` (deterministic). Agents return distributions.
`uno/eval/play.py`: `play_game` records (state, obs, probs, action) per decision;
action rng seeded from (seed, L1 info state) = common random numbers;
`match` gives score with normal CI, win rate with Wilson CI, end kinds.

`TOY`, 4000 seeds, score for the first-named agent (draws = 1/2):

| matchup                     | seat 0 | seat 1 |
|-----------------------------|--------|--------|
| random vs random            | 0.656  | 0.344  |
| heuristic vs random         | 0.738  | 0.399  |
| heuristic vs heuristic      | 0.698  | 0.302  |

Heuristic edge over random: +8 points from seat 0, +5.5 from seat 1. (On the
rejected 2-color toy it was +10 / -3: the heuristic lost to random when moving
second, another symptom of the pure race.) Exhaustion endings: 19 % of games
under play-first random; 12 % of all games are ties (about 60 % of the
exhaustion endings).


# Step 4.5: fixes (2026-09-14)

- `step(state, action, check=False)` skips the legality check; the tree walk
  uses it (it just computed `legal_actions`). Census pass 9.5 s -> ~8 s.
- `match` only treats a `ValueError` from the *deal* as an invalid seed; an
  error inside a game now propagates instead of being counted as invalid.
- The stable-hash test asserted `x == constant or True`; pinned the constant.
- Observation "leak" at non-decision nodes: not fixable and not a problem. The
  leak is the timing (opponent still to move after their own draw event =>
  the drawn card was playable), not a field. Documented in [spec section 10](../spec.md):
  L1/L2 observations are only meaningful at the observer's own decisions.
- Pass after a playable draw stays legal under forced play (intended).
