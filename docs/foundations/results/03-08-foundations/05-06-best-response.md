# Steps 5-6 results: exact best responses on `TOY` (2026-09-14, `uno/reference/solve.py`, `uno ladder`)

> Commit `30a1e1a`, re-run at `0ab8e8d` (2026-09-14). Rerun: `uv run uno ladder TOY` (reference solvers, minutes). The Heuristic columns are the step 4 Heuristic; the step 8.6 values are in [step 8.6](08.6-symmetry.md) and [index](../index.md). *Step 8.8: the on-seeds check here compares against the total-variance CI, which is conservative ([step 8.8](08.8-review.md)).*

One best-response routine, three information-set keys (L1 `info_state`, L2 =
L1 + opponent's initial hand + opponent's draws in order, L3 = full state with
pile order). L1/L2 walk the lazy tree; L3 enumerates *worlds* (deal x pile
order) and walks the bare engine. (The first version keyed L2 on the
opponent's *current* hand to keep its memory out of L2; that partition has
imperfect recall, see "Information sets and recall" in [the plan](../../history/original-plan.md). Same values on
Random and Heuristic to 4 decimals.)

Cost on `TOY`: 12-16 s and ~1 GB per L1/L2 solve against Random (0.94M
histories reachable under Random's support, out of 1.25M), 4-7 s against the
deterministic Heuristic. No array-compiled tree yet: that is for CFR (step 8).

Checks (all in `tests/reference/test_solve.py`): value == max over every pure strategy
on tiny rulesets (independent brute force); L3 worlds == distinct deck
permutations with their frequencies; table replayed through `evaluate` ==
solver value; L1 <= L2 <= L3; BR dominates Random / Heuristic / UniformLegal
as responder; the three levels coincide on a pile-0 ruleset; simulation
matches the exact value on the same deals.

**Deal luck is visible at 4000 seeds.** The first simulation check (exact
value inside the plain CI) failed for seat 1: seeds 0-4000 favour seat 0 by
~1.5 points, and every measurement on that seed range shifts together (the
step 4 table too: random vs random sampled 0.656 / 0.344, exact 0.649 /
0.351). `evaluate_on_seeds` computes the exact value on the dealt seeds with
their real pile order (bare-engine expansion from the real initial state), and
the simulation matches that to within action-sampling noise only; for two
deterministic agents the two are equal. A
1.5-point shift at 4000 games is 2 sigma, the miss a plain 95 % CI test
produces one time in twenty by construction; the on-seeds comparison is the
test, and it now covers L3 too. This
is the mirrored evaluator's and AIVAT's job in numbers: at 4000 games the
sampling noise of the deal alone (1.5 points) is larger than the L3 - L2 gap
(a value of information, 0.2-0.9 points): different units, but the noise
swamps the signal all the same.

Exact values of the row agents (score of the first-named, seat 0 / seat 1;
step 4 sampled values in brackets):

| matchup                | seat 0          | seat 1          |
|------------------------|-----------------|-----------------|
| random vs random       | 0.6493 [0.656]  | 0.3507 [0.344]  |
| heuristic vs random    | 0.7390 [0.738]  | 0.4077 [0.399]  |
| heuristic vs heuristic | 0.6891 [0.698]  | 0.3109 [0.302]  |

Best-response columns (exact score of the responder, per seat):

| opponent  | seat | BR L1  | BR L2  | BR L3  | L2-L1 | L3-L2 | row agent itself |
|-----------|------|--------|--------|--------|-------|-------|------------------|
| random    | 0    | 0.7934 | 0.8188 | 0.8280 | +2.5  | +0.9  | 0.6493 (random)  |
| random    | 1    | 0.5513 | 0.5937 | 0.5989 | +4.2  | +0.5  | 0.3507 (random)  |
| heuristic | 0    | 0.7447 | 0.7770 | 0.7837 | +3.2  | +0.7  | 0.6891 (heur.)   |
| heuristic | 1    | 0.4525 | 0.4806 | 0.4825 | +2.8  | +0.2  | 0.3109 (heur.)   |

Re-run 2026-09-14 after the L2-key / on-seeds fixes: all 12 best-response
values unchanged to 4 decimals; every row passes the on-seeds check, L3 now
included (`evaluate` builds its default roots lazily, so the per-seed L3 value
costs 0.8 s for 4000 seeds instead of 15 min). Against the deterministic
Heuristic the simulated score equals the on-seeds exact value to the last
digit, as it must; against Random the residual is 0.2-1.1 points of action
noise, inside the CI. The seat-1 deal luck of seeds 0-4000 is visible as the
on-seeds column sitting ~1.1 points under the exact column on every seat-1 row.

Reading:

1. **There is skill to learn at L1.** Against Random, the L1 ceiling is +14.4
   points over Random itself from seat 0 and +20.1 from seat 1; the Heuristic
   only takes +9.0 / +5.7 of that. So the Heuristic leaves 5.4 / 14.4 points
   on the table against Random, most of it from seat 1. The old "60-70 %
   rulebot, PPO no better" story is testable now: on this toy an L1 learner
   has ~5-14 points of room above the rulebot.
2. **Hidden information is worth 2.5-4.2 points, the pile order 0.2-0.9.** The
   L3 - L2 gap is small because the pile holds 2 cards; the L2 - L1 gap (the
   opponent's hand) is the bigger prize. BR L1 already infers the opponent's
   hand as well as the public history allows, so the L2 - L1 gap is the part
   that *no* inference can recover; the room for opponent modelling in an L1
   agent is BR L1 minus the agent's own value (5.4 / 14.4 points for the
   Heuristic against Random).
3. **Seat dominates everything.** BR L1 vs Random: 0.79 from seat 0, 0.55 from
   seat 1. Even the L3 oracle moving second against Random only reaches 0.60.
   All ladder cells stay per seat (rule from step 3).
4. **The Heuristic is exploitable but not badly**: BR L1 to it gets 0.745 /
   0.453, less than BR L1 to Random (0.793 / 0.551), so it is the stronger
   opponent, as expected. Its exploitability at L1 is BR L1 against it minus
   the Nash L1 value at that seat (step 8); that difference, not the BR score
   itself, is the quality measure for step 10 onwards.
