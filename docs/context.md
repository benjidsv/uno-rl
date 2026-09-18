# UNO RL v5: project context

**Status update, 2026-09-18:** all of **10a.3, 10B and 10C** are complete. Step
10C froze four cheap policies as an equal-weight league after an exact toy
transfer study and full-UNO DEV selection, then confirmed their scores and
behavior on fresh EVAL block 03 ([results](results/10c-results.md)). The minimal
memory ablation has the best held-out aggregate, but no variant establishes an
improvement over the Heuristic across both seats and columns. The coarse-memory
evidence has been repaired and refreshed (10a.3A); the learner observation
contract is defined with three comparison arms (10a.3B,
[docs/observation-contract.md](observation-contract.md)); and the **LBR attack
instrument** is built and calibrated (10a.3C, [spec section 32](spec.md),
[results](results/10a3c-results.md)). **Step 11** follows. See
[the execution plan](10-to-11-plan.md), [the review](results/10-review.md), and
[the repair results](results/10a3a-results.md). Older memory-sufficiency claims
below must be read with this update. A side diagnostic of the same day, **9.5g**
(section 5.26, [results](results/09.5g-results.md)), declines a proposal to replace
PIMC's uniform world sampling and leaves one fact behind that 10C should carry:
PIMC's *action choice* at the working set is ~12 % noisy while its *score* is flat.

The project now has a luck-free quality measure that does not need an exact tree:
an attack's achieved score bounds NashConv from below (`NashConv >= b0 + b1 - 1`),
calibrated on the toys at 77-97 % of each target's *exact* NashConv. It is a
**lower** bound — a small one means a weak attack, never a safe checkpoint.

A self-contained briefing on the project: the goal, the method, why each
technique was chosen, what it bought, what went wrong, and what the numbers say.
Written to be pasted into a fresh conversation (brainstorming, review, planning)
with no access to the repo. State as of **step 10C completion, 2026-09-18** (on
top of commit `6ffe6f5`): the seed, reporting,
training-target, shaping, belief, column-free-player, corrected toy-memory
contracts, learner observation contract, calibrated attack instrument, 10B
figures, and frozen 10C rulebot league are built. Next is Step 11.

The repo's detailed sources, if you have them: `docs/plan.md` (build order,
motivation), `docs/spec.md` (the rules and every algorithm; it wins over the
code), `docs/results/<step>.md` (one file per step, numbers as measured then),
`docs/results/index.md` (current headline numbers). If this file and
`index.md` disagree on a number, `index.md` is right.

---

## 1. Why this project exists

Fifth iteration of RL on 1v1 UNO. Earlier rounds (rlcard + PPO):

- rulebots reached ~60 % raw winrate against a random player, a genetically
  tuned parameterised rulebot ~70 %;
- PPO, including self-play, never measurably beat that.

The core problem is **skill vs luck**: UNO's outcome variance is huge, the
reward is sparse (win/loss at the end), and most of a game is decided by the
draw. Three suspected causes, to be *tested, not assumed*:

1. **The evaluator was too noisy** to see small improvements. A 3-point edge
   near 60 % is one sigma at ~270 games and three sigma at ~2400 (where a
   3-sigma test still misses a real edge half the time; ~3,900 for 80 %
   power); no fixed evaluation budget was ever set.
2. **rlcard's rules were inconsistent** with its own legal-move function, so
   the game may have had a low or buggy skill ceiling.
3. **The ceiling was never known**, so "no improvement" could have meant
   "already optimal".

Hence the thesis of v5: **measurement first, learning second.** No RL until a
"ladder" of matchup values is filled with *exact* numbers on a toy UNO small
enough to solve. A second purpose emerged along the way: the exact toys are a
**testbed for the sampled methods** planned for full UNO (AIVAT, PIMC, MCTS):
implement on the toy, check against the exact value (bias and variance), then
trust them on the full game with some re-testing.

Status of the three suspected causes today:

| cause | status |
|-------|--------|
| noisy evaluator | **supported.** On the toy a 3-point edge at one seat reaches 3 sigma at 1,700-2,400 raw games (50 % power; 2,800-3,900 for 80 %). On full UNO, pairing Heuristic against Random removed only ~10 % of the variance, so a 3-point comparison reaches 3 sigma at ~9,000 games either way (~15,000 at 80 % power); between near-identical variants on a shared action-rng stream pairing removes 77-85 % (1,250-1,800 deals, step 9). AIVAT with a plain logistic critic removes 24-29 % between dissimilar policies (~3,900 → ~2,900 deals at 80 % power, step 9.5b). Iterations 1-4 never had that budget. |
| buggy rules | sidestepped: own engine with a written spec and property tests. Not measured on rlcard itself. |
| unknown ceiling | **measured on the toys**: the exact best response beats Random by 14-20 points at the same seat, and the rulebot ("Heuristic") leaves 5-14 points on the table. The full-UNO ceiling is still unknown. |

## 2. Vocabulary (used everywhere below)

- **Score**: P(win) of the named player with draws worth 1/2, i.e.
  `(return + 1) / 2`, returns in {-1, 0, +1}.
- **Seat 0 / seat 1**: seat 0 moves first. On the toys seat 0 is worth a lot,
  so **every matchup is reported per seat**, never pooled. A skill check
  compares with the random baseline *at the same seat*, not with 0.5.
- **Seat gap**: score of a profile at seat 0 minus its score at seat 1, in
  points. **Edge**: seat-0 score minus 0.5 (half the gap for self-play).
- **Information levels** (a property of the observer, not the game):
  - **L1**: own hand + public information. What a real player / the RL agent
    sees. The information state is (seat, initial hand, full public history,
    own draws *in order*), which gives **perfect recall**.
  - **L2**: L1 + the opponent's hand, carried as the opponent's initial hand
    and draws in order (keeps perfect recall).
  - **L3**: L2 + the draw-pile order: perfect information, an oracle.
- **Best response (BR Lx)**: the best strategy against a *fixed, known*
  opponent policy, computed at information level x. Exact, one backward pass.
  One-sided L2 and L3 "optimal" only exist as best responses: once the
  opponent's policy is known and their hand visible there is nothing hidden to
  be in equilibrium about. **Two-sided** games, where both players see both hands
  (L2) or everything (L3), do have minimax values (`two_sided`, step 9).
- **Nash L1**: the only level where "optimal" can mean an equilibrium (both
  players hide information). Computed with **CFR** (counterfactual regret
  minimisation). The Nash *value* is unique, the strategy is not, so every
  cell using the Nash agent carries its iteration count and NashConv.
- **Exploitability** of an agent at seat s: the BR L1 score against it minus
  the Nash score of the responder's seat, in points. **NashConv** = sum over
  the two seats. It is independent of luck: the single best quality measure
  for an RL checkpoint.
- **Values of information**: L2 - L1 (worth of seeing the opponent's hand
  *beyond what can be inferred*: BR L1 already does exact Bayesian inference)
  and L3 - L2 (worth of knowing the pile order). These are ceilings moving,
  **not variance components**; never compare them with sampling noise as the
  same unit. The headroom for opponent modelling in an L1 agent is BR L1 minus
  the agent's own value, never L2 - L1.
- **The ladder**: rows are agents (Random, Heuristic, Nash L1, BR L1 to
  Random, later RL agents), columns are opponents (Random, Heuristic, Nash L1,
  BR L1 / L2 / L3 *to the row*). Each cell: seat 0 / seat 1 values, by
  estimator (raw, mirrored, AIVAT) and exact on the toy.
- **Seeds**: `DEV_SEEDS` 0-3999 for building, testing, tuning;
  `EVAL_SEEDS` 1,000,000-1,003,999 held out for reported numbers, nothing is
  ever chosen on them.

## 3. The game and the engine: design decisions and why

**Own engine, not rlcard or pyspiel.** rlcard's rules are unreliable, and the
project needs clone/replay for search, best response and AIVAT. pyspiel's
objects don't fit the oracle levels; it would only be useful for CFR/BR on L1,
and that turned out to be easy to write (and to vectorise) ourselves. Pure
Python 3.14 + numpy, managed with uv.

**One parametrised ruleset.** Toy and full UNO are the same code with
different parameters: colors, copies per rank, skip/reverse/+2 per color,
wilds, wild +4, hand size, `draw_then_may_play`, `forced_play`, `reshuffle`
(`none` or `seeded`), `max_turns`. Two players always. Since step 9 also
`draws` (`pile` or `tickets`), which is not a rule: same game, different
coupling of games on one seed (section 5.14).

**Deterministic and immutable.** All chance is the initial permutation (plus
seeded reshuffles on full UNO). A game is a pure function of (permutation,
seed, actions). States are frozen; `step(state, action) -> state`. This makes
cloning, replay, mirrored evaluation and exact solving trivial.

**Cards are a multiset of types.** Two red 3s are the same type everywhere, so
duplicate deals collapse in exact enumeration.

**Rules worth knowing** (full detail in the spec):

- Playable: wild, or same color, or same number, or same action kind.
- Wilds are joint actions (card, chosen color). No stacking, no challenge,
  +4 always playable. Skip and reverse both mean "move again" with 2 players.
- After a voluntary draw, if `draw_then_may_play` and the card is playable,
  the player may play it or **Pass**. Pass writes no public event: the opponent
  cannot tell "could not play" from "chose not to".
- **Forced play**: if a card in hand is playable, drawing is illegal. Used on
  the toys to control tree size (section 4); the free versions (`_FULL`) keep
  the voluntary draw.
- **Toys have no reshuffle.** Empty pile + a required draw = game over by
  **exhaustion**: fewer cards wins, equal is a draw. Deliberate: it turns
  official UNO's "keep your hand low" scoring into the terminal condition
  without infinite loops. Known distortion: **pile parity**. Whoever draws the
  last card hands the opponent the option to end the game, and on free-play
  toys a player can draw on an empty pile to *claim* the game when ahead. It is
  real, computable skill on the toy but does not exist in full UNO. Measured
  impact turned out small (section 6.8).
- Full UNO: 108 cards, hand 7, seeded reshuffle, cap 1000 actions (a draw).

**An observation leak that is not a bug.** Observing a state where the
opponent is still to move right after their draw reveals the drawn card was
playable. It is timing, not a field, so L1/L2 observations are only defined at
the observer's own decision nodes. Agents and solvers only use those.

**Agents return distributions, never sampled actions**, because best response
tabulates a policy and AIVAT needs action probabilities. The runner samples
with an rng keyed on **(match seed, agent name, information state)**: common
random numbers (same agent, same situation, same choice across games and
opponents). The name is in the key so two different stochastic agents never
share a stream (section 5.5). So **a name must identify a policy**: two different
agents with one name raise; self-play passes the same object twice.

**Baseline agents:**

- `UniformLegal`: uniform over every legal action, including silly draws and
  passes. Sanity only.
- `PlayFirstRandom`: **the ladder's "Random" row.** Uniform over legal plays,
  draws only when nothing plays, always plays a playable drawn card.
- `Heuristic`: a rulebot. Play over draw; action cards first when the opponent
  holds ≤ 2 cards; then the color held most; then keep the current color; then
  a number card; ties uniform. Wilds held unless nothing else plays; wild color
  = color held most. **Symmetric** under color/rank renaming since step 8.6
  (before, ties went to the lowest type id; numbers from steps 3-8.5 used that
  version). Never draws voluntarily.

## 4. The toys: how they were chosen

Target for a toy: full tree ≤ ~10^6 histories so BR and CFR run in seconds,
and it must keep the decisions that carry skill in UNO (which color to play,
hold vs spend the wild, wild color choice from inferred voids, draw vs play,
action-card timing).

**Step 3, round 1 (size).** The planned candidate, 2 colors x 3 ranks x 2
copies + wild, hand 2 (13 cards), was **> 4·10^7 histories: rejected.** What
drives size is the pile: every voluntary draw branches over ~5 types, so
histories grow like (types x branching)^pile. **Forced play shrank a
candidate 65x** without changing anything the play-first agents do (they never
draw voluntarily).

**Step 3, round 2 (tempo).** The round-1 pick (2c x (2,2) + W, forced) had
seat 0 winning **72 %** of random games. Diagnosis: 59 % of games had no draw
at all and seat 0 won 100 % of those. With so few colors every card plays on
every top, nobody gets stuck, and the game is a race the first mover wins by
construction. **Per-card playability is the lever**, not deck size. Candidates
with more colors cut the effect. Hand 4 on 9 cards left a 0-card pile (71 %
exhaustion): degenerate.

**`TOY`** = 4 colors x ranks (1, 1), one copy each, 1 wild, hand 3, forced
play. 9 cards, pile 2. 1,252,040 histories, 558,656 L1 sets. Seat 0 still
worth ~15 points (random vs random 0.649 / 0.351). Caveat recorded at the
time: the toy was picked after looking at outcomes (the tempo criterion was
added once 72 % showed up), so steps 5-6 were the first numbers on a ruleset
fixed *before* running them.

**Step 8.5 asked "does the toy transfer?"** (section 6.7). Answer: the gap was
not size but **missing decisions** (the voluntary draw is *available* at 60 % of
full-UNO decisions, action cards among other plays at 20 %, both absent on `TOY`)
and **game length**.

**`TOY_FULL`** = `TOY` without forced play. 774k quotient histories. Measures
the worth of the voluntary draw. Solvable only after the symmetry quotient
(step 8.6).

**Step 8.7 sizing (position DP, section 5.10)** over ~45 candidates:

- **13-card decks are out of reach**: ≥ 34M quotient histories forced, over a
  billion free. Would need a new representation, not a faster walker.
- **Free play multiplies the tree 20-400x.**
- **Seat parity moves cheaply with a third rank or a skip on 3 colors**
  (random vs random 0.649 → 0.588 at 312k histories). Bigger decks buy
  nothing more (5 colors + 2 wilds: 76M histories for 0.585).
- **Things that went the wrong way**: duplicate ranks bring the race back
  (0.66); hand 2 on a longer pile (0.61); forbidding play after a draw makes
  `TOY` worse (0.672); +2 cards at 10 cards end 58 % of games by exhaustion;
  hand 4 on a 1-card pile is 36-38 % exhaustion.

Picked: **`TOY2`** = 3 colors x ranks (1,1,1) + 1 wild, hand 3, forced
(10 cards, pile 3, 312k quotient histories). **`TOY2_FULL`** = same, free
(28M, the only free candidate with a real pile inside the 20-50M memory
budget). **`TOY2_SKIP`** = 3 colors x ranks (1,1) + 1 skip per color + wild,
forced (923k). Kept as the action-card toy; its free version (77M) is out of
reach.

| ruleset | deck / pile | forced | quotient histories | L1 sets | status |
|---------|-------------|--------|--------------------|---------|--------|
| `TOY` | 9 / 2 | yes | 33,646 (full 1,252,040) | 12,036 (full 558,656) | solved |
| `TOY_FULL` | 9 / 2 | no | 774,007 | 138,876 | solved |
| `TOY2` | 10 / 3 | yes | 311,819 | 129,428 | solved |
| `TOY2_SKIP` | 10 / 3 | yes | 922,744 | 439,268 | solved |
| `TOY2_FULL` | 10 / 3 | no | 28,057,516 | 3,753,530 | solved (DCFR) |
| `FULL_UNO` | 108 | no | n/a | n/a | sampled only |

## 5. Methods: why, what they bought, what went wrong

Ordered roughly as built. Each entry: the problem, the technique, the gain,
and the bugs or corrections it produced.

### 5.1 Lazy-chance tree instead of permutations (step 3)

Problem: exact solving by enumerating deck permutations explodes. Technique:
a draw is a chance node over the *distinct types* left in the unknown pile;
the deal is a chance node over (hand multisets, setup prefix) with
multivariate hypergeometric probabilities. L3 instead enumerates *worlds*
(deal x pile order) and runs the bare engine below. Cross-checked against brute
force permutation enumeration on tiny decks. Pure Python walks ~150k
histories/s.

**Bug (found step 8.9):** the hand-multiset probability multiplied by the
number of orderings at every recursion level instead of once. Wrong for hands
of 3+ from decks with duplicate cards (masses summed to 1.7). **No named toy
was affected** (one copy per card, so the error was a constant factor removed
by normalisation), but step 8.7 sizing rows with duplicate ranks or 2-3
identical wilds moved by up to 0.4 points (sizes unchanged, no decision
changed). Missed because the solver tests only generated hands of 1-2 cards.
Now pinned by tests that fail on the old code.

### 5.2 Exact best response, own implementation (steps 5-6)

Two passes: forward, collecting counterfactual reach of every history (chance
x opponent, never the responder) grouped by information set; backward,
memoised, each set picks the action maximising the reach-weighted sum of its
histories' continuation values. Valid because every level has perfect recall.
Validated against brute force over all pure strategies on tiny rulesets. Cost
on `TOY` in pure Python: 12-16 s and ~1 GB per solve. That was enough for the
twelve step 5-6 values, too slow for CFR.

**Imperfect-recall key (step 6, fixed same day).** L2 was first keyed on
L1 + the opponent's *current* hand. That loses which card the opponent drew,
so 13-23 % of L2 sets merged histories with different opponent memories. The
values were identical to 4 decimals against Random and Heuristic (neither
has memory), but a memoryful opponent (CFR, an RL agent) could have been
under-reported. Now keyed on the opponent's initial hand + draws in order, and
the solver asserts all histories of a set share legal actions.

### 5.3 Deal luck and the on-seeds check (steps 5-8.8)

**Finding:** a fixed seed range carries its own deal luck. Seeds 0-3999 favour
seat 0 by ~1.5 points on `TOY`, and every simulation on that range shifts
together. A plain "exact value inside the 95 % CI" test failed at seat 1 for
that reason (a test like that also fails 1 time in 20 by construction). At 4000
games the deal noise alone (1.5 points) exceeds the L3 - L2 gap. That is the
job of mirroring and AIVAT in one number.

**Technique:** `evaluate_on_seeds` computes the exact value on the dealt seeds
with their real pile order, so a simulation is compared with the exact value
*of the same games*, and differs only by action-sampling noise (exactly zero
for two deterministic agents).

**Held-out seeds (step 7):** `EVAL_SEEDS` introduced because the dev range's
tilt becomes a *bias* the moment anything is chosen for scoring well there.

**Correction (step 8.8, M1):** the check compared the difference with the
simulated score's CI, which counts deal and pile variance the seeds already
fix. So it was conservative and weak at catching bugs. Now **z = (sim - exact
on seeds) / sqrt(mean Var(score | seed) / n), |z| ≤ 3** (0.27 % false alarms
per check), computed exactly from each seed's outcome distribution.

The one lingering miss: `TOY2`, BR L2 seat 0 vs Random, z = -3.01 on the
held-out seeds. Examined on six disjoint ranges of 4000 seeds: pooled over
24,000 seeds z = -1.48 (L1 +0.07, L3 -0.42), and the three levels share the
held-out range's low draw (same Random streams on the same seeds). About 100
checks per full run make one |z| just over 3 roughly a 1-in-4 event. Judged
chance; threshold kept at 3 (chosen before the re-run).

### 5.4 Mirrored evaluator (step 7)

**Technique:** play every seed from both seats (agent A at seat 0 vs B, then
swapped, same deal and pile order), keep the traces, and compute every
estimator as a pure function of the traces ("play once, estimate many ways";
AIVAT becomes a fourth estimator on the same games). Pair score = mean of the
two games. It estimates the **same quantity** as seat-averaged raw winrate; the
whole gain is the negative covariance between the two seats' scores on a deal.
Per-seat numbers are always reported next to it (the pair score alone pools
seats). `exact_mirrored` computes the exact means, variances, covariance and
split share over all 10,080 worlds of `TOY`, so sampled moments can be
validated.

**What it bought on `TOY` (exact):**

| matchup | corr(seat 0, seat 1) | split pairs | var ratio mirrored / raw | games for a 3-pt edge at 3σ (50 % power), raw → mirrored |
|---------|----------------------|-------------|--------------------------|--------------------------------------------|
| heuristic vs random | -0.30 | 0.61 | 0.70 | 1993 → 1404 |
| BR L1(random) vs random | -0.41 | 0.55 | 0.61 | 1815 → 1098 |
| BR L1(heuristic) vs heuristic | -0.61 | 0.77 | 0.39 | 2100 → 828 |
| BR L1(random) vs heuristic | -0.65 | 0.80 | 0.36 | 2112 → 753 |

Mirroring removes **30-65 %** of the variance, more when the deal decides more.
Same picture on `TOY2` (30-50 %). The split share is **not** the share of
games decided by the deal: two independent games at these per-seat scores
already split 39-50 % of the time, so the deal accounts for the excess, 13-29
points (exact on `TOY`, step 9 review, scratch), and only 4 % of `TOY` worlds
have a result no play can change. The correlation is the luck readout.
*(Corrected in step 9: this read "3 games in 5 are decided before anyone
moves".)*

**Mirroring does not transfer to full UNO** (step 8.5): corr -0.10, var ratio
0.90. The luck there is in the draws during a ~44-action game, so the two games
of a pair decouple almost immediately.

**Self-play pairs are degenerate** under common random numbers: both games of an
A-vs-A pair are the same game, every pair scores 1/2. A mirrored self-play cell
says nothing; compare two agents through a third fixed column.

### 5.5 Paired comparisons and common random numbers (step 7)

**Technique:** `compare(a, b)`: per seat, the difference between two row agents
that played the *same seeds* against the *same column*; the column's rng is
keyed on the information state, so it makes the same choices in both matchups
until histories diverge.

**What it bought on `TOY`:** heuristic - random vs random measured with 45 %
of the unpaired variance; best response - heuristic with 20-30 %; two
near-identical policies (0.9 points apart) with **4-8 %**, i.e. **370-570
games instead of 7,000-9,500** to see 3 points at 3σ (50 % power; 1.64x for
80 %). For steps 10-11, "is this
checkpoint better than the rulebot" is always a paired comparison, never two raw
winrates. `TOY2`: 55-90 % removed.

**On full UNO pairing Heuristic against Random removes only ~10 %** (var ratio
0.89); ~9,000 games for a 3-point comparison at 3σ (~15,000 at 80 % power).
**Between near-identical variants on one action-rng stream it removes 77-85 %**
(step 9: the Heuristic against a 5 %-Random mixture or a changed threshold,
1,250-1,800 deals for 3 points at 3σ with 80 % power instead of ~8,000). The same
mixture on its own rng stream keeps 54-72 % of the variance: checkpoints of one
policy must share a stream (`stream`, spec section 13). Ticket draws (spec
section 8, same game, a player's k-th draw coupled across games) cut another
3-20 %. AIVAT (step 9.5) remains the estimator for dissimilar policies.

**Bug (step 8.6): rng coupling.** With the Heuristic now stochastic on ties,
the sampled seat correlation came out -0.364 against exact -0.300 (4.5σ). The
rng was keyed on (seed, info state) only, so the Heuristic in game 2 and Random
in game 1 (same seat, same info states) drew the *same* random numbers, coupling
the pair beyond the shared deal. Deterministic agents never triggered it. Fixed
by adding the agent name to the key; since step 8.9 two different agents with
the same name raise. Name rather than object id so runs reproduce across
processes and the same policy loaded twice keeps common random numbers. Watch
the defaults `CompiledAgent(name="compiled")` and `nash_l1`: give checkpoints
their own names.

**Correction (step 8.8, M4):** sampled moments (correlation, splits, variance
ratio) were printed next to exact ones but never *tested*. Now seeded percentile
bootstrap CIs over pairs, 99.8 % per moment (family-wise 99 % over the five
moments of a matchup, Bonferroni). A first 99 %-per-moment version flagged the
split share on the dev seeds (z +2.1); five disjoint seed ranges pooled gave z
+0.21, so no residual coupling, just dev seeds drawing high.

### 5.6 Compiled tree: numpy arrays instead of Python walkers (step 8)

**Problem:** CFR needs hundreds of passes; the Python walkers cost 7-16 s each.
**Technique:** walk the tree **once** into flat arrays (nodes, arcs = (node,
legal action), edges = (arc, chance outcome) with probabilities), sorted by a
**level** so each level is a contiguous slice. Every exact computation (reach,
values, evaluate, outcome distributions, BR at L1/L2/L3, exploitability, exact
mirrored moments, CFR) is a loop over levels of a few `reduceat` / fancy-index
operations. The pure-Python walkers stay as the trusted reference, and tests pin
the compiled versions to them.

Two things had to be invented:

- **A level constant across an information set**: `2 * len(public history) +
  [phase == TURN_START]`. Depth by action count fails: the opponent's hidden
  "drew a playable card and passed" is one action longer than "drew an
  unplayable card" and looks identical.
- **Observations rebuilt from keys** (`observation_from_key`), so an agent can be
  tabulated over 559k sets without keeping states; property-tested against
  `observe` at every decision of random games, all levels.

**Gain:** 45-130x (BR L1 13.2 s → 0.10 s; evaluate 1.8 s → 0.04 s). One CFR+
iteration 0.2 s on the full `TOY` tree.

**Bug the small tests missed:** the first compiled BR passed every hypothesis
test on tiny rulesets and matched all L2/L3 values on `TOY`, yet was 0.2-0.6
points *below* the reference at L1. The seat arguments of the forward pass
were swapped, so counterfactual reach held chance only and dropped the
opponent's probabilities. L2/L3 sets are (nearly) singletons, where reach
weights cannot matter, and on tiny decks the weights never changed an argmax.
Caught by replaying the reference's table on the compiled tree and by the twelve
known `TOY` values. **Lesson: pin known values on the real toy, not only
properties on tiny rulesets; singleton information sets hide reach bugs.**

### 5.7 CFR+ and faster variants (steps 8, 8.7)

**CFR+** (regret matching+, alternating updates, linear averaging) on the L1
keying of the compiled tree. On `TOY`: NashConv 38.3 pts at t=1, 1.41 at 10,
0.079 at 50, 0.008 at 200, **0.001 at 1000**. **40x ahead of vanilla CFR** at
equal iterations (vanilla fell roughly as 1/T here, faster than its 1/√T bound;
an early doc wrongly called 1/√T the observed curve). Every run logs
exploitability (two exact BRs) and the bracket `[1 - BR seat 1, BR seat 0]` that
must contain the Nash seat-0 score, so a stalled solver would be seen, not
trusted.

**DCFR** (Brown & Sandholm 2019: discount positive regrets by t^1.5/(t^1.5+1),
negatives by 1/2, quadratic averaging) and **PCFR+** (Farina et al. 2021:
optimistic step with the last instantaneous regret) added for `TOY2_FULL`,
where one iteration is ~4.5-6.6 s. Race over 300 iterations on `TOY`, `TOY2`,
`TOY_FULL`: **both reach 0.01 points in 2-4x fewer iterations than CFR+**
(CFR+ ~150-180, DCFR ~50-60, PCFR+ ~45-80), at the same cost per iteration.
DCFR and PCFR+ tie within noise; their combination adds nothing. DCFR is the
default above ~1M nodes.

Corrections logged in step 8.8: the gain was first written as "4-7x, 30-50
iterations everywhere" (really 2-4x). The `TOY2_FULL` run was planned as "DCFR
150 → NashConv 0.001, two hours into 17 minutes"; it reached **0.005** in 15
minutes, and at equal accuracy the saving over CFR+ is ~2x (CFR+ would need
25-35 min for 0.005, two hours for 0.001).

### 5.8 Symmetry quotient (step 8.6): the lever that made bigger toys possible

**Observation:** the rules never name a color or rank. Renaming colors (any
permutation) and ranks (among ranks with equal copy counts) maps the game onto
itself. Group order |G| = 48 on `TOY`, 36 on `TOY2`, 12 on `TOY2_SKIP`.
**Technique:** one subtree per *orbit* of deals (weighted by the orbit's mass),
one information set per orbit of keys (canonical key = lexicographic minimum
over G). Values on the quotient equal the full game's **whenever every policy
involved is symmetric**.

**Gain on `TOY`:** 5,600 deals → 144 orbits; 1,252,040 → 33,646 histories
(37x); 558,656 → 12,036 L1 sets (46x). BR 0.35 s → 0.01 s; CFR+ to NashConv
0.001 **203 s → 5.2 s**; tree 179 MB → 4.2 MB. The group grows factorially with
the deck, so this, not a faster walker, is what made `TOY_FULL` and the TOY2
family solvable.

**Bug the small rulesets missed, again:** the first symmetric BR scored 1.3-1.9
points *above* the true L1 maximum (0.8065 vs 0.7934). **A best response above
the maximum is not a strategy.** At a canonical set whose key has a non-trivial
stabilizer (hand {red 1, blue 1, W} on a green top is fixed by swapping red and
blue), a pure choice "red 1" lifted through each history's own canonicalising
element plays red 1 at some histories of the real set and blue 1 at others,
conditioning on hidden information. Fix: **a symmetric strategy is uniform
within every action orbit of the stabilizer.** BR picks the best *orbit* and
plays it uniformly (a symmetric BR always exists against a symmetric opponent),
and CFR averages regrets and strategy mass within orbits. 284 of `TOY`'s 12,036
canonical L1 sets have such orbits. L2/L3 sets are near-singletons, so they
hid it.

**Consequences:**

- The Heuristic had to be made symmetric (ties uniform instead of lowest type
  id). That made it slightly stronger vs Random and less exploitable (5.6 / 11.8
  pts, was 7.2 / 12.5).
- **Correction (step 8.8, A3):** the plan said an RL agent is made symmetric by
  "canonical observation in, action mapped back". Not enough: with a
  non-trivial stabilizer several elements reach the canonical key and the
  action would depend on which one `canonical` returns. The output must also be
  **averaged within the stabilizer's action orbits**. `Symmetrized(agent)` does
  both and is tested. Also corrected: symmetrising by *mixing* a policy's images
  never raises exploitability (convexity); averaging behaviour probabilities set
  by set is a different policy with no such guarantee. (The module docstring of
  `uno/game/symmetry.py` still has the old claim.)
- `tabulate` refuses an agent that is not orbit-uniform. A raw asymmetric agent
  needs the full tree (`symmetric=False`).

### 5.9 Memory: keys as bytes, tree cache, cache version (steps 8.6, 8.9)

Measured first: on full `TOY` with L1+L2 keys, 1.1 GB of which **90 % was Python
key dictionaries**; the numpy passes ran at 30-60M nodes/s and were never the
bottleneck. So keys are stored encoded as bytes (1.1 GB → 179 MB), float32 was
**dropped** (would save ~30 MB), and numba was **not** adopted (step 8.7 showed
per-checkpoint costs are milliseconds). Compiled trees and CFR results are
cached in `.cache/` (git-ignored). Since step 8.9 each file carries
`CACHE_VERSION`. A stale or unstamped file is recompiled, and the version must be
bumped whenever the engine, chance model, keys, symmetry or layout change what a
compile produces.

### 5.10 Sizing by position DP (step 8.7)

**Problem:** the census visits every history, so a 50M-history candidate takes
minutes to count, and one of 500M looks the same under a cap. **Insight:** the
subtree below a history depends only on its *position* (hands, unknown pile
multiset + known tail, top card, color, player, phase, drawn card). The public
history only feeds information-set keys. So histories, terminals, depth and the
exact end distribution under history-free policies are memoised over positions.
**Gain:** `TOY_FULL` 774k histories from 49k positions (16x), and the ratio grows
with tree size, which made ~45 candidates countable in ~25 min. Also gives the
exact random-vs-random seat-0 score for free. Cannot count L1/L2 information sets
(history-dependent). Cross-checked to the history against the census.

### 5.11 One keying per tree (step 8.7)

`TOY2_FULL` keys don't all fit in memory at once. Two compiles of the same
(ruleset, symmetric, roots) walk nodes in the same order whatever the keyings, so
a keying loaded from one cached file can be attached to a tree from another
(verified array-equal on `TOY2_FULL`). For L3 on a world tree every history is
its own set, so `best_response_perfect` takes the maximal arc with no keying at
all (equal to the keyed L3 BR to 1e-9). Cost: `TOY2_FULL` lazy L2 tree 3.0 GB
(516 s compile), world L1 1.9 GB, L1 1.9 GB (356 s, ~5 GB peak). Each BR 3-4 s.

### 5.12 Unreached information sets (step 8.8, A1 / M2)

**Problem found in review:** at a responder set with zero counterfactual reach
every action ties, and the solver took the first orbit. That doesn't change the
BR's value against its opponent, but it is what the BR *plays* if another
opponent steers the game there. So the exploitability of "BR L1 to Random" as a
row agent was partly a property of an arbitrary fill-in. Random never draws with
a play in hand, so on free-play toys an exploiter reaches those sets just by
drawing.

| ruleset | L1 sets Random never reaches (seat 0 / 1) | exploitability as solved | Nash-filled | least exploitable (step 9) |
|---------|-------------------------------------------|--------------------------|-------------|----------------------------|
| `TOY` | 42 % / 47 % | 1.56 / 0.36 | 1.38 / 0.29 | 1.21 / 0.24 |
| `TOY_FULL` | 76 % / 84 % | 4.55 / 1.26 | 3.06 / 2.44 | 2.83 / 1.14 |
| `TOY2` | 52 % / 60 % | 1.68 / 2.17 | 1.41 / 1.70 | 1.37 / 1.64 |
| `TOY2_SKIP` | 49 % / 59 % | 1.34 / 1.74 | 1.05 / 1.09 | 1.01 / 0.99 |
| `TOY2_FULL` | 85 % / 91 % | 6.25 / 4.89 | 3.40 / 3.30 | 3.11 / 2.59 |

Fix: `best_response(..., completion=strategy)` fills unreached sets explicitly,
and `br_row` builds a two-seat row. Reports now show both. Nash-filled is *one*
reasonable fill-in, not the least exploitable. Step 9 solved for the least
exploitable one (CFR with the reached sets fixed): on `TOY_FULL` the Nash fill
was worse than the solver's arbitrary one at seat 1, and half or more of the
free toys' reached sets are tied as well, so even these are upper bounds. **The earlier claim "BR to Random is no longer
near-Nash on free-play toys" survives at about half the size**, and its per-seat
split was an artefact.

### 5.13 Testing strategy

- **Reference vs compiled:** slow pure-Python walkers are the trusted
  implementation; compiled/quotient versions are tested equal to them on small
  rulesets and pinned on `TOY` values.
- **Hypothesis property tests** for the spec's invariants (conservation,
  legality, determinism, immutability, termination bound, terminal correctness,
  information invariance under permuting unseen cards, perfect recall) and for
  group actions commuting with the engine.
- **Pinned known values** (`tests/known.py`): the twelve `TOY` BR values, Nash
  0.6196, `TOY2` sizes and values, the duplicate-card sizing regression.
- **Bounded examples:** suite ~3 min 45 s, 102 tests. Hypothesis examples are
  capped by census size (`census(rs, node_cap=20_000)` + `assume(complete)`),
  because an uncapped example on a 9-card deck ran 10+ minutes. Long checks go in
  experiments, not pytest.
- **Full rerun** (`bash scripts/rerun_all.sh`): 18 steps, 1 h 55 min, all exit
  0 at `5be8a79`, and every number in the results docs reproduced to the printed
  digit (except the documented duplicate-card sizing rows).
- Smaller fixes along the way: `match` used to swallow any `ValueError` as an
  "invalid seed" (now only deal errors); a test asserted `x == constant or True`;
  a hypothesis filter health check was flaky on one test (suppressed).

### 5.14 Step 9 tools: robustness checks and coupled evaluation

**Problem (step 9 review):** several readings depended on choices never varied
(the scoring of exhaustion, the Nash strategy CFR happened to reach, the fill-in of
unreached sets), the on-seeds checks could not see the deal model, and "pairing
does not help on full UNO" had only been measured on dissimilar policies.

**Techniques**, all exact passes on the existing trees unless noted:

- `CompiledTree.rescored("draw" | "stuck")`: the same tree with exhaustion scored
  as a draw, or as a loss for the player stuck on the empty pile; every solver
  runs unchanged, so a sampled method can be validated against three exact games
  for free.
- `two_sided(tree)`: minimax values when both see both hands / everything, and the
  share of worlds no play changes.
- `cfr(fixed=...)`: CFR with some sets frozen; the least exploitable completion of
  a best response's unreached sets. `best_response` also counts tied reached sets.
- `OnSeeds.z_global`: mean of on-seeds exact values over many real deals against
  the whole-game value (catches a wrong deal model, which no simulation check can).
- **Ticket draws** (`draws="tickets"`, sampled only): a player's k-th draw is the
  pile card with the lowest of fresh tickets keyed on (seed, player, k); same game,
  but paired games keep drawing the same cards after their draw counts part.
- **Shared rng streams** (`stream` on an agent): variants of one policy draw the
  same action random numbers, as two checkpoints should.
- **Speed-ups for `TOY2_FULL`**: `transfer` moves a strategy between the lazy and
  world trees by key instead of tabulating again, and `store.load_or_tabulate`
  caches the baseline agents' tabulations (a Python call per set, ~50 s per agent
  per tree on 3.75M sets). The cache is keyed on the agent's name: bump
  `CACHE_VERSION` if a baseline policy changes.

**Bug caught by its own test:** the first ticket draw could pick the setup's
known tail (cards skipped at setup, public, drawn last), 4.7σ off the exact value
on a 7-card ruleset; fixed before any number was reported. The ticket paths the
named toys never take (duplicate cards, +2 / +4 penalty draws) are checked on a
10-card ruleset in `pairing`.

**What it bought:** section 6 readings now carry their scoring and fill-in
caveats (section 7); checkpoint comparisons on full UNO turned out affordable
with pairing on a shared stream. Results: `docs/results/09-results.md`.

### 5.15 Step 9.5a tools: memory ceilings, edge attribution, node lookup

**Problem (step 9 review, section D):** three questions about the toy results
had no tooling yet: how much of BR L1's edge needs the full history (memory),
where a checkpoint loses its points against a reference strategy (edge
attribution), and a state → tree lookup that step 9.5b's AIVAT will also need.

**Techniques**, all exact passes on the compiled tree unless noted:

- **Coarse keys** (`uno.compiled.coarse`): K0 (hand, top, colour, opponent hand
  size, pile size, phase, drawn card, plus the tree level so a coarse set never
  spans two levels), K1 (+ discard multiset), K2 (+ a void hint). `coarsen`
  builds a new `Keying` over a tree's own nodes/arcs by grouping L1 sets under
  a key function. Step 9.5a combined constituent L1 action orbits by their
  transitive join; 10a.3A supersedes that construction after finding that the
  L1 coordinate frames themselves leaked discarded history (section 5.24).
  `memoryless_response` is an iterated greedy
  best response with `own` (below) fixed at the current iterate: every step is
  a real, history-free policy, so the best value seen is a valid lower bound.
- **`own`** (`best_response(..., own=...)`, spec section 21): the responder's
  own per-arc reach folded into the set-level weighting. A no-op on a
  perfect-recall keying (own-reach is a positive constant within every L1/L2/L3
  set, since every history of one shares the same public history exactly), but
  meaningful on a coarse keying, where it weights merged histories by how much
  of a fixed reference policy's reach each carries.
- **Edge attribution** (`uno.compiled.edge`): the performance-difference lemma
  gives an *exact* decomposition of the gap between two strategies against a
  fixed opponent, summed over one side's own decisions, grouped by
  `uno.eval.decisions.node_kind` / `action_class` / `situation`
  (`uno.store.load_or_label`). `br_gap` (BR L1's value minus a row's) is the
  step 9 review's D4 metric for training against a fixed opponent.
- **Node lookup** (`uno.compiled.lookup.NodeIndex`): a played state's own node
  and, for an action, its arc, via the L2 keying (one node per set on a full
  tree, up to `|G|` equal-valued ones on the quotient). Feeds a **sampled
  cross-check** of the edge decomposition (play the row agent, sum the exact
  per-node quantity along real games, compare the sampled mean to the exact
  total) and will feed AIVAT's control variate directly in step 9.5b.
- **Deal model at 100,000 seeds** (`uno.experiments.deal_model`,
  `uno.experiments.common.deal_model_check`, shared with `robustness`): 3σ
  tightens from ~0.5 points (step 9, 20,000 seeds) to ~0.24.
- **Divergence statistic** (`uno.eval.estimate.compare(..., divergence=True)`):
  per seat, the share of a pairing's deals whose two rows' own games are
  byte-identical start to finish (common random numbers), and where the rest
  first diverge. Quantifies what step 9's pairing gains were actually reading.

**Four bugs, three caught before any number was reported** (all now regression
tests, or a fix documented alongside the code): a coarse policy read off one L1
set's own orbits wasn't uniform within the *coarse* set's (joined) orbits,
reproducing the section 5.8 bug one level up; `best_response`'s per-level
backward pass silently assumes a level's sets are numbered contiguously (true
by construction for `compile_tree`'s own keyings, not for a naive key-sorted
coarse one), caught by comparing an `own`-weighted value against `own=None`'s
on a perfect-recall keying (they must be identical, and were not); `own` had
been folded into `best_response`'s `unreached` mask too, silently starving
sets `own`'s current policy doesn't visit of a real choice even when chance x
opponent alone can still reach them. The fourth was caught live by the user,
watching memory climb during `pairing`'s full-UNO section: it held all 20
`(draws, column, row)` `play_pairs` results, each with a full game trace (~1
GB for a `FULL_UNO` set of 4000 games), simultaneously before computing any
statistic -- pre-dating step 9.5a (the traces were always retained, just
unused before the divergence statistic needed them), it crashed a ~17 GB
machine on ~20 GB of simultaneous traces. Fixed by computing each comparison
against `"heuristic"` as soon as both sides are available and immediately
slimming the trace away, the same pattern `uno.experiments.transfer` already
used elsewhere.

**What it bought at the time:** `docs/results/09.5a-results.md`. Its claim that
K1 recovered 96-99.8 % of BR L1's edge is historical and superseded by 5.24:
the policy was not strictly K1-measurable, and the ratio used return relative to
zero instead of skill relative to the same-seat Random baseline. The qualitative
finding that K2 buys little over K1 survives the corrected achieved-score table.
On
`TOY`, the Heuristic's seat-1 weakness against Random is 71 % wild-colour
decisions specifically (not hold-vs-spend); on `TOY2_SKIP` the skip-timing
kind is 24-40 % of the gap at both seats, both a genuine sharpening of the
qualitative claims in section 6 below, not new claims of their own.

### 5.16 Step 9.5b: AIVAT from the exact toys to full UNO

**Problem:** full UNO's luck is in-game draws, where mirroring and pairing of
dissimilar policies remove ~10 % (section 6.6). AIVAT (section 9) needed an
exactly validated implementation, a chance model with reshuffles, and a value
function that exists where no tree does.

**Techniques** (`docs/spec.md` section 23):

- **The estimator as records.** A value model emits per decision the value at the
  state, the pre-chance value of every action of the support and the value of the
  child. Per-game sums then give any subset of corrections (chance, row, column,
  deal) by subtraction. `stream_pairs` keeps no traces (the 9.5a memory lesson).
- **Exact moments.** One backward pass carries E[F] and E[F²] of the estimator
  from each node, for any value function and any subset of corrections, so every
  sampled variance has an exact target.
- **The value function is an L2-state value**, never reading the pile order.
  Otherwise the draws are deterministic given V and only the intractable deal and
  reshuffle permutations are left as chance. On the toy that is the lazy tree's
  node value. The brainstorm and plan had "L3 critic"; corrected.
- **`draw_outcomes`**: the chance model for any ruleset, including seeded
  reshuffles and tickets. A penalty draw that crosses a reshuffle gets no model and
  its corrections are skipped (0.004-0.009 % of full-UNO decisions).
- **A colour-symmetric logistic critic** (numpy IRLS), one per profile, fitted on
  the new `FIT_SEEDS`. Its played path is tested equal to the tree path, record by
  record.
- **The deal term as a deal-model check.** `nu(first state) - E_model nu` needs
  only a lookup per seed: 5,000,000 seeds in seconds, 3σ ≈ 0.04 points on `TOY`
  (was 0.24 in step 9.5a).
- **Intermediate rulesets** `MID_12` (26 cards, 13.7 actions) and `MID_25` (30
  cards, 21.3), picked by a pre-declared rule on length and decision shares; a
  second candidate round was needed.

**What went wrong or nearly did:**

- A first 200,000-seed deal-term range gave 12 positive z (up to +2.81); five
  disjoint ranges of 1,000,000 showed it was that range's deal luck.
- One toy variance miss (TOY2_SKIP, tickets) was rechecked on ten fresh ranges:
  chance.
- A FULL_UNO seat-0 AIVAT − raw miss (z −3.08) was chance. A fresh-seed run gave
  z +1.02, and a drawn-card uniformity test over ~1.7M draws showed max |z| 2.21 of 44.
  That test's first version counted both seatings of a seed as independent draws,
  although they share the pile order, and it flagged the exact TOY2 model at
  z −3.67.

**What it bought:**

- On the toys every identity holds to 5e-16, and 1 of 1,115 sampled checks
  missed.
- On the toys the luck is mostly the deal: 34-95 % of the variance is the deal's
  value; in-game draws carry 5-33 %.
- On full UNO a weak critic (R² 0.11) still removes 24-29 %. The ratio stays flat
  (0.72-0.76) from 14 to 45 decisions while R² falls four-fold: the number of
  corrected events carries the gain, so the toy R² → ratio curve does not
  transfer by R².
- Exact values without the deal correction reach 0.39-0.49 on the toys, so a
  better full-UNO V has up to twice the gain to collect. (Step 9.5c measured the
  ceiling: the full-UNO floor is 0.086-0.090, so up to 11x, of which a fitted network
  takes 2.2x.)

### 5.17 Step 9.5c: PIMC values as a teacher, a network as AIVAT's V

**Problem:** 9.5b's full-UNO V was a logistic critic (outcome R² 0.11) removing 24-29 % of the
variance, and the step closed on "a better V has up to twice the gain to collect", untested.

**Two measurements settled the design before anything was built:**

- **The deal floor.** With exact L2 values and corrections at chance and both players, AIVAT's estimate
  *is* the value of the dealt hands, so its residual is the deal's variance. Replaying dealt hands under
  fresh pile orders measures it without a tree: **0.086 / 0.090 on full UNO** (0.42-0.49 on the toys,
  0.25 at 14 decisions, 0.17 at 22). The ceiling was 8x lower than the toys suggested, and the deal
  matters less the longer the game.
- **Rollouts cannot be the V.** With V a rollout mean of the played profile the corrections telescope
  towards "n more games from the same hands", at the cost of those games, and the independent noise of
  ~90 values per full-UNO game swamps the gain: measured ratio 3.99 at n = 16, against 1.0 for no
  corrections at all.

**So PIMC is the teacher, not the V** (`uno/eval/pimc.py`, `uno/eval/value_net.py`, spec section 24):

- **Determinization at L2**: the unknown pile in a uniform random order, a fresh seed, pile draws. It is
  `draw_outcomes`' chance model, and it covers what that model skips (penalty draws across a reshuffle).
  Rollout values are then unbiased estimates of the profile's own L2 values, checked against the exact
  tree on three toys with pile and ticket draws (48 checks, 0 failures).
- **Teacher targets**: every decision state plus two siblings AIVAT also values, each with n rollouts.
- **A symmetric value network**: one block per colour plus a colour-free global block, a shared MLP over
  the colour blocks pooled by sum and max (DeepSets), so renaming colours or ranks cannot change the
  value; float64 CPU inference, so V is a deterministic function of the state whatever else is batched.
- **A Monte Carlo deal mean** over a fresh seed range, which makes the deal correction available where
  no tree exists. It carries a fixed error (~0.04 score points at 250,000 deals) that more evaluated
  games do not shrink.
- The evaluation loop is now parallel over seed chunks (it was the single-core part of the run).

**What it bought:** on full UNO the variance ratio goes **0.71 / 0.76 → 0.46 / 0.47**, i.e. 1,776 deals
instead of 3,860 raw or 2,757 with the critic, for a 3-point difference at 3σ with 80 % power; with the
deal correction 0.38 / 0.39 (1,483 deals). On the toys the network is nearly the exact value function
(0.02-0.06 with every correction). The pre-declared criterion passed at both seats. 7 failed checks of
648, all of them the critic's and both 9.5b's known events reproduced to the digit.

**What it taught:**

- **Outcome R² is a poor ordering, again.** 0.110 → 0.148 on full UNO for a ratio of 0.71 → 0.46. The
  network's R² *against the true values* is 0.83-0.84 on every ruleset, yet the ratio is flat at ~0.5
  from 6 to 45 decisions: AIVAT only sees the *change* of V across a chance event, so a smooth error
  cancels and only the local error survives.
- **The flat ratio is the 9.5b phenomenon one level up.** The critic sits at ~0.75 and the network at
  ~0.5 whatever the game length, while the floor falls with length, so the share of the *reducible*
  variance captured drops from 96 % (`TOY`) to 59 % (`FULL_UNO`).
- **AIVAT costs more CPU than it saves when the agents are cheap**: a rulebot pair is 2.6 ms, the
  corrections 30-90 ms, so 4,000 extra deals beat halving the variance of 4,000. Its gain is in *deals*,
  so it pays when a deal is expensive or scarce, which is 9.5d's search row.
- At a fixed rollout budget, **many noisy targets beat few clean ones** (n = 2 on 20,000 games over
  n = 32 on 1,250), and network width barely matters: the limit is the features.

Results: `docs/results/09.5c-results.md`.

### 5.18 Step 9.5d: PIMC as an agent, exact on the toys and sampled on full UNO

**Problem:** 9.5c used PIMC rollouts as a value function's *teacher*. Whether PIMC is a good
*player* -- and what its known biases (strategy fusion, non-locality) actually cost -- was untested.
The toys can answer it exactly, which is what they are for.

**Techniques** (`docs/spec.md` section 25):

- **Exact PIMC-infinity** (`uno/compiled/pimc.py`) on the toy **world trees**. Every non-root arc of
  a world tree has one edge of probability 1 and `reach(...)[:, 2]` is the world's prior, so the
  perfect-information value of an action is a *gather*, and neither it nor the belief depends on
  PIMC's own policy: the whole policy is **one global orbit-aware argmax**, with none of
  `best_response`'s per-level backward loop. Two axes: belief weighted by the opponent's reach
  (`posterior`) or the prior alone (`prior`, i.e. no inference), and each world solved two-sided
  (`minimax`, classic PIMC) or as a perfect-information best response to the fixed opponent
  (`response`).
- **PIMC-n** as the behavioural policy `pi_n(a | I) = P(argmax over n sampled worlds)`, estimated per
  set and then scored by one exact `evaluate`, so the numbers carry no game-playing noise.
- **The sampled agent** (`uno/eval/search.py`): `determinize_l1` samples the opponent's hand *and*
  the pile from an observation (a PIMC player is an L1 player); alpha-beta with 9.5c's network at the
  leaves; `PimcSearch` (fusion) and `PimcRollout` (no fusion, a one-step policy improvement). The
  world sampling is keyed on the information state, which makes the policy genuinely deterministic
  -- required, or AIVAT's row correction stops being zero-mean.
- **D2 with a search Q** where no tree exists, with selection and valuation on disjoint halves of the
  worlds and a control row that must score ~0.

**What went wrong: three bugs, each caught by a check written before the number was believed.**
Assuming arc `j` of a world is set-action `j` cost 21 score points on `TOY` at seat 1 and got *worse*
with more sampled worlds (the `uno.compiled.lookup` trap one level over: a set's members are
different worlds, each with its own canonicalising element). Selecting and valuing a reference on the
same worlds -- the winner's curse -- inflated a D2 total to 34-40 points per game; fixing that by
splitting the worlds then drove the *control* to -32, because a control row that is itself the search
must also draw its worlds from a different seed than the estimator.

**What it bought:** with the `response` continuation PIMC-infinity is within **0.04-0.70 points of
BR L1** on every toy, seat and opponent, so PIMC's structural bias is small there; classic `minimax`
PIMC loses up to **8.8 points** on free play, where assuming a clairvoyant opponent is expensive. On
full UNO at 32 worlds the search row beats the Heuristic at seat 0 (0.703 against 0.577) and loses at
seat 1 (0.563 against 0.643), and the world ladder had **not** plateaued. AIVAT works on it
(variance ratio 0.54-0.62) with the row correction identically zero, which is what a deterministic
agent implies. D2 puts ~97 % of the Heuristic's disagreement with the search in **action-card timing
and wild handling**, the same two kinds the exact toys flagged in step 9.5a. Extra search *depth*
does **not** help: paired at w = 32 on the same deals, depth 4 is 7.5 / 6.0 points below depth 2 for
8.3x the cost, and the scores fall monotonically with depth, which disfavours an odd-even leaf
artefact. The harm itself is not proven (z -1.7 / -1.3), so the conclusion is that depth is dominated
on cost-effectiveness, not that it is harmful. **Worlds stop helping too**: the ladder is flat past
32 (w = 128 is +3.0 / +6.0 over w = 32 for four times the cost, neither significant), so the whole
gain lives in the first few doublings, and at equal compute worlds still beat depth (w = 128 d = 2
scores 0.725 / 0.620 against w = 32 d = 3's 0.660 / 0.530).

**What is left on the table** (none of it tested): the full-UNO search solves each world by
*minimax*, i.e. assuming the opponent is clairvoyant too -- the continuation the toys measured as the
**worse** one by up to 8.8 points, and worst on free play, which full UNO is; in every ladder matchup
the opponent's policy is known, so best-responding to it instead needs no training. The leaf value
function is 9.5c's network, off-profile for a search row at outcome R^2 ~0.15, which is the likeliest
reason more worlds stopped helping: extra samples of a position that is evaluated badly buy nothing.
And the agent deals unseen cards uniformly, doing no inference, worth up to 2.6 points on the toys.
The 11-point seat-1 deficit against the Heuristic (0.560 against 0.670) in a game with no measurable
seat effect is partly a known wiring defect: the sweep and the AIVAT row hand the agent the *seat-0*
value network at both seats (`d2` uses the right one), so at seat 1 it searches with a value function
fitted on a different state distribution. One line to fix, and every seat-1 number in step 9.5d is
mildly pessimistic until it is.

Results: `docs/results/09.5d-results.md`.

### 5.19 Step 9.5e: the continuation rule, the largest full-UNO lever so far

**Problem:** 9.5d closed with three untested approximations in the full-UNO PIMC agent and a known
wiring defect. The toys had already ranked them, so this step spent the budget in that order.

**Techniques** (`docs/spec.md` section 25):

- **The `response` continuation** (`uno.eval.search.response_value`): each determinized world is
  solved as a perfect-information best response to the column's *real* policy -- max at the agent's
  nodes, the expectation under the opponent's policy at the opponent's -- instead of two-sided
  minimax, which assumes the opponent is clairvoyant too. It is the sampled analogue of
  `best_response_perfect`, i.e. exactly the axis `uno.compiled.pimc` measures on the toys.
- **Per-seat leaves** (`leaf_for`): 9.5d gave the agent the seat-0 value network at *both* seats.
  Not the one-line `nets[seat]` that step assumed -- one agent object plays both seatings of a deal
  (`evaluate_matchup` builds the pair once, `sweep` also plays it reversed), so the leaf must be
  resolved from `obs.seat` inside the agent.

**What it bought:** **+9.73 +- 1.04 points at seat 0 (z +9.3) and +4.57 +- 1.06 at seat 1
(z +4.3)** against Random at w = 32, d = 2 over 3,000 EVAL deals, so the lever is established at
**both** seats (a first 1,000-deal run gave +10.20 / +5.20, seat 1 only indicated at z +2.9). The row goes from losing to the Heuristic at seat
1 in 9.5d (0.5633 against 0.6433) to **beating it at both seats**, both against Random
(0.7660 / 0.7180 against 0.6190 / 0.6080) and head to head (0.6180 / 0.6010). The seat asymmetry
9.5d flagged is gone (two-sided now 0.6640 / 0.6660 where that step had 0.6950 / 0.5600), consistent
with the per-seat network fix, though that is a comparison across runs rather than a paired test.
AIVAT still works on the row (ratio 0.60-0.68, row correction identically zero), and 0 failed checks
across both runs.

**What it taught:**

- **A better continuation rule is *free*, not expensive.** It was scoped as 2-4x slower because
  expectimax cannot use alpha-beta; measured, it is *cheaper* (128.7 against 188.9 wall-clock
  ms/game), because expectimax expands only the actions in the opponent's support and both rulebots
  are far from uniform, so the collapsed branching at opponent nodes more than repays the lost
  pruning.
- **A toy check can be vacuous on the very axis it is meant to test.** The first closure test used
  `TOY`, where the two continuations sit 0.22 / 0.50 score points apart against ~3.2 points of noise
  at 200 games: it passed, and would have passed with the wrong continuation wired in. `TOY_FULL`
  separates them by 7-9 points, so the test moved there, and a noise-free test (the two rules, given
  the same worlds, must choose differently) now pins the axis directly.
- **The row is opponent-specific**: it needs the column's exact policy, so it is a ladder instrument
  rather than a general player, and on the toys fitting the belief to one opponent costs
  exploitability. A general full-UNO PIMC still pays the ~10 points measured here.
- **The budget is not the constraint; the leaf value function is.** A 7 h 39 m grid at 3,000 deals
  re-measured both ladders under the new rule. Worlds: the climb to 32 is large (+5.6 / +3.3 over
  w = 8) and everything past it is flat -- w = 128 is ~+1.1 points over w = 32 at z ~ 1.0, and
  w = 256 scores *below* w = 128 at both seats. Depth: nothing, at 3.7x (d = 3) and 13.5x (d = 4)
  the cost, all differences within 1.5 sigma of zero, which also disposes of 9.5d's odd-even
  artefact worry. At equal compute worlds beat depth (w = 128 d = 2 beats w = 32 d = 4 at both seats
  for a third of the price), so the setting is 64-128 worlds at depth 2. **This refuted the
  prediction made when the grid was commissioned**, that 9.5d's flattening was an artefact of its
  seat-0 leaf and two-sided continuation: both were fixed and it still flattens, which promotes the
  off-profile leaf network (R^2 0.07-0.25 on these games) from "likeliest reason" to the residual
  explanation and the top of the remaining list.

**Deferred: inference.** `determinize_l1` still splits the unseen cards uniformly (the `prior`
belief). The toys put `posterior` over `prior` at 0.0-0.6 points against Random -- the headline
column -- reaching 2.6 only against the Heuristic, and the belief is not cheaply available: a
world's likelihood needs the opponent's hand at each past decision, which cannot be reconstructed
across its draws, so doing it properly means a particle filter over the deal. That is **the
"particle filter + expectimax" of `plan.md`'s original design for sampled L1/L2, and the only part
of it still unbuilt**: PIMC was always meant to come with a belief, and 9.5d / 9.5e built it with a
uniform determinization instead. Scheduled as build-order row **9.5f**, with its exact target
already on the toys (`uno.compiled.pimc`'s `"posterior"` weighting).

**Superseded by step 9.5f (section 5.20).** Inference was built, but the particle filter named
above is the wrong instrument and does not work: weighting deal-time samples by the opponent's
actions collapses geometrically, and the version built to that description scored *worse than the
uniform prior it replaced*. What works conditions instead of weighting. The toy figure quoted
above as the reason to defer (0.0-0.6 points against Random) also did not transfer: full UNO gives
+2.83 / +4.40 at both seats.

Results: `docs/results/09.5e-results.md`.

### 5.20 Step 9.5f: the opponent-hand belief, and a design in the plan that does not work

**Problem:** `determinize_l1` splits the unseen cards uniformly, which is `uno.compiled.pimc`'s
`"prior"` weighting -- PIMC with no inference. Inference was the last unbuilt piece of `plan.md`'s
sampled-oracle design.

**The design in the plan was refuted first.** A particle filter over the deal (sample the
opponent's initial hand, replay the history, weight by their action probabilities) **collapses**:
a particle survives an observed play only if it holds the played card, `hand_size / unseen` ~ 7/98
on full UNO, so 64 particles decay 0.156 -> 0.018 -> 0.002 -> 0.0004 -> 0 over the opponent's
first five plays. No particle count fixes a geometric decay. The built version, resetting on
collapse at 62 % of decisions, scored **worse than the prior it replaced** (Brier 0.167 against
0.085). The lesson generalises: **condition, never weight** -- build a world consistent with what
was observed rather than sampling one and testing it afterwards.

**Techniques** (`docs/spec.md` section 26), both sampling the hand at the *current* split, where
consistency is automatic, and using the history only to constrain or score it:

- **`voids`**: a draw proves the drawer held nothing playable against the board then in force; if
  they have drawn `k` cards since, at most `k` of their cards now may be playable against it. It
  reads no policy *probabilities*, which is what separates it from `window` and keeps it cheap and
  column-portable. It is **not** policy-free, though: the count excludes the card drawn *at* the
  void, so it assumes a playable drawn card is played at once (every rulebot does; `Pass` is legal
  and writes no public event). Corrected after the fact -- see the follow-ups below.
- **`window`**: plus the opponent's real action probabilities over their trailing run of plays
  with no draw in between, where their hand is exactly recoverable as "the hand now plus the cards
  played since". Sampled by importance-resampling, never by argmax.

**What it bought:** against Random at w = 32, d = 2 over 3,000 EVAL deals, `voids` is
**+2.83 / +4.40** points (z +3.0 / +4.4) and `window` **+3.77 / +3.53** (z +4.0 / +3.5) -- the
second-largest full-UNO lever after 9.5e's continuation rule. `voids` costs +10 % and `window`
+29 %, and `window - voids` is +0.93 / -0.87, **within noise at both seats**: the hard constraints
carry the whole measured gain, so the cheap rung -- the one needing no policy *probabilities* -- is
the one to keep.

**What it taught:**

- **The toy figure did not transfer, in the direction that favoured building it.** The toys put
  `posterior - prior` at 0.0-0.6 points against Random, which is why 9.5e deferred the step; full
  UNO gives five to ten times that. A 41-decision game with `Draw` legal at 60 % of decisions
  accumulates draw evidence a 6-decision toy cannot. 9.5d's "toy belief sizes do not transfer",
  pointing the other way.
- **Calibration, not sharpness, is the correctness test for a belief.** Over-pruning looks like
  confidence. The refutation above came from a calibration harness written before any score was
  measured; the score alone would have been read as progress.
- **Brier on marginals is the wrong *value* test.** The constraint excludes 79.9 % of worlds while
  moving per-card marginals by 2 %, because it rules out *joint* combinations. Correctness and
  value needed different instruments.
- **Most of PIMC's worlds were impossible** (median surviving share 15.6 %: w = 64 was effectively
  w = 10). That is a second explanation for 9.5e's world ladder staying flat past 32, competing
  with the off-profile leaf network that step promoted to "the residual explanation". **Tested
  since** (below) -- and the answer is neither of them, quite.

**Follow-ups (results section 7).** Two of the step's open items needed the harness
parameterising first: `sweep` could not climb a ladder *under* a belief, and the belief ladder
hardcoded Random as its column. Both are now arguments.

- **The ladder re-run with `voids`** (3,000 EVAL deals): the belief is a **level shift of ~+3.5
  points at every world count, not a change of slope**. Both ladders climb hard to w = 32 and
  flatten after (`voids` 32 -> 64 is +1.20 / +0.77, z ~ 1). So flatness never could have separated
  the two explanations -- it is what *both* predict, which made the test a non-test as scoped. What
  does separate them was not asked for: **`voids` at w = 32 (0.7813 / 0.7603) outscores `prior` at
  w = 128 (0.7677 / 0.7403) and at w = 256**, for a quarter and an eighth of the cost. 9.5e's
  plateau was therefore not a ceiling imposed by the leaf network -- better worlds went straight
  through it. **The working set is now w = 32 with `voids`**, replacing 9.5e's "64-128 worlds".
- **The belief against the Heuristic column** (4,000 EVAL deals, paired): `voids - prior` is
  +2.27 / +3.70, so the rung carries to a second column at both seats, as one reading no policy
  *probabilities* should -- both columns satisfy the behavioural assumption it does make.
  `window - voids` is **+2.60 (z +3.1) at seat 0 and +0.65
  (z +0.8) at seat 1**: the column *was* why `window` looked worthless against Random, and the
  toys' "up to 2.6 against the Heuristic" is matched at seat 0. But one seat is not this project's
  standard; `EVAL_SEEDS` is spent, so seat 1 cannot be settled on held-out data; and 5.21's
  speed-up repriced the rung from 1.29x `voids` to **2.4x**, its likelihood being Python-side and
  batching with nothing. `voids` stays the default, `window` a ladder instrument.
- **A method note worth reusing.** The `prior` rows kept 9.5e's agent names, hence its
  world-sampling key, and reproduced that step **to the digit including the standard errors** -- a
  free regression test that the rewiring left the baseline alone. The same device verified 5.21:
  the batched agent *replays* the unbatched one's games rather than merely scoring like them, which
  matters because a `voids` world costs a **variable** number of rng draws (rejection sampling), so
  any reordering of the loop changes which worlds are drawn without changing how many -- a drift
  that would have moved scores by a point or two with nothing visibly broken.

**Follow-ups: what `soft` is worth, measured rather than argued.** Section 8.8 of the results left
the rung resting on an argument (it never excludes the truth) and a null result (no score
difference a 2,400-deal DEV sweep could resolve, which bounds it at about a point, not at zero).
Settling that on score needs ~39,000 deals. Two instruments settled it instead, neither spending
an eval seed:

- **`pimc divergence`** asks how often the rungs choose differently, against a control that
  re-seeds the world sampler *without* changing the belief (a second `voids` under a different
  name; `PimcSearch._rng` keys on it). `soft(0.1)` differs from `voids` at 19.5 % of decisions
  against Random and 21.5 % against the Heuristic, where the control alone gives **18.4 % /
  20.6 %**. **The belief moves about one decision in a hundred**, so the sweep was never
  under-powered -- there was nearly nothing there to separate, and the score question against
  honest columns is closed by arithmetic rather than deferred to a bigger run.
- **`pimc closure` now takes the opponent whose exact posterior is the target**, so the rung can
  be scored against the case its assumption is false. Against `KeepsDrawn` (`PlayFirstRandom`
  that keeps a playable drawn card half the time) the two rungs **swap places**: `voids` closes
  88-91 % of the distance to the posterior against an honest column but only 16-19 % against that
  one, where `soft` closes 58-66 % and 52-67 %. So 8.6's recommendation holds with numbers behind
  it -- `voids` for the ladder, whose columns are honest, `soft` for a column that may not be --
  and it is a **league** design rather than a caveat: the rung is chosen per opponent.
- **`pimc decay`** sweeps the decay against *both* columns' exact posteriors and finds a genuine
  balance point: **0.1 is where the two cross** (61.2 % of the distance closed against the honest
  column, 59.2 % against the passer), so the default 8.8 picked as a robustness tie-break among
  score-indistinguishable settings is also the maximin rung over the pair. The passer's own
  optimum, 0.2, buys 0.8 points of accuracy there for 16 against the honest column, so a row that
  knows its column should switch rungs rather than compromise. The sweep reproduces its own
  endpoints (`soft` 0 = `voids`, `soft` 1 = the prior) as a check, which pins the family end to
  end on the sampled rung and not only on `_void_weight`.

Two method notes. Over-pruned sets now stay in the closure average (skipping them scored `voids`
only on the sets where its assumption holds, the flattering half), and the `rulebot` column
reproducing 8.1 **to the digit** is what licenses the change. And `soft` never short-circuits --
it weighs all `tries` candidates where `voids` returns on the first survivor -- so it costs
several times `voids` per call, with `tries` as a quality/cost knob that interacts with the decay;
that one is a scratch benchmark so far, not yet a repo section.

Results: `docs/results/09.5f-results.md`.

### 5.21 After 9.5f: a 2.6-4.5x speed-up that changes no number

**Problem:** a full-UNO PIMC row costs 125-165 ms/game and the belief ladder at 4,000 deals runs
about an hour, which is the tax on every question asked of the agent. Profiling put **69 % of the
time in the leaf value network** and most of that in torch's per-call dispatch, not its arithmetic:
the search called the leaf once per leaf state, so every forward was a batch of one. Measured on
`FULL_UNO`, 78.5 us a row at batch 1 against 7.0 us at batch 32-256.

**Technique** (`docs/spec.md` section 25): a decision searches a whole *sample* of determinized
worlds, all independent, so `uno.eval.search.batched_search_q` plans every one of their
depth-limited trees before consulting the leaf (`_plan` defers each leaf state into one list and
leaves its index in the plan), evaluates them all in one call, and backs the values up (`_fold`).
`uno.eval.value_net.NetLeaf` is the batched leaf; a plain callable still works, one call per state.

**What it bought:** **2.8-3.3x** end to end on the response rows, 1.8-2.1x on `window`, 3.3x on the
two-sided `minimax` rows -- at **identical per-game scores** on the real network, across both
columns and all three belief rungs. Confirmed at full scale: re-running the 4,000-deal ladder
against the Heuristic reproduced the killed pre-optimisation run's `prior` rung to the digit
(0.6175 / 0.5897) at 24.5 ms/game against 76.2.

**What it taught:**

- **A speed-up that changes a number is not free, so this one was built not to.** The searched value
  is equal *bit for bit*, not to a tolerance: `_plan` visits the nodes `response_value` visits in
  its order and `_fold` combines them with the same operations, and the worlds are drawn before the
  search rather than between searches, so the world rng is consumed in the same order. The row
  therefore **replays** 9.5e's and 9.5f's games instead of resampling them, and the ladder stays
  comparable across the change. Both properties are pinned by tests, on all four rulesets and both
  continuations.
- **Pruning was worth less than batching at every depth measured.** Batching drops alpha-beta.
  Under `response` that is free (expectation nodes never pruned). Under `minimax` it was measured
  rather than assumed: 3.75x at depth 2, 1.91x at depth 4, a tie at depth 3.
- **A docstring claim that is one ulp too strong is still worth fixing when something starts
  resting on it.** `value_net.py` said float64 made V independent of "whatever else is in the
  batch"; batch 1 and batch 256 in fact differ by ~1.7e-16, because BLAS picks its kernel by the
  row count. The invariant that *does* hold exactly, and the one the batching needs, is that at a
  **fixed** row count a row's value depends on that row alone -- so the agent stays the
  deterministic function of the information state AIVAT's row correction assumes. That is now the
  documented claim and a test.

**Then the second pass, on what batching exposed.** With the leaf no longer dominant the profile
was flat and interpreter-bound -- `position_features` 35 %, world construction 29 %, `step` /
`freeze` 20 %, the network itself 6 % -- so each candidate was measured before being built, and
three of five were dropped on the measurement:

- **built:** `batch_position_features`, the batch's features as array products against per-ruleset
  tables (3.2x on the features, **1.3-1.6x end to end**, bit-identical); the per-decision derivations
  (`unseen_cards`, `voids`, `trailing_window`) computed once instead of once per world; and
  `playable_table`, `is_playable` tabulated (138 ns a card as a call against 24 ns as a frozenset
  lookup), worth **1.9x on the `voids` belief**, whose `satisfies` is nothing but that test.
- **dropped:** copy-on-write in `step` (1.13x on `step`, ~2 % overall -- the cost is not the
  copying); caching the board and discard features across a decision's worlds (the discard loop is
  6 % of the features, and the board is shared in only 11 of 26 groups); and a partial shuffle of
  the ~90-card pile to draw at most a few (7 %, but no early exit preserves `random.shuffle`'s
  stream, so it would resample the ladder rather than replay it).

Together the two passes, measured through the parallel CLI on the same 400 EVAL deals against the
Heuristic (ms/game, identical scores at every step):

| rung | before | batched leaves | + this pass | total |
|------|-------:|---------------:|------------:|------:|
| `prior` | 79.5 | 25.7 | **17.7** | 4.5x |
| `voids` | 91.6 | 32.8 | **21.0** | 4.4x |
| `window` | 132.7 | 68.3 | **50.3** | 2.6x |

Beyond that the remaining ~90 % is Python, which is where a Rust engine would start paying -- but
the binding constraint on what the agent is *worth* is the leaf network (R^2 0.07-0.25) and the
share of impossible worlds, neither of which more speed answers.

### 5.22 Step 10a: the contract, and two items the measurement changed

**Problem:** step 11 will produce checkpoints, and nothing decided in advance what gets
measured about one, on which seeds, or what counts as better. Three items
(`docs/results/10a-results.md`); the other three of the brainstorm's six, plus LBR,
follow in 10a.2.

- **The seed registry** (`uno.eval.seeds`, spec section 27.1). `EVAL_POOL` is partitioned
  into disjoint blocks claimed per step, checked for overlap at import. The load-bearing
  entry is the admission: **`EVAL_SEEDS` is spent.** 9.5d-9.5f chose the full-UNO working
  set (`response`, w = 32, `voids`) on 3,000-4,000-deal runs over those 4,000 seeds, so by
  this project's own rule they had been a dev range for three sub-steps and nothing
  noticed. 9.5f said it for `window` at seat 1 and stopped there. The range is kept
  unchanged (every number of steps 7-9.5c is on it) as a block marked spent, which raises
  if asked for.
- **The batched tabulation** (spec section 16). Scoped as "batch the policy call";
  profiling said that is an eighth of the cost. `observation_from_key` is **57-62 %** of
  `tabulate` (of which `_hand`'s five `Counter` objects per set are half), the policy call
  12 % with a rulebot. So both halves were built: `batch_l1_observations` (2.6-3.0x,
  array-equal field by field) and `tabulate_batched` (array-equal to `tabulate`). A
  network at batch 1 costs 69-92 us a row against 2.4-3.3 at batch 256, so for a
  *network* the two together are **~10x** on `TOY`, and `TOY2_FULL`'s 3.75M sets project
  at **~200 s -> ~21 s**. That repairs 6.9's "milliseconds to seconds", which was a claim
  about tabular rows and false for a network.
- **The report card** (`uv run uno report`, spec section 27.2), which reproduces the
  published values to the digit and cross-checks three numbers it was not pinned against
  (`br_gap` 5.18 / 13.66 against 6.2's independently computed 5.2 / 13.7; 14.41 / 20.06
  against +14.4 / +20.1; Random normalising to exactly 0.00 / 0.00).

**What it taught:**

- **A refusal is a measurement.** The first network stub was a plain MLP over raw
  features and `tabulate` *rejected* it as not orbit-uniform. So option B of the encoding
  choice (plain encoder + `Symmetrized`) does not merely pay orbit averaging per forward
  pass -- without `Symmetrized` in the path it cannot produce an exact cell on the
  quotient at all. Decision recorded: **equivariant by construction**, with `Symmetrized`
  as its test oracle (a correct equivariant net is a fixed point of it). The network is
  step 11's object; 10a builds only the seam.
- **Symmetrising a tabulation is nearly free, and that is a property of the quotient.**
  Every set's key is already canonical, so there is no `canonical` call to make, only one
  `bincount`. The brainstorm's "A2 is written against whichever encoding it is" was
  therefore too strong: the pipeline is the same shape either way.
- Batched float64 agrees to **one ulp, not bit for bit** (BLAS picks its kernel by row
  count) -- 5.21's lesson, reproduced exactly where that section said it would be.

### 5.23 Step 10a.2: the training target, and a cost that refuted the plan for it

**Problem:** 10a fixed what gets measured about a checkpoint; this fixes what a checkpoint is
*trained on*, plus the two ladder rows and the shaping artifacts that were designs rather than
files (`docs/results/10a2-results.md`).

- **The training target** (`uno.eval.target.TRAINING`, spec section 28): chance + opponent +
  deal, **never the agent's own decisions**. The argument was in the brainstorm; what is new
  is that it is a *measurement*. On a compiled tree the exact policy gradient of a tabulated
  policy is available, and so is the **exact mean of the sampled estimator under any subset of
  corrections**, so the bias is a number: 1e-17 for every subset sparing the row, and with an
  exact V the row correction drives the gradient to **identically zero**. Sampled over 2,000
  games the kept subset is unbiased (z +0.56) and removes **63 %** of the gradient variance.
- **The parameterisation is the whole argument.** The score-function estimator estimates the
  *logit* gradient, and a corrected return puts everything the corrections collected *before*
  the decision into a per-set constant. Those pre-decision terms are the same size as the
  gradient itself (4.5e-2 against a norm of 4.7e-2) and cancel only in the centred form. An
  unbiasedness claim about "the corrected return" without saying which parameterisation is a
  claim about nothing.
- **With an approximate V the row correction is not safe either, and R² does not say how
  unsafe.** A coarser V (R² 0.55) leaves 86 % of the gradient; a better one (R² 0.74) leaves
  8.5 %. This is 5.16's "R² does not carry" one level up.
- **The shaping potentials** (spec section 29) are emitted per level, and the identity they
  exist for holds exactly: read off played games, the **L2 deal potential's term is AIVAT's
  deal correction array-equal**. The deal floor by level is 0.28/0.18 (L1), 0.42/0.49 (L2),
  0.47/0.53 (L3) on `TOY`, and **0.86-0.96 at L3 under Nash self-play** -- between equals
  almost the whole outcome is the deal.
- **The belief rungs' exploitability** (spec section 30): the rungs that ship cost almost
  nothing (one seat 1.35 -> 1.50), while full opponent-fitting costs **8.2 points of NashConv**
  against the Heuristic column. They change the policy at thousands of sets and the score by
  at most 0.05 points, which is 5.20's "toy games are too short" measured rather than inferred.
- **A column-free full-UNO row exists and is strong** (spec section 30, EVAL block 01): the
  continuation best-responds to a *fixed surrogate* (the Heuristic) and the belief is `voids`,
  so nothing it does depends on who it faces. It beats the rulebot by **14.8 / 16.7** points
  against Random and **22.7 / 23.4** against the Heuristic, at both seats, z > 15 -- the first
  full-UNO rung that could legitimately be *trained against*. Handed its own column as the
  surrogate it reproduces the `response` row's games array-equal, which is what makes it a
  measurement of that row rather than a new agent.

**What it taught:**

- **"It batches" is not a cost estimate.** The brainstorm's reason to expect AIVAT to be
  affordable in a training loop was that the siblings batch. Their *values* do; the sibling
  **states** have to be built by the engine, and that is 73 % of the floor. The corrections
  cost **55x** the step's own `V(s)` calls with the features vectorised. The actionable half:
  the deal correction is **one row a game**, and it is the one that removes the deal floor.
- **A z needs a central limit, and a sparse projection has none.** A per-orbit projection
  visited **once in 200 games** reported a confident z of -5.86 for a provably unbiased
  subset (23 visits: -1.18; 232: -1.13). Fixed by choosing the projection among reached sets
  and refusing to check a z below 30 contributing games.
- **A check that passes on identically zero quantities is not a check.** The first
  finite-difference test of the exact gradient compared twelve orbits whose gradient is zero
  by construction -- single-orbit sets, which carry most of the reach. 5.19's vacuous-test
  lesson, met again while building the instrument meant to prevent it.
- **Reading the opponent was worth much less than the ladder implied.** The instrument's
  advantage over the column-free row is **+1.70 / +1.75 (z +2.1)** against Random -- indicated,
  not established -- and **+0.27 / +0.85** against the Heuristic, i.e. nothing. 5.19 read the
  `response` continuation's +9.73 / +4.57 as the value of best-responding *to the column*; it
  is mostly the value of best-responding to **something competent**, and assuming a clairvoyant
  opponent instead (`minimax`) is what costs the points (3.6-6.7, z 3.9-7.3, both seats).
- **The first number ever scored on unspent full-UNO seeds agreed with the spent ones.** The
  working-set row scored 0.7813 / 0.7603 on block 00, where it was *selected*; on the fresh
  block 01 it gives 0.7860 / 0.7685. The selection did not materially inflate it -- which is
  reassuring rather than guaranteed, and is why 5.22 kept the blocks apart in the first place.

### 5.24 Step 10a.3A: the coarse-memory coordinate leak

**Problem:** step 9.5a grouped quotient L1 sets after their full histories had
already selected canonical color/rank coordinates. Equal raw K1 inputs could
therefore land in different groups or attach the same coarse action probabilities
to different real actions. The policy was valid at L1, but not measurable from K1.
The review reproduced this at a positive-reach `TOY` set: two equal K1 inputs
returned `{1: 1}` and `{3: 1}`.

**Repair:** `coarsen` now canonicalizes K0 / K1 / K2 from their own fields and
maps every legal action and arc through that same group element. Action orbits
come from the coarse key's own stabilizer. `CoarseAgent` is the direct deployed
path; `lift_to_l1` maps the same strategy into the ordinary compiled L1 layout.
On the reproduced collision both paths return `{1: 0.5, 3: 0.5}` at the same
on-policy reach (`0.0017460317460317458`). On `SMALL`, quotient coarse groups are
exactly the symmetry orbits of the full-tree groups, and the same lifted K0 / K1 /
K2 policy has the same exact value on both representations at both seats.

**What changed:** the corrected quotient has fewer coarse sets because symmetric
copies no longer remain separated by discarded history. The effect is largest at
K0 (up to about 5.3 score points in the refreshed rows), while K1 moves by at most
about 0.6 points. Those are intended removals of undeclared information; the
Heuristic and BR L1 controls are unchanged. The refreshed K1 gaps to BR L1 remain
small on several toys but are nonzero, including 1.49 / 1.66 points against Random
on `TOY2_SKIP` and 0.39 / 0.46 on `TOY2_FULL`. They are gaps of an achieved local
iterate, not proofs of representational loss or of the K1 optimum. Full-UNO K1
sufficiency remains untested. Results: `docs/results/10a3a-results.md`.

### 5.25 Steps 10a.3B and 10a.3C: the observation contract, and an attack instrument

**10a.3B** wrote down what a Step 11 learner sees, before building one: three comparison arms (K1
feed-forward, K1 plus structured draw/play evidence, recurrent public/private history), each with an
explicit field list, the shared legality mask, colour/rank equivariance, the seat-0 frame, and what
the evaluator can query. Its load-bearing clause is negative: a recurrent target **cannot** be
queried through fabricated private history as though that were its real conditional policy
(docs/observation-contract.md, spec section 31).

**Problem for 10a.3C.** On full UNO there was still **no exploitability number of any kind**. Every
quality measure was "score against a fixed column", which cannot distinguish "learned to beat this
league" from "learned to play well" -- and that was iterations 1-4's diagnosis. LBR had been deferred
twice (the step 10 brainstorm scoped it and flagged it as the first thing to cut; 10a.2 cut it).

**Techniques** (spec section 32). An attack is a frozen-target responder whose **achieved score in
real games** lower-bounds the best-response score against that target, so it needs neither an exact
belief nor an exact Q. Four continuations, of which only `response` reads the target; a fixed
`surrogate` (the column-free construction of 10a.2), `minimax`, and a `PimcRollout` family read
nothing about it. Reported as `NashConv >= b0 + b1 - 1`, where the two unknown Nash seat values
cancel in the sum, with the standard error taken from the per-deal *sum* so the seat covariance is
carried and 0.5 is never substituted for an unknown seat value.

Calibration is exact, which is the point of having toys: an attacker is a deterministic function of
its information state, so it can be **tabulated** and scored by one `evaluate` (0.1-0.8 ms per
information set). The attack score, the BR L1 ceiling and the gap between them are then all exact.
`uno.compiled.flaws` supplies the targets that make the calibration sharp: one symmetry-closed
action class forced on or off over a Nash strategy, so the exploitability is **localized** and cannot
be collected without finding that class. Random and the Heuristic are weak everywhere and cannot
tell a sharp attack from a blunt one.

**What it bought.** On `TOY` the instrument recovers **77-97 % of each target's exact NashConv**
(missing 0.12-4.19 points of BR L1 per seat), returns a **negative** bound against Nash (-0.51 to
-4.88, clamped to the trivial zero), and separates a grafted flaw from a near-Nash target by **24.08
points**. A played attack matches the exact value on the same deals at z -0.52 / +0.26. The report
card gained an attack block (89 % and 94 % of the exact NashConv recovered for the Heuristic and
Random). On full UNO, **EVAL block 02** gives the project's first exploitability-style numbers there:
Random's NashConv is at least **54.22 +- 0.90** points and the Heuristic's at least **26.57 +- 0.96**
(4,000 deals, both seatings, at a configuration frozen from the DEV pilot before the block was
claimed). The pilot itself read those ~7.8 and ~3.9 points *higher*, which is exactly why a reported
number comes off a block nothing was selected on; the pilot also showed the bound climbing +7.0
points from w = 8 to w = 32, so it is budget-limited rather than target-limited. Against the
column-free PIMC row the attack finds nothing (+1.00 +- 6.11 on 100 DEV deals): a reported result,
not a safety claim. Results: `docs/results/10a3c-results.md`.

**What it taught.**

- **An invalid query fails silently, so the refusal has to be enforced.** The `response` continuation
  evaluates the target *inside* a determinized world, where its private history is fabricated. Asked
  there, the tabulated Nash strategy missed the keying at **233 of 293 queries (79.5 %) and every
  miss returned a point mass on the first legal action**. Nothing raises; the attack simply becomes
  insensitive. So support is *measured* per target rather than declared. Note the asymmetry: the
  **exact** instrument takes the target as a strategy over the keying and can respond to a memoryful
  target perfectly well. The restriction belongs to *sampling*, not to best response.
- **"Reads its own private history" and "reads the hidden state" are different properties, and
  conflating them fails the wrong thing.** An L1 policy is *entitled* to read its own initial hand
  and draws -- and since those plus the public plays determine its current hand, any policy that
  rebuilds its position from the L1 key answers differently when they are perturbed. My own attacker
  failed that probe when it was first used as an L1-validity test. L1 validity is the other
  direction: swap a different hidden world behind an identical L1 observation.
- **A pre-declaration can sit below the instrument's own resolution.** The world ladder was first
  declared monotone *per seat*; it failed, on wobbles of 0.12-0.29 points, against an attacker whose
  own world-sampling seed moves a single seat by 0.6-0.8 points. On the quantity actually reported --
  the two-seat sum -- it is cleanly monotone (5.16 -> 3.73 -> 2.18). Checking noise is not a check.
- **A built attacker is one realization, not the averaged policy.** Because the sampling is keyed on
  the information state, the deployed attacker is a single draw; `uno.compiled.pimc.pimc_n` is the
  average over that key. Six seeds straddle it (mean within 0.03-0.12 points of `pimc_n`), so the
  seed is a *reported* uncertainty beside the deal noise, not a formality.
- **Search Q and achieved score are different objects, and comparing them takes care.** Only the Q at
  the attacker's first decision is comparable to a whole-game score, and the pair must be collected
  per game. Two drafts were wrong -- one averaged Q over decisions at every depth, one paired 52 Q
  values against 60 games' scores -- and the surviving comparison is unresolved anyway
  (-2.70 +- 5.06, z -0.53), so its sign carries no finding.
- **A coincidence is indistinguishable from an aliasing bug without a guard.** A 120-deal pilot row
  printed b0 and b1 identical to four decimals with identical standard errors. Checked directly the
  seatings are genuinely separate (13 of 40 deals disagree), but the guard is now permanent rather
  than a diagnosis to be repeated by hand.
- **The weakest attacker family earns its place by being weakest.** `PimcRollout` recovers 77-92 %
  where the searches reach 91-97 %, and it is the only family that returns a *negative* bound against
  a genuinely flawed target. Keeping a family that visibly fails is what stops one attack's blind spot
  from being read as a property of the target.

### 5.26 Step 9.5g: where the world budget goes (a proposal, measured and mostly declined)

**The proposal.** PIMC samples worlds uniformly from the belief. A person does not: they imagine
the few worlds that *matter* ("what if he has a +2?") and weigh them by how likely they are ("but
only one is left and he already had three"), where uniform sampling spends most of its budget on
worlds that are technically distinct and informationally identical. Should PIMC do that instead?

The complaint splits in two, and only one half was unbuilt. *Weighting by likelihood* is the
belief, built in 9.5f and worth +2.83 / +4.40. *Not distinguishing worlds that make no difference*
is a **stratified (Rao-Blackwellised)** sampler -- bucket the belief by a statistic of the hidden
hand, take the bucket probabilities exactly from the belief, sample within buckets -- and its gain
is precisely the share of the per-world variance the statistic explains. That makes the proposal
a measurement, not an argument, which is why it got a step.

**Technique** (`uv run uno pimc concentration`, `uno/experiments/pimc.py`): record, at every
decision the agent faces, the full `(worlds, legal)` matrix it really averaged -- `_Probe`
records inside `_worlds_q` and inherits `policy` untouched, and is checked to replay a plain
`PimcSearch` action for action first. Then a **nested design** at every 10th decision: 16 hands
from the belief x 8 orders of the cards left over, which separates the two things a "world" is.
Every variance share is a plug-in ratio, so every one is reported against a **permutation control**
on the same grouping.

**What it found** (400 DEV deals, w = 128, d = 2, `voids`, both columns, which agree to a decimal):

- **The worlds disagree almost everywhere**: 3.59 distinct per-world argmaxes out of 4.49 legal
  actions, only **6.6 %** of decisions unanimous. The first hypothesis -- that the budget is spent
  confirming a foregone conclusion -- is simply false.
- **The hand carries 46.8 % of the per-world spread and the pile order the other 53 %.** So the
  proposal's premise is right and bounded: a *perfect* hand abstraction is worth **1.88x** the
  effective world count, **2.14x** if allocation is optimised (Neyman) as well as stratified.
- **That ceiling is one score point.** 9.5f measured doubling the worlds under `voids` at
  +1.20 / +0.77, z 1.1 / 0.7 -- unestablished. The entire idea is capped at a step the project
  already could not resolve, and establishing it would cost a fresh `EVAL_SEEDS` block.
- **The statistics a person would name recover 39-42 % of the ceiling**, so ~1.25x in practice.
  And **the motivating example is the worst feature measured**: "holds a +2 / a +4" explains
  2-3 % of the spread (4-7 % of the ceiling), against 12-13 % for dull counts -- how many of his
  cards play at all, how many are in the colour. What makes a world matter to this agent is how
  many outs the opponent has, not which frightening card he holds.

**What it taught, beyond the verdict:**

- **PIMC's decisions are far noisier than its score.** The shipped w = 32 names a different action
  from w = 128 at **12.2 %** of decisions, and two *disjoint* 64-world halves of one decision agree
  only **85.6 %** of the time -- while the score ladder is flat past w = 32. Both are true because
  the churn is between near-equal actions. 9.5f's working set stands, but "the ladder is flat"
  must never be read as "the agent has converged", and anything that reads PIMC's *action choices*
  as a signal (an attribution, a distillation target for 10C) is reading a 12 %-noisy one.
- **The one lever worth having is adaptive stopping, not better worlds.** Racing the candidate
  actions and stopping when the leader's t-statistic clears 3 uses a mean of 61 worlds and
  reproduces the w = 128 answer **98.1 %** of the time, where flat w = 32 reproduces it 87.8 %.
  That is w = 128 decision quality at ~1.9x the w = 32 cost instead of 4x, for a dozen lines on
  `_scores`. Not built -- it is off the 10 -> 11 path -- but it is the finding to spend on if PIMC
  is ever re-opened.
- **A control is what makes a variance share readable.** Every R2 here is upward biased by
  roughly (groups - 1) / (worlds - 1); the hand-identity ceiling reads 53.1 % raw and 46.8 % net.
  The test file pins that the permutation control sits *at* that bias rather than at zero, because
  a test asserting "noise scores about zero" would pass a broken implementation and delete the
  correction's reason for existing.
- **ISMCTS was considered and not pursued.** Its one structural advantage over PIMC is removing
  strategy fusion, and 9.5d already bounds that on the toys at **0.04-0.70 points** with the
  `response` continuation -- which, by best-responding to the known column, already removed the
  expensive opponent-side half (up to 8.8 points). It also has no exploitability guarantee, which
  is this project's headline quality measure. Sound search under imperfect information is CFR
  re-solving, not ISMCTS; neither is on the 10 -> 11 path.

Results: `docs/results/09.5g-results.md`.

## 6. Results: what we know

All exact unless marked sampled. Scores of the first-named player, seat 0 /
seat 1.

### 6.1 The seat effect is a toy artefact, reduced but not removed

Seat gap in points (seat-0 score minus seat-1 score of the same profile):

| profile | `TOY` | `TOY_FULL` | `TOY2` | `TOY2_SKIP` | `TOY2_FULL` | `FULL_UNO` (sampled) |
|---------|-------|------------|--------|-------------|-------------|----------------------|
| random vs random | 29.9 | 29.9 | 17.5 | 16.8 | 17.5 | 0.0 ± 3.1 |
| heuristic vs heuristic | 38.7 | 38.7 | 27.7 | 15.3 | 27.7 | 0.9 ± 3.1 |
| BR L1(random) vs random | 24.2 | 21.4 | 16.1 | 15.1 | 13.2 | |
| **Nash vs Nash** | **23.9** | **17.3** | **15.7** | **15.0** | **11.3** | |

Nash seat-0 score: `TOY` 0.6196, `TOY_FULL` 0.5866, `TOY2` 0.5784, `TOY2_SKIP`
0.5748, `TOY2_FULL` 0.5564. A third rank takes the Nash gap 23.9 → 15.7; the
voluntary draw 23.9 → 17.3 on `TOY` and 15.7 → 11.3 on `TOY2`. **The effects are
sub-additive** (12.6 together against 8.2 + 6.6; first written as "they add").
Optimal play halves seat 0's edge on `TOY2_FULL` (0.556 vs 0.620). Full UNO has
no measurable seat effect. The seat gap follows game length and per-card
playability, not deck size.

### 6.2 There is skill to learn at L1

Row matchups (exact): `TOY` random vs random 0.6493 / 0.3507, heuristic vs
random 0.7416 / 0.4147, heuristic vs heuristic 0.6934 / 0.3066. `TOY2`: 0.5875 /
0.4125, 0.6730 / 0.4474, 0.6385 / 0.3615. `TOY2_SKIP`: 0.5841 / 0.4159, 0.6414 /
0.4806, 0.5767 / 0.4233.

Edge over Random at the same seat, both against Random (points):

| ruleset | Heuristic | BR L1(random) (the L1 ceiling) | left on the table by the Heuristic |
|---------|-----------|--------------------------------|------------------------------------|
| `TOY` | +9.2 / +6.4 | +14.4 / +20.1 | 5.2 / 13.7 |
| `TOY2` | +8.6 / +3.5 | +14.3 / +15.7 | 5.8 / 12.2 |
| `TOY2_FULL` | +8.6 / +3.5 | +16.0 / +20.4 | 7.4 / 16.9 |

**The rulebot is decent from seat 0 and weak from seat 1**, consistently.
On full UNO (step 8.5, sampled, older Heuristic) Heuristic vs Random is the old
~60 % rulebot (seat mean 0.603, +11.4 / +9.3 over Random): at least as much skill
against Random as on the toys, ceiling unknown.

### 6.3 Best-response columns and values of information

Score of the best responder:

| ruleset | opponent | seat | BR L1 | BR L2 | BR L3 | L2 - L1 | L3 - L2 |
|---------|----------|------|-------|-------|-------|---------|---------|
| `TOY` | random | 0 / 1 | 0.7934 / 0.5513 | 0.8188 / 0.5937 | 0.8280 / 0.5989 | +2.5 / +4.2 | +0.9 / +0.5 |
| | heuristic | 0 / 1 | 0.7378 / 0.4361 | 0.7730 / 0.4774 | 0.7798 / 0.4786 | +3.5 / +4.1 | +0.7 / +0.1 |
| | nash | 0 / 1 | 0.6196 / 0.3804 | 0.6624 / 0.4239 | 0.6761 / 0.4311 | +4.3 / +4.4 | +1.4 / +0.7 |
| `TOY_FULL` | random | 0 / 1 | 0.8017 / 0.5882 | 0.8443 / 0.6419 | 0.8590 / 0.6941 | +4.3 / +5.4 | +1.5 / +5.2 |
| | heuristic | 0 / 1 | 0.7446 / 0.4732 | 0.8060 / 0.5131 | 0.8187 / 0.5437 | +6.1 / +4.0 | +1.3 / +3.1 |
| `TOY2` | random | 0 / 1 | 0.7305 / 0.5694 | 0.7976 / 0.6586 | 0.8191 / 0.6804 | +6.7 / +8.9 | +2.2 / +2.2 |
| | heuristic | 0 / 1 | 0.6959 / 0.4937 | 0.7728 / 0.5421 | 0.7907 / 0.5587 | +7.7 / +4.8 | +1.8 / +1.7 |
| `TOY2_SKIP` | random | 0 / 1 | 0.7425 / 0.5919 | 0.7865 / 0.6277 | 0.8063 / 0.6503 | +4.4 / +3.6 | +2.0 / +2.3 |
| | heuristic | 0 / 1 | 0.6941 / 0.5356 | 0.7538 / 0.5638 | 0.7738 / 0.5816 | +6.0 / +2.8 | +2.0 / +1.8 |
| `TOY2_FULL` | random | 0 / 1 | 0.7479 / 0.6164 | 0.8443 / 0.7508 | 0.8875 / 0.8208 | +9.6 / +13.4 | +4.3 / +7.0 |
| | heuristic | 0 / 1 | 0.7118 / 0.5371 | 0.8248 / 0.6080 | 0.8620 / 0.6600 | +11.3 / +7.1 | +3.7 / +5.2 |
| | nash | 0 / 1 | 0.5565 / 0.4436 | 0.6952 / 0.5528 | 0.7640 / 0.6271 | +13.9 / +10.9 | +6.9 / +7.4 |

Readings:

- **Hidden information grows with the toy**: seeing the opponent's hand is worth
  2.5-4.4 pts on `TOY`, 4-6 on `TOY_FULL`, 5-9 on `TOY2`, **7-14 on
  `TOY2_FULL`**; the pile order 0.1-1.4 → up to 7. Possible mechanisms (three
  ranks make hands harder to infer; a bigger pile; the voluntary draw gives the
  knowledge something to act on) are **hypotheses, not measured**.
- **Information is worth most against Nash.** One reading: the equilibrium hides
  its hand best, so L1 inference recovers less. The alternative, a hand that is
  easy to punish once seen, is not separated. (Step 8.7 first had the direction
  backwards, "L2 - L1 measures how much a policy gives away".)
- Seat dominates: even the L3 oracle moving second against Random on `TOY` only
  reaches 0.60.

### 6.4 Exploitability and the Nash agent

Exploitability of row agents (points, row at seat 0 / seat 1):

| ruleset | Random | Heuristic | BR L1(random): as solved (Nash-filled) |
|---------|--------|-----------|----------------------------------------|
| `TOY` | 17.1 / 17.4 | 5.6 / 11.8 | 1.6 / 0.4 (1.4 / 0.3) [1.2 / 0.2] |
| `TOY_FULL` | 17.5 / 21.5 | 6.0 / 15.8 | 4.6 / 1.3 (3.1 / 2.4) [2.8 / 1.1] |
| `TOY2` | 14.8 / 15.2 | 7.2 / 11.7 | 1.7 / 2.2 (1.4 / 1.7) [1.4 / 1.6] |
| `TOY2_SKIP` | 16.7 / 16.8 | 11.0 / 11.9 | 1.3 / 1.7 (1.1 / 1.1) [1.0 / 1.0] |
| `TOY2_FULL` | 17.3 / 19.1 | 9.4 / 15.5 | 6.3 / 4.9 (3.4 / 3.3) [3.1 / 2.6] |

(In square brackets: least exploitable fill-in, step 9. All under the exhaustion
rule; other scorings move these by several points, section 6.8.)

Nash row and column (seat 0 / seat 1):

| ruleset | nash vs random | nash vs heuristic | random vs nash | heuristic vs nash | BR L1(random) vs nash |
|---------|----------------|-------------------|----------------|-------------------|-----------------------|
| `TOY` | 0.7871 / 0.5235 | 0.7328 / 0.4139 | 0.4765 / 0.2129 | 0.5861 / 0.2672 | 0.6149 / 0.3792 |
| `TOY_FULL` | 0.7595 / 0.5123 | 0.7065 / 0.4372 | 0.4877 / 0.2405 | 0.5628 / 0.2935 | 0.5560 / 0.4060 |
| `TOY2` | 0.7184 / 0.5330 | 0.6784 / 0.4578 | 0.4670 / 0.2816 | 0.5422 / 0.3216 | 0.5697 / 0.4099 |
| `TOY2_SKIP` | 0.7268 / 0.5545 | 0.6712 / 0.5059 | 0.4455 / 0.2732 | 0.4941 / 0.3288 | 0.5670 / 0.4160 |
| `TOY2_FULL` | 0.7048 / 0.5478 | 0.6671 / 0.4843 | 0.4522 / 0.2952 | 0.5157 / 0.3329 | 0.5407 / 0.4238 |

Readings:

- **On forced-play toys, the best response to Random is near-Nash** (1.0-1.6
  pts exploitable with the least exploitable fill-in). Playing well against a weak player and playing
  unexploitably are almost the same thing there.
- **On free-play toys they separate, most on `TOY2_FULL`** (least exploitable
  3.1 / 2.6 pts there, 2.8 / 1.1 on `TOY_FULL`): enough room for deception that
  the exploitability column becomes informative for RL rows. First reported as
  4.6-6.3 (as solved), then 2.4-3.4 (Nash-filled, step 8.8).
- **Nash gives up little against weak opponents**: on `TOY` 0.6 / 2.8 pts vs
  Random and 0.5 / 2.2 vs the Heuristic compared with dedicated best responses. The
  cost of safety is small.
- **The Heuristic is weakest where the draw matters** (15.5 exploitable from seat
  1 on `TOY2_FULL`), and on `TOY2_SKIP` (11.0 / 11.9) its skip rule is a guess.
  Its self-play seat gap collapses from 27.7 to 15.3 on the skip toy.
- A best response to Random transfers: against the Heuristic it scores within
  0.8-1.4 pts of the dedicated best response (`TOY`, step 7, step 4 Heuristic).

### 6.5 The voluntary draw is a real decision

It is available at 60 % of full-UNO decisions (a card plays and `Draw` is legal;
how often drawing is right there is unknown), so step 8.5 asked whether it
carries skill.
BR L1 gain from allowing it (free minus forced, points, seat 0 / seat 1):
`TOY_FULL` - `TOY` +0.8 / +3.7 vs Random, +0.7 / +3.7 vs Heuristic;
`TOY2_FULL` - `TOY2` +1.7 / +4.7 vs Random, +1.6 / +4.3 vs Heuristic. **Mostly
worth it to seat 1**; worth more with information (up to +9.5 pts at L3 for seat
1 on `TOY_FULL`: with the pile known, a draw fetches a chosen card).

How often it's used (draws with a play in hand per game, seat 0 / seat 1):

| ruleset | BR L1(random) vs random | Nash vs Nash (of which claims) |
|---------|-------------------------|--------------------------------|
| `TOY_FULL` | 0.029 / 0.184 | 0.228 / 0.372 (0.039 / 0.048) |
| `TOY2_FULL` | 0.129 / 0.329 | 0.248 / 0.495 (0.055 / 0.079) |

On `TOY_FULL` it is mostly a tempo tool of the trailing seat; with three ranks
it is used more, which step 8.7 read as "a real alternative to a bad play"; that
mechanism is untested, every draw on a 2-3-card pile changes pile parity, and on the
2-card-pile `TOY_FULL` the draw's effect on the Nash seat gap changes sign with
the scoring of exhaustion (on the three-rank toys it shrinks the gap under every
scoring, -4.4 / -5.2 / -0.6 pts, step 9). **Its value is not the toy-only
claim**: forbidding the claim with a play in hand leaves BR L1 unchanged to 4
decimals on both free toys. Best responders never claim.

### 6.6 How much the evaluators help

Heuristic vs Random, 4000 held-out pairs:

| ruleset | corr(seat 0, seat 1) | var ratio mirrored / raw | paired Heuristic - Random, var ratio (seat 0 / 1) |
|---------|----------------------|--------------------------|---------------------------------------------------|
| `TOY` | -0.30 exact | 0.70 exact | 0.47 / 0.49 |
| `TOY2` | -0.31 exact | 0.69 exact | 0.42 / 0.45 |
| `FULL_UNO` | -0.10 sampled | 0.90 sampled | 0.89 / 0.89 |

Budget arithmetic (exact, `TOY`): a 3-point edge at one seat, raw, needs
1,700-2,400 games at 3σ (the intro's ~2,400 estimate holds). That is 50 % power:
a 3σ test catches a real 3-point edge half the time; 1.64x for 80 % power
(2,800-3,900), 2.4x for 95 %. Paired comparison of near-identical policies:
370-570 games (600-940 at 80 %). **On full UNO neither trick helps between
Heuristic and Random** (var ratio 0.89-0.90), but pairing near-identical variants
on a shared rng stream removes 77-85 % of the variance (step 9, section 5.5).

### 6.7 Transfer: how `TOY` differs from full UNO (step 8.5)

Decisions per game (Heuristic vs Random, both players), share of all decisions:

| decision kind | `TOY` | `FULL_UNO` |
|---------------|-------|------------|
| decisions per game | 6.2 | 44.5 |
| any choice (2+ legal actions) | 46 % | 71 % |
| which color (plays in 2+ colors) | 22 % | 11 % |
| wild: hold or spend | 12 % | 8 % |
| wild: pick a color | 18 % | 10 % |
| action card among other plays | 0 | 20 % |
| draw with a play in hand (legal, not taken) | 0 | 60 % |
| play or keep the drawn card | 8 % | 11 % |

Color and wild decisions exist on both at comparable shares (3-5x more per game
on full UNO). The two most frequent full-UNO decisions were absent on `TOY`,
which motivated `TOY_FULL`, `TOY2_FULL` (voluntary draw) and `TOY2_SKIP` (action
cards). Full-UNO games: 55 actions random vs random, 44.5 heuristic vs random,
37 heuristic self-play; no game hit the 1000-action cap. Game length (in-game
draw luck) is the part no exact toy can reproduce; it belongs to sampled
estimators validated on the toys.

### 6.8 Exhaustion: a modest share of games, a first-order rule

*Step 9: scoring exhaustion differently on the same tree (as a draw, or as a loss
for the player stuck on the empty pile) moves the Nash seat gap by up to 13
points on the forced toys (under 1 on `TOY2_SKIP`), and the alternatives
distort free play ("stuck" on both free toys, "draw" on `TOY_FULL`). Seat gaps, exploitability and voluntary-draw readings
belong to the rule; the accounting below stands.*

~19 % of `TOY` games end by exhaustion under Random, **15 % under Nash vs Nash**:
strong play doesn't drift into it. The edges of the best responder and of Nash
over Random arrive mostly through emptied hands (exhaustion contributes -2.7 to
+4.0 pts, accounting identity, not a causal split). The Heuristic *loses*
exhaustion endings (step 4 Heuristic: 0.44 within them at seat 0 vs Random's 0.57), giving back 5-6
pts, which suggests its wild-holding rule is miscalibrated (not checked). A
causal test (a ruleset scoring exhaustion differently) was not run.

### 6.9 Costs: what the RL loop will pay per checkpoint

| ruleset | load tree | tabulate agent | one BR | exploitability | CFR iteration |
|---------|-----------|----------------|--------|----------------|---------------|
| `TOY` | 0.04 s | 0.16 s | 3 ms | 7 ms | 5-8 ms |
| `TOY_FULL` | 0.05 s | 1.9 s | 62 ms | 145 ms | 110-180 ms |
| `TOY2` | cached | | 0.1 s | | 44 ms |
| `TOY2_FULL` | 1.7 s (three trees 11 s) | | 2.5-4 s | | 4.5 s |

(Tabulate uses a Python call per set, an upper bound; a network is one batched
forward pass.) **The exact side of a checkpoint is milliseconds to seconds; the
loop's cost will be training, not measurement.** Solving once: CFR+ 1000 its on
`TOY` 5 s, `TOY_FULL` ~100 s, `TOY2` 44 s, `TOY2_SKIP` 142 s; `TOY2_FULL` DCFR
150 its ~11 min.

## 7. Claims that were wrong and later corrected

Kept because they show where this kind of analysis goes wrong.

| claim as first written | correction |
|------------------------|------------|
| exact-inside-95 %-CI is the simulation check | fails 1 in 20 by design, and seed ranges carry deal luck: compare with exact *on the same seeds* |
| that on-seeds check with the score's CI | conservative; a calibrated z on the conditional (action-noise) variance replaced it |
| a `TOY2` miss at "2.1σ" is the usual 1-in-20 | really z = -3.01 under the calibrated check; pooled over 24k seeds it is chance |
| BR to Random "no longer near-Nash" on free toys, 4.6-6.3 pts, seat-1 pattern | depends on arbitrary play at unreached sets; ~half the size filled with Nash, per-seat split meaningless |
| symmetric BR 1.6 / 0.4 is an instance of "symmetrising never raises exploitability" | tie-breaks and unreached sets explain it, not symmetrisation |
| canonical observation in, action mapped back = symmetric agent | also needs averaging within stabilizer orbits (`Symmetrized`) |
| "averaging images" of a policy can't raise exploitability | true for mixing images, not for averaging behaviour probabilities |
| DCFR/PCFR+ are 4-7x faster, 30-50 iterations | 2-4x, DCFR 50-60, PCFR+ 45-80 |
| DCFR 150 on `TOY2_FULL` → 0.001, "two hours into 17 minutes" | reached 0.005; ~2x at equal accuracy |
| vanilla CFR shows the textbook 1/√T | observed ~1/T; 1/√T is the bound |
| third rank and voluntary draw effects on the seat gap add | sub-additive (12.6 vs 14.8) |
| "edge 12 vs 30 vs 38" | compared an edge with gaps: gaps 23.9 vs 29.9 vs 37.8 |
| L2 - L1 measures how much a policy gives away | opposite direction; mechanisms marked as hypotheses |
| `TOY_FULL` - `TOY` heuristic seat 1 L1 = +0.7 | +3.7 |
| mechanisms for the information gaps (ranks harder to infer) stated as facts | untested hypotheses; pile size not separated |
| the step 8.7 wild/duplicate sizing rows | deal probabilities with duplicate cards were wrong (sizes stand, R vs R moved ≤ 0.4 pts) |
| split pairs: "3 games in 5 on `TOY` are decided before anyone moves" | independent games already split 39-50 %; the deal adds 13-29 pts, and 4 % of worlds have a result no play changes (step 9) |
| "N games for a 3-point edge at 3σ" read as a budget | that N has 50 % power; 1.64x for 80 % (step 9) |
| "on full UNO neither pairing trick helps; AIVAT is the only way" | true between Heuristic and Random; near-identical variants on a shared rng stream: 77-85 % removed, tickets another 3-20 % (step 9) |
| "the voluntary draw shrinks the Nash seat gap"; "with three ranks a real alternative to a bad play" | holds on the three-rank toys under every scoring of exhaustion (-4.4 / -5.2 / -0.6 pts), but on the 2-card-pile `TOY_FULL` the sign changes with the scoring (-6.6 / +6.1 / -11.0); every draw changes pile parity; mechanism untested (step 9) |
| "exhaustion is less of a distortion than feared" | no scoring is neutral: Nash seat gap moves up to 13 pts on forced toys, the alternatives distort free play (step 9) |
| BR L1(random) filled with Nash measures the free-play separation | the Nash fill is not the least exploitable, and on `TOY_FULL` worse than the solver's at seat 1; least exploitable 2.83 / 1.14 (step 9) |
| (step 9 review) "on `TOY` hidden information favours seat 0" generalised | `TOY` only; on `TOY2` and `TOY2_SKIP` seeing everything helps seat 0 |
| "fitted L3 critic" for AIVAT (brainstorm, plan) | the value function must not read the pile order: an L2-state critic (step 9.5b) |
| (09.5a results) memory table "score of the responder" | the values are returns (step 9.5b) |
| (09.5a results) "6.01 / 17.35 ≈ 7.4 / 16.9, `TOY_FULL`'s own row" | 7.4 / 16.9 is `TOY2_FULL`'s row; `TOY_FULL` is not in that table (step 9.5b) |
| (09.5a results) "a large minority to a majority of decisions are close calls", linked to "48 % of sets at a non-trivial stabilizer" | direction reversed (33-74 % are close calls) and the 48 % is K2's share of L1 sets, not stabilizers (284 of 12,036); mechanisms for the Nash gaps marked as hypotheses (step 9.5b) |
| (9.5b plan) place the full-UNO critic on the toy R² → variance-ratio curve | the ratio is flat across game lengths while R² falls four-fold: R² does not carry (step 9.5b) |
| (9.5b) "a PIMC value is a candidate V for AIVAT" | a rollout V costs more games than it saves and its noise makes the estimator worse than raw (ratio 3.99 at n = 16); PIMC is the *teacher* of a fitted network (step 9.5c) |
| (9.5b) "a better full-UNO V has up to twice the gain to collect" | the floor is 0.086-0.090, not ~0.4, so up to 11x; the fitted network takes 2.2x of it (step 9.5c) |
| (9.5c plan) teacher targets at n = 8 rollouts per state | at a fixed budget, n = 2 on four times as many games fits better (step 9.5c) |
| (step 9.5a plan) "one `best_response` pass with `own` fixed gives the same value on the quotient as on the full tree" (K0 / K1 / K2) | step 9.5a correctly withdrew the claim because its quotient partition leaked the full-L1 coordinate frame; 10a.3A repairs the partition and checks the **same lifted policy's** value across full and quotient trees. Separate local optimization runs still need not choose the same iterate. |
| (9.5 brainstorm) the toys' PIMC-n curve anchors the full-UNO world budget | it does not: a toy belief holds 2-23 consistent worlds at the decisions actually faced and full UNO's holds astronomically more, so "n = 8 recovers nearly all of PIMC-infinity" is a statement about toy belief sizes (step 9.5d) |
| (9.5d) PIMC-n "converges to PIMC-infinity from below" | not always: it overshoots on `TOY` vs the Heuristic at seat 1, because PIMC-infinity maximises the belief-weighted *perfect-information* score and not the true value, so the deterministic limit need not be the best member of its own family (step 9.5d) |
| (9.5d, first draft) "worlds help; depth hurts" | the first half holds at many sigma (0.415 -> 0.595 -> 0.695 for 2 -> 8 -> 32 worlds); the second was read off differences of about **one standard error** at 200 unpaired deals. Retested paired: depth 4 is 7.5 / 6.0 points below depth 2 at w = 32 for 8.3x the cost, monotone in depth, but only z -1.7 / -1.3 -- so depth is *dominated*, not *proven harmful* (step 9.5d section 3a) |
| (9.5d) "sharing the world-sampling key makes these comparisons tightly paired" | it removes 8 % of the standard error when the configs differ in *depth* and 18-20 % when they differ in *worlds* (where the larger config's first `n` worlds are exactly the smaller one's), not most of it either way: the coupling lasts only until the two agents' games diverge, which on a ~50-decision game is soon, unlike step 9's near-identical policy variants on a shared stream (77-85 %) (step 9.5d) |
| (9.5c) AIVAT's corrections are cheap next to a search agent | they are 4-11 % of a search game rather than negligible, and for a *deterministic* search agent the row correction is identically zero, so the gain comes from chance and the opponent only (step 9.5d) |
| (9.5b brainstorm) "PIMC on full UNO is a strong row" | only with enough worlds: at 2 worlds, and as a rollout player at 8, it scores *below* `PlayFirstRandom`, because the argmax of a few noisy per-action estimates is biased towards whichever action's noise came out high (step 9.5d) |
| (9.5d) fixing the seat networks is "a one-line change" | one agent object plays both seatings, so the seat is unknown at construction and the leaf must be resolved per observation (step 9.5e) |
| (9.5e scoping) the `response` continuation drops alpha-beta, so expect 2-4x the cost per world | it is *cheaper* (128.7 against 188.9 wall-clock ms/game): expectimax expands only the opponent's support, and the rulebots are far from uniform, so the lost pruning is more than repaid (step 9.5e) |
| (9.5e scoping) a `TOY` closure test validates the sampled continuation rule | vacuous on that axis: the two rules are 0.2-0.5 points apart there against ~3.2 points of noise at 200 games, so the test would pass with the wrong one wired in. Moved to `TOY_FULL` (7-9 points apart) and backed by a noise-free "the two rules choose differently" test (step 9.5e) |
| (9.5d) the full-UNO PIMC row loses to the Heuristic at seat 1, and worlds/depth have stopped helping | that row used the two-sided continuation and the seat-0 network at both seats; with the `response` continuation and per-seat leaves it beats the Heuristic at *both* seats, so "the ladder is flat past 32 worlds" was measured under approximations that have since changed and needs re-testing (step 9.5e) |

| (`plan.md`, from the start) sampled L1/L2 inference is a **particle filter over the deal** | it collapses: 64 particles survive the opponent's first five plays 0.156 -> 0.018 -> 0.002 -> 0.0004 -> 0, because a particle lives only if it holds the played card (~7/98). The built version scored *worse than the prior it replaced* (Brier 0.167 against 0.085). Replaced by rungs that condition instead of weighting (step 9.5f) |
| (9.5e) inference is worth deferring: the toys put it at 0.0-0.6 points against Random, the headline column | +2.83 / +4.40 measured on full UNO against Random, established at both seats. Toy games are too short to accumulate the draw evidence the constraint feeds on (step 9.5f) |
| (9.5f scoping) the soft policy likelihood is the point, the hard constraints a cheap approximation of it | reversed: `window - voids` is +0.93 / -0.87, within noise at both seats. The rung needing no policy *probabilities* carries the whole measured gain, at +10 % cost instead of +29 % (step 9.5f) |
| (9.5f) the `voids` rung can only bite on free-play rulesets, since forced play has no voluntary draw | wrong: on forced play `legal_actions` offers `Draw` only when nothing is playable, so the *pre-draw* void is a rules fact rather than a policy inference. Live at 16.5-23.3 % of decisions on the forced toys (step 9.5f). **Half-right only**: the constraint's arithmetic still assumes the card drawn *at* the void was not kept, which is policy, not rules (follow-ups) |
| (9.5f) the belief costs 3.4x the search | that measured a harness calling it 64 times at *every* decision; the agent calls it once per world at its own decisions only, so the real cost is +10 % / +29 % (step 9.5f) |
| (9.5f, 8.8) `soft` at decay 0.1 "costs nothing this measurement can resolve", with the score question left open for a larger run | the question was smaller than the instrument, not the other way round: `soft` and `voids` choose the same action at all but ~1 decision in 100 against an honest column (19.5 % against a re-seeding control's 18.4 %), so no affordable score run could have separated them. Closed by `divergence` and the exact closure test rather than by deals (follow-ups) |
| (9.5f, 8.6) `soft` is the rung "for an opponent that might not be honest" -- on the argument that it never excludes the truth | right, and now measured against that opponent's *exact* posterior rather than argued: against `KeepsDrawn` the rungs swap places, `voids` closing 16-19 % of the distance to the posterior where `soft` closes 52-67 % (follow-ups) |
| (9.5f, 8.1) the closure test's averages describe what `voids` is worth | they described it only over the sets where its assumption holds: a set where the rung kept *nothing* was dropped from the average, which is exactly the flattering half and exactly the sets a dishonest column reaches. Kept in since the opponent became a parameter; the honest column is unchanged to the digit, so the fix bites only where it should (follow-ups) |
| (steps 7-9.5f) `EVAL_SEEDS` is held out, "nothing is ever chosen on them" | broken from 9.5d on: the full-UNO working set (`response`, w = 32, `voids`) was *chosen* by 3,000-4,000-deal runs on those seeds, so the range has been a dev range for three sub-steps. Recorded as spent block 00 of the new registry rather than quietly retired (step 10a) |
| (10a brainstorm, A2) the batched tabulation is "batch the policy call, for a network policy" | the policy call is 12 % of `tabulate`; `observation_from_key` is 57-62 %. Batching the forward alone would have left ~35 s of Python on `TOY2_FULL`. Both halves built (step 10a) |
| (10a brainstorm, A2) A2's batched tabulation "is written against whichever encoding it is" | it is not: on a quotient tree every key is already canonical, so no `canonical` call is needed and orbit averaging is one `bincount`, the same either way (step 10a) |
| (10a brainstorm, 3) the Heuristic sits at ~0.4 / ~0.2 on the normalised skill scale | measured, 0.64 / 0.32 vs Random and 0.71 / 0.27 vs the Heuristic on `TOY`. The 2:1 seat ratio is the robust part; the level was quoted before anything computed it (step 10a) |
| (10a brainstorm, 5.1) option B's cost is orbit averaging at every forward pass | that understates it: a plain encoder is *refused* by `tabulate` as not orbit-uniform, so without `Symmetrized` in the path it yields no exact cell on the quotient at all (step 10a) |
| (10a, first draft of the results) `batch_l1_observations` is 3.0-3.7x | those were the scratchpad prototype's numbers; the shipped function also rebuilds the discard pile per set (0.3-0.6 us), so it is 2.6-3.0x. Measuring the prototype and shipping the function is how a speed-up quietly becomes a smaller one (step 10a) |
| (8.7, and section 6.4 / `index.md` since) the Heuristic's `TOY2` seat-1 exploitability is 11.8 | **11.7** (11.7480). The published figure was the difference of two *rounded* values -- BR L1 0.6959 minus Nash 0.5784 is 0.1175 exactly, which rounds up, where the doubles give 0.69587054 - 0.57839081. Double rounding, and only on the one cell that lands on a boundary. Found by the report card on its first run against a ruleset it was not developed on (step 10a) |
| (10a, during the build) that 11.8 / 11.7 disagreement is the checker's banker's rounding | no: the value really is 11.748 and really does display as 11.7. Diagnosed from a 2-decimal print that showed "11.75", which was itself rounded -- the same double-rounding mistake, made one level up while investigating it (step 10a) |
| (10 brainstorm, A4) "the **L3** deal potential equals AIVAT's deal correction term on the same seeds" | it is the **L2** potential that equals it, array-equal per game -- AIVAT's value function may not read the pile order (section 23). The L3 potential has the same mean and a *larger* variance (0.47 / 0.53 against 0.42 / 0.49 on `TOY`), and is admissible for the deal term alone, where more information is strictly better (step 10a.2) |
| (10 brainstorm, 5.2) the corrections may be affordable in a training loop because "the siblings the corrections need **batch**" | their *values* batch; the sibling **states** must be built by the engine first, and that is 73 % of the floor. Measured on `FULL_UNO`: 55x the step's own `V(s)` calls with the features vectorised, 73x as shipped. The deal correction alone is one row a game (step 10a.2) |
| (`plan.md`, from the start) the shaping potential "should be a function of the agent's own information set (an L1 value)", since an L2/L3 potential adds noise the agent cannot condition on | right for a per-step shaping reward, wrong for the **deal** term, which is subtracted from every return and does not depend on the action at all. The L1 potential removes 0.28 / 0.18 of the outcome variance where L2 removes 0.42 / 0.49 and L3 0.47 / 0.53 (step 10a.2) |
| (10a.2, during the build) a z of -5.86 against the exact gradient for a subset that is provably unbiased, read as a bias | the projection was onto an orbit whose set was visited **once in 200 games**; the sample standard error of a statistic non-zero in one game is not evidence. 23 visits give z -1.18 and 232 give -1.13. Now the projection is chosen among reached sets and no z is checked below 30 contributing games (step 10a.2) |
| (9.5e) the `response` continuation is worth +9.73 / +4.57 because each world is solved as a best response to **the column's** known policy, which is what makes the row opponent-specific | the gap over two-sided minimax is real and reproduces (+5.75 / +3.75 with `voids` on both sides, z +6.6 / +4.4), but almost none of it needs the column: a row best-responding to a **fixed surrogate** it never varies collects it, and the instrument's advantage over that row is only +1.70 / +1.75 (z +2.1) against Random and +0.27 / +0.85 against the Heuristic. What the continuation buys is assuming a *competent* opponent rather than a clairvoyant one, not assuming *this* one (step 10a.2) |
| (10a.2, during the build) the exact gradient's finite-difference check passes | it passed on twelve orbits whose gradient is **identically zero** -- single-orbit sets, where `sigma == 1` and there is nothing to differentiate, and they carry most of the reach so ranking by reach selects exactly them. Also: requiring *full support* on the set leaves nothing to check at all on a free-play ruleset, where `PlayFirstRandom` puts zero on `Draw`; only the perturbed orbit needs positive probability (step 10a.2) |

## 8. Dead ends and things deliberately not done

- **rlcard, pyspiel** as the core: not used (section 3).
- **Rejected toys**: the original 13-card candidate (too big), 2-color decks (a
  pure race, seat 0 72 %), hand 4 on 9 cards (no pile), duplicate ranks (race
  returns), no play after draw (worse), +2 cards (58 % exhaustion), every
  13-card deck (≥ 34M forced, > 1B free), bigger decks with more wilds/colors (no
  seat-parity gain for 100x the size).
- **numba kernels, float32**: measured unnecessary.
- **DPCFR+** (combined discount + prediction): no gain over either alone.
- **Mirroring / pairing as the full-UNO evaluator**: gains don't transfer
  (~10 %).
- **Minimal-exploitability fill-in** of unreached sets: first skipped (Nash
  fill-in used), then done in step 9 (`cfr(fixed=...)`, section 5.12).
- **Exact mirrored moments on `TOY2_FULL`**: not computed (light mode), possible
  from the cached world tree.
- **Causal test of the exhaustion rule**: not run.
- **Pairs as a training signal**: explicitly not; mirrored scores are an
  evaluation metric, never a PPO reward.

## 9. Next: step 9.5 (AIVAT and sampled methods) and beyond

### Build order

| step | what | status |
|------|------|--------|
| 1-2 | spec, engine, property tests | done |
| 3 | census, pick the toy | done |
| 4 | Random / Heuristic agents, runner | done |
| 5-6 | exact BR L1 / L2 / L3 | done |
| 7 | mirrored evaluator | done |
| 8 | compiled tree, CFR+ Nash L1, exploitability | done |
| 8.5 | transfer check vs full UNO | done |
| 8.6 | symmetry quotient, bytes keys, cache; `TOY_FULL` solved | done |
| 8.7 | TOY2 family sized and solved, DCFR / PCFR+, validation | done |
| 8.8 | review, methodology fixes M1-M4 | done |
| 8.9 | duplicate-card deal fix, rng-name check, cache version, full rerun | done |
| 9 | review follow-ups (`docs/results/09-review.md`, `09-results.md`): split share, test power, exhaustion scorings, two-sided values, equilibrium choice, fill-in, deal model, pairing on full UNO, ticket draws, speed-ups; the review's section D (where the skill is, memory ceiling, metrics) is still to discuss | done |
| 9.5a | exact: memory keys (K0/K1/K2), edge-by-decision decomposition, fixed-opponent metric, deal model at 100k seeds, divergence statistic, node lookup, sampled D2 cross-check; brainstorm and section D verdicts in `docs/results/09.5-brainstorm.md` | done |
| 9.5b | AIVAT: exact validation on the toys (exact moments, scorings, tickets, imperfect-V curve, deal term at 5M seeds), chance model with reshuffle, L2-state logistic critic, `MID_12` / `MID_25`, full UNO, transfer curve (`docs/results/09.5b-results.md`) | done |
| 9.5c | PIMC values as a teacher: determinized rollouts against the exact toys, the deal floor, a symmetric value network as AIVAT's V (full UNO 0.71/0.76 → 0.46/0.47), the Monte Carlo deal mean, the cost of rollouts as V (`docs/results/09.5c-results.md`) | done |
| 9.5d | PIMC agent: exact PIMC-∞ on the toy world trees vs BR L1 / L2 (with and without opponent-reach weighting), PIMC-n, the full-UNO PIMC **search** row evaluated with 9.5c's network (the rollout player is a weaker comparator), full-UNO D2 with a search Q, and the worlds-versus-depth budget study (MCTS deferred) (`docs/results/09.5d-results.md`) | done |
| 9.5e | the `response` continuation (each determinized world solved as a best response to the known column: +10.2 / +5.2 points on full UNO, and *cheaper* than two-sided), per-seat leaf networks (9.5d used the seat-0 one at both seats), AIVAT on the new row; inference deferred to a particle-filter sub-step (`docs/results/09.5e-results.md`) | done |
| 9.5f | **PIMC inference**: the particle filter over the deal that `plan.md` specified **does not work** (survival 0.156 -> 0 over the opponent's first five plays; the built version scored worse than the prior it replaced). Replaced by rungs that condition instead of weighting -- `voids` (a draw proves the drawer held nothing playable; no policy *probabilities*, though it assumes a playable drawn card is played at once, so the row does **not** stay general unchanged) and `window` (plus the opponent's real probabilities over their trailing no-draw run of plays). **+2.83 / +4.40** points against Random at both seats for `voids`, where the toys predicted 0.0-0.6; the soft rung adds nothing measurable on top (`window - voids` +0.93 / -0.87) (`docs/results/09.5f-results.md`) | done |
| **10** | **formalise the toy results: figures and tables that RL relies on** | **next** |
| 11 | RL | |

Framing (step 9 review): the toys exist to validate the sampled methods of full
UNO; transfer of toy conclusions is secondary, and the toys are not tuned for
realism.

### AIVAT (Burch et al. 2016, arXiv:1612.06915)

A provably unbiased, low-variance value estimator (>10x fewer hands in poker).
Given any value function V(state) and the explicit strategies, at every chance
node and every decision whose distribution is known, add a control variate:
V(state actually reached) − E[V over what could have been reached]. Each term
has zero expectation, so the estimate stays unbiased whatever V is; a good V
cancels most of the luck.

Planned design and checks (from the plan and step 8.9):

- **V on the toy**: the exact value of the state under the *two policies being
  evaluated* (evaluate from that state as root; an RL row needs its policy
  tabulated). A best-response value is also unbiased but won't zero the
  variance. **On full UNO**: a PIMC estimate or a learned critic.
- **Checks**: V = constant reproduces the raw estimate exactly; V = exact
  matchup value drives variance to ~0, *provided* corrections are applied at
  chance nodes, the evaluated agent **and** the opponent (whose policy is
  known on the toy; with corrections only at chance + evaluated agent, a
  stochastic opponent's noise stays in); empirical variance reduction > 1 on
  held-out seeds, with V never fitted on the eval seeds.
- **Lookup**: built in step 9.5a (`uno.compiled.lookup.NodeIndex`, spec section
  21). On the full `TOY` tree the L2 keying has exactly one node per set, so the
  L2 key of a played state *is* its node; on the **quotient** up to `|G|` nodes
  share a canonical L2 set (equal values for a symmetric profile), and
  `NodeIndex` picks the tree's lowest-numbered one as the representative. An
  arc is found through the set-action (`k.arc_ia` / `k.ia_action`, not
  `tree.arc_action`, which is in each node's own frame, not the set's shared
  canonical one): found the hard way, by a test that compared the looked-up
  node's exact value to an independent reference. Works directly on the
  quotient; no need to fall back to the full tree.
- **Data**: the `play_pairs` traces already hold every acting player's
  probabilities, so AIVAT is a fourth pure estimator over existing games (raw,
  mirrored, paired, AIVAT).
- **Validation target**: AIVAT removes deal and pile luck, so over a seed range
  it estimates the **global** exact value. Check it against that with its own
  sampled variance, **not** the on-seeds value (the M1 check does not apply).
- **Closed in step 9.5b**: `uno.game.chance.draw_outcomes` models reshuffles
  (a penalty draw across a reshuffle is skipped, 0.004-0.009 % of decisions).
- **As built (step 9.5b)**: all of the checks above hold. On full UNO V is a
  fitted L2-state logistic critic, not PIMC. **Step 9.5c**: rollout (PIMC) values are
  too noisy and too costly to *be* V (ratio 3.99 at 16 rollouts per value, worse than
  no corrections); they became the teacher of a fitted network, which reaches
  0.46 / 0.47 against the critic's 0.71 / 0.76.
- Why it matters: the toy evaluators already work, but full UNO's luck is
  in-game draws where mirroring and pairing fail (§6.6). AIVAT corrects at every
  draw.

### Steps 10-11 (as first planned; since the step 9 review, step 10 formalises the toy results for RL and the items below feed it)

- **Step 10, skill check**: if BR L1 vs Random on the toy were close to the old
  rulebot numbers, PPO had nothing to learn and the fix would be the game. On the
  toys that is already answered in part: the rulebot leaves 5-17 pts.
- **Checkpoint quality** without luck: tabulate the (symmetrised) policy over the
  L1 keying → exploitability per seat (ms to s), BR L2/L3 columns, Nash row;
  compare checkpoints with **paired** comparisons (or AIVAT), never raw winrates.
  Give each checkpoint its own agent name.
- **RL agent** made symmetric with `Symmetrized` (canonical observation, orbit
  averaging, action mapped back), which is also a |G|-fold data-efficiency gain,
  and keeps every exact cell on the quotient.
- **Sparse reward**: potential-based shaping (policy-invariant for Φ(terminal) =
  0), with an **L1** potential (Nash L1 or BR L1 set values, which come out of the
  same passes; a PIMC value or critic on full UNO). An L2/L3 potential depends on
  cards the agent can't see, so it only adds noise.
- **Self-play evaluation** by exploitability, not winrate vs Random.

## 10. Practical notes (for sessions in the repo)

- Python 3.14, `uv sync`; experiments run as `uv run uno <experiment> [args]`
  (`uv run uno list`): `census`, `ladder`, `mirror`, `cfr`, `transfer`,
  `symmetry`, `sizing`, `loop_cost`, `cfr_variants`, `toy2`, `toy2_columns`,
  `validate`, since step 9 `robustness` (exhaustion scorings, two-sided
  values, equilibrium choice, fill-in, deal model) and `pairing` (ticket draws,
  near-identical policies on full UNO), and since step 9.5a `edge` (D2
  edge-by-decision, D4 `br_gap`, the sampled cross-check), `memory` (D3 memory
  ceilings, K0/K1/K2) and `deal_model` (the deal-model check at 100,000 seeds),
  and since step 9.5b `aivat` (toys, exact validation) and `aivat_uno` (the
  intermediate-ruleset sweep, critics, AIVAT on `MID_12` / `MID_25` / full UNO), and since step 9.5c
  `pimc_value` (`check` rollouts against the exact toys, `floor` the deal's share of the variance,
  `pilot` the teacher/student trade-off, `aivat` the network as V, `mc` the cost of rollouts as V),
  and since step 9.5d `pimc` (`exact` PIMC-infinity on the toy world trees, `pimcn` the finite-world
  curve, `uno` the full-UNO search row under AIVAT, `d2` edge attribution with a search Q, `sweep`
  the worlds-versus-depth budget study). Since step 9.5e `pimc sweep` takes an `s<worlds>x<depth>r`
  config token and `pimc uno` a trailing `r` argument, both selecting the `response` continuation
  (each determinized world solved as a best response to the known column). Since step 9.5f
  `pimc belief [deals] [worlds] [depth] [workers] [column=random|heuristic]` runs the belief ladder
  (`prior` / `voids` / `window`, docs/spec.md section 26) against the named column -- the column is
  part of the question, since Random's play choices are nearly uniform and so say little, which is
  why the toys put the belief axis at 0.0-0.6 points against Random but up to 2.6 against the
  Heuristic -- and a further `v` / `w` on a `sweep` config token (`s32x2rv`) climbs any ladder
  *under* a belief. The belief is a separate field of the config's meta tuple, never folded into
  the continuation letter: the ladder groupings filter on that letter and would silently drop the
  rows rather than group them. `pimc calibration` checks that a belief is *correct* rather than
  merely confident -- the
  instrument that refuted the deal-time particle filter, with two pre-declared checks (no rung
  materially worse than the exact prior; no populated reliability bin missing by more than 0.25).
  Unlike `belief` it runs single-process, so it also reports the fallback rates; `belief` itself
  carries two checks of its own (the rungs scored the same deals; no rung 3 sigma *below* the
  uniform prior at a seat), added because as shipped in 9.5f it registered none and its
  "0 failed checks" line was vacuous. `pimc closure` closes the belief against the exact toy
  posterior and takes the **opponent that posterior belongs to** (`rulebot` or `passer`, the
  latter `uno.agents.baseline.KeepsDrawn`), which is what turns it from a check of the
  implementation into a measurement of the assumption; `pimc divergence` measures how often two
  rungs choose different actions against a re-seeding control, the cheap pre-check before any rung
  comparison is taken to the score ladder; and `pimc decay` sweeps `soft`'s decay against both
  columns at once, scoring each by total variation to that column's exact posterior.
  Since step 10a `report [ruleset] [agent] [iterations] [attack_worlds]` prints the report card of one
  agent (spec section 27.2) and checks every number against the published one to the
  digit, and `uno.eval.seeds` holds the eval-block registry (section 27.1) -- a new
  reported number claims a block there rather than reusing `EVAL_SEEDS`, which 9.5d-9.5f
  spent on selection.
  Since step 10a.3C `lbr exact | games | pilot | eval` is the attack instrument (spec section 32,
  `uno.eval.attack`): `exact` calibrates it on the toys by *tabulating* the attacker, so the attack
  score, the BR L1 ceiling and the gap between them are all exact; `games` checks the played score
  against `scores_on_seeds` (the exact value on the *same* deals, never the whole-game value, which
  differs by 1-2 points of deal luck); `pilot` is the bounded full-UNO DEV cost/sensitivity run; and
  `eval` is the frozen confirmatory measurement on EVAL block 02. The report card grew an attack
  block from the same module. Only the `response` continuation reads the target, so only it is
  restricted to position-only targets -- a tabulated target answers "first legal action" at 79.5 % of
  fabricated-history queries instead of raising. `uno.compiled.flaws` grafts one symmetry-closed
  action class onto Nash to make a target whose flaw is *localized*, which is what a vacuous
  attacker cannot pass.
  Tests: `uv run pytest` (**340 tests, 8 min 0 s** at step 10a.3C; 269 in 6 min 54 s at step 10a; 249 in 6 min 43 s after the 9.5f follow-ups; 206 in 7 min 27 s at
  step 9.5f, 154 in 6 min 44 s at 9.5c, 197 at 9.5e). Suite time is noisy on a loaded machine rather
  than monotone in the test count -- 9.5e logged an unexplained +3 min 40 s jump with no cause
  established, 9.5f ran four minutes *faster* with nine more tests, and 249 tests now run faster
  than 206 did. Check a new file's own runtime directly. Everything: `bash scripts/rerun_all.sh`
  (~8 h 30 by the script's own estimates, of which only the 18 steps of the step-8.9
  rerun have measured durations, 115 min; everything from `robustness` on is a first-run guess,
  logs in `logs/rerun-*/report.txt`).
- `.cache/` holds compiled trees and CFR strategies (`TOY2_FULL` trees are
  1.9-3.0 GB each and 2-5 GB in RAM; about 8 GB total). Bump
  `uno.store.CACHE_VERSION` when a compile's output would change.
- Code layout: `uno/game` (rules, engine, observe, keys, chance, symmetry),
  `uno/agents` (baseline, tabular, symmetrized), `uno/reference` (slow trusted
  walkers: census, solve, mirrored, sizing), `uno/compiled` (tree, solvers,
  strategies, cfr), `uno/eval` (play, estimate, aivat, critic, and since 9.5c pimc, value_net:
  the value network needs torch, a dependency since 9.5c), `uno/store.py`,
  `uno/experiments/` (one module per step).
- About the author: comfortable with RL and PPO, newer to game-tree methods
  (minimax, CFR, best response, OpenSpiel); explain those plainly.
- A session starts by reading this file and `git log`, then brainstorming.
- Workflow: brainstorm the next step before building (recommendation +
  pros/cons, few questions); a review of existing code alongside, with findings
  batched into an "X.5"-style fix step; results go in `docs/results/<step>.md`
  (header: commit + rerun command) and `index.md`; algorithm/rule changes go in
  `docs/spec.md`; build order in `docs/plan.md`; **this file** is updated at the
  end of each step. The user makes the git commits.
- Keep tests bounded: cap hypothesis example size by census, small
  `max_examples` when an example calls a reference solver, long checks in
  experiments. Check a new test file's runtime before handing over: a sudden
  jump in suite time (it once went from ~1 to 10+ minutes in one commit) usually
  means an unbounded example, not a thorough test.
