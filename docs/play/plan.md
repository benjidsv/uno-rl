# Play demo: analysis and network-visualization plan

Status 2026-09-24: planned, not started. The play demo in `uno/play/` works; this plan
turns it from "play against a bot" into "play against the model while watching it think". It is a demo, not an
RL result: nothing here changes training, and any measurement worth keeping for RL
research (the critic calibration in phase 0 may be) gets written up under
`docs/rl/results/` as usual.

## 0. Where the demo stands

`uv run uno play [--ckpt PATH[=LABEL] ...] [--cards DIR] [--ruleset NAME]` serves a
one-page table on `127.0.0.1:8765` (stdlib `ThreadingHTTPServer`, no new dependencies).

| File | Role |
|---|---|
| `uno/play/server.py` | `Game` (engine state, human seat, per-game bot `Player`), `App` (opponents, games, sprites), HTTP handler, CLI |
| `uno/play/opponents.py` | `AgentPlayer` (stateless rulebot on L1), `NeuralPlayer` (actor export; keeps evidence and recurrent memory per game, advanced exactly once per own decision like the collector), `rulebots()`, `checkpoint()` |
| `uno/play/static/index.html` | the whole front end: CSS cards or sprites, CSS-variable hand layout, Web Animations flyers, log with the played action's probability, per-opponent W/L in localStorage |
| `uno/play/slice_sheet.py` | cuts the classic spritesheet into `static/cards/*.png` (git-ignored Mattel art); coloured wild variants are used on the discard |

Protocol: `POST /api/new` and `POST /api/act` return `steps`, one per engine action,
each `{who, action, prob, events, view}`; the page replays them. `view` is the human's
L1 view plus the opponent hand at the end. `POST /api/load` adds an actor export.

Verified: all 8 rulebots, A and B exports and a random-init C export play full games;
`NeuralPlayer` probabilities equal the stateless `NeuralAgent` exactly for B. Also
fixed in passing: `load_actor` crashed on any arm-C export (schema-2 fallback built
for C). Checkpoint exports: `logs/<run>/checkpoints/{best,final,latest}_actor.pt`;
full checkpoints with the critic: `logs/<run>/checkpoints/{best,final,latest}.pt`.

## 1. Goals and non-goals

Goals, in order:

1. **Coach overlay.** At each human decision, show what the L1 model would play from
   the human's seat, and separately what the L2 actor ("peeks at the hand") would.
2. **Value curve.** A win-probability line over the game from a critic, only after its
   calibration is known (phase 0).
3. **Card-grid network view.** The network's internals drawn on the deck layout:
   activations per layer, and an attribution of each decision.
4. **C memory as a belief map.** A per-token linear probe on C's memory, shown against
   the true opponent hand and the `voids` baseline.

Non-goals: rollout search as a feature (kept only as the calibration reference and an
optional "analyse" button), raw weight visualizations, attention-style saliency
presented as explanation, anything that feeds back into training.

## 2. Information boundaries (read before implementing any of it)

The demo sees everything; each indicator must be computed at its declared level.

- **L1 coach** gets only the human seat's L1 observation. Its recurrent memory, if C, is
  a *separate* `NeuralPlayer` for the human seat, advanced once per human decision in
  order -- it must also be advanced on human decisions where the coach is hidden, or
  its memory desynchronizes.
- **L2 coach** (`arm CRITIC` actor export, loads through `load_actor`, prints the
  cheating warning) sees the human's L2 observation: the opponent's hand, never pile
  order or seed. Label it in the UI as peeking. `NeuralPlayer` already switches to L2
  for `spec.has_opponent`.
- **Critic** sees L2 of its own seat and predicts the return of *the learner's policy*
  from that seat, against the league average (the opponent identity is hidden from
  it). Consequences: (a) against one specific opponent it is biased, so calibrate per
  opponent; (b) on the human's seat it means "the model's estimate if it took over
  your seat now", not your own win probability; label it that way.
- Network views of the bot's decisions use the bot's own inputs only. Attribution for
  the coach uses the coach's inputs.
- Never let an analysis call advance a recurrent memory. Analysis runs on a *copy*
  of the memory (`Memory.state` tensor clone plus a copy of its `EventCounter`), or
  captures what it needs during the one real step.

## 3. Phase 0: critic calibration check (small, first)

Question: is `(v + 1) / 2` from a full checkpoint's critic calibrated as a win
probability against each league member, and at which points in the game?

- Script `uno/play/calibration.py` (or an experiment module if it becomes an RL result):
  load `best.pt` (critic state + `config`), rebuild the critic with
  `make_critic(spec(rs, "CRITIC"), model_config)`. Verify how `payload["config"]`
  (`Config.resolved()`) maps back to a `ModelConfig` (`config_io.model_from_dict`,
  `config_io.from_dict`) -- do not guess.
- Play the checkpoint's actor vs each of the four league bots, both seats, DEV seeds,
  ~2k games per cell, batched like `uno/rl/eval/evaluate.py` (reuse `LiveGame`).
  Record `(opponent, seat, decision index, turn, v, final score)` at every learner
  decision.
- Also check what the critic was trained to predict (plain terminal return vs a
  corrected target: `uno/rl/algo/ppo.py`, `train.py`). If the target is not the
  terminal return, the "win probability" reading needs its own justification.
- Output: reliability diagram per opponent (10 bins, per-bin SE), overall and split by
  game phase (e.g. turn < 10, 10-30, > 30); Brier score vs a constant per-opponent
  baseline; explained variance for reference. Low EV alone is expected in a
  luck-heavy game and is not the criterion.
- Decision rule (declare before running): show the raw critic if every per-opponent
  bin with >= 200 samples is within 0.05 of its empirical rate; else fit a
  per-opponent Platt correction on one DEV block, check on another, and show that;
  if even corrected it misses, show rollout values behind an "analyse" button only.

Rollout reference (also the optional button): from the current state, sample
continuations with the learner's own memory copied (resampling only what that seat
does not know keeps a recurrent memory consistent, since its L1 observations are
unchanged) and the opponent's actual policy; batched, a few hundred games per query.

## 4. Phase 1: analysis API and coach overlay

Server:

- `App.analysts`: named analysers loaded with `--coach PATH[=LABEL]` (repeatable) and
  `--critic FULL_CKPT`. The L1 coach defaults to the currently selected checkpoint
  opponent if none is given.
- Per game, one coach `NeuralPlayer` per analyser on the human seat; call it exactly
  once at each human decision (even if the UI hides it) and attach
  `analysis = {coach: {label: {action: p}}, value: {...}}` to the `view` returned
  when it is the human's turn. Bot steps get `analysis.bot = {policy: full distribution,
  value}` (the full distribution, not only the played action's probability).
- Reveal mode (no bot-vs-bot mode: the human always plays, and challenges the model
  while "cheating"): `POST /api/new` takes `reveal: true`, and every `view` then carries
  the bot's hand and the bot-side analysis (its full policy, value, network views of
  its decisions). A toggle mid-game is fine too; it only changes what the page is sent.
  The bot's inputs are unchanged: it never learns that the human can see its hand. The
  human coaches keep their own levels (the L1 coach still does not see the bot's hand),
  so in reveal mode the L2 coach is the one using the same information as the human.
  Reveal mode is step-paced by default: before each bot decision the page stops and
  shows the bot's analysis for that decision (policy, value, network views), and the
  bot's move only plays when the human clicks "Continue" (hotkey: space). Each bot
  step (a skip/reverse chain is several) waits for its own click. Server side, this means
  the bot's analysis for a pending decision must be computable *before* its move is
  sampled: either the server returns the pending analysis and waits for
  `POST /api/advance`, or it computes each step's analysis at sampling time and the page
  just holds each step until the click (simpler; the move is already fixed but not
  shown). Prefer the second unless something needs the first. An "auto-advance" toggle
  can switch back to the normal timed pace.

Page:

- Coach chips on hand cards: a small ring or bar per coach sized by probability,
  one colour per coach, wilds grouped by card with the colour split in a tooltip.
  Draw and Keep get chips too. A "coach" toggle (hotkey `c`).
- After the human moves: a one-line verdict in the log when the move differs from
  the L1 coach's top choice, plus "L2 disagrees" markers where the two coaches differ.
- Policy bars for bot moves in a side panel (all legal actions, played one highlighted).
- Value curve: a small line chart under the log, one point per decision, x = turn.
  Hover a point to scroll the log. Load the `dataviz` skill before designing it.

## 5. Phase 2: card-grid network view (A and B)

Why this works: every token is a card type (`(colour, slot)` plus wilds, see
`uno/rl/env/layout.py` for `slots`, `slot_class` and the token order), weights are
shared per token, and mixing between tokens is only pooling. So each layer is a grid
shaped like the deck.

- `uno/play/inspect.py`: `trace(actor, features) -> dict` returning, per decision,
  the input features per token, the token states after the projection and after each
  relational block, `color_ctx`, `global_ctx`, and for C the memory before/after.
  Read `Encoder.forward_reference` (`uno/rl/model/relational.py:148`) and follow its
  residual structure rather than forward-hooking block modules (a block's output is
  a residual *update*). Check the trace against `actor(...)` logits to 1e-5.
- **Activations**: fit one PCA per layer over token states from a few hundred games
  (store the 3 components in `.cache/play/`), map each token to RGB, draw the grid:
  4 colour rows x slots, a wild column. Layer tabs: input / proj / block 1 / block 2
  (/ memory). The same PCA across games keeps colours comparable.
- **Attribution, exact part**: `Actor.scores` splits the first scorer layer into four
  blocks summed before one ReLU (own token, colour context, global context, static
  action features). With the ReLU pattern of that decision fixed, a logit is exactly
  `sum over active units of w2_i * (block contributions)_i + b2`, so the score of
  the played action vs the runner-up decomposes exactly into the four blocks. Show it
  as a small stacked bar: "own card / colour / global / action".
- **Attribution, token part**: gradient x input (or integrated gradients, 16 steps,
  from a zero or mean-token baseline) of `logit(played) - logit(runner-up)` w.r.t.
  each token's projected state, summed over width -> one signed number per card
  cell. Also offer occlusion (replace one token with the mean token) as a check; if
  the two disagree badly for a decision, show neither for it. Say "which cards moved
  the decision", never "why".
- Grid UI: a collapsible panel beside the table; cells use the card sprites at low
  opacity with a colour overlay; a cell's tooltip lists its raw feature channels
  (names from `uno/rl/env/features.py`).

## 6. Phase 3: C memory probe

Needs a trained C export (`configs/rl/rec_c.toml`).

- Data: play C vs the league on DEV seeds, record at each learner decision the
  per-token memory (`n_tokens x 64`) and, per token, the opponent's true count of
  that card type (0/1/2) and whether it is > 0.
- Probe: one shared logistic regression over tokens (equivariance makes one probe per
  token class valid; fit per `slot_class` if a shared one is clearly worse). Held-out
  DEV games; report AUROC and calibration against (a) the `voids` belief
  (`uno/eval/belief.py`) and (b) a probe on B's post-block token states at the same
  decisions, which says how much the memory adds. Compare with
  `uno/experiments/belief_ceiling.py`, which already fits sequence models on L1 tapes.
- UI: three grids side by side -- probe belief, `voids` baseline, true hand (reveal
  mode only) -- plus cells flashing by the norm of the memory update at each step, so an
  opponent draw on red visibly rewrites the red row.
- Optional corner widget: the memory trajectory over a game as a 2-D PCA path with
  event markers.
- Caveat in the UI and any write-up: a probe shows what is decodable from memory, not
  what the policy uses; phase 2's attribution on the memory channels is the partial
  answer.

## 7. Order, checks, and open questions

Order: phase 0 -> phase 1 (coach first, value curve after the calibration decision) ->
phase 2 on B -> phase 3 when C is trained. Each phase is usable alone.

Checks per phase: the coach's probabilities equal an independent `NeuralPlayer` fed
the same sequence (and a recurrent coach is not advanced by analysis calls); the trace
reproduces logits; the exact decomposition sums to the logit difference; the probe is
scored on held-out games only. Keep the page working with no analysers loaded. Report
commands actually run. Benji commits.

Open questions for Benji:

- Paths: the L2 ("critic-level") actor export, and the full checkpoint whose critic
  to use.
- Which C checkpoint to use once trained, and whether phase 3's probe comparison
  should become a `docs/rl/results/` entry (it bears on the A/B/C memory question).
- Whether reveal mode should also show the bot's *upcoming* policy on the human's turn
  (what it would do if the human passed the turn back) or only its actual decisions.
