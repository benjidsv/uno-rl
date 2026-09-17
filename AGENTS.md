# UNO v5: working guide

## Working with Benji

- **Do not commit, push, or otherwise publish changes unless explicitly asked.**
  Give a summary of changes, verification, and unresolved issues; Benji commits.
- Discuss research direction and trade-offs before implementing a new step.
  A request to review or brainstorm authorizes investigation, not a new algorithm.
- **Prefer batched/vectorized paths for engine, simulation, feature extraction,
  inference, and exact computations.** Expect large slowdowns from Python loops
  and per-state/per-network calls. Batch state construction as well as network
  evaluation where practical; measure end-to-end throughput and memory.
- Preserve slow reference implementations as correctness oracles. Performance
  changes must preserve game semantics, information boundaries, and reproducibility;
  explicitly report changes to random streams or numerical behavior.
- Benji knows RL/PPO well and is newer to extensive-form game methods. Explain
  BR, CFR, information sets, and statistical implications in plain language.
- Distinguish measured facts, hypotheses, bounds, and recommendations. Challenge
  conclusions when the experiment did not actually test them.

## Purpose and sources

Fifth iteration of two-player UNO RL. Earlier PPO attempts did not clearly beat
rulebots (~60% against Random; a historical tuned bot ~70%). The project first
built a trustworthy engine, exact toy solutions, and low-variance evaluators to
separate learning failure from noise and unknown headroom. **Measurement first,
learning second.** Toys validate methods; numerical conclusions need separate
validation on full UNO.

Start with this guide, `git status --short`, and recent `git log`. For a research
session read `docs/context.md` and `docs/results/index.md`, then relevant code.

- `docs/spec.md`: rules and algorithm contracts; intended behavior wins over code.
  Report a suspected problem in the specification instead of silently copying it.
- `docs/results/index.md`: current headline measurements; wins over context when
  numerical summaries disagree. Read later corrections alongside older tables.
- `docs/context.md`: long-form rationale, findings, failed approaches, practical
  notes. Some opening/status sections lag its later additions.
- `docs/plan.md`: build order and status. Earlier prose includes superseded plans.
- `docs/10-to-11-plan.md`: current execution plan and exit criteria: insert 10a.3
  before 10B, then retain 10B → 10C → Step 11's three phases.
- `docs/results/<step>.md`: historical experiment, command, provenance, caveats.
- `docs/results/10-review.md`: September 2026 second opinion, including a reproduced
  K1 measurability problem. Recommendations there are proposals, not implemented work.

Keep these roles separate: results are not specifications, and old brainstorms
are not proof that an algorithm has been built or validated.

## Current position (2026-09-17, reviewed at `f12ffaf`)

- Steps 1–9.5f: engine, exact solvers, symmetry, CFR, memory/edge diagnostics,
  AIVAT, PIMC, value networks, and belief experiments built.
- 10a and 10a.2: seed registry, batched tabulation, report card, gradient/target
  validation, shaping artifacts, belief exploitability, and column-free PIMC built.
- LBR was deferred in 10a.2; the revised plan puts its baseline instrument in
  10a.3 alongside the memory-evidence repair. No implementation is complete yet.
  10b figures/generated tables and 10c distilled rulebot follow in the same order.
- No RL learner yet. Benji's intended Step 11: (1) exact toy measurements to
  validate learning and approach appropriate benchmarks, (2) sampled full-UNO
  training/evaluation, (3) further RL iterations if substantial headroom remains.
- Reopen the full-UNO observation/memory choice. **K1 sufficiency is not established.**
  Keep K1 as an ablation, not a settled representation ceiling. LBR and stronger
  memory/inference are planned comparisons. Full-UNO learner comparisons and
  recurrent-target attack support belong to Step 11; a new particle filter is
  not required before 10B or 10C.

## Game and measurement vocabulary

- One parameterized, deterministic, immutable engine; `step(state, action)` returns
  a new state. Full UNO: 108 cards, hand 7, seeded reshuffle, 1000-action draw cap.
- Wild plays are joint `(card, color)` actions. No stacking/challenge; +4 is
  always playable. In two-player play, skip and reverse both grant another turn.
- A playable drawn card can be kept. `Pass` emits no public event. Never infer
  that an opponent's draw proves a void without stating the policy assumption.
- Toys have no reshuffle: required draw on an empty pile ends by hand-size
  comparison. Exhaustion/pile parity distort strategic conclusions.
- `TOY`: 9 cards, forced play; `TOY_FULL`: voluntary draw allowed.
  `TOY2`: 10 cards/three ranks; `TOY2_SKIP`: action-card toy;
  `TOY2_FULL`: large free-play toy (~28M quotient histories).
- **L1**: own initial hand, public history, own draws in order (perfect recall).
  **L2** adds opponent initial hand/draws; **L3** adds pile information.
  Actor inputs must respect the declared information level. L1/L2 observations
  are used at the observer's own decision nodes to avoid timing leaks.
- **Score** = `(return + 1) / 2`, draws worth 1/2. Report both seats; seat 0
  moves first and toy seat advantages are large. Compare to the same-seat baseline.
- **BR L1** is optimal against one fixed, known opponent. **Nash** is an
  equilibrium objective. A learner maximizing league score need not minimize
  exploitability or attain every opponent-specific BR simultaneously.
- Exploitability per row seat = opponent BR score minus that responder seat's
  Nash score. NashConv is the sum across seats, in score points here.
- L2−L1 and L3−L2 are values of extra information, not noise decompositions or
  learnable L1 headroom. PIMC is an achievable benchmark, **not an upper bound**.

## Architecture and commands

Python >=3.14, `uv`, NumPy, PyTorch; dependency/config source is `pyproject.toml`.

| Location | Responsibility |
|---|---|
| `uno/game/` | Rules, immutable engine, chance, observations, keys, symmetry |
| `uno/agents/` | Baselines, tabular adapters, symmetrization |
| `uno/reference/` | Slow trusted exact walkers and sizing |
| `uno/compiled/` | Array trees, exact evaluation/BR/CFR, coarse keys, gradients, edge attribution |
| `uno/eval/` | Game runners, estimators, seeds, search, beliefs, critics, targets, shaping |
| `uno/experiments/` | Reproducible step experiments, dispatched by `uno/cli.py` |
| `uno/store.py` | Versioned tree, strategy, and tabulation cache |
| `tests/` | Mirrored package structure; pinned values in `tests/known.py` |

```sh
uv sync
uv run uno list
uv run uno report TOY heuristic
uv run pytest tests/<relevant-path>
uv run pytest
```

Read an experiment's module docstring for positional arguments. The existing
`.venv/bin/uno` and `.venv/bin/python` also run the installed environment.
`bash scripts/rerun_all.sh` is an hours-long experiment suite, not a routine check.
Do not launch large trees or full sweeps merely to inspect the repo.

## Correctness and performance guardrails

- Agents return **action distributions**, not sampled actions. Probabilities must
  match the actual policy; AIVAT relies on this. Names identify policies. Give
  checkpoints distinct names; use explicit shared `stream` for coupled comparisons.
- Exact quotient evaluation requires color/rank symmetry and uniform probabilities
  within stabilizer action orbits. Canonicalizing observations alone is insufficient.
  An equivariant policy is the proposed learner design; `Symmetrized` is a toy oracle.
  Do not enumerate full UNO's enormous permutation group in a training loop.
- **Coarse-key warning:** K0 includes a public-history clock; K1 adds discard
  counts; current K2 adds only draw colors since the opponent's last play.
  K2 is not the later `voids` belief. `memoryless_response` is an iterative search,
  not a certified optimal imperfect-recall solver. Current quotient coarsening
  can retain history through its coordinate choice; a reproduced policy is not
  a function of raw K1 alone. See the review before relying on memory ceilings.
- `voids` assumes play-first behavior and immediate play of playable drawn cards.
  `soft(0.1)` relaxes that assumption; its toy robustness is not universal calibration.
  The failed deal-time particle sampler does not disprove all particle filters.
- Current PIMC working budget: 32 worlds, depth 2. `response` uses the known
  target policy; fixed-Heuristic continuation is a measured column-free player.
  Search settings and strength must be rechecked when beliefs/leaves/opponents change.
- `determinize_l1` fabricates opponent private history. Existing rulebots tolerate
  this; memoryful/recurrent opponent models need consistent reconstructed histories.
- AIVAT's chance value model uses L2, not real pile order. Evaluation can correct
  both players; the shipped training target corrects chance + opponent + deal,
  **never the learner's own actions**. Do not silently use evaluation returns as
  training returns. The existing gradient check does not validate all PPO/GAE choices.
- Deal shaping/correction can be cheap; full sibling corrections were expensive
  even after batched inference. Judge variance reduction per wall time, not just
  per trajectory. Preserve terminal potential zero and avoid double correction.
- Keep computations vectorized and stream summaries. Thousands of full-UNO traces
  cost GB; do not retain every matchup's traces concurrently.
- `.cache/` is ignored and expensive to rebuild. `TOY2_FULL` files are ~2–3 GB
  each with multi-GB RAM usage. Bump `CACHE_VERSION` when engine/chance/key/symmetry/
  layout changes affect caches, or cached baseline policy behavior changes.

## Experiments, validation, and handoff

- DEV pool: `[0, 1_000_000)` for fitting/tuning/selection. EVAL pool:
  `[1_000_000, 2_000_000)`, persistent claims in `uno/eval/seeds.py`.
  Block 00 is spent on selection; block 01 belongs to 10a.2. Claim fresh blocks
  for new confirmatory experiments; do not repeatedly select checkpoints on a test set.
- Pair comparisons on the same deals and opponent streams; retain per-seat
  results and uncertainty. Mirrored self-play is degenerate under shared randomness.
  Report power, units, sample counts, and whether intervals are SEs or CIs.
- Exact toy values contain no trajectory sampling noise; training seeds still
  produce variation across learned policies. Full-UNO LBR is sampled too.
- Validate against reference solvers and meaningful pinned toy values, not only
  tiny/singleton cases. Check that a test can detect the defect it targets.
- Keep Hypothesis/reference examples bounded; put long measurements in experiments.
  Run relevant checks and report what was actually run, not historical suite counts.
- Results belong in `docs/results/<step>.md` with base commit, rerun command,
  configuration, seed provenance, checks, and limitations; update the results index,
  spec, plan, and context when the corresponding facts change. Preserve historical
  results and explicitly mark superseded interpretations.
- Handoff: summarize changed files, substantive findings, checks, and next decisions.
  Leave commits to Benji.
