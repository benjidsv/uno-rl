# UNO v5: working guide

## Working with Benji

- **Do not commit, push, or otherwise publish changes unless explicitly asked.**
  Give a summary of changes, verification, and unresolved issues; Benji commits.
- Discuss research direction and trade-offs before implementing a new algorithm or
  experiment. A request to review or brainstorm authorizes investigation, not a
  new research direction.
- Distinguish measured facts, hypotheses, bounds, and recommendations. Challenge
  conclusions when the experiment did not test them.
- Benji knows RL/PPO well and is newer to extensive-form game methods. Explain BR,
  CFR, information sets, and statistical implications in plain language.
- **Prefer batched/vectorized paths for engine, simulation, feature extraction,
  inference, and exact computations.** Batch state construction as well as network
  evaluation where practical; measure end-to-end throughput and memory.
- Preserve slow reference implementations as correctness oracles. Performance
  changes must preserve game semantics, information boundaries, and reproducibility;
  report changes to random streams or numerical behavior explicitly.

## Two documentation projects

Steps 1-10 are a completed **foundations project**: the engine, exact toys,
solvers, evaluators, search tools, baselines, and the evidence that validates or
limits them. Step 11 is the active **RL project**, which consumes those foundations.
Do not make every RL session reconstruct the foundations' history.

- `docs/rl/`: active RL status, plan, contracts, handoff, and learner results.
- `docs/foundations/`: authoritative technical specification, searchable rationale,
  Steps 1-10 results, and historical plans.
- `docs/generated/`: reproducible figures and small source tables, owned by their
  generators. Ignore it unless working on figures or their producer; do not hand-edit it.

Authority depends on the kind of claim:

| Claim | Authoritative source |
|---|---|
| intended engine or foundation behavior | `docs/foundations/spec.md` |
| learner inputs and representation boundaries | `docs/rl/observation-contract.md` |
| current RL direction and phase gates | `docs/rl/plan.md` |
| current RL status and next decision | `docs/rl/README.md` |
| current Steps 1-10 measurements | `docs/foundations/results/index.md` |
| current learner measurements | `docs/rl/results/index.md` |
| provenance and caveats for a measurement | its individual result file |
| historical reasoning or failed approaches | `docs/foundations/rationale.md` |

Results are not specifications, and old plans or brainstorms are not proof that an
algorithm was built or validated. If code and a contract disagree, report the
disagreement instead of silently copying the code.

## Start and route the work

Start every session with this guide, `git status --short`, and recent `git log`.
The active project is RL unless the task says otherwise.

For RL research, planning, training, or evaluation, read `docs/rl/README.md`, then
only the documents it routes to. For foundation maintenance, start with
`docs/foundations/README.md`.

Do **not** read `docs/foundations/spec.md`, `docs/foundations/rationale.md`, or a
complete results index by default. Search their headings and terms, then read the
smallest relevant section and any directly linked result:

```sh
rg -n '^#{1,4} ' docs/foundations/spec.md docs/foundations/rationale.md \
  docs/foundations/results/index.md
rg -n -i '<topic>' docs/foundations docs/rl
```

| Task | Read next |
|---|---|
| plan or review RL research | RL README, RL plan, relevant RL result |
| implement learner, features, or recurrent state | RL starting point, observation contract, relevant spec section |
| interpret a foundation number | matching foundation-index section, then linked result if needed |
| modify engine, solver, evaluator, or search | relevant spec section, module docstring, mirrored tests |
| understand why something exists | matching rationale section only |
| work on a historical experiment | its result file and the contract it cites |
| broad audit | expand scope deliberately; do not default to reading every document |

Read an experiment module's docstring for positional arguments. Do not launch
large trees, full sweeps, or `scripts/rerun_all.sh` merely to inspect the repo.

## Domain contracts that cross both projects

- One parameterized, deterministic, immutable engine; `step(state, action)` returns
  a new state. Wild plays are joint `(card, color)` actions. In two-player play,
  skip and reverse grant another turn. A playable drawn card may be kept, and
  `Pass` emits no public event.
- **L1** is own initial hand, public history, and own draws in order. L2 adds the
  opponent's initial hand/draws; L3 adds pile information. Actor inputs must obey
  their declared information level. Do not infer a void from an opponent draw
  without stating the policy assumption.
- **Score** is `(return + 1) / 2`, with draws worth 1/2. Report both seats and
  compare to the same-seat baseline. Toy seat advantages are large.
- BR is optimal against one fixed, known opponent; Nash is an equilibrium objective.
  A league-score learner need not minimize exploitability or attain all
  opponent-specific BRs simultaneously.
- Values of extra information are not learnable L1 headroom. PIMC is an achievable
  benchmark, **not an upper bound**.
- An attack's achieved two-seat score gives a NashConv **lower** bound. A small or
  negative bound is a weak attack, never evidence that a checkpoint is safe.

The complete rules and algorithm contracts live in the foundation spec. The
learner's three representation arms and actor/critic/attacker boundaries live in
the observation contract. K1 is an ablation, not an established representation
ceiling.

## Correctness and performance guardrails

- Agents return **action distributions**, not sampled actions. Probabilities must
  match the policy actually played; AIVAT relies on this. Give checkpoints distinct
  names and use an explicit shared stream for coupled comparisons.
- Exact quotient evaluation requires color/rank symmetry and uniform probabilities
  within stabilizer action orbits. Canonicalizing observations alone is insufficient.
  The learner should be equivariant; do not enumerate full UNO's enormous
  permutation group in a training loop.
- Preserve information boundaries in critics, training targets, search, and attacks.
  `determinize_l1` fabricates private history; recurrent or memoryful target policies
  require consistent reconstructed histories before they can be queried there.
- AIVAT evaluation may correct both players. The shipped training target corrects
  chance, opponent, and deal, **never the learner's own actions**. Do not turn an
  evaluation return into a training return silently. Existing gradient checks do
  not validate every future PPO/GAE choice.
- Belief and search settings are measured working points, not universal defaults.
  The `voids` belief assumes play-first behavior and immediate play of playable
  drawn cards. Recheck search strength when beliefs, leaves, targets, or opponents change.
- Judge variance reduction per wall time, not only per trajectory. Preserve terminal
  potential zero and avoid double correction.
- Stream summaries instead of retaining thousands of full-UNO traces. Measure
  simulation, feature construction, inference, and learner update throughput together.
- `.cache/` is ignored and expensive to rebuild. Bump `CACHE_VERSION` when changes
  to engine, chance, keys, symmetry, layout, or cached baseline behavior invalidate it.

## Experiments and evidence

- DEV seeds `[0, 1_000_000)` are for fitting, tuning, checkpoint selection, and
  pilots. Persistent confirmatory claims use a fresh registered block from the
  EVAL pool `[1_000_000, 2_000_000)` in `uno/eval/seeds.py`. Do not repeatedly
  select on an EVAL block.
- Pair comparisons on the same deals and opponent streams. Retain per-seat results,
  covariance where required, uncertainty, sample counts, units, and whether intervals
  are SEs or CIs. Mirrored self-play is degenerate under shared randomness.
- Exact toy values have no trajectory-sampling noise; learned policies still vary
  across training seeds. Full-UNO scores and LBR attacks are sampled.
- Validate on meaningful pinned toy values and reference solvers, not only tiny or
  singleton cases. Check that a test can detect the defect it targets.
- Keep Hypothesis/reference examples bounded. Put long measurements in experiments.
  Report commands actually run rather than historical suite counts.
- New RL result files belong under `docs/rl/results/` and begin with the question,
  predeclared decision rule, configuration and seeds, result, interpretation,
  limitations, and next decision. Exact values get one canonical home.
- Foundation result files are historical records. Correct them explicitly when
  necessary; do not rewrite them to tell a cleaner retrospective story.
- Keep `docs/rl/README.md` short by replacing completed active-state prose with a
  link to its result. Do not append a new project history to it.

## Architecture and commands

Python >=3.14, `uv`, NumPy, and PyTorch; `pyproject.toml` is the dependency source.

| Location | Responsibility |
|---|---|
| `uno/game/` | rules, immutable engine, chance, observations, keys, symmetry |
| `uno/agents/` | baselines, tabular adapters, symmetrization |
| `uno/reference/` | slow trusted exact walkers and sizing |
| `uno/compiled/` | array trees, exact evaluation/BR/CFR, gradients, diagnostics |
| `uno/eval/` | runners, estimators, seeds, search, beliefs, critics, targets, attacks |
| `uno/experiments/` | reproducible experiments dispatched by `uno/cli.py` |
| `uno/store.py` | versioned tree, strategy, and tabulation cache |
| `tests/` | mirrored package structure; pinned values in `tests/known.py` |

```sh
uv sync
uv run uno list
uv run uno report TOY heuristic
uv run pytest tests/<relevant-path>
uv run pytest
```

Handoffs summarize changed files, substantive findings, checks, and unresolved
decisions. Leave commits to Benji.
