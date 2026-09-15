# UNO RL v5

Measurement first, learning second: an own UNO engine, a small toy game solved
exactly, and evaluators that can tell skill from luck, before any RL.

- [`docs/plan.md`](docs/plan.md): motivation, information levels, evaluators, the
  ladder, and the build order with its status.
- [`docs/spec.md`](docs/spec.md): the spec (engine, solvers, estimators, compiled
  tree, CFR, symmetry, sizing). If the code disagrees with it, the code is wrong.
- [`docs/results/`](docs/results/index.md): one file per step, and `index.md`
  with the current headline numbers of every toy.

## Setup

```sh
uv sync            # Python 3.14, numpy; dev: pytest, hypothesis; installs the `uno` command
```

## Commands

Every experiment is a module of `uno/experiments/` run through the CLI;
`uv run uno list` shows them. Trees and CFR results are cached in `.cache/`
(git-ignored, `uno/store.py`).

| what | command | was |
|------|---------|-----|
| all tests (102, ~4 min) | `uv run pytest` | |
| every experiment below, in dependency order, one log per step in `logs/` (hours) | `caffeinate -is bash scripts/rerun_all.sh` | |
| step 3: tree-size census of the toy candidates | `uv run uno census [node_cap=2e6] [games=2000] [letters e.g. S,O]` | `scripts/census.py` |
| steps 5-6: exact best responses L1/L2/L3 (pure-Python reference solvers) | `uv run uno ladder [ruleset=TOY] [games=4000]` | `scripts/ladder.py` |
| step 7: mirrored evaluator vs exact values | `uv run uno mirror [ruleset=TOY] [games=4000]` | `scripts/mirror.py` |
| step 8: compiled tree, CFR+ Nash, exploitability, Nash row/column | `uv run uno cfr [ruleset=TOY] [iterations=1000] [games=4000]` | `scripts/cfr.py` |
| step 8.5: transfer check, `TOY` against `FULL_UNO` | `uv run uno transfer [games=4000] [cfr_iterations=200]` | `scripts/transfer.py` |
| step 8.6: quotient trees, `TOY_FULL` solved (recompiles the `TOY` trees on purpose) | `uv run uno symmetry [cfr_iterations=1000]` | `scripts/symmetry.py` |
| step 8.7: exact sizes of the candidates (position DP) | `uv run uno sizing [position_cap=4000000]` | `scripts/sizing.py` |
| step 8.7: cost of one checkpoint of the RL loop | `uv run uno loop_cost` | `scripts/loop_cost.py` |
| step 8.7: CFR variants raced | `uv run uno cfr_variants [iterations=300] [rulesets=TOY,TOY_FULL,TOY2]` | `scripts/cfr_variants.py` |
| step 8.7: a TOY2-family ruleset solved | `uv run uno toy2 [ruleset=TOY2] [cfr_iterations=1000] [light] [variant=cfr+]` | `scripts/toy2.py` |
| step 8.7: L2 / L3 columns of `TOY2_FULL`, one keying per tree | `uv run uno toy2_columns [ruleset=TOY2_FULL]` | `scripts/toy2_columns.py` |
| step 8.7: validation checks from the compiled trees | `uv run uno validate [ruleset=TOY] [games=4000] [light]` | `scripts/validate.py` |

Experiments report on the held-out `EVAL_SEEDS` (1,000,000+); build and test on
`DEV_SEEDS` (0-3999).

## Code map

| module | what |
|--------|------|
| `uno/game/rules.py` | `Ruleset`, card types, actions, named rulesets (`NAMED`: `TOY`, `TOY_FULL`, `TOY2`, `TOY2_FULL`, `TOY2_SKIP`, `FULL_UNO`, ...) |
| `uno/game/engine.py` | immutable state, `initial_state`, `legal_actions`, `step`, `returns` |
| `uno/game/observe.py` | L1 / L2 / L3 observations, `observation_from_key` |
| `uno/game/keys.py` | information-set keys per level and their byte encoding |
| `uno/game/chance.py` | the chance model: lazy-chance children, deals, worlds, pile decomposition |
| `uno/game/symmetry.py` | color / rank symmetry group, canonical keys, orbit representatives ([spec section 18](docs/spec.md)) |
| `uno/agents/baseline.py` | `UniformLegal`, `PlayFirstRandom` (the "Random" row), `Heuristic`, `BySeat` |
| `uno/agents/tabular.py` | `TabularAgent`, `PureStrategy`, `CompiledAgent`: agents that play a table |
| `uno/agents/symmetrized.py` | `Symmetrized`: any agent made symmetric, so it can be evaluated on the quotient tree |
| `uno/reference/` | pure-Python reference walkers (slow, trusted): `census`, `solve` (evaluate, best response), `mirrored` (exact moments), `sizing` (position DP) |
| `uno/compiled/tree.py` | the tree as numpy arrays (quotient under the symmetry by default), keyings, `compile_tree` |
| `uno/compiled/solvers.py` | fast `evaluate`, `best_response`, `best_response_perfect`, `exploitability`, `exact_mirrored` |
| `uno/compiled/strategies.py` | `tabulate`, `tabulate_table`, `br_row` (a best-response row with a fill-in for unreached sets) |
| `uno/compiled/cfr.py` | CFR+ / vanilla / DCFR / PCFR+ on the compiled tree |
| `uno/eval/play.py` | `play_game`, `match` (common random numbers), seed ranges |
| `uno/eval/estimate.py` | `play_pairs`, `estimate_mirrored`, `compare`, statistics |
| `uno/store.py` | the cache: `load_or_compile`, `load_nash`, `load_or_solve_nash` |
| `uno/experiments/` | one module per step experiment, `common.py` for shared helpers; `uno/cli.py` runs them |

Tests mirror the packages (`tests/game/`, `tests/reference/`, ...); shared
hypothesis strategies are in `tests/strategies.py`, pinned values in `tests/known.py`.

## Quick examples

Sampled match (score of the first agent, seat 0):

```python
from uno import TOY
from uno.agents.baseline import Heuristic, PlayFirstRandom
from uno.eval.play import match

print(match(TOY, (Heuristic(TOY), PlayFirstRandom(TOY)), range(1000)))
```

Exact exploitability of an agent (~2 s on the quotient tree; `symmetric=False` gives the full tree, ~15 s):

```python
from uno import TOY, Level
from uno.agents.baseline import Heuristic
from uno.compiled.solvers import exploitability
from uno.compiled.strategies import tabulate
from uno.store import load_or_compile

tree = load_or_compile(TOY)
sigma = tabulate(tree, Level.L1, Heuristic(TOY))
print(exploitability(tree, Level.L1, sigma))
```
