# UNO RL v5

Two-player UNO research in two phases: Steps 1-10 built and validated the engine,
exact toys, evaluators, search tools, and baselines; the active project uses those
foundations to train and evaluate learned policies.

- [`docs/rl/`](docs/rl/README.md): current RL status, research plan, learner
  contracts, foundation handoff, and learner results.
- [`docs/foundations/`](docs/foundations/README.md): searchable Steps 1-10
  specification, rationale, measurements, and historical plans.
- [`docs/`](docs/README.md): the complete documentation map.

## Setup

```sh
uv sync            # Python 3.14, numpy; dev: pytest, hypothesis; installs the `uno` command
```

## Commands

Every experiment is a module of `uno/experiments/` run through the CLI. Trees and
CFR results are cached in `.cache/` (git-ignored, `uno/store.py`).

```sh
uv run uno list                 # experiment names and arguments
uv run pytest tests/<area>      # relevant checks while developing
uv run pytest                   # complete suite
```

`bash scripts/rerun_all.sh` reruns the multi-hour foundation suite and is not a
routine check. Seed-pool ownership and claimed EVAL blocks live in
`uno/eval/seeds.py`.

## Code map

| module | what |
|--------|------|
| `uno/game/rules.py` | `Ruleset`, card types, actions, named rulesets (`NAMED`: `TOY`, `TOY_FULL`, `TOY2`, `TOY2_FULL`, `TOY2_SKIP`, `FULL_UNO`, ...) |
| `uno/game/engine.py` | immutable state, `initial_state`, `legal_actions`, `step`, `returns` |
| `uno/game/observe.py` | L1 / L2 / L3 observations, `observation_from_key` |
| `uno/game/keys.py` | information-set keys per level and their byte encoding |
| `uno/game/chance.py` | the chance model: lazy-chance children, deals, worlds, pile decomposition |
| `uno/game/symmetry.py` | color / rank symmetry group, canonical keys, orbit representatives ([spec section 18](docs/foundations/spec.md)) |
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
