# UNO RL v5

Measurement first, learning second: an own UNO engine, a small toy game solved
exactly, and evaluators that can tell skill from luck, before any RL.

- `RULES.md`: the spec (engine, solvers, estimators, compiled tree, CFR). If
  the code disagrees with it, the code is wrong.
- `notes.md`: the plan, the build order and every step's results tables.

## Setup

```sh
uv sync            # Python 3.14, numpy; dev: pytest, hypothesis
```

## Commands

| what | command | rough time on `TOY` (not re-timed) |
|------|---------|---------------|
| all tests | `uv run pytest` | ~2.5 min |
| one test file | `uv run pytest tests/test_cfr.py -q` | |
| step 3: tree-size census of the toy candidates | `uv run python scripts/census.py [node_cap=2e6] [games=2000] [letters e.g. S,O]` | long: some candidates hit the node cap |
| steps 5-6: exact best responses L1/L2/L3 (pure-Python reference solvers) | `uv run python scripts/ladder.py [ruleset=TOY] [games=4000]` | minutes: 12 solves at 4-16 s each |
| step 7: mirrored evaluator vs exact values | `uv run python scripts/mirror.py [ruleset=TOY] [games=4000]` | minutes (reference BR solves first) |
| step 8: compiled tree, CFR+ Nash, exploitability, Nash row/column | `uv run python scripts/cfr.py [ruleset=TOY] [iterations=1000] [games=4000]` | 1000 CFR+ iterations = 203 s on the first run, then CFR is loaded from `.cache/cfr_TOY.npz` |
| step 8.5: transfer check, `TOY` against `FULL_UNO` (sampled) plus exact endings on `TOY` | `uv run python scripts/transfer.py [games=4000] [cfr_iterations=200]` | ~1.5 min (FULL_UNO pairs 40 s); uses the CFR cache if present |
| step 8.6: symmetric (quotient) trees, `TOY` full vs quotient, Nash on the quotient, `TOY_FULL` solved exactly | `uv run python scripts/symmetry.py [cfr_iterations=1000]` | ~5 min the first time (full `TOY` trees compile in 14 s each, `TOY_FULL` CFR+), seconds afterwards from `.cache/` |

Scripts report on the held-out `EVAL_SEEDS` (1,000,000+); build and test on
`DEV_SEEDS` (0-3999).

## Code map

| module | what |
|--------|------|
| `uno/rules.py` | `Ruleset`, card types, actions, named rulesets `TOY`, `FULL_UNO`, ... |
| `uno/engine.py` | immutable state, `initial_state`, `legal_actions`, `step`, `returns` |
| `uno/observe.py` | L1 / L2 / L3 observations and information-state keys |
| `uno/agents.py` | `UniformLegal`, `PlayFirstRandom` (the "Random" row), `Heuristic`, `BySeat` |
| `uno/play.py` | `play_game`, `match` (common random numbers), seed ranges |
| `uno/tree.py` | lazy-chance game tree, `census` |
| `uno/solve.py` | reference exact solvers: `evaluate`, `best_response` (slow, trusted) |
| `uno/estimate.py` | `play_pairs`, `estimate_mirrored`, `compare`, `exact_mirrored` |
| `uno/symmetry.py` | color / rank symmetry group, canonical keys, orbit representatives (RULES.md section 18) |
| `uno/compiled.py` | the tree as numpy arrays (quotient under the symmetry by default, cached in `.cache/`): fast `evaluate`, `best_response`, `exploitability`, `tabulate` |
| `uno/cfr.py` | CFR+ / vanilla CFR on the compiled tree |

## Quick examples

Sampled match (score of the first agent, seat 0):

```python
from uno import TOY
from uno.agents import Heuristic, PlayFirstRandom
from uno.play import match

print(match(TOY, (Heuristic(TOY), PlayFirstRandom(TOY)), range(1000)))
```

Exact exploitability of an agent (~2 s on the quotient tree; `symmetric=False` gives the full tree, ~15 s):

```python
from uno import TOY
from uno.agents import Heuristic
from uno.compiled import compile_tree, exploitability, tabulate
from uno.observe import Level

tree = compile_tree(TOY)
sigma = tabulate(tree, Level.L1, Heuristic(TOY))
print(exploitability(tree, Level.L1, sigma))
```
