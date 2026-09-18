# Step 10C: memory rulebots and frozen league (2026-09-18)

> On top of commit `6ffe6f5`. Reproduce the final figures and tables with
> `uv run uno rulebots 4000 docs/generated/rulebots 500 eval` (EVAL block 03).
> Reproduce the preceding selection diagnostic with
> `uv run uno rulebots 2000 <output_root> 500 dev`.
> Reproduce the bounded toy-only parameter selection with
> `uv run uno rulebots select docs/generated/rulebots`.
> Parameters and membership were frozen before EVAL block 03 was read.

## What was built

`uno/agents/rulebot.py` adds four cheap, symmetric, position-only policies. The
normative policy and metric contracts are in [spec section 33](../../spec.md).

1. `heuristic_memory`: the existing Heuristic, with public draw-derived void
   evidence used only to break an otherwise equal colour-support tie. With no
   live evidence its distribution is exactly the Heuristic's.
2. `rulebot_transfer_toy_v1`: the parameterized tactical scorer selected by an
   exact toy objective, without looking at full-UNO scores.
3. `rulebot_full_dev_v1`: a separately named full-UNO DEV selection. It is not
   presented as transfer evidence.
4. `rulebot_unpredictable_t0.15_m0.2_d0.02_k0.04_kw0.1`: the full-UNO scorer
   with a soft distribution over near-best plays, 2% voluntary draw, 4% keep of
   an ordinary playable draw, and 10% keep of a playable drawn wild. The agent
   reports the complete distribution; sampling remains in the game runner.

The memory is reconstructed from the current public history rather than stored
as mutable episode state. This makes all four valid position-only targets for
the existing `response` attack and preserves exact tabulation. The tactical
features cover hand colour support, rank/action-trait shedding, colour pivots,
public void targeting, follow-up plays, opponent hand pressure, action-card
penalties, and wild retention. `RuleWeights` exposes the parameters without
requiring a genetic algorithm.

The frozen training league contains those four policies at equal weight. Names,
order, and weights are asserted by `FROZEN_RULEBOT_LEAGUE` and
`FROZEN_RULEBOT_WEIGHTS`; changing a policy name without updating the frozen
declaration raises.

`uno/eval/behavior.py` streams traces into mergeable sufficient statistics. Its
conditional choice metrics use the probability mass reported by the policy,
not only its sampled action. The package measures voluntary draw, draw while
holding a wild, keep-drawn, keep-drawn-wild, plain-wild and +4 hoarding,
**+2 use and skip/reverse use separately**, sustainable action-card timing,
public-void targeting, normalized policy entropy, and determinism. It also
reports decisions, voluntary and forced draws, penalty cards received, and
terminal wilds per game. Common-state policy diversity is mean total-variation
distance on a bounded Heuristic-vs-Random probe bank, with represented decision
kinds weighted equally.

## Exact toy selection and transfer

The transfer selection objective was the unweighted mean of the row's exact L1
score against Random and Heuristic, at both seats, on `TOY` and `TOY2_SKIP`.
`TOY_FULL` is redundant for deterministic play-first candidates, so it is shown
in the output but was not counted twice during selection. A bounded coordinate
sweep around the hand-written tactical preset selected `denial=2.0` and
`singleton=0.0`:

| policy | exact selection objective |
|---|---:|
| initial tactical preset | 0.545569 |
| selected transfer preset | **0.547769** |
| Heuristic reference | 0.534791 |
| later full-UNO DEV preset, evaluated on the toys | 0.542452 |

The selected bot's exact scores are:

| ruleset | vs Random, seat 0 / 1 | vs Heuristic, seat 0 / 1 |
|---|---:|---:|
| `TOY` | 0.7414 / 0.4128 | 0.6749 / 0.3110 |
| `TOY_FULL` (display only) | 0.7414 / 0.4128 | 0.6749 / 0.3110 |
| `TOY2_SKIP` | 0.6783 / 0.5004 | 0.6182 / 0.4452 |

This looked negative on the 2,000-deal DEV diagnostic: its four-cell mean was
0.5444 versus the Heuristic's 0.5531, and its head-to-head mean was 0.4778.
That did not replicate on held-out block 03: its four-cell mean is 0.5502 versus
0.5506 for the Heuristic. The transfer result is therefore **null**, not a
successful generalization claim.

## Full-UNO DEV selection and diagnostic

The full-UNO variant came from a small shared-action-stream comparison on the
first 1,000 DEV deals. The useful change was *less* trust in inferred voids
(`denial=0.5`) plus a mild unique-trait preference (`singleton=0.5`); more
aggressive action-card and denial weights lost. Its apparent four-cell mean in
that selection was 0.5797 versus 0.5665 for the initial preset. Those are
selection numbers, not estimates of the true gain.

The generated diagnostic uses 2,000 DEV deals per cell, both seats, against
Random and the Heuristic. Each displayed cell has a raw SE of 0.0107-0.0112;
the CSV contains the cell-specific SE. The four-cell mean has no independent
inferential status because this range includes the selection deals.

| row agent | vs Random, seat 0 / 1 | vs Heuristic, seat 0 / 1 | four-cell mean | ms/game |
|---|---:|---:|---:|---:|
| Random | 0.5035 / 0.4965 | 0.4075 / 0.3800 | 0.4469 | 2.04 |
| Heuristic | 0.6200 / 0.5925 | 0.5185 / 0.4815 | 0.5531 | 1.70 |
| memory ablation | **0.6465 / 0.6105** | 0.5125 / 0.4940 | **0.5659** | 2.02 |
| toy-transfer bot | 0.6240 / 0.5980 | 0.4955 / 0.4600 | 0.5444 | 2.38 |
| full-UNO DEV bot | 0.6365 / 0.6135 | **0.5205 / 0.4885** | 0.5648 | 1.98 |
| unpredictable bot | 0.6165 / 0.5950 | 0.4985 / 0.4720 | 0.5455 | 2.11 |

The two memoryful candidates are essentially tied in this selection diagnostic:
the minimal memory ablation has the largest four-cell point estimate, while the
full scorer has the largest head-to-head estimate. That is evidence against
claiming that the larger scorer is already a proven improvement. Both remain
about 10-25 times cheaper than the 20-50 ms/game column-free PIMC rows measured
in 10a.2, although this table's timings include trace construction and behavioral
reduction and should only be compared directly within the table. The separate
8.11 s/game number belongs to an LBR repeatedly attacking a search target, not
to playing that PIMC row once.

## Frozen full-UNO evaluation

EVAL block 03 is `[1,012,000, 1,016,000)`: 4,000 deals per cell, both seats,
against Random and the Heuristic. Nothing was selected after reading it. Raw
cell SEs are 0.00762-0.00791.

| row agent | vs Random, seat 0 / 1 | vs Heuristic, seat 0 / 1 | four-cell mean | ms/game |
|---|---:|---:|---:|---:|
| Random | 0.5172 / 0.4828 | 0.4052 / 0.3922 | 0.4494 | 1.95 |
| Heuristic | 0.6078 / 0.5948 | 0.5185 / 0.4815 | 0.5506 | 1.63 |
| memory ablation | **0.6258 / 0.6078** | **0.5327 / 0.4875** | **0.5634** | 1.97 |
| toy-transfer bot | 0.6180 / 0.5990 | 0.5060 / 0.4778 | 0.5502 | 2.37 |
| full-UNO DEV bot | **0.6332 / 0.6032** | 0.5010 / 0.4893 | 0.5567 | 2.00 |
| unpredictable bot | 0.6162 / 0.5710 | 0.4860 / 0.4515 | 0.5312 | 2.14 |

Paired differences against the Heuristic on the same deals (points ± one SE;
`z = delta / SE`):

| row | vs Random s0 / s1 | vs Heuristic s0 / s1 |
|---|---:|---:|
| memory ablation | +1.80 ± 0.91 (`z=1.99`) / +1.30 ± 0.90 (`z=1.45`) | +1.43 ± 0.83 (`z=1.72`) / +0.60 ± 0.83 (`z=0.72`) |
| toy-transfer bot | +1.03 ± 1.02 (`z=1.01`) / +0.43 ± 1.02 (`z=0.42`) | -1.25 ± 0.98 (`z=-1.27`) / -0.38 ± 0.98 (`z=-0.38`) |
| full-UNO DEV bot | +2.55 ± 0.99 (`z=2.57`) / +0.85 ± 1.01 (`z=0.84`) | -1.75 ± 0.99 (`z=-1.77`) / +0.78 ± 0.97 (`z=0.80`) |
| unpredictable bot | +0.85 ± 1.00 (`z=0.85`) / -2.38 ± 1.03 (`z=-2.30`) | -3.25 ± 0.98 (`z=-3.31`) / -3.00 ± 1.00 (`z=-3.01`) |

The minimal memory ablation has the best held-out four-cell point estimate, but
no candidate establishes a positive improvement over the Heuristic at both
seats and against both columns. The larger DEV bot's only `|z| > 2` gain is
against Random at seat 0. Thus cheap public memory looks useful, but this does
**not** close most of the 10a.2 PIMC gap. The unpredictable bot pays a real
strength cost head-to-head, yet still beats the Random baseline by 9.9 / 8.8
points at the same seats and stays within 0.8 points of the Heuristic's mean
score against Random. It is a competent assumption-breaking league opponent,
not a replacement for the strongest deterministic member.

## Behavioral fingerprints

Opportunity-normalized rates from EVAL block 03:

| agent | voluntary draw | keep drawn / wild | hoard wild / +4 | use +2 | use skip/reverse | target void | entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Heuristic | 0.0% | 0.0% / 0.0% | 100.0% / 100.0% | 31.6% | 32.3% | 19.7% | 14.6% |
| memory ablation | 0.0% | 0.0% / 0.0% | 100.0% / 100.0% | 34.7% | 34.2% | 31.9% | 12.4% |
| toy-transfer bot | 0.0% | 0.0% / 0.0% | 99.6% / 11.2% | 93.6% | 79.3% | 47.8% | 12.6% |
| full-UNO DEV bot | 0.0% | 0.0% / 0.0% | 99.6% / 8.1% | 93.9% | 79.0% | 32.2% | 8.1% |
| unpredictable bot | **2.0%** | **5.3% / 10.0%** | 99.5% / 10.3% | 91.6% | 77.8% | 26.7% | **21.4%** |

Thus +2 and block/switch timing are observably different and are not collapsed
into one “action card” rate. The unpredictable bot remains close to Heuristic
strength against Random on held-out EVAL while actually violating both assumptions used
by the hard `voids` inference: it sometimes draws with a play available and
sometimes keeps a playable drawn card. Its common-state total-variation distance
from the deterministic full bot is 0.0447; the transfer/full distance is 0.0604,
and the Heuristic/memory-ablation distance is 0.0311.

## Generated artifacts and checks

- [`rulebot-toy-transfer.svg`](../../generated/rulebots/figures/rulebot-toy-transfer.svg): exact toy profiles.
- [`rulebot-strength.svg`](../../generated/rulebots/figures/rulebot-strength.svg): full-UNO EVAL strength and cost.
- [`rulebot-behavior.svg`](../../generated/rulebots/figures/rulebot-behavior.svg): conditional rates and per-game counts.
- [`rulebot-diversity.svg`](../../generated/rulebots/figures/rulebot-diversity.svg): common-state policy distance.
- `rulebot-toy-scores.csv`, `rulebot-scores.csv`, `rulebot-behavior.csv`, and
  `rulebot-policy-distance.csv`: generated source tables, including denominators
  and score SEs, under `docs/generated/rulebots/data/`.
- `rulebot-deltas.csv`: per-seat paired differences and SEs versus the Heuristic.
- `rulebot-transfer-selection.csv`: every candidate and all eight exact cells in
  the bounded toy-only coordinate sweep; the command asserts that the recorded
  `TRANSFER_WEIGHTS` are still its winner.

Focused verification: `pytest -q tests/agents/test_rulebot.py
tests/eval/test_behavior.py` -- 8 passed. Expanded nearby verification over
`tests/agents`, the behavior/belief/attack/seed evaluators, and game symmetry --
67 passed in 45.91 s. Tests cover symmetry/equivariance,
legal normalized distributions, exact draw/pass probabilities, memory identity
without evidence, use of live evidence, attack position-only support, separate
+2 and skip/reverse metrics, and policy-distance invariants.

## Limitations carried into Step 11

The league intentionally retains all four requested roles at equal weight even
though the transfer and full-UNO scorers do not beat the minimal ablation on the
held-out aggregate: they induce materially different action-card and +4 behavior,
and league diversity is not the same objective as selecting one champion. The
existing LBR attack accepts every member as a position-only target, but no new
full-UNO exploitability bound is claimed here. A later small bound would still
mean only that the attack was weak, never that a league member was safe.
