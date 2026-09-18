# Step 10a: the contract (2026-09-17)

> On top of commit `29bccad`. Plan: [plan.md](../../plan.md) row 10, brainstorm:
> [10-brainstorm.md](10-brainstorm.md) section 5.
> Re-run: `uv run uno report TOY heuristic`, `uv run uno report TOY random`,
> `uv run uno report TOY2 heuristic`, `uv run uno report TOY2_SKIP heuristic`; tests
> `uv run pytest tests/eval/test_seeds.py tests/compiled/test_strategies.py tests/game/test_observe_keys.py`.

Step 11 will produce checkpoints. This commit decides, **before any of them exist**, what
gets measured about one, on which seeds, and what counts as better. Three of the
brainstorm's six items: **A1** the seed registry, **A2** the batched tabulation, **A3**
the report card. A4 (shaping artifacts), A5 (the two missing ladder rows), A6 (the
training target) and LBR follow in a second pass.

Two of the three changed shape once measured, which is the point of measuring first.

## 1. A1: the seed registry, and the admission that `EVAL_SEEDS` is spent

`uno.eval.seeds` widens the pools (`DEV_POOL` `[0, 10^6)`, `EVAL_POOL` `[10^6, 2*10^6)`)
and partitions the eval pool into disjoint **blocks**, each claimed by one step for one
purpose, recorded in a table in the module rather than in whichever experiment runs
first. Overlap is checked at import; a name cannot be claimed twice
(`tests/eval/test_seeds.py`, 10 tests).

The load-bearing entry is **block 00, marked spent**: 1,000,000-1,003,999, the original
`EVAL_SEEDS`. Steps 9.5d-9.5f chose the full-UNO working set -- the `response`
continuation, w = 32, the `voids` belief -- on 3,000-4,000-deal runs over exactly those
seeds, so by this project's own rule ("nothing is ever chosen on them") they have been a
dev range since 9.5d and nothing noticed. 09.5f-results.md section 7.3 said so for
`window` at seat 1 and stopped there; the rule was broken for the whole working set.

`EVAL_SEEDS` itself is left **unchanged**, because every number steps 7-9.5c reported is
on it. It is kept in the registry as a spent block instead of retired, so a step-11
report card cannot silently reuse the seeds its own opponent was tuned on:
`seeds.block("00")` raises rather than returning them.

**Seeds were never the scarce resource** -- the engine is a pure function of a 64-bit
seed. The declaration was, because there was one range and no record of what had been
spent on what.

## 2. A2: the measurement that changed the item

A2 was scoped as "`observation_from_key` over all sets -> batched features -> one
forward pass -> orbit-average, for a network policy", i.e. batch the *policy* call. The
first thing built was the profile, and it says the scoping was half right.

**Where `tabulate` actually spends its time** (per information set, Heuristic):

| | `TOY` (12,036 sets) | `TOY2` (129,428 sets) |
|---|---|---|
| total | 14.5 us | 13.6 us |
| `observation_from_key` | **8.2 us (57 %)** | **8.5 us (62 %)** |
| `agent.policy` | 1.7 us (12 %) | 1.7 us (13 %) |
| `decode_key` | 1.3 us (9 %) | 1.4 us (11 %) |
| bookkeeping | 0.8 us (6 %) | 0.8 us (6 %) |

So with a *rulebot* the policy call is an eighth of the cost. With a *network* the
arithmetic flips -- a batch-of-1 forward is 69-92 us (below) -- but only until the
forward is batched, after which the observation rebuild is the dominant term again.
**A2 needs both halves**, and the half the brainstorm named would have left ~35 s of
pure Python on `TOY2_FULL`'s 3.75M sets.

Inside `observation_from_key` on `TOY2` (8.5 us): `_hand` **4.1 us**, `_public_summary`
2.3 us, the frozen `Observation.__init__` 2.1 us. `_hand` builds five `Counter` objects
per call -- 647,140 allocations for 129,428 sets -- to produce a hand tuple a batched
consumer immediately counts again.

### 2.1 `batch_l1_observations`

Keeps the one genuinely sequential part (the scan of the public history, mean 8.7
events) and drops the other two: hands come out as a `(sets, num_types)` multiplicity
matrix, the rest as parallel arrays, no `Observation` object.

| ruleset | `observation_from_key` | batched | |
|---|---|---|---|
| `TOY` | 8.51 us/set | 2.99 us/set | **2.85x** |
| `TOY2` | 9.07 us/set | 3.51 us/set | **2.59x** |
| `TOY_FULL` | 9.06 us/set | 3.06 us/set | **2.97x** |

**Array-equal on every field**, never `allclose` -- the bar
`uno.eval.value_net.batch_position_features` is held to, and the one a report card that
must reproduce its values *to the digit* needs.

*(A first draft of this file quoted 3.13 / 3.01 / 3.66x. Those were the scratchpad
prototype's, and the shipped function does more: it also rebuilds the discard pile per
set, worth 0.3-0.6 us/set. Kept, because every real L1 encoder needs the discard -- it is
what separates memory key K1 from K0, `context.md` 5.15 -- but the figures above are the
ones that ship. Measuring the prototype and shipping the function is how a speed-up
quietly becomes a smaller speed-up.)*

It is reusable: `uno.eval.decisions.set_labels` (`uno.store.load_or_label`) walks the
same sets and pays the same ~9 us today, and needs no discard, so it would go faster still.

### 2.2 The forward pass, and why batching it is not optional

The project's own value-network shape, float64 on CPU, one thread, per row:

| batch | `TOY` | `TOY2` | `FULL_UNO` |
|---|---|---|---|
| 1 | 91.8 us | 69.0 us | 68.1 us |
| 32 | 4.5 us | 7.1 us | 4.6 us |
| 256 | **2.7 us** | **2.4 us** | **3.0 us** |
| 4096 | 3.3 us | 2.6 us | 3.1 us |
| 8192 | 5.0 us | 3.6 us | 4.5 us |

A **25-30x** gap between batch 1 and batch 256, at every ruleset width. This is step
5.21's finding one level up: there it was a search's leaves, here a report card's
information sets. Past ~4096 rows it turns back up, so the chunk is a real knob, not
"as big as possible".

### 2.3 `tabulate_batched`

`uno.compiled.strategies.tabulate_batched(tree, level, policy, chunk=65_536)`, written
against a `BatchPolicy` protocol whose `policy_batch` receives an `L1Batch` and returns
one probability per set-action in the keying's own `ia` layout. Working in `ia` order
rather than per-set dicts is what keeps the path vectorised end to end: the result
scatters into `sigma` with no Python loop and `Keying.orbit_mean` symmetrises it in one
`bincount`.

Against the row-at-a-time `tabulate`, **array-equal** in every case:

| | rulebot policy | network policy (`TOY`) |
|---|---|---|
| `TOY` | 12.5 -> 4.9 us/set (2.6x) | **54-59 -> 5.5 us/set (~10x)** |
| `TOY2` | 11.6 -> 6.2 us/set (1.9x) | |
| `TOY_FULL` | 11.5 -> 5.7 us/set (2.0x) | |

The network figure is the one the item exists for, and it repairs the brainstorm's hole
(b): tabulating a *network* over `TOY2_FULL`'s 3.75M sets projects at **~200 s -> ~21 s**.
(The row-at-a-time side is noisy -- two runs gave 53.9 and 58.6 us/set, 9.8x and 10.7x --
because it is 12,036 separate framework calls; the batched side reproduced 5.5 us/set
exactly. Quoted as ~10x rather than to two figures it does not have.)
(The brainstorm's "~50 s on `TOY2_FULL`" was the *rulebot* cost; a network at batch 1 is
four times that.) `context.md` 6.9's "the exact side of a checkpoint is milliseconds to
seconds" is now true for a network, where the brainstorm correctly flagged it as false.

Two caveats worth keeping:

- The network paths agree to **2.22e-16, one ulp, not bit for bit** -- BLAS picks its
  kernel by row count, exactly as `NetLeaf`'s docstring records. The invariant that does
  hold is the one batching needs: at a fixed row count a row's value depends on that row
  alone. Far below any margin that flips an argmax.
- **Symmetry is nearly free here, and that is a property of the quotient, not of this
  code.** On a quotient tree every set's key is already canonical, so there is no
  `canonical` call to make -- only the orbit averaging, which is one `bincount`. This
  weakens the brainstorm's claim that "A2's batched tabulation is written against
  whichever [encoding] it is": the pipeline is the same shape either way.

### 2.4 An unplanned finding: a plain encoder cannot be tabulated at all

The first network stub was a plain MLP over raw features, and `tabulate` **refused it**:
not orbit-uniform. That is the section 5.8 bug class working as designed, and it is
sharper evidence for the encoding decision than the brainstorm's argument was. Option B
(plain encoder + `Symmetrized`) does not merely *pay* orbit averaging at every forward
pass -- without `Symmetrized` in the path it cannot produce an exact cell on the
quotient at all, because `tabulate` will not accept the policy. Pinned as a test
(`test_batched_tabulation_refuses_an_asymmetric_policy`).

**Decision recorded** (brainstorm open question 1): **option A**, equivariant by
construction, with `Symmetrized` kept as its test oracle -- a correctly equivariant
network must be a fixed point of it. The network itself is step 11's object; 10a builds
only the seam it plugs into.

## 3. A3: the report card

`uv run uno report [ruleset=TOY] [agent=heuristic|random|nash] [iterations=1000]`.
One command, one checkpoint, fixed format: score against each column per seat,
**normalised skill**, `br_gap` (the D4 fixed-opponent metric), **exploitability** and
NashConv, and edge by decision. Two numbers carry it -- position on the range against
the training column, and exploitability; the rest is diagnosis.

```
== report card: heuristic on TOY
exact
   vs random     0.7416 / 0.4147   normalised 0.64 / 0.32   (BR L1 0.7934 / 0.5513)  ok
                 br_gap vs this column                      5.18 / 13.66 pts
   vs heuristic  0.6934 / 0.3066   normalised 0.71 / 0.27   (BR L1 0.7378 / 0.4361)  ok
   exploitability                              5.57 / 11.82 pts   NashConv 17.39  ok
   edge by decision vs random     wild 35 %, color 34 %, drawn 28 %, other 2 %
== 0 failed checks
```

**The checks.** Every printed number for the two pinned agents must reproduce
[index.md](../index.md) to the digit -- the 9.5f device. Heuristic on `TOY`: 0.7416 /
0.4147, 0.6934 / 0.3066, exploitability 5.6 / 11.8. Random: 0.6493 / 0.3507,
exploitability 17.1 / 17.4. **0 failed checks** on both.

Three independent cross-checks the card was not pinned against, and passes anyway:

- `br_gap` of the Heuristic against Random, **5.18 / 13.66 pts**, against `context.md`
  6.2's independently computed "left on the table by the Heuristic: 5.2 / 13.7";
- `br_gap` of Random against Random, **14.41 / 20.06**, against that table's BR L1 edge
  over Random of "+14.4 / +20.1";
- Random normalises to exactly **0.00 / 0.00**, which it must, being the 0 of the scale.

**A correction to the brainstorm.** Section 3 predicted the Heuristic would land at
"~0.4 from seat 0 and ~0.2 from seat 1" on the normalised scale. Measured, with 0 =
Random and 1 = BR L1 at the same seat against the same column, it is **0.64 / 0.32** vs
Random and **0.71 / 0.27** vs the Heuristic on `TOY`. The 2:1 seat ratio the brainstorm
was pointing at is real and is the robust part; the level is not, and the figure was
quoted before anything computed it.

Normalised skill of the Heuristic across the toys, seat 0 / seat 1:

| ruleset | vs Random | vs Heuristic |
|---|---|---|
| `TOY` | 0.64 / 0.32 | 0.71 / 0.27 |
| `TOY2` | 0.60 / 0.22 | 0.60 / 0.21 |
| `TOY2_SKIP` | **0.36 / 0.37** | **0.33 / 0.37** |

The scale reproduces a qualitative finding it was not built for, which is the best
evidence that it is measuring something. On `TOY` and `TOY2` the rulebot recovers about
twice as much of the available skill from seat 0 as from seat 1 -- `context.md` 6.2's
"decent from seat 0 and weak from seat 1". On `TOY2_SKIP` that asymmetry **vanishes**,
and the level drops at both seats: the same toy where 6.4 records the Heuristic's
self-play seat gap collapsing from 27.7 to 15.3 and its exploitability rising to
11.0 / 11.9 because "its skip rule is a guess". A rule that is wrong at both seats costs
seat 0 what it already cost seat 1.

The sampled block of the brainstorm's sketch is deliberately absent: on a toy every cell
is exact and no deals are spent. A full-UNO card claims a fresh block from A1's registry
rather than re-spending block 00.

### 3.1 The card's first run against a new ruleset found a published number wrong

`report TOY2 heuristic` failed its own check: seat-1 exploitability **11.7480**, against
the **11.8** on [index.md](../index.md) and in `context.md` 6.4. The card is right and the
published figure is a **double rounding**:

    published:  BR L1 0.6959    - Nash 0.5784    = 0.117500 -> 11.8
    computed:   BR L1 0.69587054 - Nash 0.57839081 = 0.1174797 -> 11.7

It was obtained by differencing two values that had already been rounded to four
decimals. The current Nash strategy's residual is 0.000245 pts, two orders of magnitude
below the 0.002 margin, so the result is not solver wobble. Every other pinned cell
displays correctly (`TOY2` random 14.775 / 15.209, `TOY2_SKIP` 11.03 / 11.93, `TOY`
5.57 / 11.82 and 17.09 / 17.38): this is the only one that lands on a boundary.

`index.md` and `context.md` 6.4 are corrected to 11.7;
[08.7](../03-08-foundations/08.7-toy2.md) keeps its number as measured then, per that page's own convention.

**This is the instrument working.** A3 was specified as "run it on `Heuristic` and
`PlayFirstRandom` and require the pinned values" -- the 9.5f reproduces-to-the-digit
device -- and the first time it was pointed at a ruleset it had not been developed
against, it disagreed with the documentation and was right.

Two method notes, both mistakes made during this step and worth keeping:

- **The first diagnosis was wrong.** The disagreement was written off as the checker's
  banker's rounding, on the theory that the value was exactly 11.75 -- inferred from a
  *two-decimal* print reading "11.75". That is the same double-rounding error as the bug,
  committed one level up while investigating it. Printing four decimals settled it in one
  run. A number read off a rounded display is not evidence about a rounding boundary.
- **The comparator was still worth fixing**, but not for the reason first given.
  `round(value, dp) == published` is the wrong test -- the published figure is itself a
  rounded display and `round` is banker's rounding -- so `reproduces` compares against the
  half-width of the last published digit. It is deliberately tight: it must still fail on
  a genuine disagreement, which is exactly what it did here.

## 4. What is not in this commit

A4 (shaping artifacts), A5 (column-free PIMC and the belief rungs' exploitability), A6
(the training target and its exact-gradient check) and LBR -- kept in 10a per the
scoping decision, landing in the second pass.

## 5. Checks

| what | result |
|---|---|
| `tests/eval/test_seeds.py` | 10 passed |
| `tests/compiled/test_strategies.py` | 8 passed (2 pre-existing + 6) |
| `tests/game/test_observe_keys.py` | 5 passed (1 pre-existing + 4) |
| full suite | **269 passed** (249 before this step), 6 min 54 s and 8 min 01 s on two runs -- suite time is noisy on a loaded machine, as 9.5e and 9.5f both recorded |
| `report TOY heuristic` / `TOY random` / `TOY2 heuristic` / `TOY2_SKIP heuristic` | **0 failed checks each**, after correcting the published `TOY2` cell above |
| batched observations vs row-at-a-time | array-equal, every field, 3 rulesets |
| `tabulate_batched` vs `tabulate` | array-equal, 3 rulesets, 5 chunk sizes |
