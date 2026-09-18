# Step 10B: reproducible figures (2026-09-18)

> On top of commit `9a30e80`. Rerun:
> `uv run uno figures [output_root=docs/generated/report-card]`.
> New code: `uno/experiments/figures.py`. New dependency: `matplotlib` (Agg/SVG backend
> only). Design: [`docs/foundations/results/10-readiness/10-brainstorm.md`](10-brainstorm.md) section 6,
> [`docs/foundations/history/transition-to-rl-plan.md`](../../history/transition-to-rl-plan.md).

## What this is for

`docs/foundations/results/index.md`'s numbers have been hand-copied from each step's stdout into
markdown since step 3. The brainstorm named the risk directly: "staleness has bitten
before" (review item B6), and step 11 was expected to produce rows faster than
hand-editing survives. `uv run uno figures` is the first script that turns already-published
numbers into a picture rather than a new measurement, and it recomputes the exact toy
numbers it draws **live from `.cache/`** rather than copying them from `index.md`, checking
each one against the published digit with `uno.experiments.report.reproduces` -- the same
helper and convention the report card already uses. This run: **0 failed checks**.

## Two scope cuts, named rather than silently pinned

1. **`TOY2_FULL`'s BR L2 / BR L3 columns are not live here.** They need an L1+L3 keyed
   tree that was never cached -- `uno.experiments.toy2`'s `light` mode exists precisely
   because that tree does not fit memory with three keyings at once on this ruleset (28M
   quotient histories). Recompiling it is out of scope for this step, per 10a.3's own
   "no engine rewrite, bounded" instruction. These two cells come from `BR_PUBLISHED`
   (the same dict that is the live check's target for every other cell) instead of a
   fresh `best_response` call. Everything else `TOY2_FULL` needs for figure 1 --
   BR L1, the matchup scores, the Nash value -- only needs the L1 tree, which is cached
   and cheap, and *is* live-recomputed and checked.
2. **`index.md`'s hand-written tables are not spliced over by this command.** Its
   tables interleave prose, footnotes and provenance notes with the numbers throughout
   (e.g. the TOY2 11.7/11.8 correction note, or the TOY_FULL == TOY aside on the deal
   floor), so generating replacement markdown for the whole file is real doc surgery,
   not "generate the tables." The digit-check above is the concrete form of the
   brainstorm's actual regression concern -- staleness -- without committing to a full
   markdown-generation/replacement pipeline in this step. A future step can still do the
   splice; this one builds and proves the check it would need to be safe.

## The five figures

All in `docs/generated/report-card/figures/`, written as SVG (matplotlib, Agg
backend). "Live" means
recomputed from `.cache/` by this run and checked to the digit; "historical" means a
sampled full-UNO number that cannot be cheaply re-derived here and is read from the
`HISTORICAL` dict in `uno/experiments/figures.py`, each entry commented with its source.

1. **`range.svg`** -- Random / Heuristic / Nash / BR L1(random), per ruleset and seat,
   against the fixed Random column, with the achievable range (0 = Random, 1 = BR
   L1(random)) as a grey bar underneath. All five rulesets, fully live. Reading it: Nash
   is not trained against Random and both can and does fall outside the Random-Heuristic
   range at seat 0 for several toys (it isn't optimizing this column at all), which is
   the visual case for why "beats Random" and "is not exploitable" are different claims
   (the project's running theme, see `index.md`'s exploitability table).
2. **`values-of-information.svg`** -- the L2-L1 (opponent hand) and L3-L2 (pile order)
   best-response gaps, min-max range and mean over {random, heuristic} x {seat 0, seat 1},
   per ruleset. Live for TOY, TOY_FULL, TOY2, TOY2_SKIP; `TOY2_FULL` is drawn hollow
   (scope cut 1). Reproduces the brainstorm's own summary of this figure ("2.5-4.4 points
   on TOY to 7-14 on TOY2_FULL") as the visible growth of the range across rulesets.
3. **`budget.svg`** -- AIVAT variance ratio (raw = 1.0) by estimator, TOY next to
   FULL_UNO, seat 0 / seat 1, with the one known deal-count reduction (3,860 -> 1,776
   deals at seat 0, 3,956 -> 1,854 at seat 1, fitted network, for a 3-point claim)
   annotated. Entirely historical (09.5b/c, 10a2): a literal "deals for 3 points" count
   was never computed for the toys (toy AIVAT results are reported as variance ratios,
   not deal counts, since the toys are used to validate the estimator, not to budget an
   experiment), so this is the variance-ratio proxy for the brainstorm's "budget" figure,
   not a rerun of it, and the results file says so rather than fabricating toy deal
   counts that were never measured.
4. **`edge-by-decision.svg`** -- the Heuristic's gap to BR L1(random) on TOY, live, by
   `uno.eval.decisions.node_kind` (wild 35%, color 34%, drawn 28%, other/single 2%/0%),
   next to the full-UNO D2 finding (action-card timing ~70%, wild handling ~27%),
   historical. These are two different taxonomies over two different reference points
   (BR L1 vs. a w=32 search) -- the figure places them side by side rather than
   stacking them against shared categories, which would misrepresent them as the same
   breakdown.
5. **`lever-ladder.svg`** -- the full-UNO lever waterfall: continuation (+9.73/+4.57),
   belief (+2.83/+4.40), worlds 8->32 at d=2 (+5.57/+3.33), depth (~0, "every difference
   within 1.5 sigma of zero"). Entirely historical (09.5e, 09.5f section 7.2) -- this is
   specifically about full-UNO PIMC behaviour and has no toy analogue.

## The check

`uno/experiments/figures.py` recomputes, per ruleset, from cached trees and CFR results
(`uno.experiments.common.columns` / `columns_l1`, `uno.store.load_or_tabulate`,
`uno.store.load_or_solve_nash`): the BR L1/L2/L3 columns against Random and the
Heuristic, the three self-play/cross matchup scores, and the Nash seat-0 score. Every
one that has a published pin (`uno/experiments/figures.py`'s own `BR_PUBLISHED` and
`NASH_SEAT0`, plus `uno.experiments.report.PINNED` for the matchup scores where it has
an entry) is checked with `reproduces(value, published, decimals)` at 4 decimals for
scores. `report.PINNED` only covers TOY, TOY2 and TOY2_SKIP's matchup scores (it was
built for the report card, which has only ever been run against those three); TOY_FULL
and TOY2_FULL's matchup rungs are live but unchecked at that precision for want of a
published pin, and are said to be so rather than presented as verified. This run: 5
rulesets gathered, **0 failed checks**.

## What is deliberately not built here

- A `uv run uno tables` command or any markdown-generation pipeline (scope cut 2).
- Re-sampling any full-UNO number: every `HISTORICAL` entry is read, not re-measured,
  which is the whole point of not re-spending an EVAL block to draw a picture.
- A `TOY2_FULL` L1+L3 tree (scope cut 1). If a future step needs it live, `toy2.py`'s
  `light` mode and its own module docstring already say why it doesn't fit today.
