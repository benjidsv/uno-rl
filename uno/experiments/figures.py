"""Step 10B: reproducible report-card figures
(`docs/foundations/results/10-readiness/10-brainstorm.md` section 6).

    uv run uno figures [output_root=docs/generated/report-card]

Five SVGs, one function each, written with matplotlib (Agg backend, no display
needed). Every number that is cheap to recompute exactly (the toy best-response
columns, matchups and Nash values) is recomputed live from `.cache/` -- not
copied from `index.md` -- and checked against the published digit with
`uno.experiments.report.reproduces`, the same helper and convention the report
card already uses. A trailing `== N failed checks` line means either the code
moved a published number or `index.md` is stale.

Two things are **not** recomputed here, named rather than silently pinned:

- `TOY2_FULL`'s BR L2 / BR L3 columns. They need an L1+L3 keyed tree that was
  never cached (`uno.experiments.toy2`'s `light` mode exists precisely because
  that tree does not fit memory with three keyings at once); recompiling it is
  out of scope for this step, per 10a.3's own "no engine rewrite" bound. These
  two cells come from `BR_PUBLISHED` instead of a live `best_response` call.
- Every full-UNO number (the budget picture, the lever ladder, and the D2 share
  of figure 4's right-hand bar). These are sampled results from real full-UNO
  deals (09.5b/c, 09.5e, 09.5f, 10a2) -- reproducing them here would mean
  re-spending an EVAL block, not drawing a picture of one. They live in
  `HISTORICAL`, each entry commented with its source results file, and are
  never digit-checked (there is nothing cheap to check them against).

`index.md`'s hand-written tables are not spliced over by this command: its
tables interleave prose and footnotes with the numbers throughout, so
generating replacement markdown for the whole file is out of scope here (see
`docs/foundations/results/10-readiness/10b-results.md`). The check above is the regression instrument
the brainstorm's "staleness has bitten before" concern actually asked for.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from uno import store
from uno.agents.baseline import Heuristic, PlayFirstRandom
from uno.compiled.edge import mirror_terms
from uno.compiled.solvers import best_response, evaluate
from uno.compiled.tree import compile_tree
from uno.experiments.common import columns, score
from uno.experiments.report import PINNED, reproduces
from uno.experiments.toy2 import columns_l1
from uno.game import rules
from uno.game.observe import Level

L1, L2, L3 = Level.L1, Level.L2, Level.L3
FAILS: list[str] = []

RULESETS = ("TOY", "TOY_FULL", "TOY2", "TOY2_SKIP", "TOY2_FULL")
NASH_CONFIG = {"TOY": (1000, "cfr+"), "TOY_FULL": (1000, "cfr+"), "TOY2": (1000, "cfr+"), "TOY2_SKIP": (1000, "cfr+"), "TOY2_FULL": (150, "dcfr")}

# Nash seat-0 score, docs/foundations/results/index.md "Nash value and seat gaps".
NASH_SEAT0 = {"TOY": 0.6196, "TOY_FULL": 0.5866, "TOY2": 0.5784, "TOY2_SKIP": 0.5748, "TOY2_FULL": 0.5564}

# (ruleset, opponent, seat) -> (BR L1, BR L2, BR L3) score, docs/foundations/results/index.md
# "Best-response columns". Doubles as the digit-check target for every cell this script
# recomputes live, and as the only source for TOY2_FULL's L2/L3 (never live here, see above).
BR_PUBLISHED: dict[tuple[str, str, int], tuple[float, float, float]] = {
    ("TOY", "random", 0): (0.7934, 0.8188, 0.8280), ("TOY", "random", 1): (0.5513, 0.5937, 0.5989),
    ("TOY", "heuristic", 0): (0.7378, 0.7730, 0.7798), ("TOY", "heuristic", 1): (0.4361, 0.4774, 0.4786),
    ("TOY_FULL", "random", 0): (0.8017, 0.8443, 0.8590), ("TOY_FULL", "random", 1): (0.5882, 0.6419, 0.6941),
    ("TOY_FULL", "heuristic", 0): (0.7446, 0.8060, 0.8187), ("TOY_FULL", "heuristic", 1): (0.4732, 0.5131, 0.5437),
    ("TOY2", "random", 0): (0.7305, 0.7976, 0.8191), ("TOY2", "random", 1): (0.5694, 0.6586, 0.6804),
    ("TOY2", "heuristic", 0): (0.6959, 0.7728, 0.7907), ("TOY2", "heuristic", 1): (0.4937, 0.5421, 0.5587),
    ("TOY2_SKIP", "random", 0): (0.7425, 0.7865, 0.8063), ("TOY2_SKIP", "random", 1): (0.5919, 0.6277, 0.6503),
    ("TOY2_SKIP", "heuristic", 0): (0.6941, 0.7538, 0.7738), ("TOY2_SKIP", "heuristic", 1): (0.5356, 0.5638, 0.5816),
    ("TOY2_FULL", "random", 0): (0.7479, 0.8443, 0.8875), ("TOY2_FULL", "random", 1): (0.6164, 0.7508, 0.8208),
    ("TOY2_FULL", "heuristic", 0): (0.7118, 0.8248, 0.8620), ("TOY2_FULL", "heuristic", 1): (0.5371, 0.6080, 0.6600),
}  # fmt: skip

# Sampled full-UNO results that are not cheaply reproducible here (source file in each comment).
HISTORICAL = {
    # variance ratio (raw = 1.0), heuristic vs random, seat 0 / seat 1 (docs/foundations/results/index.md "AIVAT").
    "variance_ratio": {
        ("TOY", "fitted critic, no deal"): (0.73, 0.88),
        ("TOY", "fitted network, no deal"): (0.45, 0.51),
        ("TOY", "fitted network, + deal"): (0.04, 0.06),
        ("FULL_UNO", "fitted critic, no deal"): (0.71, 0.76),
        ("FULL_UNO", "fitted network, no deal"): (0.46, 0.47),
        ("FULL_UNO", "fitted network, + deal"): (0.38, 0.39),
    },
    # deals for a 3-point claim at 3 sigma / 80% power, raw -> AIVAT fitted network (09.5c),
    # heuristic vs random (docs/foundations/results/index.md "AIVAT").
    "deals_3pt": {"seat 0": (3860, 1776), "seat 1": (3956, 1854)},
    # the full-UNO lever ladder, points at both seats (docs/foundations/results/09-evaluation/09.5e-results.md, 09.5f section 7.2).
    "levers": [
        ("continuation\n(response vs minimax)", 9.73, 4.57),
        ("belief\n(voids vs prior)", 2.83, 4.40),
        ("worlds\n(8 -> 32, d=2)", 5.57, 3.33),
        ("depth\n(2 -> 3, w=32)", 0.0, 0.0),  # "every difference within 1.5 sigma of zero" (09.5e)
    ],
    # D2 (docs/foundations/results/index.md "D2 with a search Q"): ~97% of the Heuristic's disagreement with a
    # w=32 search falls in these two node-kind-like buckets (a different taxonomy from
    # uno.eval.decisions.NODE_KINDS, so not stacked against the live TOY bar's own categories).
    "d2_shares": {"action-card timing": (0.68, 0.73), "wild handling": (0.25, 0.29)},
}


def check(ok: bool, what: str) -> str:
    if not ok:
        FAILS.append(what)
    return "ok" if ok else "XX"


def gather(name: str) -> dict:
    """Live BR L1(/L2/L3), matchup and Nash data for one ruleset, checked against the
    published digit wherever a pin exists (`BR_PUBLISHED`, `NASH_SEAT0`, `report.PINNED`)."""
    rs = getattr(rules, name)
    light = name == "TOY2_FULL"  # see the module docstring
    lazy = compile_tree(rs, (L1,) if light else (L1, L2), True, cache=store.CACHE)
    if light:
        cols = columns_l1(lazy, name)
    else:
        world = compile_tree(rs, (L1, L3), True, cache=store.CACHE)
        cols = columns(lazy, world, name)

    br = {}
    for opp in ("random", "heuristic"):
        for seat in (0, 1):
            got = tuple(r.score for r in cols[(opp, seat)])
            pub = BR_PUBLISHED[(name, opp, seat)]
            if light:
                got = (got[0], pub[1], pub[2])
            else:
                check(all(reproduces(got[i], pub[i], 4) for i in range(3)), f"BR L1/L2/L3 {opp} seat {seat} on {name} ({got} vs published {pub})")
            br[(opp, seat)] = got

    rnd_s = (L1, store.load_or_tabulate(lazy, L1, PlayFirstRandom(rs)))
    heu_s = (L1, store.load_or_tabulate(lazy, L1, Heuristic(rs)))
    m = {}
    for label, a, b in (("random", rnd_s, rnd_s), ("heuristic_vs_random", heu_s, rnd_s), ("heuristic", heu_s, heu_s)):
        m[label] = (score(evaluate(lazy, a, b)[0]), score(evaluate(lazy, b, a)[1]))
    pin_r, pin_h = PINNED.get((name, "random"), {}), PINNED.get((name, "heuristic"), {})
    if "vs random" in pin_r:
        check(all(reproduces(m["random"][i], pin_r["vs random"][i], 4) for i in (0, 1)), f"random vs random on {name}")
    if "vs random" in pin_h:
        check(all(reproduces(m["heuristic_vs_random"][i], pin_h["vs random"][i], 4) for i in (0, 1)), f"heuristic vs random on {name}")
    if "vs heuristic" in pin_h:
        check(all(reproduces(m["heuristic"][i], pin_h["vs heuristic"][i], 4) for i in (0, 1)), f"heuristic vs heuristic on {name}")

    iterations, variant = NASH_CONFIG[name]
    res = store.load_or_solve_nash(lazy, iterations, variant)
    nash_s = (L1, res.avg)
    nash0 = score(evaluate(lazy, nash_s, nash_s)[0])
    check(reproduces(nash0, NASH_SEAT0[name], 4), f"Nash seat-0 score on {name}")

    return dict(lazy=lazy, cols=cols, br=br, m=m, nash=(nash0, 1 - nash0))


def fig_range(data: dict, path: Path) -> None:
    """1. The range: Random / Heuristic / Nash / BR L1(random), per ruleset and seat,
    against the fixed 'random' column. The light grey segment is the achievable range
    (0 = Random, 1 = BR L1(random)); a rung outside it is not achievable against this
    column at all (Nash is not trained against Random, so it can and does miss)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    colors = {"Random": "#888888", "Heuristic": "#1f77b4", "Nash": "#d62728", "BR L1(random)": "#2ca02c"}
    for seat, ax in enumerate(axes):
        for row, name in enumerate(RULESETS):
            d = data[name]
            base = d["m"]["random"][seat]
            ceil = d["br"][("random", seat)][0]
            heu = d["m"]["heuristic_vs_random"][seat]
            nash = d["nash"][seat]
            ax.plot([base, ceil], [row, row], color="#cccccc", lw=6, solid_capstyle="round", zorder=1)
            for label, x in (("Random", base), ("BR L1(random)", ceil), ("Heuristic", heu), ("Nash", nash)):
                ax.scatter([x], [row], color=colors[label], s=60, zorder=2, label=label if row == 0 else None)
            for label, x, dy in (("0", base, -0.28), ("1", ceil, -0.28)):
                ax.annotate(label, (x, row + dy), ha="center", fontsize=7, color="#999999")
        ax.set_title(f"seat {seat}")
        ax.set_xlabel("score")
        ax.set_xlim(0, 1)
    axes[0].set_yticks(range(len(RULESETS)))
    axes[0].set_yticklabels(RULESETS)
    axes[1].legend(loc="lower right", fontsize=8, framealpha=0.9)
    fig.suptitle("The range: where each agent sits against Random, per ruleset and seat")
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)


def fig_values_of_information(data: dict, path: Path) -> None:
    """2. Values of information: the L2-L1 (opponent hand) and L3-L2 (pile order) gaps,
    min-max range and mean over {random, heuristic} x {seat 0, seat 1}, per ruleset.
    TOY2_FULL's range is drawn hollow: those two levels are not live-recomputed here."""
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(RULESETS))
    width = 0.32
    for offset, (gap_label, color) in zip((-1, 1), (("L2 - L1 (opponent hand)", "#1f77b4"), ("L3 - L2 (pile order)", "#ff7f0e"))):
        lo, hi, mean = [], [], []
        for name in RULESETS:
            d = data[name]
            vals = []
            for opp in ("random", "heuristic"):
                for seat in (0, 1):
                    l1, l2, l3 = d["br"][(opp, seat)]
                    vals.append(100 * (l2 - l1) if gap_label.startswith("L2") else 100 * (l3 - l2))
            lo.append(min(vals)), hi.append(max(vals)), mean.append(sum(vals) / len(vals))
        xs = x + offset * width / 2
        hollow = [name == "TOY2_FULL" for name in RULESETS]
        for xi, l, h, mn, hol in zip(xs, lo, hi, mean, hollow):
            ax.plot([xi, xi], [l, h], color=color, lw=3, alpha=0.4 if hol else 1.0)
            ax.scatter([xi], [mn], color=color, s=50, facecolors="none" if hol else color, label=gap_label if xi == xs[0] else None, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(RULESETS, rotation=20, ha="right")
    ax.set_ylabel("score points")
    ax.set_title("Values of information: hidden information grows with the toy\n(hollow = TOY2_FULL, not live-recomputed; docs/foundations/results/index.md)")
    ax.legend(fontsize=8)
    ax.axhline(0, color="#dddddd", lw=1, zorder=0)
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)


def fig_budget(path: Path) -> None:
    """3. The budget picture: AIVAT variance ratio (raw = 1.0) by estimator, toys next to
    full UNO, seat 0 / seat 1. Lower is better -- less of the estimator is luck. Where the
    deal count for a 3-point claim is known (fitted network, docs/foundations/results/index.md 'AIVAT'),
    it is annotated on the bar. All values: HISTORICAL, sampled full-UNO/toy results."""
    ratios = HISTORICAL["variance_ratio"]
    cats = ["fitted critic, no deal", "fitted network, no deal", "fitted network, + deal"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    y = np.arange(len(cats))
    h = 0.18
    for i, ruleset in enumerate(("TOY", "FULL_UNO")):
        for j, seat in enumerate((0, 1)):
            vals = [ratios[(ruleset, c)][seat] for c in cats]
            yy = y + (i * 2 + j - 1.5) * h
            ax.barh(yy, vals, height=h, color="#1f77b4" if ruleset == "TOY" else "#d62728", alpha=0.55 if seat == 0 else 0.95, label=f"{ruleset} seat {seat}")
    ax.axvline(1.0, color="#333333", lw=1, ls="--", label="raw (ratio = 1)")
    deals = HISTORICAL["deals_3pt"]
    ax.annotate(f"3,860 -> {deals['seat 0'][1]:,} deals (seat 0)\n3,956 -> {deals['seat 1'][1]:,} deals (seat 1)\nfor a 3-point claim, FULL_UNO", xy=(0.46, 1.0), xytext=(0.55, 2.05), fontsize=7, color="#d62728", arrowprops=dict(arrowstyle="->", color="#d62728", lw=0.8))
    ax.set_yticks(y)
    ax.set_yticklabels(cats)
    ax.set_xlabel("variance ratio, AIVAT / raw (lower = fewer deals needed)")
    ax.set_title("The budget picture: heuristic vs random,\ntoy vs full UNO (docs/foundations/results/index.md 'AIVAT')")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)


def fig_edge_by_decision(data: dict, path: Path) -> None:
    """4. Edge by decision: the Heuristic's gap to BR L1(random), live on TOY, by
    `uno.eval.decisions.node_kind`; next to the full-UNO D2 finding (docs/foundations/results/index.md
    'D2 with a search Q'), a different taxonomy (action-card timing / wild handling), so the
    two bars are shown side by side rather than stacked against the same categories."""
    tree = data["TOY"]["lazy"]
    rs = tree.rs
    heu = (L1, store.load_or_tabulate(tree, L1, Heuristic(rs)))
    rnd = (L1, store.load_or_tabulate(tree, L1, PlayFirstRandom(rs)))
    labels = store.load_or_label(tree)
    shares = []
    for seat in (0, 1):
        ref = (L1, best_response(tree, L1, rnd, seat).sigma)
        terms = mirror_terms(tree, heu, ref, rnd, seat, labels)
        by_kind: dict[str, float] = {}
        for kind, val in zip(terms.kind, terms.term):
            if kind:
                by_kind[kind] = by_kind.get(kind, 0.0) + float(val)
        total = sum(by_kind.values())
        shares.append({k: v / total for k, v in by_kind.items()} if abs(total) > 1e-12 else {})
    toy_merged = {k: (shares[0].get(k, 0.0) + shares[1].get(k, 0.0)) / 2 for k in set(shares[0]) | set(shares[1])}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, (title, merged) in zip(axes, (("TOY: Heuristic vs BR L1(random)\n(uno.eval.decisions.node_kind)", toy_merged), ("FULL_UNO: Heuristic vs a w=32 search\n(D2's own categories, not node_kind)", {k: sum(v) / 2 for k, v in HISTORICAL["d2_shares"].items()}))):
        items = sorted(merged.items(), key=lambda kv: -kv[1])
        left = 0.0
        for k, v in items:
            ax.barh([0], [v], left=[left], label=f"{k} ({100 * v:.0f}%)")
            left += v
        ax.set_xlim(0, max(1.0, left))
        ax.set_yticks([])
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2)
    fig.suptitle("Edge by decision: where the gap to a stronger responder is spent")
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)


def fig_lever_ladder(path: Path) -> None:
    """5. The full-UNO lever ladder: where 9.5's points came from (docs/foundations/results/09-evaluation/09.5e-results.md,
    09.5f section 7.2), seat 0 / seat 1. HISTORICAL -- specifically about full-UNO PIMC."""
    levers = HISTORICAL["levers"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    y = np.arange(len(levers))
    h = 0.32
    ax.barh(y + h / 2, [v[1] for v in levers], height=h, color="#1f77b4", label="seat 0")
    ax.barh(y - h / 2, [v[2] for v in levers], height=h, color="#ff7f0e", label="seat 1")
    for yi, (_, s0, s1) in zip(y, levers):
        ax.annotate(f"+{s0:.2f}" if s0 else "~0", (s0, yi + h / 2), va="center", fontsize=8, xytext=(4, 0), textcoords="offset points")
        ax.annotate(f"+{s1:.2f}" if s1 else "~0", (s1, yi - h / 2), va="center", fontsize=8, xytext=(4, 0), textcoords="offset points")
    ax.set_yticks(y)
    ax.set_yticklabels([v[0] for v in levers])
    ax.set_xlabel("score points, vs Random, w=32 d=2 unless noted")
    ax.set_title("The full-UNO lever ladder")
    ax.legend(fontsize=8)
    ax.axvline(0, color="#dddddd", lw=1)
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    output_root = Path(argv[0]) if argv else Path("docs/generated/report-card")
    figure_dir = output_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    print("== gathering exact toy data (cached trees/CFR; a fresh solve only if the cache is stale)")
    data = {}
    for name in RULESETS:
        data[name] = gather(name)
        print(f"   {name} done")
        sys.stdout.flush()

    print(f"\n== writing figures to {figure_dir}")
    fig_range(data, figure_dir / "range.svg")
    fig_values_of_information(data, figure_dir / "values-of-information.svg")
    fig_budget(figure_dir / "budget.svg")
    fig_edge_by_decision(data, figure_dir / "edge-by-decision.svg")
    fig_lever_ladder(figure_dir / "lever-ladder.svg")
    for name in ("range", "values-of-information", "budget", "edge-by-decision", "lever-ladder"):
        print(f"   {figure_dir / (name + '.svg')}")

    print(f"\n== {len(FAILS)} failed checks" + (": " + "; ".join(FAILS) if FAILS else ""))


if __name__ == "__main__":
    main()
