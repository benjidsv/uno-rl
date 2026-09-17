"""Step 10a: the report card -- one command, one checkpoint, a fixed format (item A3).

    uv run uno report [ruleset=TOY] [agent=heuristic|random|nash] [iterations=1000]

Step 11 will produce checkpoints. This decides, **before any of them exist**, what gets
measured about one, on which seeds, and what counts as better -- the project's whole
thesis being that iterations 1-4 failed because that was never fixed in advance. The
failure mode to avoid is a step that draws pictures of numbers we already have while
the thing step 11 actually trips on goes unbuilt; so this is the thing that takes a
checkpoint and runs every instrument on it.

**The exact block** (toys only, and free: no deals are spent):

- score against each column, per seat;
- **normalised skill**, 0 = Random at that seat against that column, 1 = BR L1 against
  it. An affine rescaling per cell, so a difference in scale units is *not* score
  points: both are printed, always, and nothing is ever *selected* on the normalised
  number (docs/results/10-brainstorm.md section 3);
- `br_gap` against each column (`uno.compiled.edge`): the D4 fixed-opponent metric,
  the row's gap to the best response to its own training column;
- **exploitability** per seat and NashConv: the one luck-free quality measure, and the
  only thing that can tell "learned to exploit the league" from "learned to play well"
  -- the characteristic failure of step 11's league design, which score against the
  league cannot see;
- **edge by decision**: where the gap to BR L1 is spent, by `node_kind`.

Two numbers carry the card -- position on the range against the training column, and
exploitability. Everything else is diagnosis.

**The checks.** Run on the two agents whose values are pinned (`random`, `heuristic`),
every printed number must reproduce the published one *to the digit*
(`docs/results/index.md`), which is the step 9.5f device: a report card that agreed
only to a tolerance would not catch a rewiring that moved a number by a point. The
sampled block is deliberately absent here: on a toy every cell is exact, and a
full-UNO card claims a fresh seed block from `uno.eval.seeds` (item A1) rather than
re-spending the range 9.5d-9.5f selected on.
"""

from __future__ import annotations

import sys

import numpy as np

from uno import store
from uno.agents.baseline import Heuristic, PlayFirstRandom
from uno.compiled.edge import br_gap, mirror_terms
from uno.compiled.solvers import best_response, evaluate, exploitability
from uno.compiled.tree import compile_tree
from uno.eval import seeds as seed_registry
from uno.game import rules
from uno.game.observe import Level

L1 = Level.L1
FAILS: list[str] = []

# Published values every card must reproduce to the digit (docs/results/index.md).
# (ruleset, row): {"vs <column>": (seat 0, seat 1), "exploitability": (seat 0, seat 1)}
PINNED: dict[tuple[str, str], dict[str, tuple[float, float]]] = {
    ("TOY", "random"): {"vs random": (0.6493, 0.3507), "exploitability": (17.1, 17.4)},
    ("TOY", "heuristic"): {"vs random": (0.7416, 0.4147), "vs heuristic": (0.6934, 0.3066), "exploitability": (5.6, 11.8)},
    ("TOY2", "random"): {"vs random": (0.5875, 0.4125), "exploitability": (14.8, 15.2)},
    # seat 1 is 11.7, not the 11.8 published until step 10a: that figure was the difference of
    # two *rounded* values (0.6959 - 0.5784 = 0.1175 exactly), where the underlying doubles give
    # 11.7480. The card caught it on its first run against this ruleset.
    ("TOY2", "heuristic"): {"vs random": (0.6730, 0.4474), "vs heuristic": (0.6385, 0.3615), "exploitability": (7.2, 11.7)},
    ("TOY2_SKIP", "random"): {"vs random": (0.5841, 0.4159), "exploitability": (16.7, 16.8)},
    ("TOY2_SKIP", "heuristic"): {"vs random": (0.6414, 0.4806), "vs heuristic": (0.5767, 0.4233), "exploitability": (11.0, 11.9)},
}


def check(ok: bool, what: str) -> str:
    if not ok:
        FAILS.append(what)
    return "ok" if ok else "XX"


def reproduces(value: float, published: float, decimals: int) -> bool:
    """Whether `value` is consistent with `published` read at that many decimals.

    **Not** `round(value, decimals) == published`: the published figure is itself a
    rounded display of the same computation, `round` is banker's rounding, and a
    trailing 5 is rarely exactly representable in binary, so that form disagrees with a
    published number for reasons that have nothing to do with the quantity. Comparing
    against the half-width of the last published digit asks the question that was meant
    -- is the published rounding consistent with what was just computed -- and the
    epsilon covers the representation error exactly at the boundary.

    It stays deliberately tight: a value that genuinely does not round to the published
    one must fail. Widening this to silence a disagreement would turn the card from an
    instrument into a decoration."""
    return abs(value - published) <= 0.5 * 10**-decimals + 1e-6


def score(v: float) -> float:
    return (v + 1) / 2


def seat_scores(tree, row, column) -> tuple[float, float]:
    """The row's score against the column, at seat 0 and at seat 1."""
    return (score(evaluate(tree, row, column)[0]), score(evaluate(tree, column, row)[1]))


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    name = argv[0] if len(argv) > 0 else "TOY"
    who = argv[1] if len(argv) > 1 else "heuristic"
    iterations = int(argv[2]) if len(argv) > 2 else 1000
    rs = getattr(rules, name)

    tree = compile_tree(rs, (L1,), True, cache=store.CACHE)
    print(f"== report card: {who} on {name}")
    print(f"   {tree}")
    print(f"   seeds: exact, none spent ({seed_registry.describe().splitlines()[0]})")

    rnd, heu = PlayFirstRandom(rs), Heuristic(rs)
    columns = {a.name: (L1, store.load_or_tabulate(tree, L1, a)) for a in (rnd, heu)}
    nash = store.load_or_solve_nash(tree, iterations, "cfr+")
    rows = dict(columns)
    rows["nash"] = (L1, nash.avg)
    if who not in rows:
        raise SystemExit(f"unknown agent {who!r}; one of {', '.join(rows)}")
    row = rows[who]
    print(f"   nash: cfr+ {nash.iterations} its, NashConv {100 * nash.final.nashconv:.3f} pts")

    # -- the range, per column, per seat -------------------------------------------------
    print("\nexact")
    pins = PINNED.get((name, who), {})
    for col_name, col in columns.items():
        s = seat_scores(tree, row, col)
        base = seat_scores(tree, columns["random"], col)  # 0 of the normalised scale
        ceil = tuple(best_response(tree, L1, col, seat).score for seat in (0, 1))  # 1 of it: BR L1
        norm = tuple((s[i] - base[i]) / (ceil[i] - base[i]) if ceil[i] > base[i] else float("nan") for i in (0, 1))
        # `br_gap` is in return units; a score point is 0.02 of them (as `uno.experiments.edge`).
        gaps = tuple(50 * br_gap(tree, row, col, seat) for seat in (0, 1))
        line = f"   vs {col_name:<10s} {s[0]:.4f} / {s[1]:.4f}   normalised {norm[0]:.2f} / {norm[1]:.2f}   (BR L1 {ceil[0]:.4f} / {ceil[1]:.4f})"
        pin = pins.get(f"vs {col_name}")
        if pin is not None:
            ok = check(reproduces(s[0], pin[0], 4) and reproduces(s[1], pin[1], 4), f"{who} vs {col_name} on {name} ({s[0]:.6f} / {s[1]:.6f} against the published {pin[0]} / {pin[1]})")
            line += f"  {ok}"
        print(line)
        print(f"   {'':13s} br_gap vs this column                      {gaps[0]:.2f} / {gaps[1]:.2f} pts")

    # -- exploitability --------------------------------------------------------------
    ex = exploitability(tree, L1, row[1])
    nash_value = score(evaluate(tree, rows["nash"], rows["nash"])[0])
    per_seat = tuple(100 * (ex.br[seat ^ 1].score - (1 - nash_value if seat == 0 else nash_value)) for seat in (0, 1))
    line = f"   exploitability                              {per_seat[0]:.2f} / {per_seat[1]:.2f} pts   NashConv {per_seat[0] + per_seat[1]:.2f}"
    pin = pins.get("exploitability")
    if pin is not None:
        ok = check(reproduces(per_seat[0], pin[0], 1) and reproduces(per_seat[1], pin[1], 1), f"exploitability of {who} on {name} ({per_seat[0]:.4f} / {per_seat[1]:.4f} against the published {pin[0]} / {pin[1]})")
        line += f"  {ok}"
    print(line)

    # -- edge by decision ------------------------------------------------------------
    kinds, sits, actions = store.load_or_label(tree)
    labels = (kinds, sits, actions)
    for col_name, col in columns.items():
        shares = []
        for seat in (0, 1):
            ref = (L1, best_response(tree, L1, col, seat).sigma)
            terms = mirror_terms(tree, row, ref, col, seat, labels)
            # `term` is per arc and non-zero only on this seat's own arcs; `kind` labels
            # those same arcs ("" elsewhere), so the split is one grouped sum.
            by_kind: dict[str, float] = {}
            for kind, x in zip(terms.kind, terms.term):
                if kind:
                    by_kind[kind] = by_kind.get(kind, 0.0) + float(x)
            total = sum(by_kind.values())
            shares.append({k: v / total for k, v in by_kind.items()} if abs(total) > 1e-12 else {})
        merged = {k: (shares[0].get(k, 0.0) + shares[1].get(k, 0.0)) / 2 for k in set(shares[0]) | set(shares[1])}
        top = ", ".join(f"{k} {100 * v:.0f} %" for k, v in sorted(merged.items(), key=lambda kv: -kv[1])[:4])
        print(f"   edge by decision vs {col_name:<10s} {top or '(row is the best response)'}")

    print(f"\n== {len(FAILS)} failed checks" + (": " + "; ".join(FAILS) if FAILS else ""))


if __name__ == "__main__":
    main()
