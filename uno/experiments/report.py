"""Step 10a: the report card -- one command, one checkpoint, a fixed format (item A3).

    uv run uno report [ruleset=TOY] [agent=heuristic|random|nash] [iterations=1000] [attack_worlds]

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

**The attack block** (step 10a.3C, `uno.eval.attack`): per-seat attack scores against this
checkpoint, the compute budget that produced them, the spread across the attacker's own
world-sampling seeds, and the two-seat NashConv **lower** bound `b0 + b1 - 1`. On a toy this is
instrument validation rather than news -- the exact exploitability is printed two lines above, and
the bound must come out below it -- but it is the same instrument that will carry the only luck-free
quality number a full-UNO checkpoint can have, so the card exercises it here where the truth is
known. It runs by default only on trees small enough for an attacker tabulation to be cheap (one
tabulation per seed; ~4 s on `TOY`, minutes on `TOY2_SKIP`); pass `attack_worlds` explicitly to
force it on or `0` to skip it.

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


# Attacker tabulation is one Python call per information set, so the attack block is cheap on the
# small toys and minutes long on the big ones. Default it on only where it is cheap, and say so.
ATTACK_SET_LIMIT = 50_000
ATTACK_SEEDS = 2


def attack_block(tree, rs, row_sigma, who: str, target_agent, true_nashconv: float, worlds: int) -> None:
    """Per-seat attack scores, budget, seed spread, and the two-seat NashConv lower bound.

    The attacker is tabulated on the quotient tree, so on a toy every number here is exact and the
    only uncertainty is the attacker's own world-sampling seed (a built attacker is one realization
    of it, not the averaged behavioural policy -- `uno.eval.attack`). The pre-declared check is that
    the bound stays **below** the exact NashConv it bounds: an attack that claimed more than the
    truth would mean the attacker is seeing something it must not."""
    from uno.agents.symmetrized import Symmetrized
    from uno.agents.tabular import CompiledAgent
    from uno.compiled.strategies import tabulate
    from uno.eval import attack as atk
    from uno.eval.attack import Budget

    # A checkpoint stored as a strategy over the keying reads its own L1 history, so it cannot be
    # queried inside a determinized world; the `response` continuation is then refused and a fixed
    # surrogate opponent is used instead. Where a position-only agent exists, `response` is allowed.
    probe_target = target_agent if target_agent is not None else CompiledAgent(tree, L1, row_sigma, name=f"target_{who}")
    sup = atk.supports(rs, probe_target, games=20)
    family = "response" if sup.position_only else "surrogate"
    depth = 40  # always terminal on the toys, so no value network is involved

    def no_leaf(s):
        raise AssertionError("the attack search hit its depth limit on a toy")

    kw = {} if family == "response" else dict(surrogate=Heuristic(rs))
    per_seed = []
    for k in range(ATTACK_SEEDS):
        att = atk.build(rs, probe_target, Budget(continuation=family, worlds=worlds, depth=depth, belief="prior", seed=k), leaf=no_leaf, **kw)
        sigma = tabulate(tree, L1, Symmetrized(att, rs, name=f"sym_{att.name}"))
        per_seed.append(
            (
                score(evaluate(tree, (L1, sigma), (L1, row_sigma))[0]),
                score(evaluate(tree, (L1, row_sigma), (L1, sigma))[1]),
            )
        )
    b0 = np.array([p[0] for p in per_seed])
    b1 = np.array([p[1] for p in per_seed])
    bounds = 100 * (b0 + b1 - 1)
    best = float(bounds.max())  # the strongest attack found is the best bound available
    print(
        f"   attack (LBR)                                {b0.mean():.4f} / {b1.mean():.4f}"
        f"   NashConv >= {max(0.0, best):.2f} pts   ({family}, w={worlds}, {ATTACK_SEEDS} seeds)"
    )
    print(
        f"   {'':13s} seed spread {100 * b0.std(ddof=0):.2f} / {100 * b1.std(ddof=0):.2f} pts"
        f"   exact NashConv {true_nashconv:.2f}, so the bound recovers "
        + (f"{best / true_nashconv:.0%}" if true_nashconv > 0.05 else "n/a (the target is unexploitable)")
    )
    if not sup.position_only:
        print(f"   {'':13s} `response` refused: this target reads its own private history, so a fixed surrogate opponent was used instead")
    check(
        best <= true_nashconv + 1e-6,
        f"the attack bound ({best:.2f}) exceeds the exact NashConv it bounds ({true_nashconv:.2f}) for {who} on {tree.rs!r}",
    )


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    name = argv[0] if len(argv) > 0 else "TOY"
    who = argv[1] if len(argv) > 1 else "heuristic"
    iterations = int(argv[2]) if len(argv) > 2 else 1000
    attack_worlds = int(argv[3]) if len(argv) > 3 else None
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

    # -- the attack instrument (step 10a.3C) -----------------------------------------
    n_sets = tree.keyings[L1].n_info
    worlds = attack_worlds if attack_worlds is not None else (8 if n_sets <= ATTACK_SET_LIMIT else 0)
    if worlds:
        attack_block(tree, rs, row[1], who, {"random": rnd, "heuristic": heu}.get(who), per_seat[0] + per_seat[1], worlds)
    else:
        print(f"   attack (LBR)                                skipped: {n_sets:,} sets above the {ATTACK_SET_LIMIT:,} default limit; pass attack_worlds to run it")

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
