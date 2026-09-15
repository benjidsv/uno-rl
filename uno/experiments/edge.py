"""Step 9.5a: where a row agent's score against a fixed opponent is won or lost,
by decision (D2 of the step 9 review's section D), and the exact gap-to-ceiling
metric for training against a fixed opponent (D4).

    uv run uno edge [ruleset=TOY] [iterations=1000] [light]

Rows: Random, Heuristic, Nash L1. Opponents: Random, Heuristic. Reference: the best
response to the opponent (`uno.compiled.edge.br_gap` is the D4 metric, the row's
gap to that ceiling). For each (row, opponent, seat):

- the mirror and forward decompositions (`uno.compiled.edge.mirror_terms` /
  `forward_terms`), checked against `br_gap` and against each other to 1e-9;
- the mirror total by `uno.eval.decisions.node_kind`, checked to sum to the whole;
- the mirror total by `action_class`, with the wild "spend" / "color" split
  (`uno.compiled.edge.wild_split`) alongside as a separate, always non-negative
  diagnostic (not a term of the mirror decomposition itself);
- the mirror total by opponent-hand-size situation bucket;
- stakes (`uno.compiled.edge.stakes`): the reach-weighted distribution of
  max_a Q^ref - min_a Q^ref per decision, as quantiles and the share above 1 score
  point (0.02 return units).

Every mirror set-action total must be >= 0 when the reference is the best response
(docs/spec.md section 21): checked at set level, printed ok / XX per (row, opponent,
seat). `light` (TOY2_FULL): DCFR only, fewer iterations for Nash, as `robustness`.

Also, unless `light`: a sampled cross-check of the exact decomposition
(`uno.compiled.edge.sampled_mirror`, the "sampled D2 check") -- heuristic vs
random, both seats, reference BR L1(random), 2,000 held-out seeds distinct from
`DEV_SEEDS` / `EVAL_SEEDS`. Needs the tree's L2 keying (`uno.compiled.lookup`),
skipped on `TOY2_FULL`, whose L2 keying alone is ~3 GB.
"""

from __future__ import annotations

import sys

import numpy as np

from uno import store
from uno.agents.baseline import Heuristic, PlayFirstRandom
from uno.compiled.edge import br_gap, forward_terms, mirror_terms, sampled_mirror, stakes, wild_split
from uno.compiled.lookup import build as build_index
from uno.compiled.solvers import best_response
from uno.compiled.tree import compile_tree
from uno.game import rules
from uno.game.observe import Level

L1 = Level.L1
FAILS: list[str] = []
POINT = 0.02  # return units per score point


def check(ok: bool, what: str) -> str:
    if not ok:
        FAILS.append(what)
    return "ok" if ok else "XX"


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    if len(values) == 0 or weights.sum() <= 0:
        return float("nan")
    order = np.argsort(values)
    v, w = values[order], weights[order]
    c = np.cumsum(w) / w.sum()
    i = int(np.searchsorted(c, q))
    return float(v[min(i, len(v) - 1)])


def group_sum(term: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for label in sorted(set(labels.tolist()) - {""}):
        out[label] = float(term[labels == label].sum())
    return out


def print_group(title: str, groups: dict[str, float], total: float) -> None:
    print(f"      {title}:")
    for label, v in sorted(groups.items(), key=lambda kv: -abs(kv[1])):
        print(f"        {label:40s} {100 * v / 2:+7.2f} pts")
    check(abs(sum(groups.values()) - total) < 1e-9, f"{title}: kind totals sum to the whole")


def report(tree, row_name: str, row, opponent_name: str, opponent, seat: int, labels, kinds) -> None:
    br = best_response(tree, L1, opponent, seat)
    ref = (L1, br.sigma)
    gap = br_gap(tree, row, opponent, seat)
    m = mirror_terms(tree, row, ref, opponent, seat, labels)
    f = forward_terms(tree, row, ref, opponent, seat, labels)
    ok1 = check(abs(m.total - gap) < 1e-9, f"{row_name} vs {opponent_name} seat {seat}: mirror total == br_gap")
    ok2 = check(abs(f.total - gap) < 1e-9, f"{row_name} vs {opponent_name} seat {seat}: forward total == br_gap")
    k = tree.keyings[L1]
    mine = tree.arc_player == seat
    set_id = k.node_info[tree.arc_node]
    totals = np.bincount(np.where(mine, set_id, 0), weights=np.where(mine, m.term, 0.0), minlength=k.n_info)
    seat_sets = np.nonzero(k.info_player == seat)[0]
    ok3 = check(bool(np.all(totals[seat_sets] >= -1e-9)), f"{row_name} vs {opponent_name} seat {seat}: mirror set totals nonneg (ref = BR L1)")
    print(f"\n   {row_name} vs {opponent_name}, seat {seat}: br_gap {100 * gap / 2:+.2f} pts, mirror {100 * m.total / 2:+.2f} ({ok1}), forward {100 * f.total / 2:+.2f} ({ok2}), set-level {ok3}")
    print_group("by node kind", group_sum(m.term, m.kind), m.total)
    ac = group_sum(m.term, m.action)
    print_group("by action class", ac, m.total)
    ws = wild_split(tree, row, ref, opponent, seat, kinds)
    if len(ws.node) > 0 and ws.weight.sum() > 0:
        spend = 100 * float(ws.weight @ ws.spend) / (2 * ws.weight.sum())
        color = 100 * float(ws.weight @ ws.color) / (2 * ws.weight.sum())
        print(f"      wild split (reach-weighted mean, pts): spend {spend:.2f}, color {color:.2f}")
    sit_opp = np.array([s.split("/")[1] if s else "" for s in m.situation])
    print_group("by opponent hand size", group_sum(m.term, sit_opp), m.total)
    st = stakes(tree, row, ref, opponent, seat)
    if len(st.node) > 0 and st.weight.sum() > 0:
        q50, q90 = weighted_quantile(st.spread, st.weight, 0.5), weighted_quantile(st.spread, st.weight, 0.9)
        above = float(st.weight[st.spread > POINT].sum()) / float(st.weight.sum())
        print(f"      stakes (max-min under the reference, return units): median {q50:.3f}, 90th pct {q90:.3f}, share > 1 pt {above:.3f}")


def sampled_check(rs, seeds) -> None:
    """The sampled D2 cross-check: heuristic vs random, both seats, reference BR
    L1(random), on the tree's L2 keying (`uno.compiled.lookup`)."""
    tree = compile_tree(rs, (L1, Level.L2), True, cache=store.CACHE)
    rnd, heu = PlayFirstRandom(rs), Heuristic(rs)
    opp = (L1, store.load_or_tabulate(tree, L1, rnd))
    idx = build_index(tree, Level.L2)
    print(f"\n== sampled D2 check ({tree.keyings[Level.L2].n_info:,} L2 sets, {len(seeds)} seeds): heuristic vs random, reference BR L1(random)")
    for seat in (0, 1):
        br = best_response(tree, L1, opp, seat)
        ref = (L1, br.sigma)
        row = (L1, store.load_or_tabulate(tree, L1, heu))
        exact = mirror_terms(tree, row, ref, opp, seat, store.load_or_label(tree, L1))
        sm = sampled_mirror(tree, idx, heu, rnd, ref, seat, seeds)
        z = (sm.total.mean - exact.total) / sm.total.se if sm.total.se > 0 else 0.0
        ok = check(abs(z) <= 3, f"sampled D2 seat {seat}: total (z = {z:+.2f})")
        print(f"   seat {seat}: exact {100 * exact.total / 2:+.2f} pts, sampled {100 * sm.total.mean / 2:+.2f} ± {100 * sm.total.half / 2:.2f} pts (n={sm.n}), z {z:+.2f}  {ok}")
        exact_by_kind = group_sum(exact.term, exact.kind)
        for kind, est in sm.by_kind.items():
            if kind not in exact_by_kind or est.se == 0:
                continue
            zk = (est.mean - exact_by_kind[kind]) / est.se
            okk = check(abs(zk) <= 3, f"sampled D2 seat {seat} kind {kind} (z = {zk:+.2f})")
            print(f"      {kind:10s} exact {100 * exact_by_kind[kind] / 2:+7.2f} pts, sampled {100 * est.mean / 2:+7.2f} ± {100 * est.half / 2:5.2f} pts, z {zk:+6.2f}  {okk}")
        sys.stdout.flush()


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    name = argv[0] if len(argv) > 0 else "TOY"
    iterations = int(argv[1]) if len(argv) > 1 else 1000
    light = len(argv) > 2 and argv[2] == "light"
    rs = getattr(rules, name)
    variant = "dcfr" if light else "cfr+"
    tree = compile_tree(rs, (L1,), True, cache=store.CACHE)
    print(f"== {name}: {tree}")
    kinds, sits, actions = store.load_or_label(tree)
    labels = (kinds, sits, actions)
    rnd, heu = PlayFirstRandom(rs), Heuristic(rs)
    tab = {a.name: (L1, store.load_or_tabulate(tree, L1, a)) for a in (rnd, heu)}
    nash = store.load_or_solve_nash(tree, iterations, variant)
    tab["nash"] = (L1, nash.avg)
    print(f"   nash ({variant}, {nash.iterations} its, NashConv {100 * nash.final.nashconv:.4f} pts)")
    for opponent_name in ("random", "heuristic"):
        for row_name in ("random", "heuristic", "nash"):
            for seat in (0, 1):
                report(tree, row_name, tab[row_name], opponent_name, tab[opponent_name], seat, labels, kinds)
                sys.stdout.flush()

    if not light:
        sampled_check(rs, range(5_000_000, 5_004_000))

    print(f"\n== {len(FAILS)} failed checks" + (": " + "; ".join(FAILS) if FAILS else ""))


if __name__ == "__main__":
    main()
