"""Step 10B addendum: the full-UNO league (win rates, cost, and the deal floor).

    uv run uno league [games=150] [worlds=32] [depth=2] [workers=4]
        [output_root=docs/generated/league]

Three more figures on top of `uv run uno figures`. This one plays real full-UNO games
(minutes, not seconds) rather than exact toy math, on **DEV-pool seeds**
(`uno.eval.seeds`): it is a diagnostic visualization of league composition, not a
reported EVAL number, so nothing here claims a fresh block.

**League membership** is every *general* (opponent-oblivious) full-UNO policy this
project has actually built: true uniform-random (`UniformLegal`), the ladder's
`PlayFirstRandom`, the `Heuristic` rulebot, and the two column-free PIMC rows of step
10a.2 (`uno.experiments.colfree.column_free_row`, assume-heuristic and minimax, both
with the `voids` belief, w=32 d=2 -- the working general-player set that step
established). The `response` / `window` PIMC continuations are **excluded on purpose**:
they read their column's real policy by construction (docs/foundations/results/10-readiness/10a2-results.md,
docs/foundations/results/09-evaluation/09.5f-results.md), so "X vs Random" and "X vs Heuristic" are not the same
policy under those continuations -- they are ladder *instruments*, and a name identifies
a policy (docs/foundations/spec.md section 13). Folding them into one league row would silently
report several different agents under one name.

Round-robin: every ordered (row, column) pair, including self-play, scored at both
seats from the same `games` DEV seeds (`uno.experiments.colfree.play`, which already
plays both seatings of each deal and times the wall clock -- reused rather than
reimplemented). Figures:

- `league-winrate.svg`: two heatmaps (seat 0, seat 1), row agent's score against
  the column agent.
- `cost-vs-strength.svg`: ms/game (vs the Heuristic, the project's standard cost
  reference, e.g. 09.5e) against mean league score (both seats, every opponent
  including self-play), log-x, with the Pareto-efficient points connected.
- `deal-floor.svg`: the deal floor by game length (`uv run uno pimc_value floor`,
  docs/foundations/results/index.md "PIMC as an agent"), entirely historical -- five already-published
  numbers, not re-measured.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from uno.agents.baseline import Heuristic, PlayFirstRandom, UniformLegal
from uno.eval import seeds as seed_registry
from uno.experiments.colfree import column_free_row
from uno.experiments.pimc import net_for
from uno.experiments.colfree import play as play_pair
from uno.game import rules

# Average decisions ("actions") per game and the deal floor's share of the outcome variance
# (Heuristic vs Random, 2,000 EVAL deals x 32 pile determinizations), seat 0 / seat 1.
# docs/foundations/results/index.md "PIMC as an agent" / "the deal floor"; docs/foundations/results/09-evaluation/09.5b-results.md
# section 4 (actions/game); docs/foundations/results/03-08-foundations/08.7-toy2.md (TOY / TOY2 dec/game).
DEAL_FLOOR = [
    ("TOY", 6.5, 0.42, 0.47),
    ("TOY2", 7.1, 0.38, 0.40),
    ("MID_12", 13.7, 0.25, 0.26),
    ("MID_25", 21.3, 0.17, 0.16),
    ("FULL_UNO", 45.8, 0.086, 0.090),
]


def league_agents(rs, leaves, worlds: int, depth: int) -> dict:
    heu = Heuristic(rs)
    return {
        "uniform_legal": UniformLegal(rs),
        "random": PlayFirstRandom(rs),
        "heuristic": heu,
        "pimc_cf_minimax": column_free_row(rs, leaves, None, worlds, depth),
        "pimc_cf_heuristic": column_free_row(rs, leaves, heu, worlds, depth),
    }


def round_robin(rs, agents: dict, seeds, workers: int | None) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    names = list(agents)
    n = len(names)
    seat0, seat1, ms = (np.zeros((n, n)) for _ in range(3))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            arr, cost = play_pair(rs, agents[a], agents[b], seeds, workers)
            seat0[i, j], seat1[i, j], ms[i, j] = arr[:, 0].mean(), arr[:, 1].mean(), cost
            print(f"   {a:20s} vs {b:20s}  seat0 {seat0[i, j]:.4f}  seat1 {seat1[i, j]:.4f}  {cost:6.1f} ms/game")
            sys.stdout.flush()
    return names, seat0, seat1, ms


def fig_winrate(names: list[str], seat0: np.ndarray, seat1: np.ndarray, path: Path) -> None:
    """6. League win rates: two heatmaps, row agent's score against the column agent."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    for k, (ax, mat, title) in enumerate(zip(axes, (seat0, seat1), ("row's score at seat 0", "row's score at seat 1"))):
        im = ax.imshow(mat, vmin=0, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(len(names)))
        if k == 0:
            ax.set_yticklabels(names, fontsize=8)
        else:
            ax.tick_params(labelleft=False)
        ax.set_title(title, fontsize=10)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7.5, color="white" if abs(mat[i, j] - 0.5) > 0.25 else "black")
        ax.set_xlabel("column agent")
    axes[0].set_ylabel("row agent")
    fig.subplots_adjust(wspace=0.08)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.03, label="score of the row")
    fig.suptitle("The full-UNO league: round-robin win rates (DEV seeds, diagnostic)")
    fig.savefig(path, format="svg")
    plt.close(fig)


def fig_cost_vs_strength(names: list[str], seat0: np.ndarray, seat1: np.ndarray, ms: np.ndarray, path: Path) -> None:
    """7. Cost vs strength: ms/game against the Heuristic (x, log) vs mean league score (y)."""
    heu_i = names.index("heuristic")
    cost = ms[:, heu_i]
    strength = (seat0.mean(axis=1) + seat1.mean(axis=1)) / 2
    order = np.argsort(cost)
    pareto_x, pareto_y, best = [], [], -1.0
    for i in order:  # cheapest first: a point is Pareto-efficient if no cheaper point scores as high
        if strength[i] > best:
            pareto_x.append(cost[i]), pareto_y.append(strength[i])
            best = strength[i]

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(pareto_x, pareto_y, color="#cccccc", lw=2, zorder=1, label="Pareto frontier")
    ax.scatter(cost, strength, color="#1f77b4", s=70, zorder=2)
    for i, name in enumerate(names):
        ax.annotate(name, (cost[i], strength[i]), fontsize=8, xytext=(6, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("ms/game vs the Heuristic (log scale)")
    ax.set_ylabel("mean score across the league (both seats, incl. self-play)")
    ax.set_title("Cost vs strength: the full-UNO league")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)


def fig_deal_floor(path: Path) -> None:
    """8. The deal floor by game length: entirely historical (docs/foundations/results/index.md)."""
    labels = [d[0] for d in DEAL_FLOOR]
    x = [d[1] for d in DEAL_FLOOR]
    y0 = [d[2] for d in DEAL_FLOOR]
    y1 = [d[3] for d in DEAL_FLOOR]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(x, y0, "o-", color="#1f77b4", label="seat 0")
    ax.plot(x, y1, "o--", color="#ff7f0e", alpha=0.7, label="seat 1")
    for xi, yi, label in zip(x, y0, labels):
        ax.annotate(label, (xi, yi), fontsize=8, xytext=(6, 6), textcoords="offset points")
    ax.set_xlabel("average decisions per game")
    ax.set_ylabel("deal floor: share of outcome variance fixed at the deal")
    ax.set_title("The deal floor falls as the game gets longer\n(heuristic vs random, docs/foundations/results/index.md 'the deal floor')")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 0.55)
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    games = int(argv[0]) if len(argv) > 0 else 150
    worlds = int(argv[1]) if len(argv) > 1 else 32
    depth = int(argv[2]) if len(argv) > 2 else 2
    workers = int(argv[3]) if len(argv) > 3 else 4
    output_root = Path(argv[4]) if len(argv) > 4 else Path("docs/generated/league")
    figure_dir = output_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    rs = rules.FULL_UNO

    print(f"== the full-UNO league: {games} DEV deals/pair, w={worlds} d={depth}, {workers} workers")
    nets = [net_for(rs, "heuristic", "random", s) for s in (0, 1)]
    leaves = (nets[0].leaf, nets[1].leaf)
    agents = league_agents(rs, leaves, worlds, depth)
    seeds = seed_registry.dev_block(0, games)
    names, seat0, seat1, ms = round_robin(rs, agents, seeds, workers)

    print(f"\n== writing figures to {figure_dir}")
    fig_winrate(names, seat0, seat1, figure_dir / "league-winrate.svg")
    fig_cost_vs_strength(names, seat0, seat1, ms, figure_dir / "cost-vs-strength.svg")
    fig_deal_floor(figure_dir / "deal-floor.svg")
    for name in ("league-winrate", "cost-vs-strength", "deal-floor"):
        print(f"   {figure_dir / (name + '.svg')}")


if __name__ == "__main__":
    main()
