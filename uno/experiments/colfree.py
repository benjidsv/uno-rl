"""Step 10a.2 item A5a: a column-free PIMC row on full UNO, one that reads nothing of its opponent.

    uv run uno colfree [deals=4000] [worlds=32] [depth=2] [workers] [identity_deals=200]

The full-UNO PIMC row that carries the ladder is not a player. Both levers behind its points
read the column's *real policy*: the `response` continuation solves each determinized world
as a best response to it (step 9.5e, +9.73 / +4.57 points), and the `window` belief scores
sampled hands by its action probabilities (9.5f). A best response is *supposed* to read its
row, so those are legitimate **instruments** -- but a rung a step-11 agent could actually be
trained against, or that could be called a strong general player, may not read its opponent
at all, and 9.5f's own open list records that no such PIMC has ever been measured.

**What makes a row column-free.** Two choices, made in advance and never per opponent:

- the continuation assumes a **fixed surrogate** opponent (here the Heuristic: "assume a
  competent play-first player"), or assumes a clairvoyant one (`minimax`, which reads
  nothing whatsoever). Either way the assumption does not change with who sits opposite;
- the belief is `voids`, which reads no policy *probabilities*. It is not assumption-free --
  it assumes a playable drawn card is played at once (docs/foundations/spec.md section 26.2) -- but that
  assumption is a fixed property of the row, not a reading of the column. `window` is
  therefore excluded by construction, whatever it scores.

**The check that makes this a measurement rather than a new agent.** Handed its own column
as the surrogate, the column-free row *is* the `response` row: same worlds, same searches,
same games. Since a name identifies a policy (docs/foundations/spec.md section 13) and these two
constructions are one policy, they share a name -- hence a world-sampling stream -- and the
per-game scores must be **array-equal**, not merely close. If the builder differs from the
ladder's own construction in any parameter (the per-seat leaves, the belief, the budget), the
games diverge and the equality fails.

Seeds: **EVAL block 01** (`uno.eval.seeds`), claimed for this item. Block 00 -- the original
`EVAL_SEEDS` -- is spent: 9.5d-9.5f selected the working set on it, so a row reported there
would be scored on the seeds its own budget was chosen on (docs/foundations/spec.md section 27.1).
"""

from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

from uno.agents.baseline import Heuristic, PlayFirstRandom
from uno.eval import seeds as seed_registry
from uno.eval.belief import Belief
from uno.eval.pimc import chunks, pmap
from uno.eval.search import PimcSearch
from uno.experiments.pimc import _belief_chunk, net_for
from uno.game import rules

FAILS: list[str] = []


def check(ok: bool, what: str) -> str:
    if not ok:
        FAILS.append(what)
    return "ok" if ok else "XX"


def z_of(diff: float, se: float) -> float:
    if se < 1e-15:
        return 0.0 if abs(diff) <= 1e-12 else math.copysign(math.inf, diff)
    return diff / se


def response_row(rs, leaves, column, worlds: int, depth: int, name: str | None = None) -> PimcSearch:
    """The ladder's row: the `response` continuation against the column's *real* policy plus
    the `voids` belief -- the working set of 9.5d-9.5f. Built exactly as
    `uno.experiments.pimc.belief_section` builds it, including the name, so that a row
    constructed here replays that section's games rather than merely resampling them."""
    return PimcSearch(rs, leaves, worlds=worlds, depth=depth, name=name or f"pimc_resp_{column.name}_voids", opponent=column, belief=Belief(rs, kind="voids"))


def column_free_row(rs, leaves, surrogate, worlds: int, depth: int, name: str | None = None) -> PimcSearch:
    """The column-free row: the continuation best-responds to a **fixed** `surrogate` policy
    (or assumes a clairvoyant opponent when `surrogate is None`), and the belief is `voids`,
    which reads no policy probabilities. Nothing here depends on who the row actually plays."""
    label = "minimax" if surrogate is None else surrogate.name
    return PimcSearch(rs, leaves, worlds=worlds, depth=depth, name=name or f"pimc_cf_{label}_voids", opponent=surrogate, belief=Belief(rs, kind="voids"))


def play(rs, agent, column, seeds, workers: int | None) -> tuple[np.ndarray, float]:
    """Both seatings of every deal; the row's score per seat, per deal."""
    k = (os.cpu_count() or 1) * 8 if workers != 1 else 1
    t0 = time.perf_counter()
    parts = pmap(_belief_chunk, [(rs, (agent, column), c) for c in chunks(seeds, k)], workers)
    arr = np.array([g for p in parts for g in p])
    return arr, 1000 * (time.perf_counter() - t0) / max(2 * len(arr), 1)


def identity_section(rs, leaves, worlds: int, depth: int, workers: int | None, deals: int) -> None:
    """Handed its own column, the column-free row must be the `response` row, game for game."""
    print(f"\n== 1. the identity: handed its own column as the surrogate, the column-free row is the `response` row ({deals} dev deals)")
    seeds = seed_registry.dev_block(0, deals)
    for column in (PlayFirstRandom(rs), Heuristic(rs)):
        name = f"pimc_resp_{column.name}_voids"  # one policy, one name (docs/foundations/spec.md section 13)
        ref, _ = play(rs, response_row(rs, leaves, column, worlds, depth), column, seeds, workers)
        cf, _ = play(rs, column_free_row(rs, leaves, column, worlds, depth, name=name), column, seeds, workers)
        same = ref.shape == cf.shape and np.array_equal(ref, cf)
        ok = check(same, f"identity vs {column.name}: the column-free row handed its own column does not reproduce the response row game for game")
        print(f"   vs {column.name:10s} {len(ref)} deals, per-game scores array-equal: {same}  {ok}  (row scores {ref[:, 0].mean():.4f} / {ref[:, 1].mean():.4f})")
        sys.stdout.flush()


def ladder_section(rs, leaves, worlds: int, depth: int, workers: int | None, deals: int) -> None:
    """The column-free rungs against both columns, on the fresh eval block."""
    seeds = seed_registry.block("01")[:deals]
    heu, rnd = Heuristic(rs), PlayFirstRandom(rs)
    print(f"\n== 2. the ladder on EVAL block 01 ({deals} deals, w={worlds}, d={depth}, both seatings)")
    print(f"   {seed_registry.REGISTRY['01']}")
    print(f"   {'row':34s} {'vs':10s} {'seat 0':>17s} {'seat 1':>17s} {'ms/game':>8s}")
    rows = {
        "column-free: assume heuristic, voids": lambda: column_free_row(rs, leaves, heu, worlds, depth),
        "column-free: minimax, voids": lambda: column_free_row(rs, leaves, None, worlds, depth),
    }
    scores: dict[tuple[str, str], np.ndarray] = {}
    for column in (rnd, heu):
        # the instrument, for reference: it reads this column's real policy
        for label, build in (*rows.items(), (f"instrument: response({column.name}), voids", lambda: response_row(rs, leaves, column, worlds, depth))):
            agent = build()
            arr, ms = play(rs, agent, column, seeds, workers)
            scores[(label, column.name)] = arr
            cells = [f"{arr[:, s].mean():.4f}+-{arr[:, s].std(ddof=1) / math.sqrt(len(arr)):.4f}" for s in (0, 1)]
            print(f"   {label:34s} {column.name:10s} {cells[0]:>17s} {cells[1]:>17s} {ms:8.1f}")
            sys.stdout.flush()
        # the rulebot's own row on the same deals, as the floor a general player must clear
        arr, ms = play(rs, heu if column is rnd else PlayFirstRandom(rs), column, seeds, workers)
        scores[("baseline", column.name)] = arr
        cells = [f"{arr[:, s].mean():.4f}+-{arr[:, s].std(ddof=1) / math.sqrt(len(arr)):.4f}" for s in (0, 1)]
        print(f"   {'baseline: ' + ('heuristic' if column is rnd else 'random'):34s} {column.name:10s} {cells[0]:>17s} {cells[1]:>17s} {ms:8.1f}")

    sizes = {k: len(v) for k, v in scores.items()}
    check(len(set(sizes.values())) == 1, f"the rows did not score the same deals ({sizes}), so the paired differences below are not paired")

    print("\n== 3. what reading the column is worth: the instrument minus each column-free row (score points, paired on the deal)")
    print(f"   {'comparison':46s} {'vs':10s} {'seat 0':>20s} {'seat 1':>20s}")
    for column in (rnd, heu):
        inst = scores[(f"instrument: response({column.name}), voids", column.name)]
        for label in rows:
            cf = scores[(label, column.name)]
            cells = []
            for s in (0, 1):
                d = inst[:, s] - cf[:, s]
                se = d.std(ddof=1) / math.sqrt(len(d))
                cells.append(f"{100 * d.mean():+6.2f}+-{100 * se:.2f} (z {z_of(d.mean(), se):+.1f})")
            print(f"   {'response - (' + label + ')':46s} {column.name:10s} {cells[0]:>20s} {cells[1]:>20s}")
        # and the claim a general player has to make: it beats the rulebot at both seats
        base = scores[("baseline", column.name)]
        for label in rows:
            cf = scores[(label, column.name)]
            zs = []
            cells = []
            for s in (0, 1):
                d = cf[:, s] - base[:, s]
                se = d.std(ddof=1) / math.sqrt(len(d))
                zs.append(z_of(d.mean(), se))
                cells.append(f"{100 * d.mean():+6.2f}+-{100 * se:.2f} (z {zs[-1]:+.1f})")
            print(f"   {'(' + label + ') - the rulebot':46s} {column.name:10s} {cells[0]:>20s} {cells[1]:>20s}")
        sys.stdout.flush()


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    deals = int(argv[0]) if len(argv) > 0 else 4000
    worlds = int(argv[1]) if len(argv) > 1 else 32
    depth = int(argv[2]) if len(argv) > 2 else 2
    workers = int(argv[3]) if len(argv) > 3 and argv[3] else None
    identity_deals = int(argv[4]) if len(argv) > 4 else 200
    rs = rules.FULL_UNO
    nets = [net_for(rs, "heuristic", "random", s) for s in (0, 1)]
    leaves = (nets[0].leaf, nets[1].leaf)  # per seat (step 9.5e)
    print("== FULL_UNO: a column-free PIMC row (item A5a)")
    print(f"   w={worlds}, d={depth}; leaves are step 9.5c's per-seat value networks")
    if identity_deals:
        identity_section(rs, leaves, worlds, depth, workers, identity_deals)
    if deals:
        ladder_section(rs, leaves, worlds, depth, workers, deals)
    print(f"\n== {len(FAILS)} failed checks" + (": " + "; ".join(FAILS) if FAILS else ""))


if __name__ == "__main__":
    main()
