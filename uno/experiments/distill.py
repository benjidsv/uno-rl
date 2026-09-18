"""Arm-B PIMC distillation smoke run.

    uv run uno distill toy [bootstrap-seeds=24] [dagger-seeds=24]
    uv run uno distill full [bootstrap-seeds=200] [dagger-seeds=200]

The command is intentionally a bounded experiment, not a claim about the
10C/EVAL rows.  It bootstraps from frozen column-free PIMC trajectories,
aggregates one round of student-reached states, and prints paired scores on a
disjoint DEV tail.  Full-UNO claims still require a fresh registered block.
"""

from __future__ import annotations

import sys
from pathlib import Path

from uno import store
from uno.agents.baseline import Heuristic, PlayFirstRandom
from uno.eval.belief import Belief
from uno.eval.distill import DistillBuffer, PimcQTeacher, fit_arm_b
from uno.eval.play import match
from uno.eval.search import PimcSearch
from uno.eval.value_net import ValueNet
from uno.game import rules


def _net_paths(name: str) -> tuple[Path, Path]:
    root = Path(store.CACHE) / "value_net"
    if name == "TOY":
        suffix = "n4_g300_w128_v1.pt"
    elif name == "FULL_UNO":
        suffix = "n2_g20000_w128_v1.pt"
    else:
        raise ValueError(f"no distillation net convention for {name}")
    return tuple(root / f"{name}_heuristic_random_s{s}_{suffix}" for s in (0, 1))


def _teacher(rs, name: str, worlds: int):
    paths = _net_paths(name)
    if not all(p.exists() for p in paths):
        raise SystemExit("missing cached value networks; run `uv run uno pimc_value aivat first`")
    leaves = tuple(ValueNet.load(p, rs).leaf for p in paths)
    surrogate = Heuristic(rs)
    search = PimcSearch(rs, leaves, worlds=worlds, depth=2, name=f"distill_teacher_{name}", opponent=surrogate, belief=Belief(rs, kind="voids"))
    return search, PimcQTeacher(search)


def run(name: str, bootstrap: int, dagger: int) -> None:
    rs = rules.NAMED[name]
    worlds = 8 if name == "TOY" else 32
    teacher, qteacher = _teacher(rs, name, worlds)
    columns = (PlayFirstRandom(rs), Heuristic(rs))
    # Bootstrap labels come from teacher-reached states.  The second call is the
    # first DAgger round, after which the student is evaluated only on fresh seeds.
    buf = DistillBuffer(rs)
    seeds0 = range(20_000, 20_000 + bootstrap)
    added0 = buf.aggregate(teacher, qteacher, columns, seeds0, temperature=0.05)
    student_raw, report0 = fit_arm_b(rs, buf.rows, width=128, max_epochs=40, patience=6, seed=0, policy_temperature=0.05)
    student = student_raw.symmetric(name=f"arm_b_distilled_{name.lower()}_r0")
    seeds1 = range(30_000, 30_000 + dagger)
    added1 = buf.aggregate(student, qteacher, columns, seeds1, start_game=10_000_000, temperature=0.05)
    student_raw, report1 = fit_arm_b(rs, buf.rows, width=128, max_epochs=60, patience=8, seed=1, policy_temperature=0.05)
    student = student_raw.symmetric(name=f"arm_b_distilled_{name.lower()}_r1")
    out = Path(store.CACHE) / "distill"
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / f"{name.lower()}_arm_b_r1.pt"
    student.save(str(checkpoint))
    print(f"Arm B distillation {name}: bootstrap {added0:,}, dagger {added1:,}; {report0}; {report1}; checkpoint {checkpoint}")
    eval_seeds = range(40_000, 40_000 + max(20, min(100, dagger)))
    for column in columns:
        for label, row in (("teacher", teacher), ("student", student)):
            s0 = match(rs, (row, column), eval_seeds).score
            s1 = match(rs, (row, column), eval_seeds, swap=True).score
            print(f"  {label:7s} vs {column.name:9s}: {s0:.4f} / {s1:.4f}")


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    name = {"toy": "TOY", "full": "FULL_UNO"}.get(argv[0].lower() if argv else "toy")
    if name is None:
        raise SystemExit("usage: uno distill toy|full [bootstrap-seeds] [dagger-seeds]")
    bootstrap = int(argv[1]) if len(argv) > 1 else (24 if name == "TOY" else 200)
    dagger = int(argv[2]) if len(argv) > 2 else bootstrap
    run(name, bootstrap, dagger)


if __name__ == "__main__":
    main()
