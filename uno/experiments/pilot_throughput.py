"""RL open item: hardware/pilot throughput for the compute budget (docs/rl/compute-budget.md).

    uv run uno pilot_throughput [games=1000]

No learner network exists yet, so this cannot measure a real end-to-end
learner rate. It measures three separate ceilings instead, matched to the
components compute-budget.md's `R` formula wants separated:

1. Engine-only rate: `PlayFirstRandom` (near-zero policy cost) against each
   frozen league member (`uno.agents.rulebot.rulebot_league`), both seats, on
   `FULL_UNO`. This is engine step + observation construction + a trivial
   Python policy call per decision -- close to a pure simulation floor.
2. Simulation-with-policy rate: the same match with `Heuristic` standing in
   for the learner's actor. `Heuristic` is a nontrivial per-decision Python
   scorer, so this is closer to (but still cheaper than) a real actor that
   also builds feature tensors and runs a batched forward pass. Compare
   against (1) to see how much a Python-level policy already costs.
3. Synthetic inference/update ceiling: a generic MLP at the plan's pilot
   width (128, plan.md sec 2) and the compute-budget candidate batch sizes
   (512, 8192), timed forward-only and forward+backward, on every available
   torch device. This is a matmul-throughput proxy, not the real structured
   encoder, and excludes real feature construction.

None of these three numbers is the learner's real decisions/s. They bound it
from different directions: (1)/(2) bound collection without a real actor,
(3) bounds what the network math alone costs once collection produces batches.
Replace the "Assumed R" table in compute-budget.md with measured numbers here,
not with any single row of this script's output.
"""

from __future__ import annotations

import platform
import sys
import time

import torch

from uno.agents.baseline import Heuristic, PlayFirstRandom
from uno.agents.rulebot import rulebot_league
from uno.eval.play import DEV_SEEDS, match
from uno.game import rules


def peak_rss_mib() -> float:
    ru = 0  # this is macos specific and was removed because im too lazy to fix
    return ru / (1024 * 1024) if sys.platform == "darwin" else ru / 1024


def simulate(label: str, learner, games: int) -> None:
    rs = rules.FULL_UNO
    league = rulebot_league(rs)
    seeds = list(DEV_SEEDS)[:games]
    print(
        f"\n{label}: FULL_UNO, {games} deals/cell x {len(league)} opponents x 2 seats"
    )
    print(
        f"{'opponent':45s} {'games/s':>9s} {'dec/s (both seats)':>19s} {'len':>6s} {'score':>7s}"
    )
    total_games = total_decisions = 0
    t_all = time.perf_counter()
    for name, opp in league.items():
        for swap in (False, True):
            t0 = time.perf_counter()
            res = match(rs, (learner, opp), seeds, swap=swap)
            dt = time.perf_counter() - t0
            decisions = res.n * res.mean_length
            total_games += res.n
            total_decisions += decisions
            seat = "seat1" if swap else "seat0"
            print(
                f"{name + f' ({seat})':45s} {res.n / dt:9.1f} {decisions / dt:19.1f} {res.mean_length:6.1f} {res.score:7.3f}"
            )
    dt_all = time.perf_counter() - t_all
    print(
        f"  -> {total_games} games, {total_decisions:.0f} total decisions in {dt_all:.2f}s = "
        f"{total_games / dt_all:.1f} games/s, {total_decisions / dt_all:.1f} total decisions/s "
        f"(~{total_decisions / dt_all / 2:.1f} learner-seat decisions/s)"
    )


def mlp_bench(
    width: int = 128, in_dim: int = 512, out_dim: int = 64, reps: int = 20
) -> None:
    print(
        f"\nsynthetic MLP proxy (width={width}, in={in_dim}, out={out_dim}): matmul ceiling only, not the real encoder or feature construction"
    )
    print(f"{'device':6s} {'batch':>7s} {'fwd dec/s':>12s} {'fwd+bwd dec/s':>15s}")
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    for device in devices:
        net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, width), torch.nn.ReLU(),
            torch.nn.Linear(width, width), torch.nn.ReLU(),
            torch.nn.Linear(width, out_dim),
        ).to(device)  # fmt: skip
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        for batch in (512, 8192):
            x = torch.randn(batch, in_dim, device=device)
            for _ in range(3):
                net(x)
            if device == "mps":
                torch.mps.synchronize()
            t0 = time.perf_counter()
            for _ in range(reps):
                with torch.no_grad():
                    net(x)
            if device == "mps":
                torch.mps.synchronize()
            fwd = (time.perf_counter() - t0) / reps

            t0 = time.perf_counter()
            for _ in range(reps):
                opt.zero_grad()
                net(x).sum().backward()
                opt.step()
            if device == "mps":
                torch.mps.synchronize()
            fb = (time.perf_counter() - t0) / reps
            print(f"{device:6s} {batch:7d} {batch / fwd:12,.0f} {batch / fb:15,.0f}")


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    games = int(argv[0]) if argv else 1000
    print(
        f"host: {platform.platform()}, {platform.processor() or platform.machine()}, torch {torch.__version__}"
    )
    print(f"mps available: {torch.backends.mps.is_available()}")
    rs = rules.FULL_UNO
    simulate("engine-only (PlayFirstRandom stand-in)", PlayFirstRandom(rs), games)
    simulate("simulation-with-policy (Heuristic stand-in)", Heuristic(rs), games)
    mlp_bench()
    print(f"\npeak RSS: {peak_rss_mib():.0f} MiB")


if __name__ == "__main__":
    main()
