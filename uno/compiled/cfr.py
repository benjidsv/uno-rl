"""Counterfactual regret minimisation on the compiled tree: Nash L1 (step 8).

See RULES.md section 17 and notes.md ("The oracle levels"). Both seats are
solved at once; the result is one strategy vector over the L1 keying of the
tree (the average strategy, whose exploitability goes to zero), plus the
exploitability log that must go next to every ladder cell using it.

One iteration, for player i (vanilla: both players from the same pass;
alternating: player 1 sees player 0's updated strategy):

1. forward pass for the reach probabilities, backward pass for the values of
   every history and arc under the current strategy profile;
2. counterfactual regret of each information-set action: sum over the set's
   histories of (chance x opponent reach) x (arc value - history value);
3. regret matching: the current strategy is proportional to the positive
   regrets (uniform where none); CFR+ clips the accumulated regrets at zero;
4. the average strategy accumulates own reach x current strategy, weighted by
   the iteration number (linear averaging, CFR+) or by 1 (vanilla).

Vanilla CFR converges as O(1 / sqrt(T)); CFR+ (regret matching+, alternating
updates, linear averaging) has the same bound and is far faster in practice.

Two accelerations on the same passes (step 8.7, notes_toy2.md):

- **Discounting** (DCFR, Brown & Sandholm 2019): after iteration t the
  accumulated positive regrets are multiplied by t^a / (t^a + 1), the
  negative ones by t^b / (t^b + 1), and a contribution to the average
  strategy made at iteration t carries weight t^g. CFR+ is (a = inf, b =
  -inf, g = 1): no discount, negatives clipped, linear averaging. DCFR's
  published choice is (1.5, 0, 2).
- **Prediction** (PCFR+, Farina, Kroer & Sandholm 2021): the current
  strategy is regret matching on (accumulated regrets + the last
  instantaneous regret) instead of the accumulated regrets alone, an
  optimistic step; quadratic averaging (g = 2).

`variant` names the presets; the flags compose (prediction with discounting
is a valid combination). Every variant produces an average strategy whose
exploitability is measured exactly, so a variant that stalls is seen, not
trusted.

Exploitability (`uno.compiled.exploitability`, two best responses) is measured
at the logged iterations; NashConv in score points is
score(BR seat 0) + score(BR seat 1) - 1, and the Nash seat-0 score lies in
[1 - score(BR seat 1), score(BR seat 0)].
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from uno.agents.tabular import CompiledAgent
from uno.compiled.solvers import Exploitability, exploitability
from uno.compiled.tree import CompiledTree
from uno.game.observe import Level

DEFAULT_LOG_AT = (1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 5000)

# name -> (plus, alternating, predictive, alpha, beta, gamma); None = no discount on that sign
VARIANTS: dict[str, tuple[bool, bool, bool, float | None, float | None, float]] = {
    "vanilla": (False, False, False, None, None, 0.0),
    "cfr+": (True, True, False, None, None, 1.0),
    "dcfr": (False, True, False, 1.5, 0.0, 2.0),
    "pcfr+": (True, True, True, None, None, 2.0),
    "dpcfr+": (True, True, True, 1.5, None, 2.0),  # prediction with DCFR's positive-regret discount
}


@dataclass(frozen=True)
class LogEntry:
    t: int
    seconds: float  # since the start of the solve, exploitability computations excluded
    br: tuple[float, float]  # score of the best response to the average strategy, per seat
    value: float  # the average profile's seat-0 score
    nashconv: float  # score points

    @property
    def bracket(self) -> tuple[float, float]:
        return (1.0 - self.br[1], self.br[0])

    def __str__(self) -> str:
        return (
            f"t={self.t:5d} {self.seconds:7.1f}s  BR0={self.br[0]:.4f} BR1={self.br[1]:.4f}  "
            f"nashconv={100 * self.nashconv:.3f} pts  value={self.value:.4f} in [{self.bracket[0]:.4f}, {self.bracket[1]:.4f}]"
        )


@dataclass
class CFRResult:
    level: Level
    iterations: int
    avg: np.ndarray  # normalised average strategy over the keying's ia, both seats
    mass: np.ndarray  # its unnormalised accumulator
    regrets: np.ndarray
    current: np.ndarray  # last current strategy
    log: list[LogEntry]
    seconds: float
    config: dict = field(default_factory=dict)

    @property
    def final(self) -> LogEntry:
        return self.log[-1]

    def agent(self, tree: CompiledTree, name: str = "nash_l1") -> CompiledAgent:
        return CompiledAgent(tree, self.level, self.avg, name=name)

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            avg=self.avg,
            mass=self.mass,
            regrets=self.regrets,
            current=self.current,
            log=np.array([(e.t, e.seconds, e.br[0], e.br[1], e.value, e.nashconv) for e in self.log]),
            meta=np.array([self.level.value, self.iterations, self.seconds]),
            config=np.array([f"{k}={v}" for k, v in self.config.items()]),
        )

    @classmethod
    def load(cls, path: str | Path) -> CFRResult:
        z = np.load(path)
        log = [LogEntry(int(t), s, (b0, b1), v, nc) for t, s, b0, b1, v, nc in z["log"]]
        cfg = {k: _config_value(v) for k, v in (s.split("=", 1) for s in z["config"].tolist())}
        lv, it, sec = z["meta"]
        return cls(Level(int(lv)), int(it), z["avg"], z["mass"], z["regrets"], z["current"], log, float(sec), cfg)


def _config_value(s: str):
    """A saved config value back to None / bool / int / float, else the string."""
    if s in ("None", "True", "False"):
        return {"None": None, "True": True, "False": False}[s]
    for t in (int, float):
        try:
            return t(s)
        except ValueError:
            pass
    return s


def cfr(
    tree: CompiledTree,
    level: Level = Level.L1,
    iterations: int = 1000,
    plus: bool = True,
    alternating: bool = True,
    linear: bool = True,
    log_at=DEFAULT_LOG_AT,
    target: float | None = None,
    verbose: bool = False,
    variant: str | None = None,
    predictive: bool = False,
    alpha: float | None = None,
    beta: float | None = None,
    gamma: float | None = None,
) -> CFRResult:
    """CFR+ by default (`plus`: regret matching+, `alternating`, `linear`
    averaging); all three off is vanilla CFR. `variant` picks a preset from
    `VARIANTS` and overrides the flags; `predictive`, `alpha`, `beta`, `gamma`
    (see the module docstring) refine them, `gamma` defaulting to 1 with
    `linear` and 0 without. Logs exploitability at the iterations in `log_at`
    (the last iteration always) and stops early once NashConv is below
    `target` score points."""
    if variant is not None:
        plus, alternating, predictive, alpha, beta, g = VARIANTS[variant]
        gamma = g if gamma is None else gamma
    if gamma is None:
        gamma = 1.0 if linear else 0.0
    t0 = time.perf_counter()
    k = tree.keyings[level]
    K = k.n_ia
    ia_of = (k.ia_of(0), k.ia_of(1))
    arc_ia = (k.arc_ia[tree.arcs_of[0]], k.arc_ia[tree.arcs_of[1]])
    arc_node = (tree.arc_node[tree.arcs_of[0]], tree.arc_node[tree.arcs_of[1]])
    util = tree.seat_util(0)
    sigma = k.uniform()
    regrets = np.zeros(K)
    mass = np.zeros(K)
    log: list[LogEntry] = []
    paused = 0.0
    t = 0

    def passes(sig):
        w0, w1 = tree.weights(level, sig, 0), tree.weights(level, sig, 1)
        r = tree.reach(w0, w1)
        v, av = tree.values(w0 * w1, util)
        return r, v[:, 0], av[:, 0]

    def update(i: int, r, v, av, weight: float) -> None:
        arcs, nodes, ia = tree.arcs_of[i], arc_node[i], arc_ia[i]
        cf = r[nodes, 2] * r[nodes, 1 - i]
        sign = 1.0 if i == 0 else -1.0
        # on a symmetric tree the regret and the mass are averaged within every action orbit
        # (RULES.md section 18): the strategies stay symmetric, and CFR runs on the game whose
        # actions are the orbits
        mine = ia_of[i]
        inst = k.orbit_mean(np.bincount(ia, weights=cf * (av[arcs] - v[nodes]) * sign, minlength=K))[mine]
        R = regrets[mine] + inst
        if plus:
            R = np.maximum(R, 0.0)
        if alpha is not None or beta is not None:
            # DCFR: discount the accumulated regrets after the iteration's update
            fp = 1.0 if alpha is None else t**alpha / (t**alpha + 1.0)
            fn = 1.0 if beta is None else t**beta / (t**beta + 1.0)
            R = np.where(R > 0, R * fp, R * fn)
        regrets[mine] = R
        mass[mine] += weight * k.orbit_mean(np.bincount(ia, weights=r[nodes, i] * sigma[ia], minlength=K))[mine]
        # the next current strategy: regret matching on the (predicted) regrets
        base = regrets.copy()
        if predictive:
            base[mine] += inst
        sigma[mine] = k.normalize(np.maximum(base, 0.0))[mine]

    log_at = set(log_at) | {iterations}
    for t in range(1, iterations + 1):
        weight = float(t) ** gamma
        if alternating:
            for i in (0, 1):
                update(i, *passes(sigma), weight)
        else:
            r, v, av = passes(sigma)
            update(0, r, v, av, weight)
            update(1, r, v, av, weight)
        if t in log_at:
            p0 = time.perf_counter()
            ex = exploitability(tree, level, k.normalize(mass))
            entry = LogEntry(t, p0 - t0 - paused, (ex.br[0].score, ex.br[1].score), (ex.value + 1) / 2, ex.score_points)
            paused += time.perf_counter() - p0
            log.append(entry)
            if verbose:
                print("  ", entry, flush=True)
            if target is not None and entry.nashconv < target:
                break
    return CFRResult(
        level=level,
        iterations=t,
        avg=k.normalize(mass),
        mass=mass,
        regrets=regrets,
        current=sigma,
        log=log,
        seconds=time.perf_counter() - t0 - paused,
        config=dict(plus=plus, alternating=alternating, predictive=predictive, alpha=alpha, beta=beta, gamma=gamma, variant=variant, iterations=iterations, target=target),
    )


def exploitability_of(tree: CompiledTree, result: CFRResult) -> Exploitability:
    return exploitability(tree, result.level, result.avg)
