"""Exact PIMC-infinity on a world tree (step 9.5d). See docs/spec.md section 25.

PIMC (perfect-information Monte Carlo) samples worlds consistent with what the player
knows, solves each with perfect information, and plays the action that scores best on
average. This module computes the **n -> infinity limit exactly**: the average over the
*whole* belief instead of a sample of it, so the toy numbers isolate PIMC's own bias
(strategy fusion and non-locality) from any sampling noise.

Why it is cheap. On a world tree the roots are the worlds and everything below them is
the bare engine, so every non-root arc has exactly one edge of probability 1
(`uno.game.chance.engine_children`) and `reach(...)[:, 2]` at any node is simply that
world's prior. Two consequences:

- the perfect-information value of the position after an action is a *gather* off a
  node-value vector, not a pass of its own (`world_values`);
- neither the belief nor those values depend on PIMC's own policy (own reach is a
  positive constant within an L1 set, which has perfect recall), so the policy is fixed
  by **one global grouped argmax** -- no per-level backward loop, unlike
  `uno.compiled.solvers.best_response`, whose levels exist precisely because a set's
  value there depends on what the responder does deeper down.

At an L1 set `I` the choice is `argmax_a sum_{h in I} b(h) * V_L3(h, a)` with

- `b(h) ~ prior(h) * opponent_reach(h)` (`weighting="posterior"`): the true Bayesian
  belief given the opponent's known strategy -- PIMC *with* inference;
- `b(h) ~ prior(h)` (`weighting="prior"`): worlds weighted by chance alone -- PIMC with
  **no inference**, which is what a real PIMC does when it deals the unseen cards
  uniformly, and the only one that exists on full UNO.

and `V_L3(h, a)` the perfect-information value after `a`, under either continuation:

- `"minimax"`: the world is solved two-sided (classic PIMC: the opponent is assumed to
  play perfectly with perfect information too);
- `"response"`: the world is solved as a perfect-information best response to the
  opponent's actual fixed strategy (`best_response_perfect`, whose `sigma` replays
  through `tree.values` to give exactly these per-arc values).

**Orbits.** On a quotient tree the choice is an *orbit* of actions played uniformly, not
a pure action, exactly as `best_response` does: a pure choice at a set with a non-trivial
stabilizer is not a symmetric strategy and would let the policy condition on hidden
information (docs/spec.md section 18, the step 8.6 bug -- "a best response above the
maximum is not a strategy"). The world tree's L1 keying carries the same orbit partition
as the lazy tree's, so this is the same discipline one step over.

**Unreached sets.** Under `"posterior"` the sets the opponent never lets the player reach
carry no belief mass and get no real choice, exactly as in `best_response` (docs/spec.md
section 16, the step 8.8 fill-in problem): they take their first orbit, or `completion`.
Under `"prior"` every set of the tree has at least one world and the mass is always
positive, so no fill-in arises at all -- which is worth keeping in mind when comparing
the two weightings, since a chunk of `"posterior"`'s sets are then an arbitrary fill-in
rather than a decision (42-84 % of them on the toys, mirroring the best response's own
table in docs/context.md section 5.12).

`value` is the PIMC policy's *actual* return against the opponent, from an ordinary
`evaluate` pass -- never the argmax's own optimistic sum, which is the perfect-information
value PIMC wrongly believes it will get. That gap is the bias this step measures, and
`value <= best_response(tree, Level.L1, opponent, seat).value` must hold: PIMC-infinity is
an L1 policy, and BR L1 is the best of those.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from uno.compiled.solvers import best_response_perfect, evaluate, extremal_values
from uno.compiled.tree import CompiledTree, Keying, Strategy
from uno.game.observe import Level

WEIGHTINGS = ("posterior", "prior")
CONTINUATIONS = ("minimax", "response")


@dataclass
class Pimc:
    """A PIMC-infinity policy on a world tree and what it is actually worth."""

    seat: int
    weighting: str  # "posterior" (inference) | "prior" (none)
    continuation: str  # "minimax" (two-sided) | "response" (to the fixed opponent)
    value: float  # the policy's own return at `seat` against the opponent, in [-1, 1]
    believed: float  # the perfect-information value its own model promises at the root (`world_values`)
    sigma: np.ndarray  # strategy over the tree's L1 keying (both seats' sets; the opponent's hold their first orbit)
    unreached: int  # responder sets with no belief mass: a fill-in, not a choice (always 0 when weighting == "prior")
    ties: int  # reached responder sets where 2+ action orbits tie for best (relative 1e-9)
    seconds: float

    @property
    def score(self) -> float:
        return (self.value + 1.0) / 2.0

    @property
    def fusion_gap(self) -> float:
        """`believed - value`, in return units. **Only an upper reference under
        `continuation="response"`**, where `believed` is BR L3's value: the same player seeing
        the pile as well, so the gap is what PIMC gives up by not really having that
        information (fusion and non-locality, plus the genuine L3 - L1 value of the pile,
        which is why the report puts it next to BR L1 and BR L3 rather than reading it alone).

        Under `"minimax"` it is **not** an upper reference and routinely comes out negative: the
        two-sided value assumes a clairvoyant *opponent* as well, so against a weak fixed
        opponent the policy collects more than that value (on `TOY_FULL` vs Random, believed
        0.634 against an achieved 0.718 in score points). Compare within a continuation, never
        across."""
        return self.believed - self.value


def _profile(seat: int, mine: Strategy, opp: Strategy) -> tuple[Strategy, Strategy]:
    return (mine, opp) if seat == 0 else (opp, mine)


def world_values(tree: CompiledTree, opponent: Strategy, seat: int, continuation: str = "minimax") -> tuple[np.ndarray, float]:
    """((m,) perfect-information value of the position each arc leads to, in `seat`'s return
    units; the same solve's value at the root). The per-arc part is a gather: on a world tree
    every non-root arc has exactly one edge of probability 1, so the arc's value *is* its
    child's node value. The root arc's entry is meaningless (the root is chance) and never
    read: only the seat's own arcs are grouped.

    The root value is what PIMC's own model promises: the game's value if the player really
    did see everything from here on. It is BR L3's value under `"response"` and the two-sided
    minimax value under `"minimax"`, and it is an upper reference the policy does not collect
    (`Pimc.fusion_gap`)."""
    if tree.roots_level is not Level.L3:
        raise ValueError("world_values needs a world tree (compile_tree(..., world=True) or an L3 keying)")
    if continuation == "response":
        # `best_response_perfect`'s sigma is the whole profile per arc, and `tree.values`
        # replays it: `av` is then the post-action value with the responder playing
        # perfectly with full information from there on and the opponent at its own strategy.
        br = best_response_perfect(tree, opponent, seat)
        v, av = tree.values(br.sigma, tree.seat_util(seat))
        return av[:, 0], float(v[0, 0])
    if continuation != "minimax":
        raise ValueError(f"unknown continuation {continuation!r}: one of {CONTINUATIONS}")
    v = extremal_values(tree, tree.seat_util(0), "minimax")  # seat 0 maximises, seat 1 minimises the seat-0 return
    child = tree.edge_child[tree.arc_edge_start[:-1]]  # the single edge of every non-root arc
    sign = 1.0 if seat == 0 else -1.0
    return sign * v[child], sign * float(v[0])


def belief(tree: CompiledTree, opponent: Strategy, seat: int, weighting: str = "posterior") -> np.ndarray:
    """(n,) unnormalised belief over the histories (worlds) at each node: the world's prior,
    times the opponent's reach under `"posterior"`. Normalisation is per information set and
    cancels in the argmax, so it is not applied."""
    if weighting not in WEIGHTINGS:
        raise ValueError(f"unknown weighting {weighting!r}: one of {WEIGHTINGS}")
    opp = seat ^ 1
    w_opp = tree.weights(opponent[0], opponent[1], opp)
    ones = np.ones(tree.m)
    r = tree.reach(w_opp if opp == 0 else ones, w_opp if opp == 1 else ones)
    return r[:, 2] * r[:, opp] if weighting == "posterior" else r[:, 2]


def pimc_response(
    tree: CompiledTree,
    opponent: Strategy,
    seat: int,
    weighting: str = "posterior",
    continuation: str = "minimax",
    completion: np.ndarray | None = None,
    level: Level | Keying = Level.L1,
    b: np.ndarray | None = None,
) -> Pimc:
    """PIMC-infinity for `seat` against a fixed `opponent` on a world tree: the belief-weighted
    perfect-information argmax at every information set, played uniformly within the chosen
    action orbit, then evaluated for what it is actually worth. See the module docstring.

    `b`: an explicit (n,) belief over the worlds at each node, instead of one of the two
    `WEIGHTINGS`. Step 10a.2 item A5b passes the *sampled rungs'* own beliefs here
    (`uno.compiled.rungs.rung_belief`: `voids`, `soft`), so that they are measured on this
    same axis rather than through a parallel implementation; `weighting` is then only the
    label the result carries."""
    t0 = time.perf_counter()
    k = tree.keying_of(level)
    av, believed = world_values(tree, opponent, seat, continuation)
    if b is None:
        b = belief(tree, opponent, seat, weighting)
    elif len(b) != tree.n:
        raise ValueError(f"an explicit belief must have one weight per node ({len(b)} for {tree.n} nodes)")

    mine = np.nonzero(tree.arc_player == seat)[0]
    ia = k.arc_ia[mine]
    # the belief-weighted value of each (set, action), then of each action *orbit* played
    # uniformly: a symmetric strategy is uniform within an orbit (docs/spec.md section 18).
    q = np.bincount(ia, weights=b[tree.arc_node[mine]] * av[mine], minlength=k.n_ia)
    qo = np.bincount(k.ia_orbit, weights=q, minlength=k.n_orbits) / k.orbit_size

    mx = np.maximum.reduceat(qo, k.info_orbit_start[:-1])  # every set has at least one orbit
    at_max = qo == mx[k.orbit_info]
    near = np.abs(qo - mx[k.orbit_info]) <= 1e-9 * np.abs(mx[k.orbit_info])
    first = np.full(k.n_info, k.n_orbits, dtype=np.int64)
    np.minimum.at(first, k.orbit_info[at_max], np.nonzero(at_max)[0])  # ties: the lowest orbit, as best_response
    chosen = np.zeros(k.n_orbits)
    chosen[first[first < k.n_orbits]] = 1.0
    sigma = chosen[k.ia_orbit] / k.orbit_size[k.ia_orbit]

    dec = np.nonzero(tree.player == seat)[0]
    mass = np.bincount(k.node_info[dec], weights=b[dec], minlength=k.n_info)
    is_mine = k.info_player == seat
    unreached = (mass <= 0) & is_mine
    ties = int(((np.bincount(k.orbit_info[near], minlength=k.n_info) > 1) & ~unreached & is_mine).sum())

    # the opponent's sets, and the seat's own sets with no belief mass, get no choice here
    fill = (~is_mine | unreached)[k.ia_info]
    sigma[fill] = (completion if completion is not None else k.first_orbit_uniform())[fill]

    value = float(evaluate(tree, *_profile(seat, (k, sigma), opponent))[seat])
    return Pimc(
        seat=seat,
        weighting=weighting,
        continuation=continuation,
        value=value,
        believed=believed,
        sigma=sigma,
        unreached=int(unreached.sum()),
        ties=ties,
        seconds=time.perf_counter() - t0,
    )


@dataclass
class PimcN:
    """PIMC with a finite number of sampled worlds, as a behavioural policy and its exact value."""

    seat: int
    n: int  # worlds sampled per decision
    realizations: int  # resamples per set used to estimate the choice distribution
    weighting: str
    continuation: str
    value: float  # the exact return of the behavioural policy at `seat` against the opponent
    sigma: np.ndarray  # the behavioural strategy over the keying
    agreement: float  # belief-weighted share of its mass on PIMC-infinity's own choice
    seconds: float

    @property
    def score(self) -> float:
        return (self.value + 1.0) / 2.0


def _members(tree: CompiledTree, k: Keying, seat: int, b: np.ndarray):
    """The consistent worlds of every `seat`-owned information set: the set's member decision
    nodes (grouped, sets in order), their belief weights, and the per-set slice boundaries. A
    member is a history of the set, and on a world tree a history belongs to exactly one world,
    so these *are* the worlds PIMC would sample."""
    dec = np.nonzero(tree.player == seat)[0]
    dec = dec[b[dec] > 0]
    nodes = dec[np.argsort(k.node_info[dec], kind="stable")]
    counts = np.bincount(k.node_info[nodes], minlength=k.n_info)
    return nodes, b[nodes], counts, np.concatenate(([0], np.cumsum(counts)))


def pimc_n(
    tree: CompiledTree,
    opponent: Strategy,
    seat: int,
    n: int,
    realizations: int = 64,
    weighting: str = "posterior",
    continuation: str = "minimax",
    completion: np.ndarray | None = None,
    level: Level | Keying = Level.L1,
    seed: int = 0,
) -> PimcN:
    """PIMC with `n` worlds sampled per decision instead of the whole belief.

    A PIMC-n player re-samples at every decision, and perfect recall means it reaches any one
    information set at most once per game, so what it plays is the **behavioural** policy
    `pi_n(a | I) = P(the argmax over n sampled worlds is a)`. That distribution is estimated by
    resampling each set `realizations` times; the resulting mixed strategy is then scored by one
    ordinary `evaluate`, so the number reported carries no game-playing noise at all -- only the
    Monte Carlo error of `pi_n` itself, which shrinks with `realizations` and is independent of
    how the games would have gone. As `n` grows `pi_n` concentrates on `pimc_response`'s pure
    choice, so this is the finite-sample curve whose limit `pimc_response` computes exactly --
    though not necessarily from below, since that limit maximises the belief-weighted
    perfect-information score and not the true value (see `Pimc.fusion_gap`).

    Worlds are drawn with replacement, proportionally to the belief -- which is what sampling a
    determinization consistent with the player's information does.
    """
    t0 = time.perf_counter()
    k = tree.keying_of(level)
    av, _ = world_values(tree, opponent, seat, continuation)
    b = belief(tree, opponent, seat, weighting)
    nodes, w, counts, starts = _members(tree, k, seat, b)

    live = np.nonzero((k.info_player == seat) & (counts > 0))[0]
    cw = np.cumsum(w)
    base = np.concatenate(([0.0], cw))
    lo, hi = base[starts[live]], base[starts[live + 1]]
    # one row per (set, sampled world, arc of that world's node). The arc's set-action is read
    # from `k.arc_ia`, never from the arc's position under its node: the members of a set are
    # different worlds, each canonicalised by its own group element on a quotient tree, so their
    # arcs need not be listed in the set's own order (the trap `uno.compiled.lookup` documents).
    per = np.repeat(k.info_n[live], n)
    within = np.arange(int(per.sum())) - np.repeat(np.cumsum(per) - per, per)
    rng = np.random.default_rng(seed)
    tally = np.zeros(k.n_orbits)
    for _ in range(realizations):
        u = lo[:, None] + rng.random((len(live), n)) * (hi - lo)[:, None]
        drawn = nodes[np.minimum(np.searchsorted(cw, u.ravel(), side="left"), len(nodes) - 1)]
        arcs = tree.node_arc_start[np.repeat(drawn, per)] + within
        q = np.bincount(k.arc_ia[arcs], weights=av[arcs], minlength=k.n_ia) / n
        qo = np.bincount(k.ia_orbit, weights=q, minlength=k.n_orbits) / k.orbit_size
        mx = np.maximum.reduceat(qo, k.info_orbit_start[:-1])
        at_max = qo == mx[k.orbit_info]
        first = np.full(k.n_info, k.n_orbits, dtype=np.int64)
        np.minimum.at(first, k.orbit_info[at_max], np.nonzero(at_max)[0])
        tally[first[live]] += 1.0

    sigma = (tally / realizations)[k.ia_orbit] / k.orbit_size[k.ia_orbit]
    keep = np.zeros(k.n_info, dtype=bool)
    keep[live] = True
    fill = ~keep[k.ia_info]
    sigma[fill] = (completion if completion is not None else k.first_orbit_uniform())[fill]

    limit = pimc_response(tree, opponent, seat, weighting, continuation, completion, level)
    mass = np.bincount(k.node_info[np.nonzero(tree.player == seat)[0]], weights=b[np.nonzero(tree.player == seat)[0]], minlength=k.n_info)
    share = k.info_sum(sigma * (limit.sigma > 0))[live]
    agreement = float((share * mass[live]).sum() / mass[live].sum()) if mass[live].sum() > 0 else float("nan")
    return PimcN(
        seat=seat,
        n=n,
        realizations=realizations,
        weighting=weighting,
        continuation=continuation,
        value=float(evaluate(tree, *_profile(seat, (k, sigma), opponent))[seat]),
        sigma=sigma,
        agreement=agreement,
        seconds=time.perf_counter() - t0,
    )


def pimc_row(
    tree: CompiledTree,
    opponent: Strategy,
    weighting: str = "posterior",
    continuation: str = "minimax",
    completion: np.ndarray | None = None,
    level: Level | Keying = Level.L1,
    b: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[Strategy, tuple[Pimc, Pimc]]:
    """The ladder's "PIMC(opponent)" row agent: both seats' PIMC-infinity policies as one
    strategy over the keying, and the two results. Mirrors `uno.compiled.strategies.br_row`,
    so the row can be handed to `best_response` for its exploitability.

    `b`: the two seats' explicit beliefs (`pimc_response`'s own `b`). A belief is per seat,
    since what a player can rule out depends on which draws were the *opponent's*."""
    ps = (
        pimc_response(tree, opponent, 0, weighting, continuation, completion, level, None if b is None else b[0]),
        pimc_response(tree, opponent, 1, weighting, continuation, completion, level, None if b is None else b[1]),
    )
    k = tree.keying_of(level)
    sigma = np.where(k.info_player[k.ia_info] == 0, ps[0].sigma, ps[1].sigma)
    return (level, sigma), ps
