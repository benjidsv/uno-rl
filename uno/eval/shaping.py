"""Potential-based reward shaping as artifacts (step 10a.2, item A4). See docs/spec.md
section 29.

D5 of the step 9 review asked for shaping and got a design; this emits the objects. For a
potential `Phi` over states, shaping replaces the reward of a transition `s -> s'` by

    r'(s, a, s') = r(s, a, s') + gamma * Phi(s') - Phi(s)

which leaves the optimal policy unchanged for any `Phi` (Ng, Harada & Russell 1999). This
project has **no discount** (`gamma = 1`) and one reward, at the terminal, so with the
convention `Phi(terminal) = 0` the shaping terms of a whole game telescope to

    sum over the game's transitions of [Phi(s') - Phi(s)]  =  Phi(terminal) - Phi(s0)  =  -Phi(s0)

**so a shaped return is the plain return minus `Phi(s0)`** -- exactly what AIVAT's deal
correction subtracts when its value function is that same `Phi` (`uno.eval.aivat`:
`deal = nu(first state) - E_deals nu`, the constant being irrelevant to a gradient). The
control variate and the shaping potential are one object, which is why 5.2's training
target lists the deal correction and D5's shaping term as the same entry.

`Phi(terminal) = 0` is therefore not decoration: it is what makes the telescoping identity
hold, and it is the one place a potential differs from the value function it is built from.
`potential` returns the value function with the terminals zeroed; `values` returns it with
the terminal returns kept, which is what the exact-critic arm of the D6 ablation wants. The
two are the same array away from the terminals, and mixing them up would put a `+-1` at
every leaf of a shaping term.

**Three levels, and what each one can see.** The potential is the profile's own value
function read at a level of information:

- **L1** (`Level.L1`): the reach-weighted mean of the value over the mover's own L1
  information set -- a potential the agent could compute itself. `plan.md` argues for this
  one on the grounds that an L2/L3 potential "adds noise the agent cannot condition on";
  for the *deal* term that argument does not apply, because `-Phi(s0)` is subtracted from
  every return and does not depend on the action at all (it is a control variate, not a
  per-step reward).
- **L2** (`None` on a lazy tree): the node's own value, i.e. both hands and the public
  history but never the pile order. This is the finest level AIVAT admits as a value
  function (docs/spec.md section 23), and its deal term **is** AIVAT's deal correction.
- **L3** (`None` on a world tree): the world's value, pile order included. Inadmissible as
  AIVAT's `nu` -- a V that reads the pile order makes every draw deterministic and leaves
  only intractable expectations for the chance corrections -- but perfectly admissible for
  the **deal term alone**, which only needs a mean over deals. It removes strictly more
  luck than L2, and the experiment measures how much.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uno.compiled.aivat import arc_expectation, exact_values, full_reach, l1_set_values
from uno.compiled.tree import CompiledTree, Strategy
from uno.game.observe import Level


def values(tree: CompiledTree, seat0: Strategy, seat1: Strategy, level: Level | None = None) -> np.ndarray:
    """(n,) the profile's seat-0 value at every node, terminals included.

    `level=None`: the node's own value -- an L2-state value on a lazy tree, a world value on
    a world tree. `level=Level.L1`: the reach-weighted mean over the mover's L1 set, which
    is what an L1 agent could know of it (`uno.compiled.aivat.l1_set_values`).

    This is the array the **exact-critic** arm of the D6 ablation uses. For shaping, use
    `potential`, which zeroes the terminals."""
    nu = exact_values(tree, seat0, seat1)
    if level is None:
        return nu
    if level is not Level.L1:
        raise ValueError(f"a set-level potential is L1 only (the node's own value is {level!r}'s: pass level=None)")
    return l1_set_values(tree, seat0, seat1, nu)


def potential(tree: CompiledTree, seat0: Strategy, seat1: Strategy, level: Level | None = None) -> np.ndarray:
    """(n,) the shaping potential `Phi`: `values(...)` with **`Phi(terminal) = 0`**.

    The convention is load-bearing (see the module docstring): with no discount it is what
    makes a game's shaping terms telescope to `-Phi(s0)`, so that the potential and AIVAT's
    deal correction are the same object."""
    phi = values(tree, seat0, seat1, level)
    phi = phi.copy()
    phi[tree.terminals] = 0.0
    return phi


def set_potential(tree: CompiledTree, seat0: Strategy, seat1: Strategy, level: Level = Level.L1) -> np.ndarray:
    """(n_info,) the potential per information set of `level`, for a learner that reads its
    own information set rather than the state: the reach-weighted mean of the profile's value
    over the set, and zero where the profile never reaches it.

    This is the per-set table step 11 plugs in; `potential(..., Level.L1)` is the same
    quantity spread back over the nodes."""
    k = tree.keyings[level]
    nu = potential(tree, seat0, seat1, Level.L1 if level is Level.L1 else None)
    dec = np.nonzero(k.node_info >= 0)[0]
    rho = full_reach(tree, seat0, seat1)
    sets = k.node_info[dec]
    mass = np.bincount(sets, weights=rho[dec], minlength=k.n_info)
    num = np.bincount(sets, weights=rho[dec] * nu[dec], minlength=k.n_info)
    return np.where(mass > 0, num / np.where(mass > 0, mass, 1.0), 0.0)


@dataclass(frozen=True)
class DealTerm:
    """The deal contribution of a potential: `Phi(s0) - E_deals Phi` over the tree's roots.

    `mean` is zero by construction (it is a control variate), and `var` is the luck the
    potential removes from every return -- the quantity 9.5c called the **deal floor**, since
    with an exact value function and every other correction applied it is all the variance
    AIVAT has left."""

    level: str
    mean: float
    var: float
    roots: int

    @property
    def score_var(self) -> float:
        """The same variance in score units, where a point is 0.02 return units."""
        return self.var / 4.0


def deal_term(tree: CompiledTree, phi: np.ndarray) -> DealTerm:
    """`Phi(s0) - E Phi` over the roots, with the tree's own deal (or world) probabilities."""
    roots, p = tree.roots, tree.root_prob
    x = phi[roots] - float(arc_expectation(tree, phi)[0])
    mean = float(p @ x)
    return DealTerm(
        level="node" if tree.roots_level is not Level.L3 else "world",
        mean=mean,
        var=float(p @ (x * x)) - mean * mean,
        roots=len(roots),
    )
