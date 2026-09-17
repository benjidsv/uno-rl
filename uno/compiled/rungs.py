"""The sampled belief rungs as *exact* beliefs on a toy world tree (step 10a.2, item A5b).
See docs/spec.md section 30.

`uno.compiled.pimc` measures two beliefs exactly: `"prior"` (the uniform split, no
inference) and `"posterior"` (the true Bayesian belief given the opponent's strategy). The
rungs that actually ship on full UNO are neither: `"voids"` keeps the worlds a voluntary
draw does not rule out, and `"soft"` weights them by `decay ** excess` instead of rejecting
(docs/spec.md section 26). 9.5f closed those against the exact posterior as *distributions*
over hands (`uv run uno pimc closure`), but never asked what an agent playing them is
**worth**, or how exploitable it is -- and the toys' own warning is that fitting a belief to
one opponent *costs* exploitability (`TOY2_SKIP`: `posterior/response` 3.52 / 6.43 against
`prior/response`'s 1.78 / 2.29). That matters for step 11, because its league is this
league and the exploit an RL agent will find is the league's own.

This builds the rung's belief as a weight per world, which is exactly the `b` that
`uno.compiled.pimc.pimc_response` takes, so the rungs join the existing axis instead of
forming a parallel one and every property that module checks (orbit-uniform policies, value
below BR L1, the fill-in at massless sets) applies unchanged.

**The arithmetic is the shipped rung's, not a copy of it.** The weights come from
`uno.eval.belief`'s own `voids`, `satisfies` and `void_weight`, the same functions the
sampled agent calls. A reimplementation here would make the closure test compare two
transcriptions of one idea rather than the idea against its target -- 9.5f's `closure`
section makes the same point about `_void_weight`.

**The full tree, not the quotient.** A member world of an information set carries its own
canonicalising group element, so an opponent hand decoded from a member's L3 key on a
quotient tree is in *that member's* coordinates rather than the set's -- the trap
`uno.compiled.pimc` documents for arcs and `uv run uno pimc closure` documents for hands.
So `rung_belief` refuses a symmetric tree: the rungs are measured on the full world tree,
where a key is a raw key of the real game. One consequence to carry with the numbers: on a
full tree the PIMC choice is a *pure* per-set action rather than an orbit played uniformly,
which spec section 25 records as a genuinely different policy (0.01-0.44 score points on
`TOY`), so these rows are comparable with each other and not with the published quotient
ones. `prior` is therefore recomputed here rather than quoted.
"""

from __future__ import annotations

import numpy as np

from uno.compiled.tree import CompiledTree
from uno.eval.belief import BELIEFS, satisfies, void_weight, voids
from uno.game.keys import decode_key
from uno.game.observe import Level, observation_from_key

RUNGS = tuple(b for b in BELIEFS if b != "window")  # "window" reads policy probabilities: no exact counterpart


def rung_belief(tree: CompiledTree, seat: int, kind: str = "voids", decay: float = 0.1) -> np.ndarray:
    """(n,) the unnormalised belief a sampled rung holds over the worlds at each of `seat`'s
    decision nodes: the world's prior times what the rung makes of the opponent's hand in it.

    `"prior"` returns the prior itself (the uniform split: `uno.compiled.pimc`'s own
    `"prior"` weighting, recomputed here as the identity check of this function).
    `"voids"` zeroes the worlds a voluntary draw rules out; `"soft"` weights them by
    `decay ** excess`, so `decay = 0` reproduces `"voids"` and `decay = 1` the prior --
    the endpoints this module's experiment checks.

    Normalisation is per information set and cancels in PIMC's argmax, so it is not applied
    (as in `uno.compiled.pimc.belief`)."""
    if kind not in RUNGS:
        raise ValueError(f"unknown rung {kind!r}: one of {RUNGS} ('window' needs the opponent's action probabilities and has no exact counterpart)")
    if tree.symmetric:
        raise ValueError("rung_belief needs the full world tree (compile_tree(rs, (L1, L3), symmetric=False)): on a quotient each member world carries its own canonicalising element, so an opponent hand decoded from its L3 key is in that member's coordinates, not the set's (docs/spec.md section 30)")
    if Level.L3 not in tree.keyings or Level.L1 not in tree.keyings:
        raise ValueError("rung_belief needs both an L1 and an L3 keying: compile_tree(rs, (Level.L1, Level.L3), symmetric=False)")
    rs = tree.rs
    k1, k3 = tree.keyings[Level.L1], tree.keyings[Level.L3]
    prior = tree.reach(np.ones(tree.m), np.ones(tree.m))[:, 2]
    if kind == "prior":
        return prior
    hard = kind == "voids" or decay == 0.0

    inv1 = {i: kb for kb, i in k1.keys.items()}
    inv3 = {i: kb for kb, i in k3.keys.items()}
    opp = seat ^ 1
    b = prior.copy()
    # The constraints are a function of the public history, which every world of an
    # information set shares, so they are derived once per set and not once per world --
    # the same memo `uno.eval.belief.Belief` keeps for the same reason.
    constraints: dict[int, list] = {}
    dec = np.nonzero((tree.player == seat) & (prior > 0))[0]
    for node in dec.tolist():
        i = int(k1.node_info[node])
        vs = constraints.get(i)
        if vs is None:
            obs = observation_from_key(rs, Level.L1, decode_key(Level.L1, inv1[i]))
            vs = constraints[i] = voids(rs, obs.public, opp)
        if not vs:
            continue  # nothing the history rules out: the rung is the prior here
        hand = tuple(sorted(decode_key(Level.L3, inv3[int(k3.node_info[node])])[0][opp]))
        if hard:
            if not satisfies(rs, hand, vs):
                b[node] = 0.0
        else:
            b[node] *= void_weight(rs, hand, vs, decay)
    return b


def rung_label(kind: str, decay: float = 0.1) -> str:
    """The name of the policy a rung produces. A name identifies a policy (docs/spec.md
    section 13), and `soft` is a one-parameter family, so the decay belongs in it."""
    return f"soft({decay:g})" if kind == "soft" else kind
