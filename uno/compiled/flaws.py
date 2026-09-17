"""Targets with a **known, localized** flaw, for calibrating an attack (step 10a.3C).

An attack instrument is calibrated by pointing it at targets whose exploitability is already known
exactly. Random and the Heuristic supply two such targets and the Nash strategy a third, but they
share a weakness as calibration cases: Random and the Heuristic are weak *everywhere*, so an
attacker can score well against them without detecting anything in particular, and a mixture like
`(1 - eps) * nash + eps * uniform` is likewise diffusely bad. Neither can tell a sharp attack from
a blunt one, which is the check docs/results/10-review.md section 4 asks for: "include a case where
the attack must detect a known flaw so a vacuous detector cannot pass unnoticed".

So this module grafts **one** bad action class onto an otherwise near-optimal strategy. The result
is near-Nash everywhere except at the decisions of that class, so the exploitability that exists is
concentrated there and an attack only collects it by finding that class. The flaw is exact: the
strategy lives on the compiled keying, so `uno.compiled.solvers.exploitability` gives its true
NashConv and the attack's miss is an exact number rather than an estimate.

**The class must be symmetry-closed**, or the grafted strategy stops being symmetric and the
quotient tree no longer evaluates it correctly (docs/spec.md section 18). "Every `Pass`", "every
wild play", "every action-card play" are closed under colour and rank renaming; "play the red 3"
is not. `graft` asserts orbit-uniformity afterwards rather than trusting the argument.

Measured exactly (`uv run uno lbr exact`), NashConv in score points, on top of a CFR+ Nash whose
own NashConv is ~0.001:

| flaw | `TOY` | `TOY_FULL` | `TOY2_SKIP` |
|---|---|---|---|
| `never_pass` | 1.60 | 1.63 | 1.90 |
| `always_pass` | 12.07 | 24.51 | 26.20 |
| `always_wild_play` | 13.98 | 12.50 | 10.81 |
| `never_wild_play` | 27.47 | 80.10 | 23.33 |
| `never_action_card` | -- | -- | 36.01 |
| `always_action_card` | -- | -- | 4.47 |

Two of those are useful in opposite directions. `never_pass` is **barely exploitable at all**
(0.9-1.6 points): declining to keep a playable drawn card costs almost nothing, so it serves as a
*near-Nash* target -- the case where a correct attack must report a small number and an attack that
reports a large one is broken. `never_wild_play` on `TOY_FULL` is the other extreme at 80.10 points:
with a single wild in a 9-card deck, refusing to play it in free play is close to forfeiting.
"""

from __future__ import annotations

import numpy as np

from uno.compiled.tree import CompiledTree
from uno.game.observe import Level
from uno.game.rules import Kind, Pass, Play, Ruleset

ACTION_CARD_KINDS = (Kind.SKIP, Kind.REVERSE, Kind.DRAW_TWO)

MODES = ("never", "always")


def action_classes(rs: Ruleset) -> dict[str, np.ndarray]:
    """Symmetry-closed classes of action ids, as boolean masks over `rs.actions`.

    Each is invariant under the colour/rank group: renaming colours permutes the actions *within*
    a class and never moves one out of it, which is what keeps a grafted strategy symmetric."""
    acts = rs.actions
    types = rs.card_types
    return {
        "pass": np.array([isinstance(a, Pass) for a in acts]),
        "wild_play": np.array([isinstance(a, Play) and types[a.card].is_wild for a in acts]),
        "action_card": np.array([isinstance(a, Play) and types[a.card].kind in ACTION_CARD_KINDS for a in acts]),
    }


def graft(tree: CompiledTree, sigma: np.ndarray, klass: str, mode: str, level: Level = Level.L1) -> np.ndarray:
    """`sigma` with one action class forced on or off, renormalized per information set.

    - `mode="never"`: the class gets zero mass everywhere, and the rest of the set is rescaled.
      The flaw is "this player never does that".
    - `mode="always"`: at every set where the class is available it takes **all** the mass; other
      sets are untouched. The flaw is "this player always does that when it can".

    Raises when the class carries no mass under `sigma` and `mode="never"`, or is available at no
    set for `mode="always"`: such a graft would return the original strategy while claiming to be a
    flawed one, and a calibration target that is secretly unflawed is worse than none. Also asserts
    the result is still orbit-uniform, i.e. a symmetric strategy (docs/spec.md section 18)."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}: one of {', '.join(MODES)}")
    classes = action_classes(tree.rs)
    if klass not in classes:
        raise ValueError(f"unknown action class {klass!r}: one of {', '.join(classes)}")
    k = tree.keyings[level]
    in_class = classes[klass][k.ia_action]
    if not in_class.any():
        raise ValueError(f"no action of {tree.rs!r} belongs to the class {klass!r}: the flaw would be a no-op")
    available = k.info_sum(np.where(in_class, 1.0, 0.0))[k.ia_info] > 0
    if not available.any():
        raise ValueError(f"the class {klass!r} is available at no information set: the flaw would be a no-op")
    if mode == "never":
        mass = float(k.info_sum(np.where(in_class, sigma, 0.0)).sum())
        if mass <= 1e-12:
            raise ValueError(f"the strategy already puts no mass on {klass!r}: 'never' would be a no-op, so the target would not be flawed")
        out = np.where(in_class, 0.0, sigma)
    else:
        mass_off = float(k.info_sum(np.where(available & ~in_class, sigma, 0.0)).sum())
        if mass_off <= 1e-12:
            raise ValueError(f"the strategy already puts all its mass on {klass!r} wherever available: 'always' would be a no-op")
        out = np.where(available, np.where(in_class, sigma, 0.0), sigma)
    out = k.normalize(out)
    if tree.symmetric and not k.is_symmetric(out, 1e-9):
        raise AssertionError(f"grafting {mode} {klass} broke symmetry: the class is not closed under the colour/rank group (docs/spec.md section 18)")
    return out


def label(klass: str, mode: str) -> str:
    return f"nash_{mode}_{klass}"


def flawed(tree: CompiledTree, sigma: np.ndarray, specs=None, level: Level = Level.L1) -> dict[str, np.ndarray]:
    """`{label: strategy}` for every graft that is not a no-op on this ruleset.

    Classes absent from a ruleset are skipped rather than raising, so one call serves every toy:
    `TOY` has no action cards, and on a forced-play ruleset `Pass` exists only after a draw."""
    out: dict[str, np.ndarray] = {}
    for klass, mode in specs or [(c, m) for c in action_classes(tree.rs) for m in MODES]:
        try:
            out[label(klass, mode)] = graft(tree, sigma, klass, mode, level)
        except ValueError:
            continue  # a no-op graft on this ruleset: not a flawed target, so not offered as one
    return out
