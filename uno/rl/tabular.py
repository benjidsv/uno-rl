"""The tabular learner: one logit per action orbit of the compiled toy tree.

docs/rl/pipeline-plan.md section 13 retains the tabular gate -- five training
seeds, each within 0.5 score points of its same-seat fixed-opponent best response
at both seats, within a declared cap -- and stage 5 asks for "tabular and neural
sampled/exact paths".

The point of the tabular arm is to test the **sampled optimizer** with the
representation question removed. It is the same collector, the same PPO objective,
the same critic baseline, the same streams and the same masking convention as the
neural arm; only the actor's parameterization changes, from a network over declared
features to one logit per information-set action orbit.

**Per orbit, not per action.** A symmetric strategy is uniform within every orbit
of a set's stabilizer (foundation spec section 18), so one parameter per orbit is
exactly the symmetric parameterization -- the same one
`uno.compiled.gradient.orbit_gradient` differentiates. Parameterizing per action
would let the learner drift off the quotient and make its exact evaluation invalid.

The tabular actor is a toy instrument: it needs a compiled tree, so it exists for
`TOY`, `TOY_FULL`, `TOY2_SKIP` and their siblings, never for full UNO.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import torch
from torch import nn

from uno.compiled.tree import CompiledTree
from uno.game.keys import encode_key, obs_key
from uno.game.observe import Level, Observation
from uno.game.rules import Ruleset
from uno.game.symmetry import Symmetry
from uno.rl.exact import tree_for
from uno.rl.features import Features, Spec, spec


@cache
def _symmetry(rs: Ruleset) -> Symmetry:
    return Symmetry(rs)


class SetIndex:
    """Observation -> `(information set, group element)`, packed into one integer.

    A compiled quotient tree keys its sets by their **canonical** form, so the set
    a decision belongs to comes with the element `g` that canonicalized it, and the
    actions of the canonical set are the images `g . a` of the real ones. Losing `g`
    would silently relabel colors, so it travels with the set id: the pair is packed
    as `info * group_order + g`, which keeps the rollout's set column a plain
    integer array.
    """

    def __init__(self, tree: CompiledTree) -> None:
        self.rs = tree.rs
        self.keying = tree.keyings[Level.L1]
        self.sym = _symmetry(tree.rs) if tree.symmetric else None
        self.order = self.sym.order if self.sym is not None else 1

    def __call__(self, obs: Observation) -> int:
        key = obs_key(self.rs, obs)
        g = 0
        if self.sym is not None:
            key, g = self.sym.canonical(Level.L1, key)
        got = self.keying.keys.get(encode_key(Level.L1, key))
        if got is None:
            raise KeyError("an information set reached in play is not in the compiled tree's keying")
        return int(got) * self.order + int(g)


class TabularActor(nn.Module):
    """One logit per action orbit, masked and softmaxed exactly like the network."""

    uses_set_ids = True

    def __init__(self, tree: CompiledTree, init_scale: float = 0.0) -> None:
        super().__init__()
        k = tree.keyings[Level.L1]
        self.rs: Ruleset = tree.rs
        self.keying = k
        self.theta = nn.Parameter(torch.zeros(k.n_orbits) if init_scale == 0 else torch.randn(k.n_orbits) * init_scale)
        n_actions = tree.rs.num_actions
        # (n_info, num_actions) -> orbit id, -1 where the action is illegal in that set.
        # A dense map is what lets a whole minibatch of decisions become one gather.
        orbit_of = np.full((k.n_info, n_actions), -1, dtype=np.int64)
        for info in range(k.n_info):
            lo, hi = int(k.info_start[info]), int(k.info_start[info + 1])
            orbit_of[info, k.ia_action[lo:hi]] = k.ia_orbit[lo:hi]
        self.register_buffer("orbit_of", torch.from_numpy(orbit_of), persistent=False)
        sym = _symmetry(tree.rs) if tree.symmetric else None
        self.order = sym.order if sym is not None else 1
        images = sym.actions.astype(np.int64) if sym is not None else np.arange(n_actions, dtype=np.int64)[None]
        self.register_buffer("action_image", torch.from_numpy(images), persistent=False)  # (G, num_actions): a -> g . a

    def logits(self, set_ids: np.ndarray) -> torch.Tensor:
        """Logits in the **caller's** color/rank frame, from canonical parameters.

        `set_ids` packs `(info, g)` (`SetIndex`). The canonical set scores action
        `g . a`, so a row reads its own action `a` through the group element that
        canonicalized it. That is the symmetrization, done by indexing rather than by
        enumerating the group.
        """
        packed = torch.from_numpy(np.asarray(set_ids, dtype=np.int64))
        info, g = packed // self.order, packed % self.order
        mapped = self.action_image[g]  # (n, num_actions), the canonical action of each real one
        table = self.orbit_of[info].gather(1, mapped)  # (n, num_actions) orbit id, -1 if illegal
        # Illegal entries get a finite 0; `masked_log_probs` is what removes them, so
        # there is still exactly one masking convention in the project.
        return torch.where(table >= 0, self.theta[table.clamp(min=0)], torch.zeros((), dtype=self.theta.dtype))

    def legal(self, set_ids: np.ndarray) -> np.ndarray:
        packed = torch.from_numpy(np.asarray(set_ids, dtype=np.int64))
        info, g = packed // self.order, packed % self.order
        return (self.orbit_of[info].gather(1, self.action_image[g]) >= 0).numpy()

    @torch.no_grad()
    def sigma(self) -> np.ndarray:
        """The strategy over the keying, for exact evaluation on the quotient tree."""
        k = self.keying
        theta = self.theta.detach().numpy()[k.ia_orbit]
        out = np.empty(k.n_ia)
        for info in range(k.n_info):
            lo, hi = int(k.info_start[info]), int(k.info_start[info + 1])
            z = theta[lo:hi] - theta[lo:hi].max()
            e = np.exp(z)
            out[lo:hi] = e / e.sum()
        return out


class TabularPolicy:
    """`uno.rl.policy.ActorPolicy`'s interface, backed by `TabularActor`.

    Features are still extracted by the collector (the rollout stores them, and the
    critic needs its own), but this actor ignores them: its parameters are indexed
    by information set. That is the whole point of the arm.
    """

    def __init__(self, tree: CompiledTree, actor: TabularActor, temperature: float = 1.0) -> None:
        self.rs = tree.rs
        self.spec: Spec = spec(tree.rs, "A")  # the rollout still records declared features
        self.actor = actor
        self.temperature = temperature
        self.index = SetIndex(tree)

    @torch.no_grad()
    def log_probs(self, features: Features, masks: np.ndarray, set_ids: np.ndarray | None = None) -> np.ndarray:
        from uno.rl.model import masked_log_probs

        if set_ids is None:
            raise ValueError("the tabular actor needs information-set ids; pass set_id_fn to the collector")
        logits = self.actor.logits(set_ids)
        # The tree's keying and the engine must agree on what is legal; if they do
        # not, the set lookup is wrong and the whole tabular arm is meaningless.
        if not np.array_equal(self.actor.legal(set_ids), masks):
            raise ValueError("the compiled keying's legal actions disagree with the engine's mask")
        return masked_log_probs(logits, torch.from_numpy(masks), self.temperature).numpy()

    def sample(self, log_probs: np.ndarray, rng_values: np.ndarray) -> np.ndarray:
        from uno.rl.policy import ActorPolicy

        return ActorPolicy.sample(self, log_probs, rng_values)


def tabular_for(rs: Ruleset, init_scale: float = 0.0) -> tuple[CompiledTree, TabularActor, TabularPolicy]:
    tree = tree_for(rs)
    actor = TabularActor(tree, init_scale)
    return tree, actor, TabularPolicy(tree, actor)


__all__ = ["SetIndex", "TabularActor", "TabularPolicy", "tabular_for"]
