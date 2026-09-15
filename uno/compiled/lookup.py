"""State -> compiled-tree node/arc lookup, step 9.5a. Both AIVAT's control-variate
value function (step 9.5b) and the sampled cross-check of the exact edge
decomposition (`uno.compiled.edge`, the "sampled D2 check") need this: given a
*played* game's state and, at a decision, the action taken, find the corresponding
node and arc of a compiled tree, so `CompiledTree.values`' node/arc values (`V`,
`Q`) can be read off directly instead of recomputed.

Built on the **L2** keying, the finest one available: `docs/spec.md` section 16 --
"on the full `TOY` tree the L2 keying has exactly one node per set... the L2 key
of a played state *is* its node". On a symmetric (quotient) tree, up to `|G|`
nodes share one canonical L2 set, with equal values for a symmetric profile
(canonical-frame values are invariant under the group by construction); an
arbitrary one, here the tree's lowest-numbered node of that set, stands in for
all of them. `build` needs an L2 keying on the tree: cached for `TOY`, `TOY_FULL`,
`TOY2` and `TOY2_SKIP`; skip `TOY2_FULL`, whose L2 keying alone is ~3 GB
(docs/plan.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uno.compiled.tree import CompiledTree
from uno.game.engine import State
from uno.game.keys import encode_key, l2_key
from uno.game.observe import Level
from uno.game.symmetry import Symmetry


@dataclass
class NodeIndex:
    tree: CompiledTree
    level: Level
    first_node: np.ndarray  # (n_info,) int64: the tree's own lowest-index node per set
    sym: Symmetry | None  # None on a non-symmetric tree

    def _lookup(self, state: State) -> tuple[int, int]:
        """(the set's canonical representative node, the group element `g` with
        `g . state's own key == that representative's key`; `g = 0` when the tree
        is not symmetric)."""
        key = l2_key(state, state.player)
        g = 0
        if self.sym is not None:
            key, g = self.sym.canonical(self.level, key)
        k = self.tree.keyings[self.level]
        i = k.keys.get(encode_key(self.level, key))
        if i is None:
            # ticket-draw games are found: an L1 / L2 key never holds the pile order (step 9.5b)
            raise KeyError("state's information set is not in this tree (a different ruleset, or a state the tree's chance model does not reach)")
        return int(self.first_node[i]), g

    def node(self, state: State) -> int:
        """The compiled tree's representative node for `state`'s own L2 information
        set (i.e. `state.player`'s own set)."""
        node, _ = self._lookup(state)
        return node

    def arc(self, state: State, action: int) -> int:
        """The arc at `self.node(state)` representing `action`.

        `tree.arc_action` is in *each node's own* frame (compile_tree canonicalises
        a node's key, not the node's underlying State, so a symmetric tree's nodes
        below a canonical deal are real, un-relabelled states): comparing it
        directly against `action` mapped through this lookup's own `g` would only
        be correct at the node whose key search happened to pick that `g`, not in
        general at `self.node(state)`'s representative, which is some other node of
        the same set with its own, different canonicalising element. `k.arc_ia` /
        `k.ia_action` are what stays consistent across every node of a set (they
        are `compile_tree`'s own per-node-canonicalised action ids, used to build
        the keying in the first place), so match on those instead."""
        return self.arcs(state, (action,))[1][action]

    def arcs(self, state: State, actions) -> tuple[int, dict[int, int]]:
        """`(self.node(state), {action: self.arc(state, action)})` with one key lookup
        (step 9.5b: AIVAT reads every action of a decision's support)."""
        node, g = self._lookup(state)
        k = self.tree.keyings[self.level]
        a0, a1 = int(self.tree.node_arc_start[node]), int(self.tree.node_arc_start[node + 1])
        node_canonical_actions = k.ia_action[k.arc_ia[a0:a1]].tolist()  # consistent across every node of a set, unlike tree.arc_action
        out = {}
        for action in actions:
            canonical_action = action if self.sym is None else int(self.sym.actions[g, action])
            try:
                out[action] = a0 + node_canonical_actions.index(canonical_action)
            except ValueError:
                raise KeyError("action is not legal at this node's canonical representative") from None
        return node, out


def build(tree: CompiledTree, level: Level = Level.L2) -> NodeIndex:
    if level not in tree.keyings:
        raise ValueError(f"tree has no {level.name} keying: compile_tree(rs, (Level.L1, Level.{level.name}), ...)")
    k = tree.keyings[level]
    first_node = np.full(k.n_info, tree.n, dtype=np.int64)  # sentinel larger than any real node id, as compile_tree's own "first" does
    valid = k.node_info >= 0
    np.minimum.at(first_node, k.node_info[valid], np.nonzero(valid)[0])
    sym = Symmetry(tree.rs) if tree.symmetric else None
    return NodeIndex(tree=tree, level=level, first_node=first_node, sym=sym)
