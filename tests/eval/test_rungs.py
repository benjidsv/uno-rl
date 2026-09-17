"""The belief rungs as exact beliefs on a world tree (step 10a.2, item A5b): the family's
own identities, on the 7-card SMALL ruleset.

The measurements are `uv run uno rungs`; these are the cheap invariants."""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest

from tests.strategies import SMALL
from uno.agents.baseline import PlayFirstRandom
from uno.compiled.pimc import belief as exact_belief
from uno.compiled.pimc import pimc_response
from uno.compiled.rungs import rung_belief, rung_label
from uno.compiled.strategies import tabulate
from uno.compiled.tree import compile_tree
from uno.game.observe import Level

L1, L3 = Level.L1, Level.L3


@cache
def setup():
    # the FULL world tree: on a quotient each member world carries its own canonicalising
    # element, so an opponent hand decoded from its L3 key is in that member's coordinates
    tree = compile_tree(SMALL, (L1, L3), symmetric=False)
    opp = (L1, tabulate(tree, L1, PlayFirstRandom(SMALL)))
    return tree, opp


def test_the_prior_rung_is_the_trees_own_prior():
    tree, opp = setup()
    for seat in (0, 1):
        assert np.array_equal(rung_belief(tree, seat, "prior"), exact_belief(tree, opp, seat, "prior"))


def test_soft_reproduces_the_family_endpoints():
    """`decay = 0` is `voids` (a hard filter) and `decay = 1` is the prior: the one-parameter
    family spans the two rungs, which is the property `uno.eval.belief` claims for it."""
    tree, _ = setup()
    for seat in (0, 1):
        assert np.array_equal(rung_belief(tree, seat, "soft", 0.0), rung_belief(tree, seat, "voids"))
        assert np.array_equal(rung_belief(tree, seat, "soft", 1.0), rung_belief(tree, seat, "prior"))


def test_voids_only_ever_removes_mass():
    """The rung is a necessary condition, so it can zero a world but never favour one."""
    tree, _ = setup()
    for seat in (0, 1):
        prior = rung_belief(tree, seat, "prior")
        voids = rung_belief(tree, seat, "voids")
        soft = rung_belief(tree, seat, "soft", 0.1)
        assert np.all(voids <= prior + 1e-15)
        assert np.all(soft <= prior + 1e-15)
        # and soft never puts zero where the prior was positive: that is its whole point
        assert not np.any((soft == 0) & (prior > 0))


def test_the_rungs_produce_playable_policies_below_br_l1():
    tree, opp = setup()
    k = tree.keyings[L1]
    for kind, decay in (("prior", 0.1), ("voids", 0.1), ("soft", 0.1)):
        for seat in (0, 1):
            p = pimc_response(tree, opp, seat, rung_label(kind, decay), "response", b=rung_belief(tree, seat, kind, decay))
            assert np.allclose(k.info_sum(p.sigma), 1.0)
            assert p.value <= 1.0 + 1e-12


def test_a_quotient_tree_is_refused():
    """The trap the module exists to avoid, as a test rather than a comment."""
    quotient = compile_tree(SMALL, (L1, L3), symmetric=True)
    with pytest.raises(ValueError, match="full world tree"):
        rung_belief(quotient, 0, "voids")


def test_an_unknown_rung_and_the_window_rung_are_refused():
    tree, _ = setup()
    with pytest.raises(ValueError, match="unknown rung"):
        rung_belief(tree, 0, "window")  # reads the opponent's action probabilities


def test_the_label_carries_the_decay():
    """A name identifies a policy, and `soft` is a one-parameter family."""
    assert rung_label("soft", 0.1) == "soft(0.1)"
    assert rung_label("soft", 0.0) == "soft(0)"
    assert rung_label("voids") == "voids"
