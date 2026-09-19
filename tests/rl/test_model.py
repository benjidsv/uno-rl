"""Stage 1 acceptance for the networks: equivariance, orbits, batching, statelessness.

docs/rl/observation-contract.md section 1.4 requires
`policy(g . obs)(g . a) = policy(obs)(a)` for color permutations and for rank
permutations within equal-copy classes, tested on *actual probabilities* with
nontrivial action choices and stabilizers -- not on input shapes or a single rank
swap. Nothing here canonicalizes an observation, so these tests measure the
architecture rather than a wrapper.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.rl.helpers import decision_states, small_actor, small_critic
from uno.game.keys import l1_key
from uno.game.observe import Level, observe
from uno.game.rules import MID_25, TOY, TOY2_SKIP
from uno.game.symmetry import Symmetry
from uno.rl import features as F
from uno.rl.evidence import Evidence
from uno.rl.model import entropy_from_log_probs, masked_log_probs, to_tensors
from uno.rl.policy import ActorPolicy, NeuralAgent

TOL = 1e-5  # float32 log-softmax round-trip; the measured error is ~6e-8


def probs(actor_policy: ActorPolicy, rs, states) -> np.ndarray:
    obs = [observe(s, s.player, Level.L1) for s in states]
    ev = [Evidence.from_public(rs, s.public, s.player ^ 1) for s in states] if actor_policy.spec.has_evidence else None
    lp, _ = actor_policy.distributions(obs, ev)
    return np.exp(lp)


@pytest.mark.parametrize("rs", (TOY, TOY2_SKIP, MID_25))
@pytest.mark.parametrize("arm", (F.ARM_A, F.ARM_B))
def test_probabilities_are_equivariant_over_the_whole_group(rs, arm):
    sym = Symmetry(rs)
    pol = ActorPolicy(F.spec(rs, arm), small_actor(rs, arm, seed=4))
    states = [s for s in decision_states(rs, range(12), max_decisions=14)][:20]
    nontrivial = 0
    worst = 0.0
    for s in states:
        base = probs(pol, rs, [s])[0]
        nontrivial += int((base > 1e-9).sum() > 1)
        mapped_states = [sym.state(g, s) for g in range(sym.order)]
        got = probs(pol, rs, mapped_states)
        for g in range(sym.order):
            want = np.zeros_like(base)
            want[np.asarray(sym.actions[g], dtype=np.int64)] = base
            worst = max(worst, float(np.abs(want - got[g]).max()))
    assert nontrivial > 5, "the test must see decisions with a real choice"
    assert worst < TOL, f"max mapped-probability error {worst}"


@pytest.mark.parametrize("rs", (TOY, TOY2_SKIP))
def test_probabilities_are_uniform_inside_stabilizer_action_orbits(rs):
    """Exact quotient evaluation needs this, and canonicalization alone does not give it
    (AGENTS.md, foundation spec section 18)."""
    sym = Symmetry(rs)
    pol = ActorPolicy(F.spec(rs, F.ARM_B), small_actor(rs, F.ARM_B, seed=1))
    checked = 0
    for s in decision_states(rs, range(40), max_decisions=10):
        key, g = sym.canonical(Level.L1, l1_key(s, s.player))
        stab = sym.stabilizer(Level.L1, key)
        if len(stab) == 1:
            continue
        canonical = sym.state(g, s)
        legal = tuple(np.nonzero(F.legal_mask(rs, observe(canonical, canonical.player, Level.L1)))[0])
        orbits = sym.action_orbits(stab, legal)
        if len(set(orbits)) == len(legal):
            continue
        p = probs(pol, rs, [canonical])[0]
        for orbit in set(orbits):
            members = [a for a, o in zip(legal, orbits) if o == orbit]
            if len(members) > 1:
                checked += 1
                assert max(p[list(members)]) - min(p[list(members)]) < TOL
    assert checked > 0, "no nontrivial stabilizer orbit was reached"


@pytest.mark.parametrize("rs", (TOY, MID_25))
def test_batched_and_single_inference_agree(rs):
    pol = ActorPolicy(F.spec(rs, F.ARM_B), small_actor(rs, F.ARM_B, seed=2))
    states = decision_states(rs, range(6))[:40]
    batch = probs(pol, rs, states)
    for i, s in enumerate(states):
        assert np.allclose(batch[i], probs(pol, rs, [s])[0], atol=TOL)


def test_the_actor_carries_no_state_between_calls():
    """Contract section 2.1: A/B have no mutable episode memory. Re-asking in a
    different order, or twice, must not move a probability."""
    rs = MID_25
    pol = ActorPolicy(F.spec(rs, F.ARM_B), small_actor(rs, F.ARM_B, seed=3))
    states = decision_states(rs, range(4))[:25]
    first = probs(pol, rs, states)
    _ = probs(pol, rs, states[::-1])
    again = probs(pol, rs, states)
    assert np.array_equal(first, again)
    shuffled = probs(pol, rs, states[::-1])[::-1]
    assert np.allclose(first, shuffled, atol=TOL)


def test_masked_probabilities_are_supported_only_on_legal_actions():
    rs = MID_25
    pol = ActorPolicy(F.spec(rs, F.ARM_A), small_actor(rs, F.ARM_A))
    states = decision_states(rs, range(4))[:30]
    obs = [observe(s, s.player, Level.L1) for s in states]
    lp, masks = pol.distributions(obs)
    p = np.exp(lp)
    assert np.allclose(p.sum(1), 1.0, atol=1e-6)
    assert (p[~masks] == 0.0).all()
    assert (p[masks] > 0.0).all()


def test_inverse_cdf_roundoff_never_assigns_mass_to_illegal_actions():
    rs = MID_25
    policy = ActorPolicy(F.spec(rs, F.ARM_A), small_actor(rs, F.ARM_A))
    # Leading, interior and trailing illegal actions exercise u=0 and the largest
    # representable uniform below one. The legal mass is deliberately float32 and
    # sums just below one before the sampler's float64 renormalization.
    logp = np.array(
        [[-np.inf, -0.6881597, -np.inf, -0.6981597, -np.inf]],
        dtype=np.float32,
    )
    chosen = policy.sample(
        np.repeat(logp, 2, axis=0),
        np.array([0.0, np.nextafter(1.0, 0.0)]),
    )
    assert chosen.tolist() == [1, 3]


def test_inverse_cdf_rejects_out_of_range_uniforms():
    rs = MID_25
    policy = ActorPolicy(F.spec(rs, F.ARM_A), small_actor(rs, F.ARM_A))
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        policy.sample(np.array([[0.0]], dtype=np.float32), np.array([1.0]))


def test_neural_agent_reports_the_distribution_it_would_play():
    from uno.game.engine import legal_actions

    rs = TOY2_SKIP
    agent = NeuralAgent(ActorPolicy(F.spec(rs, F.ARM_B), small_actor(rs, F.ARM_B)), "test_actor")
    for s in decision_states(rs, range(5))[:20]:
        legal = legal_actions(s)
        p = agent.policy(observe(s, s.player, Level.L1), legal)
        assert set(p) == set(legal)
        assert sum(p.values()) == pytest.approx(1.0, abs=1e-15)


def test_entropy_ignores_illegal_actions():
    logits = torch.zeros(1, 4)
    mask = torch.tensor([[True, True, False, False]])
    lp = masked_log_probs(logits, mask)
    assert abs(float(entropy_from_log_probs(lp)[0]) - np.log(2)) < 1e-6


def test_masking_refuses_a_row_with_no_legal_action():
    with pytest.raises(ValueError):
        masked_log_probs(torch.zeros(1, 3), torch.zeros(1, 3, dtype=torch.bool))


@pytest.mark.parametrize("rs", (TOY, MID_25))
def test_critic_is_invariant_under_the_group(rs):
    sym = Symmetry(rs)
    sp = F.spec(rs, F.ARM_CRITIC)
    critic = small_critic(rs, seed=5)
    worst = 0.0
    for s in decision_states(rs, range(8), max_decisions=10)[:10]:
        mapped = [sym.state(g, s) for g in range(sym.order)]
        obs = [observe(t, t.player, Level.L2) for t in mapped]
        ev = [Evidence.from_public(rs, t.public, t.player ^ 1) for t in mapped]
        with torch.no_grad():
            v = critic(*to_tensors(F.extract(sp, obs, ev))).numpy()
        worst = max(worst, float(np.abs(v - v[0]).max()))
    assert worst < TOL


def test_actor_and_critic_share_no_parameters():
    rs = TOY
    actor, critic = small_actor(rs, F.ARM_B), small_critic(rs)
    ids = {id(p) for p in actor.parameters()}
    assert not ids & {id(p) for p in critic.parameters()}


def test_arm_a_and_b_share_the_position_encoder_shape_and_differ_only_in_width():
    from uno.rl.model import shapes

    rs = TOY
    a = shapes(F.spec(rs, F.ARM_A), F.spec(rs, F.ARM_CRITIC))
    b = shapes(F.spec(rs, F.ARM_B), F.spec(rs, F.ARM_CRITIC))
    assert (a.tokens, a.colors, a.slots, a.wilds) == (b.tokens, b.colors, b.slots, b.wilds)
    assert a.actor_params < b.actor_params  # reported, not claimed equal
    assert a.critic_params == b.critic_params
