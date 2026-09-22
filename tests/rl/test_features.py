"""Stage 1 acceptance for the declared inputs (docs/rl/pipeline-plan.md section 13).

Legality against the engine, the information boundary of each arm, batch/scalar
agreement, and that the tests can detect a deliberately broken extractor.
"""

from __future__ import annotations

import random
from dataclasses import replace

import numpy as np
import pytest

from tests.helpers import act, position, tid
from tests.rl.helpers import decision_states, rows
from uno.game.engine import Phase, legal_actions, new_game
from uno.game.observe import Level, observe
from uno.game.rules import FULL_UNO, MID_25, TOY, TOY2_SKIP, TOY_FULL
from uno.rl.env import features as F
from uno.rl.env.evidence import Evidence

RULESETS = (TOY, TOY_FULL, TOY2_SKIP, MID_25, FULL_UNO)


# -- 1. legality ---------------------------------------------------------------


@pytest.mark.parametrize("rs", RULESETS)
def test_production_mask_equals_engine(rs):
    """Contract section 5.1: the mask an actor applies is the engine's legal set."""
    states = decision_states(rs, range(30))
    assert len(states) > 100
    seen_after_draw = seen_multi = 0
    for s in states:
        obs = observe(s, s.player, Level.L1)
        mask = F.legal_mask(rs, obs)
        assert tuple(np.nonzero(mask)[0]) == tuple(sorted(legal_actions(s)))
        seen_after_draw += s.phase is Phase.AFTER_DRAW
        seen_multi += len(legal_actions(s)) > 1
    assert seen_multi > 10
    if rs.draw_then_may_play and not rs.forced_play:
        assert seen_after_draw > 0


def test_mask_covers_playable_drawn_wild_and_pass():
    """A drawn wild offers one action per color plus Pass, and nothing else."""
    rs = FULL_UNO
    s = new_game(rs, 0)
    # drive to an AFTER_DRAW on a wild by constructing it directly
    s = replace(s, phase=Phase.AFTER_DRAW, drawn=tid(rs, "W"), hands=(tuple(sorted(s.hands[0] + (tid(rs, "W"),))), s.hands[1]))
    obs = observe(s, 0, Level.L1)
    mask = F.legal_mask(rs, obs)
    assert tuple(np.nonzero(mask)[0]) == tuple(sorted(legal_actions(s)))
    assert mask[rs.pass_action]
    assert mask.sum() == rs.colors + 1


def test_mask_detects_a_broken_extractor():
    """The legality test is not vacuous: a wrong active color changes the mask."""
    rs = FULL_UNO
    states = [s for s in decision_states(rs, range(10)) if s.phase is Phase.TURN_START]
    obs = [observe(s, s.player, Level.L1) for s in states]
    broken = [replace(o, current_color=(o.current_color + 1) % rs.colors) for o in obs]
    differ = sum(not np.array_equal(F.legal_mask(rs, a), F.legal_mask(rs, b)) for a, b in zip(obs, broken))
    assert differ > 0


# -- 2. the information boundary -----------------------------------------------


@pytest.mark.parametrize("arm", [F.ARM_A, F.ARM_B, F.ARM_C])
def test_unseen_cards_and_seed_do_not_reach_the_actor(arm):
    """Contract section 5.2: change what the actor may not see; features must not move."""
    rs = FULL_UNO
    sp = F.spec(rs, arm)
    states = decision_states(rs, range(6))[:60]
    obs, ev = rows(rs, states)
    base = F.extract(sp, obs, ev if sp.has_evidence else None)
    rng = random.Random(0)
    moved = []
    for o, s in zip(obs, states):
        opp = s.hands[s.player ^ 1]
        pile = list(s.draw_pile)
        rng.shuffle(pile)
        moved.append(replace(o, opp_hand=opp, opp_initial_hand=opp, opp_draws=(), draw_pile=tuple(pile), seed=(o.seed or 0) + 1))
    other = F.extract(sp, moved, ev if sp.has_evidence else None)
    assert np.array_equal(base.tok, other.tok)
    assert np.array_equal(base.col, other.col)
    assert np.array_equal(base.glob, other.glob)


def test_arm_a_ignores_history_beyond_its_declared_fields():
    """A has no evidence block at all, so B's summary cannot reach it."""
    rs = FULL_UNO
    a, b = F.spec(rs, F.ARM_A), F.spec(rs, F.ARM_B)
    states = decision_states(rs, range(8))[:80]
    obs, ev = rows(rs, states)
    empty = [Evidence(rs, o.seat ^ 1) for o in obs]
    fa = F.extract(a, obs, ev)
    fa_empty = F.extract(a, obs, empty)
    assert np.array_equal(fa.tok, fa_empty.tok) and np.array_equal(fa.glob, fa_empty.glob)
    # B does see it: on at least some rows the two differ
    fb, fb_empty = F.extract(b, obs, ev), F.extract(b, obs, empty)
    assert not np.array_equal(fb.tok, fb_empty.tok)
    # and A's channels are exactly B's position prefix
    assert np.array_equal(fa.tok, fb.tok[:, :, : a.tok_width])
    assert np.array_equal(fa.col, fb.col[:, :, : a.col_width])
    assert np.array_equal(fa.glob, fb.glob[:, : a.glob_width])


def test_arm_c_is_b_plus_the_opponent_play_history():
    """B's channels are C's prefix, and C's extra channels carry the play history."""
    rs = FULL_UNO
    b, c = F.spec(rs, F.ARM_B), F.spec(rs, F.ARM_C)
    states = decision_states(rs, range(8))[:80]
    obs, ev = rows(rs, states)
    fb, fc = F.extract(b, obs, ev), F.extract(c, obs, ev)
    assert np.array_equal(fb.tok, fc.tok[:, :, : b.tok_width])
    assert np.array_equal(fb.col, fc.col[:, :, : b.col_width])
    assert np.array_equal(fb.glob, fc.glob)
    assert fc.tok[:, :, F.TOK_HIST_PLAYED:F.TOK_HISTORY_END].any()
    assert fc.col[:, :, F.COL_HIST_WILD:F.COL_HISTORY_END].any()
    # the history reaches C only through the evidence it is handed
    cleared = [e.copy() for e in ev]
    for e in cleared:
        e.hist_color[:] = 0.0
        e.hist_played[:] = 0.0
    fcc = F.extract(c, obs, cleared)
    assert not fcc.tok[:, :, b.tok_width :].any() and not fcc.col[:, :, b.col_width :].any()


def test_arm_c_exists_only_under_the_current_schema():
    with pytest.raises(ValueError):
        F.spec(FULL_UNO, F.ARM_C, version=2)


def test_critic_is_b_plus_the_opponent_hand():
    """Contract section 3: the critic adds L2 hand multiplicities and nothing else."""
    rs = FULL_UNO
    b, c = F.spec(rs, F.ARM_B), F.spec(rs, F.ARM_CRITIC)
    states = decision_states(rs, range(5))[:40]
    obs2 = [observe(s, s.player, Level.L2) for s in states]
    ev = [Evidence.from_public(rs, s.public, s.player ^ 1) for s in states]
    fb = F.extract(b, [observe(s, s.player, Level.L1) for s in states], ev)
    fc = F.extract(c, obs2, ev)
    assert np.array_equal(fb.tok, fc.tok[:, :, : b.tok_width])
    assert np.array_equal(fb.glob, fc.glob)
    opp_sizes = np.array([len(s.hands[s.player ^ 1]) for s in states], dtype=np.float64)
    counted = fc.tok[:, :, F.TOK_OPP_HAND] * np.asarray(rs.deck_counts, dtype=np.float64)[np.concatenate([c.layout.grid_type.reshape(-1), c.layout.wild_type])]
    assert np.allclose(counted.sum(1), opp_sizes, atol=1e-4)


def test_critic_refuses_an_l1_observation():
    rs = TOY
    sp = F.spec(rs, F.ARM_CRITIC)
    s = new_game(rs, 0)
    with pytest.raises(ValueError):
        F.extract(sp, [observe(s, 0, Level.L1)], [Evidence(rs, 1)])


# -- 3. batching and scaling ----------------------------------------------------


@pytest.mark.parametrize("rs", (TOY, MID_25, FULL_UNO))
@pytest.mark.parametrize("arm", [F.ARM_A, F.ARM_B, F.ARM_C])
def test_batch_equals_row_at_a_time(rs, arm):
    sp = F.spec(rs, arm)
    states = decision_states(rs, range(6))[:50]
    obs, ev = rows(rs, states)
    batch = F.extract(sp, obs, ev if sp.has_evidence else None)
    for i, (o, e) in enumerate(zip(obs, ev)):
        one = F.extract_one(sp, o, e if sp.has_evidence else None)
        assert np.array_equal(one.tok[0], batch.tok[i])
        assert np.array_equal(one.col[0], batch.col[i])
        assert np.array_equal(one.glob[0], batch.glob[i])


@pytest.mark.parametrize("rs", RULESETS)
@pytest.mark.parametrize("arm", [F.ARM_B, F.ARM_C])
def test_features_are_finite_and_bounded(rs, arm):
    sp = F.spec(rs, arm)
    obs, ev = rows(rs, decision_states(rs, range(10)))
    f = F.extract(sp, obs, ev)
    f.check_finite()
    for a in (f.tok, f.col, f.glob):
        assert np.abs(a).max() <= 10.0


def test_uncapped_toy_clock_is_zero_not_a_division_by_one():
    """Readiness review: the old encoder divided by 1 where no cap exists."""
    rs = TOY  # max_turns is None
    assert rs.max_turns is None
    sp = F.spec(rs, F.ARM_A)
    obs, _ = rows(rs, decision_states(rs, range(5)))
    f = F.extract(sp, obs)
    assert np.all(f.glob[:, F.GLOB_TURN] == 0.0)


def test_hand_and_discard_channels_are_the_real_multiplicities():
    rs = FULL_UNO
    sp = F.spec(rs, F.ARM_A)
    lay = sp.layout
    order = np.concatenate([lay.grid_type.reshape(-1), lay.wild_type])
    deck = np.asarray(rs.deck_counts, dtype=np.float64)[order]
    states = decision_states(rs, range(4))[:30]
    obs, _ = rows(rs, states)
    f = F.extract(sp, obs)
    for i, s in enumerate(states):
        counts = np.bincount(np.asarray(s.hands[s.player]), minlength=rs.num_types)[order]
        assert np.allclose(f.tok[i, :, F.TOK_HAND] * deck, counts, atol=1e-5)
        disc = np.bincount(np.asarray(s.discard), minlength=rs.num_types)[order]
        assert np.allclose(f.tok[i, :, F.TOK_DISCARD] * deck, disc, atol=1e-5)


# -- 4. determinized queries (contract section 5.5) ------------------------------


def test_a_and_b_agree_on_determinized_worlds():
    """A/B read no opponent private history, so `determinize_l1`'s fabricated one
    cannot move them: the policy in a sampled world equals the policy in the real
    game. That is what makes them queryable inside a search continuation -- and it
    is a property to check, not to assume."""
    import random

    import torch

    from tests.rl.helpers import small_actor
    from uno.eval.search import determinize_l1
    from uno.rl.env.evidence import Evidence
    from uno.rl.algo.policy import ActorPolicy

    rs = MID_25
    states = [s for s in decision_states(rs, range(8)) if s.phase is Phase.TURN_START][:25]
    rng = random.Random(0)
    for arm in (F.ARM_A, F.ARM_B):
        pol = ActorPolicy(F.spec(rs, arm), small_actor(rs, arm, seed=9))
        for s in states:
            real = observe(s, s.player, Level.L1)
            world = determinize_l1(rs, real, rng)
            fake = observe(world, world.player, Level.L1)
            ev = [Evidence.from_public(rs, o.public, o.seat ^ 1) for o in (real, fake)]
            lp, masks = pol.distributions([real, fake], ev if pol.spec.has_evidence else None)
            assert np.array_equal(masks[0], masks[1])
            assert np.allclose(np.exp(lp[0]), np.exp(lp[1]), atol=1e-6)
