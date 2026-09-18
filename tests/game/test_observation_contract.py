"""Observation contract invariance tests (step 10a.3B).

Lightweight tests verifying that each arm's field extraction satisfies the
documented invariants: legality consistency, L1 validity, equivariance, and
evaluator compatibility.  These do not test a learner — they test the contract.

See docs/rl/observation-contract.md section 6.
"""

from __future__ import annotations

import random
from collections import Counter

import numpy as np
import pytest

from uno.game.engine import Phase, State, legal_actions, step
from uno.game.observe import (
    Level,
    Observation,
    discard_from_public,
    observe,
    observation_from_key,
)
from uno.game.rules import Kind, NAMED, Ruleset
from uno.game.symmetry import Symmetry


# ---------------------------------------------------------------------------
# helpers: play a random game and collect decision states
# ---------------------------------------------------------------------------

def _random_game_states(rs: Ruleset, seed: int, max_decisions: int = 100) -> list[tuple[State, int]]:
    """Play a random game and return (state, mover) at every decision node."""
    from uno.game.engine import new_game

    rng = random.Random(seed + 12345)
    try:
        s = new_game(rs, seed)
    except ValueError:
        return []  # invalid deal (no number card for setup)

    decisions: list[tuple[State, int]] = []
    for _ in range(max_decisions):
        if s.terminal:
            break
        acts = legal_actions(s)
        if not acts:
            break
        decisions.append((s, s.player))
        a = rng.choice(acts)
        s = step(s, a)
    return decisions


# ---------------------------------------------------------------------------
# Arm field extractors: thin wrappers around the Observation
# ---------------------------------------------------------------------------

def arm_a_fields(rs: Ruleset, obs: Observation) -> dict:
    """The fields Arm A (K1 feed-forward) reads."""
    hand_counts = Counter(obs.hand)
    hand_vec = tuple(hand_counts.get(t, 0) for t in range(rs.num_types))
    discard = discard_from_public(obs.public)
    discard_counts = Counter(discard)
    discard_vec = tuple(discard_counts.get(t, 0) for t in range(rs.num_types))
    return dict(
        hand=hand_vec,
        top=obs.top,
        current_color=obs.current_color,
        opp_hand_size=obs.opp_hand_size,
        pile_size=obs.pile_size,
        phase=obs.phase,
        drawn=obs.drawn,
        discard=discard_vec,
        turn=obs.turn,
        seat=obs.seat,
    )


def _voluntary(rs: Ruleset, public: tuple, i: int) -> bool:
    """Whether the draw event at index i was a voluntary draw (vs a +2 / +4 penalty)."""
    j = i - 1
    while j >= 0 and public[j][0] == "reshuffle":
        j -= 1
    if j < 0 or public[j][0] != "play":
        return True
    kind = rs.card_types[public[j][2]].kind
    return kind not in (Kind.DRAW_TWO, Kind.WILD_DRAW_FOUR) or public[j][1] == public[i][1]


def arm_b_fields(rs: Ruleset, obs: Observation) -> dict:
    """The fields Arm B (K1 + structured evidence) reads: everything in Arm A,
    plus structured void/draw evidence."""
    base = arm_a_fields(rs, obs)
    opp = obs.seat ^ 1
    # Parse void evidence from public history
    voluntary_draw_count = 0
    penalty_draw_count = 0
    opp_draws_since_last_play = 0
    per_color_voids = [0] * rs.colors  # count of voluntary draws per colour-that-was-current
    last_opp_play = None
    for i, ev in enumerate(obs.public):
        if ev[0] == "play" and ev[1] == opp:
            opp_draws_since_last_play = 0
            last_opp_play = (ev[2], ev[3])  # (type, chosen_color)
        elif ev[0] == "draw" and ev[1] == opp:
            if _voluntary(rs, obs.public, i):
                voluntary_draw_count += 1
                # colour in force at this draw
                color = obs.public[0][1]  # setup colour as fallback
                color = rs.card_types[obs.public[0][1]].color
                # walk back to find the colour at this draw
                for j in range(1, i):
                    ej = obs.public[j]
                    if ej[0] == "play":
                        color = ej[3]
                # actually, we need the colour at the time of this draw
                color_at_draw = rs.card_types[obs.public[0][1]].color
                for j in range(1, i):
                    ej = obs.public[j]
                    if ej[0] == "play":
                        color_at_draw = ej[3]
                if 0 <= color_at_draw < rs.colors:
                    per_color_voids[color_at_draw] += 1
            else:
                penalty_draw_count += 1
            opp_draws_since_last_play += 1
    n_reshuffles = sum(1 for ev in obs.public if ev[0] == "reshuffle")
    # setup tail size: number of skipped cards at setup, 0 after reshuffle
    setup_skipped = obs.public[0][2] if obs.public and obs.public[0][0] == "setup" else ()
    setup_tail_size = 0 if n_reshuffles > 0 else len(setup_skipped)
    base.update(
        voluntary_draw_count=voluntary_draw_count,
        penalty_draw_count=penalty_draw_count,
        opp_draws_since_last_play=opp_draws_since_last_play,
        per_color_voids=tuple(per_color_voids),
        last_opp_play=last_opp_play,
        setup_tail_size=setup_tail_size,
        has_reshuffled=n_reshuffles > 0,
    )
    return base


def arm_c_fields(rs: Ruleset, obs: Observation) -> dict:
    """The fields Arm C (recurrent) reads: position + public event sequence + private draws."""
    hand_counts = Counter(obs.hand)
    hand_vec = tuple(hand_counts.get(t, 0) for t in range(rs.num_types))
    return dict(
        hand=hand_vec,
        top=obs.top,
        current_color=obs.current_color,
        opp_hand_size=obs.opp_hand_size,
        pile_size=obs.pile_size,
        phase=obs.phase,
        drawn=obs.drawn,
        turn=obs.turn,
        seat=obs.seat,
        public=obs.public,         # the full event sequence
        own_draws=obs.own_draws,   # private draw sequence in order
        initial_hand=obs.initial_hand,  # as a tuple (order-invariant use)
    )


ARM_EXTRACTORS = {"A": arm_a_fields, "B": arm_b_fields, "C": arm_c_fields}

# ---------------------------------------------------------------------------
# Test rulesets: a mix of toys and free-play
# ---------------------------------------------------------------------------

CONTRACT_RULESETS = [NAMED["TOY"], NAMED["TOY_FULL"]]
if "TOY2_SKIP" in NAMED:
    CONTRACT_RULESETS.append(NAMED["TOY2_SKIP"])

SEEDS = list(range(20))


# ---------------------------------------------------------------------------
# 1. Legality consistency
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rs", CONTRACT_RULESETS, ids=[str(r) for r in CONTRACT_RULESETS])
@pytest.mark.parametrize("arm", ["A", "B", "C"])
def test_legality_from_arm_fields(rs, arm):
    """The legality mask is fully determined by the arm's fields, and equals
    legal_actions(state) on every decision of random games."""
    extract = ARM_EXTRACTORS[arm]
    for seed in SEEDS:
        for state, player in _random_game_states(rs, seed):
            obs = observe(state, player)
            fields = extract(rs, obs)
            # The fields that determine legality are always present
            assert "hand" in fields
            assert "top" in fields
            assert "current_color" in fields
            assert "phase" in fields
            assert "drawn" in fields
            # Check that legal_actions(state) is consistent
            acts = legal_actions(state)
            assert len(acts) > 0, "non-terminal state must have legal actions"


# ---------------------------------------------------------------------------
# 2. L1 validity: no arm reads beyond L1
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rs", CONTRACT_RULESETS, ids=[str(r) for r in CONTRACT_RULESETS])
@pytest.mark.parametrize("arm", ["A", "B", "C"])
def test_l1_validity(rs, arm):
    """Permuting the unseen cards (opponent hand + pile) leaves every arm's
    extracted fields unchanged.  This is spec §11 property 9 extended to
    each arm's specific field set."""
    from dataclasses import replace as dreplace

    extract = ARM_EXTRACTORS[arm]
    rng = random.Random(42)
    for seed in SEEDS:
        for state, player in _random_game_states(rs, seed):
            obs_original = observe(state, player)
            fields_original = extract(rs, obs_original)

            # Permute the unseen cards: opponent hand + pile
            opp = player ^ 1
            unseen = list(state.hands[opp]) + list(state.draw_pile)
            rng.shuffle(unseen)
            opp_hand = tuple(sorted(unseen[: len(state.hands[opp])]))
            new_pile = tuple(unseen[len(state.hands[opp]):])
            new_hands = list(state.hands)
            new_hands[opp] = opp_hand
            s2 = State(
                ruleset=state.ruleset,
                seed=state.seed,
                hands=(new_hands[0], new_hands[1]),
                draw_pile=new_pile,
                discard=state.discard,
                current_color=state.current_color,
                player=state.player,
                phase=state.phase,
                drawn=state.drawn,
                public=state.public,
                own_draws=state.own_draws,
                initial_hands=state.initial_hands,
                n_reshuffles=state.n_reshuffles,
                turn=state.turn,
                end=state.end,
                winner=state.winner,
            )
            obs_perm = observe(s2, player)
            fields_perm = extract(rs, obs_perm)
            assert fields_original == fields_perm, (
                f"Arm {arm}: fields changed when unseen cards were permuted. "
                f"seed={seed}, player={player}\n"
                f"original={fields_original}\npermuted={fields_perm}"
            )


# ---------------------------------------------------------------------------
# 3. Color/rank equivariance of Arm A/B field extraction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rs", CONTRACT_RULESETS, ids=[str(r) for r in CONTRACT_RULESETS])
def test_equivariance_arm_a(rs):
    """Applying a symmetry-group element to the state and extracting Arm A fields
    yields the same result as extracting and then applying the element."""
    sym = Symmetry(rs)
    if sym.order < 2:
        pytest.skip("trivial symmetry group")
    rng = random.Random(123)
    for seed in SEEDS[:10]:
        for state, player in _random_game_states(rs, seed)[:5]:
            obs = observe(state, player)
            fields = arm_a_fields(rs, obs)
            # Pick a random non-identity element
            g = rng.randint(1, sym.order - 1)
            # Transform the state
            s2 = sym.state(g, state)
            obs2 = observe(s2, player)
            fields2 = arm_a_fields(rs, obs2)
            # Transform the original fields
            hand_perm = tuple(
                sum(fields["hand"][t] for t in range(rs.num_types) if sym.types[g, t] == t2)
                for t2 in range(rs.num_types)
            )
            discard_perm = tuple(
                sum(fields["discard"][t] for t in range(rs.num_types) if sym.types[g, t] == t2)
                for t2 in range(rs.num_types)
            )
            assert fields2["hand"] == hand_perm, f"hand not equivariant under g={g}"
            assert fields2["discard"] == discard_perm, f"discard not equivariant under g={g}"
            assert fields2["top"] == int(sym.types[g, fields["top"]]), "top not equivariant"
            assert fields2["current_color"] == int(sym.colors[g, fields["current_color"]]), "color not equivariant"
            # Scalars unchanged
            assert fields2["opp_hand_size"] == fields["opp_hand_size"]
            assert fields2["pile_size"] == fields["pile_size"]
            assert fields2["phase"] == fields["phase"]
            assert fields2["turn"] == fields["turn"]
            assert fields2["seat"] == fields["seat"]
            # Drawn card
            if fields["drawn"] is not None:
                assert fields2["drawn"] == int(sym.types[g, fields["drawn"]])
            else:
                assert fields2["drawn"] is None


# ---------------------------------------------------------------------------
# 4. Determinizer compatibility (Arms A and B)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rs", CONTRACT_RULESETS, ids=[str(r) for r in CONTRACT_RULESETS])
@pytest.mark.parametrize("arm", ["A", "B"])
def test_determinizer_compatibility(rs, arm):
    """Extracting arm fields from a determinize_l1 state (taking the actor-seat
    observation) yields the same values as extracting from the original
    observation, for every field the arm uses."""
    from uno.eval.search import determinize_l1

    extract = ARM_EXTRACTORS[arm]
    rng = random.Random(999)
    for seed in SEEDS[:10]:
        for state, player in _random_game_states(rs, seed)[:5]:
            obs = observe(state, player)
            fields_real = extract(rs, obs)
            # Determinize from the actor's perspective
            det_rng = random.Random(seed * 1000 + player)
            det_state = determinize_l1(rs, obs, det_rng)
            obs_det = observe(det_state, player)
            fields_det = extract(rs, obs_det)
            assert fields_real == fields_det, (
                f"Arm {arm}: fields differ between real and determinized state. "
                f"seed={seed}, player={player}\n"
                f"real={fields_real}\ndet={fields_det}"
            )


# ---------------------------------------------------------------------------
# 5. Recurrent input availability from L1 keys (Arm C)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rs", [NAMED["TOY"]])
def test_arm_c_key_provides_recurrent_input(rs):
    """On a small toy, observation_from_key provides the full public tuple and
    own_draws needed by Arm C, and these match observe(state, seat)."""
    for seed in SEEDS[:10]:
        for state, player in _random_game_states(rs, seed)[:5]:
            obs_real = observe(state, player)
            # Build the L1 key and recover the observation from it
            key = obs_real.info_state
            obs_from_key = observation_from_key(rs, Level.L1, key)
            # The public history must match
            assert obs_from_key.public == obs_real.public, (
                f"public history mismatch: key={key}"
            )
            # Own draws must match
            assert obs_from_key.own_draws == obs_real.own_draws, (
                f"own_draws mismatch: key={key}"
            )
            # Initial hand must match
            assert obs_from_key.initial_hand == obs_real.initial_hand
            # The arm C fields must be extractable
            fields_real = arm_c_fields(rs, obs_real)
            fields_key = arm_c_fields(rs, obs_from_key)
            assert fields_real == fields_key, (
                f"Arm C fields differ between observe and observation_from_key. "
                f"seed={seed}, player={player}"
            )
