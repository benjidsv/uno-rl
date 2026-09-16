"""The `sweep` config grammar and the belief axis it carries (`uno.experiments.pimc`, step 9.5f).

Step 9.5f measured the belief against Random only, and left two questions that both need the
sweep harness to carry a belief of its own: whether the world ladder un-flattens once the
impossible worlds are gone, and whether the soft rung pays against a column whose play choices
actually say something. Wiring a new axis into `_parse_config` is where that goes quietly wrong,
in two specific ways these tests pin:

- the belief must be its **own** field of the meta tuple. `sweep_section`'s ladder groupings
  filter on `cont in ("m", "r")`, so a belief folded into the continuation letter would not merely
  mislabel rows -- it would silently drop every row it touched from every ladder, and the section
  would print a *shorter* table rather than fail;
- the **no-belief path must be untouched**. The prior rows are what the new numbers are read
  against, and they keep step 9.5e's bare agent names on purpose: same name, same world-sampling
  key, so on the same deals they replay that step's games exactly instead of merely resampling
  them. A stray rename would turn a regression check into a fresh sample and nobody would notice.

`TOY_FULL` throughout: free play, 9 cards, and the void constraint is live at 16.5 % of its
decisions (docs/spec.md section 26.2), so the axis has teeth here without full UNO's cost.
"""

from __future__ import annotations

import math
import random
from types import SimpleNamespace

import pytest

from uno.agents.baseline import Heuristic, PlayFirstRandom
from uno.eval.search import PimcSearch
from uno.experiments.pimc import _parse_config
from uno.game import rules
from uno.game.engine import legal_actions, new_game, step
from uno.game.observe import Level, observe


def _leaf(s):
    """A deterministic stand-in for step 9.5c's value network: these tests are about wiring, and
    loading the real nets would cost seconds and a cache hit for no extra coverage."""
    return math.sin(len(s.hands[0]) * 1.7 + len(s.hands[1]) * 2.3 + s.current_color + len(s.draw_pile) * 0.11)


def _parse(rs, token: str):
    nets = (SimpleNamespace(value=_leaf), SimpleNamespace(value=_leaf))
    return _parse_config(rs, nets, Heuristic(rs), PlayFirstRandom(rs), token)


def _decisions(rs, agent_seeds, limit=300):
    """(observation, legal actions) at every real choice of random games."""
    n = 0
    for seed in agent_seeds:
        s = new_game(rs, seed)
        rng = random.Random(seed)
        while not s.terminal and n < limit:
            legal = legal_actions(s)
            obs = observe(s, s.player, Level.L1)
            if len(legal) > 1:
                yield obs, legal
                n += 1
            s = step(s, rng.choice(legal))


@pytest.mark.parametrize(
    "token, expected",
    [
        ("s32x2", ("search", 32, 2, "m", "prior")),
        ("s32x2r", ("search", 32, 2, "r", "prior")),
        ("s32x2v", ("search", 32, 2, "m", "voids")),
        ("s32x2rv", ("search", 32, 2, "r", "voids")),
        ("s8x3rw", ("search", 8, 3, "r", "window")),
        ("r8x4", ("rollout", 8, 4, "m", "prior")),
    ],
)
def test_the_belief_is_a_separate_meta_field_from_the_continuation(token, expected):
    """The continuation field may only ever be `m` or `r`, whatever the belief, because that is
    what the ladder groupings filter on."""
    _label, _agent, meta = _parse(rules.TOY_FULL, token)
    assert meta == expected
    assert meta[3] in ("m", "r")


def test_the_no_belief_rows_keep_step_9_5e_names_and_carry_no_belief():
    """The regression that protects the baseline: `s32x2r` must still be the agent step 9.5e
    measured, by name (hence by world-sampling key) and by having no belief at all."""
    rs = rules.TOY_FULL
    assert _parse(rs, "s32x2r")[1].name == "pimc_resp_random"
    assert _parse(rs, "s32x2r")[1].belief is None
    assert _parse(rs, "s32x2")[1].name == "pimc"
    assert _parse(rs, "s32x2")[1].belief is None


def test_every_config_of_one_sweep_carries_a_distinct_name():
    """A name identifies a policy (docs/context.md section 3): two agents sharing one would share
    an action rng stream, and `play_pairs` raises on it. The belief makes a different policy, so
    it has to reach the name."""
    rs = rules.TOY_FULL
    tokens = ("s32x2", "s32x2r", "s32x2v", "s32x2rv", "s8x2rw", "r8x4")
    names = [_parse(rs, t)[1].name for t in tokens]
    assert len(set(names)) == len(names), f"two sweep configs share a name: {names}"


@pytest.mark.parametrize("token, kind", [("s8x2rv", "voids"), ("s8x2rw", "window")])
def test_the_belief_is_built_against_the_sweep_column(token, kind):
    """`window` scores a sampled hand by the opponent's *real* action probabilities, so it must be
    handed the column the sweep actually plays, which is Random."""
    agent = _parse(rules.TOY_FULL, token)[1]
    assert agent.belief is not None and agent.belief.kind == kind
    assert agent.belief.opponent.name == "random"


def test_the_rollout_player_refuses_a_belief():
    """`PimcRollout` determinizes without one, so a config asking for it is a silent no-op rather
    than the thing the token says."""
    with pytest.raises(SystemExit):
        _parse(rules.TOY_FULL, "r8x4v")


def test_a_prior_config_still_plays_exactly_the_agent_step_9_5e_measured():
    """The load-bearing regression. The new numbers are read against the prior rows, so if this
    rewiring perturbed the no-belief path -- an extra rng draw, a changed name, a different leaf
    -- every comparison in the step would be against a baseline that quietly moved."""
    rs = rules.TOY_FULL
    rnd = PlayFirstRandom(rs)
    ref = PimcSearch(rs, (_leaf, _leaf), worlds=4, depth=2, name="pimc_resp_random", opponent=rnd)
    got = _parse(rs, "s4x2r")[1]
    n = 0
    for obs, legal in _decisions(rs, range(12)):
        assert got.policy(obs, legal) == ref.policy(obs, legal)
        n += 1
    assert n > 40


def test_the_belief_axis_actually_changes_the_policy():
    """Non-vacuity, the step 9.5e lesson: every test above would pass with `PimcSearch` ignoring
    its `belief` argument entirely. The ladder is only worth the CPU if the axis moves the agent,
    so pin that it does."""
    rs = rules.TOY_FULL
    prior = _parse(rs, "s8x2r")[1]
    voids = _parse(rs, "s8x2rv")[1]
    differ = seen = 0
    for obs, legal in _decisions(rs, range(60), limit=400):
        seen += 1
        differ += prior.policy(obs, legal) != voids.policy(obs, legal)
    assert seen > 50
    assert differ > 0, "the voids belief never changed a choice: the sweep's belief axis is a no-op"
