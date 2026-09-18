import numpy as np

from uno.agents.baseline import Heuristic, PlayFirstRandom
from uno.eval.belief import Belief
from uno.eval.distill import DistillBuffer, DistillExample, PimcQTeacher, arm_b_features, feature_size, fit_arm_b, soft_q_target
from uno.eval.play import play_game
from uno.eval.search import PimcSearch
from uno.game import rules
from uno.game.engine import legal_actions, new_game
from uno.game.observe import Level, observe
from uno.game.symmetry import Symmetry


def _first_valid(rs, start=0):
    for seed in range(start, start + 100):
        try:
            return seed, new_game(rs, seed)
        except ValueError:
            pass
    raise AssertionError("no valid deal")


def test_arm_b_features_are_fixed_width_and_l1_only():
    rs = rules.TOY
    _, state = _first_valid(rs)
    obs = observe(state, state.player, Level.L1)
    x = arm_b_features(rs, obs)
    assert x.shape == (rs.num_actions, feature_size(rs))
    assert np.isfinite(x).all()
    assert obs.opp_hand is None and obs.draw_pile is None


def test_arm_b_maps_action_probabilities_under_colour_symmetry(tmp_path):
    rs = rules.TOY
    _, state = _first_valid(rs)
    raw, _ = fit_arm_b(
        rs,
        [
            DistillExample(
                arm_b_features(rs, observe(state, state.player, Level.L1)),
                soft_q_target(np.arange(rs.num_actions, dtype=float), legal_actions(state)),
                legal_actions(state),
                0,
            )
        ],
        width=16,
        max_epochs=2,
        patience=1,
        seed=1,
    )
    agent = raw.symmetric()
    path = tmp_path / "arm_b.pt"
    agent.save(str(path))
    agent = agent.load(str(path), rs)
    sym = Symmetry(rs)
    g = min(1, sym.order - 1)
    transformed = sym.state(g, state)
    p = agent.policy(observe(state, state.player, Level.L1), legal_actions(state))
    pg = agent.policy(observe(transformed, transformed.player, Level.L1), legal_actions(transformed))
    mapped = sym.map_policy(g, p)
    assert set(pg) == set(mapped)
    assert all(abs(pg[a] - mapped[a]) < 1e-10 for a in pg)


def test_soft_q_target_masks_illegal_actions():
    q = np.arange(8, dtype=float)
    target = soft_q_target(q, (1, 4, 6), temperature=0.2)
    assert np.isclose(target.sum(), 1.0)
    assert target[[0, 2, 3, 5, 7]].sum() == 0.0
    assert target[6] > target[4] > target[1]


def test_dataset_aggregation_queries_student_reached_states():
    rs = rules.TOY
    seed, state = _first_valid(rs)
    obs = observe(state, state.player, Level.L1)

    class Teacher:
        def q_values(self, obs, legal):
            out = np.full(rs.num_actions, -np.inf)
            for i, action in enumerate(legal):
                out[action] = float(i)
            return out

    raw, _ = fit_arm_b(
        rs,
        [DistillExample(arm_b_features(rs, obs), soft_q_target(Teacher().q_values(obs, legal_actions(state)), legal_actions(state)), legal_actions(state), 0)],
        width=16,
        max_epochs=2,
        patience=1,
        seed=2,
    )
    student = raw.symmetric(name="aggregation_student")
    buffer = DistillBuffer(rs)
    added = buffer.aggregate(student, Teacher(), (), (seed,))
    assert added == 0
    # A real column produces both seats' student trajectories.
    added = buffer.aggregate(student, Teacher(), (PlayFirstRandom(rs),), (seed,))
    assert added > 0
    assert all(r.target.sum() == 0 or np.isclose(r.target.sum(), 1.0) for r in buffer.rows)


def test_batched_teacher_matches_one_state_queries():
    rs = rules.TOY
    trace = play_game(rs, _first_valid(rs)[0], (Heuristic(rs), Heuristic(rs)))
    leaf = (lambda s: float(len(s.hands[0])), lambda s: float(len(s.hands[0])))
    teacher = PimcQTeacher(PimcSearch(rs, leaf, worlds=2, depth=1, name="batch_test", belief=Belief(rs, "prior")))
    records = [(step.obs, tuple(step.probs)) for step in trace.steps[:4]]
    batch = teacher.q_values_batch(records)
    for i, (obs, legal) in enumerate(records):
        single = teacher.q_values(obs, legal)
        assert np.allclose(batch[i][list(legal)], single[list(legal)])
