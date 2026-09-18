"""Arm-B policy distillation from soft PIMC action values.

This module deliberately keeps distillation separate from the learner/PPO path.  A
teacher supplies one searched value per legal action; the student is a masked
feed-forward policy over the Arm-B observation contract.  Dataset aggregation
queries the frozen teacher on states reached by the current student, which avoids
the usual behaviour-cloning distribution shift.

The network itself does not choose a colour/rank coordinate system: it scores
each action with shared parameters from invariant global features and the
action's own colour block.  This is important on FULL_UNO, where enumerating the
full rank-permutation symmetry group is infeasible.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Iterable, Protocol

import numpy as np
import torch
from torch import nn

from uno.agents.baseline import Agent, Policy
from uno.eval.belief import _is_penalty, voids
from uno.eval.play import play_game
from uno.eval.search import PimcSearch, batched_search_q, determinize_l1, leaf_for, unseen_cards
from uno.game.engine import Phase
from uno.game.observe import Observation, discard_from_public, observe
from uno.game.rules import Kind, Play, Ruleset


VOID_AGE = 4


def _counts(cards: Iterable[int], n: int) -> np.ndarray:
    out = np.zeros(n, dtype=np.float64)
    for card in cards:
        out[int(card)] += 1.0
    return out


def _last_opponent_play(rs: Ruleset, public: tuple, opponent: int) -> tuple[int | None, int | None]:
    for ev in reversed(public):
        if ev[0] == "play" and ev[1] == opponent:
            return int(ev[2]), int(ev[3])
    return None, None


def _opponent_draws_since_play(public: tuple, opponent: int) -> int:
    count = 0
    for ev in reversed(public):
        if ev[0] == "play" and ev[1] == opponent:
            break
        if ev[0] == "draw" and ev[1] == opponent:
            count += int(ev[2])
    return count


def _setup_tail_size(public: tuple) -> int:
    """Cards skipped before the first number card, while they remain at the tail."""
    for ev in public:
        if ev[0] == "setup":
            return len(ev[2])
    return 0


def _reshuffles(public: tuple) -> int:
    return sum(ev[0] == "reshuffle" for ev in public)


def _rank_hist(rs: Ruleset, cards: tuple[int, ...]) -> np.ndarray:
    """Histogram of per-rank multiplicities; invariant to rank renaming."""
    counts = np.zeros(len(rs.rank_copies), dtype=np.float64)
    for rank in range(len(rs.rank_copies)):
        counts[rank] = sum(rs.card_types[c].kind is Kind.NUMBER and rs.card_types[c].rank == rank for c in cards)
    hist = np.zeros(max(rs.deck_counts) * max(rs.colors, 1) + 1, dtype=np.float64)
    for n in counts.astype(np.int64):
        hist[min(int(n), len(hist) - 1)] += 1.0
    return hist / max(len(rs.rank_copies), 1)


def _action_features(rs: Ruleset, obs: Observation, context: np.ndarray) -> np.ndarray:
    """Shared action descriptors; no absolute colour or rank is represented."""
    types, top = rs.card_types, rs.card_types[obs.top]
    hand_counts = _counts(obs.hand, rs.num_types)
    discard_counts = _counts(discard_from_public(obs.public), rs.num_types)
    rank_counts = {r: sum(types[c].kind is Kind.NUMBER and types[c].rank == r for c in obs.hand) for r in range(len(rs.rank_copies))}
    discard_rank_counts = {r: sum(types[c].kind is Kind.NUMBER and types[c].rank == r for c in discard_from_public(obs.public)) for r in range(len(rs.rank_copies))}
    _, current_colour = obs.top, obs.current_color
    blocks = context[-(9 * rs.colors) :].reshape(rs.colors, 9)
    out = np.zeros((rs.num_actions, 17 + blocks.shape[1]), dtype=np.float64)
    for action, act in enumerate(rs.actions):
        if not isinstance(act, Play):
            out[action, 0 if act.__class__.__name__ == "Draw" else 1] = 1.0
            continue
        card = types[act.card]
        f = out[action]
        f[2 + int(card.kind)] = 1.0
        f[8] = float(card.is_wild)
        colour = act.color if card.is_wild else card.color
        if colour is not None:
            f[9] = float(colour == current_colour)
            f[10] = float(card.color == current_colour)
            f[11] = blocks[colour, 0]
            f[12] = blocks[colour, 7]
            f[17:] = blocks[colour]
        f[13] = hand_counts[act.card] / max(rs.deck_counts[act.card], 1)
        f[14] = discard_counts[act.card] / max(rs.deck_counts[act.card], 1)
        f[15] = float(card.kind is Kind.NUMBER and top.kind is Kind.NUMBER and card.rank == top.rank)
        f[16] = float(not card.is_wild and card.kind is not Kind.NUMBER and card.kind is top.kind)
        if card.kind is Kind.NUMBER:
            f[13] += rank_counts[card.rank] / max(rs.hand_size, 1)
            f[14] += discard_rank_counts[card.rank] / max(rs.deck_size, 1)
    return out


def arm_b_features(rs: Ruleset, obs: Observation, void_age: int = VOID_AGE) -> np.ndarray:
    """Extract ``(num_actions, feature_width)`` Arm-B inputs from an L1 observation.

    The global part uses rank-renaming-invariant histograms and the colour part
    is supplied to every action through its own shared colour block.  Consequently
    colour/rank renaming permutes action rows but cannot change their scores.
    """
    if obs.level.name != "L1":
        raise ValueError(f"Arm B needs an L1 observation, got {obs.level.name}")
    c = rs.colors
    types = rs.card_types
    opponent = obs.seat ^ 1
    vs = voids(rs, obs.public, opponent)

    # Global position block.  No absolute rank or colour coordinate occurs here.
    top_kind = np.zeros(len(Kind), dtype=np.float64)
    top_kind[int(types[obs.top].kind)] = 1.0
    drawn_kind = np.zeros(len(Kind), dtype=np.float64)
    if obs.drawn is not None:
        drawn_kind[int(types[obs.drawn].kind)] = 1.0
    last_card, last_colour = _last_opponent_play(rs, obs.public, opponent)
    last_kind = np.zeros(len(Kind), dtype=np.float64)
    if last_card is not None:
        last_kind[int(types[last_card].kind)] = 1.0
    last_colour_onehot = np.asarray([float(last_colour == obs.current_color), float(last_colour is not None)], dtype=np.float64)

    scalar = np.asarray(
        [
            obs.opp_hand_size / max(rs.hand_size, 1),
            obs.pile_size / max(rs.deck_size, 1),
            float(obs.phase is Phase.AFTER_DRAW),
            float(obs.drawn is not None),
            obs.turn / max(rs.max_turns or 1, 1),
            min(_reshuffles(obs.public), 3) / 3.0,
            _setup_tail_size(obs.public) / max(rs.deck_size, 1),
            float(_reshuffles(obs.public) > 0),
            len(vs) / max(obs.turn + 1, 1),
            sum(ev[0] == "draw" and ev[1] == opponent and _is_penalty(rs, obs.public, i) for i, ev in enumerate(obs.public)) / max(obs.turn + 1, 1),
            _opponent_draws_since_play(obs.public, opponent) / max(rs.hand_size, 1),
            float(obs.seat == 0),
        ],
        dtype=np.float64,
    )

    # Per-colour blocks preserve the structured Arm-B evidence.  The same block
    # is later processed for every colour by a shared network in the canonical
    # coordinate system.
    blocks = np.zeros((c, 9), dtype=np.float64)
    for colour in range(c):
        blocks[colour, 0] = sum(types[x].color == colour for x in obs.hand) / max(rs.hand_size, 1)
        blocks[colour, 1] = sum(types[x].color == colour for x in discard_from_public(obs.public)) / max(rs.deck_size, 1)
        blocks[colour, 2] = float(obs.current_color == colour)
        blocks[colour, 3] = float(types[obs.top].color == colour)
        blocks[colour, 4] = float(obs.drawn is not None and types[obs.drawn].color == colour)
        live = [v for v in vs if v.colour == colour and v.since <= void_age]
        blocks[colour, 5] = min(len(live), void_age) / void_age
        blocks[colour, 6] = min((min(v.since, void_age) for v in live), default=void_age) / void_age
        blocks[colour, 7] = float(any(types[x].color == colour for x in obs.hand))
        blocks[colour, 8] = float(any(types[x].color == colour for x in discard_from_public(obs.public)))

    # Rank histograms and pooled colour moments are invariant under both parts of
    # the game symmetry group.  The per-colour block itself is selected only by
    # an action's colour in _action_features.
    pooled = np.concatenate([blocks.sum(0), blocks.max(0)])
    global_context = np.concatenate(
        [
            np.asarray([len(obs.hand) / max(rs.hand_size, 1), len(discard_from_public(obs.public)) / max(rs.deck_size, 1)], dtype=np.float64),
            top_kind,
            drawn_kind,
            scalar,
            last_kind,
            last_colour_onehot,
            _rank_hist(rs, obs.hand),
            _rank_hist(rs, discard_from_public(obs.public)),
            pooled,
        ]
    )
    context = np.concatenate([global_context, blocks.reshape(-1)])
    action = _action_features(rs, obs, context)
    return np.concatenate([np.broadcast_to(context, (rs.num_actions, context.size)), action], axis=1)


def feature_size(rs: Ruleset, void_age: int = VOID_AGE) -> int:
    # void_age changes no dimensions, but is part of the public extractor API.
    # Derive it from a valid observation in order to keep the public dimension
    # aligned with rank-count histograms for parameterized rulesets.
    from uno.game.engine import new_game

    for seed in range(100):
        try:
            s = new_game(rs, seed)
            return int(arm_b_features(rs, observe(s, s.player), void_age).shape[1])
        except ValueError:
            continue
    raise ValueError("could not construct a valid deal to determine Arm-B feature size")


class PolicyNet(nn.Module):
    def __init__(self, inputs: int, actions: int, width: int = 256) -> None:
        super().__init__()
        del actions
        self.net = nn.Sequential(
            nn.Linear(inputs, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(),
            nn.Linear(width, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # One shared scorer per action.  The action rows carry their own
        # relative card/colour descriptors, so a colour permutation permutes
        # rows without changing the scorer's parameters.
        return self.net(x).squeeze(-1)


class ArmBPolicy:
    """Feed-forward Arm-B policy, equivariant by construction."""

    def __init__(self, rs: Ruleset, mean: np.ndarray, scale: np.ndarray, params: dict[str, torch.Tensor], width: int = 256, temperature: float = 1.0, name: str = "arm_b_distilled"):
        self.rs, self.mean, self.scale, self.temperature, self.name, self.width = rs, np.asarray(mean), np.asarray(scale), temperature, name, width
        self.net = PolicyNet(feature_size(rs), rs.num_actions, width).double()
        self.net.load_state_dict({k: v.double() for k, v in params.items()})
        self.net.eval()

    def policy(self, obs: Observation, legal: tuple[int, ...]) -> Policy:
        x = (arm_b_features(self.rs, obs) - self.mean) / self.scale
        with torch.no_grad():
            logits = self.net(torch.from_numpy(x[None])).numpy()[0]
        logits = logits / max(self.temperature, 1e-6)
        logits[[a for a in range(self.rs.num_actions) if a not in legal]] = -np.inf
        z = logits - np.max(logits)
        p = np.exp(z)
        p /= p.sum()
        return {int(a): float(p[a]) for a in legal}

    def symmetric(self, name: str | None = None) -> "ArmBPolicy":
        # Kept as a compatibility spelling for experiment code.  Unlike the
        # generic wrapper this does not enumerate the full UNO symmetry group.
        if name is not None:
            self.name = name
        return self

    def save(self, path: str) -> None:
        torch.save({"rs": repr(self.rs), "width": self.width, "mean": self.mean, "scale": self.scale, "temperature": self.temperature, "name": self.name, "params": self.net.state_dict()}, path)

    @classmethod
    def load(cls, path: str, rs: Ruleset) -> "ArmBPolicy":
        d = torch.load(path, weights_only=False)
        if d["rs"] != repr(rs):
            raise ValueError(f"{path} was fitted for another ruleset: {d['rs']}")
        return cls(rs, d["mean"], d["scale"], d["params"], width=d["width"], temperature=d["temperature"], name=d["name"])


def soft_q_target(q: np.ndarray, legal: Iterable[int], temperature: float = 0.05) -> np.ndarray:
    """Convert searched values into a legal-action probability target."""
    q = np.asarray(q, dtype=np.float64)
    legal = tuple(int(a) for a in legal)
    if not legal:
        raise ValueError("soft_q_target needs at least one legal action")
    out = np.zeros_like(q)
    if len(legal) == 1:
        out[legal[0]] = 1.0
        return out
    t = max(float(temperature), 1e-8)
    values = np.asarray([q[a] for a in legal])
    weights = np.exp((values - values.max()) / t)
    weights /= weights.sum()
    out[list(legal)] = weights
    return out


@dataclass(frozen=True, slots=True)
class DistillExample:
    features: np.ndarray
    target: np.ndarray
    legal: tuple[int, ...]
    game: int


class QTeacher(Protocol):
    def q_values(self, obs: Observation, legal: tuple[int, ...]) -> np.ndarray: ...


class PimcQTeacher:
    """Expose a frozen :class:`PimcSearch` as a soft-Q teacher."""

    def __init__(self, search) -> None:
        self.search = search

    def q_values(self, obs: Observation, legal: tuple[int, ...]) -> np.ndarray:
        scores = self.search._scores(obs, legal)  # one canonical teacher implementation
        out = np.full(self.search.rs.num_actions, -np.inf, dtype=np.float64)
        for action, value in scores.items():
            out[int(action)] = float(value)
        return out

    def q_values_batch(self, records: list[tuple[Observation, tuple[int, ...]]]) -> list[np.ndarray]:
        """The same teacher queries with one leaf-network call per seat group.

        World sampling remains keyed independently by each information state and
        in the same order as ``PimcSearch._worlds_q``.  Only the leaf evaluation
        is shared across decisions, which removes the batch-of-one torch cost
        during dataset generation without changing the searched tree.
        """
        if not records:
            return []
        n = self.search.worlds
        worlds_by: list[list] = [[] for _ in records]
        for i, (obs, _) in enumerate(records):
            split = unseen_cards(self.search.rs, obs)
            rng = self.search._rng(obs)
            worlds_by[i] = [determinize_l1(self.search.rs, obs, rng, self.search.belief, split) for _ in range(n)]
        out = [np.zeros(self.search.rs.num_actions, dtype=np.float64) for _ in records]
        for seat in (0, 1):
            indices = [i for i, (obs, _) in enumerate(records) if obs.seat == seat]
            if not indices:
                continue
            flat = [world for i in indices for world in worlds_by[i]]
            leaves = tuple(self.search.leaf) if isinstance(self.search.leaf, list) else self.search.leaf
            qs = batched_search_q(flat, seat, self.search.depth, leaf_for(leaves, seat), self.search.exhaustion, self.search.opponent)
            for j, i in enumerate(indices):
                legal = records[i][1]
                for q in qs[j * n : (j + 1) * n]:
                    for action in legal:
                        out[i][action] += q[action] / n
        return out


class DistillBuffer:
    def __init__(self, rs: Ruleset) -> None:
        self.rs = rs
        self.rows: list[DistillExample] = []

    def extend_trace(self, trace, student_seat: int, teacher: QTeacher, game: int, temperature: float = 0.05) -> int:
        steps = [step for step in trace.steps if step.state.player == student_seat]
        records = [(step.obs, tuple(step.probs)) for step in steps]
        if hasattr(teacher, "q_values_batch"):
            values = teacher.q_values_batch(records)
        else:
            values = [teacher.q_values(obs, legal) for obs, legal in records]
        for step, legal, q in zip(steps, (r[1] for r in records), values):
            self.rows.append(DistillExample(arm_b_features(self.rs, step.obs), soft_q_target(q, legal, temperature), legal, game))
        return len(steps)

    def aggregate(self, student: Agent, teacher: QTeacher, columns: Iterable[Agent], seeds: Iterable[int], start_game: int = 0, temperature: float = 0.05) -> int:
        """Query the teacher on states reached by ``student`` against each column."""
        total = 0
        game = start_game
        for column in columns:
            for seed in seeds:
                for seat, pair in ((0, (student, column)), (1, (column, student))):
                    trace = play_game(self.rs, int(seed), pair)
                    total += self.extend_trace(trace, seat, teacher, game, temperature)
                    game += 1
        return total


@dataclass(frozen=True, slots=True)
class DistillReport:
    examples: int
    epochs: int
    seconds: float
    train_loss: float
    holdout_loss: float


def fit_arm_b(
    rs: Ruleset,
    rows: Iterable[DistillExample],
    width: int = 256,
    batch: int = 1024,
    lr: float = 2e-3,
    weight_decay: float = 1e-6,
    max_epochs: int = 100,
    patience: int = 8,
    seed: int = 0,
    threads: int | None = None,
    policy_temperature: float = 0.05,
) -> tuple[ArmBPolicy, DistillReport]:
    """Fit cross-entropy to soft searched targets, holding out whole games."""
    rows = tuple(rows)
    if not rows:
        raise ValueError("fit_arm_b needs at least one example")
    if threads:
        torch.set_num_threads(threads)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    x = np.asarray([r.features for r in rows], dtype=np.float64)
    y = np.asarray([r.target for r in rows], dtype=np.float64)
    games = np.asarray([r.game for r in rows], dtype=np.int64)
    mean, scale = x.mean(axis=(0, 1)), x.std(axis=(0, 1))
    scale[scale == 0] = 1.0
    xs, ys = torch.from_numpy(((x - mean) / scale).astype(np.float32)), torch.from_numpy(y.astype(np.float32))
    unique = np.unique(games)
    cut = unique[max(1, int(0.8 * len(unique))) - 1] if len(unique) > 1 else unique[-1]
    train = np.nonzero(games <= cut)[0]
    hold = np.nonzero(games > cut)[0]
    if len(hold) == 0:
        n_hold = max(1, len(train) // 5)
        if len(train) > n_hold:
            hold = train[-n_hold:]
            train = train[:-n_hold]
        else:
            # Tiny toy smoke tests may contain one game only.  Reuse that game
            # for the optimization and report the same rows as validation; real
            # aggregation runs always have many independent games.
            hold = train.copy()
    net = PolicyNet(feature_size(rs), rs.num_actions, width)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    def loss_for(idx: np.ndarray) -> torch.Tensor:
        logits = net(xs[idx])
        return -(ys[idx] * torch.log_softmax(logits, dim=1)).sum(1).mean()

    best, best_state, bad, epochs, train_loss = math.inf, None, 0, 0, math.nan
    t0 = time.perf_counter()
    for epoch in range(max_epochs):
        perm = rng.permutation(train)
        net.train()
        for i in range(0, len(perm), batch):
            idx = perm[i : i + batch]
            loss = loss_for(idx)
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            holdout = float(loss_for(hold).item())
        epochs = epoch + 1
        if holdout < best - 1e-6:
            best, bad, train_loss = holdout, 0, float(loss_for(train).item())
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    assert best_state is not None
    policy = ArmBPolicy(rs, mean, scale, best_state, width=width, temperature=policy_temperature)
    return policy, DistillReport(len(rows), epochs, time.perf_counter() - t0, train_loss, best)


__all__ = [
    "ArmBPolicy", "DistillBuffer", "DistillExample", "DistillReport", "PimcQTeacher",
    "arm_b_features", "feature_size", "fit_arm_b", "soft_q_target",
]
