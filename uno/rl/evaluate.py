"""Scheduled DEV evaluation: the fixed panel, paired differences, and exact toys.

docs/rl/pipeline-plan.md section 9. The stochastic played policy is evaluated at
the declared temperature with the production masks, on raw terminal scores. The
panel is fixed and disjoint across checkpoints and variants, balanced over
opponents and seats, and coupled: every cell plays the same deal block, so a
paired difference groups all seats and opponents sharing a deal.

Aggregation is the **equal-weight mean of the eight cell scores**, never a
decision-count weighting. Score is `(return + 1) / 2` with draws worth 1/2, as
everywhere else in the project.

This module stores no features: a panel is thousands of games and retaining their
rollouts would cost more than the training batch. It keeps per-game scores, the
behavioral counters the report card needs, and nothing else.

`exact_value` is the exact-toy path: tabulate the learner over a compiled tree's
L1 quotient and read the exact value against a fixed opponent, with no
trajectory-sampling noise at all. It requires the learner to be symmetric, which
`uno.rl.model` is by construction and `uno.compiled.strategies.tabulate` re-checks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from uno.agents.baseline import Agent
from uno.game.engine import Phase, legal_actions, returns
from uno.game.observe import Level, observe
from uno.game.rules import Play, Ruleset
from uno.rl.collect import LiveGame, sample_action
from uno.rl.features import extract, legal_mask
from uno.rl.manifest import Manifest
from uno.rl.model import entropy_from_log_probs
from uno.rl.policy import ActorPolicy
from uno.rl.streams import EVAL, uniform


@dataclass
class Behavior:
    """The behavioral summaries section 9 asks every evaluation to report."""

    games: int = 0
    draws: int = 0
    cap_games: int = 0
    total_turns: int = 0
    learner_decisions: int = 0
    voluntary_draws: int = 0
    keeps: int = 0  # Pass at AFTER_DRAW: a playable drawn card kept
    wild_plays: int = 0
    action_card_plays: int = 0
    entropy_sum: float = 0.0

    def as_dict(self) -> dict:
        n = max(self.games, 1)
        d = max(self.learner_decisions, 1)
        return {
            "draw_rate": self.draws / n,
            "action_cap_rate": self.cap_games / n,
            "mean_game_length": self.total_turns / n,
            "voluntary_draw_rate": self.voluntary_draws / d,
            "keep_rate": self.keeps / d,
            "wild_play_rate": self.wild_plays / d,
            "action_card_rate": self.action_card_plays / d,
            "entropy": self.entropy_sum / d,
        }


@dataclass
class PanelResult:
    """One checkpoint's score on one panel."""

    cells: tuple[tuple[str, int], ...]
    scores: dict[tuple[str, int], np.ndarray]  # per cell, per deal, in deal order
    deals: np.ndarray
    behavior: Behavior = field(default_factory=Behavior)

    @property
    def per_deal_mean(self) -> np.ndarray:
        """The equal-weight league mean per deal: the paired unit."""
        return np.mean([self.scores[c] for c in self.cells], axis=0)

    @property
    def cell_means(self) -> dict[tuple[str, int], float]:
        return {c: float(self.scores[c].mean()) for c in self.cells}

    @property
    def mean(self) -> float:
        return float(np.mean(list(self.cell_means.values())))

    @property
    def standard_error(self) -> float:
        x = self.per_deal_mean
        return float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else float("nan")

    @property
    def ci95(self) -> tuple[float, float]:
        h = 1.96 * self.standard_error
        return (self.mean - h, self.mean + h)

    def __str__(self) -> str:
        cells = " ".join(f"{n}/s{s}={v:.3f}" for (n, s), v in self.cell_means.items())
        return f"league {self.mean:.4f} +- {1.96 * self.standard_error:.4f} (95% CI, {len(self.deals)} deals/cell) [{cells}]"


@dataclass
class Paired:
    """A paired difference between two panel results on the same deals."""

    difference: float
    standard_error: float
    n: int

    @property
    def ci95(self) -> tuple[float, float]:
        h = 1.96 * self.standard_error
        return (self.difference - h, self.difference + h)

    def __str__(self) -> str:
        return f"{self.difference:+.4f} +- {1.96 * self.standard_error:.4f} (paired 95% CI on {self.n} deals)"


def paired(a: PanelResult, b: PanelResult) -> Paired:
    """`a - b` on the shared deals, grouping every cell that shares one.

    Grouping is what makes this a *paired* interval: seats and opponents on one
    deal are not independent observations (AGENTS.md, evidence section).
    """
    if not np.array_equal(a.deals, b.deals):
        raise ValueError("a paired difference needs the same panel deals on both sides")
    d = a.per_deal_mean - b.per_deal_mean
    se = float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else float("nan")
    return Paired(float(d.mean()), se, len(d))


def play_panel(manifest: Manifest, actor: ActorPolicy, opponents: dict[str, Agent], training_seed: int, live_games: int = 64) -> PanelResult:
    """Play the panel with the stochastic policy, keeping scores and counters only."""
    rs: Ruleset = actor.rs
    spec = actor.spec
    by_cell: dict[tuple[str, int], dict[int, float]] = {c: {} for c in manifest.cells}
    behavior = Behavior()
    queue = list(manifest.episodes)
    active: list[LiveGame] = []
    is_play = tuple(isinstance(a, Play) for a in rs.actions)

    while True:
        while queue and len(active) < live_games:
            episode = queue.pop(0)
            active.append(LiveGame(rs, episode, opponents[episode.opponent]))
        if not active:
            break
        ready, done = [], []
        for game in active:
            game.advance_to_learner(training_seed)
            (done if game.terminal else ready).append(game)
        for game in done:
            value = float(returns(game.state)[game.seat])
            by_cell[(game.episode.opponent, game.episode.seat)][game.episode.deal_seed] = (value + 1.0) / 2.0
            behavior.games += 1
            behavior.draws += value == 0.0
            behavior.cap_games += game.state.end == "cap"
            behavior.total_turns += game.state.turn
        active = ready
        if not active:
            continue
        obs = [g.observations()[0] for g in active]
        masks = np.stack([legal_mask(rs, o) for o in obs])
        feats = extract(spec, obs, [g.evidence for g in active] if spec.has_evidence else None)
        log_probs = actor.log_probs(feats, masks)
        behavior.entropy_sum += float(entropy_from_log_probs(torch.from_numpy(log_probs)).sum())
        us = np.fromiter((uniform(training_seed, EVAL, g.episode.episode_id, g.decision_index) for g in active), dtype=np.float64, count=len(active))
        chosen = actor.sample(log_probs, us)
        for i, game in enumerate(active):
            a = int(chosen[i])
            behavior.learner_decisions += 1
            if a == rs.draw_action:
                behavior.voluntary_draws += 1
            elif a == rs.pass_action and obs[i].phase is Phase.AFTER_DRAW:
                behavior.keeps += 1
            elif is_play[a]:
                card = rs.card_types[rs.actions[a].card]
                behavior.wild_plays += card.is_wild
                behavior.action_card_plays += not card.is_wild and card.rank is None
            game.apply(a)

    deals = np.asarray(sorted({e.deal_seed for e in manifest.episodes}), dtype=np.int64)
    scores = {c: np.asarray([by_cell[c][d] for d in deals], dtype=np.float64) for c in manifest.cells}
    return PanelResult(manifest.cells, scores, deals, behavior)


def baseline_panel(manifest: Manifest, agent: Agent, opponents: dict[str, Agent], rs: Ruleset, training_seed: int) -> PanelResult:
    """The same panel played by a reference agent, for the paired comparison."""
    by_cell: dict[tuple[str, int], dict[int, float]] = {c: {} for c in manifest.cells}
    behavior = Behavior()
    for episode in manifest.episodes:
        game = LiveGame(rs, episode, opponents[episode.opponent])
        while not game.terminal:
            game.advance_to_learner(training_seed)
            if game.terminal:
                break
            legal = legal_actions(game.state)
            probs = agent.policy(observe(game.state, game.seat, Level.L1), legal)
            u = uniform(training_seed, EVAL, episode.episode_id, game.decision_index)
            game.apply(sample_action(probs, u))
        value = float(returns(game.state)[game.seat])
        by_cell[(episode.opponent, episode.seat)][episode.deal_seed] = (value + 1.0) / 2.0
        behavior.games += 1
        behavior.draws += value == 0.0
        behavior.cap_games += game.state.end == "cap"
        behavior.total_turns += game.state.turn
    deals = np.asarray(sorted({e.deal_seed for e in manifest.episodes}), dtype=np.int64)
    scores = {c: np.asarray([by_cell[c][d] for d in deals], dtype=np.float64) for c in manifest.cells}
    return PanelResult(manifest.cells, scores, deals, behavior)


def select(history) -> tuple[int, float]:
    """The DEV-selected checkpoint: largest equal-weight mean, ties to the earlier one.

    `history` is `(update, score)` in update order. Ties within `1e-6` go to the
    earlier checkpoint (pipeline plan section 9). The selected checkpoint's interval
    is descriptive, not a post-selection confirmatory interval.
    """
    best_update, best_score = -1, -math.inf
    for update, score in history:
        if score > best_score + 1e-6:
            best_update, best_score = update, score
    return best_update, best_score


# -- exact toy evaluation ----------------------------------------------------------


def exact_value(rs: Ruleset, learner: Agent, opponent: Agent, seat: int) -> float:
    """The exact score of `learner` at `seat` against `opponent`, on the compiled tree.

    No trajectory-sampling noise: this is the game value, not an estimate of it.
    The tree is the cached L1 quotient (`uno.store`), so the learner must be
    symmetric -- `tabulate` refuses it otherwise, which is itself a check on the
    architecture's equivariance.
    """
    from uno import store
    from uno.compiled.solvers import evaluate

    tree = store.load_or_compile(rs, (Level.L1,), symmetric=True)
    theirs = store.load_or_tabulate(tree, Level.L1, opponent)
    from uno.compiled.strategies import tabulate

    mine = tabulate(tree, Level.L1, learner)
    a, b = ((Level.L1, mine), (Level.L1, theirs)) if seat == 0 else ((Level.L1, theirs), (Level.L1, mine))
    return (evaluate(tree, a, b)[seat] + 1.0) / 2.0


__all__ = ["Behavior", "PanelResult", "Paired", "baseline_panel", "exact_value", "paired", "play_panel", "select"]
