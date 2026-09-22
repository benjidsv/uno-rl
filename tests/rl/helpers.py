"""Shared fixtures for the learner tests: sampled decision states and small nets."""

from __future__ import annotations

import random

from uno.game.engine import State, legal_actions, new_game, step
from uno.game.observe import Level, observe
from uno.game.rules import Ruleset
from uno.rl.env.evidence import Evidence
from uno.rl.env.features import spec
from uno.rl.model import make_actor, make_critic

SMALL_WIDTH = 24  # tests check structure, not capacity


def decision_games(rs: Ruleset, seeds, max_decisions: int = 60) -> list[list[State]]:
    """Random playouts on `seeds`, each as its list of decision states in order."""
    games: list[list[State]] = []
    for seed in seeds:
        try:
            s = new_game(rs, seed)
        except ValueError:
            continue
        rng = random.Random(seed * 7919 + 1)
        game: list[State] = []
        for _ in range(max_decisions):
            if s.terminal:
                break
            game.append(s)
            s = step(s, rng.choice(legal_actions(s)))
        games.append(game)
    return games


def decision_states(rs: Ruleset, seeds, max_decisions: int = 60) -> list[State]:
    """Every decision state of random playouts on `seeds`, in order."""
    return [s for game in decision_games(rs, seeds, max_decisions) for s in game]


def rows(rs: Ruleset, states, level: Level = Level.L1):
    """(observation, evidence) per state, from the mover's seat, rebuilt from history."""
    obs = [observe(s, s.player, level) for s in states]
    ev = [Evidence.from_public(rs, s.public, s.player ^ 1) for s in states]
    return obs, ev


def small_actor(rs: Ruleset, arm: str, seed: int = 0) -> Actor:
    import torch

    torch.manual_seed(seed)
    from uno.rl.core.config import RelationalModel
    return make_actor(spec(rs, arm), RelationalModel(width=SMALL_WIDTH, blocks=2))


def small_critic(rs: Ruleset, seed: int = 0) -> Critic:
    import torch

    torch.manual_seed(seed)
    from uno.rl.core.config import RelationalModel
    return make_critic(spec(rs, "CRITIC"), RelationalModel(width=SMALL_WIDTH, blocks=2))
