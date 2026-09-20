"""Exact-toy evaluation and exact gradients of the *same* learner parameters.

docs/rl/pipeline-plan.md section 13, stage 5: "Test the same neural parameters
against exact expected gradients, not unrelated tabular logits." That is what this
module makes possible. It tabulates the production policy -- the real features, the
real mask, the real network, the real temperature -- over a compiled L1 quotient
tree, and then:

- `value` reads the exact game value against a fixed opponent, with no
  trajectory-sampling noise at all;
- `gradient` differentiates that exact value with respect to the network's own
  parameters, by chaining `uno.compiled.gradient.exact_gradient`'s `dJ/dsigma`
  through the same forward pass autograd already knows;
- `sampled_gradient` computes the unregularized score-function estimator from a
  real rollout, per game, so the comparison carries **game-level uncertainty**
  rather than a bare number.

Tabulation requires a symmetric learner, and `Keying.is_symmetric` is checked
here exactly as `uno.compiled.strategies.tabulate` checks it. In float32 the
network's equivariance error is ~1e-7, which that check rejects, so tabulation
runs the actor in float64 -- a numerical-precision choice for the exact path only;
collection and training stay FP32 as declared.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np
import torch

from uno import store
from uno.compiled.gradient import exact_gradient
from uno.compiled.solvers import best_response, evaluate
from uno.compiled.tree import CompiledTree, Keying
from uno.game.keys import decode_key
from uno.game.observe import Level, observation_from_key
from uno.game.rules import Ruleset
from uno.rl.evidence import Evidence
from uno.rl.features import Spec, extract
from uno.rl.model import Actor, masked_log_probs

SYMMETRY_TOL = 1e-9  # the tolerance `uno.compiled.strategies.tabulate` applies


@cache
def tree_for(rs: Ruleset) -> CompiledTree:
    """The cached L1 quotient tree of a toy ruleset (`uno.store`)."""
    return store.load_or_compile(rs, (Level.L1,), symmetric=True)


@cache
def _sets(rs: Ruleset, seat: int | None) -> tuple:
    """Observations, evidence and ia ranges of every set of the keying, once per ruleset.

    Rebuilding an observation costs ~9 us and a toy keying has tens of thousands of
    sets, so this is cached: an optimization loop tabulates the same sets many times
    with different parameters.
    """
    tree = tree_for(rs)
    k = tree.keyings[Level.L1]
    order = sorted(k.keys.values())
    rev = {i: b for b, i in k.keys.items()}
    obs, evidence, ia_index, rows = [], [], [], []
    for info in order:
        if seat is not None and int(k.info_player[info]) != seat:
            continue
        o = observation_from_key(rs, Level.L1, decode_key(Level.L1, rev[info]))
        obs.append(o)
        evidence.append(Evidence.from_public(rs, o.public, o.seat ^ 1))
        lo, hi = int(k.info_start[info]), int(k.info_start[info + 1])
        ia_index.append((lo, hi))
        rows.append(info)
    # Flat gather indices, built once: a per-set Python loop over tens of thousands
    # of sets dominated the exact path before this (1.8 s a gradient step against
    # 0.2 s), and an exact-gradient optimization takes thousands of them.
    positions = np.concatenate([np.arange(lo, hi) for lo, hi in ia_index]) if ia_index else np.zeros(0, np.int64)
    row_index = np.concatenate([np.full(hi - lo, i) for i, (lo, hi) in enumerate(ia_index)]) if ia_index else np.zeros(0, np.int64)
    action_index = np.concatenate([k.ia_action[lo:hi] for lo, hi in ia_index]).astype(np.int64) if ia_index else np.zeros(0, np.int64)
    return tuple(obs), tuple(evidence), tuple(ia_index), tuple(rows), positions, row_index, action_index


def tabulate(rs: Ruleset, spec: Spec, actor: Actor, temperature: float = 1.0, seat: int | None = None, base: np.ndarray | None = None) -> torch.Tensor:
    """The learner's probability for every set-action of the keying, differentiably.

    Returns an `(n_ia,)` tensor. `seat` restricts the forward pass to one seat's
    sets, which is what a fixed-seat gradient check needs; the other seat's entries
    come from `base` (the uniform strategy when not given), since a strategy vector
    covers both seats.
    """
    tree = tree_for(rs)
    k: Keying = tree.keyings[Level.L1]
    obs, evidence, ia_index, _, positions, row_index, action_index = _sets(rs, seat)
    feats = _features(rs, spec.arm, seat)
    masks = _masks(rs, seat)
    logits = actor(
        torch.from_numpy(feats.tok).double(), torch.from_numpy(feats.col).double(), torch.from_numpy(feats.glob).double()
    )
    probs = masked_log_probs(logits, torch.from_numpy(masks), temperature).exp()
    flat = probs[torch.from_numpy(row_index), torch.from_numpy(action_index)]
    sigma = torch.from_numpy(k.uniform() if base is None else base.copy())
    return sigma.index_copy(0, torch.from_numpy(positions.astype(np.int64)), flat)


@cache
def _features(rs: Ruleset, arm: str, seat: int | None):
    """The declared features of every set, built once: they do not depend on weights."""
    from uno.rl.features import spec as spec_for

    sp = spec_for(rs, arm)
    obs, evidence, _, _, _, _, _ = _sets(rs, seat)
    return extract(sp, list(obs), list(evidence) if sp.has_evidence else None)


@cache
def _masks(rs: Ruleset, seat: int | None) -> np.ndarray:
    k = tree_for(rs).keyings[Level.L1]
    _, _, ia_index, _, _, _, _ = _sets(rs, seat)
    masks = np.zeros((len(ia_index), rs.num_actions), dtype=bool)
    for i, (lo, hi) in enumerate(ia_index):
        masks[i, k.ia_action[lo:hi]] = True
    return masks


def strategy(rs: Ruleset, spec: Spec, actor: Actor, temperature: float = 1.0, seat: int | None = None, base: np.ndarray | None = None) -> np.ndarray:
    """`tabulate` as a plain strategy vector, checked for symmetry as `tabulate` does."""
    with torch.no_grad():
        sigma = tabulate(rs, spec, actor.double(), temperature, seat, base).numpy()
    actor.float()
    k = tree_for(rs).keyings[Level.L1]
    if not k.is_symmetric(sigma, SYMMETRY_TOL):
        worst = float(np.abs(sigma - k.orbit_mean(sigma)).max())
        raise ValueError(f"the tabulated policy is not symmetric (max orbit deviation {worst:.3g}): it cannot be evaluated on a quotient tree")
    return sigma


def value(rs: Ruleset, sigma: np.ndarray, opponent_sigma: np.ndarray, seat: int) -> float:
    """The exact score of the learner at `seat` against a fixed tabulated opponent."""
    tree = tree_for(rs)
    mine, theirs = (Level.L1, sigma), (Level.L1, opponent_sigma)
    a, b = (mine, theirs) if seat == 0 else (theirs, mine)
    return (evaluate(tree, a, b)[seat] + 1.0) / 2.0


def br_score(rs: Ruleset, opponent_sigma: np.ndarray, seat: int) -> float:
    """The exact L1 best-response score at `seat` against that opponent: the ceiling."""
    return best_response(tree_for(rs), Level.L1, (Level.L1, opponent_sigma), seat).score


@dataclass
class ExactGradient:
    """The exact gradient of the learner's expected return w.r.t. its parameters."""

    value: float  # the learner's expected return, in [-1, 1]
    parameters: list[torch.Tensor]
    d_sigma: np.ndarray

    @property
    def flat(self) -> torch.Tensor:
        return torch.cat([p.reshape(-1) for p in self.parameters])


def gradient(rs: Ruleset, spec: Spec, actor: Actor, opponent_sigma: np.ndarray, seat: int, temperature: float = 1.0) -> ExactGradient:
    """`dJ/dtheta` exactly, for the network's own parameters.

    `uno.compiled.gradient.exact_gradient` gives `dJ/dsigma(I, a)` -- the
    reach-weighted arc value -- for the tabulated policy. Because the same forward
    pass produced that policy, the chain rule is just a backward pass through
    `sum_{I,a} dJ/dsigma(I, a) * sigma(I, a)` with the coefficients detached. No
    finite differences and no surrogate: this is the derivative of the exact game
    value.
    """
    actor.double()
    sigma_t = tabulate(rs, spec, actor, temperature, seat)
    sigma = sigma_t.detach().numpy()
    tree = tree_for(rs)
    g = exact_gradient(tree, (Level.L1, sigma), (Level.L1, opponent_sigma), seat)
    coeff = torch.from_numpy(g.prob)
    actor.zero_grad(set_to_none=True)
    (sigma_t * coeff).sum().backward()
    grads = [(p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)) for p in actor.parameters()]
    actor.zero_grad(set_to_none=True)
    actor.float()
    return ExactGradient(g.value, [t.float() for t in grads], g.prob)


def sampled_gradient(actor: Actor, rollout, temperature: float = 1.0, baseline: np.ndarray | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-game score-function gradients `G_g = sum_t grad log p(a_t) * A_gt`.

    Returns `(per_game, decision_mean)`, both flattened. With `baseline=None` the weight is
    the raw terminal return, which is the *unregularized* reference the plan asks
    the PPO gradient to match at collection weights; with a baseline it is the
    advantage actually used in training. Keeping the per-game vectors is what gives
    the comparison a game-level standard error instead of a point estimate. The
    aggregate is the ordinary learner-decision mean, matching PPO.
    """
    from uno.rl.ppo import new_log_probs

    weights = rollout.returns_per_decision if baseline is None else (rollout.returns_per_decision - baseline)
    n_params = sum(p.numel() for p in actor.parameters())
    per_game = torch.zeros(rollout.n_games, n_params)
    for i in range(rollout.n_games):
        rows = slice(int(rollout.offset[i]), int(rollout.offset[i]) + int(rollout.learner_decisions[i]))
        if rows.stop == rows.start:
            continue
        logp, _ = new_log_probs(actor, rollout.actor[rows], rollout.masks[rows], rollout.actions[rows], temperature, "cpu", None if rollout.set_id is None else rollout.set_id[rows])
        actor.zero_grad(set_to_none=True)
        (logp * torch.from_numpy(weights[rows].astype(np.float32))).sum().backward()
        per_game[i] = torch.cat([(p.grad.reshape(-1) if p.grad is not None else torch.zeros(p.numel())) for p in actor.parameters()])
    actor.zero_grad(set_to_none=True)
    counts = torch.from_numpy(rollout.learner_decisions.astype(np.float32))
    total = float(counts.sum())
    decision_mean = (per_game * counts[:, None]).sum(0) / total if total else per_game.sum(0)
    return per_game, decision_mean


@dataclass
class Agreement:
    """A projection test of a sampled gradient against an exact one."""

    direction: str
    exact: float
    sampled: float
    standard_error: float
    games: int

    @property
    def z(self) -> float:
        return (self.sampled - self.exact) / self.standard_error if self.standard_error else float("nan")

    def __str__(self) -> str:
        return f"{self.direction:>10s}: exact {self.exact:+.5f} sampled {self.sampled:+.5f} +- {self.standard_error:.5f}  z = {self.z:+.2f}"


def agreement(per_game: torch.Tensor, exact: torch.Tensor, extra_directions: int = 4, seed: int = 0) -> list[Agreement]:
    """Project both gradients on the exact direction and a few random ones.

    Comparing every coordinate would drown in multiple comparisons; a projection is
    one scalar with an honest game-level standard error. The random directions are
    there because agreeing along the exact direction alone would not catch an error
    orthogonal to it.
    """
    g = per_game.shape[0]
    directions = []
    norm = exact.norm()
    if float(norm) > 0:
        directions.append(("exact", exact / norm))
    rng = torch.Generator().manual_seed(seed)
    for i in range(extra_directions):
        v = torch.randn(exact.shape, generator=rng)
        directions.append((f"random{i}", v / v.norm()))
    out = []
    for name, d in directions:
        samples = per_game @ d
        out.append(Agreement(name, float(exact @ d), float(samples.mean()), float(samples.std(unbiased=True) / np.sqrt(g)), g))
    return out


__all__ = [
    "Agreement", "ExactGradient", "SYMMETRY_TOL", "agreement", "br_score", "gradient",
    "sampled_gradient", "strategy", "tabulate", "tree_for", "value",
]  # fmt: skip
