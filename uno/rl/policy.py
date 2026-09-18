"""The learner's policy as an ordinary agent, and its batched form.

One production path (pipeline plan section 5): feature extraction, encoding,
masking, probabilities, sampling, recomputation and evaluation all go through
`ActorPolicy`. `NeuralAgent` is the thin `uno.agents.baseline.Agent` adapter, so
the learner plugs into `uno.eval.play`, the mirrored evaluator and AIVAT like any
other agent -- and, as they require, it returns the **distribution actually
played**, never a sampled action.

`NeuralAgent` is stateless across calls: it rebuilds B's evidence from the
observation's public history every time. That costs a history scan per decision
and is what makes the agent safe for branched and repeated evaluator queries. The
collector keeps an incremental `uno.rl.evidence.Evidence` per live game instead,
and `tests/rl/test_evidence.py` pins the two to the same features.

Temperature is 1 by default everywhere. An argmax or low-temperature deployment
is a *different policy* and needs its own measured result (pipeline plan section 5),
so it is an explicit constructor argument rather than a default of the evaluation
path.
"""

from __future__ import annotations

import numpy as np
import torch

from uno.agents.baseline import Policy
from uno.game.observe import Observation
from uno.game.rules import Ruleset
from uno.rl.evidence import Evidence
from uno.rl.features import Features, Spec, extract, legal_masks
from uno.rl.model import Actor, masked_log_probs, to_tensors


class ActorPolicy:
    """A frozen actor with one masking/temperature convention, batched.

    Used by the collector (worker-local CPU inference), by evaluation and by the
    exact-toy tabulation. `log_probs` is the collection path;
    `uno.rl.ppo.new_log_probs` is the same computation under gradient, and stage
    2's acceptance check is that the two agree at frozen weights.
    """

    def __init__(self, spec: Spec, actor: Actor, temperature: float = 1.0, device: str = "cpu") -> None:
        self.spec, self.actor, self.temperature, self.device = spec, actor, temperature, device
        self.rs: Ruleset = spec.rs
        self.actor.eval()
        self.actor.to(device)

    @torch.no_grad()
    def log_probs(self, features: Features, masks: np.ndarray, set_ids: np.ndarray | None = None) -> np.ndarray:
        """Masked log probabilities, `(n, num_actions)`, with `-inf` on illegal actions.

        `set_ids` is ignored: a network reads its features. The argument exists so a
        tabular actor (`uno.rl.tabular`), which is parameterized by information set,
        is a drop-in for the collector."""
        logits = self.actor(*to_tensors(features, self.device))
        lp = masked_log_probs(logits, torch.from_numpy(masks).to(self.device), self.temperature)
        return lp.cpu().numpy()

    def distributions(self, observations, evidences=None) -> tuple[np.ndarray, np.ndarray]:
        """`(log probabilities, legal masks)` for a batch of decisions."""
        masks = legal_masks(self.rs, list(observations))
        features = extract(self.spec, list(observations), evidences)
        return self.log_probs(features, masks), masks

    def sample(self, log_probs: np.ndarray, rng_values: np.ndarray) -> np.ndarray:
        """Inverse-CDF sampling from masked log probabilities, one uniform per row.

        Taking the uniform from the caller is what keeps sampling keyed by logical
        work rather than by batch composition or completion order (pipeline plan
        section 4): the same episode draws the same number whatever the worker
        count or the batch it landed in.
        """
        p = np.exp(log_probs)
        cdf = np.cumsum(p, axis=1)
        cdf[:, -1] = 1.0
        return (cdf < rng_values[:, None]).sum(axis=1).clip(max=p.shape[1] - 1)


class NeuralAgent:
    """`ActorPolicy` as an `Agent`: one decision, full distribution, no hidden state."""

    def __init__(self, actor: ActorPolicy, name: str) -> None:
        # the attribute is `actor`, not `policy`: `policy` is the Agent protocol's method
        self.actor, self.name = actor, name
        self.rs = actor.rs

    def _evidence(self, obs: Observation) -> Evidence | None:
        if not self.actor.spec.has_evidence:
            return None
        return Evidence.from_public(self.rs, obs.public, obs.seat ^ 1)

    def policy(self, obs: Observation, legal: tuple[int, ...]) -> Policy:
        ev = self._evidence(obs)
        lp, masks = self.actor.distributions([obs], None if ev is None else [ev])
        engine = np.zeros(self.rs.num_actions, dtype=bool)
        engine[list(legal)] = True
        if not np.array_equal(engine, masks[0]):
            raise ValueError("the production mask disagrees with the engine's legal actions")
        p = np.exp(lp[0])
        return {int(a): float(p[a]) for a in legal}


__all__ = ["ActorPolicy", "NeuralAgent"]
