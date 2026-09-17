"""The approximate-response attack instrument (step 10a.3C, docs/spec.md section 32).

An **attack** takes a frozen target policy and tries to beat it from one seat. Its *achieved
score in real games* is a lower bound on the best-response score against that target, so an
attack needs no exact belief and no exact Q to be valid -- it only has to be an admissible L1
policy. What it can be is **insensitive**: a weak attack that fails to find a target's weakness
reports a small number, and a small number is not evidence of safety
([Lisy & Bowling 2017](https://arxiv.org/abs/1612.07547), and docs/results/10-review.md section 4).
So the instrument is calibrated against the exact toy best responses, where the answer is known.

**Why the seat sum is the reportable quantity.** On full UNO the seat-specific Nash value is
unknown, so "attack score minus 0.5" is not exploitability and must never be printed as one. With
`b0` the attacker's score at seat 0 against the target's seat-1 policy and `b1` the corresponding
seat-1 score, the unknown Nash seat values cancel in the sum:

    NashConv(target) >= b0 + b1 - 1        (score units; x100 for score points)

`nashconv_bound` estimates that from per-deal paired arrays, so the covariance between the two
seatings of a deal is carried rather than assumed away.

**The continuation rule decides which targets may be attacked at all.** `uno.eval.search`'s
`response` continuation solves each determinized world as a best response to the target's real
policy, which means it *queries the target inside the world* -- where the opponent's
`initial_hands` / `own_draws` are fabricated (`determinize_l1`). A target that reads its own
private history is then handed a history that never happened. This is not a theoretical worry:
measured on `TOY` against the tabulated Nash strategy, **79.5 % of such queries miss the keying
entirely and silently return a point mass on the first legal action**, so the search would be
best-responding to "plays its first legal action" and not to the target. Nothing raises, and the
resulting attack is merely weak -- the exact failure mode this step exists to rule out.

`supports` therefore *measures* whether a target is a function of the current position, instead of
trusting a declaration, and `check_supported` raises `UnsupportedTarget` for the combinations that
cannot be queried honestly:

| continuation | reads the target? | supported targets |
|---|---|---|
| `response` | yes, inside each world | position-only targets (the baseline rulebots) |
| `surrogate` | no: a fixed stand-in opponent | any target, memoryful included |
| `minimax` | no: assumes a clairvoyant opponent | any target |
| `rollout` | no: worlds played out by fixed L1 policies | any target |

`surrogate` is the column-free construction of `uno.experiments.colfree`, reused here: it keeps the
strength of an opponent-aware continuation without reading the target. `rollout` is the second
attacker family (plan item 2): it carries no strategy fusion, so it fails differently from
`PimcSearch`, which is what stops one family's blind spot from being read as target safety.

**The attacker's own seed is a reported uncertainty.** `PimcSearch` keys its world sampling on the
information state, so a built attacker is *one deterministic realization* of the sampling, not the
averaged behavioural policy that `uno.compiled.pimc.pimc_n` computes. Measured on `TOY` (w = 4, vs
Random), six attacker seeds spread with sd 0.84 / 0.58 score points while their mean matched
`pimc_n` at 256 realizations to 0.03 / 0.12 points. So an attack score carries two independent
uncertainties -- the deals and the attacker's seed -- and `attacker_seeds` exists to measure the
second rather than hide it.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace

import numpy as np

from uno.eval.belief import Belief
from uno.eval.search import PimcRollout, PimcSearch
from uno.game.engine import legal_actions, new_game, step
from uno.game.observe import Level, Observation, observe
from uno.game.rules import Ruleset

CONTINUATIONS = ("response", "surrogate", "minimax", "rollout")

# Continuations that evaluate the target's own policy inside a determinized world, where its
# private history is fabricated. Only these constrain which targets may be attacked.
QUERIES_TARGET = ("response",)


@dataclass(frozen=True, slots=True)
class Budget:
    """One attack configuration: what the attacker is allowed to spend, and how it believes.

    `worlds` / `depth` are `PimcSearch`'s own knobs (docs/spec.md section 25); `belief` is the
    opponent-hand rung of section 26, a **separate** field from the continuation because the two
    axes are independent and folding one into the other silently drops rows from a ladder (the
    step 9.5f lesson, pinned by `tests/eval/test_pimc_configs.py`). `seed` selects the attacker's
    world-sampling realization, which is a real source of variation and not a formality.
    """

    continuation: str = "response"
    worlds: int = 32
    depth: int = 2
    belief: str = "voids"
    seed: int = 0
    rollouts: int = 1  # `rollout` continuation only: playouts per world

    def __post_init__(self) -> None:
        if self.continuation not in CONTINUATIONS:
            raise ValueError(f"unknown continuation {self.continuation!r}: one of {', '.join(CONTINUATIONS)}")
        if self.worlds < 1 or self.depth < 1:
            raise ValueError(f"worlds and depth must be positive, got {self.worlds} / {self.depth}")

    @property
    def label(self) -> str:
        """A short, stable token for tables and agent names. The belief and the continuation stay
        separate fields of it, for the reason the class docstring gives."""
        core = f"{self.continuation}_w{self.worlds}"
        if self.continuation != "rollout":
            core += f"d{self.depth}"
        else:
            core += f"x{self.rollouts}"
        return f"{core}_{self.belief}_k{self.seed}"


# -- which targets may be attacked, measured rather than declared --------------------------------


class UnsupportedTarget(ValueError):
    """Raised when an attack would have to query a target it cannot query honestly."""


@dataclass(frozen=True, slots=True)
class Support:
    """What `supports` measured about a target.

    `tested` counts only decisions where the perturbed observation genuinely differs from the
    original: a perturbation that changes nothing cannot detect anything, and counting it would
    make the detector look thorough while testing nothing (the vacuity trap this repo keeps
    finding -- docs/spec.md section 26.5). `raised` counts targets that reject a fabricated
    observation outright, which is also a refusal to be queried.

    **This is not an L1-validity test, and must never be used as one.** An L1 policy is *entitled*
    to read its own initial hand and own draws -- that is what L1 *is* (docs/spec.md section 10).
    Worse, those fields plus the public plays *determine* the current hand, so any policy that
    rebuilds its position from the L1 key (`CompiledAgent`, `Symmetrized`, a recurrent arm) answers
    differently here for a legitimate reason. What this measures is the narrower property the
    determinized-world query needs: whether the policy is a function of the **current position
    alone**. For L1 validity -- not reading the opponent's hand or the pile order -- use
    `hidden_state_probe`."""

    name: str
    tested: int
    changed: int
    raised: int

    @property
    def position_only(self) -> bool:
        """Whether the target's policy is a function of the current position alone, so it can be
        evaluated inside a determinized world."""
        return self.tested > 0 and self.changed == 0 and self.raised == 0

    @property
    def conclusive(self) -> bool:
        """Whether the probe actually exercised the distinction at all."""
        return self.tested > 0

    def __str__(self) -> str:
        verdict = "position-only" if self.position_only else ("reads private history" if self.tested else "inconclusive")
        return f"{self.name}: {verdict} ({self.changed} changed, {self.raised} raised, of {self.tested} effective probes)"


def _probe_decisions(rs: Ruleset, games: int, cap: int = 60):
    """(state, the mover's L1 observation) along random-policy games: the positions the detector
    asks about. Random play rather than the target's own play, so the probe is not confined to the
    part of the game the target happens to visit."""
    for seed in range(games):
        try:
            s = new_game(rs, seed)
        except ValueError:
            continue
        rng = random.Random(seed)
        n = 0
        while not s.terminal and n < cap:
            yield s, observe(s, s.player, Level.L1)
            s = step(s, rng.choice(legal_actions(s)))
            n += 1


def _perturbations(obs: Observation) -> list[Observation]:
    """The same position with only the observer's *private history* altered.

    The first is exactly what `determinize_l1` fabricates for the opponent (initial hand = the
    hand now, no draws), i.e. the query an attack would really make. The second adds a draw that
    never happened, which catches a target reading `own_draws` while `initial_hand == hand` already
    holds and the first perturbation would be a no-op."""
    out = [replace(obs, initial_hand=obs.hand, own_draws=())]
    if obs.hand:
        out.append(replace(obs, own_draws=(*obs.own_draws, obs.hand[0])))
    return out


def supports(rs: Ruleset, target, games: int = 40, cap: int = 60) -> Support:
    """Measure whether `target`'s policy depends on its own private history.

    Asks the target at real decisions, then at the same positions with only the private history
    changed, and counts the disagreements. A position-only policy (every baseline rulebot) answers
    identically; a policy keyed on the full L1 information state (a tabulated strategy, an RL
    checkpoint with history features) does not. Measured on `TOY`: the four rulebots changed at
    0 of 103 decisions, the tabulated Nash strategy at 51 of 103.
    """
    tested = changed = raised = 0
    for s, obs in _probe_decisions(rs, games, cap):
        legal = legal_actions(s)
        if len(legal) < 2:
            continue  # a forced move reveals nothing about what the policy reads
        try:
            base = target.policy(obs, legal)
        except Exception:  # a target that cannot answer a real observation is not attackable here
            raised += 1
            tested += 1
            continue
        for faked in _perturbations(obs):
            if (faked.initial_hand, faked.own_draws) == (obs.initial_hand, obs.own_draws):
                continue  # no-op perturbation: it could not detect anything, so it is not a test
            tested += 1
            try:
                other = target.policy(faked, legal)
            except Exception:
                raised += 1
                continue
            if other != base:
                changed += 1
    return Support(getattr(target, "name", type(target).__name__), tested, changed, raised)


@dataclass(frozen=True, slots=True)
class HiddenProbe:
    """What `hidden_state_probe` measured: whether a policy reads the hidden split.

    `tested` counts decisions where a genuinely different world was substituted behind an identical
    L1 observation; `changed` counts the ones where the policy's answer moved."""

    name: str
    tested: int
    changed: int
    observations_differed: int

    @property
    def l1_valid(self) -> bool:
        return self.tested > 0 and self.changed == 0 and self.observations_differed == 0

    def __str__(self) -> str:
        verdict = "L1-valid" if self.l1_valid else ("reads the hidden state" if self.tested else "inconclusive")
        extra = f", {self.observations_differed} observations differed" if self.observations_differed else ""
        return f"{self.name}: {verdict} ({self.changed} of {self.tested} decisions moved{extra})"


def hidden_state_probe(rs: Ruleset, agent, games: int = 20, cap: int = 60) -> HiddenProbe:
    """Does the policy read what it cannot see? The L1-validity test (docs/spec.md section 11,
    property 9; the observation contract's invariance test 2).

    `determinize_l1` builds a state consistent with the observer's own L1 observation but with the
    **unseen** cards -- the opponent's hand and the pile order -- split differently. The observer's
    L1 observation of that world is by construction identical, so an L1 policy must return exactly
    the same distribution. A policy that peeked at the opponent's hand or the pile would not.

    This is the complement of `supports`: that one asks whether a policy reads its *own* private
    history (legitimate at L1, but fatal to a determinized-world query), this one whether it reads
    *someone else's* (never legitimate)."""
    import random as _random

    from uno.eval.search import determinize_l1

    name = getattr(agent, "name", type(agent).__name__)
    tested = changed = differed = 0
    for s, obs in _probe_decisions(rs, games, cap):
        legal = legal_actions(s)
        if len(legal) < 2:
            continue
        world = determinize_l1(rs, obs, _random.Random(1000 + tested))
        other = observe(world, obs.seat, Level.L1)
        if other != obs:
            # the engine's own invariant: an L1 observation may not depend on the hidden split
            differed += 1
        tested += 1
        if agent.policy(other, legal) != agent.policy(obs, legal):
            changed += 1
    return HiddenProbe(name, tested, changed, differed)


def check_supported(rs: Ruleset, target, continuation: str, games: int = 40) -> Support:
    """`supports`, raising `UnsupportedTarget` when the continuation cannot query this target.

    Only the continuations of `QUERIES_TARGET` constrain anything: the others never read the
    target, so any representation is attackable by them (at whatever strength they achieve).
    An *inconclusive* probe is also refused for a querying continuation -- "we could not tell"
    is not a licence to query."""
    sup = supports(rs, target, games)
    if continuation not in QUERIES_TARGET:
        return sup
    if not sup.conclusive:
        raise UnsupportedTarget(
            f"cannot tell whether {sup.name!r} reads its private history ({sup}); the {continuation!r} continuation "
            "queries the target inside determinized worlds, so refuse rather than query blindly"
        )
    if not sup.position_only:
        raise UnsupportedTarget(
            f"{sup.name!r} reads its own private history ({sup}), and the {continuation!r} continuation evaluates the "
            "target inside determinized worlds whose private history is fabricated (docs/spec.md section 32). "
            "Measured on TOY against the tabulated Nash strategy, 79.5 % of such queries miss the keying and silently "
            "return the first legal action, so the search would best-respond to a policy the target does not follow. "
            "Use continuation='surrogate' (a fixed stand-in opponent), 'minimax', or 'rollout' instead."
        )
    return sup


# -- building the attacker ------------------------------------------------------------------------


def build(rs: Ruleset, target, budget: Budget, leaf=None, surrogate=None, rollout_agents=None, name: str | None = None, check: bool = True):
    """The attacker for one (target, budget), as an ordinary agent.

    `leaf` is the depth-limited search's leaf value function (`uno.eval.value_net.ValueNet.leaf`,
    or a per-seat pair -- step 9.5e's `leaf_for`). On a toy a depth that always reaches a terminal
    makes the leaf unreachable, which is how the exact calibration avoids depending on a value
    network at all.

    `surrogate` is the fixed stand-in opponent of the `surrogate` continuation, and
    `rollout_agents` the fixed L1 pair the `rollout` continuation plays each world out with.

    `check` runs `check_supported` first. It is on by default: an attack that silently queries a
    target it cannot query is precisely the insensitive-attack failure this module is written
    against."""
    if check:
        check_supported(rs, target, budget.continuation)
    # A name identifies a policy (docs/spec.md section 13) and keys both the action rng and the
    # world sampling, so every knob that changes the policy has to appear in it -- including the
    # target, since a `response` attacker is a different policy per target.
    label = name or f"lbr_{budget.label}_vs_{getattr(target, 'name', 'target')}"
    if budget.continuation == "rollout":
        if rollout_agents is None:
            raise ValueError("the rollout continuation needs `rollout_agents`: the fixed L1 policies each world is played out with")
        if budget.belief != "prior":
            # `PimcRollout` determinizes without a belief, so any other value would put a rung in
            # the agent's name that its policy does not use -- and a name must identify a policy
            # (docs/spec.md section 13). Refuse rather than mislabel.
            raise ValueError(f"the rollout continuation has no belief rung, so belief={budget.belief!r} would mislabel it; use belief='prior'")
        return PimcRollout(rs, rollout_agents, worlds=budget.worlds, rollouts=budget.rollouts, name=label, seed=budget.seed)
    if budget.continuation == "response":
        opponent = target
    elif budget.continuation == "surrogate":
        if surrogate is None:
            raise ValueError("the surrogate continuation needs `surrogate`: the fixed opponent each world is solved against")
        opponent = surrogate
    else:  # minimax reads nothing about anyone
        opponent = None
    belief = None if budget.belief == "prior" else Belief(rs, kind=budget.belief, opponent=target if budget.belief == "window" else None)
    if leaf is None:
        raise ValueError("`leaf` is required for a search attacker: pass a value function, or a depth that always terminates plus a raising leaf")
    return PimcSearch(rs, leaf, worlds=budget.worlds, depth=budget.depth, name=label, seed=budget.seed, opponent=opponent, belief=belief)


def attacker_seeds(rs: Ruleset, target, budget: Budget, seeds, **kw) -> list:
    """The same attack at several world-sampling seeds: one attacker per seed.

    The spread across these is the attacker's *own* uncertainty, independent of the deals, and it
    is not small (0.58-0.84 score points at w = 4 on `TOY`). Reporting one seed as "the" attack
    score hides it."""
    return [build(rs, target, replace(budget, seed=s), **kw) for s in seeds]


# -- the reportable bound -------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NashConvBound:
    """A two-seat NashConv lower bound in **score points**, from paired per-deal attack scores.

    `b0 + b1 - 1` bounds NashConv because the two unknown Nash seat values sum to 1 in this
    constant-sum score convention and therefore cancel. Zero is always a valid lower bound, so a
    negative point estimate is a *weak attack*, never evidence that the target is unexploitable --
    `reported` clamps at zero while `estimate` keeps the raw number visible."""

    n: int
    b0: float  # attacker's mean score at seat 0
    b1: float  # attacker's mean score at seat 1
    estimate: float  # 100 * (b0 + b1 - 1)
    se: float  # of the same quantity, from the per-deal sum (so the seat covariance is carried)
    z: float  # the one-sided confidence multiplier used for `lower`
    lower: float  # estimate - z * se
    corr: float  # correlation between the two seats' per-deal scores

    @property
    def reported(self) -> float:
        """The lower bound as it should be printed: never below the trivial bound of zero."""
        return max(0.0, self.lower)

    def __str__(self) -> str:
        return (
            f"b0 {self.b0:.4f} b1 {self.b1:.4f}  NashConv >= {self.estimate:+.2f} +- {self.se:.2f} pts "
            f"(one-sided {self.z:g} sigma lower bound {self.lower:+.2f}, reported {self.reported:.2f}; corr {self.corr:+.2f}, n {self.n})"
        )


def nashconv_bound(b0, b1, z: float = 1.96) -> NashConvBound:
    """The `NashConv >= b0 + b1 - 1` bound from two paired per-deal score arrays.

    `b0[i]` and `b1[i]` must come from the **same deal** `i` -- the two seatings of one mirrored
    pair. The standard error is taken from the per-deal *sum*, which is what carries the covariance
    whatever its sign, so the seats are never assumed independent. The measured correlation is mild
    and not always negative: -0.30 on `TOY` and -0.10 on `FULL_UNO` for a rulebot row
    (docs/results/index.md), and +0.10 +- 0.16 for an attacker against Random over 40 full-UNO deals.
    So this is not a large correction in practice -- it is simply the one that needs no assumption."""
    x, y = np.asarray(b0, dtype=float), np.asarray(b1, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"the two seats were not scored on the same deals: {x.shape} against {y.shape}")
    if len(x) < 2:
        raise ValueError("a standard error needs at least two deals")
    total = x + y
    se = float(total.std(ddof=1) / math.sqrt(len(total)))
    est = float(total.mean() - 1.0)
    corr = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")
    return NashConvBound(
        n=len(x),
        b0=float(x.mean()),
        b1=float(y.mean()),
        estimate=100 * est,
        se=100 * se,
        z=z,
        lower=100 * (est - z * se),
        corr=corr,
    )
