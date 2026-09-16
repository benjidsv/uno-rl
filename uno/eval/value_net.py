"""A value network for AIVAT, fitted on PIMC rollout targets (step 9.5c, docs/spec.md section 24).

**What it may see**: the L2 state, as the critic (`uno.eval.critic`): both hands, the discard (so the
pile as a multiset), top card and colour, mover, phase, the drawn card, pile size, reshuffles. Never
the pile order.

**Symmetric by construction.** Features come in a global block and one block per colour; the
global block only uses colour- and rank-free quantities (sizes, wild counts, cards matching the top
card's rank or kind, ranks held in two or more colours), and the colour blocks are processed by one
shared network and pooled by sum and max (DeepSets). Renaming colours permutes the blocks, renaming
ranks within a copy class changes no feature, so the value is invariant under `uno.game.symmetry`
and quotient-tree moments equal the full game's (up to float rounding of the pooled sum). Pooling
also makes the architecture independent of the number of colours.

**Frame.** Hands in the seat-0 frame, output the seat-0 return in [-1, 1] (tanh). One network per
profile (row, column, seat), as the critic.

**Inference in float64 on CPU**, so V is a deterministic function of the state whatever else is
in the batch (AIVAT's unbiasedness wants V fixed before the chance event; float32 kernels can depend
on the batch). Training is float32.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
import torch
from torch import nn

from uno.game.engine import Phase, State, is_playable
from uno.game.observe import Observation, discard_from_public
from uno.game.rules import Kind, Ruleset

COLOURED = (Kind.NUMBER, Kind.SKIP, Kind.REVERSE, Kind.DRAW_TWO)
B = 17  # features per colour block
G = 44  # global features


@cache
def _tables(rs: Ruleset) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    types = rs.card_types
    return (
        tuple(int(t.kind) for t in types),
        tuple(-1 if t.color is None else t.color for t in types),
        tuple(-1 if t.rank is None else t.rank for t in types),
    )


def layout(rs: Ruleset) -> tuple[int, int, int]:
    """(global features, features per colour, colours): a row is the global block then the colour blocks."""
    return G, B, rs.colors


def position_features(rs: Ruleset, hands, discard, top: int, color: int, player: int, phase: Phase, drawn: int | None, pile: int, reshuffles: int) -> list[float]:
    kind, col, rank = _tables(rs)
    deck = rs.deck_size
    blocks = [[0.0] * B for _ in range(rs.colors)]
    glob: list[float] = []
    top_kind, top_rank = kind[top], rank[top]
    for h, hand in enumerate(hands):
        wild = four = playable = match_rank = match_kind = 0
        colours: dict[int, int] = {}
        rank_colours: dict[int, set] = {}
        for c in hand:
            k, cc = kind[c], col[c]
            p = is_playable(rs, c, top, color)
            playable += p
            if cc < 0:
                wild += k == Kind.WILD
                four += k == Kind.WILD_DRAW_FOUR
                continue
            blk = blocks[cc]
            blk[5 * h + k] += 1.0
            blk[5 * h + 4] += p
            colours[cc] = colours.get(cc, 0) + 1
            if k == Kind.NUMBER:
                rank_colours.setdefault(rank[c], set()).add(cc)
                match_rank += top_kind == Kind.NUMBER and rank[c] == top_rank and cc != color
            else:
                match_kind += k == top_kind and cc != color
        n = len(hand)
        glob += [n / 10.0, float(wild), float(four), float(playable), float(match_rank), float(match_kind),
                 float(sum(len(v) >= 2 for v in rank_colours.values())), float(len(colours)), float(max(colours.values(), default=0)),
                 float(n == 1), float(n == 2)]  # fmt: skip
    dwild = dfour = 0
    for c in discard:
        if col[c] < 0:
            dwild += kind[c] == Kind.WILD
            dfour += kind[c] == Kind.WILD_DRAW_FOUR
        else:
            blocks[col[c]][10 + kind[c]] += 1.0
    for c in range(rs.colors):
        blk = blocks[c]
        blk[14] = float(c == color)
        blk[15] = float(col[top] == c)
        blk[16] = float(drawn is not None and col[drawn] == c)
    top_onehot = [0.0] * 6
    top_onehot[top_kind] = 1.0
    drawn_onehot = [0.0] * 6
    if drawn is not None:
        drawn_onehot[kind[drawn]] = 1.0
    glob += [float(player == 0), float(phase is Phase.AFTER_DRAW), pile / deck, float(pile == 0), len(discard) / deck, min(reshuffles, 3) / 3.0]
    glob += top_onehot + [float(drawn is not None)] + drawn_onehot
    glob += [float(dwild), float(dfour), (len(hands[0]) - len(hands[1])) / 10.0]
    assert len(glob) == G
    return glob + [x for blk in blocks for x in blk]


def state_features(s: State) -> list[float]:
    """Module-level (it crosses processes in `uno.eval.pimc.teacher_targets`)."""
    return position_features(s.ruleset, s.hands, s.discard, s.top, s.current_color, s.player, s.phase, s.drawn, len(s.draw_pile), s.n_reshuffles)


def observation_features(rs: Ruleset, obs: Observation) -> list[float]:
    """From an L2 observation at the observer's own decision (`observation_from_key`)."""
    hands = [None, None]
    hands[obs.seat], hands[obs.seat ^ 1] = obs.hand, obs.opp_hand
    reshuffles = sum(1 for ev in obs.public if ev[0] == "reshuffle")
    return position_features(rs, hands, discard_from_public(obs.public), obs.top, obs.current_color, obs.player, obs.phase, obs.drawn, obs.pile_size, reshuffles)


# -- the network --------------------------------------------------------------------------------


class Net(nn.Module):
    def __init__(self, g: int, b: int, c: int, width: int) -> None:
        super().__init__()
        self.g, self.b, self.c = g, b, c
        self.phi = nn.Sequential(nn.Linear(b + g, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU())
        self.rho = nn.Sequential(nn.Linear(2 * width + g, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = x[:, : self.g]
        blocks = x[:, self.g :].reshape(-1, self.c, self.b)
        h = self.phi(torch.cat((blocks, g.unsqueeze(1).expand(-1, self.c, -1)), dim=2))
        return torch.tanh(self.rho(torch.cat((h.sum(1), h.amax(1), g), dim=1))).squeeze(1)


def standardizer(rs: Ruleset, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature mean and scale, pooled over colours for the colour blocks (so that standardizing
    keeps the symmetry)."""
    g, b, c = layout(rs)
    x = np.asarray(x, dtype=np.float64)
    mg, sg = x[:, :g].mean(axis=0), x[:, :g].std(axis=0)
    cb = x[:, g:].reshape(-1, c, b)
    mb, sb = cb.mean(axis=(0, 1)), cb.std(axis=(0, 1))
    mean = np.concatenate([mg, np.tile(mb, c)])
    scale = np.concatenate([sg, np.tile(sb, c)])
    scale[scale == 0] = 1.0
    return mean, scale


@dataclass
class ValueNet:
    rs: Ruleset
    width: int
    mean: np.ndarray
    scale: np.ndarray
    params: dict[str, torch.Tensor]

    def __post_init__(self) -> None:
        g, b, c = layout(self.rs)
        self.net = Net(g, b, c, self.width).double()
        self.net.load_state_dict({k: v.double() for k, v in self.params.items()})
        self.net.eval()

    def features(self, s: State) -> list[float]:
        return state_features(s)

    def observation_features(self, rs: Ruleset, obs: Observation) -> list[float]:
        return observation_features(rs, obs)

    def values(self, rows) -> np.ndarray:
        """Seat-0 return in [-1, 1] per feature row, in float64."""
        x = (np.asarray(rows, dtype=np.float64).reshape(-1, len(self.mean)) - self.mean) / self.scale
        with torch.no_grad():
            return self.net(torch.from_numpy(x)).numpy()

    def value(self, s: State) -> float:
        return float(self.values([self.features(s)])[0])

    def save(self, path: str | Path) -> None:
        torch.save({"width": self.width, "mean": self.mean, "scale": self.scale, "params": self.params, "rs": repr(self.rs)}, path)

    @classmethod
    def load(cls, path: str | Path, rs: Ruleset) -> ValueNet:
        d = torch.load(path, weights_only=False)
        if d["rs"] != repr(rs):
            raise ValueError(f"{path} was fitted for another ruleset: {d['rs']}")
        return cls(rs, d["width"], d["mean"], d["scale"], d["params"])


@dataclass(frozen=True)
class NetReport:
    states: int
    epochs: int
    seconds: float
    holdout_mse: float  # against the rollout targets
    holdout_noise: float  # mean variance of those targets' own noise
    holdout_var: float  # variance of the targets
    train_mse: float

    @property
    def r2_targets(self) -> float:
        """R^2 against the noisy targets on held-out games."""
        return 1.0 - self.holdout_mse / self.holdout_var

    @property
    def r2_values(self) -> float:
        """R^2 against the rollout values' expectations (the profile's L2 values): target noise removed
        from both the error and the variance."""
        return 1.0 - (self.holdout_mse - self.holdout_noise) / (self.holdout_var - self.holdout_noise)

    def __str__(self) -> str:
        return (
            f"{self.states:,} states, {self.epochs} epochs in {self.seconds:.0f}s: train mse {self.train_mse:.5f}, holdout mse {self.holdout_mse:.5f} "
            f"(target noise {self.holdout_noise:.5f}, target var {self.holdout_var:.5f}), R^2 vs targets {self.r2_targets:.4f}, vs values {self.r2_values:.4f}"
        )


def fit_value_net(
    rs: Ruleset, x: np.ndarray, y: np.ndarray, game: np.ndarray, noise: np.ndarray | None = None, width: int = 128, batch: int = 1024, lr: float = 1e-3,
    weight_decay: float = 1e-6, max_epochs: int = 200, patience: int = 8, holdout: float = 0.2, seed: int = 0, threads: int | None = None,
) -> tuple[ValueNet, NetReport]:  # fmt: skip
    """MSE on the targets with Adam; the last `holdout` share of the games (by index) is held out
    for early stopping, and the best epoch's weights are kept. Deterministic given `seed` and
    `threads`."""
    t0 = time.perf_counter()
    if threads:
        torch.set_num_threads(threads)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    mean, scale = standardizer(rs, x)
    xs = torch.from_numpy(((np.asarray(x, dtype=np.float64) - mean) / scale).astype(np.float32))
    ys = torch.from_numpy(np.asarray(y, dtype=np.float32))
    cut = int((1.0 - holdout) * (int(game.max()) + 1))
    train = np.nonzero(game < cut)[0]
    hold = np.nonzero(game >= cut)[0]
    g, b, c = layout(rs)
    net = Net(g, b, c, width)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    xh, yh = xs[hold], ys[hold]

    def holdout_mse() -> float:
        net.eval()
        with torch.no_grad():
            out = torch.cat([net(xh[i : i + 65536]) for i in range(0, len(hold), 65536)])
        net.train()
        return float(((out - yh) ** 2).mean())

    best, best_state, bad, epochs, train_mse = math.inf, None, 0, 0, math.nan
    for epoch in range(max_epochs):
        perm = torch.from_numpy(rng.permutation(train))
        tot = 0.0
        for i in range(0, len(perm), batch):
            idx = perm[i : i + batch]
            loss = ((net(xs[idx]) - ys[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        epochs = epoch + 1
        h = holdout_mse()
        if h < best - 1e-7:
            best, bad, train_mse = h, 0, tot / len(train)
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    hn = float(np.mean(noise[hold])) if noise is not None else 0.0
    report = NetReport(len(y), epochs, time.perf_counter() - t0, best, hn, float(np.var(y[hold])), train_mse)
    return ValueNet(rs, width, mean, scale, best_state), report
