"""How much of the opponent's hand does L1 history reveal beyond arms A, B and C?

A supervised ceiling for hand inference, with no RL. A fixed behaviour actor plays the
rulebot league; at every learner decision we record the learner's L1 information and,
as the target, the opponent's true hand (L2, used only as a label). Predictors that
see different inputs are then fitted on the same games and compared on held-out deals:

- `prior`   the counting prior: the opponent's cards are uniform over the unseen cards
            (`uno.eval.search.unseen_cards`). No learning; what A knows by bookkeeping.
- `lin-X`   a linear probe on arm X's declared features (continuity with earlier probes).
- `rel-X`   the actor's own equivariant relational encoder (`uno.rl.model.relational`)
            on arm X's features, with a per-token head: the best the arm's encoding
            supports under the architecture the actor actually uses.
- `gru`/`lstm`/`tf` a sequence model over the **raw** L1 event stream (own initial hand, setup,
            every public event, own private draws aligned to their draw events), plus the
            current position. It sees everything L1 allows, with no engineered summary.
            Trained with random color permutations, since it is not equivariant by design.

Every model predicts a distribution `p` over the 54 card types, as a correction added to
the log counting prior (types with no unseen copy are impossible). The primary metric is
the multinomial log loss in **nats per opponent card**, `-sum_t n_t log p_t / sum_t n_t`,
compared with the prior. Color-count R^2 (expected per-color counts `hand_size * p`) is
reported for continuity with the earlier linear probes (A 0.36, B 0.45, C 0.47).

The comparison that answers "is our encoding leaving information on the table" is the
paired, deal-clustered difference between the best sequence model and `rel-C`. It
measures recoverable *information about the hand* under this behaviour policy and
league, not score points; converting it to score needs a policy that uses it.

    uv run uno belief_ceiling collect --actor logs/..B../checkpoints/final_actor.pt \\
        --deals 4096 --out logs/belief/b4096
    uv run uno belief_ceiling fit --data logs/belief/b4096 --device cuda \\
        --models prior,lin-A,lin-B,lin-C,rel-A,rel-B,rel-C,gru,tf --fractions 0.25,0.5,1
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn

from uno.eval.search import unseen_cards
from uno.game.engine import Phase
from uno.rl.algo.collect import LiveGame
from uno.rl.core import manifest as mf
from uno.rl.core.checkpoint import load_actor
from uno.rl.core.streams import EVAL, uniform
from uno.rl.env.features import ARM_A, ARM_B, ARM_C, extract, legal_mask, spec
from uno.rl.eval.opponents import league
from uno.rl.model.relational import Encoder

# Raw event tokens: (kind, actor, card, color, n) small integers.
PAD, INIT, SETUP, SKIP, PLAY, DRAW, OWN_DRAW, RESHUFFLE = range(8)
N_KINDS = 8
NOBODY, SELF, OPP = 0, 1, 2
MAX_N = 15
# Data starts away from the diagnostic panel's first deals, which paired arm
# comparisons (`arm_b_diag`, 2,048 deals/cell) already use.
DEFAULT_START = mf.DIAGNOSTIC_DEALS.start + 50_000
SCALARS = 5  # opp hand, own hand, pile, turn, after-draw
SEQUENCE = ("gru", "lstm", "tf")
HYBRID_CORES = ("gru", "lstm", "tf", "none")  # `rel-C+none` is the capacity control: no history


class Tape:
    """One game's L1 event stream for one seat, extended incrementally.

    Own private draws are placed right after the public draw event they belong to,
    using its actual count, as the observation contract requires for C-style history.
    """

    def __init__(self, rs, seat: int) -> None:
        self.rs, self.seat = rs, seat
        self.none_card, self.none_color = rs.num_types, rs.colors
        self.tokens: list[tuple[int, int, int, int, int]] = []
        self.public_seen = 0
        self.own_seen = 0

    def _who(self, p: int) -> int:
        return SELF if p == self.seat else OPP

    def advance(self, obs) -> int:
        """Append everything new in `obs`; return the prefix length a query may read."""
        nc, ncol = self.none_card, self.none_color
        if not self.tokens:
            self.tokens += [(INIT, SELF, c, ncol, 0) for c in sorted(obs.initial_hand)]
        for ev in obs.public[self.public_seen:]:
            match ev[0]:
                case "setup":
                    self.tokens.append((SETUP, NOBODY, ev[1], ncol, 0))
                    self.tokens += [(SKIP, NOBODY, c, ncol, 0) for c in ev[2]]
                case "play":
                    _, p, card, color = ev
                    self.tokens.append((PLAY, self._who(p), card, color, 0))
                case "draw":
                    _, p, n = ev
                    self.tokens.append((DRAW, self._who(p), nc, ncol, min(n, MAX_N)))
                    if p == self.seat:
                        cards = obs.own_draws[self.own_seen : self.own_seen + n]
                        if len(cards) != n:
                            raise ValueError("own private draws do not align with public draw events")
                        self.own_seen += n
                        self.tokens += [(OWN_DRAW, SELF, c, ncol, 0) for c in cards]
                case "reshuffle":
                    self.tokens.append((RESHUFFLE, NOBODY, nc, ncol, 0))
                case other:
                    raise ValueError(f"unknown public event {other!r}")
        self.public_seen = len(obs.public)
        return len(self.tokens)


def _counts(cards, k: int) -> np.ndarray:
    return np.bincount(np.asarray(cards, dtype=np.int64), minlength=k).astype(np.uint8)


# -- collection -----------------------------------------------------------------


def collect(actor_path: str, deals: int, start: int, out: Path, seed: int = 0, live: int = 64,
            features: bool = True) -> None:
    """Play and record. `features=False` keeps only the raw streams (sequence models only)."""
    agent, meta = load_actor(actor_path)
    rs, policy = agent.rs, agent.actor
    opponents = league(rs, "rulebot")
    names = list(opponents)
    man = mf.diagnostic_panel(opponents, deals, start=start)
    sp_a, sp_b, sp_c = spec(rs, ARM_A), spec(rs, ARM_B), spec(rs, ARM_C)
    k = rs.num_types

    rows: dict[str, list] = {key: [] for key in (
        "game", "pos", "hand", "unseen", "target", "top", "color", "drawn", "scalars",
        "tok", "col", "glob")}
    tapes: dict[int, Tape] = {}
    game_of: dict[int, int] = {}
    games_meta: list[tuple[int, int, int, int]] = []  # deal, opponent index, seat, episode
    game_tokens: list[list] = []  # each game's stream up to its last query
    nested_checked = False

    queue, active = deque(man.episodes), []
    started = time.time()
    while queue or active:
        while queue and len(active) < live:
            e = queue.popleft()
            g = LiveGame(rs, e, opponents[e.opponent])
            game_of[id(g)] = len(games_meta)
            game_tokens.append([])
            games_meta.append((e.deal_seed, names.index(e.opponent), e.seat, e.episode_id))
            tapes[id(g)] = Tape(rs, e.seat)
            active.append(g)
        for g in active:
            g.advance_to_learner(seed)
        done = [g for g in active if g.terminal]
        for g in done:
            game_tokens[game_of.pop(id(g))] = tapes.pop(id(g)).tokens
        active = [g for g in active if not g.terminal]
        if not active:
            continue

        pairs = [g.observations() for g in active]
        obs = [p[0] for p in pairs]
        evs = [g.evidence for g in active]
        feats_b = extract(sp_b, obs, evs)
        feats_c = extract(sp_c, obs, evs) if features else None
        if features and not nested_checked:
            # A and B must be channel prefixes of C, or slicing C would mislabel an arm.
            feats_a = extract(sp_a, obs, evs)
            for sub, s in ((feats_a, sp_a), (feats_b, sp_b)):
                ok = (np.array_equal(sub.tok, feats_c.tok[..., : s.tok_width])
                      and np.array_equal(sub.col, feats_c.col[..., : s.col_width])
                      and np.array_equal(sub.glob, feats_c.glob[..., : s.glob_width]))
                if not ok:
                    raise ValueError(f"arm {s.arm} features are not a prefix of arm C's")
            nested_checked = True

        for g, (o, o2) in zip(active, pairs):
            unseen = _counts(unseen_cards(rs, o)[0], k)
            target = _counts(o2.opp_hand, k)
            if np.any(target > unseen):
                raise ValueError("an opponent card is outside the L1 unseen set")
            rows["game"].append(game_of[id(g)])
            rows["pos"].append(tapes[id(g)].advance(o))
            rows["hand"].append(_counts(o.hand, k))
            rows["unseen"].append(unseen)
            rows["target"].append(target)
            rows["top"].append(o.top)
            rows["color"].append(o.current_color)
            after = o.phase is Phase.AFTER_DRAW
            rows["drawn"].append(o.drawn if after and o.drawn is not None else k)
            rows["scalars"].append((o.opp_hand_size / rs.hand_size, len(o.hand) / rs.hand_size,
                                    o.pile_size / rs.deck_size, o.turn / 200.0, float(after)))
        if features:
            rows["tok"].append(feats_c.tok.astype(np.float16))
            rows["col"].append(feats_c.col.astype(np.float16))
            rows["glob"].append(feats_c.glob.astype(np.float16))

        masks = np.stack([legal_mask(rs, o) for o in obs])
        logp = policy.log_probs(feats_b if policy.spec.arm == ARM_B else extract(policy.spec, obs, evs), masks)
        us = np.array([uniform(seed, EVAL, g.episode.episode_id, g.decision_index) for g in active])
        for g, a in zip(active, policy.sample(logp, us)):
            g.apply(int(a))
        n_rows = len(rows["game"])
        if n_rows and n_rows % 20_000 < len(active):
            print(f"  {n_rows:,} decisions, {len(games_meta):,}/{len(man.episodes):,} games started "
                  f"({time.time() - started:.0f}s)", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    lengths = np.array([len(t) for t in game_tokens], dtype=np.int64)
    np.save(out / "token_offsets.npy", np.concatenate([[0], np.cumsum(lengths)]))
    np.save(out / "tokens.npy", np.asarray([x for t in game_tokens for x in t], dtype=np.uint8).reshape(-1, 5))
    np_rows = {key: np.asarray(v) for key, v in rows.items() if key not in ("tok", "col", "glob")}
    for key in ("tok", "col", "glob") if features else ():
        np.save(out / f"feat_{key}.npy", np.concatenate(rows[key]))
    for key, v in np_rows.items():
        np.save(out / f"{key}.npy", v)
    np.save(out / "games.npy", np.asarray(games_meta, dtype=np.int64))
    info = {
        "actor": str(actor_path), "actor_name": meta.get("name"), "actor_update": meta.get("update"),
        "actor_decisions": meta.get("trained_decisions"), "ruleset": repr(rs),
        "deals_per_cell": deals, "start": start, "seed": seed, "opponents": names,
        "schemas": {s.arm: s.hash for s in (sp_a, sp_b, sp_c)},
        "widths": {s.arm: [s.tok_width, s.col_width, s.glob_width] for s in (sp_a, sp_b, sp_c)},
        "features": features, "games": len(games_meta), "decisions": int(len(np_rows["game"])),
        "seconds": time.time() - started,
    }
    (out / "meta.json").write_text(json.dumps(info, indent=2))
    print(f"collected {info['decisions']:,} decisions from {info['games']:,} games "
          f"in {info['seconds']:.0f}s -> {out}")


# -- data -----------------------------------------------------------------------


class Data:
    """A collected dataset, with a deal-level train/val/test split.

    Splitting by deal keeps the eight cells that share a deal on one side, so no
    held-out deal was seen in training under another opponent or seat.
    """

    def __init__(self, path: Path, test_mod: int = 10) -> None:
        self.path = path
        self.meta = json.loads((path / "meta.json").read_text())
        load = lambda name, mmap=None: np.load(path / f"{name}.npy", mmap_mode=mmap)
        for key in ("game", "pos", "hand", "unseen", "target", "top", "color", "drawn", "scalars"):
            setattr(self, key, load(key))
        self.features = self.meta.get("features", True)
        self.tok, self.col, self.glob = (
            (load(f"feat_{k}", "r") for k in ("tok", "col", "glob")) if self.features else (None, None, None))
        self.games = load("games")
        self.tokens = load("tokens")
        self.offsets = load("token_offsets")
        deal = self.games[:, 0]
        bucket = deal % test_mod
        split_of_game = np.where(bucket == 0, 2, np.where(bucket == 1, 1, 0))  # 0 train, 1 val, 2 test
        self.split = split_of_game[self.game]
        self.deal = deal[self.game]

    def rows(self, split: int, fraction: float = 1.0, seed: int = 0) -> np.ndarray:
        idx = np.flatnonzero(self.split == split)
        if fraction < 1.0:
            deals = np.unique(self.deal[idx])
            keep = np.random.default_rng(seed).permutation(deals)[: max(1, int(len(deals) * fraction))]
            idx = idx[np.isin(self.deal[idx], keep)]
        return idx


def color_perm_types(rs, perm: np.ndarray) -> np.ndarray:
    """Type map of a color permutation (`perm[c]` is c's new color); wilds and 'none' fixed."""
    k = rs.num_types
    out = np.arange(k + 1)
    colored = np.arange(rs.colors * 13) if k == rs.colors * 13 + 2 else None
    if colored is None:
        raise ValueError("color permutation assumes the FULL_UNO type layout")
    out[colored] = perm[colored // 13] * 13 + colored % 13
    return out


# -- models ---------------------------------------------------------------------


def masked_log_probs(correction: torch.Tensor, unseen: torch.Tensor) -> torch.Tensor:
    """Log p over card types: the counting prior times exp(correction), impossible types excluded."""
    base = torch.where(unseen > 0, unseen.clamp_min(1).log(), torch.full_like(unseen, -math.inf))
    return torch.log_softmax(base + correction, dim=-1)


def card_nll(logp: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-row nats per opponent card."""
    ll = torch.where(target > 0, target * logp, torch.zeros_like(logp)).sum(-1)
    return -ll / target.sum(-1).clamp_min(1)


class Prior(nn.Module):
    def forward(self, batch):
        return torch.zeros_like(batch["unseen"])


class Linear(nn.Module):
    def __init__(self, widths, n_tokens: int, colors: int, k: int) -> None:
        super().__init__()
        self.widths = widths
        t, c, g = widths
        self.layer = nn.Linear(n_tokens * t + colors * c + g, k)

    def forward(self, batch):
        t, c, g = self.widths
        x = torch.cat([batch["tok"][..., :t].flatten(1), batch["col"][..., :c].flatten(1), batch["glob"][..., :g]], 1)
        return self.layer(x)


class Relational(nn.Module):
    """The actor's encoder on one arm's features, with a per-token hand head."""

    def __init__(self, sp, width: int, blocks: int) -> None:
        super().__init__()
        self.sp = sp
        self.encoder = Encoder(sp, width, blocks)
        self.grid_head = nn.Linear(width, 1)
        self.wild_head = nn.Linear(width, 1)
        lay = sp.layout
        order = np.concatenate([lay.grid_type.reshape(-1), lay.wild_type])  # token -> type
        self.register_buffer("type_to_token", torch.from_numpy(np.argsort(order)))

    def forward(self, batch):
        sp = self.sp
        grid, wild, _, _ = self.encoder(batch["tok"][..., : sp.tok_width], batch["col"][..., : sp.col_width],
                                        batch["glob"][..., : sp.glob_width])
        z = torch.cat([self.grid_head(grid).flatten(1), self.wild_head(wild).squeeze(-1)], 1)
        return z[:, self.type_to_token]


class Sequence(nn.Module):
    """GRU, LSTM or causal transformer over the raw event stream, queried at each decision.

    `kind="none"` reads no events (its history state is zero): with the same query and
    head it is the control that isolates extra capacity from extra information.
    """

    def __init__(self, rs, kind: str, width: int = 256, layers: int = 2, max_len: int = 4096) -> None:
        super().__init__()
        k, ncol = rs.num_types, rs.colors
        self.kind, self.k, self.width = kind, k, width
        self.e_kind = nn.Embedding(N_KINDS, width)
        self.e_actor = nn.Embedding(3, width)
        self.e_card = nn.Embedding(k + 1, width)
        self.e_color = nn.Embedding(ncol + 1, width)
        self.e_n = nn.Embedding(MAX_N + 1, width)
        if kind in ("gru", "lstm"):  # LSTM: on MPS, GRU is ~3x slower (torch 2.14)
            self.core = (nn.GRU if kind == "gru" else nn.LSTM)(width, width, layers, batch_first=True)
        elif kind == "tf":
            self.e_pos = nn.Embedding(max_len, width)
            block = nn.TransformerEncoderLayer(width, 4, 4 * width, dropout=0.0, batch_first=True, norm_first=True)
            self.core = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        query = 2 * k + k + ncol + (k + 1) + SCALARS
        self.query = nn.Sequential(nn.Linear(query, width), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(2 * width, 2 * width), nn.ReLU(), nn.Linear(2 * width, k))

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.kind == "none":
            return torch.zeros(*tokens.shape[:2], self.width, device=tokens.device)
        t = tokens.long()
        x = (self.e_kind(t[..., 0]) + self.e_actor(t[..., 1]) + self.e_card(t[..., 2])
             + self.e_color(t[..., 3]) + self.e_n(t[..., 4]))
        if self.kind in ("gru", "lstm"):
            return self.core(x)[0]
        n = x.shape[1]
        if n > self.e_pos.num_embeddings:
            raise ValueError(f"a game has {n} tokens, more than the transformer's {self.e_pos.num_embeddings}")
        x = x + self.e_pos(torch.arange(n, device=x.device))
        causal = torch.triu(torch.full((n, n), float("-inf"), device=x.device), 1)
        return self.core(x, mask=causal, is_causal=True)

    def forward(self, batch):
        h = self.encode(batch["tokens"])  # (games, T, w)
        h = h[batch["row_game"], batch["pos"] - 1]  # state after the last token a query may read
        k = self.k
        f = torch.nn.functional
        q = torch.cat([batch["hand"] / 2, batch["unseen"] / 2, f.one_hot(batch["top"], k).float(),
                       f.one_hot(batch["color"], self.e_color.num_embeddings - 1).float(),
                       f.one_hot(batch["drawn"], k + 1).float(), batch["scalars"]], 1)
        return self.head(torch.cat([h, self.query(q)], 1))


class Hybrid(nn.Module):
    """`rel-C` plus a sequence correction, zero-initialized so training starts at `rel-C`."""

    def __init__(self, rel: Relational, seq: Sequence) -> None:
        super().__init__()
        self.rel, self.seq = rel, seq
        nn.init.zeros_(seq.head[-1].weight)
        nn.init.zeros_(seq.head[-1].bias)

    def forward(self, batch):
        return self.rel(batch) + self.seq(batch)


def uses_games(name: str) -> bool:
    return name in SEQUENCE or "+" in name


def uses_features(name: str) -> bool:
    return name != "prior" and name not in SEQUENCE


# -- batching -------------------------------------------------------------------


def row_batch(data: Data, idx: np.ndarray, device) -> dict:
    idx = np.sort(idx)
    t = lambda a, dtype=torch.float32: torch.as_tensor(np.asarray(a[idx]), dtype=dtype, device=device)
    return {"tok": t(data.tok), "col": t(data.col), "glob": t(data.glob),
            "unseen": t(data.unseen), "target": t(data.target), "rows": idx}


def game_batch(data: Data, games: np.ndarray, rows_of_game: dict, device, rng=None,
               features: bool = False) -> dict:
    """Whole games (padded token streams) plus all their query rows, optionally color-permuted.

    Hybrids also read arm C's feature rows, which are not permuted, so they are never
    augmented (`rng` is used for them only to shuffle games).
    """
    rs_types = data.meta["_rs_types"]
    idx = np.concatenate([rows_of_game[g] for g in games])
    lens = np.array([data.offsets[g + 1] - data.offsets[g] for g in games])
    tokens = np.zeros((len(games), max(1, lens.max()), 5), dtype=np.int64)
    for i, g in enumerate(games):
        tokens[i, : lens[i]] = data.tokens[data.offsets[g] : data.offsets[g + 1]]
    local = {g: i for i, g in enumerate(games)}
    row_game = np.array([local[g] for g in data.game[idx]])
    hand, unseen, target = data.hand[idx], data.unseen[idx], data.target[idx]
    top, color, drawn = data.top[idx].astype(np.int64), data.color[idx].astype(np.int64), data.drawn[idx].astype(np.int64)
    if rng is not None and not features:  # one color permutation per batch: every field moves together
        perm = rng.permutation(rs_types.colors)
        tmap = color_perm_types(rs_types, perm)
        inv = np.argsort(tmap[:-1])
        tokens[..., 2] = tmap[tokens[..., 2]]
        cmap = np.append(perm, rs_types.colors)
        tokens[..., 3] = cmap[tokens[..., 3]]
        hand, unseen, target = hand[:, inv], unseen[:, inv], target[:, inv]
        top, drawn, color = tmap[top], tmap[drawn], perm[color]
    t = lambda a, dtype=torch.float32: torch.as_tensor(np.asarray(a), dtype=dtype, device=device)
    out = {"tokens": t(tokens, torch.long), "row_game": t(row_game, torch.long),
           "pos": t(data.pos[idx], torch.long), "hand": t(hand), "unseen": t(unseen), "target": t(target),
           "top": t(top, torch.long), "color": t(color, torch.long), "drawn": t(drawn, torch.long),
           "scalars": t(data.scalars[idx]), "rows": idx}
    if features:
        out.update(tok=t(data.tok[idx]), col=t(data.col[idx]), glob=t(data.glob[idx]))
    return out


def batches(data: Data, model_name: str, rows: np.ndarray, device, rng=None, row_bs: int = 1024, game_bs: int = 64):
    """Yield batches covering `rows` exactly once (shuffled when `rng` is given)."""
    if uses_games(model_name):
        games = np.unique(data.game[rows])
        rows_of_game = {}
        order = rows[np.argsort(data.game[rows], kind="stable")]
        cuts = np.flatnonzero(np.diff(data.game[order])) + 1
        for chunk in np.split(order, cuts):
            rows_of_game[int(data.game[chunk[0]])] = chunk
        if rng is not None:
            games = rng.permutation(games)
        for i in range(0, len(games), game_bs):
            yield game_batch(data, games[i : i + game_bs], rows_of_game, device, rng, uses_features(model_name))
    else:
        order = rng.permutation(rows) if rng is not None else rows
        for i in range(0, len(order), row_bs):
            yield row_batch(data, order[i : i + row_bs], device)


def build(name: str, rs, width: int, blocks: int):
    if name == "prior":
        return Prior()
    if name in SEQUENCE:
        return Sequence(rs, name)
    if "+" in name:
        rel_name, core = name.split("+")
        if rel_name != "rel-C" or core not in HYBRID_CORES:
            raise ValueError(f"hybrids are rel-C+{{{','.join(HYBRID_CORES)}}}, not {name!r}")
        return Hybrid(build(rel_name, rs, width, blocks), Sequence(rs, core))
    kind, arm = name.split("-")
    sp = spec(rs, arm)
    if kind == "lin":
        return Linear((sp.tok_width, sp.col_width, sp.glob_width), sp.layout.n_tokens, sp.layout.colors, rs.num_types)
    if kind == "rel":
        return Relational(sp, width, blocks)
    raise ValueError(f"unknown model {name!r}")


# -- fitting and evaluation -----------------------------------------------------


@torch.no_grad()
def predict(model, data: Data, name: str, rows: np.ndarray, device) -> tuple[np.ndarray, np.ndarray]:
    """(per-row nats/card, per-row log p) in `rows` order."""
    model.eval()
    nll = np.empty(len(data.game), dtype=np.float64)
    logp = np.empty((len(data.game), data.unseen.shape[1]), dtype=np.float32)
    for b in batches(data, name, rows, device):
        lp = masked_log_probs(model(b), b["unseen"])
        nll[b["rows"]] = card_nll(lp, b["target"]).float().cpu().numpy()
        logp[b["rows"]] = lp.float().cpu().numpy()
    return nll[rows], logp[rows]


def fit(model, data: Data, name: str, train: np.ndarray, val: np.ndarray, device, epochs: int,
        lr: float, patience: int, seed: int) -> dict:
    if not any(p.requires_grad for p in model.parameters()):
        return {"epochs": 0}
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    best, best_state, bad, history = math.inf, None, 0, []
    for epoch in range(epochs):
        model.train()
        started = time.time()
        for b in batches(data, name, train, device, rng):
            loss = card_nll(masked_log_probs(model(b), b["unseen"]), b["target"]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        v = float(predict(model, data, name, val, device)[0].mean())
        history.append(v)
        print(f"    epoch {epoch + 1}: val {v:.5f} nats/card ({time.time() - started:.0f}s)", flush=True)
        if v < best - 1e-5:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return {"epochs": len(history), "val_history": history, "best_val": best}


def color_r2(data: Data, rows: np.ndarray, logp: np.ndarray, colors: int) -> tuple[float, float]:
    """R^2 of expected per-color counts (pooled over colors) and of the wild count."""
    n = data.target[rows].sum(1, keepdims=True).astype(np.float64)
    expected = n * np.exp(logp.astype(np.float64))
    truth = data.target[rows].astype(np.float64)
    per_color = lambda a: a[:, : colors * 13].reshape(len(a), colors, 13).sum(2)
    r2 = lambda y, f: 1.0 - ((y - f) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return r2(per_color(truth), per_color(expected)), r2(truth[:, colors * 13 :].sum(1), expected[:, colors * 13 :].sum(1))


def clustered(diff: np.ndarray, clusters: np.ndarray) -> tuple[float, float]:
    """Mean of a per-row quantity and its SE with rows clustered by deal."""
    _, inv = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inv, weights=diff)
    counts = np.bincount(inv)
    mean = sums.sum() / counts.sum()
    resid = sums - mean * counts
    se = math.sqrt((resid ** 2).sum()) / counts.sum()
    return mean, se


def run_fit(args) -> None:
    from uno.game.rules import NAMED

    data = Data(Path(args.data))
    rs = next(r for r in NAMED.values() if repr(r) == data.meta["ruleset"])
    data.meta["_rs_types"] = rs
    tdata = None
    if args.train_data:
        # A larger stream-only dataset for models without features. Its split uses the
        # same deal rule, so none of its training deals is one of `data`'s test deals.
        tdata = Data(Path(args.train_data))
        tdata.meta["_rs_types"] = rs
        for key in ("actor", "start", "seed", "ruleset"):
            if tdata.meta[key] != data.meta[key]:
                raise ValueError(f"--train-data differs from --data in {key}")
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)
    train_all, val, test = data.rows(0), data.rows(1), data.rows(2)
    print(f"{data.meta['decisions']:,} decisions from {data.meta['games']:,} games; "
          f"train/val/test rows {len(train_all):,}/{len(val):,}/{len(test):,}; device {device}")
    names = args.models.split(",")
    fractions = [float(f) for f in args.fractions.split(",")]
    results, test_nll = {}, {}
    stamp = int(time.time())

    def save() -> None:
        """Written after every model, so a stopped run keeps everything it finished."""
        np.savez(Path(args.data) / f"test_nll_{stamp}.npz", rows=test, deal=data.deal[test], **test_nll)
        (Path(args.data) / f"fit_{stamp}.json").write_text(json.dumps(
            {"args": vars(args), "meta": {k: v for k, v in data.meta.items() if k != "_rs_types"},
             "results": results, "paired": comparisons}, indent=2, default=float))

    comparisons = {}
    for name in names:
        for fraction in (fractions if name != "prior" else [1.0]):
            label = name if fraction == 1.0 else f"{name}@{fraction:g}"
            source = tdata if tdata is not None and not uses_features(name) else data
            train = source.rows(0, fraction, args.seed)
            print(f"\n[{label}] {len(train):,} training rows from {source.path}", flush=True)
            model = build(name, rs, args.width, args.blocks).to(device)
            started = time.time()
            lr = args.seq_lr if name in SEQUENCE else args.lr
            info = fit(model, source, name, train, source.rows(1), device, args.epochs, lr, args.patience, args.seed)
            nll, logp = predict(model, data, name, test, device)
            r2c, r2w = color_r2(data, test, logp, rs.colors)
            info.update(test_nll=float(nll.mean()), color_r2=r2c, wild_r2=r2w, train_rows=len(train),
                        params=sum(p.numel() for p in model.parameters()), seconds=time.time() - started)
            results[label], test_nll[label] = info, nll
            print(f"  test {nll.mean():.5f} nats/card, color R^2 {r2c:.3f}, wild R^2 {r2w:.3f}")
            save()

    deal = data.deal[test]
    print("\n== test, paired by row and clustered by deal (nats per opponent card; lower is better)")
    base = test_nll.get("prior")
    print(f"{'model':14s} {'nats/card':>10s} {'gain vs prior':>16s} {'color R2':>9s} {'wild R2':>8s}")
    for label, nll in test_nll.items():
        gain = "" if base is None else "%+.5f +- %.5f" % clustered(base - nll, deal)
        r = results[label]
        print(f"{label:14s} {nll.mean():10.5f} {gain:>16s} {r['color_r2']:9.3f} {r['wild_r2']:8.3f}")
    for seq in [x for x in test_nll if uses_games(x.split("@")[0])]:
        for feat in [x for x in test_nll if x.startswith("rel-") and "@" not in x and "+" not in x]:
            m, se = clustered(test_nll[feat] - test_nll[seq], deal)
            comparisons[f"{seq} - {feat}"] = (m, se)
            line = f"{seq} - {feat}: {m:+.5f} +- {se:.5f} nats/card better"
            if base is not None:
                room = float((base - test_nll[feat]).mean())
                if room > 0:
                    line += f" (G = {m / room:+.1%} of {feat}'s own gain over the prior)"
            print(line)
    save()
    print(f"\nwrote {Path(args.data) / f'fit_{stamp}.json'} and test_nll_{stamp}.npz")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="uno belief_ceiling")
    sub = ap.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--actor", required=True, help="behaviour actor export (final_actor.pt)")
    c.add_argument("--deals", type=int, default=4096, help="deals per cell (8 cells)")
    c.add_argument("--start", type=int, default=DEFAULT_START)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--no-features", action="store_true", help="raw streams only (sequence models)")
    f = sub.add_parser("fit")
    f.add_argument("--data", required=True, help="dataset whose test split every model is scored on")
    f.add_argument("--train-data", help="larger stream-only dataset to train feature-free models on")
    f.add_argument("--models", default="prior,lin-A,lin-B,lin-C,rel-A,rel-B,rel-C,gru")
    f.add_argument("--fractions", default="1", help="training-deal fractions, for a data learning curve")
    f.add_argument("--device", default="auto")
    f.add_argument("--epochs", type=int, default=30)
    f.add_argument("--patience", type=int, default=3)
    f.add_argument("--lr", type=float, default=1e-3)
    f.add_argument("--seq-lr", type=float, default=5e-4)
    f.add_argument("--width", type=int, default=128)
    f.add_argument("--blocks", type=int, default=2)
    f.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.command == "collect":
        collect(args.actor, args.deals, args.start, args.out, args.seed, features=not args.no_features)
    else:
        run_fit(args)


if __name__ == "__main__":
    main()
