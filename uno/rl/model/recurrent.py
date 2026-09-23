"""The recurrent actor: arm X's relational actor plus an equivariant per-token memory.

Carried over from `uno.experiments.belief_ceiling.Recurrent` (`rec-X`), where it
recovered more of the opponent's hand from L1 history than the feed-forward `rel-C`.
Here the same memory feeds the policy and is trained by PPO alone.

At each learner decision the encoder reads arm X's features plus two extra inputs
on every token: the memory left by the previous decision, and `phi` of the counts
of the raw events since then (`uno.rl.env.events`). One `GRUCell`, shared by every
token, then updates the memory from the encoder's token outputs. Everything is per
token with shared weights, and events reach a token only through equalities (same
type, same color, same rank, wild, named color), so relabelling colors or
interchangeable ranks permutes tokens, memories and action probabilities together.

The projection columns that read memory and events start at zero, so a fresh
recurrent actor is a feed-forward actor that has not yet learned to use its memory,
and `load_feedforward` turns a trained `Actor` into exactly the same policy.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch
from torch import nn

from uno.rl.env.events import N_CHANNELS, N_OTHER
from uno.rl.env.features import Spec
from uno.rl.model.relational import BLOCKS, WIDTH, Actor

MEMORY = 64
EVENTS = 32
REL = 4 * N_CHANNELS + 4 + N_OTHER + 1  # per-token event inputs, see `EventExpansion`


class EventExpansion(nn.Module):
    """Per-token inputs from one interval's count tables, by equality only.

    `(..., event_width)` -> `(..., n_tokens, REL)`: for each card channel the count of
    the token's own type, of other types of its color, of other types of its rank, and
    of wilds; plays naming the token's color and naming another color; the draw and
    reshuffle counters; and whether the token is a wild. `log1p` throughout.
    """

    def __init__(self, sp: Spec) -> None:
        super().__init__()
        lay, rs = sp.layout, sp.rs
        self.k, self.colors = rs.num_types, rs.colors
        tok_type = np.concatenate([lay.grid_type.reshape(-1), lay.wild_type])  # token -> type
        tcol, tslot = lay.type_color[tok_type], lay.type_slot[tok_type]
        color_oh = np.zeros((self.k, self.colors), np.float32)
        slot_oh = np.zeros((self.k, lay.slots), np.float32)
        colored = np.flatnonzero(lay.type_color >= 0)
        color_oh[colored, lay.type_color[colored]] = 1.0
        slot_oh[colored, lay.type_slot[colored]] = 1.0
        buffers = {
            "tok_type": torch.from_numpy(tok_type).long(), "tok_color": torch.from_numpy(np.maximum(tcol, 0)).long(),
            "tok_slot": torch.from_numpy(np.maximum(tslot, 0)).long(),
            "tok_colored": torch.from_numpy((tcol >= 0).astype(np.float32)),
            "color_oh": torch.from_numpy(color_oh), "slot_oh": torch.from_numpy(slot_oh),
            "wild": torch.from_numpy((lay.type_color < 0).astype(np.float32)),
        }  # fmt: skip
        for name, value in buffers.items():
            self.register_buffer(name, value, persistent=False)

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        lead, k, c = events.shape[:-1], self.k, self.colors
        cards = events[..., : N_CHANNELS * k].reshape(*lead, N_CHANNELS, k)
        named = events[..., N_CHANNELS * k : N_CHANNELS * k + 2 * c].reshape(*lead, 2, c)
        other = events[..., N_CHANNELS * k + 2 * c :]
        colored = self.tok_colored
        same_type = cards[..., self.tok_type]  # (..., 6, n_tokens)
        same_color = ((cards @ self.color_oh)[..., self.tok_color] - same_type) * colored
        same_slot = ((cards @ self.slot_oh)[..., self.tok_slot] - same_type) * colored
        wilds = (cards @ self.wild)[..., None].expand_as(same_type)
        named_mine = named[..., self.tok_color]  # (..., 2, n_tokens)
        named_same = named_mine * colored
        named_other = (named.sum(-1, keepdim=True) - named_mine) * colored
        per_token = torch.cat([same_type, same_color, same_slot, wilds, named_same, named_other], -2)
        n_tok = len(self.tok_type)
        per_token = per_token.transpose(-1, -2)  # (..., n_tokens, 28)
        per_other = other[..., None, :].expand(*lead, n_tok, N_OTHER)
        t_wild = (1.0 - colored).expand(*lead, n_tok)[..., None]
        return torch.log1p(torch.cat([per_token, per_other, t_wild], -1))


class RecurrentActor(nn.Module):
    """`Actor` on arm X's features plus memory and event channels; see the module doc."""

    recurrent = True

    def __init__(self, sp: Spec, width: int = WIDTH, blocks: int = BLOCKS, memory: int = MEMORY,
                 events: int = EVENTS) -> None:
        super().__init__()
        self.spec, self.width, self.memory, self.events = sp, width, memory, events
        self.n_tokens = sp.layout.n_tokens
        self.body = Actor(replace(sp, tok_width=sp.tok_width + memory + events), width, blocks)
        with torch.no_grad():
            self.body.encoder.proj.weight[:, sp.tok_width : sp.tok_width + memory + events] = 0.0
        self.expand = EventExpansion(sp)
        self.phi = nn.Sequential(nn.Linear(REL, events), nn.ReLU(), nn.Linear(events, events))
        self.cell = nn.GRUCell(width, memory)

    def initial_memory(self, n: int, device=None) -> torch.Tensor:
        return torch.zeros(n, self.n_tokens, self.memory, device=device)

    def event_inputs(self, events: torch.Tensor) -> torch.Tensor:
        """`phi` of the expanded count tables: independent of memory, so batchable."""
        return self.phi(self.expand(events))

    def _step(self, tok, col, glob, e, memory) -> tuple[torch.Tensor, torch.Tensor]:
        n = tok.shape[0]
        x = torch.cat([tok, memory.to(tok.dtype), e.to(tok.dtype)], -1)
        h, hw, color_ctx, global_ctx = self.body.encoder(x, col, glob)
        logits = self.body.scores(h, hw, color_ctx, global_ctx)
        tokens = torch.cat((h.reshape(n, -1, self.width), hw), dim=1)  # token order
        new = self.cell(tokens.reshape(n * self.n_tokens, -1), memory.reshape(n * self.n_tokens, -1).to(tokens.dtype))
        return logits, new.reshape(n, self.n_tokens, self.memory).float()

    def step(self, tok, col, glob, events, memory) -> tuple[torch.Tensor, torch.Tensor]:
        """One decision for a batch of games: `(logits, next memory)`."""
        return self._step(tok, col, glob, self.event_inputs(events), memory)

    def forward(self, tok, col, glob, events, lengths) -> torch.Tensor:
        """Logits for whole episodes, rows in the given order.

        Rows are grouped by episode, each episode's decisions contiguous and in order,
        with `lengths` (host array) the episode sizes in that order. Every episode
        starts from zero memory. The loop runs over decision index on the episodes
        still in progress only (longest first), so nothing is padded.
        """
        lengths = np.asarray(lengths, dtype=np.int64)
        starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
        order = np.argsort(-lengths, kind="stable")
        by_len, first = lengths[order], starts[order]
        steps = int(by_len[0]) if len(by_len) else 0
        active = [int((by_len > i).sum()) for i in range(steps)]
        idx = np.concatenate([first[:a] + i for i, a in enumerate(active)]) if steps else np.zeros(0, np.int64)
        device = tok.device
        idx_t = torch.from_numpy(idx).to(device)
        e = self.event_inputs(events)
        memory = self.initial_memory(len(by_len), device)
        out, offset = [], 0
        for a in active:
            r = idx_t[offset : offset + a]
            logits, memory = self._step(tok[r], col[r], glob[r], e[r], memory[:a])
            out.append(logits)
            offset += a
        if not out:
            return tok.new_zeros((0, self.body.act_token.shape[0]))
        inverse = torch.from_numpy(np.argsort(idx, kind="stable")).to(device)
        return torch.cat(out)[inverse]

    def load_feedforward(self, state: dict) -> None:
        """Start from a trained `Actor` of the same arm: its weights, zero on memory/events."""
        own = self.body.state_dict()
        t, extra = self.spec.tok_width, self.memory + self.events
        for key, value in state.items():
            if key == "encoder.proj.weight":
                new = torch.zeros_like(own[key])
                new[:, :t] = value[:, :t]
                new[:, t + extra :] = value[:, t:]
                own[key] = new
            else:
                own[key] = value
        self.body.load_state_dict(own)


__all__ = ["EVENTS", "MEMORY", "EventExpansion", "RecurrentActor"]
