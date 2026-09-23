"""The feed-forward actor and the separate critic, equivariant by construction.

docs/rl/pipeline-plan.md section 5 specifies the architecture this implements:
width 128, FP32, no dropout or normalization, a shared token projection, two
shared relational blocks with residual updates, and one shared action scorer.
Temperature is 1 everywhere, in collection, learning, evaluation and replay.

**Where the symmetry lives.** Nothing canonicalizes an observation. The inputs
are indexed by color and slot (`uno.rl.env.features`), and every operation here is
one of three kinds:

- *shared per token*: the same weights for every `(color, slot)` and every wild;
- *pooled over a whole axis*: mean and max over colors, over the slots of one
  color, over the slots of one copy class, or over everything -- all of which are
  permutation invariant;
- *indexed scoring*: an action reads its own token and its own color's context.

A slot is embedded only through `slot_class`, so two interchangeable ranks are
indistinguishable to the network and two distinct semantic kinds are not merged.
Consequently relabeling colors or ranks relabels the output probabilities exactly,
including equality inside stabilizer action orbits -- asserted numerically over
the real group in `tests/rl/test_model.py` rather than assumed here.

**What this is not.** A concrete architecture proposal with recorded shapes and
parameter counts, not a proof that it can express a good policy. The critic
shares no parameters, no normalization and no state with the actor, and the actor
export excludes it (contract section 3).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from uno.rl.env.features import Features, Spec

WIDTH = 128
BLOCKS = 2


def _pool(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Mean and max over `dim`, concatenated: a permutation-invariant summary."""
    return torch.cat((x.mean(dim=dim), x.amax(dim=dim)), dim=-1)


class Encoder(nn.Module):
    """Shared token projection plus `BLOCKS` relational blocks.

    Returns `(grid, wild, color_ctx, global_ctx)`:
    `grid` is `(n, colors, slots, width)`, `wild` is `(n, wilds, width)`,
    `color_ctx` is `(n, colors, 2 * width)` and `global_ctx` is
    `(n, 6 * width)` -- the colors pooled twice over and the pooled wilds.
    """

    def __init__(self, sp: Spec, width: int = WIDTH, blocks: int = BLOCKS) -> None:
        super().__init__()
        lay = sp.layout
        self.spec, self.width, self.blocks = sp, width, blocks
        self.colors, self.slots, self.wilds = lay.colors, lay.slots, lay.wilds
        self.embed = nn.Embedding(lay.n_classes, width)
        # one projection for every token; a wild has no color block, so its color
        # features enter as zeros rather than as a second set of weights.
        self.proj = nn.Linear(
            sp.tok_width + sp.col_width + width + sp.glob_width, width
        )
        # grid update sees its own token, its color, its rank across colors, its
        # copy class and the global context: 1 + 2 + 2 + 2 + 6 widths.
        self.grid_blocks = nn.ModuleList(
            nn.Sequential(
                nn.Linear(13 * width, width), nn.ReLU(), nn.Linear(width, width)
            )
            for _ in range(blocks)
        )
        # a wild token has no color and no rank index, so it sees itself and the global context.
        self.wild_blocks = nn.ModuleList(
            nn.Sequential(
                nn.Linear(7 * width, width), nn.ReLU(), nn.Linear(width, width)
            )
            for _ in range(blocks)
        )
        # slot -> compacted copy-class id, for the class pool
        classes = sorted({lay.slot_class[s] for s in range(lay.slots)})
        remap = {c: i for i, c in enumerate(classes)}
        self.n_slot_classes = len(classes)
        self.register_buffer(
            "slot_class",
            torch.tensor(
                [remap[lay.slot_class[s]] for s in range(lay.slots)], dtype=torch.long
            ),
            persistent=False,
        )
        member = torch.zeros(len(classes), lay.slots)
        member[self.slot_class, torch.arange(lay.slots)] = 1.0
        self.register_buffer(
            "class_member", member, persistent=False
        )  # (classes, slots)
        self.register_buffer("class_size", member.sum(1), persistent=False)
        # Small pre-cast copies avoid a dtype-conversion kernel/allocation on every
        # BF16 forward while preventing FP32 promotion of large relational activations.
        self.register_buffer(
            "class_member_bf16", member.to(torch.bfloat16), persistent=False
        )
        self.register_buffer(
            "class_size_bf16", member.sum(1).to(torch.bfloat16), persistent=False
        )
        self.register_buffer(
            "token_class",
            torch.tensor(list(lay.slot_class), dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "wild_class_ids",
            torch.tensor(list(lay.wild_class), dtype=torch.long),
            persistent=False,
        )

    def _class_ctx(self, h: torch.Tensor) -> torch.Tensor:
        """Pool the slots of each copy class, over colors too, and broadcast back.

        `h` is `(n, colors, slots, width)`. Interchangeable ranks share a class, so
        this pool is invariant under the group and distinct semantic kinds keep
        their own summary.
        """
        n, c, s, w = h.shape
        k = self.n_slot_classes
        per_slot = h.permute(0, 2, 1, 3)  # (n, slots, colors, width)
        # Keep static pooling weights in the activation dtype. Under CUDA autocast,
        # ``h`` is typically BF16 after the first linear; mixing it with FP32 buffers
        # would promote the large relational concatenations back to FP32.
        if h.dtype == torch.bfloat16:
            mem = self.class_member_bf16.view(1, k, s, 1)
            class_size = self.class_size_bf16.view(1, k, 1)
        else:
            mem = self.class_member.to(dtype=h.dtype).view(1, k, s, 1)
            class_size = self.class_size.to(dtype=h.dtype).view(1, k, 1)
        means = (per_slot.mean(2).unsqueeze(1) * mem).sum(2) / class_size
        over_colors = per_slot.amax(2).unsqueeze(1).expand(n, k, s, w)
        maxes = torch.where(
            mem.bool(), over_colors, torch.full_like(over_colors, float("-inf"))
        ).amax(2)
        ctx = torch.cat((means, maxes), dim=-1)  # (n, classes, 2w)
        return ctx[:, self.slot_class]  # (n, slots, 2w)

    def forward_reference(self, tok: torch.Tensor, col: torch.Tensor, glob: torch.Tensor):
        """The literal concatenate-then-project encoder: the correctness oracle for
        `forward`, which computes the same function without materializing the
        broadcast concatenations."""
        n = tok.shape[0]
        c, s, wl, w = self.colors, self.slots, self.wilds, self.width
        grid_in = tok[:, : c * s].reshape(n, c, s, -1)
        wild_in = tok[:, c * s :]
        g = glob.unsqueeze(1)
        grid_x = torch.cat(
            (
                grid_in,
                col.unsqueeze(2).expand(n, c, s, col.shape[-1]),
                self
                .embed(self.token_class)
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(n, c, s, w),
                g.unsqueeze(1).expand(n, c, s, glob.shape[-1]),
            ),
            dim=-1,
        )
        wild_x = torch.cat(
            (
                wild_in,
                torch.zeros(n, wl, col.shape[-1], device=tok.device, dtype=tok.dtype),
                self.embed(self.wild_class_ids).unsqueeze(0).expand(n, wl, w),
                g.expand(n, wl, glob.shape[-1]),
            ),
            dim=-1,
        )
        h = torch.relu(self.proj(grid_x))
        hw = torch.relu(self.proj(wild_x))

        for grid_block, wild_block in zip(self.grid_blocks, self.wild_blocks):
            color_ctx = _pool(h, 2)  # (n, colors, 2w)
            rank_ctx = _pool(h, 1)  # (n, slots, 2w)
            class_ctx = self._class_ctx(h)  # (n, slots, 2w)
            wild_ctx = _pool(hw, 1)  # (n, 2w)
            global_ctx = torch.cat(
                (color_ctx.mean(1), color_ctx.amax(1), wild_ctx), dim=-1
            )  # (n, 6w)
            grid_update = grid_block(
                torch.cat(
                    (
                        h,
                        color_ctx.unsqueeze(2).expand(n, c, s, 2 * w),
                        rank_ctx.unsqueeze(1).expand(n, c, s, 2 * w),
                        class_ctx.unsqueeze(1).expand(n, c, s, 2 * w),
                        global_ctx.unsqueeze(1).unsqueeze(1).expand(n, c, s, 6 * w),
                    ),
                    dim=-1,
                )
            )
            wild_update = wild_block(
                torch.cat((hw, global_ctx.unsqueeze(1).expand(n, wl, 6 * w)), dim=-1)
            )
            h = h + grid_update
            hw = hw + wild_update

        color_ctx = _pool(h, 2)
        wild_ctx = _pool(hw, 1)
        global_ctx = torch.cat((color_ctx.mean(1), color_ctx.amax(1), wild_ctx), dim=-1)
        return h, hw, color_ctx, global_ctx

    def forward(self, tok: torch.Tensor, col: torch.Tensor, glob: torch.Tensor):
        """`forward_reference` without the broadcast concatenations.

        A linear layer on a concatenation is the sum of the column blocks applied to
        each piece, and broadcasting commutes with a linear map. So each context is
        projected at its own size -- per color, per slot, per row -- and only then
        broadcast-added. Same parameters and function; only floating-point summation
        order differs (`tests/rl/test_model.py` pins it to the reference).
        """
        n = tok.shape[0]
        c, s, w = self.colors, self.slots, self.width
        sp = self.spec
        Wt, Wc, We, Wg = _weight_blocks(
            self.proj, (sp.tok_width, sp.col_width, w, sp.glob_width)
        )
        tok_p = F.linear(tok, Wt)  # (n, tokens, w)
        col_p = F.linear(col, Wc)  # (n, colors, w); a wild's color block is zero
        glob_p = F.linear(glob, Wg, self.proj.bias)  # (n, w), bias added once
        emb = self.embed.weight
        grid_e = F.linear(emb[self.token_class], We)  # (slots, w)
        wild_e = F.linear(emb[self.wild_class_ids], We)  # (wilds, w)
        h = torch.relu(
            tok_p[:, : c * s].reshape(n, c, s, w)
            + col_p.unsqueeze(2)
            + grid_e
            + glob_p[:, None, None]
        )
        hw = torch.relu(tok_p[:, c * s :] + wild_e + glob_p.unsqueeze(1))

        for grid_block, wild_block in zip(self.grid_blocks, self.wild_blocks):
            color_ctx = _pool(h, 2)  # (n, colors, 2w)
            rank_ctx = _pool(h, 1)  # (n, slots, 2w)
            class_ctx = self._class_ctx(h)  # (n, slots, 2w)
            wild_ctx = _pool(hw, 1)  # (n, 2w)
            global_ctx = torch.cat(
                (color_ctx.mean(1), color_ctx.amax(1), wild_ctx), dim=-1
            )  # (n, 6w)
            first, second = grid_block[0], grid_block[2]
            Wh, Wcol, Wrank, Wcls, Wglob = _weight_blocks(
                first, (w, 2 * w, 2 * w, 2 * w, 6 * w)
            )
            grid_hidden = (
                F.linear(h, Wh)
                + F.linear(color_ctx, Wcol).unsqueeze(2)
                + (F.linear(rank_ctx, Wrank) + F.linear(class_ctx, Wcls)).unsqueeze(1)
                + F.linear(global_ctx, Wglob, first.bias)[:, None, None]
            )
            grid_update = second(torch.relu(grid_hidden))
            wfirst, wsecond = wild_block[0], wild_block[2]
            Wwh, Wwg = _weight_blocks(wfirst, (w, 6 * w))
            wild_update = wsecond(
                torch.relu(
                    F.linear(hw, Wwh) + F.linear(global_ctx, Wwg, wfirst.bias).unsqueeze(1)
                )
            )
            h = h + grid_update
            hw = hw + wild_update

        color_ctx = _pool(h, 2)
        wild_ctx = _pool(hw, 1)
        global_ctx = torch.cat((color_ctx.mean(1), color_ctx.amax(1), wild_ctx), dim=-1)
        return h, hw, color_ctx, global_ctx


def _weight_blocks(linear: nn.Linear, widths: tuple[int, ...]) -> tuple[torch.Tensor, ...]:
    """Column blocks of `linear.weight`, one per concatenated input piece.

    Under autocast the whole weight is cast once and then split: casting each view
    separately would launch one conversion kernel per block per call.
    """
    weight = linear.weight
    if weight.is_cuda and torch.is_autocast_enabled("cuda"):
        weight = weight.to(torch.get_autocast_dtype("cuda"))
    return torch.split(weight, widths, dim=1)


class Actor(nn.Module):
    """Encoder plus the shared action scorer. Outputs raw logits; masking is the
    policy's job (`masked_log_probs`), so one convention serves every path."""

    def __init__(self, sp: Spec, width: int = WIDTH, blocks: int = BLOCKS) -> None:
        super().__init__()
        lay = sp.layout
        self.spec, self.width = sp, width
        self.encoder = Encoder(sp, width, blocks)
        self.score = nn.Sequential(
            nn.Linear(width + 2 * width + 6 * width + lay.act_feat_width, width),
            nn.ReLU(),
            nn.Linear(width, 1),
        )
        self.register_buffer(
            "act_token",
            torch.tensor(lay.act_token, dtype=torch.long).clamp(min=0),
            persistent=False,
        )
        act_has_token = torch.tensor(lay.act_token >= 0, dtype=torch.float32)
        self.register_buffer("act_has_token", act_has_token, persistent=False)
        self.register_buffer(
            "act_has_token_bf16", act_has_token.to(torch.bfloat16), persistent=False
        )
        self.register_buffer(
            "act_color",
            torch.tensor(lay.act_color, dtype=torch.long).clamp(min=0),
            persistent=False,
        )
        act_has_color = torch.tensor(lay.act_color >= 0, dtype=torch.float32)
        self.register_buffer("act_has_color", act_has_color, persistent=False)
        self.register_buffer(
            "act_has_color_bf16", act_has_color.to(torch.bfloat16), persistent=False
        )
        act_feat = torch.tensor(lay.act_feat, dtype=torch.float32)
        self.register_buffer("act_feat", act_feat, persistent=False)
        self.register_buffer(
            "act_feat_bf16", act_feat.to(torch.bfloat16), persistent=False
        )

    def forward_reference(
        self, tok: torch.Tensor, col: torch.Tensor, glob: torch.Tensor
    ) -> torch.Tensor:
        """The literal concatenate-then-score actor: the oracle for `forward`."""
        n = tok.shape[0]
        h, hw, color_ctx, global_ctx = self.encoder.forward_reference(tok, col, glob)
        tokens = torch.cat(
            (h.reshape(n, -1, self.width), hw), dim=1
        )  # (n, n_tokens, w)
        a = self.act_token.shape[0]
        # Match static action buffers to the activation dtype so BF16 autocast does
        # not accidentally promote the activation-heavy scorer inputs back to FP32.
        if tokens.dtype == torch.bfloat16:
            act_has_token = self.act_has_token_bf16
            act_has_color = self.act_has_color_bf16
            act_feat = self.act_feat_bf16
        else:
            act_has_token = self.act_has_token.to(dtype=tokens.dtype)
            act_has_color = self.act_has_color.to(dtype=color_ctx.dtype)
            act_feat = self.act_feat.to(dtype=global_ctx.dtype)
        tok_a = tokens[:, self.act_token] * act_has_token.view(1, a, 1)
        col_a = color_ctx[:, self.act_color] * act_has_color.view(1, a, 1)
        x = torch.cat(
            (
                tok_a,
                col_a,
                global_ctx.unsqueeze(1).expand(n, a, global_ctx.shape[-1]),
                act_feat.unsqueeze(0).expand(n, a, -1),
            ),
            dim=-1,
        )
        return self.score(x).squeeze(-1)

    def forward(
        self, tok: torch.Tensor, col: torch.Tensor, glob: torch.Tensor
    ) -> torch.Tensor:
        """`forward_reference` with the scorer's first layer split by input block.

        Tokens are projected once (54 rows) and gathered per action (62 rows); the
        color context once per color; the global context once per row; and the
        static action features once per call, independent of the batch.
        """
        n = tok.shape[0]
        w = self.width
        h, hw, color_ctx, global_ctx = self.encoder(tok, col, glob)
        tokens = torch.cat((h.reshape(n, -1, w), hw), dim=1)  # (n, n_tokens, w)
        a = self.act_token.shape[0]
        first, second = self.score[0], self.score[2]
        Wt, Wc, Wg, Wf = _weight_blocks(
            first, (w, 2 * w, 6 * w, self.act_feat.shape[1])
        )
        tok_s = F.linear(tokens, Wt)[:, self.act_token]  # (n, a, w)
        col_s = F.linear(color_ctx, Wc)[:, self.act_color]  # (n, a, w)
        if tok_s.dtype == torch.bfloat16:
            act_has_token, act_has_color = self.act_has_token_bf16, self.act_has_color_bf16
            act_feat = self.act_feat_bf16
        else:
            act_has_token = self.act_has_token.to(dtype=tok_s.dtype)
            act_has_color = self.act_has_color.to(dtype=tok_s.dtype)
            act_feat = self.act_feat.to(dtype=Wf.dtype)
        hidden = (
            tok_s * act_has_token.view(1, a, 1)
            + col_s * act_has_color.view(1, a, 1)
            + F.linear(global_ctx, Wg, first.bias).unsqueeze(1)
            + F.linear(act_feat, Wf)
        )
        return second(torch.relu(hidden)).squeeze(-1)


class Critic(nn.Module):
    """Separate structured invariant network with a scalar linear value head."""

    def __init__(self, sp: Spec, width: int = WIDTH, blocks: int = BLOCKS) -> None:
        super().__init__()
        self.spec = sp
        self.encoder = Encoder(sp, width, blocks)
        self.head = nn.Linear(6 * width, 1)

    def forward(
        self, tok: torch.Tensor, col: torch.Tensor, glob: torch.Tensor
    ) -> torch.Tensor:
        _, _, _, global_ctx = self.encoder(tok, col, glob)
        return self.head(global_ctx).squeeze(-1)

    def forward_reference(
        self, tok: torch.Tensor, col: torch.Tensor, glob: torch.Tensor
    ) -> torch.Tensor:
        """The critic on the reference encoder: the oracle for `forward`."""
        _, _, _, global_ctx = self.encoder.forward_reference(tok, col, glob)
        return self.head(global_ctx).squeeze(-1)


# -- the one masking / probability convention ----------------------------------


def masked_log_probs(
    logits: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 1.0,
    *,
    validate: bool = True,
) -> torch.Tensor:
    """Log-softmax over the legal actions only, at `temperature`.

    The single convention used by collection, the PPO loss, evaluation and replay
    (contract section 1.1). An all-illegal row is a caller bug, not something to
    paper over with a uniform fallback, so it raises.
    """
    # ``bool(cuda_tensor)`` synchronizes the device.  Hot training paths validate
    # the complete CPU rollout once before staging and call this with
    # ``validate=False``; collection/evaluation keep the defensive check.
    if validate and not bool(mask.any(dim=-1).all()):
        raise ValueError("a decision row has no legal action")
    z = logits / temperature
    z = z.masked_fill(~mask, float("-inf"))
    return torch.log_softmax(z, dim=-1)


def entropy_from_log_probs(logp: torch.Tensor) -> torch.Tensor:
    """Per-row entropy in nats, with the -inf of illegal actions contributing zero."""
    p = logp.exp()
    return -(p * torch.where(torch.isfinite(logp), logp, torch.zeros_like(logp))).sum(
        -1
    )


def to_tensors(
    f: Features, device: torch.device | str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.from_numpy(np.ascontiguousarray(f.tok)).to(device),
        torch.from_numpy(np.ascontiguousarray(f.col)).to(device),
        torch.from_numpy(np.ascontiguousarray(f.glob)).to(device),
    )


