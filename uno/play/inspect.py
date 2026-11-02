"""Card-grid network views of one decision (docs/play/plan.md phase 2).

Every token is a card type and every weight is shared per token, so each layer of the
encoder is a grid shaped like the deck: colours x slots, plus the wilds.

- `trace` re-runs one decision through `Encoder.forward_reference`'s residual structure
  (not module hooks: a block's output is a residual *update*), keeping the token states
  after the projection and after each block, and is checked against the actor's own
  logits.
- **Activations**: one PCA per layer, fitted on a few hundred games of the actor
  against the league and cached in `.cache/play/`, maps each token state to RGB. The
  same projection for every game keeps colours comparable.
- **Exact attribution**: the scorer's first layer is a sum of four input blocks (own
  token, colour context, global context, static action features) plus a bias before
  one ReLU. With that decision's ReLU pattern fixed, a logit is exactly
  `b2 + sum_active w2_i * (sum of block contributions)_i`, so `logit(chosen) -
  logit(runner-up)` splits exactly into the blocks.
- **Token attribution**: integrated gradients (16 steps, dataset-mean-token baseline)
  of the same logit difference with respect to each token's projected state, summed
  over width; and occlusion (one token replaced by the mean token). If the two
  disagree badly for a decision, neither is shown. They say which cards moved the
  decision, not why.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from uno.play.opponents import Decision, NeuralPlayer

CACHE = Path(".cache/play")
ANALYSIS_VERSION = 2
IG_STEPS = 16
AGREEMENT = 0.5  # minimum Pearson correlation between IG and occlusion over tokens


def _body(actor):
    return actor.body if getattr(actor, "recurrent", False) else actor


def _pool(x, dim):
    return torch.cat((x.mean(dim=dim), x.amax(dim=dim)), dim=-1)


def _inputs(actor, features, events, memory) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The body's `(tok, col, glob)` rows: a recurrent actor appends memory and events."""
    from uno.rl.model import to_tensors

    tok, col, glob = to_tensors(features)
    if getattr(actor, "recurrent", False):
        n = tok.shape[0]
        mem = memory if memory is not None else actor.initial_memory(n)
        e = actor.event_inputs(torch.from_numpy(events))
        tok = torch.cat([tok, mem.to(tok.dtype), e.to(tok.dtype)], -1)
    return tok, col, glob


def project(enc, tok, col, glob):
    """Token states after the shared projection: `(grid (n,c,s,w), wild (n,wl,w))`."""
    n = tok.shape[0]
    c, s, wl, w = enc.colors, enc.slots, enc.wilds, enc.width
    grid_in = tok[:, : c * s].reshape(n, c, s, -1)
    wild_in = tok[:, c * s :]
    g = glob.unsqueeze(1)
    grid_x = torch.cat((grid_in, col.unsqueeze(2).expand(n, c, s, col.shape[-1]),
                        enc.embed(enc.token_class).unsqueeze(0).unsqueeze(0).expand(n, c, s, w),
                        g.unsqueeze(1).expand(n, c, s, glob.shape[-1])), dim=-1)
    wild_x = torch.cat((wild_in, torch.zeros(n, wl, col.shape[-1], dtype=tok.dtype),
                        enc.embed(enc.wild_class_ids).unsqueeze(0).expand(n, wl, w),
                        g.expand(n, wl, glob.shape[-1])), dim=-1)
    return torch.relu(enc.proj(grid_x)), torch.relu(enc.proj(wild_x))


def blocks(enc, h, hw):
    """The relational blocks from projected states: every layer's `(grid, wild)` and the
    final colour and global contexts (`Encoder.forward_reference`'s loop)."""
    n, c, s, w = h.shape
    wl = hw.shape[1]
    layers = []
    for grid_block, wild_block in zip(enc.grid_blocks, enc.wild_blocks):
        color_ctx, rank_ctx, class_ctx, wild_ctx = _pool(h, 2), _pool(h, 1), enc._class_ctx(h), _pool(hw, 1)
        global_ctx = torch.cat((color_ctx.mean(1), color_ctx.amax(1), wild_ctx), dim=-1)
        grid_update = grid_block(torch.cat((
            h, color_ctx.unsqueeze(2).expand(n, c, s, 2 * w), rank_ctx.unsqueeze(1).expand(n, c, s, 2 * w),
            class_ctx.unsqueeze(1).expand(n, c, s, 2 * w), global_ctx.unsqueeze(1).unsqueeze(1).expand(n, c, s, 6 * w)), dim=-1))
        wild_update = wild_block(torch.cat((hw, global_ctx.unsqueeze(1).expand(n, wl, 6 * w)), dim=-1))
        h, hw = h + grid_update, hw + wild_update
        layers.append((h, hw))
    color_ctx = _pool(h, 2)
    global_ctx = torch.cat((color_ctx.mean(1), color_ctx.amax(1), _pool(hw, 1)), dim=-1)
    return layers, color_ctx, global_ctx


def scorer_blocks(body, h, hw, color_ctx, global_ctx):
    """The scorer's first-layer pre-activation split by input block, `(n, a, w)` each,
    and the logits. Same weights and order of terms as `Actor.scores`."""
    from uno.rl.model.relational import _weight_blocks

    n, w = h.shape[0], body.width
    tokens = torch.cat((h.reshape(n, -1, w), hw), dim=1)
    first, second = body.score[0], body.score[2]
    Wt, Wc, Wg, Wf = _weight_blocks(first, (w, 2 * w, 6 * w, body.act_feat.shape[1]))
    parts = {
        "own card": torch.nn.functional.linear(tokens, Wt)[:, body.act_token] * body.act_has_token.view(1, -1, 1),
        "colour": torch.nn.functional.linear(color_ctx, Wc)[:, body.act_color] * body.act_has_color.view(1, -1, 1),
        "global": torch.nn.functional.linear(global_ctx, Wg).unsqueeze(1).expand(n, body.act_token.shape[0], w),
        "action": torch.nn.functional.linear(body.act_feat, Wf).unsqueeze(0).expand(n, -1, w),
        "bias": first.bias.view(1, 1, w).expand(n, body.act_token.shape[0], w),
    }
    hidden = sum(parts.values())
    logits = second(torch.relu(hidden)).squeeze(-1)
    return parts, hidden, logits


def logits_from_proj(body, h, hw):
    layers, color_ctx, global_ctx = blocks(body.encoder, h, hw)
    h, hw = layers[-1] if layers else (h, hw)
    return scorer_blocks(body, h, hw, color_ctx, global_ctx)[2]


@torch.no_grad()
def trace(actor, features, events=None, memory=None) -> dict:
    """Every layer's token states for a batch of decisions, checked against the actor."""
    body = _body(actor)
    enc = body.encoder
    tok, col, glob = _inputs(actor, features, events, memory)
    h, hw = project(enc, tok, col, glob)
    layers, color_ctx, global_ctx = blocks(enc, h, hw)
    hl, hwl = layers[-1] if layers else (h, hw)
    _, _, logits = scorer_blocks(body, hl, hwl, color_ctx, global_ctx)
    if getattr(actor, "recurrent", False):
        mem_in = memory if memory is not None else actor.initial_memory(tok.shape[0])
        e = torch.from_numpy(events)
        ref, new_memory = actor.step(tok[..., : actor.spec.tok_width], col, glob, e, mem_in)
    else:
        ref, new_memory = actor(tok, col, glob), None
    # relative: the production forward sums in another order (float32, logits up to ~50)
    err = float((ref - logits).abs().max()) / max(1.0, float(ref.abs().max()))
    if err > 1e-5:
        raise AssertionError(f"trace does not reproduce the actor's logits (relative error {err:.2e})")
    n = tok.shape[0]
    flat = lambda g, w: torch.cat((g.reshape(n, -1, g.shape[-1]), w), dim=1)  # noqa: E731
    out = {"input": torch.from_numpy(np.ascontiguousarray(features.tok)), "proj": flat(h, hw)}
    for i, (g, w) in enumerate(layers):
        out[f"block {i + 1}"] = flat(g, w)
    if new_memory is not None:
        out["memory"] = new_memory
    out["_logits"], out["_error"] = logits, err
    return out


# -- PCA ------------------------------------------------------------------------------


def actor_key(actor) -> str:
    h = hashlib.blake2b(digest_size=8)
    for k, v in sorted(actor.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def play_records(policy, games: int = 300, seed: int = 0) -> list[tuple[Decision, object, int, int]]:
    """`(decision, state, seat, game)` for every decision of `policy` against the league, both
    seats, on DEV diagnostic deals from 900,000: the data the PCA and the probes are fitted on."""
    import random

    from uno.game.engine import new_game, step
    from uno.play.opponents import AgentPlayer
    from uno.rl.eval.opponents import league

    rs = policy.rs
    bots = list(league(rs).values())
    rng = random.Random(seed)
    out = []
    for g in range(games):
        seat = g % 2
        try:
            s = new_game(rs, 900_000 + g)
        except ValueError:
            continue
        me = NeuralPlayer(policy, seat)
        bot = AgentPlayer(bots[(g // 2) % len(bots)], seat ^ 1)
        while not s.terminal:
            if s.player == seat:
                probs = me.policy(s)
                out.append((me.last, s, seat, g))
            else:
                probs = bot.policy(s)
            acts = list(probs)
            s = step(s, rng.choices(acts, weights=[probs[a] for a in acts])[0])
    return out


def sample_decisions(policy, games: int = 200, seed: int = 0, keep: int = 4000) -> list[Decision]:
    """A shuffled subset of `play_records`' decisions."""
    import random

    out = [r[0] for r in play_records(policy, games, seed)]
    random.Random(seed).shuffle(out)
    return out[:keep]


def stack(decisions: list[Decision], actor):
    from uno.rl.env.features import Features

    f = Features(*(np.concatenate([getattr(d.features, k) for d in decisions]) for k in ("tok", "col", "glob")))
    events = np.concatenate([d.events for d in decisions]) if getattr(actor, "recurrent", False) else None
    memory = None
    if getattr(actor, "recurrent", False):
        zero = actor.initial_memory(1)[0]
        memory = torch.stack([zero if d.memory is None else d.memory for d in decisions])
    return f, events, memory


def layer_states(policy, decisions: list[Decision], chunk: int = 256) -> dict[str, np.ndarray]:
    """Every traced layer, `(decisions, tokens, width)` float32."""
    layers: dict[str, list[np.ndarray]] = {}
    for i in range(0, len(decisions), chunk):
        f, e, m = stack(decisions[i : i + chunk], policy.actor)
        for k, v in trace(policy.actor, f, e, m).items():
            if not k.startswith("_"):
                layers.setdefault(k, []).append(v.float().numpy())
    return {k: np.concatenate(v) for k, v in layers.items()}


def fit_pca(states: dict[str, np.ndarray], keep: int = 4000, seed: int = 0) -> dict[str, dict[str, np.ndarray]]:
    """Per layer: mean, 3 components, the 2nd/98th percentiles of each component, and the
    per-token mean state (the attribution baseline)."""
    rows = np.random.default_rng(seed).permutation(len(next(iter(states.values()))))[:keep]
    out = {}
    for k, v in states.items():
        sub = v[rows]
        x = sub.reshape(-1, sub.shape[-1]).astype(np.float64)
        mean = x.mean(0)
        vals, vecs = np.linalg.eigh(np.cov((x - mean).T))
        comps = vecs[:, ::-1][:, :3].T
        # sign convention: the largest-magnitude loading is positive, for stable colours
        comps *= np.sign(comps[np.arange(3), np.abs(comps).argmax(1)])[:, None]
        z = (x - mean) @ comps.T
        out[k] = {"mean": mean, "components": comps, "lo": np.percentile(z, 2, axis=0), "hi": np.percentile(z, 98, axis=0),
                  "explained": vals[::-1][:3] / max(vals.sum(), 1e-12), "token_mean": sub.mean(0)}
    return out


# -- the opponent-hand probe ----------------------------------------------------------


def token_types(lay) -> np.ndarray:
    return np.concatenate([lay.grid_type.reshape(-1), lay.wild_type]).astype(int)


def opponent_counts(rs, state, seat: int) -> np.ndarray:
    """How many copies of each token's card the opponent of `seat` holds."""
    counts = np.bincount(np.asarray(state.hands[seat ^ 1], dtype=int), minlength=rs.num_types)
    from uno.rl.env.layout import layout

    return counts[token_types(layout(rs))]


def prior_belief(rs, state, seat: int) -> np.ndarray:
    """P(the opponent holds >= 1 of each token's card) under the uniform split of the cards
    `seat` has not seen (`uno.eval.search.determinize_l1`'s prior): exact."""
    from collections import Counter
    from math import comb

    from uno.eval.search import unseen_cards
    from uno.game.observe import Level, observe
    from uno.rl.env.layout import layout

    obs = observe(state, seat, Level.L1)
    unseen, _ = unseen_cards(rs, obs)
    n, k = len(unseen), obs.opp_hand_size
    c = Counter(unseen)
    total = comb(n, k)
    p = np.array([1.0 - comb(n - c.get(t, 0), k) / total if total else 0.0 for t in range(rs.num_types)])
    return p[token_types(layout(rs))]


def voids_belief(rs, state, seat: int, samples: int = 64, seed: int = 0) -> np.ndarray:
    """The same marginal under `uno.eval.belief`'s `"voids"` rung, from sampled hands. It
    assumes a play-first opponent that plays a playable drawn card at once."""
    import random

    from uno.eval.belief import Belief
    from uno.eval.search import unseen_cards
    from uno.game.observe import Level, observe
    from uno.rl.env.layout import layout

    obs = observe(state, seat, Level.L1)
    unseen, _ = unseen_cards(rs, obs)
    belief, rng = Belief(rs, "voids"), random.Random(seed)
    held = np.zeros(rs.num_types)
    for _ in range(samples):
        held[np.unique(np.asarray(belief.hand(obs, unseen, rng), dtype=int))] += 1
    return (held / samples)[token_types(layout(rs))]


def auroc(y: np.ndarray, p: np.ndarray) -> float:
    """Rank AUROC with ties averaged."""
    y = y.astype(bool)
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    _, inv, counts = np.unique(p, return_inverse=True, return_counts=True)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    ranks = (starts + (counts + 1) / 2.0)[inv]
    return float((ranks[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def logistic(X: np.ndarray, y: np.ndarray, ridge: float = 1e-3, iters: int = 30) -> np.ndarray:
    """Weights (last = bias) of an L2-regularized logistic regression, by Newton's method."""
    X = np.concatenate([X, np.ones((len(X), 1))], 1)
    w = np.zeros(X.shape[1])
    reg = ridge * len(X) * np.eye(X.shape[1])
    reg[-1, -1] = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ w, -30, 30)))
        g = X.T @ (p - y) + reg @ w
        H = (X * (p * (1 - p))[:, None]).T @ X + reg
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def probe_predict(probe: dict, states: np.ndarray) -> np.ndarray:
    x = (states - probe["mean"]) / probe["std"]
    return 1.0 / (1.0 + np.exp(-np.clip(x @ probe["w"][:-1] + probe["w"][-1], -30, 30)))


def fit_probes(policy, records, states: dict[str, np.ndarray], baseline_decisions: int = 600) -> tuple[dict, dict]:
    """One probe per layer, shared by every token: `P(opponent holds this card)`.

    Fitted on 70% of the games, scored on the other 30% (whole games held out, so no
    decision of a scored game was seen in fitting), against the prior and `voids`
    marginals on the same held-out decisions (`voids` on a subsample: it is sampled)."""
    rs = policy.rs
    y = np.stack([opponent_counts(rs, s, seat) > 0 for _, s, seat, _ in records]).astype(np.float64)
    game = np.array([g for *_, g in records])
    held = (game % 10) >= 7
    probes, metrics = {}, {"decisions": len(records), "held_out_decisions": int(held.sum()), "layers": {}}
    for k, v in states.items():
        x = v.reshape(len(v), -1, v.shape[-1]).astype(np.float64)
        tr = x[~held].reshape(-1, x.shape[-1])
        mean, std = tr.mean(0), tr.std(0) + 1e-6
        w = logistic((tr - mean) / std, y[~held].reshape(-1))
        probes[k] = {"w": w, "mean": mean, "std": std}
        p = probe_predict(probes[k], x[held].reshape(-1, x.shape[-1]))
        yt = y[held].reshape(-1)
        metrics["layers"][k] = {"auroc": auroc(yt, p), "brier": float(np.mean((p - yt) ** 2))}
    idx = np.flatnonzero(held)
    idx = idx[np.random.default_rng(0).permutation(len(idx))[:baseline_decisions]]
    yb = y[idx].reshape(-1)
    for name, fn in (("prior", prior_belief), ("voids", voids_belief)):
        p = np.concatenate([fn(rs, records[i][1], records[i][2]) for i in idx])
        metrics[name] = {"auroc": auroc(yb, p), "brier": float(np.mean((p - yb) ** 2)), "decisions": len(idx)}
    # the probes on the same subsample, so the comparison is like for like
    metrics["subsample"] = {}
    for k, v in states.items():
        x = v[idx].reshape(-1, v.shape[-1]).astype(np.float64)
        p = probe_predict(probes[k], x)
        metrics["subsample"][k] = {"auroc": auroc(yb, p), "brier": float(np.mean((p - yb) ** 2))}
    return probes, metrics


# -- the cached bundle ----------------------------------------------------------------


def fit_analysis(policy, games: int = 300) -> dict:
    records = play_records(policy, games)
    states = layer_states(policy, [r[0] for r in records])
    probes, metrics = fit_probes(policy, records, states)
    return {"pca": fit_pca(states), "probe": probes, "metrics": metrics}


def load_analysis(policy, games: int = 300) -> dict:
    """PCA and probes for this actor's weights, fitted once and cached in `.cache/play/`."""
    import json

    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"analysis_v{ANALYSIS_VERSION}_{actor_key(policy.actor)}.npz"
    if path.exists():
        flat = dict(np.load(path))
        out: dict = {"pca": {}, "probe": {}, "metrics": json.loads(str(flat.pop("metrics")))}
        for key, v in flat.items():
            part, layer, field = key.split("|")
            out[part].setdefault(layer, {})[field] = v
        return out
    bundle = fit_analysis(policy, games)
    arrays = {f"{part}|{layer}|{field}": v for part in ("pca", "probe")
              for layer, d in bundle[part].items() for field, v in d.items()}
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, metrics=np.asarray(json.dumps(bundle["metrics"])), **arrays)
    tmp.replace(path)
    return bundle


# -- attribution ----------------------------------------------------------------------


@torch.no_grad()
def decompose(body, h, hw, color_ctx, global_ctx, a: int, b: int) -> dict:
    """Exact split of `logit(a) - logit(b)` into the scorer's input blocks (one decision)."""
    parts, hidden, logits = scorer_blocks(body, h, hw, color_ctx, global_ctx)
    w2, b2 = body.score[2].weight[0], body.score[2].bias[0]
    out = {}
    for name, p in parts.items():
        act_a, act_b = (hidden[0, a] > 0).float(), (hidden[0, b] > 0).float()
        out[name] = float((w2 * act_a * p[0, a]).sum() - (w2 * act_b * p[0, b]).sum())
    total = float(logits[0, a] - logits[0, b])
    residual = total - sum(out.values())
    if abs(residual) > 1e-4 * max(1.0, abs(total)):
        raise AssertionError(f"the block decomposition misses the logit difference by {residual:.2e}")
    return {"blocks": out, "total": total}


def token_attribution(body, h, hw, a: int, b: int, token_mean: np.ndarray) -> dict:
    """Integrated gradients and occlusion over the projected token states."""
    n_grid = h.shape[1] * h.shape[2]
    x = torch.cat((h.reshape(1, n_grid, -1), hw), dim=1)[0]  # (tokens, w)
    base = torch.from_numpy(token_mean).to(x.dtype)
    c, s = h.shape[1], h.shape[2]

    def f(tokens):  # (m, tokens, w) -> (m,) logit difference
        g = tokens[:, :n_grid].reshape(-1, c, s, tokens.shape[-1])
        logits = logits_from_proj(body, g, tokens[:, n_grid:])
        return logits[:, a] - logits[:, b]

    alphas = (torch.arange(IG_STEPS, dtype=x.dtype) + 0.5) / IG_STEPS
    path = (base + alphas.view(-1, 1, 1) * (x - base)).detach().requires_grad_(True)
    (grad,) = torch.autograd.grad(f(path).sum(), path)
    ig = ((x - base) * grad.mean(0)).sum(-1).detach()
    with torch.no_grad():
        full = f(x.unsqueeze(0))[0]
        occluded = x.unsqueeze(0).repeat(x.shape[0], 1, 1)
        idx = torch.arange(x.shape[0])
        occluded[idx, idx] = base
        occ = full - f(occluded)
        baseline = f(base.unsqueeze(0))[0]
    ig_np, occ_np = ig.numpy(), occ.numpy()
    corr = float(np.corrcoef(ig_np, occ_np)[0, 1]) if ig_np.std() > 0 and occ_np.std() > 0 else 0.0
    return {"ig": ig_np.tolist(), "occlusion": occ_np.tolist(), "agreement": corr,
            "shown": corr >= AGREEMENT, "completeness": float(full - baseline), "ig_sum": float(ig.sum())}


# -- the page's view ------------------------------------------------------------------


def channel_names(sp) -> list[str]:
    from uno.rl.env import evidence as E
    from uno.rl.env import features as F

    names = ["hand", "in hand", "discard", "is top", "is drawn", "playable", "copies", "is wild",
             "current colour", "top colour"]
    if sp.has_evidence:
        names += [f"evidence same colour, age {i}" for i in range(E.AGE_BINS)]
        names += [f"evidence other colour, age {i}" for i in range(E.AGE_BINS)]
        names += ["opponent's last play"]
    if sp.has_history:
        names += [f"opponent played, decay {d:g}" for d in E.HISTORY_DECAYS]
    if sp.has_opponent:
        names += ["opponent hand count", "opponent holds it"]
    if len(names) != sp.tok_width:
        names += [f"channel {i}" for i in range(len(names), sp.tok_width)]
    assert F.TOK_POSITION == 10
    return names[: sp.tok_width]


class Inspector:
    """Builds the JSON the page draws for one recorded decision: raw input channels, the
    opponent-hand probe per layer with its baselines, the attribution, and the PCA colours.
    The PCA and probes are fitted once per actor (`load_analysis`)."""

    def __init__(self, rs) -> None:
        self.rs = rs
        self.bundles: dict[int, dict] = {}
        self.lock = __import__("threading").Lock()

    def bundle(self, policy) -> dict:
        with self.lock:  # one fit per actor even if the prefit thread and a request race
            if id(policy) not in self.bundles:
                self.bundles[id(policy)] = load_analysis(policy)
            return self.bundles[id(policy)]

    def view(self, policy, d: Decision, chosen: int | None = None, context: dict | None = None) -> dict:
        """`context`: `{"state", "seat", "truth"}` of the decision, for the belief baselines;
        `truth` says whether the viewer may see the opponent's real hand."""
        actor, sp = policy.actor, policy.spec
        lay = sp.layout
        bundle = self.bundle(policy)
        pca, probes, metrics = bundle["pca"], bundle["probe"], bundle["metrics"]
        f, e, m = stack([d], actor)
        tr = trace(actor, f, e, m)
        layers, belief = {}, {}
        for name, states in tr.items():
            if name.startswith("_"):
                continue
            x = states[0].double().numpy()
            if name in pca:
                p = pca[name]
                z = (x - p["mean"]) @ p["components"].T
                rgb = np.clip((z - p["lo"]) / np.maximum(p["hi"] - p["lo"], 1e-9), 0, 1)
                layers[name] = {"rgb": np.round(rgb, 3).tolist(), "explained": np.round(p["explained"], 3).tolist()}
            if name in probes:
                belief[name] = np.round(probe_predict(probes[name], x), 4).tolist()
        probs = d.probs
        ranked = sorted(probs, key=probs.get, reverse=True)
        a = chosen if chosen is not None else ranked[0]
        others = [x for x in ranked if x != a]
        result = {
            "tokens": [int(t) for t in token_types(lay)],
            "colors": lay.colors, "slots": lay.slots, "wilds": lay.wilds,
            "channels": channel_names(sp), "features": np.round(f.tok[0], 3).tolist(),
            "layers": layers, "order": list(layers), "trace_error": tr["_error"],
            "belief": belief, "belief_metrics": metrics, "peeks": bool(sp.has_opponent),
            "probs": {str(k): v for k, v in probs.items()}, "chosen": a,
        }
        if context is not None:
            st, seat = context["state"], context["seat"]
            result["baselines"] = {"prior": np.round(prior_belief(self.rs, st, seat), 4).tolist(),
                                   "voids": np.round(voids_belief(self.rs, st, seat), 4).tolist()}
            if context.get("truth"):
                result["truth"] = opponent_counts(self.rs, st, seat).tolist()
        if not others:
            result["runner_up"] = None
            return result
        b = others[0]
        body = _body(actor)
        tok, col, glob = _inputs(actor, f, e, m)
        with torch.no_grad():
            h, hw = project(body.encoder, tok, col, glob)
            blocks_, color_ctx, global_ctx = blocks(body.encoder, h, hw)
            hl, hwl = blocks_[-1] if blocks_ else (h, hw)
            result["decomposition"] = decompose(body, hl, hwl, color_ctx, global_ctx, a, b)
        with torch.enable_grad():
            result["attribution"] = token_attribution(body, h, hw, a, b, pca["proj"]["token_mean"])
        result["runner_up"] = b
        return result


def main(argv: list[str] | None = None) -> None:
    """`uv run python -m uno.play.inspect ACTOR_EXPORT [--games 300]`: fit (or load) and
    print the probe's held-out quality per layer against the prior and `voids`."""
    import argparse

    from uno.game.rules import NAMED
    from uno.play.opponents import load_policy

    ap = argparse.ArgumentParser(prog="uno.play.inspect")
    ap.add_argument("actor")
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--ruleset", default="FULL_UNO")
    args = ap.parse_args(argv)
    policy, payload, detail = load_policy(NAMED[args.ruleset], args.actor)
    m = load_analysis(policy, args.games)["metrics"]
    print(f"{payload['name']}: {detail}")
    print(f"{m['decisions']:,} decisions, {m['held_out_decisions']:,} in held-out games; "
          f"baselines on {m['prior']['decisions']} held-out decisions")
    print(f"{'':>12} {'AUROC':>7} {'Brier':>7}   {'sub AUROC':>9} {'sub Brier':>9}")
    for k, v in m["layers"].items():
        sub = m["subsample"][k]
        print(f"{k:>12} {v['auroc']:7.3f} {v['brier']:7.4f}   {sub['auroc']:9.3f} {sub['brier']:9.4f}")
    for k in ("prior", "voids"):
        print(f"{k:>12} {'':>7} {'':>7}   {m[k]['auroc']:9.3f} {m[k]['brier']:9.4f}")


if __name__ == "__main__":
    main()


__all__ = ["Inspector", "auroc", "decompose", "fit_analysis", "fit_pca", "fit_probes", "load_analysis", "token_attribution", "trace"]
