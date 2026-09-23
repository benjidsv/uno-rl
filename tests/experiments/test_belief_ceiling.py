"""`uno.experiments.belief_ceiling`: the event stream, the warm start and equivariance.

The recurrent model's equivariance is checked the way `tests/rl/test_model.py` checks
the actor: whole game states are mapped by `Symmetry` (colors and ranks within copy
classes), the stream and features are rebuilt from the mapped states, and predictions
must map with the card types. Nothing is permuted by hand.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.rl.helpers import decision_games
from uno.experiments import belief_ceiling as bc
from uno.game.observe import Level, observe
from uno.game.rules import MID_25
from uno.game.symmetry import Symmetry
from uno.rl.env import features as F
from uno.rl.env.evidence import Evidence

RS = MID_25
SEAT = 0


def _games(seeds=range(8)):
    games = [[s for s in g if s.player == SEAT] for g in decision_games(RS, seeds, max_decisions=40)]
    return [g for g in games if len(g) >= 2]


def _batch(games, sym=None, g=0):
    """A game batch as `bc.game_batch` lays it out, rebuilt from (optionally mapped) states."""
    sp = F.spec(RS, F.ARM_C)
    tokens, rows = [], []
    for i, game in enumerate(games):
        tape = bc.Tape(RS, SEAT)
        for j, s in enumerate(game):
            s = sym.state(g, s) if sym is not None else s
            o = observe(s, SEAT, Level.L1)
            rows.append((i, j, tape.advance(o), o, Evidence.from_public(RS, s.public, SEAT ^ 1)))
        tokens.append(tape.tokens)
    length = max(len(t) for t in tokens)
    pad = np.zeros((len(tokens), length, 5), dtype=np.int64)
    for i, t in enumerate(tokens):
        pad[i, : len(t)] = t
    feats = F.extract(sp, [r[3] for r in rows], [r[4] for r in rows])
    f32 = lambda a: torch.as_tensor(a, dtype=torch.float32)
    return {"tokens": torch.as_tensor(pad), "row_game": torch.tensor([r[0] for r in rows]),
            "step": torch.tensor([r[1] for r in rows]), "pos": torch.tensor([r[2] for r in rows]),
            "tok": f32(feats.tok), "col": f32(feats.col), "glob": f32(feats.glob)}


def _randomize(model, seed=0):
    torch.manual_seed(seed)
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.1)


def test_tape_reconstructs_own_hand_and_opponent_hand_size():
    for game in _games():
        tape = bc.Tape(RS, SEAT)
        for s in game:
            o = observe(s, SEAT, Level.L1)
            toks = tape.tokens[: tape.advance(o)]
            hand = np.zeros(RS.num_types, int)
            opp = RS.hand_size
            for kind, actor, card, _, n in toks:
                if kind in (bc.INIT, bc.OWN_DRAW):
                    hand[card] += 1
                elif kind == bc.PLAY:
                    hand[card] -= actor == bc.SELF
                    opp -= actor == bc.OPP
                elif kind == bc.DRAW and actor == bc.OPP:
                    opp += n
            assert np.array_equal(hand, np.bincount(o.hand, minlength=RS.num_types))
            assert opp == o.opp_hand_size


def test_warm_started_recurrent_model_starts_exactly_at_rel():
    b = _batch(_games())
    rel = bc.Relational(F.spec(RS, F.ARM_C), 32, 2)
    _randomize(rel)
    for history, memory in ((True, True), (True, False), (False, False)):
        rec = bc.Recurrent(RS, F.spec(RS, F.ARM_C), 32, 2, history=history, mem=8, ev=8, memory=memory)
        _randomize(rec, 1)  # memory/event weights nonzero; load_rel must still zero their reads
        rec.load_rel(rel.state_dict())
        with torch.no_grad():
            assert torch.allclose(rec(b), rel(b), atol=1e-5)


VARIANTS = pytest.mark.parametrize("history,memory", ((True, True), (True, False), (False, False)),
                                   ids=("rec", "recev", "recnull"))


@VARIANTS
def test_recurrent_model_is_equivariant_under_the_whole_group(history, memory):
    sym = Symmetry(RS)
    games = _games()
    rec = bc.Recurrent(RS, F.spec(RS, F.ARM_C), 32, 2, history=history, mem=8, ev=8, memory=memory)
    _randomize(rec)
    with torch.no_grad():
        base = rec(_batch(games))
        assert base.std() > 1e-2, "the test needs non-constant predictions"
        worst = 0.0
        for g in range(1, sym.order):
            got = rec(_batch(games, sym, g))
            types = torch.as_tensor(sym.types[g].astype(np.int64))
            worst = max(worst, float((got[:, types] - base).abs().max()))
    assert worst < 1e-4, f"max mapped-prediction error {worst}"


def test_equivariance_check_detects_a_color_dependent_input():
    """The oracle above must be able to fail: leak an absolute color into the stream."""
    sym = Symmetry(RS)
    games = _games()
    rec = bc.Recurrent(RS, F.spec(RS, F.ARM_C), 32, 2, mem=8, ev=8)
    _randomize(rec)
    leak = lambda b: {**b, "tokens": torch.where(b["tokens"][..., 3:4] == 0, b["tokens"] * 0 + 1, b["tokens"])}
    with torch.no_grad():
        base = rec(leak(_batch(games)))
        g = next(g for g in range(sym.order) if sym.colors[g][0] != 0)
        got = rec(leak(_batch(games, sym, g)))
    assert float((got[:, torch.as_tensor(sym.types[g].astype(np.int64))] - base).abs().max()) > 1e-3


@VARIANTS
def test_recurrent_predictions_ignore_everything_after_the_decision(history, memory):
    """Causality: rewriting later events and later decisions' features must not change
    any prediction up to decision `j` (the stream is read as one sequence per game)."""
    games = _games()
    b = _batch(games)
    rec = bc.Recurrent(RS, F.spec(RS, F.ARM_C), 32, 2, history=history, mem=8, ev=8, memory=memory)
    _randomize(rec)
    torch.manual_seed(1)
    with torch.no_grad():
        base = rec(b)
        for j in (0, 1, 3):
            later = b["step"] > j
            if not later.any():
                continue
            fake = dict(b)
            tokens = b["tokens"].clone()
            for gi in range(tokens.shape[0]):
                rows = (b["row_game"] == gi) & (b["step"] == j)
                if rows.any():
                    cut = int(b["pos"][rows][0])
                    tokens[gi, cut:, 1:4] = torch.randint_like(tokens[gi, cut:, 1:4], 0, 3)
                    tokens[gi, cut:, 0] = bc.PLAY
            fake["tokens"] = tokens
            for key in ("tok", "col", "glob"):
                x = b[key].clone()
                x[later] = torch.rand_like(x[later])
                fake[key] = x
            got = rec(fake)
            keep = ~later
            assert torch.allclose(got[keep], base[keep], atol=1e-6), f"decision <= {j} saw the future"
            assert not torch.allclose(got[later], base[later]), "the perturbation must matter later"
