"""Step 8.7: size the TOY2 candidates exactly (uno/sizing.py, notes.md).

    uv run python scripts/sizing.py [position_cap=4000000]

The step 8.6 census stopped at 6M quotient histories on every 13-card
candidate. The position DP counts a tree without walking it, so the same
candidates get exact sizes, plus the exact random-vs-random seat-0 score
(the seat-parity readout; the earlier table sampled it on 2000 seeds), the
exhaustion share and the game length under PlayFirstRandom.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uno import rules
from uno.rules import Ruleset
from uno.sizing import size


def toy(colors=4, ranks=(1, 1), skip=0, draw_two=0, wilds=1, hand=3, forced=True) -> Ruleset:
    return Ruleset(
        colors=colors, rank_copies=ranks, skip_per_color=skip, reverse_per_color=0, draw_two_per_color=draw_two,
        wild_count=wilds, wild_draw_four_count=0, hand_size=hand, draw_then_may_play=True, forced_play=forced,
        reshuffle="none", max_turns=None,
    )  # fmt: skip


CANDIDATES: list[tuple[str, Ruleset]] = [
    ("TOY 4c(1,1)+W h3 forced", rules.TOY),
    ("TOY_FULL 4c(1,1)+W h3 free", rules.TOY_FULL),
    ("5c(1,1)+W h3 forced", toy(colors=5)),
    ("5c(1,1)+W h3 free", toy(colors=5, forced=False)),
    ("4c(1,1)+2W h3 forced", toy(wilds=2)),
    ("4c(1,1)+2W h3 free", toy(wilds=2, forced=False)),
    ("4c(1,1,1)+W h3 forced", toy(ranks=(1, 1, 1))),
    ("4c(1,1,1)+W h3 free", toy(ranks=(1, 1, 1), forced=False)),
    ("4c(1,1,1)+W h4 forced", toy(ranks=(1, 1, 1), hand=4)),
    ("4c(1,1,1)+W h2 forced", toy(ranks=(1, 1, 1), hand=2)),
    ("4c(2,1)+W h3 forced", toy(ranks=(2, 1))),
    ("3c(1,1,1,1)+W h3 forced", toy(colors=3, ranks=(1, 1, 1, 1))),
    ("4c(1,1)+S+W h3 forced", toy(skip=1)),
    ("4c(1,1)+S+W h3 free", toy(skip=1, forced=False)),
    ("4c(1,1)+S+W h4 forced", toy(skip=1, hand=4)),
    ("4c(1,1)+D2+W h3 forced", toy(draw_two=1)),
    ("4c(1,1,1)+2W h3 forced", toy(ranks=(1, 1, 1), wilds=2)),
    ("5c(1,1)+W h4 forced", toy(colors=5, hand=4)),
    ("4c(1,1)+S+2W h3 forced", toy(skip=1, wilds=2)),
    ("5c(1,1,1)+W h3 forced", toy(colors=5, ranks=(1, 1, 1))),
    ("4c(1,1,1,1)+W h3 forced", toy(ranks=(1, 1, 1, 1))),
]


def main() -> None:
    cap = int(sys.argv[1]) if len(sys.argv) > 1 else 4_000_000
    print(f"{'candidate':28s} {'deck':>4s} {'pile':>4s} {'orbits':>7s} {'positions':>10s} {'histories':>12s} {'decisions':>12s} {'depth':>5s} {'R vs R p0':>9s} {'UL p0':>6s} {'exh':>5s} {'ties':>5s} {'dec/game':>8s} {'sec':>6s}")
    for name, rs in CANDIDATES:
        z = size(rs, symmetric=True, position_cap=cap)
        pile = rs.deck_size - 2 * rs.hand_size - 1
        hist = f"{z.histories:,}" + ("" if z.complete else "+")
        if z.complete:
            tail = f"{z.random_p0_score:9.4f} {z.uniform_legal_p0_score:6.3f} {z.exhaustion_share():5.3f} {z.tie_share():5.3f} {z.decisions_per_game[1]:8.2f}"
        else:
            tail = f"{'(cap)':>9s} {'':6s} {'':5s} {'':5s} {'':8s}"
        print(f"{name:28s} {rs.deck_size:4d} {pile:4d} {z.roots:7,} {z.positions:10,} {hist:>12s} {z.decision_nodes:12,} {z.max_depth:5d} {tail} {z.seconds:6.1f}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
