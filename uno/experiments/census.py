"""Step 3: tree-size census on candidate toy rulesets + cheap skill proxy.

    uv run uno census [node_cap] [games]

Prints one row per candidate: lazy-tree sizes (see uno/game/chance.py) and the match
results random-vs-random and heuristic-vs-random (both seats) as the proxy
skill check. A candidate whose tree hits the cap is reported as incomplete.
"""

from __future__ import annotations

import sys

from uno.agents.baseline import Heuristic, PlayFirstRandom, UniformLegal
from uno.eval.play import match
from uno.game.rules import Ruleset
from uno.reference.census import census


def toy(**kw) -> Ruleset:
    base = dict(
        colors=2, rank_copies=(2, 2, 2), skip_per_color=0, reverse_per_color=0,
        draw_two_per_color=0, wild_count=1, wild_draw_four_count=0, hand_size=2,
        draw_then_may_play=True, forced_play=False, reshuffle="none", max_turns=None,
    )  # fmt: skip
    base.update(kw)
    return Ruleset(**base)


CANDIDATES: dict[str, Ruleset] = {
    "A 2c(2,2,2)+W h2 [TOY_CANDIDATE]": toy(),
    "B 2c(2,2)+W h2": toy(rank_copies=(2, 2)),
    "C 2c(2,2)+W h3": toy(rank_copies=(2, 2), hand_size=3),
    "D 2c(1,1,1)+W h2": toy(rank_copies=(1, 1, 1)),
    "E 2c(2,2)+W+S h2": toy(rank_copies=(2, 2), skip_per_color=1),
    "F 2c(2,2)+D2 h2": toy(rank_copies=(2, 2), wild_count=0, draw_two_per_color=1),
    "G 2c(2,2,2)+W h2 forced": toy(forced_play=True),
    "H 2c(2,2)+W h2 forced": toy(rank_copies=(2, 2), forced_play=True),
    "I 3c(2,2)+W h2": toy(colors=3, rank_copies=(2, 2)),
    "J 2c(2,2,2)+W h3": toy(hand_size=3),
    "K 2c(1,1,1)+W h3": toy(rank_copies=(1, 1, 1), hand_size=3),
    "L 2c(2,2)+W h2 no-after-draw": toy(rank_copies=(2, 2), draw_then_may_play=False),
    "M 2c(1,1,1)+W+S h2": toy(rank_copies=(1, 1, 1), skip_per_color=1),
    "N 2c(2,2)+W+S h2 forced": toy(rank_copies=(2, 2), skip_per_color=1, forced_play=True),
    "O 2c(2,2)+W h3 forced": toy(rank_copies=(2, 2), hand_size=3, forced_play=True),
    "P 2c(2,2)+W h4": toy(rank_copies=(2, 2), hand_size=4),
    "Q 2c(2,2)+W+S h3": toy(rank_copies=(2, 2), skip_per_color=1, hand_size=3),
    "R 3c(1,1)+W h2": toy(colors=3, rank_copies=(1, 1)),
    # low per-card playability, forced play (tempo fix, see docs/foundations/results/03-08-foundations/03-census.md)
    "S 4c(1,1)+W h3 forced [TOY]": toy(colors=4, rank_copies=(1, 1), hand_size=3, forced_play=True),
    "T 3c(1,1,1)+W h3 forced": toy(colors=3, rank_copies=(1, 1, 1), hand_size=3, forced_play=True),
    "U 4c(1,1,1)+W h3 forced": toy(colors=4, rank_copies=(1, 1, 1), hand_size=3, forced_play=True),
    "V 4c(1,1)+W h3 [TOY_FULL]": toy(colors=4, rank_copies=(1, 1), hand_size=3),
}


def tempo(rs: Ruleset, seeds) -> str:
    """Share of random-vs-random games with no draw at all (pure races, which
    seat 0 always wins) and share of decisions with more than one legal action."""
    from uno.eval.play import play_game
    from uno.game.rules import Draw

    rnd = PlayFirstRandom(rs)
    games = no_draw = decisions = choices = 0
    for seed in seeds:
        try:
            t = play_game(rs, seed, (rnd, rnd))
        except ValueError:
            continue
        games += 1
        no_draw += not any(isinstance(rs.actions[s.action], Draw) for s in t.steps)
        for s in t.steps:
            decisions += 1
            choices += len(s.probs) > 1
    return f"no-draw games={no_draw / games:.2f} decisions with a choice={choices / decisions:.2f}"


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    cap = int(float(argv[0])) if len(argv) > 0 else 2_000_000
    games = int(argv[1]) if len(argv) > 1 else 2000
    only = argv[2].split(",") if len(argv) > 2 else None
    for name, rs in CANDIDATES.items():
        if only and name[0] not in only:
            continue
        print(f"== {name}  deck={rs.deck_size} types={rs.num_types} actions={rs.num_actions}")
        c = census(rs, node_cap=cap)
        print("   census:", c.row())
        if c.complete:
            print("   uniform-legal end mass:", {k: round(v, 3) for k, v in sorted(c.end_mass.items(), key=str)})
        rnd, heu, uni = PlayFirstRandom(rs), Heuristic(rs), UniformLegal(rs)
        seeds = range(games)
        print("   U vs U      :", match(rs, (uni, uni), seeds), "(exact p0 score above if complete)")
        print("   R vs R      :", match(rs, (rnd, rnd), seeds))
        print("   R vs U      :", match(rs, (rnd, uni), seeds))
        print("   H vs R      :", match(rs, (heu, rnd), seeds))
        print("   H vs R swap :", match(rs, (heu, rnd), seeds, swap=True))
        print("   tempo       :", tempo(rs, seeds))
        sys.stdout.flush()


if __name__ == "__main__":
    main()
