"""Step 9.5a: the deal model, checked at 100,000 seeds instead of step 9's 20,000
(B4 of the step 9 review: on-seeds checks cannot see a wrong deal model,
`scores_on_seeds` starts from each seed's real state, so the step 8.9 duplicate-card
bug passed every one of them; the standalone check is the mean over many seeds of
the exact score on each real deal against the whole-game exact score, z on the deal
variance). At 20,000 seeds the check resolves a roughly 0.5-point bias at 2-3 sigma,
marginal against that bug's up to 0.4-point sizing moves; at 100,000 seeds 3 sigma
is about 0.24 points (`sd(score) <= 0.5`, `3 * 0.5 / sqrt(100_000) ~ 0.0047` return
units, `~0.24` score points), a real safeguard rather than a coin flip.

    uv run uno deal_model [ruleset=TOY] [seeds=100000]

`TOY`, `TOY2`, `TOY2_SKIP` (forced play): random vs random and heuristic vs random.
`TOY_FULL`, `TOY2_FULL` (free play): nash vs random only, at a quarter of `seeds`
(the Nash agent needs a cached CFR result, `uv run uno cfr` / `symmetry` / `toy2`
first) -- random vs random and heuristic vs random are skipped there: neither
agent ever draws voluntarily, so their games are identical to the forced toy's
(`uno.experiments.pairing`'s "tickets are the same game" section already checks
this at 20,000 seeds on `TOY` / `TOY_FULL` together).
"""

from __future__ import annotations

import sys

from uno import store
from uno.agents.baseline import Heuristic, PlayFirstRandom
from uno.compiled.tree import compile_tree
from uno.experiments.common import deal_model_check
from uno.game import rules
from uno.game.observe import Level
from uno.store import CACHE

L1 = Level.L1
FAILS: list[str] = []


def check(ok: bool, what: str) -> str:
    if not ok:
        FAILS.append(what)
    return "ok" if ok else "XX"


FREE_PLAY = {"TOY_FULL", "TOY2_FULL"}


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    name = argv[0] if len(argv) > 0 else "TOY"
    n_seeds = int(argv[1]) if len(argv) > 1 else 100_000
    rs = getattr(rules, name)
    lazy = compile_tree(rs, (L1,), True, cache=CACHE)
    print(f"== {name}: {lazy}")
    rnd, heu = PlayFirstRandom(rs), Heuristic(rs)
    sl = {a.name: (L1, store.load_or_tabulate(lazy, L1, a)) for a in (rnd, heu)}

    if name in FREE_PLAY:
        nash = store.load_nash(lazy)
        if nash is None:
            raise SystemExit(f"no cached Nash strategy for {name}: run `uv run uno cfr {name}` / `symmetry` / `toy2 {name}` first")
        nash_agent = nash.agent(lazy, name="nash")
        sl["nash"] = (L1, nash.avg)
        deal_model_check(rs, lazy, [("nash vs random", nash_agent, rnd)], sl, n_seeds // 4, 6_000_000, check)
    else:
        deal_model_check(rs, lazy, [("random vs random", rnd, rnd), ("heuristic vs random", heu, rnd)], sl, n_seeds, 6_000_000, check)

    print(f"\n== {len(FAILS)} failed checks" + (": " + "; ".join(FAILS) if FAILS else ""))


if __name__ == "__main__":
    main()
