"""Step 10a.2 item A4: the shaping potentials, and why they are AIVAT's deal correction.

    uv run uno shaping [ruleset=TOY] [games=300] [iterations=1000]

D5 of the step 9 review asked for reward shaping and got a design; nothing was ever emitted.
`uno.eval.shaping` builds the potentials, and this measures the three things about them that
can be wrong or surprising:

1. **`Phi(terminal) = 0`**, per level and per profile. Not decoration: with no discount it
   is what makes a game's shaping terms telescope to `-Phi(s0)`, so that the potential and
   AIVAT's deal correction are the same object rather than two similar ones.
2. **The deal term, per level of information.** `Phi(s0) - E Phi` has mean zero by
   construction, and its variance is the deal luck the potential removes from every return
   -- which is 9.5c's **deal floor**. Three levels: L1 (what the mover can know of the
   value from its own hand), L2 (both hands, the finest a value function may read) and L3
   (the pile order too). The L2 variance must reproduce `aivat_moments`' own "every
   correction but the deal" ratio, and on `TOY` the published exact figure.
3. **The per-game identity**: the L2 potential's deal term, read off played games, is
   AIVAT's deal correction **array-equal**, not merely close. That equality is the point of
   the item -- the control variate and the shaping term are one object.

Then the L1 set tables are emitted for the profiles step 11's D6 ablation needs (Nash, BR
L1, and the Heuristic league profile), and loaded back.

**What is emitted, and what is not.** The per-information-set table is written out, because
that is what a learner reads and it is small (12k-3.75M floats). The per-*node* vectors are
not: on `TOY2_FULL` one is 28M doubles, and it is two passes over a cached tree away
(`uno.eval.shaping.potential`), so storing it would trade a gigabyte for a second.

Seeds: dev (section 3 plays games only to check an identity; nothing is selected and no
performance number is claimed). Every check prints ok / XX; the last line counts failures.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from uno import store
from uno.agents.baseline import Heuristic, PlayFirstRandom
from uno.compiled.aivat import TreeValues, aivat_moments, arc_expectation, exact_values
from uno.compiled.lookup import build
from uno.compiled.solvers import best_response
from uno.compiled.strategies import transfer
from uno.compiled.tree import CompiledTree, compile_tree
from uno.eval import seeds as seed_registry
from uno.eval.aivat import NONE, Corrections, trace_terms
from uno.eval.play import play_game
from uno.eval.shaping import deal_term, potential, set_potential, values
from uno.experiments.common import by_seat
from uno.game import rules
from uno.game.observe import Level

L1, L2, L3 = Level.L1, Level.L2, Level.L3
NO_DEAL = Corrections(chance=True, row=True, column=True, deal=False)
FAILS: list[str] = []

# The exact "every correction but the deal" variance ratio, published in
# docs/results/index.md's AIVAT table (heuristic vs random, exact values). The L2 deal
# potential's own variance ratio must reproduce it: with an exact value function and every
# other correction applied, the deal term is all the variance AIVAT has left.
PUBLISHED_NO_DEAL = {"TOY": (0.42, 0.49)}


def check(ok: bool, what: str) -> str:
    if not ok:
        FAILS.append(what)
    return "ok" if ok else "XX"


def reproduces(value: float, published: float, decimals: int) -> bool:
    """As `uno.experiments.report.reproduces`: consistent with the published figure read at
    that many decimals, never `round(...) == published`."""
    return abs(value - published) <= 0.5 * 10**-decimals + 1e-6


def profiles(lazy: CompiledTree, world: CompiledTree, iterations: int) -> dict:
    """The profiles a potential is built from: the Nash profile, the BR L1 profile and the
    league's rulebot profile, each as a pair of strategies on both trees."""
    rs = lazy.rs
    rnd, heu = PlayFirstRandom(rs), Heuristic(rs)
    r = (L1, store.load_or_tabulate(lazy, L1, rnd))
    h = (L1, store.load_or_tabulate(lazy, L1, heu))
    nash = store.load_or_solve_nash(lazy, iterations, "cfr+")
    n = (L1, nash.avg)
    br = by_seat(lazy, [best_response(lazy, L1, r, seat) for seat in (0, 1)])
    print(f"   nash: cfr+ {nash.iterations} its, NashConv {100 * nash.final.nashconv:.3f} pts")
    out = {
        "nash vs nash": (n, n),
        "br_l1(random) vs random": (br, r),
        "heuristic vs random": (h, r),
    }
    return {name: {"lazy": pair, "world": tuple((L1, transfer(lazy, world, L1, s[1])) for s in pair)} for name, pair in out.items()}


def levels_section(lazy: CompiledTree, world: CompiledTree, profs: dict) -> None:
    print("\n== 1. the potential per level of information, and the deal luck each one removes")
    print("   Phi(s0) - E Phi has mean zero by construction; its variance is the deal floor (9.5c), i.e. what AIVAT has left with every other correction applied")
    print(f"   {'profile':26s} {'seat':4s} {'level':18s} {'Phi(term)=0':11s} {'deal mean':>10s} {'deal var':>9s} {'ratio to raw':>12s}  checks")
    for name, pair in profs.items():
        for seat in (0, 1):
            lz = pair["lazy"] if seat == 0 else pair["lazy"][::-1]
            wd = pair["world"] if seat == 0 else pair["world"][::-1]
            nu = exact_values(lazy, *lz)
            raw = aivat_moments(lazy, *lz, nu, NONE, seat).var
            nodeal = aivat_moments(lazy, *lz, nu, NO_DEAL, seat).var
            for label, tree, phi in (
                ("L1 (own hand)", lazy, potential(lazy, *lz, L1)),
                ("L2 (both hands)", lazy, potential(lazy, *lz)),
                ("L3 (+ pile order)", world, potential(world, *wd)),
            ):
                zero = bool(np.all(phi[tree.terminals] == 0.0))
                dt = deal_term(tree, phi)
                ratio = dt.var / raw if raw else float("nan")
                marks = check(zero, f"{name} seat {seat} {label}: Phi(terminal) != 0")
                marks += " " + check(abs(dt.mean) < 1e-9, f"{name} seat {seat} {label}: deal term mean {dt.mean:+.2e} is not zero")
                if label.startswith("L2"):
                    # the identity: with an exact V, AIVAT's residual after every other
                    # correction *is* this potential's deal term
                    marks += " " + check(abs(dt.var - nodeal) < 1e-9, f"{name} seat {seat}: the L2 deal term variance {dt.var:.6f} != AIVAT's no-deal variance {nodeal:.6f}")
                    pub = PUBLISHED_NO_DEAL.get(store.ruleset_name(lazy.rs))
                    if pub is not None and name == "heuristic vs random":
                        marks += " " + check(reproduces(ratio, pub[seat], 2), f"{name} seat {seat}: deal floor {ratio:.4f} against the published {pub[seat]}")
                print(f"   {name:26s} {seat:<4d} {label:18s} {str(zero):11s} {dt.mean:+10.1e} {dt.var:9.6f} {ratio:12.4f}  {marks}")
        sys.stdout.flush()


def identity_section(lazy: CompiledTree, profs: dict, games: int) -> None:
    """The per-game equality: the L2 potential's deal term is AIVAT's deal correction."""
    rs = lazy.rs
    idx = build(lazy, L2)
    heu, rnd = Heuristic(rs), PlayFirstRandom(rs)
    print(f"\n== 2. per game, on {games} dev deals: the L2 deal potential's term against AIVAT's own deal correction")
    pair = profs["heuristic vs random"]["lazy"]
    for seat in (0, 1):
        lz = pair if seat == 0 else pair[::-1]
        nu = exact_values(lazy, *lz)
        phi = potential(lazy, *lz)
        mean_phi = float(arc_expectation(lazy, phi)[0])
        model = TreeValues(idx, nu)
        agents = (heu, rnd) if seat == 0 else (rnd, heu)
        mine, theirs = [], []
        t0 = time.perf_counter()
        for seed in seed_registry.dev_block(0, games):
            try:
                trace = play_game(rs, seed, agents)
            except ValueError:
                continue
            mine.append(phi[idx.node(trace.steps[0].state)] - mean_phi)
            theirs.append(trace_terms(model, seed, trace).deal)
        a, b = np.array(mine), np.array(theirs)
        ok = check(np.array_equal(a, b), f"seat {seat}: the L2 deal potential's per-game term is not array-equal to AIVAT's deal correction (max dev {np.abs(a - b).max():.2e})")
        print(f"   seat {seat}: {len(a)} games, array-equal {np.array_equal(a, b)}  {ok}  (sd of the term {a.std(ddof=1):.4f}, {time.perf_counter() - t0:.1f}s)")
        sys.stdout.flush()


def emit_section(lazy: CompiledTree, profs: dict) -> None:
    """Write the per-information-set L1 tables step 11 reads, and load them back."""
    out = Path(store.CACHE) / "shaping"
    out.mkdir(parents=True, exist_ok=True)
    name = store.ruleset_name(lazy.rs)
    k = lazy.keyings[L1]
    print(f"\n== 3. emitted L1 set potentials ({k.n_info:,} sets) -> {out}")
    print(f"   {'profile':26s} {'seat':4s} {'reached sets':>12s} {'range':>18s} {'file':44s} {'KB':>7s}  round trip")
    for pname, pair in profs.items():
        for seat in (0, 1):
            lz = pair["lazy"] if seat == 0 else pair["lazy"][::-1]
            sp = set_potential(lazy, *lz, L1)
            path = out / f"{name}_{pname.replace(' ', '_').replace('(', '').replace(')', '')}_s{seat}_L1.npz"
            np.savez(path, version=np.array(store.CACHE_VERSION), fingerprint=np.array(store.keying_fingerprint(lazy, L1)), phi=sp)
            with np.load(path) as z:
                same = np.array_equal(z["phi"], sp)
            ok = check(same, f"{pname} seat {seat}: the emitted potential does not load back array-equal")
            print(f"   {pname:26s} {seat:<4d} {int((sp != 0).sum()):12,d} [{sp.min():+.3f}, {sp.max():+.3f}] {path.name:44s} {path.stat().st_size / 1024:7.0f}  {ok}")
    # the exact-critic arm's array differs from the potential at exactly one place
    lz = profs["heuristic vs random"]["lazy"]
    v, phi = values(lazy, *lz), potential(lazy, *lz)
    same_away = np.array_equal(np.delete(v, lazy.terminals), np.delete(phi, lazy.terminals))
    ok = check(same_away and float(np.abs(v[lazy.terminals]).max()) > 0, "values() and potential() do not agree away from the terminals")
    print(f"   values() (the exact-critic arm) keeps the terminal returns, max |Phi| there {np.abs(v[lazy.terminals]).max():.1f}; away from them the two are array-equal: {same_away}  {ok}")


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    name = argv[0] if len(argv) > 0 else "TOY"
    games = int(argv[1]) if len(argv) > 1 else 300
    iterations = int(argv[2]) if len(argv) > 2 else 1000
    rs = getattr(rules, name)

    lazy = compile_tree(rs, (L1, L2), True, cache=store.CACHE)
    world = compile_tree(rs, (L1, L3), True, cache=store.CACHE)
    print(f"== {name} shaping potentials (item A4)\n   lazy  {lazy}\n   world {world}")
    profs = profiles(lazy, world, iterations)
    levels_section(lazy, world, profs)
    identity_section(lazy, profs, games)
    emit_section(lazy, profs)
    print(f"\n== {len(FAILS)} failed checks" + (": " + "; ".join(FAILS) if FAILS else ""))


if __name__ == "__main__":
    main()
