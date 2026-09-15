"""The on-disk cache of compiled trees and CFR results.

Files live in `CACHE` (the repo's `.cache/`, or `$UNO_CACHE`), git-ignored:

- `tree_<name>_<levels>[W]_<sym|full>.npz`: a compiled tree (`CompiledTree.save`),
  `W` for world roots without an L3 keying;
- `cfr_<name>_L1_<sym|full>[_<variant>].npz`: a CFR result (`CFRResult.save`), no
  suffix for CFR+, `+` spelled `p` otherwise (`_dcfr`, `_pcfrp`).

`<name>` is the ruleset's name in `uno.game.rules.NAMED`, or a hash of its repr.
Every experiment loads trees and Nash strategies through here.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import TYPE_CHECKING

from uno.game.observe import Level
from uno.game.rules import NAMED, Ruleset

if TYPE_CHECKING:
    from uno.compiled.cfr import CFRResult
    from uno.compiled.tree import CompiledTree

CACHE = Path(os.environ.get("UNO_CACHE", Path(__file__).resolve().parent.parent / ".cache"))


def ruleset_name(rs: Ruleset) -> str:
    """The ruleset's name in `NAMED`, or a short stable hash of its repr."""
    name = next((n for n, r in NAMED.items() if r == rs), None)
    return name or hashlib.blake2b(repr(rs).encode(), digest_size=6).hexdigest()


def cache_name(rs: Ruleset, levels: tuple[Level, ...], symmetric: bool, world: bool | None = None) -> str:
    """Short stable name for a compiled tree; `W` marks world roots without an L3 keying."""
    w = "W" if world and Level.L3 not in levels else ""
    return f"{ruleset_name(rs)}_{''.join(lv.name for lv in levels)}{w}_{'sym' if symmetric else 'full'}"


def load_or_compile(rs: Ruleset, levels: tuple[Level, ...] = (Level.L1,), symmetric: bool = True, world: bool | None = None, cache: str | Path = CACHE) -> CompiledTree:
    """The compiled tree from the cache, compiled and saved there if missing."""
    from uno.compiled.tree import compile_tree

    return compile_tree(rs, levels, symmetric, cache=cache, world=world)


def cfr_path(tree: CompiledTree, variant: str = "cfr+", cache: str | Path = CACHE) -> Path:
    suffix = "" if variant == "cfr+" else f"_{variant.replace('+', 'p')}"
    return Path(cache) / f"cfr_{cache_name(tree.rs, (Level.L1,), tree.symmetric)}{suffix}.npz"


def load_nash(tree: CompiledTree, variant: str | None = None, min_iterations: int = 0, cache: str | Path = CACHE) -> CFRResult | None:
    """A cached CFR result on the tree's L1 keying: of `variant`, or of any variant
    (CFR+ first, then the others by file name) when None. Only a result with at
    least `min_iterations` iterations over the same keying counts; None otherwise."""
    from uno.compiled.cfr import CFRResult

    base = cfr_path(tree, "cfr+", cache)
    paths = [cfr_path(tree, variant, cache)] if variant is not None else [base, *sorted(Path(cache).glob(base.stem + "_*.npz"))]
    n_ia = tree.keyings[Level.L1].n_ia
    for path in paths:
        if path.exists():
            res = CFRResult.load(path)
            if res.iterations >= min_iterations and len(res.avg) == n_ia:
                return res
    return None


def load_or_solve_nash(tree: CompiledTree, iterations: int, variant: str = "cfr+", verbose: bool = False, cache: str | Path = CACHE) -> CFRResult:
    """The cached CFR result of `variant` with at least `iterations` iterations, or
    a fresh run of that many iterations, saved to the cache."""
    res = load_nash(tree, variant, iterations, cache)
    if res is not None:
        if verbose:
            print(f"   loaded {cfr_path(tree, variant, cache).name}: {res.iterations} iterations, NashConv {100 * res.final.nashconv:.3f} pts, {res.seconds:.0f}s")
        return res
    from uno.compiled.cfr import cfr

    res = cfr(tree, iterations=iterations, verbose=verbose, variant=variant)
    path = cfr_path(tree, variant, cache)
    path.parent.mkdir(parents=True, exist_ok=True)
    res.save(path)
    if verbose:
        print(f"   {variant} {res.iterations} iterations in {res.seconds:.1f}s ({1000 * res.seconds / res.iterations:.1f} ms each), NashConv {100 * res.final.nashconv:.3f} pts, saved to {path.name}")
    return res
