"""The on-disk cache of compiled trees and CFR results.

Files live in `CACHE` (the repo's `.cache/`, or `$UNO_CACHE`), git-ignored:

- `tree_<name>_<levels>[W]_<sym|full>.npz`: a compiled tree (`CompiledTree.save`),
  `W` for world roots without an L3 keying;
- `cfr_<name>_L1_<sym|full>[_<variant>].npz`: a CFR result (`CFRResult.save`), no
  suffix for CFR+, `+` spelled `p` otherwise (`_dcfr`, `_pcfrp`);
- `tab_<name>_<D|W>_<sym|full>_<level>_<agent>.npz`: an agent tabulated over a
  keying (`load_or_tabulate`), `D` for deal roots, `W` for world roots.
- `label_<name>_<D|W>_<sym|full>_<level>.npz`: `uno.eval.decisions.set_labels` for a
  keying (`load_or_label`, step 9.5a): no agent, a property of the tree alone.

`<name>` is the ruleset's name in `uno.game.rules.NAMED`, or a hash of its repr.
Every experiment loads trees and Nash strategies through here.

Every file carries `CACHE_VERSION`; a file with another version (or none, before
step 8.9) is stale: trees are recompiled and CFR results solved again, and the
new file overwrites it.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from uno.game.observe import Level
from uno.game.rules import NAMED, Ruleset

if TYPE_CHECKING:
    from uno.compiled.cfr import CFRResult
    from uno.compiled.tree import CompiledTree

CACHE = Path(os.environ.get("UNO_CACHE", Path(__file__).resolve().parent.parent / ".cache"))

# Bump whenever a change to the engine, the chance model, the keys, the symmetry or the
# tree layout changes what a compile (or a CFR run on it) produces, or a baseline agent's
# policy changes (its tabulations are cached by name).
CACHE_VERSION = 1


def is_current(path: str | Path) -> bool:
    """Whether a cached file exists and was written under the current `CACHE_VERSION`."""
    path = Path(path)
    if not path.exists():
        return False
    with np.load(path) as z:
        return "version" in z.files and int(z["version"]) == CACHE_VERSION


def ruleset_name(rs: Ruleset) -> str:
    """The ruleset's name in `NAMED`, or a short stable hash of its repr."""
    name = next((n for n, r in NAMED.items() if r == rs), None)
    return name or hashlib.blake2b(repr(rs).encode(), digest_size=6).hexdigest()


def cache_name(rs: Ruleset, levels: tuple[Level, ...], symmetric: bool, world: bool | None = None) -> str:
    """Short stable name for a compiled tree; `W` marks world roots without an L3 keying."""
    w = "W" if world and Level.L3 not in levels else ""
    return f"{ruleset_name(rs)}_{''.join(lv.name for lv in levels)}{w}_{'sym' if symmetric else 'full'}"


def load_or_compile(rs: Ruleset, levels: tuple[Level, ...] = (Level.L1,), symmetric: bool = True, world: bool | None = None, cache: str | Path = CACHE) -> CompiledTree:
    """The compiled tree from the cache, compiled and saved there if missing or stale."""
    from uno.compiled.tree import compile_tree

    return compile_tree(rs, levels, symmetric, cache=cache, world=world)


def cfr_path(tree: CompiledTree, variant: str = "cfr+", cache: str | Path = CACHE) -> Path:
    suffix = "" if variant == "cfr+" else f"_{variant.replace('+', 'p')}"
    return Path(cache) / f"cfr_{cache_name(tree.rs, (Level.L1,), tree.symmetric)}{suffix}.npz"


def load_nash(tree: CompiledTree, variant: str | None = None, min_iterations: int = 0, cache: str | Path = CACHE) -> CFRResult | None:
    """A cached CFR result on the tree's L1 keying: of `variant`, or of any variant
    (CFR+ first, then the others by file name) when None. Only a current result with
    at least `min_iterations` iterations over the same keying counts; None otherwise."""
    from uno.compiled.cfr import CFRResult

    base = cfr_path(tree, "cfr+", cache)
    paths = [cfr_path(tree, variant, cache)] if variant is not None else [base, *sorted(Path(cache).glob(base.stem + "_*.npz"))]
    n_ia = tree.keyings[Level.L1].n_ia
    for path in paths:
        if is_current(path):
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


def tabulation_path(tree: CompiledTree, level: Level, name: str, cache: str | Path = CACHE) -> Path:
    """`W` for a world tree, `D` for a lazy one: the L1 numbering depends on the roots,
    not on which other keyings a tree carries."""
    roots = "W" if tree.roots_level is Level.L3 else "D"
    return Path(cache) / f"tab_{ruleset_name(tree.rs)}_{roots}_{'sym' if tree.symmetric else 'full'}_{level.name}_{name}.npz"


def keying_fingerprint(tree: CompiledTree, level: Level) -> str:
    k = tree.keyings[level]
    h = hashlib.blake2b(digest_size=8)
    h.update(k.info_start.tobytes())
    h.update(k.ia_action.tobytes())
    return h.hexdigest()


def load_or_tabulate(tree: CompiledTree, level: Level, agent, cache: str | Path = CACHE) -> np.ndarray:
    """`uno.compiled.strategies.tabulate` through the cache (step 9): a Python call per
    set costs ~50 s per agent on TOY2_FULL. Keyed on the agent's name (a name
    identifies a policy, docs/spec.md section 13) and on a fingerprint of the keying; a
    file of another `CACHE_VERSION` or keying is tabulated again and overwritten. Meant
    for the fixed baseline agents: bump `CACHE_VERSION` when one of their policies changes."""
    from uno.compiled.strategies import tabulate

    path = tabulation_path(tree, level, agent.name, cache)
    fp = keying_fingerprint(tree, level)
    if is_current(path):
        with np.load(path) as z:
            if str(z["fingerprint"]) == fp and len(z["sigma"]) == tree.keyings[level].n_ia:
                return z["sigma"].copy()
    sigma = tabulate(tree, level, agent)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, version=np.array(CACHE_VERSION), fingerprint=np.array(fp), sigma=sigma)
    return sigma


def label_path(tree: CompiledTree, level: Level, cache: str | Path = CACHE) -> Path:
    roots = "W" if tree.roots_level is Level.L3 else "D"
    return Path(cache) / f"label_{ruleset_name(tree.rs)}_{roots}_{'sym' if tree.symmetric else 'full'}_{level.name}.npz"


def load_or_label(tree: CompiledTree, level: Level = Level.L1, cache: str | Path = CACHE) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`uno.eval.decisions.set_labels` through the cache (step 9.5a): a Python call per
    set, the same cost as `load_or_tabulate`. Keyed only on a fingerprint of the
    keying (labels are a property of the tree, not of any agent); a file of another
    `CACHE_VERSION` or keying is relabelled and overwritten."""
    from uno.eval.decisions import set_labels

    path = label_path(tree, level, cache)
    fp = keying_fingerprint(tree, level)
    if is_current(path):
        with np.load(path) as z:
            if str(z["fingerprint"]) == fp:
                return z["kinds"].astype(str), z["situations"].astype(str), z["actions"].astype(str)
    kinds, sits, actions = set_labels(tree, level)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, version=np.array(CACHE_VERSION), fingerprint=np.array(fp), kinds=kinds, situations=sits, actions=actions)
    return kinds, sits, actions
