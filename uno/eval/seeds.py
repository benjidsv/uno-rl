"""The seed registry: which seeds a step is allowed to report on (step 10a, item A1).

`uno.eval.play` fixed two ranges: `DEV_SEEDS` for building, testing and tuning, and
`EVAL_SEEDS` held out for reported numbers, "nothing is ever chosen on them". Steps
9.5d-9.5f broke the second half of that rule without noticing: the working set of the
full-UNO search row -- the `response` continuation, w = 32, the `voids` belief -- was
*chosen* by 3,000-4,000-deal runs on `EVAL_SEEDS`, so those 4,000 seeds now carry
selection and are a dev range in everything but name (09.5f-results.md section 7.3
says so outright for `window` at seat 1).

Seeds were never the scarce resource: a 64-bit seed space is free and the engine is a
pure function of it. The **declaration** was scarce, because there was only one range
and no record of what had been spent on what. So this module widens the pools and
partitions the eval pool into disjoint **blocks**, each claimed by one step for one
purpose, with the claim written down here rather than in the experiment that happens
to run first.

    DEV_POOL   [0, 10^6)        build, test, tune, select; choose on these freely
    EVAL_POOL  [10^6, 2 * 10^6) report; one block per claim, never reused

The rule a block encodes: **a number is reported on a block only if nothing was
selected on that block.** Block 00 is the counter-example, kept in the table rather
than quietly retired -- it records that 1,000,000-1,003,999 is spent, so a step 11
report card cannot silently reuse the seeds its own working set was tuned on.

Claims are checked for overlap at import (`_validate`), so two steps cannot be given
the same seeds by a typo, and `claim` refuses a name that is already in the table.
"""

from __future__ import annotations

from dataclasses import dataclass

# The pools. `DEV_POOL` starts at 0 so the historical `DEV_SEEDS = range(4000)` is its
# first block and every number ever measured on dev seeds still reproduces.
DEV_POOL = range(0, 1_000_000)
EVAL_POOL = range(1_000_000, 2_000_000)

BLOCK = 4_000  # seeds per block: the project's standard eval size (step 7 onwards)


@dataclass(frozen=True, slots=True)
class Block:
    """One claimed range of `EVAL_POOL`, and what it was spent on."""

    name: str
    start: int
    stop: int
    step: str  # the build-order step that claimed it (docs/plan.md)
    purpose: str
    spent: bool = False  # True: something was *selected* on it, so it may not carry a reported number

    @property
    def seeds(self) -> range:
        return range(self.start, self.stop)

    def __len__(self) -> int:
        return self.stop - self.start

    def __str__(self) -> str:
        mark = " SPENT" if self.spent else ""
        return f"{self.name} [{self.start:,}, {self.stop:,}) {len(self):,} seeds, {self.step}: {self.purpose}{mark}"


# The registry. Append; never renumber, never widen an existing block, never reuse one.
REGISTRY: dict[str, Block] = {
    b.name: b
    for b in (
        Block(
            "00",
            1_000_000,
            1_004_000,
            "7-9.5f",
            "the original EVAL_SEEDS. Reported numbers from steps 7-9.5c, then spent: "
            "9.5d-9.5f chose the full-UNO working set (response continuation, w = 32, voids) "
            "on 3,000-4,000-deal runs over it",
            spent=True,
        ),
        Block(
            "01",
            1_004_000,
            1_008_000,
            "10a.2",
            "the column-free PIMC row on FULL_UNO (item A5a): the first full-UNO row whose "
            "continuation and belief read nothing of the column's real policy, against the "
            "`response` row it must reproduce when handed its own column",
        ),
        Block(
            "02",
            1_008_000,
            1_012_000,
            "10a.3C",
            "the LBR attack baseline on FULL_UNO: per-seat attack scores and the two-seat "
            "NashConv lower bound against Random and the Heuristic, at the configuration frozen "
            "by the 10a.3C DEV pilot (response continuation, w = 32, d = 2, voids belief, "
            "attacker seed 0). The pilot put the bound's standard error at ~4.0 points on 200 "
            "deals, so one block takes it to ~0.9; the column-free PIMC target is not on this "
            "block, since at 8.11 s/game it would need ~1,340 deals for a 5-point bound",
        ),
        Block(
            "03",
            1_012_000,
            1_016_000,
            "10C",
            "confirmatory full-UNO scores and behavioral fingerprints for the frozen four-agent "
            "rulebot league against Random and the Heuristic, both seats. Transfer weights were "
            "selected exactly on TOY/TOY2_SKIP; the full-UNO and stochastic variants were selected "
            "on DEV before this block was claimed",
        ),
    )
}


def _validate(registry: dict[str, Block]) -> None:
    """No block escapes `EVAL_POOL`, and no two blocks overlap (A1's assertion)."""
    for b in registry.values():
        if b.start < EVAL_POOL.start or b.stop > EVAL_POOL.stop:
            raise ValueError(f"block {b.name} [{b.start:,}, {b.stop:,}) is outside EVAL_POOL")
        if b.stop <= b.start:
            raise ValueError(f"block {b.name} is empty")
    ordered = sorted(registry.values(), key=lambda b: b.start)
    for a, b in zip(ordered, ordered[1:]):
        if a.stop > b.start:
            raise ValueError(f"blocks {a.name} [{a.start:,}, {a.stop:,}) and {b.name} [{b.start:,}, {b.stop:,}) overlap")


_validate(REGISTRY)


def block(name: str) -> range:
    """The seeds of a claimed block. Raises on an unknown block, and on a spent one:
    a spent block may not carry a reported number (see the module docstring)."""
    b = REGISTRY.get(name)
    if b is None:
        raise KeyError(f"no eval block {name!r}; claimed: {', '.join(sorted(REGISTRY)) or 'none'}")
    if b.spent:
        raise ValueError(f"eval block {name} is spent ({b.purpose}); claim a fresh block for a reported number")
    return b.seeds


def dev_block(start: int = 0, n: int = BLOCK) -> range:
    """A slice of `DEV_POOL`. Unclaimed by design: nothing reported comes off it, so
    two dev uses overlapping costs nothing."""
    if start < 0 or start + n > len(DEV_POOL):
        raise ValueError(f"dev block [{start:,}, {start + n:,}) is outside DEV_POOL")
    return range(DEV_POOL.start + start, DEV_POOL.start + start + n)


def claim(name: str, step: str, purpose: str, n: int = BLOCK) -> Block:
    """Claim the next free `n` seeds of `EVAL_POOL` under `name`.

    Meant to be called once, at module import, with the result written into `REGISTRY`
    above -- a claim that lives only in a running process is not a declaration. Raises
    if `name` is already claimed, which is A1's "a block cannot be claimed twice"."""
    if name in REGISTRY:
        raise ValueError(f"eval block {name} is already claimed ({REGISTRY[name].step}: {REGISTRY[name].purpose})")
    start = max((b.stop for b in REGISTRY.values()), default=EVAL_POOL.start)
    b = Block(name, start, start + n, step, purpose)
    candidate = dict(REGISTRY)
    candidate[name] = b
    _validate(candidate)
    REGISTRY[name] = b
    return b


def describe() -> str:
    """The registry as a table, for an experiment's header."""
    lines = [f"eval pool [{EVAL_POOL.start:,}, {EVAL_POOL.stop:,}), {len(REGISTRY)} claimed block(s):"]
    lines += [f"   {b}" for b in sorted(REGISTRY.values(), key=lambda b: b.start)]
    free = max((b.stop for b in REGISTRY.values()), default=EVAL_POOL.start)
    lines.append(f"   next free seed {free:,} ({(EVAL_POOL.stop - free) // BLOCK:,} blocks of {BLOCK:,} left)")
    return "\n".join(lines)
