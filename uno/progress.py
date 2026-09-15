"""Progress bars (tqdm) and loop timings for the long loops: compiles, CFR,
simulations, exact walks, sizing.

A loop shorter than 2 s shows no bar and records no timing. On a terminal a bar
redraws twice a second; otherwise it writes every 5 s, to `$UNO_PROGRESS_FILE`
when set (`scripts/rerun_all.sh` points it next to each step's log, which then
holds only the experiment's own output), to stderr otherwise. When
`$UNO_TIMINGS_FILE` is set, every loop of 2 s or more appends one line there as
it ends: `[hh:mm:ss] <what>: took <s> s for <count> <unit> (<rate> <unit>/s)`.
`UNO_PROGRESS=0` turns both off; the tests do.
"""

from __future__ import annotations

import os
import sys
import time
from typing import IO

from tqdm import tqdm

MIN_SECONDS = 2.0
_files: dict[str, IO[str]] = {}


def _enabled() -> bool:
    return os.environ.get("UNO_PROGRESS", "1") != "0"


def _open(var: str) -> IO[str] | None:
    path = os.environ.get(var)
    if not path:
        return None
    f = _files.get(path)
    if f is None:
        f = _files[path] = open(path, "a", buffering=1)
    return f


class Bar(tqdm):
    """A tqdm bar that appends its timing to `$UNO_TIMINGS_FILE` when it closes after `MIN_SECONDS` or more."""

    def close(self) -> None:
        if not self.disable and not getattr(self, "_timed", False):
            self._timed = True
            elapsed = self._time() - self.start_t
            out = _open("UNO_TIMINGS_FILE")
            if out is not None and elapsed >= MIN_SECONDS:
                unit = self.unit.strip()
                out.write(
                    f"[{time.strftime('%H:%M:%S')}] {self.desc or 'loop'}: took {elapsed:.1f} s for "
                    f"{tqdm.format_sizeof(self.n)} {unit} ({tqdm.format_sizeof(self.n / elapsed)} {unit}/s)\n"
                )
        super().close()


def bar(iterable=None, total: int | None = None, desc: str | None = None, unit: str = "items") -> Bar:
    """A bar over `iterable`, or for manual `update`s up to `total`; with neither
    a length nor a total it is a counter with a rate. `unit` is plural ("nodes")."""
    enabled = _enabled()
    out = (_open("UNO_PROGRESS_FILE") if enabled else None) or sys.stderr
    tty = out.isatty()
    if total is None and iterable is not None and hasattr(iterable, "__len__"):
        total = len(iterable)
    return Bar(
        iterable,
        total=total,
        desc=desc,
        unit=" " + unit,  # tqdm glues the scaled count to the unit: "200k nodes/s", not "200knodes/s"
        unit_scale=True,
        file=out,
        disable=not enabled,
        delay=MIN_SECONDS,
        mininterval=0.5 if tty else 5.0,
        dynamic_ncols=tty,
        leave=False,
    )
