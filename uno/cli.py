"""`uv run uno <experiment> [args...]`: run one step experiment of `uno.experiments`.

`uv run uno list` (or no argument) lists them with the first line of their docstring.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys

import uno.experiments


def experiments() -> list[str]:
    return sorted(m.name for m in pkgutil.iter_modules(uno.experiments.__path__) if m.name != "common")


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    names = experiments()
    if not argv or argv[0] in ("list", "-h", "--help"):
        print("usage: uno <experiment> [args...]\n\nexperiments:")
        for name in names:
            doc = (importlib.import_module(f"uno.experiments.{name}").__doc__ or "").strip().splitlines()
            print(f"  {name:14s} {doc[0] if doc else ''}")
        return
    if argv[0] not in names:
        raise SystemExit(f"unknown experiment {argv[0]!r}; `uno list` shows them")
    importlib.import_module(f"uno.experiments.{argv[0]}").main(argv[1:])


if __name__ == "__main__":
    main()
