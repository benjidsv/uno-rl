"""Cut the classic UNO spritesheet into the per-card files `uno play` looks for.

    uv run python -m uno.play.slice_sheet [SHEET] [OUT_DIR]

Defaults: `uno/play/static/spritesheet.png` -> `uno/play/static/cards/`. The sheet is a
12-wide grid on a dark background: row 0 holds the back, the wild (plain, then
pre-coloured yellow/red/blue/green), the wild +4 (same five) and a blank wild; then
13 cards per colour in yellow, red, blue, green order: 1-9, 0, +2, skip, reverse.
Cells are found from the background rather than a fixed pitch, which is uneven.
Pillow comes in with matplotlib.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).parent / "static"

ROW0 = ["back", "wild", "wild_yellow", "wild_red", "wild_blue", "wild_green",
        "wild_draw4", "wild_draw4_yellow", "wild_draw4_red", "wild_draw4_blue", "wild_draw4_green", None]
COLOURED = [f"{c}_{k}" for c in ("yellow", "red", "blue", "green")
            for k in (*"123456789", "0", "draw2", "skip", "reverse")]
NAMES = ROW0 + COLOURED


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return list(zip(edges[::2], edges[1::2]))


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    sheet = Path(argv[0]) if argv else HERE / "spritesheet.png"
    out = Path(argv[1]) if len(argv) > 1 else HERE / "cards"
    img = np.asarray(Image.open(sheet).convert("RGBA")).copy()
    bg = img[-1, -1, :3].astype(int)
    card = np.abs(img[..., :3].astype(int) - bg).sum(-1) > 30
    cols = _runs(card[: img.shape[0] // 8].any(0))  # first row is complete
    rows = _runs(card[:, : img.shape[1] // 12].any(1))  # first column is complete
    if len(cols) != 12:
        raise SystemExit(f"expected 12 columns, found {len(cols)}")
    out.mkdir(parents=True, exist_ok=True)
    written = 0
    for i, name in enumerate(NAMES):
        if name is None:
            continue
        (y0, y1), (x0, x1) = rows[i // 12], cols[i % 12]
        cell, bgpix = img[y0:y1, x0:x1].copy(), ~card[y0:y1, x0:x1]
        r = min(cell.shape[:2]) // 8  # only the rounded corners: dark ink inside a card stays
        for ys in (slice(0, r), slice(-r, None)):
            for xs in (slice(0, r), slice(-r, None)):
                cell[ys, xs, 3][bgpix[ys, xs]] = 0
        Image.fromarray(cell).save(out / f"{name}.png")
        written += 1
    print(f"wrote {written} cards ({x1 - x0}x{y1 - y0}) to {out}")


if __name__ == "__main__":
    main()
