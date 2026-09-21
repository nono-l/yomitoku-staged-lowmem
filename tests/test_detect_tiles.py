#!/usr/bin/env python3
"""空いた区画の切り出し。モデルを載せない。"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

from det.tiles import empty_tiles, ink_ratio


def main() -> None:
    full = [[0, 0], [800, 0], [800, 600], [0, 600]]
    none = empty_tiles(800, 600, [full], tile=320)
    if none:
        raise SystemExit("full cover must yield no tiles")

    left = [[0, 0], [400, 0], [400, 600], [0, 600]]
    tiles = empty_tiles(800, 600, [left], tile=400)
    xs = sorted((t["x"], t["w"]) for t in tiles)
    if not xs:
        raise SystemExit("right half should be empty")
    if any(x < 400 for x, _ in xs):
        raise SystemExit("left tiles must stay occupied")
    if min(x for x, _ in xs) != 400:
        raise SystemExit("first empty tile should start at 400")

    vacant = empty_tiles(640, 320, [], tile=320)
    if len(vacant) != 2:
        raise SystemExit(f"empty 640x320 / 320 -> 2 tiles, got {len(vacant)}")

    blank = np.full((320, 640, 3), 240, dtype=np.uint8)
    if empty_tiles(640, 320, [], tile=320, img=blank, min_ink=0.02):
        raise SystemExit("blank margin must drop")
    marked = blank.copy()
    marked[40:80, 40:200] = 20
    kept = empty_tiles(640, 320, [], tile=320, img=marked, min_ink=0.02)
    if len(kept) != 1:
        raise SystemExit(f"ink tile should remain, got {len(kept)}")
    if ink_ratio(blank, 0, 0, 320, 320) >= 0.02:
        raise SystemExit("blank ink_ratio")
    print("ok empty tiles", len(tiles), "vacant", len(vacant), "ink", len(kept))


if __name__ == "__main__":
    main()
