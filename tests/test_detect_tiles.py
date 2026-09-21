#!/usr/bin/env python3
"""空いた区画の切り出し。モデルを載せない。"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from det.tiles import empty_tiles


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
    print("ok empty tiles", len(tiles), "vacant", len(vacant))


if __name__ == "__main__":
    main()
