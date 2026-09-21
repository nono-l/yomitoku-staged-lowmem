#!/usr/bin/env python3
"""初回優先の箱足し。モデルを載せない。"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from det.merge_boxes import merge_points


def main() -> None:
    first = {
        "points": [[[0, 0], [40, 0], [40, 20], [0, 20]]],
        "scores": [0.9],
    }
    same = {
        "points": [[[2, 1], [38, 1], [38, 19], [2, 19]]],
        "scores": [0.99],
    }
    got = merge_points(first, same)
    if len(got["points"]) != 1:
        raise SystemExit("overlap must keep first only")
    if got["scores"][0] != 0.9:
        raise SystemExit("first score must stay")

    far = {
        "points": [[[80, 0], [120, 0], [120, 20], [80, 20]]],
        "scores": [0.4],
    }
    got = merge_points(first, far)
    if len(got["points"]) != 2:
        raise SystemExit("distant box must be added")
    if got["scores"][1] != 0.4:
        raise SystemExit("extra score")
    print("ok merge keep-first")


if __name__ == "__main__":
    main()
