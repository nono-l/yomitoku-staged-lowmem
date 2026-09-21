#!/usr/bin/env python3
"""インク検出。モデルを載せない。"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

from det.ink_boxes import boxes_from_bgr


def main() -> None:
    blank = np.full((80, 120, 3), 240, dtype=np.uint8)
    q, s = boxes_from_bgr(blank)
    if q:
        raise SystemExit(f"blank must be empty, got {len(q)}")

    marked = blank.copy()
    marked[20:36, 10:90] = 20
    q, s = boxes_from_bgr(marked)
    if len(q) < 1:
        raise SystemExit("ink bar should make a box")
    if any(len(p) != 4 for p in q):
        raise SystemExit("quad must have 4 corners")
    print("ok ink boxes", len(q), "score", s[0])


if __name__ == "__main__":
    main()
