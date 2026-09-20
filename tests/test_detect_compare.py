#!/usr/bin/env python3
"""T2。凍結した旧検出と ONNX 検出の箱が、同じ絵で重なるか。"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tests"))
from compare_detect import compare, load

OLD = os.path.join(ROOT, "tests", "fixtures", "points_torch_settei21.json")
NEW = os.path.join(ROOT, "tests", "fixtures", "points_onnx_settei21.json")


def main() -> None:
    stats = compare(load(OLD), load(NEW), thresh=0.5)
    print(stats)
    if stats["n_old"] != stats["n_new"]:
        raise SystemExit("box count differs")
    if stats["matched"] != stats["n_old"]:
        raise SystemExit("unmatched boxes")
    if stats["min_iou"] < 0.99:
        raise SystemExit("iou too low")
    print("ok detect compare")


if __name__ == "__main__":
    main()
