#!/usr/bin/env python3
"""初回の箱とタイルの箱を足す。検出器は載せない。

重なったら初回を残す。run_staged.sh の既定には入れない。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description="初回の箱に、重ならないタイル箱を足す")
    parser.add_argument("first", help="初回の points.json")
    parser.add_argument("extra", help="タイル検出の points.json")
    parser.add_argument("-o", "--out", default="results/points_merged.json")
    parser.add_argument("--max-iou", type=float, default=0.5)
    args = parser.parse_args()

    from det.merge_boxes import merge_points

    def load(path: str) -> dict:
        with open(path, encoding="utf-8") as f:
            pack = json.load(f)
        if not isinstance(pack, dict):
            raise SystemExit(f"{path} はオブジェクトである")
        return pack

    merged = merge_points(load(args.first), load(args.extra), max_iou=args.max_iou)
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(merged, f)
    print(f"n_boxes {len(merged['points'])}", flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
