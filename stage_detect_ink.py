#!/usr/bin/env python3
"""インク塊から箱を出す。DBNet も ORT も載せない。

既定の stage_detect.py は置き換えない。run_staged.sh からも呼ばない。
失敗したらこの入口だけ捨てる。
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
    parser = argparse.ArgumentParser(description="ネット無しで文字箱を試す")
    parser.add_argument("image")
    parser.add_argument("-o", "--out", default="results/points_ink.json")
    args = parser.parse_args()

    from cvsurf import cv2
    from det.ink_boxes import boxes_from_bgr

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"failed to read image: {args.image}")
    quads, scores = boxes_from_bgr(img)
    print(f"n_boxes {len(quads)}", flush=True)
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"points": quads, "scores": scores, "detector": "ink"}, f)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
