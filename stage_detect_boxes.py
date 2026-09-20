#!/usr/bin/env python3
"""予測図から箱だけ出す。検出器は載せない。"""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description="予測図から文字箱を出す")
    parser.add_argument("pred", help="stage_detect_infer の npz")
    parser.add_argument("-o", "--out", default="results/points.json")
    args = parser.parse_args()

    import numpy as np
    from det.postprocess import boxes_from_binary

    pack = np.load(args.pred)
    pred = pack["pred"].astype(np.float32)
    dest = (int(pack["dest_h"]), int(pack["dest_w"]))
    quads, scores = boxes_from_binary(pred, dest)
    print(f"n_boxes {len(quads)}", flush=True)
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"points": quads, "scores": scores}, f)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
