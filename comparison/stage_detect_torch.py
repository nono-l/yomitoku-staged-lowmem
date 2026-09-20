#!/usr/bin/env python3
"""torch 検出。比較用。既定の入口ではない。"""

from __future__ import annotations

import argparse
import gc
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


def main() -> None:
    parser = argparse.ArgumentParser(description="比較用 torch 検出")
    parser.add_argument("image")
    parser.add_argument("-o", "--out", default="results/points_torch.json")
    args = parser.parse_args()

    import cv2
    import torch
    from yomitoku.text_detector import TextDetector

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"failed to read image: {args.image}")
    torch.set_num_threads(1)
    det = TextDetector(device="cpu", visualize=False, infer_onnx=False)
    outputs, _ = det(img)
    points = outputs.points
    scores = [float(s) for s in outputs.scores]
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"points": points, "scores": scores}, f)
    print(f"n_boxes {len(points)} wrote {args.out}", flush=True)
    del det, outputs
    gc.collect()


if __name__ == "__main__":
    main()
