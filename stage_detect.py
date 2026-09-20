#!/usr/bin/env python3
"""検出だけを載せて終了する。

認識器を import しない。約 2GiB で両方を初期化すると 137 になる。
フル CLI や Layout / Table も同じ理由で呼ばない。
"""

from __future__ import annotations

import argparse
import gc
import json
import os

# 検出だけでも、デフォルトの並列はこの箱の RAM を食いつぶす。
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def main() -> None:
    parser = argparse.ArgumentParser(description="文字箱だけ出して検出器を捨てる")
    parser.add_argument("image", help="入力画像")
    parser.add_argument(
        "-o",
        "--out",
        default="results/points.json",
        help="箱の JSON（既定: results/points.json）",
    )
    args = parser.parse_args()

    import cv2
    import torch

    torch.set_num_threads(1)

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"failed to read image: {args.image}")
    print(f"img {img.shape}", flush=True)

    from yomitoku.text_detector import TextDetector

    det = TextDetector(device="cpu", visualize=False, infer_onnx=False)
    print("detector ready", flush=True)
    outputs, _ = det(img)
    points = outputs.points
    scores = [float(s) for s in outputs.scores]
    print(f"n_boxes {len(points)}", flush=True)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"points": points, "scores": scores}, f)
    print(f"wrote {args.out}", flush=True)

    del det, outputs
    gc.collect()
    print("detect done", flush=True)


if __name__ == "__main__":
    main()
