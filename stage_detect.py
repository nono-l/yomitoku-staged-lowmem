#!/usr/bin/env python3
"""検出だけを載せて終了する。

認識器を import しない。約 2GiB で両方を初期化すると 137 になる。
既定は別置き ONNX。yomitoku も torch も使わない。
torch 検出は comparison/stage_detect_torch.py。
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DEFAULT_ONNX = os.path.join(ROOT, "weights", "pinned", "detector", "model.onnx")


def detect_onnx(img, onnx_path: str):
    import onnxruntime as ort
    from det.postprocess import boxes_from_binary
    from det.preprocess import prepare_bgr

    tensor = prepare_bgr(img)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(
        onnx_path, sess_options=so, providers=["CPUExecutionProvider"]
    )
    out = sess.run(["output"], {"input": tensor})[0]
    ori_h, ori_w = img.shape[:2]
    quads, scores = boxes_from_binary(out, (ori_h, ori_w))
    return quads, scores


def main() -> None:
    parser = argparse.ArgumentParser(description="文字箱だけ出して検出器を捨てる")
    parser.add_argument("image", help="入力画像")
    parser.add_argument(
        "-o",
        "--out",
        default="results/points.json",
        help="箱の JSON（既定: results/points.json）",
    )
    parser.add_argument(
        "--onnx",
        default=DEFAULT_ONNX,
        help="別置き検出グラフ",
    )
    args = parser.parse_args()

    import cv2

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"failed to read image: {args.image}")
    print(f"img {img.shape} backend onnx", flush=True)

    if not os.path.isfile(args.onnx):
        raise SystemExit(
            f"missing ONNX: {args.onnx}\nexport: python3 weights/export_detector_onnx.py"
        )
    points, scores = detect_onnx(img, args.onnx)

    print(f"n_boxes {len(points)}", flush=True)
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"points": points, "scores": scores}, f)
    print(f"wrote {args.out}", flush=True)
    gc.collect()
    print("detect done", flush=True)


if __name__ == "__main__":
    main()
