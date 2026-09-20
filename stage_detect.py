#!/usr/bin/env python3
"""検出だけを載せて終了する。

認識器を import しない。約 2GiB で両方を初期化すると 137 になる。
既定は別置き ONNX。torch 検出は --backend torch で残す。比較用であり、既定に戻さない。
ONNX 経路では TextDetector を作らない。from_pretrained が重みを二重に広げるため。
"""

from __future__ import annotations

import argparse
import gc
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ONNX = os.path.join(ROOT, "weights", "pinned", "detector", "model.onnx")


def detect_onnx(img, onnx_path: str):
    import numpy as np
    import onnxruntime as ort
    import torch
    from yomitoku.data.functions import (
        array_to_tensor,
        resize_shortest_edge,
        standardization_image,
    )
    from yomitoku.postprocessor import DBnetPostProcessor

    ori_h, ori_w = img.shape[:2]
    x = img.copy()[:, :, ::-1].astype(np.float32)
    # 本体設定の shortest/limit。ここを変えると箱が旧面とずれる。
    x = resize_shortest_edge(x, 1280, 1600)
    x = standardization_image(x)
    tensor = array_to_tensor(x)

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(
        onnx_path, sess_options=so, providers=["CPUExecutionProvider"]
    )
    out = sess.run(["output"], {"input": tensor.numpy()})[0]
    preds = {"binary": torch.tensor(out)}
    pp = DBnetPostProcessor(
        min_size=2,
        thresh=0.3,
        box_thresh=0.4,
        max_candidates=1500,
        unclip_ratio=3.5,
    )
    quads, scores = pp(preds, (ori_h, ori_w))
    return quads, [float(s) for s in scores]


def detect_torch(img):
    import torch
    from yomitoku.text_detector import TextDetector

    torch.set_num_threads(1)
    det = TextDetector(device="cpu", visualize=False, infer_onnx=False)
    outputs, _ = det(img)
    points = outputs.points
    scores = [float(s) for s in outputs.scores]
    del det, outputs
    gc.collect()
    return points, scores


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
        "--backend",
        choices=("onnx", "torch"),
        default="onnx",
        help="既定 onnx。torch は比較用",
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
    print(f"img {img.shape} backend {args.backend}", flush=True)

    if args.backend == "onnx":
        if not os.path.isfile(args.onnx):
            raise SystemExit(
                f"missing ONNX: {args.onnx}\nexport: python3 weights/export_detector_onnx.py"
            )
        points, scores = detect_onnx(img, args.onnx)
    else:
        points, scores = detect_torch(img)

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
