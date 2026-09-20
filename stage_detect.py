#!/usr/bin/env python3
"""検出の入口。推論と箱出しは別プロセス。

認識器を import しない。約 2GiB で両方を初期化すると 137 になる。
既定は別置き ONNX。yomitoku も torch も使わない。
torch 検出は comparison/stage_detect_torch.py。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ONNX = os.path.join(ROOT, "weights", "pinned", "detector", "model.onnx")


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
    parser.add_argument(
        "--keep-pred",
        action="store_true",
        help="予測図 npz を残す",
    )
    args = parser.parse_args()

    out_abs = os.path.abspath(args.out)
    pred = out_abs[:-5] + ".pred.npz" if out_abs.endswith(".json") else out_abs + ".pred.npz"
    py = sys.executable
    subprocess.check_call(
        [
            py,
            os.path.join(ROOT, "stage_detect_infer.py"),
            args.image,
            "-o",
            pred,
            "--onnx",
            args.onnx,
        ]
    )
    subprocess.check_call(
        [
            py,
            os.path.join(ROOT, "stage_detect_boxes.py"),
            pred,
            "-o",
            args.out,
        ]
    )
    if not args.keep_pred and os.path.isfile(pred):
        os.remove(pred)
    print("detect done", flush=True)


if __name__ == "__main__":
    main()
