#!/usr/bin/env python3
"""認識の入口。encoder と decoder は別プロセス。

TextRecognizer は作らない。Dataset / Tokenizer も使わない。
作物と字表は rec/。pos_queries は decoder ONNX の中。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ROOT = os.path.dirname(os.path.abspath(__file__))
ENC = os.path.join(ROOT, "weights", "pinned", "recognizer", "encoder_dynw.onnx")
DEC = os.path.join(ROOT, "weights", "pinned", "recognizer", "decoder_step_dynw.onnx")


def main() -> None:
    parser = argparse.ArgumentParser(description="ONNX 認識。既定")
    parser.add_argument("image")
    parser.add_argument("--points", default="results/points.json")
    parser.add_argument("-o", "--out", default="results/ocr.json")
    parser.add_argument("--encoder", default=ENC)
    parser.add_argument("--decoder", default=DEC)
    parser.add_argument("--keep-mem", action="store_true", help="memory npz を残す")
    args = parser.parse_args()

    out_abs = os.path.abspath(args.out)
    mem = out_abs[:-5] + ".mem.npz" if out_abs.endswith(".json") else out_abs + ".mem.npz"
    py = sys.executable
    subprocess.check_call(
        [
            py,
            os.path.join(ROOT, "stage_recognize_encode.py"),
            args.image,
            "--points",
            args.points,
            "-o",
            mem,
            "--encoder",
            args.encoder,
        ]
    )
    subprocess.check_call(
        [
            py,
            os.path.join(ROOT, "stage_recognize_decode.py"),
            mem,
            "--points",
            args.points,
            "-o",
            args.out,
            "--decoder",
            args.decoder,
        ]
    )
    if not args.keep_mem and os.path.isfile(mem):
        os.remove(mem)
    print("recognize done", flush=True)


if __name__ == "__main__":
    main()
