#!/usr/bin/env python3
"""作物を encoder に通して memory だけ書く。decoder は載せない。"""

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

ENC = os.path.join(ROOT, "weights", "pinned", "recognizer", "encoder_dynw.onnx")


def main() -> None:
    parser = argparse.ArgumentParser(description="認識の encoder だけ")
    parser.add_argument("image")
    parser.add_argument("--points", default="results/points.json")
    parser.add_argument("-o", "--out", default="results/memory.npz")
    parser.add_argument("--encoder", default=ENC)
    args = parser.parse_args()

    import numpy as np
    import onnxruntime as ort
    from cvsurf import cv2
    from rec.crop import crop_quad

    if not os.path.isfile(args.encoder):
        raise SystemExit(f"missing {args.encoder}")
    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"failed to read image: {args.image}")
    points = json.load(open(args.points, encoding="utf-8"))["points"]
    print(f"img {img.shape} n_boxes {len(points)} encode", flush=True)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    enc = ort.InferenceSession(
        args.encoder, sess_options=so, providers=["CPUExecutionProvider"]
    )
    pack = {"n": np.int32(len(points))}
    valid = []
    for i, quad in enumerate(points):
        crop = crop_quad(img, quad)
        if crop is None:
            pack[f"m{i}"] = np.zeros((0,), dtype=np.float16)
            valid.append(0)
            continue
        memory = enc.run(["memory"], {"input": crop})[0]
        pack[f"m{i}"] = memory.astype(np.float16)
        valid.append(1)
        if (i + 1) % 10 == 0 or i + 1 == len(points):
            print(f"encode {i + 1}", flush=True)
    pack["valid"] = np.asarray(valid, dtype=np.uint8)
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(args.out, **pack)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
