#!/usr/bin/env python3
"""検出の推論だけを載せて終了する。箱出しはしない。

後処理と同じプロセスに載せると 2GiB で落ちることがある。
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DEFAULT_ONNX = os.path.join(ROOT, "weights", "pinned", "detector", "model.onnx")


def main() -> None:
    parser = argparse.ArgumentParser(description="検出の予測図だけ書いて検出器を捨てる")
    parser.add_argument("image")
    parser.add_argument("-o", "--out", default="results/pred.npz")
    parser.add_argument("--onnx", default=DEFAULT_ONNX)
    parser.add_argument(
        "--shortest",
        type=int,
        default=0,
        help="最短辺。0 なら本体と同じ 1280",
    )
    args = parser.parse_args()

    import numpy as np
    import onnxruntime as ort
    from cvsurf import cv2
    from det.preprocess import prepare_bgr

    if not os.path.isfile(args.onnx):
        raise SystemExit(f"missing ONNX: {args.onnx}")
    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"failed to read image: {args.image}")
    h, w = img.shape[:2]
    print(f"img {img.shape} infer", flush=True)
    shortest = args.shortest if args.shortest > 0 else None
    tensor = prepare_bgr(img) if shortest is None else prepare_bgr(img, shortest_edge_length=shortest)
    del img
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(
        args.onnx, sess_options=so, providers=["CPUExecutionProvider"]
    )
    out = sess.run(["output"], {"input": tensor})[0]
    del sess, tensor
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        args.out,
        pred=out.astype(np.float16),
        dest_h=np.int32(h),
        dest_w=np.int32(w),
    )
    print(f"wrote {args.out} pred {tuple(out.shape)}", flush=True)


if __name__ == "__main__":
    main()
