#!/usr/bin/env python3
"""検出器だけを単一ファイルの ONNX に書く。

認識器は import しない。比較用。既定の検出は別置き ONNX。
YomiToku の convert_onnx は外部 .data を隣に出すので、ここで一つのファイルに直す。
opset 16 指定は変換に失敗し 18 のまま残ることがある。中身の箱が一致すればよい。
"""

from __future__ import annotations

import argparse
import gc
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


def main() -> None:
    parser = argparse.ArgumentParser(description="検出グラフだけを ONNX に書く")
    parser.add_argument(
        "-o",
        "--out",
        default="weights/pinned/detector/model.onnx",
        help="単一ファイルの ONNX",
    )
    args = parser.parse_args()

    import torch
    import onnx

    torch.set_num_threads(1)
    from yomitoku.text_detector import TextDetector

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".raw"

    det = TextDetector(device="cpu", visualize=False, infer_onnx=False)
    print("detector ready", flush=True)
    det.convert_onnx(tmp)
    del det
    gc.collect()

    model = onnx.load(tmp, load_external_data=True)
    onnx.save(model, out)
    raw_size = os.path.getsize(tmp)
    full_size = os.path.getsize(out)
    sidecar = tmp + ".data"
    print(f"wrote {out} {full_size} (raw {raw_size})", flush=True)
    for p in (tmp, sidecar):
        if os.path.isfile(p):
            os.remove(p)


if __name__ == "__main__":
    main()
