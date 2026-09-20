#!/usr/bin/env python3
"""認識の encoder だけを ONNX に書く。

検出器は import しない。既定の認識はまだ torch tiny である。
本体の convert_onnx は AR 解碼を max_label 歩展開する。デコーダは 2.4MB なのに
グラフが爆発するので使わない。encoder は 22MB、一回の前進だけ。
入力は本体 ONNX 経路と同じ 1x3x32x800。dynamic_width はこのグラフでは使わない。
"""

from __future__ import annotations

import argparse
import gc
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


def main() -> None:
    parser = argparse.ArgumentParser(description="認識 encoder だけを ONNX に書く")
    parser.add_argument(
        "-o",
        "--out",
        default="weights/pinned/recognizer/encoder.onnx",
        help="単一ファイルの encoder ONNX",
    )
    args = parser.parse_args()

    import torch
    import onnx
    from yomitoku.text_recognizer import TextRecognizer

    torch.set_num_threads(1)
    rec = TextRecognizer(
        model_name="parseq-tiny-dynw-v5",
        device="cpu",
        visualize=False,
        infer_onnx=False,
        num_parallel_batches=1,
    )
    print("recognizer ready", flush=True)

    class Enc(torch.nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder

        def forward(self, images):
            return self.encoder(images)

    wrap = Enc(rec.model.encoder).eval()
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".raw"
    dummy = torch.randn(1, 3, 32, 800)
    kwargs = dict(
        opset_version=18,
        input_names=["input"],
        output_names=["memory"],
        dynamic_axes={"input": {0: "batch"}, "memory": {0: "batch"}},
    )
    try:
        torch.onnx.export(wrap, dummy, tmp, dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(wrap, dummy, tmp, **kwargs)

    del rec, wrap
    gc.collect()
    model = onnx.load(tmp, load_external_data=True)
    onnx.save(model, out)
    print(f"wrote {out} {os.path.getsize(out)}", flush=True)
    for p in (tmp, tmp + ".data"):
        if os.path.isfile(p):
            os.remove(p)


if __name__ == "__main__":
    main()
