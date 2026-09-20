#!/usr/bin/env python3
"""認識 encoder を幅動的な ONNX に書く。検出器は載せない。既定認識はまだ torch tiny。

32x800 固定の encoder.onnx とは別物。既定 rec の dynamic_width に合わせる。
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out", default="weights/pinned/recognizer/encoder_dynw.onnx")
    args = parser.parse_args()
    import torch, onnx
    from torch.export import Dim
    from yomitoku.text_recognizer import TextRecognizer

    torch.set_num_threads(1)
    rec = TextRecognizer(
        model_name="parseq-tiny-dynw-v5", device="cpu", visualize=False,
        infer_onnx=False, num_parallel_batches=1,
    )

    class Enc(torch.nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder
        def forward(self, images):
            return self.encoder(images)

    wrap = Enc(rec.model.encoder).eval()
    x = torch.randn(1, 3, 32, 160)
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".raw"
    w = Dim("width", min=32, max=1024)
    torch.onnx.export(
        wrap, x, tmp, dynamo=True, opset_version=18,
        input_names=["input"], output_names=["memory"],
        dynamic_shapes={"images": {3: w}},
    )
    model = onnx.load(tmp, load_external_data=True)
    onnx.save(model, out)
    print(f"wrote {out} {os.path.getsize(out)}", flush=True)
    for p in (tmp, tmp + ".data"):
        if os.path.isfile(p):
            os.remove(p)


if __name__ == "__main__":
    main()
