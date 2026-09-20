#!/usr/bin/env python3
"""認識デコーダの 1 ステップを ONNX に書く。

AR を 100 歩展開した convert_onnx は使わない。Python が次手を回す。
トレースで長さを固定すると次手で Reshape が壊れるので dynamo + 動的 len で書く。
一時のバッチは 1。既定認識はまだ torch tiny。
"""

from __future__ import annotations

import argparse
import gc
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


def main() -> None:
    parser = argparse.ArgumentParser(description="認識デコーダ 1 ステップを ONNX に書く")
    parser.add_argument(
        "-o",
        "--out",
        default="weights/pinned/recognizer/decoder_step.onnx",
    )
    args = parser.parse_args()

    import torch
    import onnx
    from torch.export import Dim
    from yomitoku.text_recognizer import TextRecognizer

    torch.set_num_threads(1)
    rec = TextRecognizer(
        model_name="parseq-tiny-dynw-v5",
        device="cpu",
        visualize=False,
        infer_onnx=False,
        num_parallel_batches=1,
    )
    m = rec.model.eval()
    print("recognizer ready", flush=True)
    with torch.no_grad():
        mem = m.encode(torch.randn(1, 3, 32, 800)).detach().clone()

    class Step(torch.nn.Module):
        def __init__(self, parseq):
            super().__init__()
            self.text_embed = parseq.text_embed
            self.pos_queries = parseq.pos_queries
            self.decoder = parseq.decoder
            self.head = parseq.head
            self.dropout = parseq.dropout

        def forward(self, memory, tgt, pos_query, query_mask):
            N, L = tgt.shape
            null_ctx = self.text_embed(tgt[:, :1])
            rest = self.pos_queries[:, : L - 1] + self.text_embed(tgt[:, 1:])
            tgt_emb = self.dropout(torch.cat([null_ctx, rest], dim=1))
            tgt_query = self.dropout(pos_query)
            out = self.decoder(tgt_query, tgt_emb, memory, query_mask, None, None)
            return self.head(out)

    step = Step(m).eval()
    L = 8
    tgt = torch.zeros(1, L, dtype=torch.long)
    tgt[:, 0] = m.tokenizer.bos_id
    pos_query = m.pos_queries[:, L - 1 : L].expand(1, -1, -1).contiguous()
    query_mask = torch.triu(torch.ones(L, L, dtype=torch.bool), 1)[L - 1 : L, :L]

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".raw"
    b = Dim("batch", min=1, max=1)
    ln = Dim("len", min=2, max=101)
    torch.onnx.export(
        step,
        (mem, tgt, pos_query, query_mask),
        tmp,
        dynamo=True,
        opset_version=18,
        input_names=["memory", "tgt", "pos_query", "query_mask"],
        output_names=["logits"],
        dynamic_shapes={
            "memory": {0: b},
            "tgt": {0: b, 1: ln},
            "pos_query": {0: b},
            "query_mask": {1: ln},
        },
    )
    model = onnx.load(tmp, load_external_data=True)
    onnx.save(model, out)
    print(f"wrote {out} {os.path.getsize(out)}", flush=True)
    for p in (tmp, tmp + ".data"):
        if os.path.isfile(p):
            os.remove(p)
    del rec, step
    gc.collect()


if __name__ == "__main__":
    main()
