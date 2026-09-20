#!/usr/bin/env python3
"""デコーダ 1 ステップを memory 長さ動的で書く。AR 展開はしない。既定ではない。"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out", default="weights/pinned/recognizer/decoder_step_dynw.onnx")
    args = parser.parse_args()
    import torch, onnx
    from torch.export import Dim
    from yomitoku.text_recognizer import TextRecognizer

    torch.set_num_threads(1)
    rec = TextRecognizer(
        model_name="parseq-tiny-dynw-v5", device="cpu", visualize=False,
        infer_onnx=False, num_parallel_batches=1,
    )
    m = rec.model.eval()
    with torch.no_grad():
        mem = m.encoder(torch.randn(1, 3, 32, 160)).detach().clone()

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
    pos_query = m.pos_queries[:, L - 1 : L].contiguous()
    query_mask = torch.triu(torch.ones(L, L, dtype=torch.bool), 1)[L - 1 : L, :L]
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".raw"
    seq = Dim("seq", min=32, max=1024)
    ln = Dim("len", min=2, max=101)
    torch.onnx.export(
        step, (mem, tgt, pos_query, query_mask), tmp,
        dynamo=True, opset_version=18,
        input_names=["memory", "tgt", "pos_query", "query_mask"],
        output_names=["logits"],
        dynamic_shapes={
            "memory": {1: seq},
            "tgt": {1: ln},
            "pos_query": {},
            "query_mask": {1: ln},
        },
    )
    model = onnx.load(tmp, load_external_data=True)
    onnx.save(model, out)
    print(f"wrote {out} {os.path.getsize(out)}", flush=True)
    for p in (tmp, tmp + ".data"):
        if os.path.isfile(p):
            os.remove(p)


if __name__ == "__main__":
    main()
