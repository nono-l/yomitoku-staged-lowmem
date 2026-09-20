#!/usr/bin/env python3
"""memory から活字を出す。encoder は載せない。"""

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

DEC = os.path.join(ROOT, "weights", "pinned", "recognizer", "decoder_step_dynw.onnx")


def main() -> None:
    parser = argparse.ArgumentParser(description="認識の decoder だけ")
    parser.add_argument("memory", help="stage_recognize_encode の npz")
    parser.add_argument("--points", default="results/points.json")
    parser.add_argument("-o", "--out", default="results/ocr.json")
    parser.add_argument("--decoder", default=DEC)
    args = parser.parse_args()

    import numpy as np
    import onnxruntime as ort
    from rec.decode import GreedyTokenizer, load_charset, load_replace_table
    from rec.step import load_pos_queries, logits_from_memory

    if not os.path.isfile(args.decoder):
        raise SystemExit(f"missing {args.decoder}")
    points = json.load(open(args.points, encoding="utf-8"))["points"]
    pack = np.load(args.memory)
    n = int(pack["n"])
    if n != len(points):
        raise SystemExit(f"memory n {n} != points {len(points)}")
    tokenizer = GreedyTokenizer(load_charset())
    table = load_replace_table()
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    dec = ort.InferenceSession(
        args.decoder, sess_options=so, providers=["CPUExecutionProvider"]
    )
    pos_queries = load_pos_queries(args.decoder)
    bos, pad, eos = tokenizer.bos_id, tokenizer.pad_id, tokenizer.eos_id
    valid = pack["valid"]
    words = []
    for i, quad in enumerate(points):
        if int(valid[i]) != 1:
            words.append({"content": "", "rec_score": 0.0, "points": quad})
            continue
        memory = pack[f"m{i}"].astype(np.float32)
        p = logits_from_memory(dec, pos_queries, bos, pad, eos, 101, memory)
        text, score = tokenizer.decode_one(p)
        if table:
            text = text.translate(table)
        words.append({"content": text, "rec_score": score, "points": quad})
        if (i + 1) % 10 == 0 or i + 1 == n:
            print(f"decode {i + 1}", flush=True)
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(words, f, ensure_ascii=False, indent=2)
    for w in words:
        print(f"{w['rec_score']:.2f} {w['content']}", flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
