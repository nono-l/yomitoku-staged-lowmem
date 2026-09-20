#!/usr/bin/env python3
"""認識を ONNX 二グラフで行う。既定の入口。

TextRecognizer は作らない。Dataset / Tokenizer も使わない。
作物と字表は rec/。pos_queries は decoder ONNX の中。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

ENC = os.path.join(ROOT, "weights", "pinned", "recognizer", "encoder_dynw.onnx")
DEC = os.path.join(ROOT, "weights", "pinned", "recognizer", "decoder_step_dynw.onnx")


def load_pos_queries(path):
    """decoder ONNX の initializer から取る。onnx パッケージは使わない。"""
    import numpy as np

    data = open(path, "rb").read()
    key = b"B\x0bpos_queriesJ"
    j = data.find(key)
    if j < 0:
        raise SystemExit(f"pos_queries missing in {path}")
    i = j + len(key)
    n = 0
    shift = 0
    while True:
        by = data[i]
        i += 1
        n |= (by & 0x7F) << shift
        if by < 0x80:
            break
        shift += 7
    arr = np.frombuffer(data[i : i + n], dtype=np.float32)
    if arr.size != 1 * 101 * 192:
        raise SystemExit(f"pos_queries size {arr.size}")
    return arr.reshape(1, 101, 192)


def decode_one(enc, dec, pos_queries, bos, pad, eos, num_steps, image_1chw):
    import numpy as np

    memory = enc.run(["memory"], {"input": image_1chw})[0]
    tgt = np.full((1, num_steps), pad, dtype=np.int64)
    tgt[:, 0] = bos
    mask = np.triu(np.ones((num_steps, num_steps), dtype=bool), 1)
    steps = []
    for i in range(num_steps):
        j = i + 1
        logits = dec.run(
            ["logits"],
            {
                "memory": memory,
                "tgt": tgt[:, :j],
                "pos_query": pos_queries[:, i:j],
                "query_mask": mask[i:j, :j],
            },
        )[0]
        steps.append(logits)
        if j < num_steps:
            nxt = int(logits[0, 0].argmax())
            tgt[0, j] = nxt
            if nxt == eos:
                break
    p = np.concatenate(steps, axis=1)
    p = np.exp(p - p.max(axis=-1, keepdims=True))
    p = p / p.sum(axis=-1, keepdims=True)
    return p[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="ONNX 認識。既定")
    parser.add_argument("image")
    parser.add_argument("--points", default="results/points.json")
    parser.add_argument("-o", "--out", default="results/ocr.json")
    parser.add_argument("--encoder", default=ENC)
    parser.add_argument("--decoder", default=DEC)
    args = parser.parse_args()

    import onnxruntime as ort
    from cvsurf import cv2
    from rec.crop import crop_quad
    from rec.decode import GreedyTokenizer, load_charset, load_replace_table

    for p in (args.encoder, args.decoder):
        if not os.path.isfile(p):
            raise SystemExit(f"missing {p}")

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"failed to read image: {args.image}")
    with open(args.points, encoding="utf-8") as f:
        payload = json.load(f)
    points = payload["points"]
    print(f"img {img.shape} n_boxes {len(points)}", flush=True)

    tokenizer = GreedyTokenizer(load_charset())
    table = load_replace_table()
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    enc = ort.InferenceSession(args.encoder, sess_options=so, providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession(args.decoder, sess_options=so, providers=["CPUExecutionProvider"])
    pos_queries = load_pos_queries(args.decoder)
    bos, pad, eos = tokenizer.bos_id, tokenizer.pad_id, tokenizer.eos_id
    num_steps = 101

    contents = []
    scores = []
    kept_points = []
    for i, quad in enumerate(points):
        crop = crop_quad(img, quad)
        if crop is None:
            contents.append("")
            scores.append(0.0)
            kept_points.append(quad)
            continue
        p = decode_one(enc, dec, pos_queries, bos, pad, eos, num_steps, crop)
        text, score = tokenizer.decode_one(p)
        if table:
            text = text.translate(table)
        contents.append(text)
        scores.append(score)
        kept_points.append(quad)
        if (i + 1) % 10 == 0 or i + 1 == len(points):
            print(f"offset {i + 1}", flush=True)

    words = [
        {"content": t, "rec_score": sc, "points": pt}
        for t, sc, pt in zip(contents, scores, kept_points)
    ]
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
