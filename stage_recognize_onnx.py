#!/usr/bin/env python3
"""認識を ONNX 二グラフで行う。比較用。既定の入口ではない。

TextRecognizer は作らない。36MB の torch 重みを載せない。
作物と字表は yomitoku の Dataset / Tokenizer だけ使う。
pos_queries は decoder ONNX の中にある。
"""

from __future__ import annotations

import argparse
import json
import os
import unicodedata

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = os.path.dirname(os.path.abspath(__file__))
ENC = os.path.join(ROOT, "weights", "pinned", "recognizer", "encoder_dynw.onnx")
DEC = os.path.join(ROOT, "weights", "pinned", "recognizer", "decoder_step_dynw.onnx")


def load_pos_queries(path):
    from onnx import numpy_helper
    import onnx

    model = onnx.load(path)
    for init in model.graph.initializer:
        if init.name == "pos_queries":
            return numpy_helper.to_array(init)
    raise SystemExit(f"pos_queries missing in {path}")


def decode_one(enc, dec, pos_queries, bos, pad, eos, num_steps, image_1chw):
    import numpy as np
    import torch

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
    return torch.tensor(p)


def main() -> None:
    parser = argparse.ArgumentParser(description="比較用 ONNX 認識。既定ではない")
    parser.add_argument("image")
    parser.add_argument("--points", default="results/points.json")
    parser.add_argument("-o", "--out", default="results/ocr_onnx.json")
    parser.add_argument("--encoder", default=ENC)
    parser.add_argument("--decoder", default=DEC)
    args = parser.parse_args()

    import cv2
    import onnxruntime as ort
    from yomitoku.configs.cfg_text_recognizer_parseq_tiny_dynw_v5 import (
        TextRecognizerPARSeqTinyDynwV5Config,
    )
    from yomitoku.data.dataset import ParseqDataset
    from yomitoku.postprocessor import ParseqTokenizer as Tokenizer
    from yomitoku.utils.misc import load_char_replace_table, load_charset

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

    cfg = TextRecognizerPARSeqTinyDynwV5Config()
    charset = load_charset(cfg.charset)
    tokenizer = Tokenizer(charset)
    table = (
        load_char_replace_table(cfg.char_replace_table)
        if cfg.char_replace_table
        else None
    )
    dataset = ParseqDataset(cfg, img, points, num_workers=1, dynamic_width=True)
    print(f"dataset {len(dataset)}", flush=True)

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    enc = ort.InferenceSession(args.encoder, sess_options=so, providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession(args.decoder, sess_options=so, providers=["CPUExecutionProvider"])
    pos_queries = load_pos_queries(args.decoder)
    bos, pad, eos = tokenizer.bos_id, tokenizer.pad_id, tokenizer.eos_id
    num_steps = cfg.max_label_length + 1

    contents = []
    scores = []
    for i in range(len(dataset)):
        crop = dataset[i].unsqueeze(0).numpy()
        p = decode_one(enc, dec, pos_queries, bos, pad, eos, num_steps, crop)
        pred, score = tokenizer.decode(p)
        if cfg.nfkc_normalize:
            pred = [unicodedata.normalize("NFKC", x) for x in pred]
        if table is not None:
            pred = [x.translate(table) for x in pred]
        contents.append(pred[0])
        scores.append(float(score[0].mean()) if hasattr(score[0], "mean") else float(score[0]))
        if (i + 1) % 10 == 0 or i + 1 == len(dataset):
            print(f"offset {i + 1}", flush=True)

    words = [
        {"content": t, "rec_score": sc, "points": pt}
        for t, sc, pt in zip(contents, scores, points)
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
