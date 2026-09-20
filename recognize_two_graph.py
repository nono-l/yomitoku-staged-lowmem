#!/usr/bin/env python3
"""二グラフ認識。比較用であり、既定の入口ではない。

encoder と decoder 1 ステップを別置き ONNX で回し、検出順へ戻す。
幅は既定 rec と同じ dynamic_width。バッチ復元を忘れると行が入れ替わる。
"""

from __future__ import annotations

import argparse
import gc
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = os.path.dirname(os.path.abspath(__file__))
ENC = os.path.join(ROOT, "weights", "pinned", "recognizer", "encoder_dynw.onnx")
DEC = os.path.join(ROOT, "weights", "pinned", "recognizer", "decoder_step_dynw.onnx")


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
                "pos_query": pos_queries[:, i:j].numpy(),
                "query_mask": mask[i:j, :j],
            },
        )[0]
        steps.append(logits)
        if j < num_steps:
            nxt = int(logits[0, 0].argmax())
            tgt[0, j] = nxt
            if nxt == eos:
                break
    return torch.tensor(np.concatenate(steps, axis=1)).softmax(-1)


def main() -> None:
    parser = argparse.ArgumentParser(description="比較用の二グラフ認識。既定ではない")
    parser.add_argument("image")
    parser.add_argument("--points", default="results/points.json")
    parser.add_argument("-o", "--out", default="results/ocr_two_graph.json")
    parser.add_argument("--encoder", default=ENC)
    parser.add_argument("--decoder", default=DEC)
    args = parser.parse_args()

    import cv2
    import numpy as np
    import onnxruntime as ort
    from yomitoku.text_recognizer import TextRecognizer

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

    rec = TextRecognizer(
        model_name="parseq-tiny-dynw-v5",
        device="cpu",
        visualize=False,
        infer_onnx=False,
        num_parallel_batches=1,
    )
    m = rec.model
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    enc = ort.InferenceSession(args.encoder, sess_options=so, providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession(args.decoder, sess_options=so, providers=["CPUExecutionProvider"])

    loader, pts, dataset, order = rec.preprocess(img, points)
    sorted_points = [pts[i] for i in order] if order is not None else pts
    pos_queries = m.pos_queries.detach().cpu()
    bos, pad, eos = m.tokenizer.bos_id, m.tokenizer.pad_id, m.tokenizer.eos_id
    num_steps = m.max_label_length + 1

    raw_pred = []
    raw_score = []
    offset = 0
    for data in loader:
        arr = data.numpy()
        for k in range(arr.shape[0]):
            p = decode_one(
                enc, dec, pos_queries, bos, pad, eos, num_steps, arr[k : k + 1]
            )
            pred, score, direction = rec.postprocess(p, [sorted_points[offset]])
            raw_pred.extend(pred)
            raw_score.extend(float(s) for s in score)
            offset += 1
        print(f"offset {offset}", flush=True)

    if order is not None:
        inv = np.argsort(order)
        contents = [raw_pred[i] for i in inv]
        scores = [raw_score[i] for i in inv]
    else:
        contents, scores = raw_pred, raw_score

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
    del rec
    gc.collect()


if __name__ == "__main__":
    main()
