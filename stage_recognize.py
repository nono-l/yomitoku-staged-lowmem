#!/usr/bin/env python3
"""認識だけを載せて終了する。

stage_detect.py の後に、別プロセスで起こす。検出器の重みが残っていると 137 に戻る。
tiny を使うのは、middle がこの箱に載らないため。
"""

from __future__ import annotations

import argparse
import gc
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def main() -> None:
    parser = argparse.ArgumentParser(description="保存した箱を小分割で認識する")
    parser.add_argument("image", help="検出と同じ入力画像")
    parser.add_argument(
        "--points",
        default="results/points.json",
        help="stage_detect.py が書いた JSON",
    )
    parser.add_argument(
        "-o",
        "--out",
        default="results/ocr.json",
        help="語の JSON（既定: results/ocr.json）",
    )
    parser.add_argument("--chunk", type=int, default=6, help="1 回の箱数。増やすとピーク RAM が上がる")
    parser.add_argument(
        "--model",
        default="parseq-tiny-dynw-v5",
        help="認識モデル名（lite 既定）",
    )
    args = parser.parse_args()

    import cv2
    import torch

    torch.set_num_threads(1)

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"failed to read image: {args.image}")
    with open(args.points, encoding="utf-8") as f:
        payload = json.load(f)
    points = payload["points"]
    det_scores = payload.get("scores") or [None] * len(points)
    print(f"img {img.shape} n_boxes {len(points)}", flush=True)

    from yomitoku.text_recognizer import TextRecognizer

    rec = TextRecognizer(
        model_name=args.model,
        device="cpu",
        visualize=False,
        infer_onnx=False,
        num_parallel_batches=1,
    )
    print("recognizer ready", flush=True)

    contents = []
    scores = []
    directions = []
    for i in range(0, len(points), args.chunk):
        chunk = points[i : i + args.chunk]
        print(f"chunk {i}:{i + len(chunk)}", flush=True)
        out, _ = rec(img, chunk, vis=None)
        contents.extend(out.contents)
        scores.extend(float(s) for s in out.scores)
        directions.extend(list(out.directions))
        gc.collect()

    words = []
    for pts, text, sc, direction, det_sc in zip(
        points, contents, scores, directions, det_scores
    ):
        words.append(
            {
                "content": text,
                "rec_score": sc,
                "det_score": det_sc,
                "direction": direction,
                "points": pts,
            }
        )

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(words, f, ensure_ascii=False, indent=2)

    print("--- texts ---", flush=True)
    for w in words:
        print(f"{w['rec_score']:.2f} {w['content']}", flush=True)
    print(f"wrote {args.out}", flush=True)

    del rec
    gc.collect()


if __name__ == "__main__":
    main()
