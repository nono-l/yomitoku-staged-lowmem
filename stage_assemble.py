#!/usr/bin/env python3
"""ocr_raw/v0 を読み順の本文に戻す。

重みを載せない。検出器と認識器が残っているプロセスで呼ぶと、
また同時展開に戻るので、別プロセス・別段にする。
店名や項目名は付けない。残すのは読み順と隔離だけ。
"""

from __future__ import annotations

import argparse
import json
import os
import re

SCHEMA = "document_ocr/v1"
ROW_Y = 16
NOISE_SCORE = 0.25
NOISE_SHORT = re.compile(r"^[\d\W_]{1,2}$")


def box_key(points: list) -> tuple:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (int(min(ys) // ROW_Y), min(xs), min(ys))


def is_noise(word: dict) -> bool:
    text = (word.get("content") or "").strip()
    if not text:
        return True
    score = word.get("rec_score")
    if score is not None and float(score) < NOISE_SCORE:
        return True
    if NOISE_SHORT.match(text):
        return True
    return False


def assemble(words: list, source: str | None = None) -> dict:
    indexed = list(enumerate(words))
    indexed.sort(key=lambda pair: box_key(pair[1]["points"]))

    body = []
    noise = []
    order = 0
    for _, word in indexed:
        if is_noise(word):
            noise.append(word)
            continue
        item = dict(word)
        item["order"] = order
        body.append(item)
        order += 1

    return {
        "schema": SCHEMA,
        "source": source,
        "text_in_order": [w["content"] for w in body],
        "noise": noise,
        "words": body,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="v0 の箱列を読み順の本文に戻す")
    parser.add_argument("ocr_json", help="stage_recognize_onnx.py が出した v0 JSON")
    parser.add_argument(
        "-o",
        "--out",
        default="results/document.json",
        help="v1 JSON（既定: results/document.json）",
    )
    args = parser.parse_args()

    with open(args.ocr_json, encoding="utf-8") as f:
        words = json.load(f)
    if not isinstance(words, list):
        raise SystemExit("ocr_raw/v0 は配列である")

    doc = assemble(words, source=os.path.basename(args.ocr_json))

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)

    print(f"schema {doc['schema']} body {len(doc['words'])} noise {len(doc['noise'])}", flush=True)
    print("--- text_in_order ---", flush=True)
    for line in doc["text_in_order"]:
        print(line, flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
