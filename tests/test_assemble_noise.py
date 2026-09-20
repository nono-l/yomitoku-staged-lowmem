#!/usr/bin/env python3
"""隔離の境界 0.25 を固定する。モデルを載せない。"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from stage_assemble import NOISE_SCORE, assemble

QUAD = [[0, 0], [100, 0], [100, 20], [0, 20]]


def word(text, score):
    return {"content": text, "rec_score": score, "points": QUAD}


def main() -> None:
    if abs(NOISE_SCORE - 0.25) > 1e-12:
        raise SystemExit(f"NOISE_SCORE moved {NOISE_SCORE}")
    long_title = "オリジナル設定・午前だけのティーアトリエ制服"
    doc = assemble(
        [
            word(long_title, 0.24),
            word("はかりバッジ", 0.25),
            word("12", 0.99),
        ]
    )
    body = "".join(doc["text_in_order"])
    noise = "".join(w["content"] for w in doc["noise"])
    if "ティーアトリエ" in body:
        raise SystemExit("0.24 title must stay noise")
    if "ティーアトリエ" not in noise:
        raise SystemExit("0.24 title missing from noise")
    if "はかりバッジ" not in body:
        raise SystemExit("0.25 text must stay body")
    if "12" in body:
        raise SystemExit("short digits must stay noise")
    print("ok isolate 0.24 / keep 0.25 / short digits")


if __name__ == "__main__":
    main()
