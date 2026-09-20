#!/usr/bin/env python3
"""T3。凍結した 32x800 torch 認識と encoder-ONNX 混成の活字。モデルは載せない。"""

from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(ROOT, "tests", "fixtures", "ocr_encoder_onnx_settei21.json")
EXPECTED = os.path.join(ROOT, "tests", "expected_settei21.json")


def main() -> None:
    with open(FIXTURE, encoding="utf-8") as f:
        data = json.load(f)
    with open(EXPECTED, encoding="utf-8") as f:
        want = json.load(f)

    if data["torch800"] != data["encoder_onnx_hybrid"]:
        raise SystemExit("encoder ONNX hybrid texts differ from torch800")
    blob = "".join(data["encoder_onnx_hybrid"])
    missing = [s for s in want["must_remain"] if s not in blob]
    if missing:
        raise SystemExit("本文が欠けた: " + ", ".join(missing))
    print("ok exact", len(data["encoder_onnx_hybrid"]), "remain", ", ".join(want["must_remain"]))


if __name__ == "__main__":
    main()
