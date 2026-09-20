#!/usr/bin/env python3
"""重みなし ONNX 認識の活字。本文が残る。既定切り替えではない。"""

from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(ROOT, "tests", "fixtures", "ocr_onnx_lite_settei21.json")
TORCH = os.path.join(ROOT, "tests", "fixtures", "ocr_two_graph_dynw_settei21.json")


def main() -> None:
    with open(FIXTURE, encoding="utf-8") as f:
        data = json.load(f)
    with open(TORCH, encoding="utf-8") as f:
        torch_side = json.load(f)
    blob = "".join(data["lite"])
    missing = [s for s in data["must_remain"] if s not in blob]
    if missing:
        raise SystemExit("本文が欠けた: " + ", ".join(missing))
    a, b = torch_side["torch_dynw"], data["lite"]
    diffs = [(i, x, y) for i, (x, y) in enumerate(zip(a, b)) if x != y]
    if len(diffs) > 4:
        raise SystemExit(f"too many diffs: {diffs}")
    for i, x, y in diffs:
        for s in data["must_remain"]:
            if s in x or s in y:
                raise SystemExit(f"本文の行が違う: {i} {x!r} {y!r}")
    print("ok remain", ", ".join(data["must_remain"]), "diffs", len(diffs))


if __name__ == "__main__":
    main()
