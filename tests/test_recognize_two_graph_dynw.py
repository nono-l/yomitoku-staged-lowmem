#!/usr/bin/env python3
"""T5。既定 rec と dynw 二グラフ。本文が残り、行ずれは 1 件まで。既定切り替えではない。"""

from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(ROOT, "tests", "fixtures", "ocr_two_graph_dynw_settei21.json")


def main() -> None:
    with open(FIXTURE, encoding="utf-8") as f:
        data = json.load(f)
    a, b = data["torch_dynw"], data["two_graph_dynw"]
    if len(a) != len(b):
        raise SystemExit("length differs")
    blob = "".join(b)
    missing = [s for s in data["must_remain"] if s not in blob]
    if missing:
        raise SystemExit("本文が欠けた: " + ", ".join(missing))
    diffs = [(i, x, y) for i, (x, y) in enumerate(zip(a, b)) if x != y]
    if len(diffs) > 1:
        raise SystemExit(f"too many diffs: {diffs}")
    for i, x, y in diffs:
        for s in data["must_remain"]:
            if s in x or s in y:
                raise SystemExit(f"本文の行が違う: {i} {x!r} {y!r}")
    print("ok remain", ", ".join(data["must_remain"]), "diffs", len(diffs))


if __name__ == "__main__":
    main()
