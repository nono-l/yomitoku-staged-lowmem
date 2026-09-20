#!/usr/bin/env python3
"""T4。二グラフ AR の活字に本文が残るか。順序は見ない。既定切り替えではない。"""

from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(ROOT, "tests", "fixtures", "ocr_two_graph_settei21.json")


def main() -> None:
    with open(FIXTURE, encoding="utf-8") as f:
        data = json.load(f)
    blob = "".join(data["two_graph"])
    missing = [s for s in data["must_remain"] if s not in blob]
    if missing:
        raise SystemExit("本文が欠けた: " + ", ".join(missing))
    print("ok remain", ", ".join(data["must_remain"]))


if __name__ == "__main__":
    main()
