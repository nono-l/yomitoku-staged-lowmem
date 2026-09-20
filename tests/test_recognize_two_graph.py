#!/usr/bin/env python3
"""T4 順序復元後。本文が残り、行ずれは端の短い数字 1 件まで。既定切り替えではない。"""

from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(ROOT, "tests", "fixtures", "ocr_two_graph_settei21.json")


def main() -> None:
    with open(FIXTURE, encoding="utf-8") as f:
        data = json.load(f)
    a = data["torch800"]
    b = data["two_graph"]
    if len(a) != len(b):
        raise SystemExit("length differs")
    blob = "".join(b)
    missing = [s for s in data["must_remain"] if s not in blob]
    if missing:
        raise SystemExit("本文が欠けた: " + ", ".join(missing))
    diffs = [(i, x, y) for i, (x, y) in enumerate(zip(a, b)) if x != y]
    if len(diffs) > 1:
        raise SystemExit(f"too many diffs: {diffs}")
    if diffs:
        i, x, y = diffs[0]
        if not (x.replace(".", "").replace("-", "").isdigit() and y.replace(".", "").replace("-", "").isdigit()):
            raise SystemExit(f"non-digit mismatch at {i}: {x!r} vs {y!r}")
    print("ok remain", ", ".join(data["must_remain"]), "diffs", len(diffs))


if __name__ == "__main__":
    main()
