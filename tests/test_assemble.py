#!/usr/bin/env python3
"""提出前の証。モデルは載せない。凍結した v0 から本文が残るかを見る。"""

from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from stage_assemble import assemble

FIXTURE = os.path.join(ROOT, "tests", "fixtures", "settei21_ocr.json")
EXPECTED = os.path.join(ROOT, "tests", "expected_settei21.json")


def main() -> None:
    with open(FIXTURE, encoding="utf-8") as f:
        words = json.load(f)
    with open(EXPECTED, encoding="utf-8") as f:
        want = json.load(f)

    doc = assemble(words, source="settei21_ocr.json")
    blob = "".join(doc["text_in_order"])

    missing = [s for s in want["must_remain"] if s not in blob]
    if missing:
        raise SystemExit("本文が欠けた: " + ", ".join(missing))

    if doc["schema"] != "document_ocr/v1":
        raise SystemExit("schema が v1 ではない")

    print("ok body", len(doc["words"]), "noise", len(doc["noise"]))
    print("ok remain", ", ".join(want["must_remain"]))


if __name__ == "__main__":
    main()
