#!/usr/bin/env python3
"""offset の mini 箱が凍結されている。モデルを載せない。"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from det.offset import offset_round
from det.postprocess import _mini_boxes

FIX = os.path.join(ROOT, "tests", "fixtures", "offset_cases.json")


def main() -> None:
    with open(FIX, encoding="utf-8") as f:
        data = json.load(f)
    for i, case in enumerate(data["cases"]):
        src = np.array(case["src"], dtype=np.float32)
        expanded = np.array(offset_round(src, case["distance"]), dtype=np.float32).reshape(
            -1, 1, 2
        )
        mini, _ = _mini_boxes(expanded)
        got = np.array(mini, dtype=np.float32).tolist()
        if got != case["mini"]:
            raise SystemExit(f"case {i} mini differs: {got} != {case['mini']}")
    print(f"ok offset cases {len(data['cases'])}")


if __name__ == "__main__":
    main()
