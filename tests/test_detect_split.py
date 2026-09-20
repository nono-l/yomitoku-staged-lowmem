#!/usr/bin/env python3
"""推論と箱出しが別入口であること。モデルを載せない。"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _imports(path: str) -> set[str]:
    tree = ast.parse(open(path, encoding="utf-8").read())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_entry_imports() -> None:
    infer = _imports(os.path.join(ROOT, "stage_detect_infer.py"))
    boxes = _imports(os.path.join(ROOT, "stage_detect_boxes.py"))
    gate = _imports(os.path.join(ROOT, "stage_detect.py"))
    if "onnxruntime" in boxes or "onnxruntime" in gate:
        raise SystemExit("boxes/gate must not import onnxruntime at top")
    if "det" in gate:
        raise SystemExit("gate must not import det")
    if "subprocess" not in gate:
        raise SystemExit("gate must spawn processes")
    if "onnxruntime" not in infer:
        raise SystemExit("infer should name onnxruntime")
    print("ok split imports")


def test_boxes_from_fake_pred() -> None:
    pred = np.zeros((1, 1, 64, 64), dtype=np.float16)
    pred[0, 0, 20:28, 8:50] = 1
    with tempfile.TemporaryDirectory() as td:
        pack = os.path.join(td, "pred.npz")
        out = os.path.join(td, "points.json")
        np.savez_compressed(pack, pred=pred, dest_h=np.int32(64), dest_w=np.int32(64))
        subprocess.check_call(
            [sys.executable, os.path.join(ROOT, "stage_detect_boxes.py"), pack, "-o", out]
        )
        data = json.load(open(out, encoding="utf-8"))
        if len(data["points"]) < 1:
            raise SystemExit("fake pred produced no box")
    print("ok boxes from fake pred")


def main() -> None:
    test_entry_imports()
    test_boxes_from_fake_pred()
    print("ok detect split")


if __name__ == "__main__":
    main()
