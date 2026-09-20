#!/usr/bin/env python3
"""既定経路が yomitoku を import しない。モデルを載せない。"""

from __future__ import annotations

import ast
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT = [
    "stage_detect.py",
    "stage_recognize_onnx.py",
    "stage_assemble.py",
    "run_staged.sh",
    "det/preprocess.py",
    "det/postprocess.py",
    "rec/crop.py",
    "rec/decode.py",
    "weights/verify_weights.py",
]


def imports_of(path: str) -> list[str]:
    if path.endswith(".sh"):
        return []
    tree = ast.parse(open(os.path.join(ROOT, path), encoding="utf-8").read(), filename=path)
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module.split(".")[0])
    return names


def main() -> None:
    bad = []
    for rel in DEFAULT:
        names = imports_of(rel)
        if "yomitoku" in names or "torch" in names:
            bad.append(f"{rel}: {names}")
        text = open(os.path.join(ROOT, rel), encoding="utf-8").read()
        if rel.endswith(".sh"):
            if "stage_recognize.py" in text and "stage_recognize_onnx.py" not in text:
                bad.append(f"{rel} still calls torch rec")
            if "yomitoku" in text:
                bad.append(f"{rel} mentions yomitoku")
    if bad:
        raise SystemExit("default path still depends:\n" + "\n".join(bad))
    print("ok default path has no yomitoku/torch import")


if __name__ == "__main__":
    main()
