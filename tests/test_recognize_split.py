#!/usr/bin/env python3
"""認識の encoder と decoder が別入口であること。モデルを載せない。"""

from __future__ import annotations

import ast
import os

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


def main() -> None:
    enc = _imports(os.path.join(ROOT, "stage_recognize_encode.py"))
    dec = _imports(os.path.join(ROOT, "stage_recognize_decode.py"))
    gate = _imports(os.path.join(ROOT, "stage_recognize_onnx.py"))
    if "onnxruntime" in gate:
        raise SystemExit("gate must not import onnxruntime")
    if "subprocess" not in gate:
        raise SystemExit("gate must spawn processes")
    if "onnxruntime" not in enc or "onnxruntime" not in dec:
        raise SystemExit("encode/decode should name onnxruntime")
    src_enc = open(os.path.join(ROOT, "stage_recognize_encode.py"), encoding="utf-8").read()
    src_dec = open(os.path.join(ROOT, "stage_recognize_decode.py"), encoding="utf-8").read()
    if "decoder_step" in src_enc:
        raise SystemExit("encode mentions decoder graph")
    if "encoder_dynw" in src_dec:
        raise SystemExit("decode mentions encoder graph")
    print("ok recognize split imports")


if __name__ == "__main__":
    main()
