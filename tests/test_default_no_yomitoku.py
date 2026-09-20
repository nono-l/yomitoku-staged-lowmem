#!/usr/bin/env python3
"""yomitoku / torch の import は comparison/ だけ。モデルを載せない。"""

from __future__ import annotations

import ast
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALLOWED = ("comparison/",)


def py_files():
    for dirpath, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", "weights/pinned"}]
        for name in files:
            if name.endswith(".py"):
                yield os.path.relpath(os.path.join(dirpath, name), ROOT)


def imports_of(rel: str) -> list[str]:
    tree = ast.parse(open(os.path.join(ROOT, rel), encoding="utf-8").read(), filename=rel)
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module.split(".")[0])
    return names


def main() -> None:
    bad = []
    allowed_hits = []
    for rel in sorted(py_files()):
        names = imports_of(rel)
        banned = [n for n in names if n in ("yomitoku", "torch")]
        if not banned:
            continue
        if rel.startswith(ALLOWED):
            allowed_hits.append(rel)
            continue
        bad.append(f"{rel}: {banned}")
    sh = open(os.path.join(ROOT, "run_staged.sh"), encoding="utf-8").read()
    if "yomitoku" in sh:
        bad.append("run_staged.sh mentions yomitoku")
    if "stage_recognize.py" in sh and "stage_recognize_onnx.py" not in sh:
        bad.append("run_staged.sh still calls torch rec")
    if bad:
        raise SystemExit("yomitoku/torch leaked out of comparison/:\n" + "\n".join(bad))
    if not allowed_hits:
        raise SystemExit("comparison/ has no yomitoku imports; export was lost?")
    print("ok yomitoku/torch only in", ", ".join(allowed_hits))


if __name__ == "__main__":
    main()
