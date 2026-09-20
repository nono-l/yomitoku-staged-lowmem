#!/usr/bin/env python3
"""既定経路が使ってよい cv2 名を凍結する。モデルを載せない。

imread / resize / 輪郭 / warp は exact が死ぬので残す。
rotate と mean は numpy で足りた。
"""

from __future__ import annotations

import ast
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_PREFIX = ("comparison/", "tests/", "weights/")
ALLOWED = {
    "CHAIN_APPROX_SIMPLE",
    "INTER_AREA",
    "INTER_LINEAR",
    "RETR_LIST",
    "fillPoly",
    "findContours",
    "getPerspectiveTransform",
    "imread",
    "minAreaRect",
    "resize",
    "warpPerspective",
}


def py_files():
    for dirpath, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", "weights"}]
        for name in files:
            if name.endswith(".py"):
                yield os.path.relpath(os.path.join(dirpath, name), ROOT)


def cv2_attrs(rel: str) -> set[str]:
    tree = ast.parse(open(os.path.join(ROOT, rel), encoding="utf-8").read(), filename=rel)
    names = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "cv2"
        ):
            names.add(node.attr)
    return names


def main() -> None:
    found = set()
    extra = []
    for rel in sorted(py_files()):
        if rel.startswith(SKIP_PREFIX):
            continue
        names = cv2_attrs(rel)
        found |= names
        bad = names - ALLOWED
        if bad:
            extra.append(f"{rel}: {sorted(bad)}")
    if extra:
        raise SystemExit("cv2 names outside allowlist:\n" + "\n".join(extra))
    missing = ALLOWED - found
    if missing:
        raise SystemExit("allowlist stale, unused: " + ", ".join(sorted(missing)))
    print("ok cv2 surface", ", ".join(sorted(found)))


if __name__ == "__main__":
    main()
