#!/usr/bin/env python3
"""別置きのハッシュを見る。

既定は実行に使う graphs だけ。safetensors は比較・再 export 用で、
入口の条件にしない。無い・違うなら失敗する。ネットへ救済しに行かない。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _check(root: str, items: list) -> list:
    errors = []
    for item in items:
        path = os.path.join(root, item["path"])
        if not os.path.isfile(path):
            errors.append(f"missing {item['path']}")
            continue
        size = os.path.getsize(path)
        if size != item["bytes"]:
            errors.append(f"size {item['path']} {size} != {item['bytes']}")
            continue
        digest = sha256_file(path)
        if digest != item["sha256"]:
            errors.append(f"hash {item['path']} {digest} != {item['sha256']}")
    return errors


def load_manifest(root: str) -> dict:
    man_path = os.path.join(root, "weights", "manifest.json")
    with open(man_path, encoding="utf-8") as f:
        return json.load(f)


def runtime_graphs(man: dict) -> list:
    items = [g for g in man.get("graphs", []) if g.get("runtime")]
    if not items:
        return []
    return items


def verify(root: str = ROOT, what: str = "runtime") -> list:
    man = load_manifest(root)
    if what == "runtime":
        items = runtime_graphs(man)
        if not items:
            return ["manifest has no runtime graphs"]
        return _check(root, items)
    if what == "origin":
        return _check(root, man.get("files", []))
    raise ValueError(f"unknown what={what}")


def main() -> None:
    parser = argparse.ArgumentParser(description="別置きのハッシュを見る")
    parser.add_argument(
        "--origin",
        action="store_true",
        help="safetensors も見る。比較用。既定の入口では使わない",
    )
    args = parser.parse_args()
    errors = verify(ROOT, what="origin" if args.origin else "runtime")
    if errors:
        print("weights pin failed:", file=sys.stderr)
        for e in errors:
            print(" ", e, file=sys.stderr)
        raise SystemExit(1)
    print("ok weights pin", "origin" if args.origin else "runtime")


if __name__ == "__main__":
    main()
