#!/usr/bin/env python3
"""別置き重みのハッシュを見る。

無い・違うなら失敗する。ここが落ちなければ、実行は既知のファイルを指している。
ネットワークへ救済しに行かない。
"""

from __future__ import annotations

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


def verify(root: str = ROOT) -> list:
    man_path = os.path.join(root, "weights", "manifest.json")
    with open(man_path, encoding="utf-8") as f:
        man = json.load(f)
    errors = []
    for item in man["files"]:
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


def main() -> None:
    errors = verify(ROOT)
    if errors:
        print("weights pin failed:", file=sys.stderr)
        for e in errors:
            print(" ", e, file=sys.stderr)
        raise SystemExit(1)
    print("ok weights pin")


if __name__ == "__main__":
    main()
