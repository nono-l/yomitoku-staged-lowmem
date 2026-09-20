#!/usr/bin/env python3
"""parts の結合が元に戻る。モデルを載せない。"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "weights"))
from verify_weights import join_item, sha256_file


def main() -> None:
    blob = b"abc" * 1000 + b"END"
    digest = hashlib.sha256(blob).hexdigest()
    with tempfile.TemporaryDirectory() as tmp:
        parts_dir = os.path.join(tmp, "weights", "parts")
        os.makedirs(parts_dir)
        a, b = blob[:1000], blob[1000:]
        p0 = "weights/parts/x.bin.part00"
        p1 = "weights/parts/x.bin.part01"
        open(os.path.join(tmp, p0), "wb").write(a)
        open(os.path.join(tmp, p1), "wb").write(b)
        item = {
            "id": "x",
            "path": "weights/pinned/x.bin",
            "sha256": digest,
            "bytes": len(blob),
            "parts": [
                {"path": p0, "sha256": hashlib.sha256(a).hexdigest(), "bytes": len(a), "index": 0},
                {"path": p1, "sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b), "index": 1},
            ],
        }
        join_item(tmp, item)
        got = open(os.path.join(tmp, item["path"]), "rb").read()
        if got != blob:
            raise SystemExit("join did not restore bytes")
        if sha256_file(os.path.join(tmp, item["path"])) != digest:
            raise SystemExit("join hash")
        print("ok join restores bytes")

    man = json.load(open(os.path.join(ROOT, "weights", "manifest.json"), encoding="utf-8"))
    if man.get("part_bytes", 0) >= 50 * 1024 * 1024:
        raise SystemExit("part_bytes must stay under GitHub 50MB warning")
    n_parts = 0
    for item in list(man.get("graphs", [])) + list(man.get("files", [])):
        for part in item.get("parts") or []:
            if part["bytes"] > 40 * 1024 * 1024:
                raise SystemExit(f"part too large {part['path']}")
            path = os.path.join(ROOT, part["path"])
            if not os.path.isfile(path):
                raise SystemExit(f"missing part {part['path']}")
            size = os.path.getsize(path)
            if size != part["bytes"]:
                raise SystemExit(f"size {part['path']} {size} != {part['bytes']}")
            digest = sha256_file(path)
            if digest != part["sha256"]:
                raise SystemExit(f"hash {part['path']}")
            n_parts += 1
    if n_parts < 1:
        raise SystemExit("manifest has no parts")
    print(f"ok {n_parts} parts on disk match manifest")


if __name__ == "__main__":
    main()
