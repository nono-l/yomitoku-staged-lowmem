#!/usr/bin/env python3
"""結合後の実体を 40MiB 未満の parts に切る。Git の 50MB 警告を避ける。

parts は git に載せる。結合後は weights/pinned/ で gitignore。
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PART = 40 * 1024 * 1024


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parts_dir_for(joined_path: str) -> str:
    rel = joined_path
    if rel.startswith("weights/pinned/"):
        rel = "weights/parts/" + rel[len("weights/pinned/") :]
    else:
        rel = "weights/parts/" + os.path.basename(joined_path)
    return rel


def split_one(root: str, item: dict) -> list:
    src = os.path.join(root, item["path"])
    if not os.path.isfile(src):
        raise SystemExit(f"missing {item['path']}")
    size = os.path.getsize(src)
    digest = sha256_file(src)
    if size != item["bytes"] or digest != item["sha256"]:
        raise SystemExit(f"refusing to split dirty {item['path']}")
    dest_prefix = parts_dir_for(item["path"])
    os.makedirs(os.path.join(root, os.path.dirname(dest_prefix)), exist_ok=True)
    parts = []
    with open(src, "rb") as f:
        index = 0
        while True:
            blob = f.read(PART)
            if not blob:
                break
            part_path = f"{dest_prefix}.part{index:02d}"
            full = os.path.join(root, part_path)
            with open(full, "wb") as out:
                out.write(blob)
            ph = hashlib.sha256(blob).hexdigest()
            parts.append(
                {
                    "path": part_path,
                    "sha256": ph,
                    "bytes": len(blob),
                    "index": index,
                }
            )
            print(f"  {part_path} {len(blob)}", flush=True)
            index += 1
    item["parts"] = parts
    return parts


def main() -> None:
    man_path = os.path.join(ROOT, "weights", "manifest.json")
    with open(man_path, encoding="utf-8") as f:
        man = json.load(f)
    targets = [g for g in man.get("graphs", []) if g.get("runtime")]
    targets += list(man.get("files", []))
    for item in targets:
        print("split", item["id"], flush=True)
        split_one(ROOT, item)
    man["schema"] = "weights_pin/v1"
    man["part_bytes"] = PART
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print("ok split", flush=True)


if __name__ == "__main__":
    main()
