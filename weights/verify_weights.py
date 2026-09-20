#!/usr/bin/env python3
"""別置きと字表のハッシュを見る。

既定は実行に使う graphs と字表。safetensors は比較・再 export 用で、
入口の条件にしない。無い・違うなら失敗する。ネットへ救済しに行かない。
parts があるのに pinned が無いときは結合してから見る。
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
    return [g for g in man.get("graphs", []) if g.get("runtime")]


def runtime_resources(man: dict) -> list:
    return [r for r in man.get("resources", []) if r.get("runtime")]


def runtime_items(man: dict) -> list:
    return runtime_graphs(man) + runtime_resources(man)


def join_item(root: str, item: dict) -> None:
    dest = os.path.join(root, item["path"])
    parts = sorted(item.get("parts") or [], key=lambda p: p["index"])
    if not parts:
        return
    if os.path.isfile(dest) and os.path.getsize(dest) == item["bytes"]:
        if sha256_file(dest) == item["sha256"]:
            return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".joining"
    with open(tmp, "wb") as out:
        for part in parts:
            src = os.path.join(root, part["path"])
            if not os.path.isfile(src):
                raise FileNotFoundError(part["path"])
            if os.path.getsize(src) != part["bytes"]:
                raise ValueError(f"size {part['path']}")
            if sha256_file(src) != part["sha256"]:
                raise ValueError(f"hash {part['path']}")
            with open(src, "rb") as f:
                while True:
                    chunk = f.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
    os.replace(tmp, dest)
    if os.path.getsize(dest) != item["bytes"] or sha256_file(dest) != item["sha256"]:
        raise ValueError(f"joined {item['path']}")


def join_items(root: str, items: list) -> list:
    errors = []
    for item in items:
        if not item.get("parts"):
            continue
        try:
            join_item(root, item)
        except FileNotFoundError as e:
            errors.append(f"missing {e}")
        except ValueError as e:
            errors.append(str(e))
    return errors


def verify(root: str = ROOT, what: str = "runtime") -> list:
    man = load_manifest(root)
    if what == "runtime":
        graphs = runtime_graphs(man)
        if not graphs:
            return ["manifest has no runtime graphs"]
        errors = join_items(root, graphs)
        if errors:
            return errors
        return _check(root, runtime_items(man))
    if what == "origin":
        files = man.get("files", [])
        errors = join_items(root, files)
        if errors:
            return errors
        return _check(root, files)
    raise ValueError(f"unknown what={what}")


def main() -> None:
    parser = argparse.ArgumentParser(description="別置きと字表のハッシュを見る")
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
