#!/usr/bin/env python3
"""parts を結合して weights/pinned/ へ戻す。

無い・ハッシュが違うと失敗する。ネットへ行かない。
既定は runtime graphs。origin safetensors は --origin。
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "weights"))
from verify_weights import join_item, load_manifest, runtime_graphs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="parts を結合して pinned へ戻す")
    parser.add_argument(
        "--origin",
        action="store_true",
        help="safetensors も結合する。比較用",
    )
    args = parser.parse_args()
    man = load_manifest(ROOT)
    items = list(runtime_graphs(man))
    if args.origin:
        items += list(man.get("files", []))
    for item in items:
        if not item.get("parts"):
            print("skip", item["id"], "(no parts)", flush=True)
            continue
        dest = os.path.join(ROOT, item["path"])
        existed = os.path.isfile(dest)
        join_item(ROOT, item)
        print(("have" if existed else "joined"), item["path"], flush=True)
    print("ok join", "origin" if args.origin else "runtime")


if __name__ == "__main__":
    main()
