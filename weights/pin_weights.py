#!/usr/bin/env python3
"""manifest の origin safetensors を書いた revision だけ別置きへ取る。

latest は見ない。既定の graphs は Drive。ここでは取らない。
既にハッシュが一致していれば触らない。
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "weights"))
from verify_weights import sha256_file, verify  # noqa: E402


def download(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    with urllib.request.urlopen(url) as src, open(tmp, "wb") as out:
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    os.replace(tmp, dest)


def main() -> None:
    man_path = os.path.join(ROOT, "weights", "manifest.json")
    with open(man_path, encoding="utf-8") as f:
        man = json.load(f)

    for item in man["files"]:
        dest = os.path.join(ROOT, item["path"])
        if os.path.isfile(dest) and sha256_file(dest) == item["sha256"]:
            print("have", item["path"], flush=True)
            continue
        url = (
            "https://huggingface.co/"
            + item["hf_repo"]
            + "/resolve/"
            + item["revision"]
            + "/"
            + item["remote_name"]
        )
        print("get", url, flush=True)
        download(url, dest)

    errors = verify(ROOT, what="origin")
    if errors:
        print("origin pin failed:", file=sys.stderr)
        for e in errors:
            print(" ", e, file=sys.stderr)
        raise SystemExit(1)
    print("ok origin pinned")


if __name__ == "__main__":
    main()
