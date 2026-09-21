#!/usr/bin/env python3
"""箱の目録から空いた区画だけ出す。検出器も認識器も載せない。

二度目の検出は別プロセス。この段は矩形の JSON だけ書く。
run_staged.sh の既定には入れない。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description="既存の箱を避けたタイルを出す")
    parser.add_argument("points", help="stage_detect.py の points.json")
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--tile", type=int, default=320)
    parser.add_argument("--image", default="", help="余白を落とすとき渡す")
    parser.add_argument("--min-ink", type=float, default=-1.0)
    parser.add_argument("-o", "--out", default="results/tiles.json")
    args = parser.parse_args()

    from det.tiles import empty_tiles

    with open(args.points, encoding="utf-8") as f:
        pack = json.load(f)
    if not isinstance(pack, dict):
        raise SystemExit("points.json はオブジェクトである")
    quads = pack.get("points") or []
    img = None
    min_ink = 0.0
    if args.image:
        from cvsurf import cv2

        img = cv2.imread(args.image)
        if img is None:
            raise SystemExit(f"failed to read image: {args.image}")
        min_ink = 0.02 if args.min_ink < 0 else args.min_ink
    elif args.min_ink > 0:
        raise SystemExit("min-ink には image が要る")
    tiles = empty_tiles(
        args.width,
        args.height,
        quads,
        tile=args.tile,
        img=img,
        min_ink=min_ink,
    )
    doc = {
        "schema": "detect_tiles/v0",
        "width": args.width,
        "height": args.height,
        "tile": args.tile,
        "min_ink": min_ink,
        "tiles": tiles,
    }
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    print(f"schema detect_tiles/v0 tiles {len(tiles)}", flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
