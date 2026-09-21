#!/usr/bin/env python3
"""空いたタイルを切って、既存の検出を一枚ずつ別プロセスで呼ぶ。

検出器をこのプロセスに載せない。既定の最短辺は変えない。
run_staged.sh の既定には入れない。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description="タイルごとに検出を呼ぶ")
    parser.add_argument("image")
    parser.add_argument("tiles", help="detect_tiles/v0")
    parser.add_argument("-o", "--out", default="results/points_tiles.json")
    parser.add_argument("--detect", default=os.path.join(ROOT, "stage_detect.py"))
    args = parser.parse_args()

    from cvsurf import cv2
    from det.tile_run import crop_bgr, remap_quads, write_bmp

    with open(args.tiles, encoding="utf-8") as f:
        pack = json.load(f)
    if not isinstance(pack, dict) or pack.get("schema") != "detect_tiles/v0":
        raise SystemExit("tiles は detect_tiles/v0")
    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"failed to read image: {args.image}")

    out_abs = os.path.abspath(args.out)
    work = out_abs + ".work"
    os.makedirs(work, exist_ok=True)
    points = []
    scores = []
    py = sys.executable
    for i, tile in enumerate(pack.get("tiles") or []):
        if int(tile["w"]) < 32 or int(tile["h"]) < 32:
            continue
        crop = crop_bgr(img, tile)
        bmp = os.path.join(work, f"tile_{i:03d}.bmp")
        pts = os.path.join(work, f"tile_{i:03d}.json")
        write_bmp(bmp, crop)
        subprocess.check_call([py, args.detect, bmp, "-o", pts])
        with open(pts, encoding="utf-8") as f:
            got = json.load(f)
        quads = remap_quads(got.get("points") or [], int(tile["x"]), int(tile["y"]))
        points.extend(quads)
        scores.extend(got.get("scores") or [0.0] * len(quads))

    dest_dir = os.path.dirname(out_abs)
    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"points": points, "scores": scores}, f)
    print(f"n_boxes {len(points)}", flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
