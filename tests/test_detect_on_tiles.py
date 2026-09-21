#!/usr/bin/env python3
"""タイル切りと座標戻し。モデルを載せない。"""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from det.tile_run import crop_bgr, remap_quads, write_bmp
from cvsurf.io import imread


def _imports(path: str) -> set[str]:
    tree = ast.parse(open(path, encoding="utf-8").read())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def main() -> None:
    gate = _imports(os.path.join(ROOT, "stage_detect_on_tiles.py"))
    if "onnxruntime" in gate:
        raise SystemExit("tile detect gate must not import onnxruntime at top")
    if "subprocess" not in gate:
        raise SystemExit("tile detect must spawn processes")

    img = np.zeros((80, 120, 3), dtype=np.uint8)
    img[10:50, 20:80] = (0, 0, 200)
    crop = crop_bgr(img, {"x": 20, "y": 10, "w": 60, "h": 40})
    if crop.shape != (40, 60, 3):
        raise SystemExit(f"crop shape {crop.shape}")
    if int(crop[0, 0, 2]) != 200:
        raise SystemExit("crop pixel")

    moved = remap_quads([[[1, 2], [3, 2], [3, 4], [1, 4]]], 20, 10)
    if moved[0][0] != [21, 12] or moved[0][2] != [23, 14]:
        raise SystemExit(f"remap {moved}")

    path = os.path.join(ROOT, "tests", "_tmp_tile.bmp")
    write_bmp(path, crop)
    back = imread(path)
    os.remove(path)
    if back.shape != crop.shape:
        raise SystemExit(f"bmp shape {back.shape}")
    if int(back[0, 0, 2]) != 200:
        raise SystemExit("bmp pixel")
    fake = os.path.join(ROOT, "tests", "fake_detect.py")
    img_path = os.path.join(ROOT, "tests", "_tmp_src.bmp")
    write_bmp(img_path, img)
    tiles = {
        "schema": "detect_tiles/v0",
        "width": 120,
        "height": 80,
        "tile": 60,
        "tiles": [{"x": 20, "y": 10, "w": 60, "h": 40, "row": 0, "col": 0}],
    }
    tpath = os.path.join(ROOT, "tests", "_tmp_tiles.json")
    opath = os.path.join(ROOT, "tests", "_tmp_pts.json")
    with open(tpath, "w", encoding="utf-8") as f:
        json.dump(tiles, f)
    subprocess.check_call(
        [
            sys.executable,
            os.path.join(ROOT, "stage_detect_on_tiles.py"),
            img_path,
            tpath,
            "-o",
            opath,
            "--detect",
            fake,
        ]
    )
    got = json.load(open(opath, encoding="utf-8"))
    for p in (img_path, tpath, opath):
        os.remove(p)
    work = opath + ".work"
    if os.path.isdir(work):
        shutil.rmtree(work)
    box = got["points"][0]
    if box[0] != [20, 10] or box[2] != [80, 50]:
        raise SystemExit(f"merged box {box}")
    print("ok tile run remap bmp")


if __name__ == "__main__":
    main()
