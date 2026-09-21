#!/usr/bin/env python3
"""試験用。画像全体を一つの箱として書く。重みは載せない。"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cvsurf import cv2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("-o", "--out", required=True)
    args = parser.parse_args()
    img = cv2.imread(args.image)
    h, w = img.shape[:2]
    quad = [[0, 0], [w, 0], [w, h], [0, h]]
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"points": [quad], "scores": [1.0]}, f)


if __name__ == "__main__":
    main()
