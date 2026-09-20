#!/usr/bin/env python3
"""cvsurf の幾何の往復。モデルを載せない。公式 cv2 が無い環境でも自前だけ見る。"""

from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cvsurf import cv2 as surf
from cvsurf.consts import CHAIN_APPROX_SIMPLE, INTER_LINEAR, RETR_LIST


def test_homography_roundtrip():
    src = np.float32([[0, 0], [40, 0], [40, 20], [0, 20]])
    dst = np.float32([[2, 3], [38, 1], [41, 22], [1, 19]])
    m = surf.getPerspectiveTransform(src, dst)
    ones = np.ones((4, 1), dtype=np.float64)
    hp = np.hstack([src, ones])
    got = (m @ hp.T).T
    got = got[:, :2] / got[:, 2:3]
    if np.max(np.abs(got - dst)) > 1e-6:
        raise SystemExit("homography roundtrip")
    print("ok homography")


def test_fill_poly_closed():
    mask = np.zeros((20, 20), dtype=np.uint8)
    pts = [np.array([[[2, 2]], [[17, 2]], [[17, 17]], [[2, 17]]], dtype=np.int32)]
    surf.fillPoly(mask, pts, 1)
    if mask[10, 10] != 1 or mask[0, 0] != 0:
        raise SystemExit("fillPoly")
    print("ok fillPoly")


def test_resize_shape():
    img = np.arange(60, dtype=np.uint8).reshape(6, 10)
    out = surf.resize(img, (5, 3), interpolation=INTER_LINEAR)
    if out.shape != (3, 5):
        raise SystemExit(f"resize shape {out.shape}")
    print("ok resize shape")


def test_min_area_rect_axis():
    pts = np.array([[0, 0], [10, 0], [10, 4], [0, 4]], dtype=np.float32)
    (cx, cy), (bw, bh), _ang = surf.minAreaRect(pts)
    if abs(cx - 5) > 0.2 or abs(cy - 2) > 0.2:
        raise SystemExit(f"minAreaRect center {cx},{cy}")
    sides = sorted([bw, bh])
    if abs(sides[0] - 4) > 0.3 or abs(sides[1] - 10) > 0.3:
        raise SystemExit(f"minAreaRect size {bw},{bh}")
    print("ok minAreaRect")


def test_find_contours_square():
    img = np.zeros((20, 20), dtype=np.uint8)
    img[5:15, 5:15] = 255
    cnts, _ = surf.findContours(img, RETR_LIST, CHAIN_APPROX_SIMPLE)
    if len(cnts) < 1:
        raise SystemExit("no contour")
    pts = cnts[0].reshape(-1, 2)
    if pts[:, 0].min() > 5 or pts[:, 0].max() < 14:
        raise SystemExit(f"contour xs {pts}")
    print("ok findContours")


def main():
    test_homography_roundtrip()
    test_fill_poly_closed()
    test_resize_shape()
    test_min_area_rect_axis()
    test_find_contours_square()
    print("ok cvsurf geom")


if __name__ == "__main__":
    main()
