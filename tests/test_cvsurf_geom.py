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


def test_area_integer_scale():
    from cvsurf.consts import INTER_AREA

    src = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    up = surf.resize(src, (4, 4), interpolation=INTER_AREA)
    expect_up = np.array(
        [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]], dtype=np.uint8
    )
    if not np.array_equal(up, expect_up):
        raise SystemExit(f"area 2x up {up}")
    block = np.array([[10, 10, 20, 20], [10, 10, 20, 20], [30, 30, 40, 40], [30, 30, 40, 40]], dtype=np.uint8)
    down = surf.resize(block, (2, 2), interpolation=INTER_AREA)
    if not np.array_equal(down, src * 10):
        raise SystemExit(f"area 2x down {down}")
    print("ok area integer scale")


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


def test_find_contours_wide():
    img = np.zeros((200, 300), dtype=np.uint8)
    img[40:80, 50:180] = 255
    cnts, _ = surf.findContours(img, RETR_LIST, CHAIN_APPROX_SIMPLE)
    if len(cnts) < 1:
        raise SystemExit("wide image no contour")
    pts = cnts[0].reshape(-1, 2)
    if pts[:, 0].min() > 50 or pts[:, 0].max() < 179:
        raise SystemExit(f"wide contour xs {pts[:, 0].min()} {pts[:, 0].max()}")
    print("ok findContours wide")


def test_warp_keeps_last_row():
    img = np.arange(30, dtype=np.uint8).reshape(3, 10)
    m = np.eye(3, dtype=np.float64)
    out = surf.warpPerspective(img, m, (10, 3))
    if out.shape != (3, 10):
        raise SystemExit(f"warp shape {out.shape}")
    if int(out[2].sum()) == 0:
        raise SystemExit("warp dropped last row")
    if not np.array_equal(out, img):
        raise SystemExit(f"identity warp {out}")
    print("ok warp last row")


def main():
    test_homography_roundtrip()
    test_fill_poly_closed()
    test_resize_shape()
    test_area_integer_scale()
    test_min_area_rect_axis()
    test_find_contours_square()
    test_find_contours_wide()
    test_warp_keeps_last_row()
    print("ok cvsurf geom")


if __name__ == "__main__":
    main()
