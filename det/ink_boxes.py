"""インク塊から箱を出す。検出ネットは載せない。

DBNet の重みも予測図も使わない。余白から外れた画素の輪郭だけ見る。
既定経路には入れない。細字や接触した行は落とす。
"""

from __future__ import annotations

import math

import numpy as np

from cvsurf import cv2


def _quad(rect) -> list[list[int]]:
    (cx, cy), (w, h), ang = rect
    rad = math.radians(ang)
    c, s = math.cos(rad), math.sin(rad)
    hw, hh = w / 2.0, h / 2.0
    local = ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))
    out = []
    for x, y in local:
        out.append([int(round(cx + x * c - y * s)), int(round(cy + x * s + y * c))])
    return out


def boxes_from_bgr(
    bgr,
    min_area: int = 40,
    max_fill: float = 0.45,
    delta: float = 24.0,
) -> tuple[list[list[list[int]]], list[float]]:
    """BGR から四隅と仮スコア。ネットは走らない。"""
    if bgr is None or bgr.size == 0:
        raise ValueError("image")
    gray = bgr.mean(axis=2).astype(np.float32)
    med = float(np.median(gray))
    ink = (np.abs(gray - med) >= delta).astype(np.uint8) * 255
    contours, _ = cv2.findContours(ink, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape
    img_area = float(h * w)
    quads = []
    scores = []
    for cnt in contours:
        pts = np.asarray(cnt).reshape(-1, 2)
        if len(pts) < 4:
            continue
        rect = cv2.minAreaRect(pts)
        (_, _), (rw, rh), _ = rect
        area = float(rw * rh)
        if area < min_area or area > img_area * max_fill:
            continue
        short, long = sorted((rw, rh))
        if short < 2 or long / max(short, 1.0) > 40:
            continue
        quads.append(_quad(rect))
        scores.append(round(min(area / 4000.0, 1.0), 4))
    return quads, scores
