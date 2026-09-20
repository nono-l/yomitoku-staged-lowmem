"""輪郭塩。スコア計算のマスクだけに使う。アンチエイリアスはしない。"""

from __future__ import annotations

import numpy as np


def fillPoly(img, pts, color):
    img = np.asarray(img)
    if img.ndim != 2:
        raise ValueError("fillPoly expects 2d mask")
    for poly in pts:
        poly = np.asarray(poly, dtype=np.int32).reshape(-1, 2)
        _fill_one(img, poly, int(color))
    return img


def _fill_one(img, poly, color):
    if len(poly) < 3:
        return
    h, w = img.shape
    ys = poly[:, 1]
    ymin = max(int(ys.min()), 0)
    ymax = min(int(ys.max()), h - 1)
    n = len(poly)
    for y in range(ymin, ymax + 1):
        xs = []
        for i in range(n):
            x0, y0 = poly[i]
            x1, y1 = poly[(i + 1) % n]
            if y0 == y1:
                continue
            if y0 > y1:
                x0, y0, x1, y1 = x1, y1, x0, y0
            if y < y0 or y >= y1:
                continue
            t = (y - y0) / float(y1 - y0)
            xs.append(x0 + t * (x1 - x0))
        if len(xs) < 2:
            continue
        xs.sort()
        for a, b in zip(xs[0::2], xs[1::2]):
            xa = max(int(np.ceil(min(a, b))), 0)
            xb = min(int(np.floor(max(a, b))), w - 1)
            if xa <= xb:
                img[y, xa : xb + 1] = color
