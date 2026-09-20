"""単応、歪み補正、回転最小矩形。線形代数は numpy。

minAreaRect の角度は、長辺が x に近いとき 0 に近い。
OpenCV 5 の符号規約と一致するとは限らない。箱の4点は _box_points 側で組む。
"""

from __future__ import annotations

import numpy as np


def getPerspectiveTransform(src, dst):
    src = np.asarray(src, dtype=np.float64).reshape(4, 2)
    dst = np.asarray(dst, dtype=np.float64).reshape(4, 2)
    a = np.zeros((8, 8), dtype=np.float64)
    b = np.zeros(8, dtype=np.float64)
    for i in range(4):
        x, y = src[i]
        u, v = dst[i]
        a[2 * i] = [x, y, 1, 0, 0, 0, -x * u, -y * u]
        a[2 * i + 1] = [0, 0, 0, x, y, 1, -x * v, -y * v]
        b[2 * i] = u
        b[2 * i + 1] = v
    h8 = np.linalg.solve(a, b)
    h = np.append(h8, 1.0).reshape(3, 3)
    return h.astype(np.float64)


def warpPerspective(src, m, dsize):
    src = np.asarray(src)
    m = np.asarray(m, dtype=np.float64)
    w, h = int(dsize[0]), int(dsize[1])
    inv = np.linalg.inv(m)
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    ones = np.ones_like(xx, dtype=np.float64)
    dest = np.stack([xx.ravel(), yy.ravel(), ones.ravel()], axis=0)
    src_xy = inv @ dest
    src_xy /= src_xy[2]
    xs = src_xy[0].reshape(h, w)
    ys = src_xy[1].reshape(h, w)
    return _sample_linear(src, ys, xs)


def _sample_linear(src, ys, xs):
    old_h, old_w = src.shape[:2]
    y0 = np.floor(ys).astype(np.int32)
    x0 = np.floor(xs).astype(np.int32)
    wy = ys - y0
    wx = xs - x0
    inside = (y0 >= 0) & (y0 < old_h - 1) & (x0 >= 0) & (x0 < old_w - 1)
    y0c = np.clip(y0, 0, old_h - 1)
    x0c = np.clip(x0, 0, old_w - 1)
    y1c = np.clip(y0 + 1, 0, old_h - 1)
    x1c = np.clip(x0 + 1, 0, old_w - 1)
    if src.ndim == 2:
        a = src[y0c, x0c]
        b = src[y0c, x1c]
        c = src[y1c, x0c]
        d = src[y1c, x1c]
        out = a * (1 - wy) * (1 - wx) + b * (1 - wy) * wx + c * wy * (1 - wx) + d * wy * wx
        out = np.where(inside, out, 0)
    else:
        wy3, wx3 = wy[..., None], wx[..., None]
        a = src[y0c, x0c]
        b = src[y0c, x1c]
        c = src[y1c, x0c]
        d = src[y1c, x1c]
        out = a * (1 - wy3) * (1 - wx3) + b * (1 - wy3) * wx3 + c * wy3 * (1 - wx3) + d * wy3 * wx3
        out = np.where(inside[..., None], out, 0)
    if np.issubdtype(src.dtype, np.integer):
        return np.clip(np.rint(out), 0, np.iinfo(src.dtype).max).astype(src.dtype)
    return out.astype(src.dtype, copy=False)


def minAreaRect(points):
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 1:
        return ((0.0, 0.0), (0.0, 0.0), 0.0)
    if len(pts) == 1:
        x, y = pts[0]
        return ((float(x), float(y)), (0.0, 0.0), 0.0)
    hull = _convex_hull(pts)
    return _rotating_calipers(hull)


def _convex_hull(pts):
    pts = np.unique(pts, axis=0)
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def _rotating_calipers(hull):
    n = len(hull)
    if n == 0:
        return ((0.0, 0.0), (0.0, 0.0), 0.0)
    if n == 1:
        x, y = hull[0]
        return ((float(x), float(y)), (0.0, 0.0), 0.0)
    if n == 2:
        d = hull[1] - hull[0]
        length = float(np.hypot(d[0], d[1]))
        ang = float(np.degrees(np.arctan2(d[1], d[0])))
        cx, cy = (hull[0] + hull[1]) / 2
        return ((float(cx), float(cy)), (length, 0.0), ang)
    best = None
    best_area = None
    for i in range(n):
        a = hull[i]
        b = hull[(i + 1) % n]
        edge = b - a
        elen = np.hypot(edge[0], edge[1])
        if elen < 1e-12:
            continue
        ux, uy = edge / elen
        vx, vy = -uy, ux
        proj_u = hull[:, 0] * ux + hull[:, 1] * uy
        proj_v = hull[:, 0] * vx + hull[:, 1] * vy
        min_u, max_u = proj_u.min(), proj_u.max()
        min_v, max_v = proj_v.min(), proj_v.max()
        width = max_u - min_u
        height = max_v - min_v
        area = width * height
        if best_area is None or area < best_area:
            best_area = area
            cu = 0.5 * (min_u + max_u)
            cv = 0.5 * (min_v + max_v)
            cx = cu * ux + cv * vx
            cy = cu * uy + cv * vy
            ang = float(np.degrees(np.arctan2(uy, ux)))
            best = ((float(cx), float(cy)), (float(width), float(height)), ang)
    if best is None:
        x, y = hull.mean(axis=0)
        return ((float(x), float(y)), (0.0, 0.0), 0.0)
    (cx, cy), (bw, bh), ang = best
    if bw < bh:
        bw, bh = bh, bw
        ang += 90.0
    ang = ((ang + 90.0) % 180.0) - 90.0
    return ((cx, cy), (float(bw), float(bh)), float(ang))
