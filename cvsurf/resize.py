"""縮小・拡大。OpenCV の名前を残し、補間はここで固定する。

INTER_AREA は画素の重なり面積で平均する。拡大でも線形に落とさない。
公式と同じく端は分数重み。整数倍の拡大は元画素の繰り返しになる。
"""

from __future__ import annotations

import numpy as np

from cvsurf.consts import INTER_AREA, INTER_LINEAR


def resize(src, dsize, interpolation=INTER_LINEAR):
    src = np.asarray(src)
    if src.size == 0:
        raise ValueError("empty image")
    w, h = int(dsize[0]), int(dsize[1])
    if w < 1 or h < 1:
        raise ValueError("dsize")
    if interpolation == INTER_AREA:
        return _area(src, h, w)
    return _linear(src, h, w)


def _linear(src, new_h, new_w):
    old_h, old_w = src.shape[:2]
    if old_h == new_h and old_w == new_w:
        return src.copy()
    ys = np.linspace(0, old_h - 1, new_h)
    xs = np.linspace(0, old_w - 1, new_w)
    y0 = np.floor(ys).astype(np.int32)
    x0 = np.floor(xs).astype(np.int32)
    y1 = np.clip(y0 + 1, 0, old_h - 1)
    x1 = np.clip(x0 + 1, 0, old_w - 1)
    wy = (ys - y0)[:, None]
    wx = (xs - x0)[None, :]
    y0 = np.clip(y0, 0, old_h - 1)
    x0 = np.clip(x0, 0, old_w - 1)
    a = src[y0][:, x0]
    b = src[y0][:, x1]
    c = src[y1][:, x0]
    d = src[y1][:, x1]
    if src.ndim == 2:
        wy = wy[:, :, 0] if wy.ndim == 3 else wy
        out = (
            a * (1 - wy) * (1 - wx)
            + b * (1 - wy) * wx
            + c * wy * (1 - wx)
            + d * wy * wx
        )
    else:
        wy = wy[:, :, None]
        wx = wx[:, :, None]
        out = (
            a * (1 - wy) * (1 - wx)
            + b * (1 - wy) * wx
            + c * wy * (1 - wx)
            + d * wy * wx
        )
    if np.issubdtype(src.dtype, np.integer):
        return np.clip(np.rint(out), 0, np.iinfo(src.dtype).max).astype(src.dtype)
    return out.astype(src.dtype, copy=False)


def _area(src, new_h, new_w):
    """重なり面積の平均。軸は独立なので幅→高さの順。"""
    old_h, old_w = src.shape[:2]
    if old_h == new_h and old_w == new_w:
        return src.copy()
    x = _box_1d(src.astype(np.float64), old_w, new_w, axis=1)
    out = _box_1d(x, old_h, new_h, axis=0)
    if np.issubdtype(src.dtype, np.integer):
        return np.clip(np.rint(out), 0, np.iinfo(src.dtype).max).astype(src.dtype)
    return out.astype(src.dtype, copy=False)


def _box_1d(arr, old, new, axis):
    if old == new:
        return arr
    arr = np.moveaxis(arr, axis, -1)
    lead = arr.shape[:-1]
    out = np.zeros(lead + (new,), dtype=np.float64)
    for i in range(new):
        a = i * old / new
        b = (i + 1) * old / new
        i0 = int(np.floor(a))
        i1 = min(int(np.ceil(b - 1e-12)), old - 1)
        wsum = 0.0
        acc = np.zeros(lead, dtype=np.float64)
        for k in range(i0, i1 + 1):
            w = min(b, k + 1) - max(a, k)
            if w <= 0:
                continue
            acc += arr[..., k] * w
            wsum += w
        if wsum > 0:
            out[..., i] = acc / wsum
    return np.moveaxis(out, -1, axis)
