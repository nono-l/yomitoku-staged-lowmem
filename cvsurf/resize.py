"""縮小・拡大。OpenCV の名前を残し、補間はここで固定する。

INTER_AREA は縮小の平均、INTER_LINEAR は双線形。
公式 INTER_AREA は端の分数重みを使う。ここはブロック平均なので画素が違う。
settei21 では公式縮小＋既定後処理で 36 箱、ここの縮小だと 29 箱。輪郭の差ではない。
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
    if interpolation == INTER_AREA and (h < src.shape[0] or w < src.shape[1]):
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
    old_h, old_w = src.shape[:2]
    ys = np.linspace(0, old_h, new_h + 1)
    xs = np.linspace(0, old_w, new_w + 1)
    out_shape = (new_h, new_w) + src.shape[2:]
    acc = np.zeros(out_shape, dtype=np.float64)
    src_f = src.astype(np.float64)
    for i in range(new_h):
        y0, y1 = ys[i], ys[i + 1]
        iy0 = int(np.floor(y0))
        iy1 = int(np.ceil(y1)) - 1
        iy1 = min(iy1, old_h - 1)
        for j in range(new_w):
            x0, x1 = xs[j], xs[j + 1]
            ix0 = int(np.floor(x0))
            ix1 = int(np.ceil(x1)) - 1
            ix1 = min(ix1, old_w - 1)
            block = src_f[iy0 : iy1 + 1, ix0 : ix1 + 1]
            if block.size == 0:
                continue
            acc[i, j] = block.mean(axis=(0, 1)) if src.ndim == 3 else block.mean()
    if np.issubdtype(src.dtype, np.integer):
        return np.clip(np.rint(acc), 0, np.iinfo(src.dtype).max).astype(src.dtype)
    return acc.astype(src.dtype, copy=False)
