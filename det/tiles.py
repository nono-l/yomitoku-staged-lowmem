"""既にある箱を避けて、空いた区画の矩形を出す。検出器は載せない。

全域の最短辺を上げない。空いたところだけ二度目の検出に渡すための目録である。
伸長もしない。絵だけの区画も落とさない。落とすと細字も落ちる。
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def aabb(quad: list) -> tuple[int, int, int, int]:
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def overlap_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0
    return (x1 - x0) * (y1 - y0)


def ink_ratio(img, x: int, y: int, w: int, h: int, delta: float = 16.0) -> float:
    """中央値から外れた画素の割合。余白は 0 に近い。"""
    patch = img[y : y + h, x : x + w]
    if patch.size == 0:
        return 0.0
    gray = patch.mean(axis=2) if patch.ndim == 3 else patch
    med = float(np.median(gray))
    return float((np.abs(gray.astype(np.float32) - med) >= delta).mean())


def empty_tiles(
    width: int,
    height: int,
    quads: Iterable[list],
    tile: int = 320,
    max_fill: float = 0.05,
    img=None,
    min_ink: float = 0.0,
) -> list[dict]:
    """箱がほとんど無く、必要ならインクもあるタイル。座標は元画像。"""
    if width < 1 or height < 1:
        raise ValueError("size")
    if tile < 32:
        raise ValueError("tile")
    if min_ink > 0 and img is None:
        raise ValueError("min_ink needs img")
    boxes = [aabb(q) for q in quads]
    out = []
    y = 0
    row = 0
    while y < height:
        x = 0
        col = 0
        th = min(tile, height - y)
        while x < width:
            tw = min(tile, width - x)
            cell = (x, y, x + tw, y + th)
            area = tw * th
            filled = sum(overlap_area(cell, b) for b in boxes)
            if area > 0 and filled / area <= max_fill:
                ink = ink_ratio(img, x, y, tw, th) if img is not None else None
                if min_ink <= 0 or (ink is not None and ink >= min_ink):
                    rec = {
                        "x": x,
                        "y": y,
                        "w": tw,
                        "h": th,
                        "row": row,
                        "col": col,
                    }
                    if ink is not None:
                        rec["ink"] = round(float(ink), 4)
                    out.append(rec)
            x += tile
            col += 1
        y += tile
        row += 1
    return out
