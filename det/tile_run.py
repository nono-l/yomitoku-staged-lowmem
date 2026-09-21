"""タイル画素の切り出しと、箱座標を元画像へ戻す。検出器は載せない。"""

from __future__ import annotations

import struct

import numpy as np


def crop_bgr(img, tile: dict) -> np.ndarray:
    x, y, w, h = int(tile["x"]), int(tile["y"]), int(tile["w"]), int(tile["h"])
    if w < 32 or h < 32:
        raise ValueError("tile too small")
    return img[y : y + h, x : x + w].copy()


def remap_quads(quads, ox: int, oy: int) -> list:
    out = []
    for q in quads:
        out.append([[int(p[0]) + ox, int(p[1]) + oy] for p in q])
    return out


def write_bmp(path: str, bgr: np.ndarray) -> None:
    h, w = bgr.shape[:2]
    row = ((w * 3 + 3) // 4) * 4
    raw = bytearray()
    for y in range(h - 1, -1, -1):
        raw.extend(bytes(bgr[y].reshape(-1)))
        raw.extend(b"\x00" * (row - w * 3))
    off = 54
    size = off + len(raw)
    fileh = struct.pack("<2sIHHI", b"BM", size, 0, 0, off)
    dib = struct.pack("<IiiHHIIiiII", 40, w, h, 1, 24, 0, len(raw), 0, 0, 0, 0)
    with open(path, "wb") as f:
        f.write(fileh)
        f.write(dib)
        f.write(raw)
