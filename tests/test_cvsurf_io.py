#!/usr/bin/env python3
"""imread が BMP / PNG を BGR で返すか。モデルを載せない。"""

from __future__ import annotations

import os
import struct
import sys
import tempfile
import zlib

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cvsurf import cv2 as surf


def _write_bmp(path, bgr):
    h, w = bgr.shape[:2]
    row = ((w * 3 + 3) // 4) * 4
    payload = bytearray()
    for y in range(h - 1, -1, -1):
        payload.extend(bgr[y].tobytes())
        payload.extend(b"\x00" * (row - w * 3))
    off = 54
    size = off + len(payload)
    buf = bytearray()
    buf += b"BM"
    buf += struct.pack("<IHHI", size, 0, 0, off)
    buf += struct.pack("<IiiHHIIiiII", 40, w, h, 1, 24, 0, len(payload), 0, 0, 0, 0)
    buf += payload
    open(path, "wb").write(buf)


def _write_png(path, bgr):
    rgb = bgr[:, :, ::-1]
    h, w = rgb.shape[:2]
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)

    def chunk(tag, data):
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    blob = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    open(path, "wb").write(blob)


def main():
    rng = np.random.default_rng(0)
    bgr = rng.integers(0, 256, (12, 16, 3), dtype=np.uint8)
    with tempfile.TemporaryDirectory() as td:
        bmp = os.path.join(td, "a.bmp")
        png = os.path.join(td, "a.png")
        _write_bmp(bmp, bgr)
        _write_png(png, bgr)
        got_bmp = surf.imread(bmp)
        got_png = surf.imread(png)
        if got_bmp.shape != bgr.shape:
            raise SystemExit(f"bmp shape {got_bmp.shape}")
        if not np.array_equal(got_bmp, bgr):
            raise SystemExit("bmp pixels")
        if got_png.shape != bgr.shape:
            raise SystemExit(f"png shape {got_png.shape}")
        if not np.array_equal(got_png, bgr):
            raise SystemExit("png pixels")
    print("ok imread bmp/png")


if __name__ == "__main__":
    main()
