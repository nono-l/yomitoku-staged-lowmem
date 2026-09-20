"""画像読み。JPEG は基線 DCT だけ。拡張・プログレッシブは捨てる。

PIL / 公式 cv2 とは画素が一致しない。この部品の既定画素はここ。
PNG は zlib で展開する。BMP は無圧縮だけ。
"""

from __future__ import annotations

import struct
import zlib

import numpy as np

from cvsurf.jpegdec import decode_jpeg


def imread(path):
    with open(path, "rb") as f:
        data = f.read()
    if data.startswith(b"\xff\xd8"):
        return decode_jpeg(data)
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return _png(data)
    if data.startswith(b"BM"):
        return _bmp(data)
    raise SystemExit(f"unsupported image: {path}")


def _png(data):
    pos = 8
    width = height = None
    bit_depth = color_type = None
    raw = []
    while pos + 8 <= len(data):
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        ctype = data[pos + 4 : pos + 8]
        chunk = data[pos + 8 : pos + 8 + length]
        pos += 12 + length
        if ctype == b"IHDR":
            width, height, bit_depth, color_type, *_ = struct.unpack(">IIBBBBB", chunk)
        elif ctype == b"IDAT":
            raw.append(chunk)
        elif ctype == b"IEND":
            break
    if width is None:
        raise ValueError("png ihdr")
    if bit_depth != 8:
        raise ValueError("png bit depth")
    inflated = zlib.decompress(b"".join(raw))
    if color_type == 2:
        bpp = 3
    elif color_type == 6:
        bpp = 4
    elif color_type == 0:
        bpp = 1
    else:
        raise ValueError("png color")
    stride = width * bpp + 1
    rows = []
    prev = bytearray(width * bpp)
    src = memoryview(inflated)
    off = 0
    for _ in range(height):
        filt = src[off]
        scan = bytearray(src[off + 1 : off + stride])
        off += stride
        _paeth_unfilter(filt, scan, prev, bpp)
        prev = scan
        rows.append(scan)
    arr = np.frombuffer(b"".join(rows), dtype=np.uint8).reshape(height, width, bpp)
    if bpp == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif bpp == 4:
        arr = arr[:, :, :3]
    return arr[:, :, ::-1].copy()


def _paeth_unfilter(filt, scan, prev, bpp):
    n = len(scan)
    if filt == 0:
        return
    if filt == 1:
        for i in range(n):
            left = scan[i - bpp] if i >= bpp else 0
            scan[i] = (scan[i] + left) & 255
        return
    if filt == 2:
        for i in range(n):
            scan[i] = (scan[i] + prev[i]) & 255
        return
    if filt == 3:
        for i in range(n):
            left = scan[i - bpp] if i >= bpp else 0
            scan[i] = (scan[i] + ((left + prev[i]) // 2)) & 255
        return
    if filt == 4:
        for i in range(n):
            a = scan[i - bpp] if i >= bpp else 0
            b = prev[i]
            c = prev[i - bpp] if i >= bpp else 0
            p = a + b - c
            pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
            pr = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
            scan[i] = (scan[i] + pr) & 255
        return
    raise ValueError("png filter")


def _bmp(data):
    offset = struct.unpack_from("<I", data, 10)[0]
    width = struct.unpack_from("<i", data, 18)[0]
    height = struct.unpack_from("<i", data, 22)[0]
    bpp = struct.unpack_from("<H", data, 28)[0]
    comp = struct.unpack_from("<I", data, 30)[0]
    if bpp != 24 or comp != 0:
        raise ValueError("bmp format")
    flip = height > 0
    height = abs(height)
    row_stride = ((width * 3 + 3) // 4) * 4
    rows = []
    pos = offset
    for _ in range(height):
        row = np.frombuffer(data, dtype=np.uint8, count=width * 3, offset=pos).reshape(width, 3)
        rows.append(row.copy())
        pos += row_stride
    return np.stack(rows[::-1] if flip else rows, axis=0)
