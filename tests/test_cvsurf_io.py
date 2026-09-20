#!/usr/bin/env python3
"""imread が BMP / PNG / 基線 JPEG を BGR で返すか。モデルを載せない。"""

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
    baseline = bytes.fromhex(
        "ffd8ffe000104a46494600010100000100010000ffdb0043000201010101010201"
        "010102020202020403020202020504040304060506060605060606070908060709"
        "070606080b08090a0a0a0a0a06080b0c0b0a0c090a0a0affdb0043010202020202"
        "02050303050a0706070a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a"
        "0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0affc000110800"
        "08001003011100021101031101ffc4001f00000105010101010101000000000000"
        "00000102030405060708090a0bffc400b510000201030302040305050404000001"
        "7d01020300041105122131410613516107227114328191a1082342b1c11552d1f0"
        "2433627282090a161718191a25262728292a3435363738393a434445464748494a"
        "535455565758595a636465666768696a737475767778797a838485868788898a92"
        "939495969798999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9bac2c3c4c5c6c7c8"
        "c9cad2d3d4d5d6d7d8d9dae1e2e3e4e5e6e7e8e9eaf1f2f3f4f5f6f7f8f9faffc4"
        "001f0100030101010101010101010000000000000102030405060708090a0bffc4"
        "00b511000201020404030407050404000102770001020311040521310612415107"
        "61711322328108144291a1b1c109233352f0156272d10a162434e125f11718191a"
        "262728292a35363738393a434445464748494a535455565758595a636465666768"
        "696a737475767778797a82838485868788898a92939495969798999aa2a3a4a5a6"
        "a7a8a9aab2b3b4b5b6b7b8b9bac2c3c4c5c6c7c8c9cad2d3d4d5d6d7d8d9dae2e3"
        "e4e5e6e7e8e9eaf2f3f4f5f6f7f8f9faffda000c03010002110311003f00fca8d5"
        "755b0d46c34cb3b3f0cd8d8496162d05d5d5a4939935190cf2ca2e2612caeab204"
        "912102258e3f2e08c9432192493f5b49a6f5ff0081fd6fa9f18ddca54c47ffd9"
    )
    progressive = bytes.fromhex(
        "ffd8ffe000104a46494600010100000100010000ffdb0043000201010101010201"
        "010102020202020403020202020504040304060506060605060606070908060709"
        "070606080b08090a0a0a0a0a06080b0c0b0a0c090a0a0affdb0043010202020202"
        "02050303050a0706070a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a"
        "0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0affc200110800"
        "08001003012200021101031101ffc4001600010101000000000000000000000000"
        "000308ffc4001501010100000000000000000000000000000506ffda000c030100"
        "02100310000001ca712b85ffc40016100101010000000000000000000000000003"
        "0200ffda000801010001050255848dffc400141101000000000000000000000000"
        "00000000ffda0008010301013f017fffc400141101000000000000000000000000"
        "00000000ffda0008010201013f017fffc4001d1000030002020300000000000000"
        "00000001021103210004123141ffda0008010100063f02c689d64428919949b90d"
        "27c8d3ef7350403ed279ffc4001910010002030000000000000000000000000110"
        "11213141ffda0008010100013f214b3befa2205062d0a03fffda000c0301000200"
        "030000001083ffc40014110100000000000000000000000000000000ffda000801"
        "0301013f107fffc4001511010100000000000000000000000000000031ffda0008"
        "010201013f108fffc400161001010100000000000000000000000000011011ffda"
        "0008010100013f107f0a4aa381c146c25a83ffd9"
    )
    with tempfile.TemporaryDirectory() as td:
        base = os.path.join(td, "b.jpg")
        prog = os.path.join(td, "p.jpg")
        open(base, "wb").write(baseline)
        open(prog, "wb").write(progressive)
        img = surf.imread(base)
        if img.ndim != 3 or img.shape[2] != 3:
            raise SystemExit(f"baseline jpeg shape {img.shape}")
        print(f"ok imread baseline jpeg {img.shape}")
        try:
            surf.imread(prog)
            raise SystemExit("progressive jpeg should fail")
        except ValueError as e:
            if "SOF0" not in str(e) and "0xC2" not in str(e):
                raise SystemExit(f"wrong progressive error: {e}")
            print("ok reject progressive jpeg")


if __name__ == "__main__":
    main()
