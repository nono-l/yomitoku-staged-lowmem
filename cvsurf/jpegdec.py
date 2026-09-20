"""基線 JPEG を BGR uint8 にする。プログレッシブ・CMYK は捨てる。

ハフマンと 8x8 IDCT だけ。公式 cv2 の libjpeg 設定とは画素が違う。
"""

from __future__ import annotations

import math
import struct

import numpy as np

_ZIGZAG = [
    0, 1, 5, 6, 14, 15, 27, 28,
    2, 4, 7, 13, 16, 26, 29, 42,
    3, 8, 12, 17, 25, 30, 41, 43,
    9, 11, 18, 24, 31, 40, 44, 53,
    10, 19, 23, 32, 39, 45, 52, 54,
    20, 22, 33, 38, 46, 51, 55, 60,
    21, 34, 37, 47, 50, 56, 59, 61,
    35, 36, 48, 49, 57, 58, 62, 63,
]


def decode_jpeg(data: bytes) -> np.ndarray:
    p = _Parser(data)
    p.parse()
    return p.decode_image()


class _Huff:
    def __init__(self, lengths, symbols):
        self.by_len = {}
        code = 0
        k = 0
        for i, n in enumerate(lengths):
            length = i + 1
            bucket = {}
            for _ in range(n):
                bucket[code] = symbols[k]
                k += 1
                code += 1
            self.by_len[length] = bucket
            code <<= 1


class _Parser:
    def __init__(self, data):
        self.data = data
        self.pos = 0
        self.qtab = {}
        self.ht_dc = {}
        self.ht_ac = {}
        self.width = self.height = 0
        self.comps = []
        self.sos = None
        self.scan = b""
        self.restart = 0

    def _u8(self):
        v = self.data[self.pos]
        self.pos += 1
        return v

    def _u16(self):
        v = struct.unpack_from(">H", self.data, self.pos)[0]
        self.pos += 2
        return v

    def parse(self):
        if self.data[0:2] != b"\xff\xd8":
            raise ValueError("soi")
        self.pos = 2
        while self.pos < len(self.data):
            if self.data[self.pos] != 0xFF:
                self.pos += 1
                continue
            while self.data[self.pos] == 0xFF:
                self.pos += 1
            marker = self._u8()
            if marker == 0xD9:
                break
            if marker == 0xDA:
                self._sos()
                break
            length = self._u16()
            payload = self.data[self.pos : self.pos + length - 2]
            self.pos += length - 2
            if marker == 0xC0:
                self._sof(payload)
            elif marker in (0xC1, 0xC2, 0xC3):
                raise ValueError(
                    f"jpeg not baseline SOF0 (got 0x{marker:02X}); convert to BMP/PNG"
                )
            elif marker == 0xDB:
                self._dqt(payload)
            elif marker == 0xC4:
                self._dht(payload)
            elif marker == 0xDD:
                self.restart = struct.unpack(">H", payload)[0]

    def _sof(self, p):
        precision = p[0]
        if precision != 8:
            raise ValueError("jpeg precision")
        self.height, self.width = struct.unpack(">HH", p[1:5])
        n = p[5]
        self.comps = []
        off = 6
        for _ in range(n):
            cid, samp, qid = p[off], p[off + 1], p[off + 2]
            self.comps.append({"id": cid, "h": samp >> 4, "v": samp & 15, "qid": qid})
            off += 3

    def _dqt(self, p):
        off = 0
        while off < len(p):
            info = p[off]
            off += 1
            pq, tid = info >> 4, info & 15
            if pq != 0:
                raise ValueError("qtab 16")
            raw = list(p[off : off + 64])
            off += 64
            tab = [0] * 64
            for i, z in enumerate(_ZIGZAG):
                tab[i] = raw[z]
            self.qtab[tid] = np.array(tab, dtype=np.int32).reshape(8, 8)

    def _dht(self, p):
        off = 0
        while off < len(p):
            info = p[off]
            off += 1
            lengths = list(p[off : off + 16])
            off += 16
            n = sum(lengths)
            symbols = list(p[off : off + n])
            off += n
            tbl = _Huff(lengths, symbols)
            cls, tid = info >> 4, info & 15
            if cls == 0:
                self.ht_dc[tid] = tbl
            else:
                self.ht_ac[tid] = tbl

    def _sos(self):
        length = self._u16()
        payload = self.data[self.pos : self.pos + length - 2]
        self.pos += length - 2
        n = payload[0]
        spec = []
        off = 1
        for _ in range(n):
            cid = payload[off]
            tdta = payload[off + 1]
            spec.append({"id": cid, "td": tdta >> 4, "ta": tdta & 15})
            off += 2
        self.sos = spec
        scan = bytearray()
        d = self.data
        i = self.pos
        while i < len(d):
            b = d[i]
            if b == 0xFF:
                if i + 1 < len(d) and d[i + 1] == 0x00:
                    scan.append(0xFF)
                    i += 2
                    continue
                if i + 1 < len(d) and 0xD0 <= d[i + 1] <= 0xD7:
                    i += 2
                    continue
                break
            scan.append(b)
            i += 1
        self.scan = bytes(scan)

    def decode_image(self):
        if not self.comps:
            raise ValueError("sof")
        bits = _BitReader(self.scan)
        hmax = max(c["h"] for c in self.comps)
        vmax = max(c["v"] for c in self.comps)
        mcu_w = 8 * hmax
        mcu_h = 8 * vmax
        nx = math.ceil(self.width / mcu_w)
        ny = math.ceil(self.height / mcu_h)
        planes = {}
        for c in self.comps:
            ph = ny * c["v"] * 8
            pw = nx * c["h"] * 8
            planes[c["id"]] = np.zeros((ph, pw), dtype=np.float32)
        pred = {c["id"]: 0 for c in self.comps}
        mcu_i = 0
        for my in range(ny):
            for mx in range(nx):
                if self.restart and mcu_i and mcu_i % self.restart == 0:
                    bits.align()
                    for k in pred:
                        pred[k] = 0
                for c, s in zip(self.comps, self.sos):
                    q = self.qtab[c["qid"]]
                    hdc = self.ht_dc[s["td"]]
                    hac = self.ht_ac[s["ta"]]
                    for vy in range(c["v"]):
                        for hx in range(c["h"]):
                            block = _decode_block(bits, hdc, hac, q, pred, c["id"])
                            y0 = my * c["v"] * 8 + vy * 8
                            x0 = mx * c["h"] * 8 + hx * 8
                            planes[c["id"]][y0 : y0 + 8, x0 : x0 + 8] = block
                mcu_i += 1
        yid = self.comps[0]["id"]
        ypl = _scale_plane(planes[yid], self.height, self.width)
        if len(self.comps) == 1:
            g = np.clip(ypl + 128, 0, 255).astype(np.uint8)
            return np.repeat(g[:, :, None], 3, axis=2)
        cb = _scale_plane(planes[self.comps[1]["id"]], self.height, self.width)
        cr = _scale_plane(planes[self.comps[2]["id"]], self.height, self.width)
        y = ypl + 128.0
        r = y + 1.402 * cr
        g = y - 0.344136 * cb - 0.714136 * cr
        b = y + 1.772 * cb
        rgb = np.stack([r, g, b], axis=2)
        return np.clip(rgb[:, :, ::-1], 0, 255).astype(np.uint8)


def _scale_plane(pl, h, w):
    if pl.shape[0] == h and pl.shape[1] == w:
        return pl
    ys = np.linspace(0, pl.shape[0] - 1, h)
    xs = np.linspace(0, pl.shape[1] - 1, w)
    y0 = np.floor(ys).astype(np.int32)
    x0 = np.floor(xs).astype(np.int32)
    return pl[y0][:, x0]


def _decode_block(bits, hdc, hac, q, pred, cid):
    dc_len = bits.read_huff(hdc)
    dc_diff = bits.receive_extend(dc_len)
    pred[cid] += dc_diff
    zz = [0] * 64
    zz[0] = pred[cid]
    k = 1
    while k < 64:
        rs = bits.read_huff(hac)
        rrrr, ssss = rs >> 4, rs & 15
        if ssss == 0:
            if rrrr == 15:
                k += 16
                continue
            break
        k += rrrr
        if k >= 64:
            break
        zz[k] = bits.receive_extend(ssss)
        k += 1
    natural = [0] * 64
    for i, z in enumerate(_ZIGZAG):
        natural[i] = zz[z]
    coef = np.array(natural, dtype=np.float32).reshape(8, 8) * q.astype(np.float32)
    return _idct8(coef)


def _idct8(coef):
    t = np.empty((8, 8), dtype=np.float32)
    out = np.empty((8, 8), dtype=np.float32)
    for i in range(8):
        t[:, i] = _idct1(coef[:, i])
    for i in range(8):
        out[i, :] = _idct1(t[i, :])
    return out


_C = [math.cos((2 * x + 1) * u * math.pi / 16.0) for u in range(8) for x in range(8)]


def _idct1(vec):
    s = np.zeros(8, dtype=np.float32)
    for u in range(8):
        cu = 1.0 / math.sqrt(2) if u == 0 else 1.0
        for x in range(8):
            s[x] += cu * vec[u] * _C[u * 8 + x]
    return s * 0.5


class _BitReader:
    def __init__(self, data):
        self.data = data
        self.pos = 0
        self.bit = 0
        self.cur = 0

    def align(self):
        self.bit = 0

    def _fill(self):
        if self.pos >= len(self.data):
            return 0
        b = self.data[self.pos]
        self.pos += 1
        return b

    def read_bit(self):
        if self.bit == 0:
            self.cur = self._fill()
            self.bit = 8
        self.bit -= 1
        return (self.cur >> self.bit) & 1

    def read_bits(self, n):
        v = 0
        for _ in range(n):
            v = (v << 1) | self.read_bit()
        return v

    def read_huff(self, table):
        code = 0
        for length in range(1, 17):
            code = (code << 1) | self.read_bit()
            hit = table.by_len.get(length, {}).get(code)
            if hit is not None:
                return hit
        raise ValueError("huff")

    def receive_extend(self, ssss):
        if ssss == 0:
            return 0
        v = self.read_bits(ssss)
        vt = 1 << (ssss - 1)
        if v < vt:
            return v + (-1 << ssss) + 1
        return v
