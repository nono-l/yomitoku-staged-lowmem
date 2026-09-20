"""二値画像の輪郭。RETR_LIST と CHAIN_APPROX_SIMPLE だけ実装する。

穴も外輪郭も同じリストに並べる。階層は返すが使わない。
開始点は左が空の前景だけを配列で拾う。全画素の二重ループはしない。
"""

from __future__ import annotations

import numpy as np

from cvsurf.consts import CHAIN_APPROX_SIMPLE, RETR_LIST

# 8 近傍、時計回り。開始は右。
_DX = [1, 1, 0, -1, -1, -1, 0, 1]
_DY = [0, 1, 1, 1, 0, -1, -1, -1]


def findContours(image, mode, method):
    if mode != RETR_LIST:
        raise ValueError("only RETR_LIST")
    if method != CHAIN_APPROX_SIMPLE:
        raise ValueError("only CHAIN_APPROX_SIMPLE")
    img = np.asarray(image)
    if img.ndim != 2:
        raise ValueError("findContours expects 2d")
    binary = img > 0
    h, w = binary.shape
    visited = np.zeros((h, w), dtype=np.uint8)
    left_empty = np.empty_like(binary)
    left_empty[:, 0] = True
    left_empty[:, 1:] = ~binary[:, :-1]
    ys, xs = np.nonzero(binary & left_empty)
    contours = []
    for y, x in zip(ys.tolist(), xs.tolist()):
        if visited[y, x]:
            continue
        chain = _follow(binary, int(x), int(y), visited)
        if chain:
            contours.append(_approx(chain))
    hierarchy = np.zeros((len(contours), 4), dtype=np.int32)
    return contours, hierarchy


def _follow(binary, sx, sy, visited):
    h, w = binary.shape
    path = [(sx, sy)]
    x, y = sx, sy
    direction = 0
    visited[y, x] = 1
    # 閉じないときの上限。h*w*4 だと 2GiB で落ちる。
    limit = min(h * w, 20000)
    for _ in range(limit):
        found = False
        for k in range(8):
            nd = (direction + k) % 8
            nx, ny = x + _DX[nd], y + _DY[nd]
            if 0 <= nx < w and 0 <= ny < h and binary[ny, nx]:
                x, y = nx, ny
                direction = (nd + 6) % 8
                found = True
                visited[y, x] = 1
                if (x, y) == (sx, sy) and len(path) > 2:
                    return path
                path.append((x, y))
                break
        if not found:
            return path
    return path


def _approx(chain):
    if len(chain) <= 2:
        return np.asarray(chain, dtype=np.int32).reshape(-1, 1, 2)
    pts = [chain[0]]
    for i in range(1, len(chain) - 1):
        x0, y0 = pts[-1]
        x1, y1 = chain[i]
        x2, y2 = chain[i + 1]
        if (x1 - x0) * (y2 - y1) == (x2 - x1) * (y1 - y0):
            continue
        pts.append(chain[i])
    if chain[-1] != pts[-1]:
        pts.append(chain[-1])
    return np.asarray(pts, dtype=np.int32).reshape(-1, 1, 2)
