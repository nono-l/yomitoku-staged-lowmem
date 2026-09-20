"""閉じた多角形の外側オフセット。Clipper 6.4.2 の JT_ROUND 相当。

pyclipper の C++ 拡張を置かない。入力は検出の minAreaRect 四隅。
穴・交差・オープパスは扱わない。箱の整数化は切り捨て（AddPath と同じ）。
"""

from __future__ import annotations

import math

ARC_TOL = 0.25


def _iround(val: float) -> int:
    if val < 0:
        return int(val - 0.5)
    return int(val + 0.5)


def _unit_normal(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if dx == 0 and dy == 0:
        return (0.0, 0.0)
    f = math.hypot(dx, dy)
    dx /= f
    dy /= f
    return (dy, -dx)


def offset_round(path, delta, arc_tol: float = ARC_TOL):
    pts = [(int(p[0]), int(p[1])) for p in path]
    n = len(pts)
    if n < 3 or delta == 0:
        return pts
    norms = [_unit_normal(pts[i], pts[(i + 1) % n]) for i in range(n)]
    abs_delta = abs(delta)
    if arc_tol <= 0:
        y = ARC_TOL
    elif arc_tol > abs_delta * ARC_TOL:
        y = abs_delta * ARC_TOL
    else:
        y = arc_tol
    steps = math.pi / math.acos(1 - y / abs_delta)
    if steps > abs_delta * math.pi:
        steps = abs_delta * math.pi
    two_pi = 2 * math.pi
    m_sin = math.sin(two_pi / steps)
    m_cos = math.cos(two_pi / steps)
    steps_per_rad = steps / two_pi
    if delta < 0:
        m_sin = -m_sin
    dest = []
    k = n - 1
    for j in range(n):
        nk = norms[k]
        nj = norms[j]
        sin_a = nk[0] * nj[1] - nj[0] * nk[1]
        if abs(sin_a * delta) < 1.0:
            cos_a = nk[0] * nj[0] + nj[1] * nk[1]
            if cos_a > 0:
                dest.append(
                    (
                        _iround(pts[j][0] + nk[0] * delta),
                        _iround(pts[j][1] + nk[1] * delta),
                    )
                )
                k = j
                continue
        else:
            if sin_a > 1:
                sin_a = 1.0
            elif sin_a < -1:
                sin_a = -1.0
        if sin_a * delta < 0:
            dest.append(
                (
                    _iround(pts[j][0] + nk[0] * delta),
                    _iround(pts[j][1] + nk[1] * delta),
                )
            )
            dest.append(pts[j])
            dest.append(
                (
                    _iround(pts[j][0] + nj[0] * delta),
                    _iround(pts[j][1] + nj[1] * delta),
                )
            )
        else:
            a = math.atan2(sin_a, nk[0] * nj[0] + nk[1] * nj[1])
            nsteps = max(_iround(steps_per_rad * abs(a)), 1)
            x, yv = nk[0], nk[1]
            for _ in range(nsteps):
                dest.append(
                    (
                        _iround(pts[j][0] + x * delta),
                        _iround(pts[j][1] + yv * delta),
                    )
                )
                x2 = x
                x = x * m_cos - m_sin * yv
                yv = x2 * m_sin + yv * m_cos
            dest.append(
                (
                    _iround(pts[j][0] + nj[0] * delta),
                    _iround(pts[j][1] + nj[1] * delta),
                )
            )
        k = j
    return dest
