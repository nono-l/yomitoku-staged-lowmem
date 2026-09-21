"""初回の箱を残し、重ならないタイル箱だけ足す。検出器は載せない。

同じ帯を二度出さない。重なったら初回を残す。
"""

from __future__ import annotations

from det.tiles import aabb, overlap_area


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    inter = overlap_area(a, b)
    if inter == 0:
        return 0.0
    aa = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    ba = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    den = aa + ba - inter
    if den <= 0:
        return 0.0
    return inter / den


def merge_points(first: dict, extra: dict, max_iou: float = 0.5) -> dict:
    """first の points/scores を残し、重ならない extra だけ後ろへ足す。"""
    fp = list(first.get("points") or [])
    fs = list(first.get("scores") or [0.0] * len(fp))
    if len(fs) < len(fp):
        fs.extend([0.0] * (len(fp) - len(fs)))
    kept = [aabb(q) for q in fp]
    ep = list(extra.get("points") or [])
    es = list(extra.get("scores") or [0.0] * len(ep))
    if len(es) < len(ep):
        es.extend([0.0] * (len(ep) - len(es)))
    for q, sc in zip(ep, es):
        box = aabb(q)
        if any(iou(box, k) > max_iou for k in kept):
            continue
        fp.append(q)
        fs.append(sc)
        kept.append(box)
    return {"points": fp, "scores": fs}
