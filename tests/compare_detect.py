#!/usr/bin/env python3
"""旧検出と ONNX 検出の箱を比較する。モデルは載せない。"""

from __future__ import annotations

import json


def aabb(pts: list) -> tuple:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def iou(a: list, b: list) -> float:
    ax1, ay1, ax2, ay2 = aabb(a)
    bx1, by1, bx2, by2 = aabb(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def compare(old: dict, new: dict, thresh: float = 0.5) -> dict:
    used = set()
    ious = []
    for p in old["points"]:
        best = -1.0
        bj = None
        for j, q in enumerate(new["points"]):
            if j in used:
                continue
            v = iou(p, q)
            if v > best:
                best = v
                bj = j
        if bj is not None and best >= thresh:
            used.add(bj)
            ious.append(best)
    n = len(old["points"])
    return {
        "n_old": n,
        "n_new": len(new["points"]),
        "matched": len(ious),
        "mean_iou": (sum(ious) / len(ious)) if ious else 0.0,
        "min_iou": min(ious) if ious else 0.0,
        "exact": old["points"] == new["points"],
    }


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
