"""検出入力。使っていた 3 関数だけ。"""

from __future__ import annotations

import numpy as np

from cvsurf import cv2

SHORTEST = 1280
MAX_LENGTH = 1600
RGB_MEAN = (0.485, 0.456, 0.406)
RGB_STD = (0.229, 0.224, 0.225)


def resize_shortest_edge(img, shortest_edge_length=SHORTEST, max_length=MAX_LENGTH):
    h, w = img.shape[:2]
    scale = shortest_edge_length / min(h, w)
    if h < w:
        new_h, new_w = shortest_edge_length, int(w * scale)
    else:
        new_h, new_w = int(h * scale), shortest_edge_length
    if max(new_h, new_w) > max_length:
        scale = float(max_length) / max(new_h, new_w)
        new_h, new_w = int(new_h * scale), int(new_w * scale)
    neww = max(int(new_w / 32) * 32, 32)
    newh = max(int(new_h / 32) * 32, 32)
    return cv2.resize(img, (neww, newh), interpolation=cv2.INTER_AREA)


def standardization_image(img, rgb=RGB_MEAN, std=RGB_STD):
    img = img[:, :, ::-1]
    img = img / 255.0
    img = (img - np.array(rgb)) / np.array(std)
    return img.astype(np.float32)


def to_nchw(img):
    return np.transpose(img, (2, 0, 1))[None, ...].astype(np.float32)


def prepare_bgr(bgr):
    """BGR uint8 -> encoder 入力 (1,3,H,W)。本体と同じ色の順。"""
    x = bgr.copy()[:, :, ::-1].astype(np.float32)
    x = resize_shortest_edge(x)
    x = standardization_image(x)
    return to_nchw(x)
