"""DBNet の箱出し。torch は使わない。area/length は shoelace。offset は pyclipper。"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pyclipper

MIN_SIZE = 2
THRESH = 0.3
BOX_THRESH = 0.4
MAX_CANDIDATES = 1500
UNCLIP_RATIO = 3.5


def _area_length(box):
    x = box[:, 0]
    y = box[:, 1]
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    d = np.diff(box, axis=0, append=box[:1])
    length = np.hypot(d[:, 0], d[:, 1]).sum()
    return float(area), float(length)


def _unclip(box, unclip_ratio):
    width = box[:, 0].max() - box[:, 0].min()
    height = box[:, 1].max() - box[:, 1].min()
    box_dist = min(width, height)
    ratio = unclip_ratio / math.sqrt(box_dist)
    area, length = _area_length(box)
    distance = area * ratio / length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = np.array(offset.Execute(distance))
    return expanded


def _mini_boxes(contour):
    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
    if points[1][1] > points[0][1]:
        index_1, index_4 = 0, 1
    else:
        index_1, index_4 = 1, 0
    if points[3][1] > points[2][1]:
        index_2, index_3 = 2, 3
    else:
        index_2, index_3 = 3, 2
    box = [points[index_1], points[index_2], points[index_3], points[index_4]]
    return box, min(bounding_box[1])


def _box_score(bitmap, contour):
    h, w = bitmap.shape[:2]
    box = contour.copy()
    xmin = np.clip(np.floor(box[:, 0].min()).astype(int), 0, w - 1)
    xmax = np.clip(np.ceil(box[:, 0].max()).astype(int), 0, w - 1)
    ymin = np.clip(np.floor(box[:, 1].min()).astype(int), 0, h - 1)
    ymax = np.clip(np.ceil(box[:, 1].max()).astype(int), 0, h - 1)
    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    box[:, 0] = box[:, 0] - xmin
    box[:, 1] = box[:, 1] - ymin
    cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
    return cv2.mean(bitmap[ymin : ymax + 1, xmin : xmax + 1], mask)[0]


def boxes_from_binary(
    binary,
    dest_size,
    min_size=MIN_SIZE,
    thresh=THRESH,
    box_thresh=BOX_THRESH,
    max_candidates=MAX_CANDIDATES,
    unclip_ratio=UNCLIP_RATIO,
):
    arr = np.asarray(binary)
    if arr.ndim == 4:
        pred = arr[0, 0]
    elif arr.ndim == 3:
        pred = arr[0]
    else:
        pred = arr
    bitmap = pred > thresh
    dest_height, dest_width = dest_size
    height, width = bitmap.shape
    contours, _ = cv2.findContours(
        (bitmap.astype(np.uint8) * 255),
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    num_contours = min(len(contours), max_candidates)
    boxes, scores = [], []
    for index in range(num_contours):
        contour = contours[index].squeeze(1)
        points, sside = _mini_boxes(contour)
        if sside < min_size:
            continue
        points = np.array(points)
        score = _box_score(pred, contour)
        if box_thresh > score:
            continue
        box = _unclip(points, unclip_ratio=unclip_ratio).reshape(-1, 1, 2)
        box, sside = _mini_boxes(box)
        if sside < min_size + 2:
            continue
        box = np.array(box)
        box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
        box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
        boxes.append(box.astype(np.int16).tolist())
        scores.append(float(score))
    return boxes, scores
