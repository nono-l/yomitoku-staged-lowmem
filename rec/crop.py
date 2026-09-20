"""箱から 32px 高の作物を切る。使っていた前処理だけ。"""

from __future__ import annotations

import cv2
import numpy as np

IMG_H = 32
IMG_W = 800
ALIGN = 8
MARGIN = 96
RESIZE_POLICY = "fit"


def validate_quads(img, quad):
    if len(quad) != 4:
        return None
    for point in quad:
        if len(point) != 2:
            return None
    quad = np.array(quad, dtype=int)
    x1 = np.min(quad[:, 0])
    x2 = np.max(quad[:, 0])
    y1 = np.min(quad[:, 1])
    y2 = np.max(quad[:, 1])
    h, w = img.shape[:2]
    if x1 < 0 or x2 > w or y1 < 0 or y2 > h:
        return None
    return True


def extract_roi_with_perspective(img, quad):
    quad = np.array(quad, dtype=np.int64)
    roi_img = img[
        int(min(quad[:, 1])) : int(max(quad[:, 1])),
        int(min(quad[:, 0])) : int(max(quad[:, 0])),
        :,
    ]
    quad = quad.copy()
    quad[:, 0] -= int(min(quad[:, 0]))
    quad[:, 1] -= int(min(quad[:, 1]))
    width = int(np.linalg.norm(quad[0] - quad[1]))
    height = int(np.linalg.norm(quad[1] - quad[2]))
    pts1 = np.float32(quad)
    pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(roi_img, M, (width, height))


def rotate_text_image(img, thresh_aspect=2):
    h, w = img.shape[:2]
    if h > thresh_aspect * w:
        # cv2.ROTATE_90_COUNTERCLOCKWISE と同じ。90度だけ。
        img = np.rot90(img, 1)
    return img


def calc_resize_without_padding(img, target_size, resize_policy="fit"):
    if resize_policy not in ("fit", "downscale"):
        raise ValueError(f"Unknown resize_policy: {resize_policy}")
    h, w = img.shape[:2]
    scale = min(target_size[0] / h, target_size[1] / w)
    if resize_policy == "downscale":
        scale = min(1.0, scale)
    return max(1, int(h * scale)), max(1, int(w * scale))


def resize_with_dynamic_padding(
    img,
    target_size,
    align=ALIGN,
    margin=MARGIN,
    background_color=(0, 0, 0),
    resize_policy=RESIZE_POLICY,
):
    h = img.shape[0]
    new_h, new_w = calc_resize_without_padding(img, target_size, resize_policy)
    interp = cv2.INTER_AREA if new_h <= h else cv2.INTER_LINEAR
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
    canvas_w = min(target_size[1], ((new_w + margin + align - 1) // align) * align)
    canvas = np.zeros((target_size[0], canvas_w, 3), dtype=np.uint8)
    canvas[:, :] = background_color
    canvas[:new_h, :new_w, :] = resized
    return canvas


def to_nchw(rgb_u8):
    x = rgb_u8.astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    return np.transpose(x, (2, 0, 1))


def crop_quad(bgr, quad, img_size=(IMG_H, IMG_W), margin=MARGIN):
    """BGR 画像と四隅から、encoder 入力 (1,3,32,W) を返す。無効なら None。"""
    rgb = bgr[:, :, ::-1]
    if validate_quads(rgb, quad) is None:
        return None
    roi = extract_roi_with_perspective(rgb, quad)
    if roi is None or roi.size == 0:
        return None
    roi = rotate_text_image(roi, thresh_aspect=2)
    resized = resize_with_dynamic_padding(
        roi, img_size, margin=margin, resize_policy=RESIZE_POLICY
    )
    return to_nchw(resized)[None, ...]
