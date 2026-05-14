"""Advanced 2D warps for the Pro mode: Delaunay piecewise affine + TPS.

Standalone — does NOT modify the existing pipeline. The existing modes use
a single global affine (5-kps similarity or 478-pt 6-DOF affine) which is
fine for similar head pose but breaks down for tilted / yawed / pitched
target faces. These warps give per-triangle / smooth-nonrigid deformation.
"""

from __future__ import annotations

import cv2
import numpy as np


def piecewise_affine_warp(
    source_img: np.ndarray,
    source_lmks: np.ndarray,
    target_lmks: np.ndarray,
    target_shape: tuple,
) -> tuple:
    """Per-triangle affine warp via Delaunay triangulation of the target
    landmarks. Each triangle is warped independently so the face shape
    deforms freely (jaw can move, mouth can open, chin can rotate).

    ``source_lmks`` and ``target_lmks`` must be Nx2 corresponding point arrays.
    Returns (warped_image, coverage_mask) where coverage is the union of all
    successfully-warped target triangles.
    """
    h, w = target_shape[:2]
    source_lmks = np.asarray(source_lmks, dtype=np.float32)
    target_lmks = np.asarray(target_lmks, dtype=np.float32)

    # Clip target to inside image bounds for Subdiv2D.
    tgt_clipped = target_lmks.copy()
    tgt_clipped[:, 0] = np.clip(tgt_clipped[:, 0], 0, w - 1)
    tgt_clipped[:, 1] = np.clip(tgt_clipped[:, 1], 0, h - 1)

    rect = (0, 0, w, h)
    subdiv = cv2.Subdiv2D(rect)
    for p in tgt_clipped:
        try:
            subdiv.insert((float(p[0]), float(p[1])))
        except cv2.error:
            continue
    triangles = subdiv.getTriangleList()

    warped = np.zeros((h, w, 3), dtype=np.uint8)
    coverage = np.zeros((h, w), dtype=np.uint8)

    def nearest_idx(x, y):
        d2 = (tgt_clipped[:, 0] - x) ** 2 + (tgt_clipped[:, 1] - y) ** 2
        i = int(np.argmin(d2))
        return i if d2[i] < 1.5 else -1

    for t in triangles:
        x1, y1, x2, y2, x3, y3 = t
        if min(x1, x2, x3) < 0 or min(y1, y2, y3) < 0:
            continue
        if max(x1, x2, x3) >= w or max(y1, y2, y3) >= h:
            continue
        i1 = nearest_idx(x1, y1)
        i2 = nearest_idx(x2, y2)
        i3 = nearest_idx(x3, y3)
        if -1 in (i1, i2, i3) or len({i1, i2, i3}) < 3:
            continue

        src_tri = np.ascontiguousarray(
            source_lmks[[i1, i2, i3]], dtype=np.float32,
        )
        tgt_tri = np.ascontiguousarray(
            np.array([[x1, y1], [x2, y2], [x3, y3]], dtype=np.float32),
        )

        try:
            M = cv2.getAffineTransform(src_tri, tgt_tri)
        except cv2.error:
            continue

        warped_tri = cv2.warpAffine(
            source_img, M, (w, h),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        tri_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(
            tri_mask, tgt_tri.astype(np.int32), 255, lineType=cv2.LINE_AA,
        )
        m = tri_mask > 0
        warped[m] = warped_tri[m]
        coverage[m] = 255

    return warped, coverage


def tps_warp(
    source_img: np.ndarray,
    source_lmks: np.ndarray,
    target_lmks: np.ndarray,
    target_shape: tuple,
):
    """Thin-Plate-Spline warp from source landmarks to target landmarks.

    Returns (warped_image, coverage_mask) where coverage covers the full
    target rect (TPS warps everywhere). Requires opencv-contrib for
    ``cv2.createThinPlateSplineShapeTransformer`` — if unavailable, falls
    back to ``piecewise_affine_warp``.

    TPS gives smooth non-rigid deformation, ideal for expression mismatch
    (open mouth → closed mouth) and small pose drift. For large pose
    differences (>30°) you need a real 3D method (3DDFA / DECA / FLAME).
    """
    h, w = target_shape[:2]
    try:
        tps = cv2.createThinPlateSplineShapeTransformer()
    except AttributeError:
        return piecewise_affine_warp(
            source_img, source_lmks, target_lmks, target_shape,
        )

    src = np.asarray(source_lmks, dtype=np.float32).reshape(1, -1, 2)
    tgt = np.asarray(target_lmks, dtype=np.float32).reshape(1, -1, 2)
    matches = [cv2.DMatch(i, i, 0) for i in range(src.shape[1])]

    # TPS wants (target → source) for image warping (it inverts internally).
    try:
        tps.estimateTransformation(tgt, src, matches)
        warped = tps.warpImage(source_img)
    except cv2.error:
        return piecewise_affine_warp(
            source_img, source_lmks, target_lmks, target_shape,
        )

    if warped.shape[:2] != (h, w):
        warped = cv2.resize(
            warped, (w, h), interpolation=cv2.INTER_LANCZOS4,
        )
    coverage = np.full((h, w), 255, dtype=np.uint8)
    return warped, coverage
