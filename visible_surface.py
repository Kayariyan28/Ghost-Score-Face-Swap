"""Z-buffered visible-surface mask using 3DDFA_V2's dense 3D face mesh.

For an extreme yaw or pitch target, only ~half of the face is geometrically
visible. Pasting a full source face onto it produces the "face sticker on
side view" artifact — pixels appear over parts of the face that physically
don't exist in the target pose. This module fixes that by:

  1. Fitting 3DDFA_V2's dense 3D face mesh (~38 000 vertices) to the target
     face using its official mobilenet-V1 120×120 checkpoint.
  2. Computing per-triangle normals.
  3. Rejecting triangles whose normal points AWAY from the camera (the
     z-buffer / back-face cull).
  4. Rasterising the surviving triangles into a binary mask in target-image
     coordinates.

The resulting `visible_mask` is intersected with BiSeNet skin and any
occluder masks to produce the final blend mask. Source pixels are only
deposited where the target face is BOTH skin AND camera-facing AND
unoccluded.

If 3DDFA_V2's checkpoint isn't present we fall back to insightface's
1k3d68 sparse 3D landmarks: project to 2D, take the convex hull of the
positive-z (front-facing) subset. Coarser but still functional.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


_3DDFA_DIR = Path(__file__).resolve().parent / 'models' / 'pro' / '3ddfa_v2'
_3DDFA_SRC = Path(__file__).resolve().parent / 'models' / 'pro' / '3ddfa_v2_src'


class _DDFASession:
    """Wraps the official 3DDFA_V2 inference. Lazy-loaded on first call."""

    _instance = None

    @classmethod
    def get(cls):
        if cls._instance is not None:
            return cls._instance
        try:
            cls._instance = cls()
        except Exception as e:
            print(f'[visible_surface] 3DDFA_V2 unavailable ({e}); '
                  f'using insightface 1k3d68 fallback')
            cls._instance = False  # marker so we don't retry every call
        return cls._instance if cls._instance else None

    def __init__(self):
        # Make the cloned 3DDFA_V2 source importable.
        if str(_3DDFA_SRC) not in sys.path:
            sys.path.insert(0, str(_3DDFA_SRC))
        # 3DDFA_V2 uses np.long (removed in numpy >= 1.20) and np.int. Shim
        # them so the cloned source imports cleanly.
        if not hasattr(np, 'long'):
            np.long = int  # type: ignore[attr-defined]
        if not hasattr(np, 'int'):
            np.int = int  # type: ignore[attr-defined]
        if not hasattr(np, 'float'):
            np.float = float  # type: ignore[attr-defined]
        if not hasattr(np, 'bool'):
            np.bool = bool  # type: ignore[attr-defined]
        # Use the PyTorch TDDFA, not the ONNX one — official repo's TDDFA
        # class wraps the MobileNet param-regressor + BFM reconstruction.
        from TDDFA import TDDFA
        cfg = {
            'arch': 'mobilenet',
            'widen_factor': 1.0,
            'checkpoint_fp': str(_3DDFA_DIR / 'mb1_120x120.pth'),
            'bfm_fp': str(_3DDFA_DIR / 'bfm_noneck_v3.pkl'),
            'size': 120,
            'num_params': 62,
        }
        # 3DDFA_V2 looks up config paths via relative os.path lookups; chdir
        # to the cloned src dir so its internal pkl/npy lookups resolve.
        old_cwd = os.getcwd()
        try:
            os.chdir(str(_3DDFA_SRC))
            self.tddfa = TDDFA(**cfg)
        finally:
            os.chdir(old_cwd)
        # Precompute triangle index array (constant across all faces).
        import pickle
        with open(_3DDFA_DIR / 'tri.pkl', 'rb') as f:
            tri = pickle.load(f)  # (3, n_tri) int32 — vertex indices
        self.tri = np.asarray(tri).T.astype(np.int32)  # (n_tri, 3)

    def fit_vertices(self, image_bgr: np.ndarray, face_bbox) -> np.ndarray:
        """Fit the BFM mesh to the face. Returns vertices Nx3 in image
        coordinates (x, y, z) where z is depth (smaller = closer)."""
        x1, y1, x2, y2 = face_bbox.astype(int)
        # 3DDFA expects boxes in [x1, y1, x2, y2] format.
        param_lst, roi_box_lst = self.tddfa(image_bgr, [[x1, y1, x2, y2]])
        ver_lst = self.tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)
        verts = ver_lst[0].T  # (n_verts, 3)
        return verts


def _visible_mask_from_mesh(verts: np.ndarray, tri: np.ndarray,
                            image_shape) -> np.ndarray:
    """Build a visible-surface mask from a 3D mesh fit.

    The mesh vertices live in IMAGE coordinates (x, y) with z = depth
    (3DDFA_V2 returns it this way). Back-face culling: a triangle is
    camera-facing iff its 2D winding is counter-clockwise in (x, y), which
    on a Y-down image plane corresponds to the normal pointing toward the
    camera. Rasterise the surviving triangles.
    """
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    # Vectorised triangle winding test:
    #   (b - a) × (c - a) > 0  ⇒  CCW in image-y-down  ⇒  front-facing
    a = verts[tri[:, 0], :2]
    b = verts[tri[:, 1], :2]
    c = verts[tri[:, 2], :2]
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) \
          - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    visible_tri = tri[cross > 0]
    # Rasterise. For ~38k triangles this is the slow loop, but it's a
    # one-shot per swap and Python's overhead is the bottleneck, not the
    # fillConvexPoly call itself.
    pts_a = verts[visible_tri[:, 0], :2].astype(np.int32)
    pts_b = verts[visible_tri[:, 1], :2].astype(np.int32)
    pts_c = verts[visible_tri[:, 2], :2].astype(np.int32)
    for ai, bi, ci in zip(pts_a, pts_b, pts_c):
        cv2.fillConvexPoly(mask, np.array([ai, bi, ci]), 255, lineType=cv2.LINE_AA)
    return mask


def _fallback_from_1k3d68(face) -> Optional[np.ndarray]:
    """Coarse fallback when 3DDFA_V2 isn't available. Uses insightface's
    68-point 3D landmarks (already loaded as part of buffalo_l): project
    to 2D, keep landmarks whose z is in the front half (z < median), and
    take the convex hull. This produces a rough visible-region mask."""
    pts = getattr(face, 'landmark_3d_68', None)
    if pts is None:
        return None
    pts = np.asarray(pts)
    if pts.shape[-1] < 3:
        return None
    median_z = float(np.median(pts[:, 2]))
    front = pts[pts[:, 2] <= median_z + 1e-3]
    return cv2.convexHull(front[:, :2].astype(np.int32))


def compute_visible_mask(image: np.ndarray, face) -> Optional[np.ndarray]:
    """Return a uint8 0/255 mask of the camera-facing face surface in the
    given image. Tries 3DDFA_V2 first, falls back to 1k3d68 + convex hull."""
    sess = _DDFASession.get()
    h, w = image.shape[:2]
    if sess is not None and sess is not False:
        try:
            verts = sess.fit_vertices(image, face.bbox)
            return _visible_mask_from_mesh(verts, sess.tri, image.shape)
        except Exception as e:
            print(f'[visible_surface] 3DDFA_V2 fit failed ({e}); fallback')

    hull = _fallback_from_1k3d68(face)
    if hull is None:
        return None
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    return mask
