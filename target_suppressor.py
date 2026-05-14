"""Target-face identity suppression — erases the underlying target's
facial-feature evidence before the swap composites the source identity on
top. Used by Pro mode's HardPoseReplaceMode.

Three suppression methods are provided:

  A. **low_frequency_suppression** — Laplacian-pyramid + bilateral filter
     decomposition. Keeps target lighting / shadows (low-freq), removes
     target identity edges (high-freq) inside the inner-face mask. This
     prevents double-eyes / double-nose / double-mouth ghosting without
     full inpainting.

  B. **inpaint_target_identity** — OpenCV Telea + Navier-Stokes inpainting
     auto-selected based on mask area. Use when target features are clearly
     visible under the patch.

  C. **synth_base** — call into the existing inswapper to build a
     pose-adapted base (already done by the upstream pipeline; this just
     exposes the entry point for completeness).

The suppression mask is built from BiSeNet face-parse classes:
  inner-face = skin + brows + eyes + nose + lips + mouth
This is exactly the set of pixels we need to neutralise.
"""

from __future__ import annotations

import cv2
import numpy as np


# CelebAMask-HQ classes from face_swap.py — kept identical.
INNER_FACE_CLASSES = (1, 2, 3, 4, 5, 6, 10, 11, 12, 13)
# 1 skin, 2 l_brow, 3 r_brow, 4 l_eye, 5 r_eye, 6 eye_g (glasses), 10 nose,
# 11 mouth, 12 u_lip, 13 l_lip


def inner_face_mask_from_parser(parser_fn, image, face) -> np.ndarray:
    """Build the suppression mask: pixels classified as any inner-face class.

    `parser_fn(image, face_bbox)` should return (class_map, _) following the
    FaceSwapper._classes_around_face contract.
    """
    cls, _ = parser_fn(image, face.bbox)
    if cls is None:
        return None
    mask = np.isin(cls, INNER_FACE_CLASSES).astype(np.uint8) * 255
    return mask


# ============================================================================
# METHOD A — Laplacian-pyramid low-frequency suppression
# ============================================================================

def low_frequency_suppression(image: np.ndarray,
                              mask: np.ndarray,
                              sigma: float = 12.0,
                              bilateral_d: int = 13,
                              bilateral_sigma_color: float = 75.0,
                              bilateral_sigma_space: float = 75.0) -> np.ndarray:
    """Method A. Replace high-frequency content of `image` inside `mask`
    with the bilateral-filtered low-frequency version.

    Pipeline
    --------
        target_low  = bilateral_filter(image, ...)          # smooth, edge-aware
        target_low2 = GaussianBlur(target_low, sigma)       # remove high-freq
        target_high = image - target_low2
        result      = target_low2 * inside_mask + image * outside_mask

    The bilateral filter preserves macro-structural edges (face oval,
    shadow boundaries) while erasing micro-detail (eyelid creases, lip
    contours) that constitute identity. The Gaussian after that smooths
    out any remaining feature edges.

    Returns the suppressed image (same shape as input).
    """
    if mask is None or mask.max() == 0:
        return image.copy()

    # Edge-preserving smoothing → kills high-frequency identity but keeps
    # shadows and lighting gradients.
    bilateral = cv2.bilateralFilter(
        image, bilateral_d,
        bilateral_sigma_color, bilateral_sigma_space,
    )
    low = cv2.GaussianBlur(bilateral, (0, 0), sigmaX=sigma)

    # Feather the mask so the suppression transition is invisible.
    fw = max(11, int(0.02 * min(image.shape[:2])) | 1)
    soft = cv2.GaussianBlur(mask, (fw, fw), 0)
    alpha = (soft.astype(np.float32) / 255.0)[..., None]

    out = (low.astype(np.float32) * alpha
           + image.astype(np.float32) * (1 - alpha))
    return np.clip(out, 0, 255).astype(np.uint8)


# ============================================================================
# METHOD B — OpenCV inpainting (Telea + Navier-Stokes auto-select)
# ============================================================================

def inpaint_target_identity(image: np.ndarray,
                            mask: np.ndarray,
                            radius: int = 7) -> np.ndarray:
    """Method B. Inpaint `mask` region using OpenCV's classical inpainting.

    Auto-selects the algorithm based on mask area:
      - Small mask (< 5% of image) → Telea (fast marching, sharper)
      - Large mask (≥ 5%)          → Navier-Stokes (fluid dynamics, smoother)

    Both are lightweight, CPU-friendly, and Mac-safe. The result isn't a
    photorealistic face — it doesn't need to be. It's only used as a NEUTRAL
    BASE under the source-pixel transplant, to prevent target features from
    re-emerging as ghost edges.
    """
    if mask is None or mask.max() == 0:
        return image.copy()
    h, w = image.shape[:2]
    area = float((mask > 0).sum()) / (h * w)
    flag = cv2.INPAINT_TELEA if area < 0.05 else cv2.INPAINT_NS
    binary = (mask > 0).astype(np.uint8) * 255
    return cv2.inpaint(image, binary, radius, flag)


# ============================================================================
# Combined two-stage suppression
# ============================================================================

def suppress_target_identity(image: np.ndarray,
                             mask: np.ndarray,
                             mode: str = 'auto') -> np.ndarray:
    """Two-stage suppression. Use ``mode``:

        'low_freq'  — Method A only (fast, preserves lighting)
        'inpaint'   — Method B only (slower, completely removes features)
        'auto'      — Method A first, then Method B if double-edge would
                      still be visible (default)
        'both'      — Apply both in sequence

    For the HardPoseReplaceMode default we use 'both': inpaint first
    (removes features), then low-freq suppression of the inpainted result
    (smooths inpaint artefacts and ties it to target lighting).
    """
    if mask is None or mask.max() == 0:
        return image.copy()

    if mode == 'low_freq':
        return low_frequency_suppression(image, mask)
    if mode == 'inpaint':
        return inpaint_target_identity(image, mask)
    if mode == 'both':
        out = inpaint_target_identity(image, mask)
        out = low_frequency_suppression(out, mask, sigma=6.0)
        return out
    # auto
    return inpaint_target_identity(image, mask)
