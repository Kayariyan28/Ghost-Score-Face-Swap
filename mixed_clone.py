"""Adaptive cloning mode selector for the final boundary blend.

Pérez et al.'s Poisson image editing offers three operation modes:

  - ``NORMAL_CLONE`` (cv2.NORMAL_CLONE)
        Solves ∇²f = div(g_src) inside the mask. Replaces the destination's
        gradients with the source's. Best when the source REGION has rich
        detail you want to keep and the destination is smooth.

  - ``MIXED_CLONE`` (cv2.MIXED_CLONE)
        Solves ∇²f = div(max(|g_src|, |g_dst|)) — at each pixel, takes
        whichever has the stronger gradient. Best when the destination has
        STRONG features (target shadow lines, motion-blur streaks, hair
        whisps) that would otherwise be over-written by NORMAL_CLONE.

  - ``MONOCHROME_TRANSFER`` — colour-only transfer, not useful here.

For HardPoseReplaceMode we additionally offer a 5-band Laplacian-pyramid
blend for cases where Poisson cloning would over-attenuate source
high-frequency content.

The selector measures the target's gradient energy inside the swap mask:

    ratio = mean|∇_target| / mean|∇_source|

  ratio < 0.7   → NORMAL_CLONE     (source has more detail; replace)
  ratio in [0.7, 1.4]   → MIXED_CLONE      (similar energy; merge gradients)
  ratio > 1.4   → LAPLACIAN_GRAFT  (target has stronger structure; keep
                                    its low-freq, only graft source's
                                    high-freq)
"""

from __future__ import annotations

import cv2
import numpy as np


_NORMAL = 'normal_clone'
_MIXED = 'mixed_clone'
_LAPL = 'laplacian_graft'


def _grad_mag(image: np.ndarray, mask: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    if mask.max() == 0:
        return float(mag.mean())
    return float(mag[mask > 128].mean())


def select_mode(source_img: np.ndarray, target_img: np.ndarray,
                mask: np.ndarray) -> str:
    """Pick the cloning method based on gradient-energy ratio."""
    src_g = _grad_mag(source_img, mask) + 1e-6
    tgt_g = _grad_mag(target_img, mask) + 1e-6
    ratio = tgt_g / src_g
    if ratio < 0.7:
        return _NORMAL
    if ratio < 1.4:
        return _MIXED
    return _LAPL


def _laplacian_pyramid_graft(base: np.ndarray, source: np.ndarray,
                             mask: np.ndarray, levels: int = 5) -> np.ndarray:
    """Multi-scale Laplacian-pyramid blend.

    Builds Gaussian pyramids of (base, source, mask), then Laplacian
    residuals at each level. At each level we use the source's residual
    where mask is high and base's where mask is low. Reconstructs by
    summing residuals up the pyramid. Standard Burt-Adelson approach.
    """
    base_f = base.astype(np.float32)
    src_f = source.astype(np.float32)
    mask_f = (mask.astype(np.float32) / 255.0)[..., None]

    g_base = [base_f]
    g_src = [src_f]
    g_mask = [mask_f]
    for _ in range(levels - 1):
        g_base.append(cv2.pyrDown(g_base[-1]))
        g_src.append(cv2.pyrDown(g_src[-1]))
        g_mask.append(cv2.pyrDown(g_mask[-1]))

    l_base = []
    l_src = []
    for i in range(levels - 1):
        size = (g_base[i].shape[1], g_base[i].shape[0])
        up_b = cv2.pyrUp(g_base[i + 1], dstsize=size)
        up_s = cv2.pyrUp(g_src[i + 1], dstsize=size)
        l_base.append(g_base[i] - up_b)
        l_src.append(g_src[i] - up_s)
    # Lowest band keeps the Gaussian.
    l_base.append(g_base[-1])
    l_src.append(g_src[-1])

    # Blend each band. Broadcast 2D mask to 3 channels if necessary.
    blended = []
    for i in range(levels):
        m = g_mask[i] if i < levels - 1 else g_mask[-1]
        if m.ndim == 2 and l_src[i].ndim == 3:
            m = m[..., None]
        blended.append(l_src[i] * m + l_base[i] * (1.0 - m))

    # Reconstruct.
    out = blended[-1]
    for i in range(levels - 2, -1, -1):
        size = (blended[i].shape[1], blended[i].shape[0])
        out = cv2.pyrUp(out, dstsize=size)
        out = out + blended[i]
    return np.clip(out, 0, 255).astype(np.uint8)


def adaptive_clone(source_img: np.ndarray, target_img: np.ndarray,
                   mask: np.ndarray) -> tuple:
    """Pick a method and perform the clone. Returns (result_image, method_name)."""
    if mask is None or mask.max() == 0:
        return target_img.copy(), 'none'
    method = select_mode(source_img, target_img, mask)
    # Center of the mask for cv2.seamlessClone.
    ys, xs = np.where(mask > 128)
    if ys.size == 0:
        return target_img.copy(), 'none'
    h, w = target_img.shape[:2]
    cx = int(max(1, min(w - 2, (xs.min() + xs.max()) // 2)))
    cy = int(max(1, min(h - 2, (ys.min() + ys.max()) // 2)))

    if method == _LAPL:
        return _laplacian_pyramid_graft(target_img, source_img, mask), _LAPL

    flag = cv2.NORMAL_CLONE if method == _NORMAL else cv2.MIXED_CLONE
    try:
        result = cv2.seamlessClone(source_img, target_img, mask, (cx, cy), flag)
        return result, method
    except cv2.error:
        # Poisson failure (tiny mask, mask at image edge). Fall back to
        # Laplacian-pyramid blend which has no such constraint.
        return _laplacian_pyramid_graft(target_img, source_img, mask), _LAPL
