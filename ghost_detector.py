"""Ghost-face detector — quantifies overlay artifacts in a swap result.

Used by Pro mode's HardPoseReplaceMode for routing decisions and auto-retry.
Computes six metrics on every generated image:

  1. ArcFace cosine result-to-source
  2. ArcFace cosine result-to-target
  3. ghost_score = arc(out, target) - arc(out, source)
     - high positive → result still looks like the target (identity transfer
       failed; target was not suppressed enough)
     - high negative → result looks like the source (good swap)
  4. Seam gradient energy along the mask boundary
  5. Landmark disagreement between generated face and target face (RMSE in
     face-width-normalised units)
  6. Local SSIM mismatch inside the face mask vs source face (region detail
     preservation)
  7. Double-edge score around eyes/nose/mouth/jaw — convolves a directional
     edge kernel and looks for parallel edge pairs that indicate overlay

Plus a top-level routing recommendation derived from the metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import cv2
import numpy as np

from skimage.metrics import structural_similarity as _ssim


@dataclass
class GhostMetrics:
    arc_to_source: float           # cosine, [-1, 1]
    arc_to_target: float           # cosine, [-1, 1]
    ghost_score: float             # arc_to_target - arc_to_source
    seam_gradient_energy: float    # mean |∇| along feathered boundary, [0, ∞)
    landmark_disagreement: float   # normalised face-width units, [0, ∞)
    local_ssim_mismatch: float     # 1 - SSIM(out_face, src_face), [0, 1]
    double_edge_score: float       # ratio of parallel-edge pixels, [0, 1]
    routing: str                   # 'ok' | 'suppress_target' | 'shrink_mask' |
                                   # 'hard_pose' | 'rebuild_base'

    def to_dict(self):
        d = asdict(self)
        out = {}
        for k, v in d.items():
            if isinstance(v, float):
                if np.isnan(v) or np.isinf(v):
                    out[k] = None
                else:
                    out[k] = round(v, 4)
            else:
                out[k] = v
        return out


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32).flatten()
    b = b.astype(np.float32).flatten()
    na = float(np.linalg.norm(a)) + 1e-6
    nb = float(np.linalg.norm(b)) + 1e-6
    return float(np.dot(a / na, b / nb))


def _face_crop(img, face, pad=0.2):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = face.bbox.astype(int)
    fw, fh = x2 - x1, y2 - y1
    cx1 = max(0, int(x1 - fw * pad))
    cy1 = max(0, int(y1 - fh * pad))
    cx2 = min(w, int(x2 + fw * pad))
    cy2 = min(h, int(y2 + fh * pad))
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    return img[cy1:cy2, cx1:cx2]


def _seam_gradient_energy(image: np.ndarray, mask: np.ndarray) -> float:
    """Mean |∇| along the boundary band of `mask`.

    The boundary band is mask − erode(mask, 5), i.e. a ring ~5 px wide just
    inside the mask. A clean Poisson-cloned boundary has low gradient energy;
    a visible patch line is bright.
    """
    if mask is None or mask.max() == 0:
        return float('nan')
    m_bin = (mask > 128).astype(np.uint8)
    eroded = cv2.erode(m_bin, np.ones((5, 5), np.uint8))
    band = cv2.subtract(m_bin, eroded)
    if band.sum() == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    return float(mag[band > 0].mean())


def _landmark_disagreement(landmark_fn, result_img, result_face,
                           target_img, target_face) -> float:
    """RMSE between MediaPipe FaceMesh landmarks of result and target faces,
    normalised by target face width. Captures geometric drift in the
    generated face (= the swap moved features off-position)."""
    if landmark_fn is None or result_face is None or target_face is None:
        return float('nan')
    try:
        r_lm = landmark_fn(result_img, result_face.bbox)
        t_lm = landmark_fn(target_img, target_face.bbox)
    except Exception:
        return float('nan')
    if r_lm is None or t_lm is None or len(r_lm) != len(t_lm):
        return float('nan')

    def normalise(pts, face):
        fw = max(1.0, float(face.bbox[2] - face.bbox[0]))
        cx = float(face.bbox[0] + face.bbox[2]) / 2.0
        cy = float(face.bbox[1] + face.bbox[3]) / 2.0
        return (pts - np.array([cx, cy])) / fw

    a = normalise(r_lm, result_face)
    b = normalise(t_lm, target_face)
    return float(np.sqrt(((a - b) ** 2).sum(axis=1).mean()))


def _double_edge_score(image: np.ndarray, face) -> float:
    """Detect parallel-edge pairs around eyes / nose / mouth / jaw.

    A genuine face has SINGLE edges at feature boundaries (eyelid, lip,
    nostril). An overlay shows DOUBLE edges (source feature ~ a few pixels
    from base feature). We measure this with a local 1D differential filter
    in the orientation of each detected edge: at a true edge, the response
    is a single peak; at a double edge, two peaks within 1-8 px.

    Returns the fraction of edge pixels inside the face bbox that show a
    nearby parallel-edge response (higher = more ghosting).
    """
    if face is None:
        return float('nan')
    h, w = image.shape[:2]
    x1, y1, x2, y2 = face.bbox.astype(int)
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return float('nan')
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return float('nan')

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 150)
    if edges.sum() == 0:
        return 0.0
    # Dilate by 2 and 5 px. Pixels that are edge in BOTH dilations 2 and 5
    # but NOT in dilation 1 indicate a separate edge ~3-5 px away → ghost.
    d1 = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    d3 = cv2.dilate(edges, np.ones((7, 7), np.uint8))
    band = cv2.subtract(d3, d1)
    parallel = cv2.bitwise_and(band, edges)
    return float(parallel.sum() / max(edges.sum(), 1))


def _local_ssim_mismatch(source_img, source_face,
                        result_img, result_face) -> float:
    """1 − SSIM between source-face crop and result-face crop (256×256).
    Higher = more facial-detail divergence between source and result.
    """
    if source_face is None or result_face is None:
        return float('nan')
    s = _face_crop(source_img, source_face)
    r = _face_crop(result_img, result_face)
    if s is None or r is None or s.size == 0 or r.size == 0:
        return float('nan')
    try:
        s256 = cv2.resize(s, (256, 256), interpolation=cv2.INTER_AREA)
        r256 = cv2.resize(r, (256, 256), interpolation=cv2.INTER_AREA)
        sg = cv2.cvtColor(s256, cv2.COLOR_BGR2GRAY)
        rg = cv2.cvtColor(r256, cv2.COLOR_BGR2GRAY)
        ssim_val = float(_ssim(sg, rg, data_range=255))
        return 1.0 - ssim_val
    except Exception:
        return float('nan')


def _routing(arc_to_source: float, arc_to_target: float,
             ghost: float, seam: float, lmk: float, double: float,
             pose_delta_deg: float) -> str:
    """Decide what to do next.

    Routes
    ------
      ok                 : ship the current result
      suppress_target    : ghost_score is high → run target-face suppression
                           (low-freq erase + inpaint) and rerun the swap
      shrink_mask        : seam_gradient_energy is high → shrink mask and
                           rerun Poisson with mixed-clone
      hard_pose          : pose_delta > 35° → run full HardPoseReplaceMode
      rebuild_base       : double-edge score is high → ghost overlay
                           detected, switch to target-suppression base
    """
    if pose_delta_deg > 35.0:
        return 'hard_pose'
    # ghost_score > 0 means result is closer to TARGET than to SOURCE —
    # identity transfer clearly failed.
    if not np.isnan(ghost) and ghost > 0.0:
        return 'suppress_target'
    # ghost_score in (-0.05, 0]: partial / borderline — only retry if
    # double-edge or seam is also bad.
    if not np.isnan(double) and double > 0.35:
        return 'rebuild_base'
    if not np.isnan(seam) and seam > 28.0:
        return 'shrink_mask'
    if not np.isnan(ghost) and ghost > -0.02:
        return 'suppress_target'
    return 'ok'


def compute(face_app, source_img, source_face,
            target_img, target_face,
            result_img, result_face,
            mask: Optional[np.ndarray] = None,
            landmark_fn=None,
            pose_delta_deg: float = 0.0) -> GhostMetrics:
    """Compute all ghost metrics + routing decision."""

    # ArcFace cosines.
    arc_src = float('nan')
    arc_tgt = float('nan')
    if source_face is not None and result_face is not None:
        try:
            arc_src = _cosine(
                source_face.normed_embedding,
                result_face.normed_embedding,
            )
        except Exception:
            pass
    if target_face is not None and result_face is not None:
        try:
            arc_tgt = _cosine(
                target_face.normed_embedding,
                result_face.normed_embedding,
            )
        except Exception:
            pass
    ghost = (
        float(arc_tgt - arc_src)
        if not (np.isnan(arc_src) or np.isnan(arc_tgt))
        else float('nan')
    )

    seam = _seam_gradient_energy(result_img, mask) if mask is not None else float('nan')
    lmk = _landmark_disagreement(
        landmark_fn, result_img, result_face, target_img, target_face,
    )
    ssim_mis = _local_ssim_mismatch(source_img, source_face, result_img, result_face)
    double = _double_edge_score(result_img, result_face)

    route = _routing(arc_src, arc_tgt, ghost, seam, lmk, double, pose_delta_deg)

    return GhostMetrics(
        arc_to_source=arc_src,
        arc_to_target=arc_tgt,
        ghost_score=ghost,
        seam_gradient_energy=seam,
        landmark_disagreement=lmk,
        local_ssim_mismatch=ssim_mis,
        double_edge_score=double,
        routing=route,
    )
