"""Quantitative quality metrics for a face swap result.

For every swap we compute:
  - ArcFace cosine similarity between the SOURCE face and the SWAPPED face
    in the result. > 0.6 = same person (high confidence). > 0.4 = same
    person (moderate). < 0.3 = identity drift.
  - SSIM between the result's face crop and the warped source face crop.
  - PSNR (peak signal-to-noise) for the face crop.
  - CIEDE2000 (ΔE00) mean color difference under the face skin mask —
    measures how close the result's color is to the source's.
  - Landmark RMSE — how far each MediaPipe FaceMesh landmark is from where
    we'd expect it to be (compared to target geometry, in normalised face
    width units). Indicates alignment quality.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import cv2
import numpy as np

try:
    from skimage.metrics import structural_similarity as _ssim
    from skimage.color import rgb2lab, deltaE_ciede2000
    _SKIMAGE_OK = True
except Exception:
    _SKIMAGE_OK = False


@dataclass
class QualityScores:
    arcface_cosine: float          # source vs result identity, [-1, 1]
    arcface_verdict: str           # 'match' / 'partial' / 'drift'
    ssim: float                    # source face crop vs result face crop, [0, 1]
    psnr: float                    # dB
    delta_e_mean: float            # CIEDE2000 mean, lower = closer color
    landmark_rmse_norm: float      # normalised face-width units
    elapsed_ms: int

    def to_dict(self):
        """JSON-safe: NaN / Infinity become None."""
        d = asdict(self)
        out = {}
        for k, v in d.items():
            if isinstance(v, float):
                if np.isnan(v) or np.isinf(v):
                    out[k] = None
                    continue
            out[k] = v
        return out


def _crop_face(img, face, pad=0.20):
    """Crop face region from img with given padding ratio. Returns crop + bbox."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = face.bbox.astype(int)
    fw, fh = x2 - x1, y2 - y1
    cx1 = max(0, int(x1 - fw * pad))
    cy1 = max(0, int(y1 - fh * pad))
    cx2 = min(w, int(x2 + fw * pad))
    cy2 = min(h, int(y2 + fh * pad))
    if cx2 <= cx1 or cy2 <= cy1:
        return None, (0, 0, 0, 0)
    return img[cy1:cy2, cx1:cx2], (cx1, cy1, cx2, cy2)


def cosine_similarity(a, b):
    a = a.astype(np.float32).flatten()
    b = b.astype(np.float32).flatten()
    na = np.linalg.norm(a) + 1e-6
    nb = np.linalg.norm(b) + 1e-6
    return float(np.dot(a / na, b / nb))


def arcface_verdict_from_cosine(c):
    if c >= 0.6:
        return 'match'
    if c >= 0.4:
        return 'partial'
    return 'drift'


def compute_metrics(face_app, source_img, result_img, target_face,
                    dense_landmarks_fn=None, source_face=None,
                    elapsed_ms=0) -> QualityScores:
    """Compute all metrics in one call. ``face_app`` is the InsightFace
    FaceAnalysis. ``dense_landmarks_fn`` is optional (returns Nx2 float32
    landmarks via MediaPipe). ``target_face`` is the (rescaled) face object
    detected in the *result* image. ``source_face`` is the face object in the
    source image (already detected); if None we re-detect."""
    # ---- ArcFace cosine identity ----
    if source_face is None:
        s_faces = face_app.get(source_img)
        if not s_faces:
            arc = float('nan')
        else:
            source_face = max(
                s_faces,
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            )
    arc = float('nan')
    if source_face is not None and target_face is not None:
        try:
            arc = cosine_similarity(
                source_face.normed_embedding, target_face.normed_embedding,
            )
        except Exception:
            arc = float('nan')
    verdict = arcface_verdict_from_cosine(arc) if not np.isnan(arc) else 'unknown'

    # ---- SSIM + PSNR over a matched face crop ----
    ssim_val = float('nan')
    psnr_val = float('nan')
    delta_e = float('nan')
    if source_face is not None and target_face is not None:
        src_crop, _ = _crop_face(source_img, source_face)
        res_crop, _ = _crop_face(result_img, target_face)
        if src_crop is not None and res_crop is not None \
                and src_crop.size > 0 and res_crop.size > 0:
            # Resize both to a common 256×256 for comparison.
            try:
                s256 = cv2.resize(src_crop, (256, 256), interpolation=cv2.INTER_AREA)
                r256 = cv2.resize(res_crop, (256, 256), interpolation=cv2.INTER_AREA)
                if _SKIMAGE_OK:
                    s_gray = cv2.cvtColor(s256, cv2.COLOR_BGR2GRAY)
                    r_gray = cv2.cvtColor(r256, cv2.COLOR_BGR2GRAY)
                    ssim_val = float(_ssim(s_gray, r_gray, data_range=255))
                mse = float(((s256.astype(np.float32) - r256.astype(np.float32)) ** 2).mean())
                if mse > 0:
                    psnr_val = float(10.0 * np.log10(255.0 * 255.0 / mse))
                else:
                    psnr_val = float('inf')
                if _SKIMAGE_OK:
                    s_lab = rgb2lab(cv2.cvtColor(s256, cv2.COLOR_BGR2RGB) / 255.0)
                    r_lab = rgb2lab(cv2.cvtColor(r256, cv2.COLOR_BGR2RGB) / 255.0)
                    delta_e = float(deltaE_ciede2000(s_lab, r_lab).mean())
            except Exception:
                pass

    # ---- Landmark RMSE between source and result face geometries ----
    lm_rmse = float('nan')
    if dense_landmarks_fn is not None and source_face is not None and target_face is not None:
        try:
            src_lm = dense_landmarks_fn(source_img, source_face.bbox)
            res_lm = dense_landmarks_fn(result_img, target_face.bbox)
            if src_lm is not None and res_lm is not None and len(src_lm) == len(res_lm):
                # Normalise both to a unit-width face frame, then compare positions.
                # Subtract centroid, scale by face bbox width, then RMSE.
                def normalise(pts, face):
                    fw = face.bbox[2] - face.bbox[0]
                    cx = (face.bbox[0] + face.bbox[2]) / 2.0
                    cy = (face.bbox[1] + face.bbox[3]) / 2.0
                    return (pts - np.array([cx, cy])) / max(fw, 1)
                a = normalise(src_lm, source_face)
                b = normalise(res_lm, target_face)
                lm_rmse = float(np.sqrt(((a - b) ** 2).sum(axis=1).mean()))
        except Exception:
            pass

    return QualityScores(
        arcface_cosine=float(round(arc, 4)) if not np.isnan(arc) else float('nan'),
        arcface_verdict=verdict,
        ssim=float(round(ssim_val, 4)) if not np.isnan(ssim_val) else float('nan'),
        psnr=float(round(psnr_val, 2)) if not np.isnan(psnr_val) and psnr_val != float('inf') else psnr_val,
        delta_e_mean=float(round(delta_e, 3)) if not np.isnan(delta_e) else float('nan'),
        landmark_rmse_norm=float(round(lm_rmse, 5)) if not np.isnan(lm_rmse) else float('nan'),
        elapsed_ms=int(elapsed_ms),
    )
