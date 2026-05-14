"""Blur / noise / motion matching — degrade the warped source to match the
target's local imaging characteristics before blending.

A pin-sharp source face on a motion-blurred or noisy target looks like an
overlay even if geometry is perfect. We measure the target face region's:

  - Laplacian variance (blur amount; lower variance = blurrier)
  - FFT directional energy (motion-blur orientation, if any)
  - Local noise σ via Median-Absolute-Deviation on high-pass residual

Then apply matched degradation to the source projection:

  - Gaussian blur with σ chosen so source Laplacian variance ≈ target
  - Directional motion-blur kernel along the detected orientation
  - Additive Gaussian noise at target σ
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ImagingProfile:
    laplacian_variance: float    # blur metric (higher = sharper)
    noise_sigma: float           # estimated per-channel noise σ in [0..255]
    motion_angle_deg: float      # 0..180, NaN if no clear motion blur
    motion_strength: float       # 0..1, ratio of motion-band energy to total


def _region_crop(image: np.ndarray, face) -> np.ndarray:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = face.bbox.astype(int)
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)
    return image[y1:y2, x1:x2]


def _laplacian_variance(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _noise_sigma(gray: np.ndarray) -> float:
    """Noise σ via Donoho's median-absolute-deviation on wavelet HH band
    (approximated here by a 3×3 Laplacian high-pass)."""
    high = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    mad = float(np.median(np.abs(high - np.median(high))))
    return mad / 0.6745  # Donoho scaling


def _motion_blur_orientation(gray: np.ndarray) -> tuple:
    """Estimate motion-blur orientation via FFT magnitude spectrum.

    A motion-blurred image has a SINGLE elongated bright stripe through
    the origin of its 2D FFT spectrum, perpendicular to the motion
    direction. We extract that orientation via the structure tensor of
    the log-magnitude spectrum.

    Returns (angle_deg in [0, 180], strength in [0, 1])."""
    if gray.shape[0] < 32 or gray.shape[1] < 32:
        return float('nan'), 0.0
    h, w = gray.shape
    # Pad to next power of 2 for stable FFT.
    nh = int(2 ** np.ceil(np.log2(h)))
    nw = int(2 ** np.ceil(np.log2(w)))
    img = np.zeros((nh, nw), dtype=np.float32)
    img[:h, :w] = gray.astype(np.float32)
    img -= img.mean()
    F = np.fft.fftshift(np.fft.fft2(img))
    # FFT output is complex128 → abs/log1p give float64. Sobel can't go from
    # float64 source to CV_32F destination on this build; downcast explicitly.
    mag = np.log1p(np.abs(F)).astype(np.float32)
    # Structure tensor on log-magnitude.
    gx = cv2.Sobel(mag, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(mag, cv2.CV_32F, 0, 1)
    Jxx = float((gx * gx).sum())
    Jyy = float((gy * gy).sum())
    Jxy = float((gx * gy).sum())
    # Dominant gradient direction → motion line is perpendicular.
    theta = 0.5 * np.arctan2(2 * Jxy, Jxx - Jyy)
    angle_deg = float(np.degrees(theta) + 90.0) % 180.0
    # Strength: anisotropy of structure tensor.
    trace = Jxx + Jyy
    det = Jxx * Jyy - Jxy * Jxy
    if trace < 1e-3:
        return float('nan'), 0.0
    aniso = (trace ** 2 - 4 * det) / max(trace ** 2, 1e-6)
    return angle_deg, float(np.clip(aniso, 0.0, 1.0))


def estimate_profile(image: np.ndarray, face) -> ImagingProfile:
    """Estimate a face-region imaging profile."""
    crop = _region_crop(image, face)
    if crop.size == 0:
        return ImagingProfile(0.0, 0.0, float('nan'), 0.0)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    lv = _laplacian_variance(gray)
    nz = _noise_sigma(gray)
    ang, st = _motion_blur_orientation(gray)
    return ImagingProfile(lv, nz, ang, st)


def _motion_kernel(length: int, angle_deg: float) -> np.ndarray:
    """Build a directional motion-blur kernel of given length & angle."""
    length = max(3, int(length))
    if length % 2 == 0:
        length += 1
    k = np.zeros((length, length), dtype=np.float32)
    k[length // 2, :] = 1.0 / length
    M = cv2.getRotationMatrix2D((length / 2, length / 2), angle_deg, 1.0)
    k = cv2.warpAffine(k, M, (length, length))
    k /= max(k.sum(), 1e-6)
    return k


def match_to_target(source_warp: np.ndarray,
                    target_profile: ImagingProfile,
                    source_profile: ImagingProfile,
                    apply_noise: bool = True) -> np.ndarray:
    """Apply matched degradation to ``source_warp`` so its imaging
    characteristics resemble those of the target.

    Steps
    -----
    1. If target Laplacian variance < source's, apply Gaussian blur with σ
       chosen empirically: σ ≈ 0.5 + 0.6 * (ratio - 1) where ratio is the
       sqrt of (source_var / target_var).
    2. If target shows strong directional motion (strength > 0.55), apply
       a directional motion kernel along that angle.
    3. If target noise σ exceeds source's, add matched Gaussian noise.
    """
    out = source_warp.copy()

    # Gaussian-blur match. Cap σ at 1.5 so we never destroy source identity
    # — heavy blur on the source projection collapses the ArcFace embedding.
    if (target_profile.laplacian_variance > 0
            and source_profile.laplacian_variance > 2.0 * target_profile.laplacian_variance):
        ratio = float(np.sqrt(
            source_profile.laplacian_variance
            / max(target_profile.laplacian_variance, 1e-3),
        ))
        sigma = float(np.clip(0.5 + 0.4 * (ratio - 1.0), 0.5, 1.5))
        out = cv2.GaussianBlur(out, (0, 0), sigmaX=sigma)

    # Motion-blur match. cv2.filter2D needs the destination depth to match
    # the source's depth class. Use uint8 to uint8 explicitly so we don't
    # trip on a CV_32F kernel + CV_8U buffer mismatch.
    if (not np.isnan(target_profile.motion_angle_deg)
            and target_profile.motion_strength > 0.55):
        length = int(np.clip(
            3 + (target_profile.motion_strength - 0.55) * 18, 3, 15,
        ))
        kernel = _motion_kernel(length, target_profile.motion_angle_deg)
        out = cv2.filter2D(out, cv2.CV_8U, kernel)

    # Noise match.
    if apply_noise and target_profile.noise_sigma > source_profile.noise_sigma:
        delta = float(np.sqrt(max(
            target_profile.noise_sigma ** 2 - source_profile.noise_sigma ** 2,
            0.0,
        )))
        if delta > 0.5:
            noise = np.random.normal(0.0, delta, out.shape).astype(np.float32)
            out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return out
