"""Region-wise compositing — each facial region blends source and base
under a different rule, as specified by the HardPoseReplaceMode spec.

A single oval face mask produces overlay artifacts (every pixel uses the
same alpha). Faces have semantically different regions that demand
different blending rules:

    eyes        → source identity priority, high detail, tight mask
    brows       → source detail + target lighting
    nose        → 3D projection priority
    cheeks      → source texture + target lighting
    lips        → expression-aware, conservative
    teeth/mouth → preserve target geometry
    jawline     → target shadow/shape priority
    forehead    → source texture unless hair occludes
    ears        → target priority (especially in side view)

This module produces a per-pixel WEIGHT MAP for the source image in each
region. The caller then composites:

    result = source_warp * w_source + base * (1 - w_source)
"""

from __future__ import annotations

from typing import Dict, Optional

import cv2
import numpy as np


# CelebAMask-HQ class index → (source_weight, region_label)
REGION_RULES: Dict[int, tuple] = {
    1:  (0.88, 'skin'),         # cheek + forehead skin
    2:  (1.00, 'l_brow'),
    3:  (1.00, 'r_brow'),
    4:  (1.00, 'l_eye'),
    5:  (1.00, 'r_eye'),
    6:  (0.00, 'eye_g'),        # eyeglasses — target priority
    7:  (0.10, 'l_ear'),        # ear — target priority
    8:  (0.10, 'r_ear'),
    10: (0.95, 'nose'),
    11: (0.15, 'mouth'),        # inner mouth — preserve target
    12: (1.00, 'u_lip'),
    13: (1.00, 'l_lip'),
    14: (0.00, 'neck'),         # not a face region
    17: (0.00, 'hair'),         # never paint over hair
    18: (0.00, 'hat'),
}


def build_region_alpha(class_map: np.ndarray,
                       face_bbox: Optional[np.ndarray] = None,
                       extra_visibility_mask: Optional[np.ndarray] = None,
                       extra_occluder_mask: Optional[np.ndarray] = None,
                       ) -> Dict[str, np.ndarray]:
    """Build a multi-region alpha map.

    Parameters
    ----------
    class_map : H×W uint8 from BiSeNet
    face_bbox : optional [x1, y1, x2, y2] for jawline region (BiSeNet doesn't
                emit a 'jaw' class, so we derive it from the lower portion
                of the skin class within the face bbox)
    extra_visibility_mask : optional H×W uint8 from 3DDFA_V2 visible surface.
                When provided, source weight is zeroed where the surface is
                self-occluded.
    extra_occluder_mask : optional H×W uint8 from SAM 2 / Depth-Anything.
                When provided, source weight is zeroed where a foreground
                object covers the face.

    Returns
    -------
    dict with keys:
        'alpha'        : H×W float32 in [0, 1] — final per-pixel source weight
        'per_region'   : dict[region_label, H×W float32 alpha] for debugging
        'jawline'      : H×W uint8 mask of the jawline band (target priority)
    """
    h, w = class_map.shape[:2]
    alpha = np.zeros((h, w), dtype=np.float32)
    per_region = {}

    for cls_idx, (weight, label) in REGION_RULES.items():
        region = (class_map == cls_idx)
        if not region.any():
            continue
        per_region[label] = region.astype(np.float32) * weight
        alpha[region] = np.maximum(alpha[region], weight)

    # Jawline band: lower 30% of the skin region inside the face bbox,
    # weighted toward target (because the jawline carries the target's face
    # shape silhouette; pasting source jaw onto a wider/narrower target
    # leaves a visible step).
    jawline_band = np.zeros((h, w), dtype=np.uint8)
    if face_bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in face_bbox]
        fh = max(1, y2 - y1)
        y_jaw_start = y1 + int(fh * 0.70)
        skin = (class_map == 1)
        for y in range(max(0, y_jaw_start), min(h, y2)):
            jawline_band[y, max(0, x1):min(w, x2)] = 255
        jaw_skin = skin & (jawline_band > 0)
        # Drop source weight in the jaw band by 50%.
        alpha[jaw_skin] *= 0.50
        per_region['jawline'] = jaw_skin.astype(np.float32) * 0.50

    # Apply 3DDFA_V2 visible-surface mask: zero out hidden regions.
    if extra_visibility_mask is not None:
        vis = (extra_visibility_mask > 128).astype(np.float32)
        # Feather slightly so the visibility edge isn't a hard cut.
        vis = cv2.GaussianBlur(vis, (11, 11), 0)
        alpha = alpha * vis

    # Apply foreground occluder mask: zero out covered regions.
    if extra_occluder_mask is not None:
        occ = (extra_occluder_mask > 128).astype(np.float32)
        occ = cv2.GaussianBlur(occ, (9, 9), 0)
        alpha = alpha * (1.0 - occ)

    # Final feather so per-region jumps are smoothed at boundaries.
    fw = max(7, int(0.012 * min(h, w))) | 1
    alpha = cv2.GaussianBlur(alpha, (fw, fw), 0)
    alpha = np.clip(alpha, 0.0, 1.0)

    return {
        'alpha': alpha,
        'per_region': per_region,
        'jawline': jawline_band,
    }


def region_aware_composite(base: np.ndarray, source_warp: np.ndarray,
                           alpha_map: np.ndarray) -> np.ndarray:
    """Linear composite: result = source*alpha + base*(1-alpha)."""
    a = alpha_map[..., None].astype(np.float32)
    out = (source_warp.astype(np.float32) * a
           + base.astype(np.float32) * (1.0 - a))
    return np.clip(out, 0, 255).astype(np.uint8)
