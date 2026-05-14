"""SAM 2 promptable segmentation for face occluders.

Used by Pro mode's HardPoseReplaceMode to refine masks for objects that
fall in front of the target face: hair strands, glasses frames,
microphone, hands, hat brim. SAM 2 takes point/box prompts and returns
high-resolution masks — far more accurate than BiSeNet for arbitrary
foreground objects, especially in action scenes.

This is the official SAM 2 base_plus checkpoint (~308 MB) downloaded by
``download_pro_models.py``.

Calling pattern:
    sam = OccluderSAM.get()
    if sam is not None:
        mask = sam.predict_box(image_bgr, [x1, y1, x2, y2])

Prompts in HardPoseReplaceMode:
    1. Negative bbox prompt = face bbox  (SAM should NOT include face skin)
    2. Positive point prompts on BiSeNet hair/glasses/hand pixels
    3. Negative point prompts on BiSeNet skin pixels
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np


_SAM_DIR = Path(__file__).resolve().parent / 'models' / 'pro' / 'sam2'
_SAM_CKPT = _SAM_DIR / 'sam2_hiera_base_plus.pt'
_SAM_CFG = 'sam2_hiera_b+.yaml'


class OccluderSAM:
    _instance = None

    @classmethod
    def get(cls):
        if cls._instance is not None:
            return cls._instance if cls._instance is not False else None
        if not _SAM_CKPT.exists():
            print(f'[occluder_sam] no checkpoint at {_SAM_CKPT}; disabled')
            cls._instance = False
            return None
        try:
            cls._instance = cls()
        except Exception as e:
            print(f'[occluder_sam] init failed: {e}')
            cls._instance = False
            return None
        return cls._instance

    def __init__(self):
        import os
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        # SAM 2 uses upsample_bicubic2d which isn't implemented on MPS in
        # current PyTorch. Force CPU (slower but correct) and tell PyTorch
        # to allow CPU fallback for any other unimplemented MPS ops.
        os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'  # NOT mps — bicubic2d isn't implemented
        sam_model = build_sam2(_SAM_CFG, str(_SAM_CKPT), device=device)
        self.predictor = SAM2ImagePredictor(sam_model)
        self.device = device
        print(f'[occluder_sam] SAM 2 base_plus ready on {device}')

    def _set_image(self, image_bgr: np.ndarray):
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(rgb)

    def predict_box(self, image_bgr: np.ndarray,
                    box_xyxy: Sequence[int]) -> Optional[np.ndarray]:
        """Segment the object inside `box_xyxy`. Returns uint8 0/255 mask."""
        try:
            self._set_image(image_bgr)
            box = np.array(box_xyxy, dtype=np.float32)[None, :]
            masks, scores, _ = self.predictor.predict(box=box, multimask_output=False)
        except Exception as e:
            print(f'[occluder_sam] predict_box failed: {e}')
            return None
        return (masks[0] > 0).astype(np.uint8) * 255

    def predict_points(self, image_bgr: np.ndarray,
                       pos_points: np.ndarray,
                       neg_points: Optional[np.ndarray] = None
                       ) -> Optional[np.ndarray]:
        """Segment from point prompts. pos_points (N, 2), neg_points (M, 2)."""
        try:
            self._set_image(image_bgr)
            pts = pos_points.astype(np.float32)
            labels = np.ones(len(pts), dtype=np.int32)
            if neg_points is not None and len(neg_points) > 0:
                pts = np.vstack([pts, neg_points.astype(np.float32)])
                labels = np.concatenate(
                    [labels, np.zeros(len(neg_points), dtype=np.int32)],
                )
            masks, scores, _ = self.predictor.predict(
                point_coords=pts, point_labels=labels, multimask_output=False,
            )
        except Exception as e:
            print(f'[occluder_sam] predict_points failed: {e}')
            return None
        return (masks[0] > 0).astype(np.uint8) * 255


def refine_occluder_mask(image: np.ndarray,
                         bisenet_occluder: np.ndarray,
                         face_bbox: Sequence[int]) -> Optional[np.ndarray]:
    """Use SAM 2 to refine BiSeNet's coarse occluder mask.

    Strategy: take 5–8 random positive points from the BiSeNet occluder
    mask, 5–8 random negative points from the face-skin interior, and let
    SAM 2 propagate to a high-resolution mask. Returns refined uint8 mask
    or None if SAM 2 is unavailable.
    """
    sam = OccluderSAM.get()
    if sam is None or bisenet_occluder is None or bisenet_occluder.max() == 0:
        return bisenet_occluder
    pos_idx = np.argwhere(bisenet_occluder > 200)
    if pos_idx.size == 0:
        return bisenet_occluder
    rng = np.random.default_rng(42)
    pos_sample = pos_idx[rng.choice(len(pos_idx), size=min(8, len(pos_idx)),
                                     replace=False)]
    pos_points = pos_sample[:, ::-1]  # (y, x) → (x, y)

    # Negative points = face skin interior. Eroded face bbox.
    x1, y1, x2, y2 = [int(v) for v in face_bbox]
    fw, fh = x2 - x1, y2 - y1
    nx1 = x1 + fw // 4
    ny1 = y1 + fh // 3
    nx2 = x2 - fw // 4
    ny2 = y2 - fh // 4
    neg_points = np.array([
        [(nx1 + nx2) // 2, (ny1 + ny2) // 2],
        [nx1, (ny1 + ny2) // 2],
        [nx2, (ny1 + ny2) // 2],
        [(nx1 + nx2) // 2, ny1],
    ], dtype=np.float32)

    refined = sam.predict_points(image, pos_points, neg_points)
    if refined is None:
        return bisenet_occluder
    # Take the union of BiSeNet + SAM 2 to be safe: SAM 2 can sometimes miss
    # thin strands.
    return cv2.bitwise_or(bisenet_occluder, refined)
