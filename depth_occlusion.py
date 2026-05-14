"""Depth-Anything-V2 monocular depth → foreground-occluder mask.

A face being swapped can be partially occluded by objects in front: hands,
microphone, glasses frame, hair strand. Standard BiSeNet doesn't catch
arbitrary occluders. We use Depth Anything V2 to estimate per-pixel depth,
then mark pixels in front of the face's median depth as "occluders" and
mask them out of the swap region.

Depth Anything V2 small ONNX (~99 MB) is downloaded by
``download_pro_models.py`` into ``models/pro/depth_anything_v2/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np


_DEPTH_DIR = Path(__file__).resolve().parent / 'models' / 'pro' / 'depth_anything_v2'
_DEPTH_ONNX = _DEPTH_DIR / 'depth_anything_v2_small.onnx'


class _DepthSession:
    _instance = None

    @classmethod
    def get(cls):
        if cls._instance is not None:
            return cls._instance
        if not _DEPTH_ONNX.exists():
            print(f'[depth_occlusion] no checkpoint at {_DEPTH_ONNX}; disabled')
            cls._instance = False
            return None
        try:
            import onnxruntime as ort
            cls._instance = cls()
        except Exception as e:
            print(f'[depth_occlusion] init failed: {e}')
            cls._instance = False
            return None
        return cls._instance

    def __init__(self):
        import onnxruntime as ort
        providers = (
            ['CUDAExecutionProvider', 'CPUExecutionProvider']
            if 'CUDAExecutionProvider' in ort.get_available_providers()
            else ['CPUExecutionProvider']
        )
        self.session = ort.InferenceSession(
            str(_DEPTH_ONNX), providers=providers,
        )
        # Figure out input shape from the model. Depth Anything V2 ONNX
        # exports usually have a dynamic batch dim with H=W=518 fixed.
        inp = self.session.get_inputs()[0]
        shape = inp.shape  # e.g. ['batch', 3, 518, 518] or [1, 3, 518, 518]
        self.input_name = inp.name
        try:
            self.h = int(shape[2])
            self.w = int(shape[3])
        except (TypeError, ValueError):
            self.h = self.w = 518

    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        """Return per-pixel relative depth at the input image resolution.
        Larger values = closer to camera (Depth Anything V2 convention)."""
        orig_h, orig_w = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.w, self.h), interpolation=cv2.INTER_CUBIC)
        # Depth Anything V2 preprocess: normalize to ImageNet stats.
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        x = resized.astype(np.float32) / 255.0
        x = (x - mean) / std
        x = x.transpose(2, 0, 1)[None, ...]  # (1, 3, H, W)
        out = self.session.run(None, {self.input_name: x})[0]
        depth = np.squeeze(out)
        if depth.shape != (orig_h, orig_w):
            depth = cv2.resize(
                depth, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR,
            )
        return depth.astype(np.float32)


def foreground_occluder_mask(image: np.ndarray, face,
                             ahead_quantile: float = 0.85) -> Optional[np.ndarray]:
    """Build a mask of pixels that are MORE FORWARD than the face's median
    depth. Such pixels are physically in front of the face plane and
    should occlude it (hands, mic, hair strand drifting in front, etc.).

    ``ahead_quantile`` is the relative-depth quantile (within the face
    bbox) above which we consider a pixel "in front". Tuned conservatively
    so we don't accidentally cull face skin.

    Returns uint8 mask 0/255, or None if Depth Anything V2 isn't available.
    """
    sess = _DepthSession.get()
    if sess is None or sess is False:
        return None
    try:
        depth = sess.predict(image)
    except Exception as e:
        print(f'[depth_occlusion] predict failed: {e}')
        return None

    x1, y1, x2, y2 = face.bbox.astype(int)
    h, w = image.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None

    face_depth = depth[y1:y2, x1:x2]
    threshold = float(np.quantile(face_depth, ahead_quantile))
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[depth > threshold] = 255
    # Restrict to the face's bounding region — distant background pixels
    # closer to the camera (due to depth-model quirks) shouldn't count.
    box_mask = np.zeros((h, w), dtype=np.uint8)
    box_mask[y1:y2, x1:x2] = 255
    mask = cv2.bitwise_and(mask, box_mask)

    # Smooth + feather slightly so the occluder boundary blends.
    fw = max(5, int(0.01 * min(h, w))) | 1
    mask = cv2.medianBlur(mask, fw)
    mask = cv2.GaussianBlur(mask, (fw, fw), 0)
    return mask
