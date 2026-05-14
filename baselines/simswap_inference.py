"""Standalone SimSwap-256 inference wrapper for our benchmark.

Loads the published SimSwap-256 generator + ArcFace ID encoder from the
official checkpoint and swaps the target face inside an aligned 256x256
crop. Re-pastes the swapped crop into the original target frame via the
inverse of the alignment transform.

Used by `run_baseline_simswap.py` to produce a head-to-head row against
our raw_inswapper / AI / Classical / Pro on the same benchmark pairs.
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Bring in SimSwap's Generator_Adain_Upsample architecture.
SIMSWAP_REPO = os.path.join(os.path.dirname(__file__), 'SimSwap')
if SIMSWAP_REPO not in sys.path:
    sys.path.insert(0, SIMSWAP_REPO)
from models.fs_networks import Generator_Adain_Upsample  # noqa: E402


# ---- alignment to the canonical 5-point face-recognition template ----
# Same template SimSwap uses (insightface 112x112 standard, scaled to 256).
_ARCFACE_DST_112 = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def _align_face(image: np.ndarray, kps: np.ndarray, size: int = 256):
    """Affine-warp `image` so the 5 keypoints land on the canonical template,
    scaled to `size` x `size`. Returns (aligned, M) where M is the 2x3 forward
    transform (so cv2.invertAffineTransform(M) reverses it)."""
    dst = _ARCFACE_DST_112 * (size / 112.0)
    M, _ = cv2.estimateAffinePartial2D(kps.astype(np.float32), dst,
                                       method=cv2.LMEDS)
    aligned = cv2.warpAffine(image, M, (size, size),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
    return aligned, M


def _paste_back(target_full: np.ndarray, swapped_256: np.ndarray,
                M: np.ndarray, size: int = 256):
    """Warp swapped 256x256 back into the target frame using inverse M,
    blended by a soft circular mask centred on the face."""
    h, w = target_full.shape[:2]
    M_inv = cv2.invertAffineTransform(M)
    swapped_full = cv2.warpAffine(swapped_256, M_inv, (w, h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(0, 0, 0))
    mask_256 = np.zeros((size, size), dtype=np.float32)
    cv2.circle(mask_256, (size // 2, size // 2), int(size * 0.40),
               1.0, thickness=-1)
    mask_256 = cv2.GaussianBlur(mask_256, (0, 0), sigmaX=size * 0.06)
    mask_full = cv2.warpAffine(mask_256, M_inv, (w, h),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=0)
    mask_full = mask_full[..., None]
    out = (swapped_full.astype(np.float32) * mask_full
           + target_full.astype(np.float32) * (1.0 - mask_full))
    return out.astype(np.uint8)


class SimSwapWrapper:
    """Minimal SimSwap-256 inference wrapper."""

    def __init__(self, gen_path='baselines/simswap_models/simswap_256.pth',
                 arc_path='baselines/simswap_models/arcface_model.jit',
                 device='cpu'):
        self.device = torch.device(device)

        # Build the generator that matches the published 256 checkpoint.
        gen = Generator_Adain_Upsample(input_nc=3, output_nc=3,
                                       latent_size=512, n_blocks=9, deep=False)
        sd = torch.load(gen_path, map_location='cpu', weights_only=False)
        gen.load_state_dict(sd, strict=True)
        gen.eval()
        self.gen = gen.to(self.device)

        # Load SimSwap's TorchScripted ArcFace ID encoder (consumes 112x112 RGB).
        arc = torch.jit.load(arc_path, map_location='cpu')
        arc.eval()
        self.arc = arc.to(self.device)

    @torch.no_grad()
    def _id_vec(self, aligned_256_bgr: np.ndarray) -> torch.Tensor:
        """Compute SimSwap's identity embedding from a 256x256 aligned RGB face."""
        rgb = cv2.cvtColor(aligned_256_bgr, cv2.COLOR_BGR2RGB)
        # SimSwap ArcFace expects 112x112 [0,1] then normalized
        rgb112 = cv2.resize(rgb, (112, 112), interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(rgb112.astype(np.float32) / 255.0)
        t = t.permute(2, 0, 1).unsqueeze(0)
        # SimSwap normalises with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        t = (t - mean) / std
        t = t.to(self.device)
        emb = self.arc(t)
        # L2 normalise (SimSwap uses cosine ID loss internally).
        emb = F.normalize(emb, p=2, dim=1)
        return emb

    @torch.no_grad()
    def swap(self, source_img: np.ndarray, target_img: np.ndarray,
             source_kps: np.ndarray, target_kps: np.ndarray) -> np.ndarray:
        """Replace the target face (defined by `target_kps`) with the source
        face (defined by `source_kps`). Returns a BGR uint8 image the same
        size as `target_img`.
        """
        # Align both faces into the 256x256 canonical frame.
        src_align, _ = _align_face(source_img, source_kps, size=256)
        tgt_align, M_tgt = _align_face(target_img, target_kps, size=256)

        # ID embedding from source.
        id_vec = self._id_vec(src_align)

        # Target tensor in [0,1] BGR-as-RGB matches SimSwap test_one_image convention.
        tgt_rgb = cv2.cvtColor(tgt_align, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tgt_t = torch.from_numpy(tgt_rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)

        # Run the generator: outputs in [0,1] (per `(x+1)/2` final step).
        out_t = self.gen(tgt_t, id_vec)
        out = out_t.squeeze(0).clamp(0, 1).cpu().numpy()
        out = (out.transpose(1, 2, 0) * 255.0).astype(np.uint8)
        out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

        # Paste the 256x256 swap back into the original target frame.
        full = _paste_back(target_img, out_bgr, M_tgt, size=256)
        return full
