"""Pro mode — adaptive face swap with pose gating + multi-pipeline scoring.

This module is STRICTLY ADDITIVE. It imports from ``face_swap`` (the existing
FaceSwapper) to reuse the model instances (inswapper, GFPGAN, BiSeNet,
MediaPipe), but never modifies them. The existing ``photoshop`` and ``ai``
modes work *exactly* as before — Pro is a third option triggered only when
the user explicitly selects it.

Strategy
--------
1. **Pose estimation gate**: estimate yaw/pitch/roll on both source and
   target via OpenCV solvePnP. Compute pose-difference magnitude (max
   single-axis delta in degrees).

2. **Pipeline candidate set**: based on pose difference, run multiple
   variants of the swap pipeline in parallel slots:
     - photoshop (existing) — baseline
     - photoshop tweaked (detail=1.3, preserve_source_tone=True) — max
       source fidelity
     - tps_overlay — Pro's TPS warp of full-res source on top of inswapper
       base, with region-aware blending
     - ai_synthesis (existing) — pose-tolerant fallback if pose diff is large

3. **Multi-pipeline scoring**: every candidate gets ArcFace cosine identity
   + landmark RMSE + SSIM. The winner is the candidate with the HIGHEST
   ArcFace cosine (identity preservation is the dominant metric).

4. **Region-aware blending in tps_overlay**: BiSeNet's 19-class output is
   used to weight different facial regions differently:
       eyes  → 100% source (preserve identity)
       skin  → blend with target lighting
       lips  → 100% source
       inner mouth → target (avoid grafted teeth)
       hair / glasses → target (hair-aware composite handles this anyway)

5. **Future hooks**: 3DDFA_V2 / SAM 2 / manual editor are noted but not
   implemented here — they require separate model downloads + UI work.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from pose_estimator import estimate_pose, pose_diff_magnitude, PoseAngles
from pro_warps import tps_warp, piecewise_affine_warp
from quality_metrics import compute_metrics, QualityScores

# Pro-mode-only modules — additive. Photoshop / AI modes never touch these.
import ghost_detector
import hard_pose_replace


@dataclass
class ProResult:
    """Output of the Pro pipeline."""
    image: np.ndarray
    scores: QualityScores
    chosen_route: str
    source_pose: PoseAngles
    target_pose: PoseAngles
    pose_diff: float
    candidates: list  # list of (route_name, scores_dict_with_ghost)
    ghost: dict = None  # ghost-detector metrics for the winning candidate

    def info_dict(self):
        return {
            'chosen_route': self.chosen_route,
            'source_pose': self.source_pose.to_dict(),
            'target_pose': self.target_pose.to_dict(),
            'pose_diff_deg': round(self.pose_diff, 2),
            'ghost': self.ghost,
            'candidates': [
                {'route': name, 'scores': scores}
                for name, scores in self.candidates
            ],
        }


def pro_swap(swapper, source_img: np.ndarray, target_img: np.ndarray,
             **kwargs) -> ProResult:
    """Adaptive pipeline entry point. ``swapper`` is the existing FaceSwapper
    instance — we use its face_app, swapper, enhancer, _dense_landmarks,
    _skin_mask, etc. without modifying them. ``kwargs`` are the same form
    fields as ``FaceSwapper.swap``."""

    # Detect source + target faces upfront so we can compute pose ONCE.
    source_face = swapper._pick_source_face(source_img)
    if source_face is None:
        raise ValueError('No face detected in the base (source) image.')
    swap_all = kwargs.get('swap_all_targets', False)
    target_faces = swapper._target_faces(target_img, swap_all)
    if not target_faces:
        raise ValueError('No face detected in the target (body) image.')
    target_face = target_faces[0]  # Pro mode only swaps the largest face

    src_pose = estimate_pose(source_img, source_face)
    tgt_pose = estimate_pose(target_img, target_face)
    pose_diff = pose_diff_magnitude(src_pose, tgt_pose)

    # Decide which routes to try. The set scales with pose difficulty.
    # `ai_synthesis` is ALWAYS in the candidate set so the auto-best selector
    # is never forced to ship a worse result than the AI baseline could
    # produce — the original pose-gated set was leaving AI off the menu on
    # frontal pairs and Pro could land below AI on paired Δα_s (Sec.~V-I).
    routes = ['photoshop', 'ai_synthesis']
    if pose_diff > 8.0:
        routes.append('photoshop_strong')      # detail=1.3, tone-preserve
    if pose_diff > 12.0:
        routes.append('tps_overlay')           # Pro's TPS warp
    # HardPoseReplaceMode for extreme pose / tilted / action targets.
    # This is the route that:
    #   1. suppresses target identity (inpaint + low-freq)
    #   2. fits 3DDFA_V2 dense mesh for visible-surface masking
    #   3. SAM 2 / Depth-Anything occluder masks
    #   4. region-aware blending
    #   5. blur / noise / motion matching
    #   6. adaptive Poisson / Laplacian clone
    #   7. ghost-score scoring + auto-retry with stronger suppression
    if pose_diff > 25.0:
        routes.append('hard_pose')

    candidates = []
    for route in routes:
        print(f'[pro_swap] running route: {route}')
        try:
            img, elapsed_ms = _run_route(
                route, swapper, source_img, target_img,
                source_face, target_face, **kwargs,
            )
        except Exception as e:
            print(f'[pro_swap] route "{route}" failed: {e}')
            try:
                swapper.cleanup()
            except Exception:
                pass
            continue
        # Quality metrics (ArcFace identity, SSIM, ΔE, landmark RMSE).
        r_faces = swapper._target_faces(img, False)
        tf_r = r_faces[0] if r_faces else None
        scores = compute_metrics(
            swapper.face_app,
            source_img, img, tf_r,
            dense_landmarks_fn=swapper._dense_landmarks,
            source_face=source_face,
            elapsed_ms=elapsed_ms,
        )
        # Ghost-face detector: routing-aware metrics.
        ghost = ghost_detector.compute(
            swapper.face_app,
            source_img, source_face,
            target_img, target_face,
            img, tf_r,
            mask=None,
            landmark_fn=swapper._dense_landmarks,
            pose_delta_deg=pose_diff,
        )
        candidates.append((route, img, scores, ghost))
        try:
            swapper.cleanup()
        except Exception:
            pass
        print(f'[pro_swap]   → arc={scores.arcface_cosine}, '
              f'ghost={ghost.ghost_score:.3f}, route_hint={ghost.routing}, '
              f'ms={elapsed_ms}')

    # SECOND PASS: ghost-score-driven re-routing. If any first-pass candidate
    # advises `suppress_target` or `rebuild_base`, that means the result is
    # still too close to the target identity — overlay blending didn't fully
    # transfer the source. Run HardPoseReplaceMode (which does target
    # suppression + 3D visible-surface masking + region-aware blending) as
    # an extra candidate so the auto-best selector can compare.
    if 'hard_pose' not in [r for r, _, _, _ in candidates]:
        wants_suppression = any(
            g.routing in ('suppress_target', 'rebuild_base', 'hard_pose')
            for _, _, _, g in candidates
        )
        if wants_suppression:
            print('[pro_swap] ghost-routing → adding hard_pose candidate')
            try:
                img, elapsed_ms = _run_route(
                    'hard_pose', swapper, source_img, target_img,
                    source_face, target_face, **kwargs,
                )
                r_faces = swapper._target_faces(img, False)
                tf_r = r_faces[0] if r_faces else None
                s = compute_metrics(
                    swapper.face_app,
                    source_img, img, tf_r,
                    dense_landmarks_fn=swapper._dense_landmarks,
                    source_face=source_face,
                    elapsed_ms=elapsed_ms,
                )
                g = ghost_detector.compute(
                    swapper.face_app,
                    source_img, source_face,
                    target_img, target_face,
                    img, tf_r,
                    mask=None,
                    landmark_fn=swapper._dense_landmarks,
                    pose_delta_deg=pose_diff,
                )
                candidates.append(('hard_pose', img, s, g))
                print(f'[pro_swap]   hard_pose → arc={s.arcface_cosine}, '
                      f'ghost={g.ghost_score:.3f}, '
                      f'route_hint={g.routing}, ms={elapsed_ms}')
                try:
                    swapper.cleanup()
                except Exception:
                    pass
            except Exception as e:
                import traceback
                print(f'[pro_swap] hard_pose second-pass failed: {e}')
                traceback.print_exc()

    if not candidates:
        raise RuntimeError('All Pro pipeline routes failed.')

    # Selection metric: ghost_score is the dominant signal (more negative =
    # source identity dominates, target identity gone). ArcFace-to-source
    # is the tiebreak (higher = closer to source identity).
    def key(c):
        _, _, s, g = c
        ghost = g.ghost_score if not np.isnan(g.ghost_score) else 99.0
        arc = s.arcface_cosine if not np.isnan(s.arcface_cosine) else -1.0
        # primary: minimise ghost_score; tiebreak: maximise arc-to-source
        return (-ghost, arc)
    candidates.sort(key=key, reverse=True)
    best_route, best_img, best_scores, best_ghost = candidates[0]

    return ProResult(
        image=best_img,
        scores=best_scores,
        chosen_route=best_route,
        source_pose=src_pose,
        target_pose=tgt_pose,
        pose_diff=pose_diff,
        candidates=[
            (name, {**s.to_dict(), 'ghost': g.to_dict()})
            for name, _, s, g in candidates
        ],
        ghost=best_ghost.to_dict(),
    )


def _run_route(route, swapper, source_img, target_img, source_face,
               target_face, **kwargs) -> tuple:
    """Dispatch to the implementation for a named route. Returns (image, ms)."""
    t0 = time.perf_counter()
    if route == 'photoshop':
        img = swapper.swap(source_img, target_img, mode='photoshop', **kwargs)
    elif route == 'photoshop_strong':
        ps_kwargs = dict(kwargs)
        ps_kwargs['detail'] = 1.3
        ps_kwargs['preserve_source_tone'] = True
        img = swapper.swap(source_img, target_img, mode='photoshop', **ps_kwargs)
    elif route == 'ai_synthesis':
        ai_kwargs = dict(kwargs)
        ai_kwargs['fidelity'] = 0.9
        img = swapper.swap(source_img, target_img, mode='ai', **ai_kwargs)
    elif route == 'tps_overlay':
        img = _tps_overlay_swap(
            swapper, source_img, target_img, source_face, target_face, **kwargs,
        )
    elif route == 'hard_pose':
        # The HardPoseReplaceMode pipeline (14 steps). Includes its own
        # auto-retry with stronger suppression if ghost_score > -0.05.
        hp_result = hard_pose_replace.run_hard_pose_with_retry(
            swapper, source_img, target_img, source_face, target_face,
            enhance=kwargs.get('enhance', True),
            hd=kwargs.get('hd', True),
            preserve_hair=kwargs.get('preserve_hair', True),
        )
        img = hp_result.image
    else:
        raise ValueError(f'Unknown route: {route}')
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    return img, elapsed_ms


def _tps_overlay_swap(swapper, source_img, target_img, source_face,
                      target_face, **kwargs) -> np.ndarray:
    """Pro's TPS-based overlay swap.

    Pipeline
    --------
    1. Run inswapper to get the base (pose / expression / lighting-adapted).
    2. Run GFPGAN 2× for HD canvas.
    3. Get dense MediaPipe FaceMesh landmarks on both source and base.
    4. **TPS warp** the full-resolution source image onto the base
       landmarks. TPS handles non-rigid expression / jaw / cheek motion
       smoothly, unlike a single affine.
    5. Compute region-aware alpha:
         eyes + lips      → 100% source pixels (identity-critical)
         skin / cheek     → 80% source / 20% base
         inner mouth      → base only (no transplant)
         hair / glasses   → base only (handled by hair composite later)
    6. Composite warped source onto base with the per-region alpha.
    7. Poisson seamless clone for the boundary.
    8. Hair-aware composite is added by swap() afterwards (we delegate by
       calling swapper.swap() with mode=photoshop, then redo the graft —
       too complex; instead we replicate the bits we need here).
    """
    swap_all = kwargs.get('swap_all_targets', False)
    enhance = kwargs.get('enhance', True)
    hd = kwargs.get('hd', True)
    sharpen = float(kwargs.get('sharpen', 0.35))
    detail = float(kwargs.get('detail', 1.0))
    preserve_tone = kwargs.get('preserve_source_tone', False)
    preserve_hair = kwargs.get('preserve_hair', True)

    # Step 1: inswapper base.
    base = target_img.copy()
    base = swapper.swapper.get(base, target_face, source_face, paste_back=True)

    # Step 2: GFPGAN 2× upscale (uses MPS via existing FaceSwapper config).
    if enhance:
        swapper._set_enhancer_scale(2 if hd else 1)
        _, _, base = swapper.enhancer.enhance(
            base, has_aligned=False, only_center_face=True,
            paste_back=True, weight=0.4,
        )

    # Step 3: re-detect face in the upscaled base.
    rescaled = swapper._target_faces(base, False)
    if not rescaled:
        return base
    base_face = rescaled[0]

    # Step 4: dense landmarks.
    src_lm = swapper._dense_landmarks(source_img, source_face.bbox)
    tgt_lm = swapper._dense_landmarks(base, base_face.bbox)
    if src_lm is None or tgt_lm is None or len(src_lm) != len(tgt_lm):
        # Fall back to a clean photoshop result.
        return swapper.swap(source_img, target_img, mode='photoshop', **kwargs)

    # Step 5: TPS warp source → target geometry.
    warped, _coverage = tps_warp(source_img, src_lm, tgt_lm, base.shape)

    # Step 6: region-aware alpha mask.
    alpha = _region_aware_alpha(swapper, base, base_face)
    if alpha is None:
        # Fall back to a clean photoshop result.
        return swapper.swap(source_img, target_img, mode='photoshop', **kwargs)

    # Optional LAB color match in skin region.
    if not preserve_tone:
        # Use the binary skin mask for color stats.
        warped = swapper._color_transfer_lab(
            warped, base, (alpha * 255).astype(np.uint8),
        )

    # Step 7: alpha composite warped source onto base.
    alpha_3 = alpha[..., None] * detail
    alpha_3 = np.clip(alpha_3, 0.0, 1.0)
    result = (warped.astype(np.float32) * alpha_3
              + base.astype(np.float32) * (1.0 - alpha_3))
    result = np.clip(result, 0, 255).astype(np.uint8)

    # Step 8: hair-aware composite. We call into the existing helper so the
    # target's hair / glasses / accessories sit on top in z-order.
    if preserve_hair and swapper._parser_available:
        target_resized = target_img
        if result.shape[:2] != target_img.shape[:2]:
            target_resized = cv2.resize(
                target_img, (result.shape[1], result.shape[0]),
                interpolation=cv2.INTER_LANCZOS4,
            )
        occ = swapper._occluder_mask(target_img, target_face)
        if occ is not None and occ.max() > 0:
            if occ.shape != result.shape[:2]:
                occ = cv2.resize(
                    occ, (result.shape[1], result.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
            a = (occ.astype(np.float32) / 255.0)[..., None]
            result = (result.astype(np.float32) * (1 - a)
                      + target_resized.astype(np.float32) * a)
            result = np.clip(result, 0, 255).astype(np.uint8)

    # Final unsharp polish.
    if sharpen > 0:
        result = swapper._unsharp_mask(result, sharpen)
    return result


def _region_aware_alpha(swapper, base, target_face):
    """Build a per-pixel float32 alpha mask in [0, 1] inside the face area.

    Different regions get different weights:
        eyes + brows + lips  → 1.0 (full source)
        nose                 → 0.95
        skin                 → 0.85
        inner mouth (11)     → 0.10 (mostly target — avoid weird teeth)
        background / hair    → 0.0

    Returns None if the parser isn't available or fails.
    """
    if not swapper._parser_available:
        return None
    cls, _bbox = swapper._classes_around_face(base, target_face.bbox)
    if cls is None:
        return None
    h, w = base.shape[:2]
    alpha = np.zeros((h, w), dtype=np.float32)
    # CelebAMask-HQ class weights — match the FACE_SURFACE_CLASSES used by
    # the existing pipeline but with per-region opacity.
    weights = {
        1: 0.88,    # skin
        2: 1.00,    # l_brow
        3: 1.00,    # r_brow
        4: 1.00,    # l_eye
        5: 1.00,    # r_eye
        10: 0.95,   # nose
        11: 0.20,   # mouth (interior)
        12: 1.00,   # u_lip
        13: 1.00,   # l_lip
    }
    for cls_idx, w_val in weights.items():
        alpha[cls == cls_idx] = w_val

    # Feather the boundary so we don't get a sharp edge.
    face_w = max(1, int(target_face.bbox[2] - target_face.bbox[0]))
    feather_k = max(11, int(face_w * 0.025) | 1)
    alpha = cv2.GaussianBlur(alpha, (feather_k, feather_k), 0)
    return alpha
