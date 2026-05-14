"""HardPoseReplaceMode — the 14-step pipeline for tilted / occluded / action
target faces. Used by Pro mode when:

  - pose_delta > 35°, OR
  - first attempt produced high ghost_score (target identity still visible)

Pipeline
--------
 1. Detect source / target face (already done by the caller; passed in).
 2. Estimate pose: yaw, pitch, roll.
 3. If yaw/pitch is extreme, disable direct full-face transplant.
 4. Generate target identity-suppression mask
    (eyes + brows + nose + lips + cheek high-frequency).
 5. Suppress target high-frequency identity inside mask
    (inpaint + low-freq smoothing).
 6. Fit 3D target face → 3DDFA_V2 dense mesh.
 7. Render source identity / texture only onto visible target face surfaces.
 8. Build occlusion mask: hair + glasses + hand + foreground + depth-front.
 9. Region-wise blend per facial region (eyes / nose / lips / jaw / cheek).
10. Match target blur / noise / motion.
11. Run mixed-gradient / Laplacian blend (adaptive).
12. Restore target occlusions above the face.
13. Score ghosting: arc(out, src) vs arc(out, tgt).
14. If ghosting remains, increase target-suppression and retry.

This module is STRICTLY USED BY Pro mode. The photoshop / ai modes never
call into it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from pose_estimator import estimate_pose, pose_diff_magnitude
import target_suppressor
import visible_surface
import depth_occlusion
import occluder_sam
import region_blender
import blur_matcher
import mixed_clone
import ghost_detector


@dataclass
class HardPoseResult:
    image: np.ndarray
    ghost_metrics: ghost_detector.GhostMetrics
    clone_method: str
    suppression_mode: str          # 'low_freq' | 'inpaint' | 'both'
    occluder_sources: list         # ['bisenet'] / ['bisenet', 'sam2'] / etc.
    elapsed_ms: int
    notes: list


def run_hard_pose(swapper, source_img, target_img,
                  source_face, target_face,
                  suppression_mode: str = 'low_freq',
                  enhance: bool = True,
                  hd: bool = True,
                  preserve_hair: bool = True) -> HardPoseResult:
    """Execute the 14-step HardPoseReplaceMode pipeline.

    ``swapper`` is the existing FaceSwapper instance. We use its
    face_app, swapper (inswapper), enhancer (GFPGAN), _classes_around_face
    (BiSeNet), and _dense_landmarks (MediaPipe) read-only.
    """
    t0 = time.perf_counter()
    notes = []

    # === Step 2: pose ====================================================
    src_pose = estimate_pose(source_img, source_face)
    tgt_pose = estimate_pose(target_img, target_face)
    pose_delta = pose_diff_magnitude(src_pose, tgt_pose)
    notes.append(f'pose Δ {pose_delta:.1f}°  (src {src_pose.magnitude:.1f}, tgt {tgt_pose.magnitude:.1f})')

    # === Step 4: target identity-suppression mask ========================
    if not swapper._parser_available:
        notes.append('BiSeNet unavailable — falling back to bbox mask')
        h, w = target_img.shape[:2]
        x1, y1, x2, y2 = target_face.bbox.astype(int)
        suppression_mask = np.zeros((h, w), dtype=np.uint8)
        suppression_mask[y1:y2, x1:x2] = 255
    else:
        suppression_mask = target_suppressor.inner_face_mask_from_parser(
            swapper._classes_around_face, target_img, target_face,
        )
        if suppression_mask is None:
            h, w = target_img.shape[:2]
            x1, y1, x2, y2 = target_face.bbox.astype(int)
            suppression_mask = np.zeros((h, w), dtype=np.uint8)
            suppression_mask[y1:y2, x1:x2] = 255

    # === Step 5: suppress target high-frequency identity =================
    target_suppressed = target_suppressor.suppress_target_identity(
        target_img, suppression_mask, mode=suppression_mode,
    )
    notes.append(f'target suppression: {suppression_mode}')

    # === Step 1-build: inswapper base on the SUPPRESSED target ===========
    # Inswapper writes the source identity into the now-neutral target. The
    # combination of "neutral underlay + inswapper writes on top" leaves
    # NO target identity evidence under the swap → eliminates double-edge.
    base = target_suppressed.copy()
    base = swapper.swapper.get(base, target_face, source_face, paste_back=True)

    # GFPGAN 2x HD canvas.
    if enhance:
        swapper._set_enhancer_scale(2 if hd else 1)
        _, _, base = swapper.enhancer.enhance(
            base, has_aligned=False, only_center_face=True,
            paste_back=True, weight=0.4,
        )

    # === Re-detect face in the upscaled base ==============================
    r_faces = swapper._target_faces(base, False)
    if not r_faces:
        # Detection lost it; fall back gracefully.
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        return HardPoseResult(
            image=base,
            ghost_metrics=ghost_detector.compute(
                swapper.face_app, source_img, source_face,
                target_img, target_face, base, None,
                landmark_fn=swapper._dense_landmarks,
                pose_delta_deg=pose_delta,
            ),
            clone_method='none',
            suppression_mode=suppression_mode,
            occluder_sources=[],
            elapsed_ms=elapsed_ms,
            notes=notes + ['face detection lost in upscaled base'],
        )
    base_face = r_faces[0]

    # === Step 6: fit 3D mesh, compute visible surface ====================
    visible_mask = visible_surface.compute_visible_mask(base, base_face)
    if visible_mask is not None:
        notes.append('3D visible-surface mask: ON')
    else:
        notes.append('3D visible-surface mask: unavailable')

    # === Step 7: warp full-res source onto base landmarks ================
    src_dense = swapper._dense_landmarks(source_img, source_face.bbox)
    tgt_dense = swapper._dense_landmarks(base, base_face.bbox)
    if (src_dense is not None and tgt_dense is not None
            and len(src_dense) == len(tgt_dense)):
        M, _ = cv2.estimateAffine2D(src_dense, tgt_dense, method=cv2.LMEDS)
    else:
        M, _ = cv2.estimateAffinePartial2D(
            source_face.kps.astype(np.float32),
            target_face.kps.astype(np.float32),
            method=cv2.LMEDS,
        )
    if M is None:
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        return HardPoseResult(
            image=base, ghost_metrics=ghost_detector.compute(
                swapper.face_app, source_img, source_face,
                target_img, target_face, base, base_face,
                landmark_fn=swapper._dense_landmarks,
                pose_delta_deg=pose_delta,
            ),
            clone_method='none', suppression_mode=suppression_mode,
            occluder_sources=[], elapsed_ms=elapsed_ms,
            notes=notes + ['affine fit failed'],
        )

    h, w = base.shape[:2]
    source_warp = cv2.warpAffine(
        source_img, M, (w, h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    # === Step 8: build occluder mask =====================================
    occluder_sources = []
    occluder_mask = None
    bise_occ = swapper._occluder_mask(target_img, target_face)
    if bise_occ is not None and bise_occ.max() > 0:
        occluder_sources.append('bisenet')
        # Refine with SAM 2 if available.
        if hasattr(occluder_sam, 'refine_occluder_mask'):
            refined = occluder_sam.refine_occluder_mask(
                target_img, bise_occ, target_face.bbox,
            )
            if refined is not None and refined is not bise_occ:
                occluder_sources.append('sam2')
                bise_occ = refined
        occluder_mask = bise_occ

    # Add depth-front occluders.
    depth_occ = depth_occlusion.foreground_occluder_mask(target_img, target_face)
    if depth_occ is not None and depth_occ.max() > 0:
        occluder_sources.append('depth_anything_v2')
        if occluder_mask is None:
            occluder_mask = depth_occ
        else:
            occluder_mask = cv2.bitwise_or(occluder_mask, depth_occ)

    # Scale occluder mask to base size.
    if occluder_mask is not None and occluder_mask.shape != base.shape[:2]:
        occluder_mask = cv2.resize(
            occluder_mask, (base.shape[1], base.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    notes.append(f'occluders: {occluder_sources or ["none"]}')

    # === Step 9: region-wise alpha =======================================
    cls, _ = swapper._classes_around_face(base, base_face.bbox) \
        if swapper._parser_available else (None, None)
    if cls is not None:
        rb = region_blender.build_region_alpha(
            cls,
            face_bbox=base_face.bbox,
            extra_visibility_mask=visible_mask,
            extra_occluder_mask=occluder_mask,
        )
        alpha = rb['alpha']
    else:
        # Hull fallback.
        hull_mask, _ = swapper._face_mask(base_face, base.shape) \
            if hasattr(swapper, '_face_mask') else (None, None)
        alpha = (hull_mask.astype(np.float32) / 255.0) if hull_mask is not None \
            else np.zeros(base.shape[:2], dtype=np.float32)

    # === Step 10: match target blur / noise / motion =====================
    tgt_profile = blur_matcher.estimate_profile(base, base_face)
    src_profile = blur_matcher.estimate_profile(source_img, source_face)
    source_warp_matched = blur_matcher.match_to_target(
        source_warp, tgt_profile, src_profile,
    )
    notes.append(
        f'imaging match: tgt_lap_var={tgt_profile.laplacian_variance:.1f}, '
        f'noise σ={tgt_profile.noise_sigma:.2f}, '
        f'motion {tgt_profile.motion_angle_deg:.0f}° '
        f'(strength {tgt_profile.motion_strength:.2f})'
    )

    # === Step 11: mixed-gradient / Laplacian clone =======================
    # Composite the matched source via region alpha first to get a smooth
    # candidate, THEN run an adaptive Poisson / Laplacian blend over the
    # union of the alpha to remove any remaining boundary energy.
    composited = region_blender.region_aware_composite(
        base, source_warp_matched, alpha,
    )
    mask_for_clone = (alpha * 255).astype(np.uint8)
    # Erode the clone mask slightly so the boundary sits inside the visible
    # face area (where it's least noticeable).
    erode_k = max(3, int(0.012 * min(h, w)))
    clone_mask = cv2.erode(mask_for_clone, np.ones((erode_k, erode_k), np.uint8))
    cloned, clone_method = mixed_clone.adaptive_clone(
        composited, base, clone_mask,
    )
    notes.append(f'clone method: {clone_method}')

    # === Step 12: restore target occluders on top ========================
    if preserve_hair and swapper._parser_available:
        occ_for_top = swapper._occluder_mask(target_img, target_face)
        if occ_for_top is not None and occ_for_top.max() > 0:
            target_resized = target_img
            if cloned.shape[:2] != target_img.shape[:2]:
                target_resized = cv2.resize(
                    target_img, (cloned.shape[1], cloned.shape[0]),
                    interpolation=cv2.INTER_LANCZOS4,
                )
                occ_for_top = cv2.resize(
                    occ_for_top, (cloned.shape[1], cloned.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
            a = (occ_for_top.astype(np.float32) / 255.0)[..., None]
            cloned = (
                cloned.astype(np.float32) * (1 - a)
                + target_resized.astype(np.float32) * a
            )
            cloned = np.clip(cloned, 0, 255).astype(np.uint8)

    # === Step 13: ghost-score the result =================================
    r_faces2 = swapper._target_faces(cloned, False)
    rf = r_faces2[0] if r_faces2 else None
    ghost = ghost_detector.compute(
        swapper.face_app, source_img, source_face,
        target_img, target_face, cloned, rf,
        mask=mask_for_clone,
        landmark_fn=swapper._dense_landmarks,
        pose_delta_deg=pose_delta,
    )
    notes.append(
        f'ghost: arc_src={ghost.arc_to_source:.3f} '
        f'arc_tgt={ghost.arc_to_target:.3f} '
        f'score={ghost.ghost_score:.3f}  '
        f'seam={ghost.seam_gradient_energy:.1f}  '
        f'route→{ghost.routing}'
    )

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    return HardPoseResult(
        image=cloned,
        ghost_metrics=ghost,
        clone_method=clone_method,
        suppression_mode=suppression_mode,
        occluder_sources=occluder_sources,
        elapsed_ms=elapsed_ms,
        notes=notes,
    )


# Step 14: retry with stronger suppression if ghosting remains.
def run_hard_pose_with_retry(swapper, source_img, target_img,
                             source_face, target_face,
                             max_attempts: int = 2,
                             **kwargs) -> HardPoseResult:
    """Run HardPoseReplaceMode. If `ghost_score > -0.05` after the first
    attempt (target identity still visible), retry with stronger
    suppression. Max two attempts."""
    # Order: gentlest first. Aggressive modes only used if ghost remains
    # bad after the gentle one (low-freq preserves source-identity-anchor
    # structures inswapper relies on; full-inpaint can wipe them out and
    # collapse the result to a no-identity blank face).
    suppression_modes = ['low_freq', 'both', 'inpaint']
    best = None
    for i in range(min(max_attempts, len(suppression_modes))):
        mode = suppression_modes[i]
        result = run_hard_pose(
            swapper, source_img, target_img, source_face, target_face,
            suppression_mode=mode, **kwargs,
        )
        if best is None or (
            not np.isnan(result.ghost_metrics.ghost_score)
            and (np.isnan(best.ghost_metrics.ghost_score)
                 or result.ghost_metrics.ghost_score < best.ghost_metrics.ghost_score)
        ):
            best = result
        # Stop if the routing decision says we're good.
        if result.ghost_metrics.routing == 'ok':
            return result
        # Stop if ghost_score is already very negative (source dominates).
        if (not np.isnan(result.ghost_metrics.ghost_score)
                and result.ghost_metrics.ghost_score < -0.15):
            return result
    return best
