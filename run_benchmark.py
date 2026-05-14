"""Multi-pair benchmark runner for the IEEE paper.

For each cross-identity pair (src, tgt) in a curated list:
  - run the AI mode swap
  - run the Classical Compositing mode swap (a.k.a. 'photoshop' internally)
  - compute ArcFace cosines (alpha_s, alpha_t), ghost score g = alpha_t - alpha_s,
    pose difference theta_pose, and wall-clock time per mode
  - log everything to a CSV + JSON

This produces the multi-N aggregate evidence the paper needs to move
beyond a single-pair claim.

Run with:
    source venv/bin/activate
    python run_benchmark.py --max-pairs 12 --modes ai,classical --output paper/figures/benchmark.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from face_swap import FaceSwapper


def list_portraits(root: Path, min_size_kb: int = 200) -> list[Path]:
    """Find candidate portrait images under `root`."""
    exts = {'.png', '.jpg', '.jpeg'}
    out: list[Path] = []
    for p in root.rglob('*'):
        if p.suffix.lower() not in exts:
            continue
        try:
            if p.stat().st_size < min_size_kb * 1024:
                continue
        except OSError:
            continue
        # heuristic: prefer files whose names hint at a portrait subject
        name = p.name.lower()
        bad_tokens = ('abstract', 'art_nouveau', 'oilpainting', 'legs_hanging',
                      'screenshot', 'cyc_world', 'face-swap')
        if any(t in name for t in bad_tokens):
            continue
        out.append(p)
    return out


def detect_one_face(swapper: FaceSwapper, img: np.ndarray):
    """Return the largest face in the image, or None."""
    faces = swapper.face_app.get(img)
    if not faces:
        return None
    faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
               reverse=True)
    return faces[0]


def pose_yaw_pitch(face) -> tuple[float, float]:
    """Approximate yaw/pitch in degrees from 5 keypoints."""
    if not hasattr(face, 'pose'):
        # insightface 0.7 carries `pose` = (yaw, pitch, roll) on some
        # versions; fall back to keypoint geometry otherwise.
        kps = face.kps
        eye_l, eye_r, nose, mo_l, mo_r = kps
        face_w = float(np.linalg.norm(eye_r - eye_l))
        yaw_proxy = float((nose[0] - 0.5 * (eye_l[0] + eye_r[0])) / max(face_w, 1.0))
        pitch_proxy = float((nose[1] - 0.5 * (eye_l[1] + eye_r[1])) / max(face_w, 1.0))
        return yaw_proxy * 90.0, pitch_proxy * 90.0
    yaw, pitch, _roll = face.pose
    return float(yaw), float(pitch)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32).flatten()
    b = b.astype(np.float32).flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--portraits-root', default=str(Path.home() / 'Downloads'),
                    help='Directory to scan for portrait images.')
    ap.add_argument('--max-pairs', type=int, default=12,
                    help='Number of cross-identity pairs to evaluate.')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--modes', default='ai,classical',
                    help='Comma-separated modes to run: ai, classical, pro.')
    ap.add_argument('--output-csv', default='paper/figures/benchmark.csv')
    ap.add_argument('--output-json', default='paper/figures/benchmark.json')
    ap.add_argument('--max-side', type=int, default=1280,
                    help='Resize the longer side of each input to this many '
                         'px to keep wall-clock manageable. The paper\'s '
                         'main hero pair runs at the original 4608x8192; '
                         'this benchmark trades absolute speed numbers for '
                         'tractability across N pairs.')
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print('[1/5] scanning for portraits...')
    portraits = list_portraits(Path(args.portraits_root))
    print(f'    found {len(portraits)} candidates under {args.portraits_root}')
    if len(portraits) < args.max_pairs * 2:
        print('WARNING: not enough portraits for the requested number of pairs')

    print('[2/5] loading FaceSwapper...')
    t0 = time.time()
    swapper = FaceSwapper()
    print(f'    ready in {time.time()-t0:.1f}s')

    print('[3/5] face-validating candidate portraits...')
    valid = []
    for p in portraits:
        img = cv2.imread(str(p))
        if img is None:
            continue
        # resize for speed
        h, w = img.shape[:2]
        if max(h, w) > args.max_side:
            scale = args.max_side / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)
        face = detect_one_face(swapper, img)
        if face is None:
            continue
        valid.append((p, img, face))
        if len(valid) >= 2 * args.max_pairs + 6:
            break
    print(f'    {len(valid)} validated portraits with detectable faces')

    if len(valid) < 4:
        print('ERROR: not enough validated portraits to build pairs')
        return 1

    print('[4/5] building cross-identity pairs...')
    pairs = []
    used = set()
    # Pair portraits whose ArcFace embeddings are far apart (different identity).
    while len(pairs) < args.max_pairs and len(used) < len(valid) - 1:
        cands = [i for i in range(len(valid)) if i not in used]
        if len(cands) < 2:
            break
        i = random.choice(cands)
        # pick a partner with low cosine to ensure cross-identity
        best_j = None
        best_score = 1.0
        for j in cands:
            if j == i:
                continue
            c = cosine(valid[i][2].normed_embedding, valid[j][2].normed_embedding)
            if c < best_score:
                best_score = c
                best_j = j
        if best_j is None:
            break
        if best_score > 0.45:
            # too similar; skip this i
            used.add(i)
            continue
        pairs.append((i, best_j, best_score))
        used.add(i)
        used.add(best_j)
    print(f'    selected {len(pairs)} cross-identity pairs '
          f'(target cosine in source/target < 0.45)')

    modes = [m.strip() for m in args.modes.split(',') if m.strip()]
    rows = []

    print('[5/5] running swaps...')
    for k, (si, ti, src_tgt_cos) in enumerate(pairs):
        src_path, src_img, src_face = valid[si]
        tgt_path, tgt_img, tgt_face = valid[ti]
        src_yaw, src_pitch = pose_yaw_pitch(src_face)
        tgt_yaw, tgt_pitch = pose_yaw_pitch(tgt_face)
        pose_diff = max(abs(src_yaw - tgt_yaw), abs(src_pitch - tgt_pitch))

        for mode in modes:
            tag = 'classical' if mode == 'classical' else mode
            internal_mode = 'photoshop' if mode == 'classical' else mode
            try:
                t0 = time.time()
                result = swapper.swap(src_img, tgt_img, mode=internal_mode,
                                      hd=True, enhance=True)
                dt = time.time() - t0
                # Re-detect the face in the result for embedding cosines
                r_face = detect_one_face(swapper, result)
                if r_face is None:
                    a_s = a_t = float('nan')
                else:
                    a_s = cosine(src_face.normed_embedding, r_face.normed_embedding)
                    a_t = cosine(tgt_face.normed_embedding, r_face.normed_embedding)
                g = a_t - a_s
                rows.append({
                    'pair': k,
                    'mode': tag,
                    'src': src_path.name,
                    'tgt': tgt_path.name,
                    'src_tgt_arc_cos': round(src_tgt_cos, 4),
                    'pose_diff_deg': round(pose_diff, 2),
                    'alpha_s': round(a_s, 4) if a_s == a_s else None,
                    'alpha_t': round(a_t, 4) if a_t == a_t else None,
                    'ghost_score': round(g, 4) if g == g else None,
                    'time_s': round(dt, 2),
                })
                print(f'  pair {k:2d} {tag:10s} pose={pose_diff:5.1f}° '
                      f'α_s={a_s:.3f} α_t={a_t:.3f} g={g:+.3f} ({dt:.1f}s)')
            except Exception as e:
                print(f'  pair {k} {mode} FAILED: {e}')
                rows.append({
                    'pair': k, 'mode': tag,
                    'src': src_path.name, 'tgt': tgt_path.name,
                    'error': str(e),
                })

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, 'w', newline='') as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r.keys()}))
            w.writeheader()
            w.writerows(rows)
    with open(args.output_json, 'w') as f:
        json.dump({'pairs': len(pairs), 'rows': rows}, f, indent=2)

    print(f'\nWrote {args.output_csv} and {args.output_json}')

    # Aggregate stats
    if rows:
        print('\nAggregate:')
        for mode in modes:
            tag = 'classical' if mode == 'classical' else mode
            vals_as = [r['alpha_s'] for r in rows
                       if r.get('mode') == tag and r.get('alpha_s') is not None]
            vals_g = [r['ghost_score'] for r in rows
                      if r.get('mode') == tag and r.get('ghost_score') is not None]
            vals_t = [r['time_s'] for r in rows
                      if r.get('mode') == tag and 'time_s' in r]
            if vals_as:
                print(f'  {tag:10s}  α_s = {np.mean(vals_as):.3f} ± {np.std(vals_as):.3f}  '
                      f'g = {np.mean(vals_g):+.3f} ± {np.std(vals_g):.3f}  '
                      f't = {np.mean(vals_t):.1f}s  N = {len(vals_as)}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
