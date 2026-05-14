"""Run SimSwap-256 as an external baseline on the SAME 200 pairs (seed=42)
that the main benchmark uses.

This produces an apples-to-apples row for Table III of the paper:
SimSwap is the published prior-art face-swap GAN we compare against.
ArcFace cosines are computed with the EXACT SAME InsightFace recognition
model used for our internal modes, so the numbers are directly
comparable.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from face_swap import FaceSwapper

sys.path.insert(0, str(Path(__file__).parent / 'baselines'))
from simswap_inference import SimSwapWrapper


def list_portraits(root: Path, min_size_kb: int = 200) -> list[Path]:
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
        name = p.name.lower()
        bad = ('abstract', 'art_nouveau', 'oilpainting', 'legs_hanging',
               'screenshot', 'cyc_world', 'face-swap')
        if any(t in name for t in bad):
            continue
        out.append(p)
    return out


def cosine(a, b):
    a = a.astype(np.float32).flatten()
    b = b.astype(np.float32).flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def detect_one_face(swapper, img):
    faces = swapper.face_app.get(img)
    if not faces:
        return None
    faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
               reverse=True)
    return faces[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--portraits-root', default=str(Path.home() / 'Downloads'))
    ap.add_argument('--max-pairs', type=int, default=200)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max-side', type=int, default=1280)
    ap.add_argument('--output-csv', default='paper/figures/benchmark_simswap.csv')
    ap.add_argument('--output-json', default='paper/figures/benchmark_simswap.json')
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print('[1/4] scanning + loading models...')
    portraits = list_portraits(Path(args.portraits_root))
    fs = FaceSwapper()
    ss = SimSwapWrapper()
    print(f'    ready, {len(portraits)} portrait candidates')

    print('[2/4] face-validating portraits (512px for speed, cap at 120)...')
    valid = []
    random.shuffle(portraits)
    for p in portraits:
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        # Fast pre-check at 512px
        if max(h, w) > 512:
            s_small = 512 / max(h, w)
            img_small = cv2.resize(img, (int(w * s_small), int(h * s_small)),
                                   interpolation=cv2.INTER_AREA)
        else:
            img_small = img
        if detect_one_face(fs, img_small) is None:
            continue
        # Re-load at benchmark resolution for the actual swap run
        if max(h, w) > args.max_side:
            s = args.max_side / max(h, w)
            img_run = cv2.resize(img, (int(w * s), int(h * s)),
                                 interpolation=cv2.INTER_AREA)
        else:
            img_run = img
        face = detect_one_face(fs, img_run)
        if face is None:
            continue
        valid.append((p, img_run, face))
        if len(valid) >= 120:
            break
    print(f'    {len(valid)} validated portraits')

    print('[3/4] building cross-identity pairs (deterministic with seed=42)...')
    pairs = []
    seen = set()
    attempts = 0
    while len(pairs) < args.max_pairs and attempts < args.max_pairs * 20:
        attempts += 1
        i, j = random.sample(range(len(valid)), 2)
        key = tuple(sorted((i, j)))
        if key in seen:
            continue
        c = cosine(valid[i][2].normed_embedding, valid[j][2].normed_embedding)
        if c > 0.45:
            continue
        pairs.append((i, j, c))
        seen.add(key)
    print(f'    {len(pairs)} pairs (same as main benchmark)')

    rows = []
    print(f'[4/4] running SimSwap on {len(pairs)} pairs...')
    for k, (si, ti, src_tgt_cos) in enumerate(pairs):
        src_path, src_img, src_face = valid[si]
        tgt_path, tgt_img, tgt_face = valid[ti]
        try:
            t0 = time.time()
            out = ss.swap(src_img, tgt_img, src_face.kps, tgt_face.kps)
            dt = time.time() - t0

            r_face = detect_one_face(fs, out)
            if r_face is None:
                a_s = a_t = float('nan')
            else:
                a_s = cosine(src_face.normed_embedding, r_face.normed_embedding)
                a_t = cosine(tgt_face.normed_embedding, r_face.normed_embedding)
            g = a_t - a_s
            rows.append({
                'pair': k, 'mode': 'simswap_256',
                'src': src_path.name, 'tgt': tgt_path.name,
                'src_tgt_arc_cos': round(src_tgt_cos, 4),
                'alpha_s': round(a_s, 4) if a_s == a_s else None,
                'alpha_t': round(a_t, 4) if a_t == a_t else None,
                'ghost_score': round(g, 4) if g == g else None,
                'time_s': round(dt, 2),
            })
            print(f'  pair {k:3d} simswap_256  α_s={a_s:.3f} α_t={a_t:.3f} g={g:+.3f} ({dt:.2f}s)')
        except Exception as e:
            print(f'  pair {k} FAILED: {e}')
            rows.append({'pair': k, 'mode': 'simswap_256',
                         'src': src_path.name, 'tgt': tgt_path.name,
                         'error': str(e)})

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, 'w', newline='') as f:
        if rows:
            keys = sorted({k for r in rows for k in r.keys()})
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    with open(args.output_json, 'w') as f:
        json.dump({'pairs': len(pairs), 'rows': rows}, f, indent=2)

    print(f'\nWrote {args.output_csv} and {args.output_json}')

    # Aggregate
    vals_as = [r['alpha_s'] for r in rows if r.get('alpha_s') is not None]
    vals_g  = [r['ghost_score'] for r in rows if r.get('ghost_score') is not None]
    vals_t  = [r['time_s'] for r in rows if 'time_s' in r]
    if vals_as:
        m = sum(1 for v in vals_as if v >= 0.6)
        print(f'\nSimSwap-256 on {len(vals_as)} valid pairs:')
        print(f'  α_s = {np.mean(vals_as):.3f} ± {np.std(vals_as):.3f}')
        print(f'  g   = {np.mean(vals_g):+.3f} ± {np.std(vals_g):.3f}')
        print(f'  time = {np.mean(vals_t):.2f}±{np.std(vals_t):.2f}s')
        print(f'  match rate (α_s ≥ 0.6) = {m}/{len(vals_as)} = {100*m/len(vals_as):.1f}%')


if __name__ == '__main__':
    sys.exit(main())
