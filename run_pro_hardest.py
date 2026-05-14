"""Run Pro mode on the HARDEST 10 pairs from the N=60 fast benchmark.

"Hardest" = pairs where Classical Compositing produced the largest
positive ghost score (worst failures) or the smallest α_s. This is the
population where Pro mode is supposed to help — running it here is a
fairer test than random-10.
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
import pro_swap


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
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max-side', type=int, default=1280)
    ap.add_argument('--hardest-from', default='paper/figures/benchmark_n200.json',
                    help='Benchmark JSON to select hardest pairs from. Falls back to '
                         'benchmark_fast.json if benchmark_n200.json absent.')
    ap.add_argument('--n-hardest', type=int, default=10)
    ap.add_argument('--output-csv', default='paper/figures/benchmark_pro_hardest.csv')
    ap.add_argument('--output-json', default='paper/figures/benchmark_pro_hardest.json')
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load source benchmark to pick hardest pairs (those where Classical failed worst).
    src_path = args.hardest_from
    if not Path(src_path).exists():
        src_path = 'paper/figures/benchmark_fast.json'
    print(f'Selecting hardest from {src_path}')
    src_data = json.load(open(src_path))
    src_rows = src_data['rows']
    classical = {r['pair']: r for r in src_rows
                 if r.get('mode') == 'classical' and r.get('ghost_score') is not None}
    # rank by ghost_score descending (largest g = worst failure)
    ranked = sorted(classical.items(), key=lambda kv: kv[1]['ghost_score'], reverse=True)
    hardest = [p for p, _ in ranked[:args.n_hardest]]
    print(f'Selected pair indices: {hardest}')
    print('  with g values:', [round(classical[p]["ghost_score"], 3) for p in hardest])

    # Recreate the same pair list deterministically (same seed=42 + same scan order)
    print('\nLoading FaceSwapper + face-validating portraits...')
    fs = FaceSwapper()
    portraits = list_portraits(Path(args.portraits_root))
    valid = []
    for p in portraits:
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        if max(h, w) > args.max_side:
            s = args.max_side / max(h, w)
            img = cv2.resize(img, (int(w * s), int(h * s)),
                             interpolation=cv2.INTER_AREA)
        face = detect_one_face(fs, img)
        if face is None:
            continue
        valid.append((p, img, face))
        if len(valid) >= 120:
            break
    print(f'  {len(valid)} validated')

    # Re-build the pair list with same seed
    pairs = []
    seen = set()
    attempts = 0
    while len(pairs) < max(hardest) + 1 and attempts < 4000:
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
    print(f'  rebuilt pair list ({len(pairs)} pairs)')

    rows = []
    print(f'\nRunning Pro on {len(hardest)} hardest pairs...')
    for k in hardest:
        if k >= len(pairs):
            print(f'  pair {k} not reachable (only {len(pairs)} pairs rebuilt)')
            continue
        si, ti, src_tgt_cos = pairs[k]
        src_path, src_img, src_face = valid[si]
        tgt_path, tgt_img, tgt_face = valid[ti]
        prev_g = classical[k]['ghost_score']
        prev_a = classical[k]['alpha_s']
        try:
            t0 = time.time()
            pr = pro_swap.pro_swap(fs, src_img, tgt_img, hd=True, enhance=True)
            result = pr.image
            dt = time.time() - t0
            r_face = detect_one_face(fs, result)
            if r_face is None:
                a_s = a_t = float('nan')
            else:
                a_s = cosine(src_face.normed_embedding, r_face.normed_embedding)
                a_t = cosine(tgt_face.normed_embedding, r_face.normed_embedding)
            g = a_t - a_s
            rows.append({
                'pair': k, 'mode': 'pro_new_router',
                'src': src_path.name, 'tgt': tgt_path.name,
                'pose_diff_deg': classical[k].get('pose_diff_deg'),
                'alpha_s': round(a_s, 4) if a_s == a_s else None,
                'alpha_t': round(a_t, 4) if a_t == a_t else None,
                'ghost_score': round(g, 4) if g == g else None,
                'classical_alpha_s': round(prev_a, 4),
                'classical_ghost_score': round(prev_g, 4),
                'delta_alpha_s': round((a_s if a_s == a_s else 0) - prev_a, 4),
                'time_s': round(dt, 2),
            })
            print(f'  pair {k:3d}  Cl→Pro α_s: {prev_a:.3f}→{a_s:.3f}  '
                  f'Cl→Pro g: {prev_g:+.3f}→{g:+.3f}  Δα_s={a_s-prev_a:+.3f}  ({dt:.0f}s)')
        except Exception as e:
            print(f'  pair {k} FAILED: {e}')
            rows.append({'pair': k, 'mode': 'pro_new_router',
                         'error': str(e)})
        fs.cleanup()

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, 'w', newline='') as f:
        if rows:
            keys = sorted({k for r in rows for k in r.keys()})
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    with open(args.output_json, 'w') as f:
        json.dump({'rows': rows}, f, indent=2)

    valid_rows = [r for r in rows if r.get('alpha_s') is not None]
    if valid_rows:
        deltas = [r['delta_alpha_s'] for r in valid_rows]
        rescues = [r for r in valid_rows if r['delta_alpha_s'] > 0.1]
        print(f'\n=== Aggregate on hardest pairs ===')
        print(f'  Δα_s (Pro - Classical):  mean = {np.mean(deltas):+.3f}, '
              f'median = {np.median(deltas):+.3f}')
        print(f'  rescues (Δα_s > +0.1):    {len(rescues)} / {len(valid_rows)}')
        print(f'  Pro α_s ≥ 0.6 (match):    '
              f'{sum(1 for r in valid_rows if r["alpha_s"] >= 0.6)} / {len(valid_rows)}')

    print(f'\nWrote {args.output_csv}')


if __name__ == '__main__':
    sys.exit(main())
