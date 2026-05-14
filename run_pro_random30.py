"""Run Pro mode on 30 RANDOM pairs from the N=200 benchmark (not just the
hardest), to characterise Pro's average-case behaviour and show that the
10/10 rescue on the hardest pairs is not cherry-picking."""

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
    ap.add_argument('--n-random', type=int, default=30)
    ap.add_argument('--source-bench', default='paper/figures/benchmark_n200.json')
    ap.add_argument('--output-csv', default='paper/figures/benchmark_pro_random30.csv')
    ap.add_argument('--output-json', default='paper/figures/benchmark_pro_random30.json')
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load reference benchmark to know how to rebuild the same pair list
    bench = json.load(open(args.source_bench))['rows']
    classical = {r['pair']: r for r in bench
                 if r.get('mode') == 'classical' and r.get('ghost_score') is not None}
    n_pairs_total = max(classical.keys()) + 1
    # Pick a fresh random sub-set of 30 pairs — disjoint from the hardest 10
    random.seed(123)  # different seed for selecting these 30
    hardest_set = set(sorted(classical.keys(),
                              key=lambda k: -classical[k]['ghost_score'])[:10])
    rest = [k for k in classical if k not in hardest_set]
    random.shuffle(rest)
    sel = sorted(rest[:args.n_random])
    print(f'Selected {len(sel)} random pairs (disjoint from hardest 10): {sel}')
    print(f'Their Classical g distribution: '
          f'min={min(classical[k]["ghost_score"] for k in sel):.3f}, '
          f'max={max(classical[k]["ghost_score"] for k in sel):.3f}')

    # Rebuild the pair list deterministically
    random.seed(args.seed)
    np.random.seed(args.seed)
    print('\nLoading FaceSwapper + face-validating portraits...')
    fs = FaceSwapper()
    portraits = list_portraits(Path(args.portraits_root))
    random.shuffle(portraits)
    valid = []
    for p in portraits:
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        if max(h, w) > 512:
            s_small = 512 / max(h, w)
            img_small = cv2.resize(img, (int(w * s_small), int(h * s_small)),
                                   interpolation=cv2.INTER_AREA)
        else:
            img_small = img
        if detect_one_face(fs, img_small) is None:
            continue
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

    # Rebuild pairs with same seed=42
    pairs = []
    seen = set()
    attempts = 0
    while len(pairs) < n_pairs_total and attempts < n_pairs_total * 30:
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
    print(f'  rebuilt {len(pairs)} pairs')

    rows = []
    print(f'\nRunning Pro on {len(sel)} random pairs...')
    for k in sel:
        if k >= len(pairs):
            print(f'  pair {k} unreachable (only {len(pairs)} pairs rebuilt)')
            continue
        si, ti, _ = pairs[k]
        src_path, src_img, src_face = valid[si]
        tgt_path, tgt_img, tgt_face = valid[ti]
        cl_row = classical[k]
        prev_a = cl_row['alpha_s']
        prev_g = cl_row['ghost_score']
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
                'pair': k, 'mode': 'pro_random',
                'src': src_path.name, 'tgt': tgt_path.name,
                'pose_diff_deg': cl_row.get('pose_diff_deg'),
                'classical_alpha_s': round(prev_a, 4),
                'classical_ghost_score': round(prev_g, 4),
                'alpha_s': round(a_s, 4) if a_s == a_s else None,
                'alpha_t': round(a_t, 4) if a_t == a_t else None,
                'ghost_score': round(g, 4) if g == g else None,
                'delta_alpha_s': round((a_s if a_s == a_s else 0) - prev_a, 4),
                'time_s': round(dt, 2),
            })
            print(f'  pair {k:3d}  Cl→Pro α_s: {prev_a:+.3f}→{a_s:+.3f}  '
                  f'Cl→Pro g: {prev_g:+.3f}→{g:+.3f}  '
                  f'Δ={a_s-prev_a:+.3f}  ({dt:.0f}s)')
        except Exception as e:
            print(f'  pair {k} FAILED: {e}')
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
        d = np.array([r['delta_alpha_s'] for r in valid_rows])
        a = np.array([r['alpha_s'] for r in valid_rows])
        rescues = sum(1 for x in d if x > 0.1)
        no_harm = sum(1 for x in d if x >= -0.05)
        ci = 1.96 * d.std() / np.sqrt(len(d))
        print(f'\n=== Pro on {len(valid_rows)} random pairs ===')
        print(f'  Pro α_s         = {a.mean():.3f} ± {a.std():.3f}')
        print(f'  Δα_s (Pro-Cl)   = {d.mean():+.3f} ± {d.std():.3f} (95% CI ±{ci:.3f})')
        print(f'  rescues (Δ>0.1): {rescues}/{len(valid_rows)}')
        print(f'  no-harm (Δ≥-0.05): {no_harm}/{len(valid_rows)}')
        print(f'  matches (α_s≥0.6): {sum(1 for v in a if v>=0.6)}/{len(valid_rows)}')


if __name__ == '__main__':
    sys.exit(main())
