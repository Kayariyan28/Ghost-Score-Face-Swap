"""Re-run SimSwap-256 on the same 200 pairs and evaluate identity with
BOTH ArcFace heads:

  * InsightFace `buffalo_l/w600k_r50` (what every row of Table III uses).
  * SimSwap's OWN ArcFace head (the JIT module it was trained against).

This addresses the standard reviewer objection that evaluating SimSwap
with an ArcFace head it never saw at training time depresses its
numerical score. By reporting both:

  - the SimSwap-head numbers should be in the published-paper range
    (~0.5–0.6), demonstrating the checkpoint is loaded correctly;
  - the InsightFace-head numbers stay the apples-to-apples paired
    comparison signal because every row uses the same yardstick.
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
import torch
import torch.nn.functional as F

from face_swap import FaceSwapper

sys.path.insert(0, str(Path(__file__).parent / 'baselines'))
from simswap_inference import SimSwapWrapper, _ARCFACE_DST_112


def list_portraits(root: Path, min_size_kb: int = 200):
    exts = {'.png', '.jpg', '.jpeg'}
    out = []
    for p in root.rglob('*'):
        if p.suffix.lower() not in exts:
            continue
        try:
            if p.stat().st_size < min_size_kb * 1024:
                continue
        except OSError:
            continue
        name = p.name.lower()
        if any(t in name for t in ('abstract','art_nouveau','oilpainting',
                                    'legs_hanging','screenshot','cyc_world','face-swap')):
            continue
        out.append(p)
    return out


def cos(a, b):
    a = a.astype(np.float32).flatten()
    b = b.astype(np.float32).flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def detect(fs, img):
    f = fs.face_app.get(img)
    if not f:
        return None
    f.sort(key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
    return f[0]


def simswap_id_vec(arc_jit, img_bgr, kps_5pt):
    """Compute SimSwap-ArcFace embedding from a face crop aligned to the
    SimSwap canonical 5-point template (the SAME alignment as training)."""
    dst = _ARCFACE_DST_112  # 112x112 template
    M, _ = cv2.estimateAffinePartial2D(kps_5pt.astype(np.float32), dst,
                                       method=cv2.LMEDS)
    aligned = cv2.warpAffine(img_bgr, M, (112, 112),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
    rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    t = (t - mean) / std
    with torch.no_grad():
        emb = arc_jit(t)
    emb = F.normalize(emb, p=2, dim=1).squeeze(0).cpu().numpy()
    return emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--portraits-root', default=str(Path.home() / 'Downloads'))
    ap.add_argument('--max-pairs', type=int, default=200)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max-side', type=int, default=1280)
    ap.add_argument('--output-csv', default='paper/figures/benchmark_simswap_dualhead.csv')
    ap.add_argument('--output-json', default='paper/figures/benchmark_simswap_dualhead.json')
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)

    fs = FaceSwapper()
    ss = SimSwapWrapper()
    print('[ready] FaceSwapper + SimSwapWrapper + ArcFace JIT loaded')

    portraits = list_portraits(Path(args.portraits_root))
    random.shuffle(portraits)
    valid = []
    for p in portraits:
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        if max(h, w) > 512:
            s = 512 / max(h, w)
            small = cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
        else:
            small = img
        if detect(fs, small) is None:
            continue
        if max(h, w) > args.max_side:
            s = args.max_side / max(h, w)
            img_run = cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
        else:
            img_run = img
        face = detect(fs, img_run)
        if face is None:
            continue
        valid.append((p, img_run, face))
        if len(valid) >= 120:
            break
    print(f'[ready] {len(valid)} validated portraits')

    pairs = []; seen = set(); attempts = 0
    while len(pairs) < args.max_pairs and attempts < args.max_pairs * 30:
        attempts += 1
        i, j = random.sample(range(len(valid)), 2)
        key = tuple(sorted((i, j)))
        if key in seen:
            continue
        c = cos(valid[i][2].normed_embedding, valid[j][2].normed_embedding)
        if c > 0.45:
            continue
        pairs.append((i, j, c)); seen.add(key)
    print(f'[ready] {len(pairs)} cross-identity pairs (same seed=42)')

    rows = []
    for k, (si, ti, _) in enumerate(pairs):
        src_path, src_img, src_face = valid[si]
        tgt_path, tgt_img, tgt_face = valid[ti]
        try:
            t0 = time.time()
            out = ss.swap(src_img, tgt_img, src_face.kps, tgt_face.kps)
            dt = time.time() - t0

            r_face = detect(fs, out)
            if r_face is None:
                continue

            # Head 1: InsightFace w600k_r50 (the table-wide yardstick)
            a_s_if = cos(src_face.normed_embedding, r_face.normed_embedding)
            a_t_if = cos(tgt_face.normed_embedding, r_face.normed_embedding)

            # Head 2: SimSwap's own ArcFace JIT module on the same crops
            arc = ss.arc
            v_s_ss = simswap_id_vec(arc, src_img, src_face.kps)
            v_t_ss = simswap_id_vec(arc, tgt_img, tgt_face.kps)
            v_r_ss = simswap_id_vec(arc, out, r_face.kps)
            a_s_ss = cos(v_s_ss, v_r_ss)
            a_t_ss = cos(v_t_ss, v_r_ss)

            rows.append({
                'pair': k, 'mode': 'simswap_256',
                'src': src_path.name, 'tgt': tgt_path.name,
                # InsightFace head (table III evaluator)
                'alpha_s_insightface': round(a_s_if, 4),
                'alpha_t_insightface': round(a_t_if, 4),
                'g_insightface':       round(a_t_if - a_s_if, 4),
                # SimSwap's own ArcFace head (training-distribution evaluator)
                'alpha_s_simswap':     round(a_s_ss, 4),
                'alpha_t_simswap':     round(a_t_ss, 4),
                'g_simswap':           round(a_t_ss - a_s_ss, 4),
                'time_s': round(dt, 2),
            })
            print(f'  pair {k:3d}  IF: α_s={a_s_if:+.3f} g={a_t_if-a_s_if:+.3f}  '
                  f'SS-head: α_s={a_s_ss:+.3f} g={a_t_ss-a_s_ss:+.3f}  ({dt:.2f}s)')
        except Exception as e:
            print(f'  pair {k} FAILED: {e}')

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, 'w', newline='') as f:
        if rows:
            keys = sorted({k for r in rows for k in r.keys()})
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(rows)
    with open(args.output_json, 'w') as f:
        json.dump({'rows': rows}, f, indent=2)

    if rows:
        a_if = np.array([r['alpha_s_insightface'] for r in rows])
        a_ss = np.array([r['alpha_s_simswap'] for r in rows])
        g_if = np.array([r['g_insightface'] for r in rows])
        g_ss = np.array([r['g_simswap'] for r in rows])
        ci_if = 1.96 * a_if.std() / np.sqrt(len(a_if))
        ci_ss = 1.96 * a_ss.std() / np.sqrt(len(a_ss))
        m_if = sum(1 for v in a_if if v >= 0.6)
        m_ss = sum(1 for v in a_ss if v >= 0.6)
        print(f'\n=== SimSwap-256 dual-head on {len(rows)} pairs ===')
        print(f'InsightFace head (table III yardstick):')
        print(f'  α_s = {a_if.mean():.3f} ± {a_if.std():.3f} (95%CI ±{ci_if:.3f})')
        print(f'  g   = {g_if.mean():+.3f} ± {g_if.std():.3f}')
        print(f'  match ({a_if.mean():.3f}≥0.6): {m_if}/{len(a_if)} = {100*m_if/len(a_if):.1f}%')
        print(f'SimSwap own ArcFace head (training-distribution):')
        print(f'  α_s = {a_ss.mean():.3f} ± {a_ss.std():.3f} (95%CI ±{ci_ss:.3f})')
        print(f'  g   = {g_ss.mean():+.3f} ± {g_ss.std():.3f}')
        print(f'  match: {m_ss}/{len(a_ss)} = {100*m_ss/len(a_ss):.1f}%')


if __name__ == '__main__':
    sys.exit(main())
