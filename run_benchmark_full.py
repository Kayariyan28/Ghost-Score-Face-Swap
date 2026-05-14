"""Extended multi-pair benchmark for T-PAMI-class evidence.

For each cross-identity pair we evaluate FOUR pipelines:
  * raw_inswapper  -- baseline: inswapper_128 ONLY, no GFPGAN, no Laplacian
                       graft, no hair composite. This is the "published prior
                       art on a tee" reference point.
  * ai             -- our AI Synthesis mode (raw_inswapper + GFPGAN).
  * classical      -- our Classical Compositing mode.
  * pro            -- our Pro adaptive mode (multi-pipeline auto-best,
                       optionally falling back to HardPoseReplaceMode).

We collect per-mode: ArcFace identity (alpha_s), target leak (alpha_t),
ghost score (g), wall-clock time, AND an LPIPS perceptual distance
between the result face crop and the source face crop. LPIPS gives an
independent third-party perceptual signal (the AlexNet/VGG features
were not trained on the swap distribution) that is a defensible proxy
for a human study within the constraints of an automated benchmark.

We also re-time the first N pairs forcing the entire stack onto CPU,
so we can report a cross-platform-comparable runtime number (CPU is
roughly comparable across the Apple Silicon lineup, since GPU/MPS is
what scales between M1 Pro / M2 / M4 Max).

Run:
    source venv/bin/activate
    python run_benchmark_full.py --max-pairs 50 --cpu-pairs 6
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

from face_swap import FaceSwapper


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
        bad_tokens = ('abstract', 'art_nouveau', 'oilpainting', 'legs_hanging',
                      'screenshot', 'cyc_world', 'face-swap')
        if any(t in name for t in bad_tokens):
            continue
        out.append(p)
    return out


def detect_one_face(swapper: FaceSwapper, img):
    faces = swapper.face_app.get(img)
    if not faces:
        return None
    faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
               reverse=True)
    return faces[0]


def pose_yaw_pitch(face):
    if not hasattr(face, 'pose'):
        kps = face.kps
        eye_l, eye_r, nose, _mol, _mor = kps
        face_w = float(np.linalg.norm(eye_r - eye_l))
        yaw_proxy = float((nose[0] - 0.5 * (eye_l[0] + eye_r[0])) / max(face_w, 1.0))
        pitch_proxy = float((nose[1] - 0.5 * (eye_l[1] + eye_r[1])) / max(face_w, 1.0))
        return yaw_proxy * 90.0, pitch_proxy * 90.0
    yaw, pitch, _r = face.pose
    return float(yaw), float(pitch)


def cosine(a, b):
    a = a.astype(np.float32).flatten()
    b = b.astype(np.float32).flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def face_crop_from(img, face, size=256, pad=0.2):
    if face is None:
        return None
    h, w = img.shape[:2]
    x1, y1, x2, y2 = face.bbox.astype(int)
    fw, fh = x2 - x1, y2 - y1
    cx1 = max(0, int(x1 - fw * pad))
    cy1 = max(0, int(y1 - fh * pad))
    cx2 = min(w, int(x2 + fw * pad))
    cy2 = min(h, int(y2 + fh * pad))
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    crop = img[cy1:cy2, cx1:cx2]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def lpips_distance(net, a, b):
    """LPIPS distance between two BGR uint8 face crops."""
    if a is None or b is None:
        return float('nan')

    def to_tensor(x):
        # BGR -> RGB, [0,255] -> [-1,1], HWC -> 1CHW
        rgb = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)

    with torch.no_grad():
        d = net(to_tensor(a), to_tensor(b))
    return float(d.item())


def raw_inswapper_swap(swapper, src_img, tgt_img):
    """Baseline: bare inswapper_128, no GFPGAN, no graft, no hair composite."""
    src_faces = swapper.face_app.get(src_img)
    tgt_faces = swapper.face_app.get(tgt_img)
    if not src_faces or not tgt_faces:
        raise ValueError('Face detection failed in raw baseline.')
    src_faces.sort(key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]),
                   reverse=True)
    tgt_faces.sort(key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]),
                   reverse=True)
    result = swapper.swapper.get(tgt_img.copy(), tgt_faces[0], src_faces[0],
                                 paste_back=True)
    return result


def run_one(swapper, src_img, tgt_img, mode):
    """Run a single swap. Returns (result_img, elapsed_s)."""
    t0 = time.time()
    if mode == 'raw_inswapper':
        result = raw_inswapper_swap(swapper, src_img, tgt_img)
    elif mode == 'ai':
        result = swapper.swap(src_img, tgt_img, mode='ai', hd=True, enhance=True)
    elif mode == 'classical':
        result = swapper.swap(src_img, tgt_img, mode='photoshop', hd=True, enhance=True)
    elif mode == 'pro':
        import pro_swap
        pr = pro_swap.pro_swap(swapper, src_img, tgt_img, hd=True, enhance=True)
        result = pr.image
    else:
        raise ValueError(f'unknown mode: {mode}')
    return result, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--portraits-root', default=str(Path.home() / 'Downloads'))
    ap.add_argument('--max-pairs', type=int, default=50)
    ap.add_argument('--cpu-pairs', type=int, default=6,
                    help='Number of pairs to additionally time with the GFPGAN '
                         'stack forced onto CPU as a portable-runtime proxy.')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--modes', default='raw_inswapper,ai,classical,pro')
    ap.add_argument('--output-csv', default='paper/figures/benchmark_full.csv')
    ap.add_argument('--output-json', default='paper/figures/benchmark_full.json')
    ap.add_argument('--max-side', type=int, default=1280)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print('[1/6] scanning portraits...')
    portraits = list_portraits(Path(args.portraits_root))
    print(f'    {len(portraits)} candidates under {args.portraits_root}')

    print('[2/6] loading FaceSwapper...')
    t0 = time.time()
    swapper = FaceSwapper()
    print(f'    ready in {time.time()-t0:.1f}s')

    print('[3/6] loading LPIPS perceptual models (AlexNet + VGG)...')
    import lpips
    lpips_alex = lpips.LPIPS(net='alex')
    lpips_alex.eval()
    lpips_vgg = lpips.LPIPS(net='vgg')
    lpips_vgg.eval()
    print('    LPIPS ready (AlexNet + VGG backbones)')

    print('[4/6] face-validating portraits (downsampled to 512px for speed)...')
    valid = []
    # Pre-shuffle for diversity; then face-detect at SMALL size (512 px) which is
    # ~5x faster than 1280 — InsightFace handles 512 face detection in <2 s.
    random.shuffle(portraits)
    for p in portraits:
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        # Downsample for face validation (re-load full-size later for actual swap)
        max_validate = 512
        if max(h, w) > max_validate:
            scale = max_validate / max(h, w)
            img_small = cv2.resize(img, (int(w * scale), int(h * scale)),
                                   interpolation=cv2.INTER_AREA)
        else:
            img_small = img
        face = detect_one_face(swapper, img_small)
        if face is None:
            continue
        # Re-load and resize to benchmark resolution (max_side) for swap runs
        if max(h, w) > args.max_side:
            scale = args.max_side / max(h, w)
            img_run = cv2.resize(img, (int(w * scale), int(h * scale)),
                                 interpolation=cv2.INTER_AREA)
        else:
            img_run = img
        # Re-detect on benchmark-size image so embedding/landmarks are consistent
        face = detect_one_face(swapper, img_run)
        if face is None:
            continue
        valid.append((p, img_run, face))
        if len(valid) >= 120:
            break
    print(f'    {len(valid)} validated portraits')

    if len(valid) < 4:
        print('ERROR: not enough validated portraits')
        return 1

    print('[5/6] building cross-identity pairs (with replacement)...')
    pairs = []
    attempts = 0
    seen = set()
    while len(pairs) < args.max_pairs and attempts < args.max_pairs * 20:
        attempts += 1
        i, j = random.sample(range(len(valid)), 2)
        key = tuple(sorted((i, j)))
        if key in seen:
            continue
        c = cosine(valid[i][2].normed_embedding, valid[j][2].normed_embedding)
        if c > 0.45:                  # too similar -> probably same identity
            continue
        pairs.append((i, j, c))
        seen.add(key)
    print(f'    selected {len(pairs)} cross-identity pairs (cos<0.45)')

    modes = [m.strip() for m in args.modes.split(',') if m.strip()]
    rows = []

    print(f'[6/6] running swaps over {len(pairs)} pairs x {len(modes)} modes...')
    for k, (si, ti, src_tgt_cos) in enumerate(pairs):
        src_path, src_img, src_face = valid[si]
        tgt_path, tgt_img, tgt_face = valid[ti]
        src_yaw, src_pitch = pose_yaw_pitch(src_face)
        tgt_yaw, tgt_pitch = pose_yaw_pitch(tgt_face)
        pose_diff = max(abs(src_yaw - tgt_yaw), abs(src_pitch - tgt_pitch))
        src_crop = face_crop_from(src_img, src_face)

        for mode in modes:
            try:
                result, dt = run_one(swapper, src_img, tgt_img, mode)
                r_face = detect_one_face(swapper, result)
                if r_face is None:
                    a_s = a_t = lp_a = lp_v = float('nan')
                else:
                    a_s = cosine(src_face.normed_embedding, r_face.normed_embedding)
                    a_t = cosine(tgt_face.normed_embedding, r_face.normed_embedding)
                    r_crop = face_crop_from(result, r_face)
                    lp_a = lpips_distance(lpips_alex, src_crop, r_crop)
                    lp_v = lpips_distance(lpips_vgg, src_crop, r_crop)
                g = a_t - a_s
                rows.append({
                    'pair': k, 'mode': mode,
                    'src': src_path.name, 'tgt': tgt_path.name,
                    'src_tgt_arc_cos': round(src_tgt_cos, 4),
                    'pose_diff_deg': round(pose_diff, 2),
                    'src_age': getattr(src_face, 'age', None),
                    'src_gender': int(getattr(src_face, 'gender', -1)),
                    'tgt_age': getattr(tgt_face, 'age', None),
                    'tgt_gender': int(getattr(tgt_face, 'gender', -1)),
                    'alpha_s': round(a_s, 4) if a_s == a_s else None,
                    'alpha_t': round(a_t, 4) if a_t == a_t else None,
                    'ghost_score': round(g, 4) if g == g else None,
                    'lpips_alex': round(lp_a, 4) if lp_a == lp_a else None,
                    'lpips_vgg': round(lp_v, 4) if lp_v == lp_v else None,
                    'time_s': round(dt, 2),
                })
                print(f'  pair {k:3d} {mode:14s} pose={pose_diff:5.1f}° '
                      f'α_s={a_s:.3f} α_t={a_t:.3f} g={g:+.3f} '
                      f'LP_a={lp_a:.3f} LP_v={lp_v:.3f} ({dt:.1f}s)')
            except Exception as e:
                print(f'  pair {k} {mode} FAILED: {e}')
                rows.append({'pair': k, 'mode': mode,
                             'src': src_path.name, 'tgt': tgt_path.name,
                             'error': str(e)})
            swapper.cleanup()

    # ------- Cross-platform CPU timing pass --------
    if args.cpu_pairs and 'classical' in modes:
        print(f'\n[CPU timing] running classical on {args.cpu_pairs} pairs '
              'with GFPGAN forced to CPU as portable-runtime proxy...')
        try:
            # The GFPGAN enhancer is held inside the swapper. Force its
            # device to CPU. This estimates what runtime would look like on
            # a machine without MPS / when the Metal backend is unavailable
            # (the bottleneck on Apple Silicon is GFPGAN, not the InsightFace
            # ONNX stack which is already on CPU).
            old_device = None
            if hasattr(swapper, 'enhancer') and swapper.enhancer is not None:
                old_device = getattr(swapper.enhancer.gfpgan, 'device', None)
                swapper.enhancer.gfpgan = swapper.enhancer.gfpgan.cpu()
                swapper.enhancer.device = torch.device('cpu')
            for k in range(min(args.cpu_pairs, len(pairs))):
                si, ti, _ = pairs[k]
                src_path, src_img, src_face = valid[si]
                tgt_path, tgt_img, tgt_face = valid[ti]
                try:
                    _, dt = run_one(swapper, src_img, tgt_img, 'classical')
                    rows.append({'pair': k, 'mode': 'classical_cpu_only',
                                 'src': src_path.name, 'tgt': tgt_path.name,
                                 'time_s': round(dt, 2)})
                    print(f'  CPU pair {k:3d} classical_cpu_only  ({dt:.1f}s)')
                except Exception as e:
                    print(f'  CPU pair {k} FAILED: {e}')
                swapper.cleanup()
        except Exception as e:
            print(f'CPU timing pass skipped: {e}')

    # ------- write outputs --------
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

    # ------- aggregate stats --------
    print('\nAggregate stats:')
    for mode in modes + ['classical_cpu_only']:
        a_s_vals = [r['alpha_s'] for r in rows if r.get('mode')==mode
                    and r.get('alpha_s') is not None]
        g_vals = [r['ghost_score'] for r in rows if r.get('mode')==mode
                  and r.get('ghost_score') is not None]
        lp_vals = [r['lpips_alex'] for r in rows if r.get('mode')==mode
                   and r.get('lpips_alex') is not None]
        lpv_vals = [r['lpips_vgg'] for r in rows if r.get('mode')==mode
                    and r.get('lpips_vgg') is not None]
        t_vals = [r['time_s'] for r in rows if r.get('mode')==mode
                  and 'time_s' in r]
        if not t_vals:
            continue
        line = f'  {mode:20s}  N={len(t_vals):3d}  t={np.mean(t_vals):5.1f}s'
        if a_s_vals:
            line += f'  α_s={np.mean(a_s_vals):.3f}±{np.std(a_s_vals):.3f}'
        if g_vals:
            line += f'  g={np.mean(g_vals):+.3f}±{np.std(g_vals):.3f}'
        if lp_vals:
            line += f'  LPIPS_a={np.mean(lp_vals):.3f}'
        if lpv_vals:
            line += f'  LPIPS_v={np.mean(lpv_vals):.3f}'
        print(line)

    return 0


if __name__ == '__main__':
    sys.exit(main())
