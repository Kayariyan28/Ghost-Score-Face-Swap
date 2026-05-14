"""Run Pro on ALL 200 pairs from the N=200 benchmark (full coverage)."""
from __future__ import annotations
import argparse, csv, json, random, sys, time
from pathlib import Path
import cv2, numpy as np

from face_swap import FaceSwapper
import pro_swap


def list_portraits(root: Path, min_size_kb: int = 200):
    exts = {'.png', '.jpg', '.jpeg'}; out = []
    for p in root.rglob('*'):
        if p.suffix.lower() not in exts: continue
        try:
            if p.stat().st_size < min_size_kb * 1024: continue
        except OSError: continue
        name = p.name.lower()
        if any(t in name for t in ('abstract','art_nouveau','oilpainting',
                                    'legs_hanging','screenshot','cyc_world','face-swap')):
            continue
        out.append(p)
    return out


def cos(a,b):
    a=a.astype(np.float32).flatten(); b=b.astype(np.float32).flatten()
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-8))


def detect(fs,img):
    fs2 = fs.face_app.get(img)
    if not fs2: return None
    fs2.sort(key=lambda f:(f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]),reverse=True)
    return fs2[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--portraits-root', default=str(Path.home()/'Downloads'))
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max-side', type=int, default=1280)
    ap.add_argument('--source-bench', default='paper/figures/benchmark_n200.json')
    ap.add_argument('--output-csv', default='paper/figures/benchmark_pro_all200.csv')
    ap.add_argument('--output-json', default='paper/figures/benchmark_pro_all200.json')
    ap.add_argument('--start-from', type=int, default=0,
                    help='resume after N pairs already done (for crash recovery)')
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    src_data = json.load(open(args.source_bench))['rows']
    classical = {r['pair']: r for r in src_data
                 if r.get('mode')=='classical' and r.get('ghost_score') is not None}
    n_pairs = max(classical.keys()) + 1
    print(f'Will run Pro on all {n_pairs} pairs')

    fs = FaceSwapper()
    portraits = list_portraits(Path(args.portraits_root))
    random.shuffle(portraits)
    valid = []
    for p in portraits:
        img = cv2.imread(str(p))
        if img is None: continue
        h,w = img.shape[:2]
        if max(h,w) > 512:
            s = 512/max(h,w)
            small = cv2.resize(img,(int(w*s),int(h*s)),interpolation=cv2.INTER_AREA)
        else: small = img
        if detect(fs,small) is None: continue
        if max(h,w) > args.max_side:
            s = args.max_side/max(h,w)
            img_run = cv2.resize(img,(int(w*s),int(h*s)),interpolation=cv2.INTER_AREA)
        else: img_run = img
        face = detect(fs,img_run)
        if face is None: continue
        valid.append((p,img_run,face))
        if len(valid)>=120: break

    pairs = []; seen=set(); attempts=0
    while len(pairs) < n_pairs and attempts < n_pairs*30:
        attempts+=1
        i,j = random.sample(range(len(valid)),2)
        key = tuple(sorted((i,j)))
        if key in seen: continue
        c = cos(valid[i][2].normed_embedding, valid[j][2].normed_embedding)
        if c > 0.45: continue
        pairs.append((i,j,c)); seen.add(key)
    print(f'rebuilt {len(pairs)} pairs')

    rows = []
    # Resume support: load existing rows if any
    if args.start_from > 0 and Path(args.output_json).exists():
        rows = json.load(open(args.output_json))['rows']
        print(f'resuming after {len(rows)} pairs already complete')

    done_pairs = {r['pair'] for r in rows}

    for k in range(n_pairs):
        if k in done_pairs:
            continue
        if k >= len(pairs): continue
        si,ti,_ = pairs[k]
        src_path,src_img,src_face = valid[si]
        tgt_path,tgt_img,tgt_face = valid[ti]
        cl_row = classical.get(k, {})
        prev_a = cl_row.get('alpha_s', 0.0)
        prev_g = cl_row.get('ghost_score', 0.0)
        try:
            t0 = time.time()
            pr = pro_swap.pro_swap(fs, src_img, tgt_img, hd=True, enhance=True)
            r = pr.image; dt = time.time()-t0
            rf = detect(fs, r)
            if rf is None: a_s=a_t=float('nan')
            else:
                a_s = cos(src_face.normed_embedding, rf.normed_embedding)
                a_t = cos(tgt_face.normed_embedding, rf.normed_embedding)
            g = a_t - a_s
            rows.append({
                'pair': k, 'mode': 'pro_all200',
                'src': src_path.name, 'tgt': tgt_path.name,
                'pose_diff_deg': cl_row.get('pose_diff_deg'),
                'classical_alpha_s': round(prev_a,4),
                'classical_ghost_score': round(prev_g,4),
                'alpha_s': round(a_s,4) if a_s==a_s else None,
                'alpha_t': round(a_t,4) if a_t==a_t else None,
                'ghost_score': round(g,4) if g==g else None,
                'delta_alpha_s': round((a_s if a_s==a_s else 0)-prev_a,4),
                'time_s': round(dt,2),
            })
            print(f'  pair {k:3d}  Cl→Pro α_s: {prev_a:+.3f}→{a_s:+.3f}  '
                  f'g: {prev_g:+.3f}→{g:+.3f}  Δ={a_s-prev_a:+.3f}  ({dt:.0f}s)')
        except Exception as e:
            print(f'  pair {k} FAILED: {e}')
            rows.append({'pair':k,'mode':'pro_all200','error':str(e)})
        fs.cleanup()
        # Save partial every 10 pairs
        if (len(rows) % 10) == 0:
            with open(args.output_json,'w') as f: json.dump({'rows':rows},f,indent=2)

    Path(args.output_csv).parent.mkdir(parents=True,exist_ok=True)
    with open(args.output_csv,'w',newline='') as f:
        if rows:
            keys = sorted({k for r in rows for k in r.keys()})
            w = csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)
    with open(args.output_json,'w') as f: json.dump({'rows':rows},f,indent=2)

    valid_rows = [r for r in rows if r.get('alpha_s') is not None]
    if valid_rows:
        d = np.array([r['delta_alpha_s'] for r in valid_rows])
        a = np.array([r['alpha_s'] for r in valid_rows])
        from scipy.stats import wilcoxon
        stat,p = wilcoxon(d, alternative='greater')
        print(f'\n=== Pro on {len(valid_rows)} pairs ===')
        print(f'Pro α_s          = {a.mean():.3f}±{a.std():.3f}')
        print(f'Δα_s (Pro-Cl)    = {d.mean():+.3f}±{d.std():.3f}  CI=±{1.96*d.std()/np.sqrt(len(d)):.3f}  median={np.median(d):+.3f}')
        print(f'Wilcoxon (Pro>Cl): W={stat:.0f}, p={p:.3g}')
        print(f'Rescues (Δ>0.1):  {sum(1 for x in d if x>0.1)}/{len(d)}')
        print(f'No-harm (Δ≥-0.05): {sum(1 for x in d if x>=-0.05)}/{len(d)}')
        print(f'Matches (α_s≥0.6): {sum(1 for v in a if v>=0.6)}/{len(a)}')


if __name__ == '__main__': sys.exit(main())
