"""Download the extra checkpoints needed by Pro mode's HardPoseReplaceMode.

  - 3DDFA_V2  (mb1_120x120.onnx + BFM stats)  ~3 MB total — dense 3D face fit
  - Depth Anything V2 small (vits)            ~99 MB     — monocular depth
  - SAM 2 base_plus checkpoint                ~165 MB    — promptable segmentation

These are *only* used by Pro mode. The existing photoshop/ai paths do not
depend on them. If any download fails, Pro mode falls back gracefully.
"""

import os
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent / 'models' / 'pro'
ROOT.mkdir(parents=True, exist_ok=True)


def _progress(name):
    def hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 / total_size)
            sys.stdout.write(
                f'\r  {name}: {pct:5.1f}%  '
                f'({downloaded/1024/1024:6.1f}/{total_size/1024/1024:.1f} MB)'
            )
        else:
            sys.stdout.write(f'\r  {name}: {downloaded/1024/1024:.1f} MB')
        sys.stdout.flush()
    return hook


def _download(url, dest, name, min_bytes=1_000_000):
    if dest.exists() and dest.stat().st_size >= min_bytes:
        print(f'  Already present: {dest.name} ({dest.stat().st_size/1024/1024:.1f} MB)')
        return True
    print(f'Downloading {name} from {url}')
    try:
        urllib.request.urlretrieve(url, dest, reporthook=_progress(name))
        print()
        return True
    except Exception as e:
        print(f'\n  Failed: {e}')
        if dest.exists():
            dest.unlink()
        return False


# ---------------------------------------------------------------------------
# 3DDFA_V2  (cleardusk/3DDFA_V2 official ONNX weights)
# ---------------------------------------------------------------------------
TDDFA_DIR = ROOT / '3ddfa_v2'
TDDFA_DIR.mkdir(parents=True, exist_ok=True)
TDDFA_URLS = {
    'mb1_120x120.onnx': [
        'https://huggingface.co/onnx-community/3ddfa_v2/resolve/main/mb1_120x120.onnx',
        'https://github.com/cleardusk/3DDFA_V2/raw/master/weights/mb1_120x120.onnx',
    ],
    'bfm_noneck_v3.pkl': [
        'https://huggingface.co/onnx-community/3ddfa_v2/resolve/main/bfm_noneck_v3.pkl',
        'https://github.com/cleardusk/3DDFA_V2/raw/master/configs/bfm_noneck_v3.pkl',
    ],
    'param_mean_std_62d_120x120.pkl': [
        'https://huggingface.co/onnx-community/3ddfa_v2/resolve/main/param_mean_std_62d_120x120.pkl',
        'https://github.com/cleardusk/3DDFA_V2/raw/master/configs/param_mean_std_62d_120x120.pkl',
    ],
}


def fetch_3ddfa():
    ok = True
    for fname, urls in TDDFA_URLS.items():
        dest = TDDFA_DIR / fname
        got = False
        for url in urls:
            if _download(url, dest, fname, min_bytes=1024):
                got = True
                break
        ok &= got
    return ok


# ---------------------------------------------------------------------------
# Depth Anything V2 small (vits) — official ONNX checkpoint
# ---------------------------------------------------------------------------
DEPTH_DIR = ROOT / 'depth_anything_v2'
DEPTH_DIR.mkdir(parents=True, exist_ok=True)
DEPTH_URLS = [
    'https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx',
    'https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth',
]


def fetch_depth_anything():
    dest = DEPTH_DIR / 'depth_anything_v2_small.onnx'
    if dest.exists() and dest.stat().st_size > 10_000_000:
        print(f'  Already present: {dest.name}')
        return True
    for url in DEPTH_URLS:
        if _download(url, dest, 'depth_anything_v2', min_bytes=10_000_000):
            return True
    return False


# ---------------------------------------------------------------------------
# SAM 2 — base_plus checkpoint (165 MB)
# ---------------------------------------------------------------------------
SAM_DIR = ROOT / 'sam2'
SAM_DIR.mkdir(parents=True, exist_ok=True)
SAM_URLS = [
    'https://huggingface.co/facebook/sam2-hiera-base-plus/resolve/main/sam2_hiera_base_plus.pt',
    'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt',
]
SAM_CONFIG_URLS = [
    'https://raw.githubusercontent.com/facebookresearch/segment-anything-2/main/sam2_configs/sam2_hiera_b%2B.yaml',
]


def fetch_sam2():
    ckpt = SAM_DIR / 'sam2_hiera_base_plus.pt'
    cfg = SAM_DIR / 'sam2_hiera_b+.yaml'
    ok = True
    if not (ckpt.exists() and ckpt.stat().st_size > 50_000_000):
        ok2 = False
        for url in SAM_URLS:
            if _download(url, ckpt, 'sam2_hiera_base_plus.pt', min_bytes=50_000_000):
                ok2 = True
                break
        ok &= ok2
    else:
        print(f'  Already present: {ckpt.name}')
    if not cfg.exists():
        for url in SAM_CONFIG_URLS:
            try:
                urllib.request.urlretrieve(url, cfg)
                print(f'  Saved config: {cfg.name}')
                break
            except Exception:
                continue
    return ok


def main():
    print(f'Saving to: {ROOT}')
    a = fetch_3ddfa()
    b = fetch_depth_anything()
    c = fetch_sam2()
    print(f'\n3DDFA_V2: {"OK" if a else "FAIL"}'
          f'   DepthAnythingV2: {"OK" if b else "FAIL"}'
          f'   SAM2: {"OK" if c else "FAIL"}')


if __name__ == '__main__':
    main()
