"""Download the model weights needed by the face-swap pipeline.

Files dropped into ./models/ :
  - inswapper_128.onnx                    (~554 MB)  identity swap network
  - GFPGANv1.4.pth                        (~333 MB)  face restoration / detail
  - models/buffalo_l/{5 ONNX files}       (~325 MB)  face detection + embedding
"""

import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent / 'models'
MODELS_DIR.mkdir(exist_ok=True)
(MODELS_DIR / 'models').mkdir(exist_ok=True)

# Each entry: name -> list of mirror URLs (first one that works wins).
# Note: GitHub's releases CDN is heavily throttled (~50–200 KB/s) so we put
# HuggingFace mirrors first wherever possible.
SIMPLE_DOWNLOADS = {
    'inswapper_128.onnx': [
        'https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx',
        'https://huggingface.co/deepinsight/inswapper/resolve/main/inswapper_128.onnx',
        'https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx',
    ],
    'GFPGANv1.4.pth': [
        'https://huggingface.co/leonelhs/gfpgan/resolve/main/GFPGANv1.4.pth',
        'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth',
    ],
}

# buffalo_l ships as a zip that we extract under ./models/models/buffalo_l/
BUFFALO_L_MIRRORS = [
    'https://huggingface.co/vladmandic/insightface-faceanalysis/resolve/main/buffalo_l.zip',
    'https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip',
    'https://sourceforge.net/projects/insightface.mirror/files/v0.7/buffalo_l.zip/download',
]
BUFFALO_L_REQUIRED_FILES = [
    '1k3d68.onnx', '2d106det.onnx', 'det_10g.onnx', 'genderage.onnx', 'w600k_r50.onnx',
]

# GFPGAN's GFPGANer auto-downloads these into ./gfpgan/weights/ on first run.
# We pre-fetch them from HuggingFace because the GitHub releases CDN is throttled.
GFPGAN_AUX_DIR = Path(__file__).resolve().parent / 'gfpgan' / 'weights'
GFPGAN_AUX_FILES = {
    'detection_Resnet50_Final.pth': [
        'https://huggingface.co/salmonrk/facedetection/resolve/main/detection_Resnet50_Final.pth',
        'https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth',
    ],
    'parsing_parsenet.pth': [
        'https://huggingface.co/nlightcho/gfpgan_v14/resolve/main/parsing_parsenet.pth',
        'https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth',
    ],
}


def _progress(name):
    def hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 / total_size)
            sys.stdout.write(
                f'\r  {name}: {pct:5.1f}%  '
                f'({downloaded / 1024 / 1024:6.1f} / {total_size / 1024 / 1024:.1f} MB)'
            )
        else:
            sys.stdout.write(f'\r  {name}: {downloaded / 1024 / 1024:6.1f} MB')
        sys.stdout.flush()
    return hook


def _download(url: str, dest: Path, name: str) -> bool:
    print(f'Downloading {name} from {url}')
    try:
        urllib.request.urlretrieve(url, dest, reporthook=_progress(name))
        print()
        return True
    except Exception as e:
        print(f'\n  Mirror failed: {e}')
        if dest.exists():
            dest.unlink()
        return False


def download_simple(name: str, urls):
    path = MODELS_DIR / name
    if path.exists() and path.stat().st_size > 1024 * 1024:
        print(f'  Already present: {name} ({path.stat().st_size / 1024 / 1024:.1f} MB)')
        return True
    for url in urls:
        if _download(url, path, name):
            return True
    print(f'  ALL mirrors failed for {name}.', file=sys.stderr)
    return False


def download_buffalo_l() -> bool:
    target_dir = MODELS_DIR / 'models' / 'buffalo_l'
    if target_dir.exists() and all((target_dir / f).exists() for f in BUFFALO_L_REQUIRED_FILES):
        print(f'  Already present: buffalo_l ({target_dir})')
        return True

    zip_path = MODELS_DIR / 'models' / 'buffalo_l.zip'
    ok = False
    for url in BUFFALO_L_MIRRORS:
        if _download(url, zip_path, 'buffalo_l.zip'):
            ok = True
            break
    if not ok:
        return False

    print('Extracting buffalo_l.zip...')
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target_dir)

    # Some mirrors wrap files in an extra buffalo_l/ subdirectory. Flatten if so.
    nested = target_dir / 'buffalo_l'
    if nested.is_dir():
        for item in nested.iterdir():
            shutil.move(str(item), str(target_dir / item.name))
        nested.rmdir()

    zip_path.unlink()
    missing = [f for f in BUFFALO_L_REQUIRED_FILES if not (target_dir / f).exists()]
    if missing:
        print(f'  buffalo_l is missing: {missing}', file=sys.stderr)
        return False
    print(f'  buffalo_l extracted to {target_dir}')
    return True


def download_gfpgan_aux():
    GFPGAN_AUX_DIR.mkdir(parents=True, exist_ok=True)
    ok = True
    for name, urls in GFPGAN_AUX_FILES.items():
        path = GFPGAN_AUX_DIR / name
        if path.exists() and path.stat().st_size > 1024 * 1024:
            print(f'  Already present: gfpgan/weights/{name} ({path.stat().st_size / 1024 / 1024:.1f} MB)')
            continue
        success = False
        for url in urls:
            if _download(url, path, f'gfpgan/weights/{name}'):
                success = True
                break
        if not success:
            print(f'  ALL mirrors failed for {name}.', file=sys.stderr)
            ok = False
    return ok


def main():
    print(f'Saving to: {MODELS_DIR}')
    ok = True
    for name, urls in SIMPLE_DOWNLOADS.items():
        ok &= download_simple(name, urls)
    ok &= download_buffalo_l()
    ok &= download_gfpgan_aux()
    if ok:
        print('\nAll models ready.')
    else:
        print('\nOne or more downloads failed. See messages above.', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
