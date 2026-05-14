"""FastAPI server that exposes the face-swap pipeline as a simple HTTP API."""

import asyncio
import concurrent.futures
import io
import os
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

from face_swap import FaceSwapper
from pro_swap import pro_swap as _pro_swap

BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / 'outputs'
STATIC_DIR = BASE_DIR / 'static'
OUTPUTS_DIR.mkdir(exist_ok=True)

app = FastAPI(title='Face Swap')

print('[app] Loading models (first run will also download buffalo_l ~280 MB)...')
swapper = FaceSwapper(models_dir=str(BASE_DIR / 'models'))
print('[app] Models ready.')

# --- Concurrency & timeout guardrails --------------------------------------
# Only one swap runs at a time. Concurrent HTTP requests queue up rather than
# racing on the same PyTorch / MPS context (which was the original cause of
# hangs after a few rapid requests).
_swap_lock = asyncio.Lock()
_swap_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix='swap-worker',
)
# A swap that exceeds this is killed cleanly and returns 504. 240 s is
# generous — most swaps finish in 20–40 s on Apple Silicon.
SWAP_TIMEOUT_S = 240
_session_started_at = time.time()
_session_swap_count = 0

# Pipeline-reload thresholds. When ANY of these trips, we tear down the
# FaceSwapper instance and rebuild it before the next swap — guarantees a
# fresh PyTorch / MPS context, MediaPipe state, GFPGAN face_helper, etc.
RELOAD_AFTER_SWAPS = 20
RELOAD_RSS_MB = 24_000  # 24 GB


def _rss_mb():
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1e6
    except Exception:
        return 0.0


def _maybe_full_reload():
    """If memory pressure or swap count threshold is reached, rebuild the
    FaceSwapper from scratch. Returns True if a reload happened."""
    global swapper
    rss = _rss_mb()
    if _session_swap_count >= RELOAD_AFTER_SWAPS or rss >= RELOAD_RSS_MB:
        print(f'[app] FULL PIPELINE RELOAD '
              f'(swaps={_session_swap_count}, rss={rss:.0f} MB)')
        # Best-effort cleanup of the old instance before reload.
        try:
            swapper.cleanup()
        except Exception:
            pass
        old = swapper
        try:
            import gc
            import torch
            del old
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
                torch.mps.synchronize()
        except Exception:
            pass
        swapper = FaceSwapper(models_dir=str(BASE_DIR / 'models'))
        return True
    return False


def _decode(upload_bytes: bytes) -> np.ndarray:
    # Fast path: OpenCV handles JPG / PNG / WEBP natively.
    arr = np.frombuffer(upload_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is not None:
        return img
    # Fallback path: Pillow handles HEIC (iPhone), AVIF, BMP, TIFF, etc.
    try:
        pil = Image.open(io.BytesIO(upload_bytes))
        if pil.mode != 'RGB':
            pil = pil.convert('RGB')
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail='Could not decode image. Supported formats: JPG, PNG, WEBP, HEIC.',
        )


@app.get('/outputs/{filename}')
async def get_output(filename: str):
    # Reject anything that tries to escape the outputs dir.
    if '/' in filename or '\\' in filename or '..' in filename:
        raise HTTPException(status_code=400, detail='Invalid filename.')
    path = OUTPUTS_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail='Not found.')
    return FileResponse(str(path))


@app.post('/api/swap')
async def api_swap(
    source: UploadFile = File(...),
    target: UploadFile = File(...),
    mode: str = Form('photoshop'),
    enhance: str = Form('true'),
    swap_all: str = Form('false'),
    fidelity: str = Form('0.75'),
    hd: str = Form('true'),
    sharpen: str = Form('0.35'),
    detail: str = Form('0.9'),
    preserve_hair: str = Form('true'),
    preserve_source_tone: str = Form('false'),
):
    source_img = _decode(await source.read())
    target_img = _decode(await target.read())

    valid_modes = ('photoshop', 'ai', 'pro')
    m = mode.lower()
    if m in ('pure', 'hybrid', 'diffusion'):
        m = 'photoshop'  # legacy values map to photoshop
    mode_val = m if m in valid_modes else 'photoshop'
    enhance_flag = enhance.lower() in ('1', 'true', 'yes', 'on')
    swap_all_flag = swap_all.lower() in ('1', 'true', 'yes', 'on')
    hd_flag = hd.lower() in ('1', 'true', 'yes', 'on')
    try:
        fidelity_val = float(fidelity)
    except ValueError:
        fidelity_val = 0.75
    try:
        sharpen_val = float(sharpen)
    except ValueError:
        sharpen_val = 0.35
    try:
        detail_val = float(detail)
    except ValueError:
        detail_val = 0.9
    preserve_hair_flag = preserve_hair.lower() in ('1', 'true', 'yes', 'on')
    preserve_tone_flag = preserve_source_tone.lower() in ('1', 'true', 'yes', 'on')

    # Serialize swaps + hard timeout. Concurrent requests wait on the lock;
    # any single swap that exceeds SWAP_TIMEOUT_S is killed via thread-pool
    # cancel and returns 504. This is the watchdog that fixes the "stuck
    # after N generations" pattern.
    common_kwargs = dict(
        enhance=enhance_flag,
        swap_all_targets=swap_all_flag,
        fidelity=fidelity_val,
        hd=hd_flag,
        sharpen=sharpen_val,
        detail=detail_val,
        preserve_hair=preserve_hair_flag,
        preserve_source_tone=preserve_tone_flag,
    )

    def _do_swap():
        if mode_val == 'pro':
            # Pro mode: separate pipeline (pro_swap.py). Uses the same
            # swapper instance but never modifies it. Returns a ProResult
            # which we adapt to the same (image, scores, retried) tuple.
            pro = _pro_swap(swapper, source_img, target_img, **common_kwargs)
            # cleanup() after the run — same as swap_with_metrics does.
            try:
                swapper.cleanup()
            except Exception:
                pass
            return pro.image, pro.scores, False, pro.info_dict()
        result, scores, retried = swapper.swap_with_metrics(
            source_img, target_img, mode=mode_val, **common_kwargs,
        )
        return result, scores, retried, None

    global _session_swap_count
    async with _swap_lock:
        # Pre-swap watchdog — reload pipeline if it's reached memory or
        # swap-count thresholds. Done under the lock so no other request
        # is mid-pipeline when we tear down.
        if _maybe_full_reload():
            _session_swap_count = 0

        t_swap_start = time.time()
        print(f'[app] swap #{_session_swap_count + 1} start '
              f'(mode={mode_val}, rss={_rss_mb():.0f} MB)')

        future = _swap_executor.submit(_do_swap)
        loop = asyncio.get_event_loop()
        try:
            result, scores, retried, pro_info = await asyncio.wait_for(
                asyncio.wrap_future(future, loop=loop),
                timeout=SWAP_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            future.cancel()
            # Trigger cleanup of the runtime even though the worker is
            # blocked — best-effort, the next request will get a fresh slate.
            try:
                swapper.cleanup()
            except Exception:
                pass
            raise HTTPException(
                status_code=504,
                detail=(
                    f'Swap exceeded {SWAP_TIMEOUT_S}s timeout. Pipeline reset; '
                    'next request should work. If this keeps happening, '
                    'restart the server.'
                ),
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f'Swap failed: {e}')
        _session_swap_count += 1
        print(f'[app] swap #{_session_swap_count} done '
              f'in {time.time() - t_swap_start:.1f}s '
              f'(rss={_rss_mb():.0f} MB, mp_calls={swapper._mp_call_count})')

    out_name = f'{uuid.uuid4().hex}.png'
    out_path = OUTPUTS_DIR / out_name
    cv2.imwrite(str(out_path), result)

    payload = {
        'success': True,
        'url': f'/outputs/{out_name}',
        'metrics': scores.to_dict(),
        'retried': bool(retried),
    }
    if pro_info is not None:
        payload['pro'] = pro_info  # pose, chosen_route, candidates
    return payload


@app.get('/healthz')
def healthz():
    """Detailed health check — process state + memory + swap counts."""
    rss_mb = None
    try:
        import psutil
        rss_mb = round(psutil.Process(os.getpid()).memory_info().rss / 1e6, 1)
    except Exception:
        pass
    uptime_s = int(time.time() - _session_started_at)
    return {
        'ok': True,
        'uptime_s': uptime_s,
        'swaps_completed': _session_swap_count,
        'rss_mb': rss_mb,
        'swap_lock_held': _swap_lock.locked(),
        'mediapipe_calls_since_reset': swapper._mp_call_count,
    }


@app.post('/api/reset')
async def api_reset():
    """Manual pipeline reset — force MPS cache flush, GFPGAN buffer clear,
    MediaPipe rebuild. Use this if the dashboard shows the server is in a
    weird state."""
    swapper.cleanup()
    # Force a MediaPipe rebuild regardless of count.
    swapper._mp_call_count = swapper.MEDIAPIPE_RESET_EVERY
    swapper.cleanup()
    return {'ok': True, 'reset': True}


# Mount static UI last so explicit routes above win.
app.mount('/', StaticFiles(directory=str(STATIC_DIR), html=True), name='static')


if __name__ == '__main__':
    import uvicorn
    host = os.environ.get('HOST', '127.0.0.1')
    port = int(os.environ.get('PORT', '8000'))
    uvicorn.run(app, host=host, port=port)
