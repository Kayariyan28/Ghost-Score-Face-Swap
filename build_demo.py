"""Build a demo video of the Face Swap web app.

Captures real screenshots from the running FastAPI app at localhost:8000
using Playwright, then stitches them together with title cards, captions
and result reveals into a polished MP4 suitable for GitHub upload.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080  # 1080p
FPS = 30


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for fp in ['/System/Library/Fonts/Supplemental/Arial Bold.ttf',
               '/Library/Fonts/Arial Bold.ttf',
               '/System/Library/Fonts/Supplemental/Arial.ttf',
               '/Library/Fonts/Arial.ttf']:
        if 'Bold' not in fp and bold:
            continue
        if os.path.exists(fp):
            return ImageFont.truetype(fp, size)
    return ImageFont.load_default()


def gradient_bg(w: int = W, h: int = H,
                top=(12, 14, 18), bot=(28, 32, 42)) -> Image.Image:
    g = Image.new('RGB', (w, h))
    gd = ImageDraw.Draw(g)
    for y in range(h):
        t = y / h
        gd.line([(0, y), (w, y)],
                 fill=tuple(int(top[i] + (bot[i] - top[i]) * t) for i in range(3)))
    return g


def title_card(headline: str, subline: str = '') -> Image.Image:
    img = gradient_bg()
    d = ImageDraw.Draw(img)
    fb = font(78, bold=True)
    bbox = d.textbbox((0, 0), headline, font=fb)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    d.text(((W - tw) // 2, H // 2 - 80), headline,
           fill=(255, 255, 255), font=fb)
    if subline:
        fs = font(36)
        bbox = d.textbbox((0, 0), subline, font=fs)
        tw = bbox[2] - bbox[0]
        d.text(((W - tw) // 2, H // 2 + 20), subline,
               fill=(180, 190, 200), font=fs)
    return img


def frame_with_caption(content: Image.Image, caption_top: str = '',
                       caption_bot: str = '') -> Image.Image:
    """Place content centred, with bg gradient + optional captions."""
    bg = gradient_bg()
    d = ImageDraw.Draw(bg)
    if caption_top:
        ft = font(38, bold=True)
        bbox = d.textbbox((0, 0), caption_top, font=ft)
        tw = bbox[2] - bbox[0]
        d.text(((W - tw) // 2, 50), caption_top,
               fill=(255, 255, 255), font=ft)
    # Centre content
    cw, ch = content.size
    max_w, max_h = W - 200, H - 280
    s = min(max_w / cw, max_h / ch, 1.0)
    if s < 1.0:
        content = content.resize((int(cw * s), int(ch * s)), Image.LANCZOS)
        cw, ch = content.size
    cx = (W - cw) // 2
    cy = (H - ch) // 2 + (30 if caption_top else 0)
    # Drop shadow
    shadow = Image.new('RGBA', (cw + 40, ch + 40), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rectangle(
        [20, 20, cw + 20, ch + 20], fill=(0, 0, 0, 100))
    bg.paste(shadow, (cx - 20, cy - 20), shadow)
    bg.paste(content, (cx, cy))
    if caption_bot:
        fb = font(24)
        bbox = d.textbbox((0, 0), caption_bot, font=fb)
        tw = bbox[2] - bbox[0]
        d = ImageDraw.Draw(bg)
        d.text(((W - tw) // 2, H - 80), caption_bot,
               fill=(170, 180, 190), font=fb)
    return bg


def grab_ui_screenshots(out_dir: Path):
    """Drive the local app via Playwright and screenshot each UI state."""
    from playwright.sync_api import sync_playwright

    out_dir.mkdir(parents=True, exist_ok=True)
    URL = 'http://127.0.0.1:8000'
    src_img = '/Users/karanchandradey/Face Swap/paper/figures/hero_unsplash/unsplash_subject_A_source.jpg'
    tgt_img = '/Users/karanchandradey/Face Swap/paper/figures/hero_unsplash/unsplash_subject_B_target.jpg'

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={'width': 1600, 'height': 1000},
                                  device_scale_factor=2)
        page = ctx.new_page()

        # 1) Landing page
        page.goto(URL, wait_until='networkidle')
        page.wait_for_timeout(800)
        page.screenshot(path=str(out_dir / '01_landing.png'))
        print('  captured 01_landing')

        # 2) Source uploaded
        page.set_input_files('#sourceInput', src_img)
        page.wait_for_timeout(900)
        page.screenshot(path=str(out_dir / '02_source_uploaded.png'))
        print('  captured 02_source_uploaded')

        # 3) Both uploaded
        page.set_input_files('#targetInput', tgt_img)
        page.wait_for_timeout(900)
        page.screenshot(path=str(out_dir / '03_both_uploaded.png'))
        print('  captured 03_both_uploaded')

        # 4) Scroll to mode picker / options
        page.evaluate('window.scrollTo(0, 600)')
        page.wait_for_timeout(500)
        page.screenshot(path=str(out_dir / '04_options.png'))
        print('  captured 04_options')

        browser.close()


def cmd_run(args, **kw):
    print(f'$ {" ".join(args)}')
    return subprocess.run(args, check=True, **kw)


def main():
    project = Path('/Users/karanchandradey/Face Swap')
    figs = project / 'paper/figures'
    docs = project / 'docs'
    docs.mkdir(exist_ok=True)
    ui_dir = docs / 'ui_shots'
    frames_dir = Path(tempfile.mkdtemp(prefix='demo_frames_'))
    print('frames -> ', frames_dir)

    # Step 1: capture UI screenshots if not already present
    if not (ui_dir / '04_options.png').exists():
        print('[1/4] capturing UI from localhost:8000...')
        grab_ui_screenshots(ui_dir)
    else:
        print('[1/4] UI screenshots already present')

    # Step 2: build the frame list
    print('[2/4] composing frames...')
    timeline = []  # (image, hold_seconds)

    # Intro title
    timeline += [
        (title_card('Ghost-Score-Driven', 'Multi-Pipeline Face Swap'), 2.5),
        (title_card('Apple-Silicon Native', 'On-device. No cloud.'), 2.0),
    ]

    # Show banner
    banner = Image.open(docs / 'banner.png').convert('RGB')
    timeline.append((frame_with_caption(banner), 3.0))

    # UI walk-through
    ui_steps = [
        ('01_landing.png',         'Open the local web UI',
         'http://localhost:8000 — FastAPI + static HTML/JS, no cloud calls'),
        ('02_source_uploaded.png', 'Upload the source face',
         'Drop in a portrait of the person whose identity you want to transplant'),
        ('03_both_uploaded.png',   'Add the target image',
         'Different subject. Cross-identity. Source-pixel graft preserves real skin detail.'),
        ('04_options.png',         'Pick a mode',
         'Classical · AI Synthesis · Pro (3D-aware adaptive)'),
    ]
    for fn, top, bot in ui_steps:
        try:
            im = Image.open(ui_dir / fn).convert('RGB')
            timeline.append((frame_with_caption(im, top, bot), 2.8))
        except FileNotFoundError:
            print(f'  WARNING: missing {fn}')

    # Architecture preview
    arch = title_card('Inside the pipeline',
                       'Multi-pipeline auto-best · Ghost-score routing · HardPoseReplaceMode')
    timeline.append((arch, 2.0))

    # Result: triplet
    triplet = Image.open(figs / 'experiment_triplet.png').convert('RGB')
    timeline.append((frame_with_caption(triplet,
        'Result on the cross-identity hero pair',
        'Source A · Target B · Pro mode result (Pro α_s = 0.896, g = −0.848)'), 3.5))

    # Mode comparison
    if (figs / 'experiment_modes.png').exists():
        modes = Image.open(figs / 'experiment_modes.png').convert('RGB')
        timeline.append((frame_with_caption(modes,
            'All three modes succeed on this pair',
            'Source · Target · AI · Classical · Pro'), 3.5))

    # Identity inspection
    if (figs / 'experiment_face_crops.png').exists():
        crops = Image.open(figs / 'experiment_face_crops.png').convert('RGB')
        timeline.append((frame_with_caption(crops,
            'Identity inspection',
            "Eyelid contour, iris colour, brow density and lip shape inherited from source"), 3.0))

    # Eye-zoom
    if (figs / 'experiment_eye_zoom.png').exists():
        eye = Image.open(figs / 'experiment_eye_zoom.png').convert('RGB')
        timeline.append((frame_with_caption(eye,
            'Eye-region zoom',
            'Micro-features at full HD resolution'), 2.8))

    # Quality dashboard / metrics card
    metrics = gradient_bg()
    d = ImageDraw.Draw(metrics)
    d.text((100, 80), 'Quality dashboard (returned in JSON)',
           fill=(255, 255, 255), font=font(48, bold=True))
    rows = [
        ('α_s  (source ArcFace cosine)',          '0.896',   'higher → identity match'),
        ('α_t  (target ArcFace cosine)',          '0.048',   'should be low'),
        ('g    (ghost score)',                     '−0.848',  'lower → cleaner swap'),
        ('seam gradient energy',                   '14.7',    'low → no patch line'),
        ('landmark RMSE',                          '0.012',   'low → no geometric drift'),
        ('double-edge ratio  ρ_d',                 '0.07',    'low → no ghost overlay'),
        ('LPIPS-AlexNet  /  -VGG',                 '0.41 / 0.39', 'perceptual distance to source'),
    ]
    y = 200
    for k, v, h in rows:
        d.text((140, y), k, fill=(200, 210, 220), font=font(28))
        d.text((900, y), v, fill=(80, 200, 140), font=font(32, bold=True))
        d.text((1180, y), h, fill=(140, 150, 160), font=font(22))
        y += 70
    d.text((140, y + 40),
           'Returned with every /api/swap call · client decides what to render.',
           fill=(150, 160, 170), font=font(22))
    timeline.append((metrics, 4.5))

    # Numbers card: N=200 benchmark
    nums = gradient_bg()
    d = ImageDraw.Draw(nums)
    d.text((100, 70), 'N=200 multi-pair benchmark',
           fill=(255, 255, 255), font=font(48, bold=True))
    d.text((100, 130),
           'real-photo cross-identity portraits, pose 1.3° – 140.6°',
           fill=(170, 180, 190), font=font(26))
    bigs = [
        ('92.5%',  'Pro match rate'),
        ('+0.150', 'Δα_s Pro − Classical'),
        ('p<10⁻²¹', 'Wilcoxon paired'),
        ('10/10',  'Hardest-pair rescue'),
        ('25.5 s', 'Pro median wall-clock'),
        ('24 GB',  'RSS reload watchdog'),
    ]
    bx, by = 100, 260
    for i, (val, lab) in enumerate(bigs):
        col = i % 3
        row = i // 3
        rx = bx + col * 600
        ry = by + row * 240
        # Card
        card = Image.new('RGB', (520, 200), (40, 48, 60))
        nums.paste(card, (rx, ry))
        nd = ImageDraw.Draw(nums)
        bbox = nd.textbbox((0, 0), val, font=font(74, bold=True))
        tw = bbox[2] - bbox[0]
        nd.text((rx + (520 - tw) // 2, ry + 20), val,
                fill=(255, 220, 100), font=font(74, bold=True))
        bbox = nd.textbbox((0, 0), lab, font=font(24))
        tw = bbox[2] - bbox[0]
        nd.text((rx + (520 - tw) // 2, ry + 130), lab,
                fill=(220, 230, 240), font=font(24))
    timeline.append((nums, 4.0))

    # Outro
    outro = title_card('Run it locally',
                       'github.com/yourname/face-swap · MIT Licence (code), Unsplash Licence (hero pair)')
    timeline.append((outro, 3.0))

    # Step 3: write frames
    print('[3/4] writing frames...')
    frame_idx = 0
    for img, hold_s in timeline:
        if img.size != (W, H):
            img = img.resize((W, H), Image.LANCZOS)
        n_frames = int(hold_s * FPS)
        for _ in range(n_frames):
            img.save(frames_dir / f'frame_{frame_idx:06d}.png',
                     compress_level=4)
            frame_idx += 1
    print(f'  wrote {frame_idx} frames')

    # Step 4: encode with ffmpeg
    print('[4/4] encoding mp4...')
    out_mp4 = docs / 'demo.mp4'
    cmd_run([
        'ffmpeg', '-y',
        '-framerate', str(FPS),
        '-i', str(frames_dir / 'frame_%06d.png'),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '20',
        '-preset', 'medium',
        '-movflags', '+faststart',
        str(out_mp4),
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f'  wrote {out_mp4} ({out_mp4.stat().st_size / 1e6:.1f} MB)')

    # Clean up frames
    shutil.rmtree(frames_dir)
    print('done.')


if __name__ == '__main__':
    sys.exit(main() or 0)
