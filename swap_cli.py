"""Command-line wrapper for quick testing without spinning up the web server.

Usage:
    python swap_cli.py <source_face.jpg> <target_body.jpg> [output.png]
"""

import sys
from pathlib import Path

import cv2

from face_swap import FaceSwapper


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    src_path = Path(sys.argv[1])
    tgt_path = Path(sys.argv[2])
    out_path = Path(sys.argv[3]) if len(sys.argv) > 3 else Path('outputs/cli_swap.png')
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for p in (src_path, tgt_path):
        if not p.exists():
            print(f'File not found: {p}', file=sys.stderr)
            sys.exit(1)

    source = cv2.imread(str(src_path))
    target = cv2.imread(str(tgt_path))
    if source is None or target is None:
        print('Could not read one of the images (use JPG/PNG/WEBP).', file=sys.stderr)
        sys.exit(1)

    print('Loading models...')
    swapper = FaceSwapper()
    print('Swapping...')
    result = swapper.swap(source, target, enhance=True, swap_all_targets=False)
    cv2.imwrite(str(out_path), result)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
