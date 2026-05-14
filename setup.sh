#!/usr/bin/env bash
# One-shot setup: create venv, install deps, download models.
#
# Install strategy:
#   1. Install torch / numpy / cython FIRST (with build isolation, binary wheels)
#   2. Install everything else with --no-build-isolation
#      → basicsr builds against the already-installed torch instead of
#        downloading its OWN copy of torch (~300 MB saved + ~5 min faster).
set -e
cd "$(dirname "$0")"

PYTHON_BIN=""
for candidate in python3.11 python3.10 python3.12; do
  if command -v "$candidate" >/dev/null 2>&1; then
    PYTHON_BIN="$candidate"
    break
  fi
done

if [ -z "$PYTHON_BIN" ]; then
  echo "Error: need python3.10 / 3.11 / 3.12 on PATH."
  echo "Install with:  brew install python@3.11"
  exit 1
fi

echo "==> Using $PYTHON_BIN ($($PYTHON_BIN --version))"

if [ ! -d "venv" ]; then
  echo "==> Creating virtual environment in ./venv"
  "$PYTHON_BIN" -m venv venv
fi

# shellcheck disable=SC1091
source venv/bin/activate

echo "==> Upgrading pip / wheel"
python -m pip install --upgrade pip wheel setuptools

echo "==> Step 1/3: installing build prerequisites (torch + numpy + cython)"
pip install "cython>=3.0" "numpy==1.26.4" "torch==2.2.2" "torchvision==0.17.2"

echo "==> Step 2/3: installing easy binary deps"
pip install --no-build-isolation \
  "fastapi==0.110.0" \
  "uvicorn[standard]==0.27.1" \
  "python-multipart==0.0.9" \
  "onnxruntime==1.17.1" \
  "opencv-python==4.9.0.80" \
  "scipy==1.12.0" \
  "tqdm==4.66.2" \
  "pillow-heif==0.16.0"

echo "==> Step 3/3: installing face-swap libs (basicsr builds from source — 1-2 min)"
pip install --no-build-isolation \
  "basicsr==1.4.2" \
  "facexlib==0.3.0" \
  "gfpgan==1.3.8" \
  "insightface==0.7.3"

echo "==> Downloading AI models (~900 MB)"
python download_models.py

echo ""
echo "Setup complete."
echo "Run with:  ./run.sh    (or: source venv/bin/activate && python app.py)"
echo "Then open: http://localhost:8000"
