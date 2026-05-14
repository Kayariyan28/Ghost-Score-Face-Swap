#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
# shellcheck disable=SC1091
source venv/bin/activate
exec python app.py
