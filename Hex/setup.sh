#!/usr/bin/env bash
set -euo pipefail

# Run from the script directory
cd "$(dirname "$0")"

# Find a suitable python executable (prefer 3.11 if available)
PY=python3.11
if ! command -v "$PY" >/dev/null 2>&1; then
  PY=python3
  if ! command -v "$PY" >/dev/null 2>&1; then
    PY=python
  fi
fi

echo "Using Python: $PY"

echo "Creating virtual environment in ./venv..."
$PY -m venv venv

echo "Activating virtualenv..."
# shellcheck disable=SC1091
source venv/bin/activate

echo "Upgrading pip and installing requirements..."
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

echo "Setup complete. Activate the environment with:"
echo "  source $(pwd)/venv/bin/activate"
