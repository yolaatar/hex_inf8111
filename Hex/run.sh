#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ -f venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
else
  echo "No virtual environment found. Run ./setup.sh first."
  exit 1
fi

python main_hex.py "$@"
