#!/usr/bin/env bash
set -euo pipefail

VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi

echo "Activating venv..."
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "Upgrading pip..."
python -m pip install --upgrade pip

echo "Installing API requirements and test helpers..."
python -m pip install -r requirements-api.txt
python -m pip install pytest pytest-asyncio pillow pandas rapidfuzz

if [ "${1-}" = "--full" ]; then
  echo "Installing full requirements (this may take a long time)..."
  python -m pip install -r requirements.txt
fi

echo "Setup complete. Activate the venv with: source .venv/bin/activate"
