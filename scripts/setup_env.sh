#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="$REPO_ROOT/.venv"
FACE_SERVICE=${1:-cpu}
INSTALL_DEV=${2:-false}

echo "Repository root: $REPO_ROOT"
echo "Venv: $VENV_DIR"

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtualenv..."
  python3 -m venv "$VENV_DIR"
else
  echo "Virtualenv already exists"
fi

PIP="$VENV_DIR/bin/pip"
PYTHON="$VENV_DIR/bin/python"

echo "Upgrading pip, setuptools, wheel..."
$PYTHON -m pip install --upgrade pip setuptools wheel

echo "Installing runtime requirements..."
$PIP install -r "$REPO_ROOT/requirements-full.txt"

if [ "$INSTALL_DEV" = "true" ]; then
  echo "Installing dev requirements..."
  $PIP install -r "$REPO_ROOT/requirements-dev.txt"
fi

if [ "$FACE_SERVICE" = "gpu" ]; then
  if [ -f "$REPO_ROOT/face-service/requirements-gpu.txt" ]; then
    echo "Installing face-service GPU requirements..."
    $PIP install -r "$REPO_ROOT/face-service/requirements-gpu.txt"
  else
    echo "GPU requirements not found; falling back to CPU requirements"
    $PIP install -r "$REPO_ROOT/face-service/requirements-cpu.txt"
  fi
else
  if [ -f "$REPO_ROOT/face-service/requirements-cpu.txt" ]; then
    echo "Installing face-service CPU requirements..."
    $PIP install -r "$REPO_ROOT/face-service/requirements-cpu.txt"
  fi
fi

# Run quick import check
$PYTHON "$REPO_ROOT/scripts/test_imports.py"

echo "Setup complete. To activate the venv run: source .venv/bin/activate"
