#!/usr/bin/env bash
# Generic POSIX Fallback Installation Script for Adaptive ML Platform.
# Installs packages using python3/pip/venv in environments where Poetry or Homebrew are unavailable.

set -e

echo "=== Adaptive ML Platform: Generic fallback Installer ==="

# Check Python3
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required. Please install Python 3.9+ to continue."
    exit 1
fi

# Set up standard python3 venv
echo "-> Creating python3 virtual environment in '.venv' folder..."
python3 -m venv .venv
source .venv/bin/activate

echo "-> Upgrading pip and setting up standard packaging tools..."
pip install --upgrade pip setuptools wheel

echo "-> Installing backend requirements using setup.py/pyproject.toml fallback..."
pip install -e .

# Install frontend if present
if [ -d "frontend" ]; then
    cd frontend
    if command -v npm &> /dev/null; then
        echo "-> Installing frontend NPM packages..."
        npm install
    else
        echo "Warning: npm not found. Frontend must be installed manually."
    fi
    cd ..
fi

echo "=== Installation Completed Successfully! ==="
echo "Activate your virtual environment using:"
echo "  source .venv/bin/activate"
echo "To run the backend ModelServer:"
echo "  python3 backend/app/api/inference.py"
