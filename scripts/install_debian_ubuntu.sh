#!/usr/bin/env bash
# Debian/Ubuntu Installation Script for Adaptive ML Platform.
# Installs Python, Node.js, Poetry, and sets up backend and frontend environments.

set -e

echo "=== Adaptive ML Platform: Debian/Ubuntu Installer ==="

# Check for root/sudo permissions
if [ "$EUID" -ne 0 ] && [ -z "$(which sudo)" ]; then
    echo "Error: This script requires sudo or root privileges."
    exit 1
fi

SUDO=""
if [ "$EUID" -ne 0 ]; then
    SUDO="sudo"
fi

echo "-> Updating system package lists..."
$SUDO apt-get update -y

echo "-> Installing system dependencies (Python, Node.js, git, curl, build essentials)..."
$SUDO apt-get install -y python3 python3-pip python3-venv python3-dev git curl build-essential

# Check and install Node.js/NPM if not present
if ! command -v node &> /dev/null; then
    echo "-> Node.js not found. Installing from NodeSource or default repos..."
    $SUDO apt-get install -y nodejs npm
fi

# Ensure Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "-> Poetry not found. Installing Poetry via official installer..."
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="$HOME/.local/bin:$PATH"
fi

# Set up Python virtual environment
echo "-> Creating and configuring Python virtual environment..."
poetry config virtualenvs.create true --local
poetry env use python3

echo "-> Installing backend dependencies..."
poetry install --no-root

echo "-> Installing frontend dashboard dependencies..."
if [ -d "frontend" ]; then
    cd frontend
    if command -v npm &> /dev/null; then
        echo "-> Running npm install..."
        npm install
    else
        echo "Warning: npm is not available. Frontend dependencies must be installed manually."
    fi
    cd ..
fi

echo "=== Installation Completed Successfully! ==="
echo "To run the backend ModelServer:"
echo "  poetry run adaptivebrain serve --port 8000"
echo "To view system dashboard status:"
echo "  poetry run adaptivebrain status"
