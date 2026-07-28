#!/usr/bin/env bash
# macOS Installation Script for Adaptive ML Platform.
# Uses Homebrew to set up dependencies.

set -e

echo "=== Adaptive ML Platform: macOS Installer ==="

# Check for Homebrew
if ! command -v brew &> /dev/null; then
    echo "Homebrew is required for this installation script."
    echo "Would you like to install Homebrew now? (y/n)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    else
        echo "Error: Homebrew is required to install dependencies. Halting."
        exit 1
    fi
fi

echo "-> Updating Homebrew formulas..."
brew update

echo "-> Installing system dependencies (Python, Node, git, poetry)..."
brew install python node git poetry curl

# Create local virtualenv config
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
