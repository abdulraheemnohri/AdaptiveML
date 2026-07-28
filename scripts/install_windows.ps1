# Windows Installation Script for Adaptive ML Platform.
# Sets up Python, Poetry, Node.js, and dependencies.

Write-Output "=== Adaptive ML Platform: Windows PowerShell Installer ==="

# Check for Administrator privileges
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Warning "Running as standard user. If dependencies are missing, administrator privilege may be required."
}

# Check for Git
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Output "-> Git not found. Installing via Winget..."
    winget install --id Git.Git -e --source winget
}

# Check for Python 3
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Output "-> Python not found. Installing via Winget..."
    winget install --id Python.Python.3.11 -e --source winget
}

# Check for Node.js
if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    Write-Output "-> Node.js not found. Installing via Winget..."
    winget install --id OpenJS.NodeJS -e --source winget
}

# Check for Poetry
if (-not (Get-Command poetry -ErrorAction SilentlyContinue)) {
    Write-Output "-> Poetry not found. Installing via official script..."
    (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
    $env:Path += ";$env:APPDATA\Python\Scripts"
}

# Configure virtual environment
Write-Output "-> Creating and configuring Python virtual environment..."
poetry config virtualenvs.create true --local
poetry env use python

Write-Output "-> Installing backend dependencies..."
poetry install --no-root

# Frontend
if (Test-Path "frontend") {
    cd frontend
    if (Get-Command npm -ErrorAction SilentlyContinue) {
        Write-Output "-> Running npm install..."
        npm install
    } else {
        Write-Warning "npm not found. Please install Node.js and run 'npm install' in 'frontend' folder."
    }
    cd ..
}

Write-Output "=== Installation Completed Successfully! ==="
Write-Output "To run the backend ModelServer:"
Write-Output "  poetry run adaptivebrain serve --port 8000"
Write-Output "To view system dashboard status:"
Write-Output "  poetry run adaptivebrain status"
