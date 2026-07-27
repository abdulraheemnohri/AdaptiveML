# 📦 Adaptive ML Platform: Installation & Setup Guide

This guide describes how to install and configure the **Adaptive ML Platform** on various operating systems. The platform uses standard, lightweight, and widely available tools to ensure seamless deployment.

---

## 🚀 Quick Start (Automated Installer)

The repository includes pre-configured, automated installation scripts that detect and set up Python, Node.js, Poetry, and system dependencies automatically.

### **Debian / Ubuntu**
```bash
chmod +x scripts/install_debian_ubuntu.sh
./scripts/install_debian_ubuntu.sh
```

### **macOS**
```bash
chmod +x scripts/install_macos.sh
./scripts/install_macos.sh
```

### **Windows (PowerShell)**
```powershell
Set-ExecutionPolicy Bypass -Scope Process
.\scripts\install_windows.ps1
```

### **Generic POSIX Fallback (no Poetry)**
```bash
chmod +x scripts/install_generic.sh
./scripts/install_generic.sh
```

---

## 🛠️ Manual Installation Guide

If you prefer to configure your environment manually, follow these system-by-system steps.

### **1. Prerequisites**
Ensure you have the following packages installed on your host system:
* **Python 3.9 to 3.12** (Python 3.12 is highly recommended)
* **Node.js v18+** & **NPM** (for the React-based Dashboard frontend)
* **Git** and **Curl**

---

### **2. Manual Setup Steps (Debian/Ubuntu)**

#### **A. System Package Setup**
```bash
sudo apt-get update -y
sudo apt-get install -y python3 python3-pip python3-venv python3-dev git curl nodejs npm build-essential
```

#### **B. Setup Poetry**
```bash
curl -sSL https://install.python-poetry.org | python3 -
export PATH="$HOME/.local/bin:$PATH"
```

#### **C. Set Up Virtual Environment & Dependencies**
```bash
# Clone the repository
git clone https://github.com/abdulraheemnohri/AdaptiveML.git
cd AdaptiveML

# Configure poetry virtualenv locally
poetry config virtualenvs.create true --local
poetry env use python3

# Install python dependencies
poetry install --no-root
```

#### **D. Set Up React Frontend**
```bash
cd frontend
npm install
cd ..
```

---

### **3. Manual Setup Steps (macOS)**

#### **A. Install System Packages**
Ensure you have **Homebrew** installed, then execute:
```bash
brew install python node git curl poetry
```

#### **B. Set Up Environment**
```bash
# Clone and enter repo
git clone https://github.com/abdulraheemnohri/AdaptiveML.git
cd AdaptiveML

# Configure poetry env
poetry config virtualenvs.create true --local
poetry env use python3
poetry install --no-root

# Install React dashboard packages
cd frontend
npm install
cd ..
```

---

### **4. Manual Setup Steps (Windows)**

#### **A. Install Prerequisites**
Install Git, Python, and Node.js using **Winget**:
```powershell
winget install --id Git.Git -e --source winget
winget install --id Python.Python.3.11 -e --source winget
winget install --id OpenJS.NodeJS -e --source winget
```

#### **B. Install Poetry**
```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
$env:Path += ";$env:APPDATA\Python\Scripts"
```

#### **C. Install Dependencies**
```powershell
poetry config virtualenvs.create true --local
poetry env use python
poetry install --no-root

cd frontend
npm install
cd ..
```

---

## 🏃 Run the Platform

Once installed, use the following commands to start and verify your platform:

### **Run Python Test Suite**
```bash
poetry run python -m pytest
```

### **Start REST API Server / Backend ModelServer**
```bash
poetry run adaptivebrain serve --port 8000
```

### **Verify Local status**
```bash
poetry run adaptivebrain status
```
