#!/bin/bash
# AvatarVoice Local Installation Script
# Run this on your local machine

set -e

echo "=========================================="
echo "  AvatarVoice Local Installation"
echo "=========================================="

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "Project root: $PROJECT_ROOT"
echo ""

# Check Python version
echo "[1/5] Checking Python..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
    echo "Python version: $PYTHON_VERSION"
else
    echo "Error: Python 3 not found. Please install Python 3.9+."
    exit 1
fi

# Create virtual environment if it doesn't exist
echo ""
echo "[2/5] Setting up virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "Created virtual environment"
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
echo ""
echo "[3/5] Installing Python dependencies..."
pip install -q --upgrade pip
pip install -q -e .

# Check for .env file
echo ""
echo "[4/5] Checking configuration..."
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "Created .env from .env.example"
        echo ""
        echo "IMPORTANT: Edit .env and set your API keys:"
        echo "  - GEMINI_API_KEY: Your Google Gemini API key"
        echo "  - VIBEVOICE_ENDPOINT: Your GPU's Gradio URL (e.g., https://xxx.gradio.live)"
    else
        echo "Warning: No .env file found. Please create one."
    fi
else
    echo ".env file exists"
fi

# Check for data files
echo ""
echo "[5/5] Checking data files..."
if [ -d "data/crema_d/AudioWAV" ] && [ -f "data/voice_database.sqlite" ]; then
    AUDIO_COUNT=$(ls -1 data/crema_d/AudioWAV/*.wav 2>/dev/null | wc -l)
    echo "Data files found: $AUDIO_COUNT audio samples"
else
    echo "Warning: Data files not found."
    echo ""
    echo "To download CREMA-D dataset:"
    echo "  python scripts/download_cremad.py"
    echo "  python scripts/build_database.py"
    echo ""
    echo "Or copy existing data folder from another installation."
fi

echo ""
echo "=========================================="
echo "  Installation Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Edit .env with your API keys"
echo "2. Ensure data files exist (or run download_cremad.py)"
echo "3. Start the server: bash scripts/local_run.sh"
echo ""
