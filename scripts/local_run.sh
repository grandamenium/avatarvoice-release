#!/bin/bash
# AvatarVoice Local Startup Script
# Run this to start the local AvatarVoice UI

set -e

echo "=========================================="
echo "  Starting AvatarVoice"
echo "=========================================="

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "Activated virtual environment"
else
    echo "Warning: No virtual environment found. Using system Python."
fi

# Check for .env
if [ ! -f ".env" ]; then
    echo "Error: .env file not found. Please run local_install.sh first."
    exit 1
fi

# Load and check environment
source .env

if [ -z "$GEMINI_API_KEY" ]; then
    echo "Warning: GEMINI_API_KEY not set in .env"
fi

if [ -z "$VIBEVOICE_ENDPOINT" ]; then
    echo "Warning: VIBEVOICE_ENDPOINT not set in .env"
    echo "You'll need this to generate speech. Set it to your GPU's Gradio URL."
fi

# Check for data files
if [ ! -d "data/crema_d/AudioWAV" ]; then
    echo "Error: Audio data not found at data/crema_d/AudioWAV"
    echo "Please run download_cremad.py or copy data from another installation."
    exit 1
fi

if [ ! -f "data/voice_database.sqlite" ]; then
    echo "Error: Database not found at data/voice_database.sqlite"
    echo "Please run build_database.py first."
    exit 1
fi

echo ""
echo "Configuration:"
echo "  VIBEVOICE_ENDPOINT: ${VIBEVOICE_ENDPOINT:-<not set>}"
echo "  GEMINI_API_KEY: ${GEMINI_API_KEY:0:10}..."
echo ""
echo "Starting Gradio UI..."
echo ""

# Run the app
python -m src.avatarvoice_ui.app
