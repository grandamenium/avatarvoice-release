#!/bin/bash
# Download CREMA-D dataset and build database
# Run this if you don't have the data files

set -e

echo "=========================================="
echo "  CREMA-D Dataset Setup"
echo "=========================================="

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Activate venv if exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo ""
echo "[1/2] Downloading CREMA-D dataset from HuggingFace..."
echo "This will download ~600MB of audio files."
echo ""

python scripts/download_cremad.py

echo ""
echo "[2/2] Building voice database..."
echo ""

python scripts/build_database.py

echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "Data files created:"
ls -lh data/voice_database.sqlite 2>/dev/null || echo "  - Database: Not found"
echo "  - Audio files: $(ls -1 data/crema_d/AudioWAV/*.wav 2>/dev/null | wc -l | tr -d ' ') wav files"
echo ""
