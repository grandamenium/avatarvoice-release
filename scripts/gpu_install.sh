#!/bin/bash
# VibeVoice GPU Installation Script
# Run this on your GPU instance (e.g., ArchitectDock)

set -e

echo "=========================================="
echo "  VibeVoice GPU Installation"
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "gradio_demo.py" ]; then
    echo "Error: gradio_demo.py not found."
    echo "Please run this script from the VibeVoice demo directory."
    echo "Example: cd /workspace/VibeVoiceTTS/demo && bash scripts/gpu_install.sh"
    exit 1
fi

# Install dependencies
echo ""
echo "[1/4] Installing Python dependencies..."
pip install -q gradio torch transformers huggingface_hub soundfile librosa numpy

# Create models directory
echo ""
echo "[2/4] Creating models directory..."
mkdir -p ./models

# Check if model exists
if [ -d "./models/VibeVoice-Large" ] || [ -d "./models/VibeVoice-1.5B" ]; then
    echo "[3/4] Model already downloaded. Skipping download."
else
    echo "[3/4] Downloading VibeVoice model..."
    echo "This may take a while (~2.7GB for 1.5B model)..."
    python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='microsoft/VibeVoice-1.5B',
    local_dir='./models/VibeVoice-1.5B',
    local_dir_use_symlinks=False
)
print('Download complete!')
"
fi

# Create voices directory if it doesn't exist
echo ""
echo "[4/4] Setting up voices directory..."
mkdir -p ./voices

echo ""
echo "=========================================="
echo "  Installation Complete!"
echo "=========================================="
echo ""
echo "To start VibeVoice, run:"
echo "  bash scripts/gpu_run.sh"
echo ""
echo "Or manually:"
echo "  python gradio_demo.py --model_path ./models/VibeVoice-1.5B"
echo ""
