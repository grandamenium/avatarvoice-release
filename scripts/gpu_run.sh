#!/bin/bash
# VibeVoice GPU Startup Script
# Run this on your GPU instance to start the VibeVoice TTS server

set -e

echo "=========================================="
echo "  Starting VibeVoice TTS Server"
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "gradio_demo.py" ]; then
    echo "Error: gradio_demo.py not found."
    echo "Please run this script from the VibeVoice demo directory."
    exit 1
fi

# Detect model path
MODEL_PATH=""
if [ -d "./models/VibeVoice-Large" ]; then
    MODEL_PATH="./models/VibeVoice-Large"
elif [ -d "./models/VibeVoice-1.5B" ]; then
    MODEL_PATH="./models/VibeVoice-1.5B"
else
    echo "Error: No model found in ./models/"
    echo "Please run gpu_install.sh first or download a model manually."
    exit 1
fi

echo "Using model: $MODEL_PATH"
echo ""

# Start the server with auto-load
echo "Starting Gradio server with auto-load..."
echo "The model will load automatically on startup."
echo ""
echo "Once started, copy the public URL (*.gradio.live) and set it as"
echo "VIBEVOICE_ENDPOINT in your local .env file."
echo ""

python gradio_demo.py --model_path "$MODEL_PATH" --share
