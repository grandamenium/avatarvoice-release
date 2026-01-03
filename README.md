# AvatarVoice

AI-powered voice cloning pipeline that analyzes avatar images, matches demographics to voice actors, and generates speech using VibeVoice TTS.

## How It Works

1. **Image Analysis** - Upload an avatar image, Gemini Vision extracts demographics (age, gender, ethnicity, emotion)
2. **Voice Matching** - Matches demographics against 91 voice actors from CREMA-D dataset (7,442 samples)
3. **Speech Generation** - Generates speech using VibeVoice TTS on a GPU server

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     LOCAL MACHINE                            │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              AvatarVoice UI (Gradio)                │    │
│  │                  localhost:7861                      │    │
│  └──────────────────────┬──────────────────────────────┘    │
│                         │                                    │
│  ┌──────────────────────┴──────────────────────────────┐    │
│  │              VoiceMatch API                          │    │
│  │  • Gemini Vision (image analysis)                   │    │
│  │  • SQLite database (voice matching)                 │    │
│  │  • CREMA-D audio samples                            │    │
│  └──────────────────────┬──────────────────────────────┘    │
└─────────────────────────┼───────────────────────────────────┘
                          │ HTTPS
┌─────────────────────────┼───────────────────────────────────┐
│                     GPU SERVER                               │
│  ┌──────────────────────┴──────────────────────────────┐    │
│  │           VibeVoice TTS (Gradio)                    │    │
│  │            https://xxx.gradio.live                   │    │
│  │  • VibeVoice-Large model                            │    │
│  │  • Voice cloning from reference audio               │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- ArchitectDock account with GPU access
- Gemini API key ([get one here](https://makersuite.google.com/app/apikey))
- Git installed locally

### 1. GPU Setup (ArchitectDock)

**Deploy a new GPU instance:**
- Image: `hygoinc/avatar-to-vibe-voice`
- GPU: RTX 4090 (or similar)
- Network Volume: `vibevoice-webui-prod` (California region)
- Mount Path: `/workspace`

**Start VibeVoice** (via SSH or web terminal):

```bash
cd /workspace/VibeVoiceTTS
source .venv/bin/activate
cd demo
python gradio_demo.py --model_path ../models/VibeVoice-Large --share
```

**Copy the public URL** from the output:
```
Running on public URL: https://xxxxx.gradio.live
```

### 2. Local Setup

```bash
# Clone the repo (includes ~600MB of voice data)
git clone https://github.com/grandamenium/avatarvoice.git
cd avatarvoice

# Install dependencies
bash scripts/local_install.sh

# Configure environment
cp .env.example .env
# Edit .env and set:
#   GEMINI_API_KEY=your_key_here
#   VIBEVOICE_ENDPOINT=https://xxxxx.gradio.live  (from GPU step)

# Start the UI
bash scripts/local_run.sh
```

**Open http://localhost:7861**

## Configuration

Edit `.env` with your settings:

```bash
# Required
GEMINI_API_KEY=your_gemini_api_key

# VibeVoice GPU endpoint (from gpu_run.sh output)
VIBEVOICE_ENDPOINT=https://xxxx.gradio.live
```

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/local_install.sh` | Set up local environment (venv, deps, .env) |
| `scripts/local_run.sh` | Start AvatarVoice UI on localhost:7861 |
| `scripts/gpu_install.sh` | Install VibeVoice on GPU (one-time) |
| `scripts/gpu_run.sh` | Start VibeVoice TTS server on GPU |
| `scripts/setup_data.sh` | Download CREMA-D dataset (if not included) |
| `scripts/package_data.sh` | Package data for transfer |

## Data

The repository includes:
- `data/voice_database.sqlite` - Voice actor metadata (91 actors)
- `data/crema_d/AudioWAV/` - 7,442 audio samples (~600MB)
- `data/crema_d/VideoDemographics.csv` - Actor demographics

## API Endpoints

The system exposes REST APIs for programmatic access:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/analyze/image` | POST | Analyze avatar image |
| `/analyze/matches` | POST | Find matching voices |
| `/voices` | GET | List voice actors |
| `/voices/{id}/sample` | GET | Get voice sample |
| `/generate/audio` | POST | Generate speech |
| `/pipeline/generate` | POST | Full pipeline (SSE) |

## Development

```bash
# Run tests
pytest

# Type checking
mypy src/

# Linting
ruff check src/
```

## Troubleshooting

**"No sample found"** - Check DATA_DIR in .env (should be `.` or omitted)

**"VIBEVOICE_ENDPOINT not set"** - Set the GPU's Gradio URL in .env

**GPU won't start** - Ensure model exists at `/workspace/VibeVoiceTTS/models/VibeVoice-Large`

## License

MIT
