# AvatarVoice - Claude Code Context

## Project Overview

AvatarVoice is a two-part system:
- **Local**: Gradio UI + VoiceMatch API (image analysis, voice matching)
- **GPU**: VibeVoice TTS server (speech generation)

The local app connects to a remote GPU running VibeVoice via Gradio's public URL.

---

## Quick Start (Full Setup)

### Prerequisites
- ArchitectDock account with GPU access
- Gemini API key
- Git installed locally

### Step 1: GPU Server (ArchitectDock)

1. **Deploy GPU instance**
   - Image: `hygoinc/avatar-to-vibe-voice`
   - GPU: RTX 4090 (or similar)
   - Network Volume: `vibevoice-webui-prod` (California)
   - Mount Path: `/workspace`

2. **Start VibeVoice server** (via SSH or web terminal)
   ```bash
   cd /workspace/VibeVoiceTTS
   source .venv/bin/activate
   cd demo
   python gradio_demo.py --model_path ../models/VibeVoice-Large --share
   ```

3. **Copy the public URL** from output:
   ```
   Running on public URL: https://xxxxx.gradio.live
   ```

### Step 2: Local Setup

1. **Clone the repo**
   ```bash
   git clone <repo-url>
   cd avatarvoice
   ```

2. **Install dependencies**
   ```bash
   bash scripts/local_install.sh
   ```

3. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env:
   #   GEMINI_API_KEY=your-gemini-api-key
   #   VIBEVOICE_ENDPOINT=https://xxxxx.gradio.live  (from GPU step)
   ```

4. **Start the app**
   ```bash
   bash scripts/local_run.sh
   ```

5. **Open in browser**: http://localhost:7861

---

## GPU Instance (ArchitectDock)

### Infrastructure

- **Platform**: ArchitectDock (GPU cloud)
- **GPU**: RTX 4090 24GB
- **Docker Image**: `hygoinc/avatar-to-vibe-voice`
- **Network Volume**: `vibevoice-webui-prod` (100GB, California)
- **Mount Path**: `/workspace`

### Volume Contents

```
/workspace/
└── VibeVoiceTTS/
    ├── .venv/                  # Python virtual environment
    ├── demo/
    │   └── gradio_demo.py      # Main Gradio app
    ├── models/
    │   └── VibeVoice-Large/    # TTS model (~3GB)
    └── vibevoice/              # Core library
```

### Starting the GPU Server

```bash
# SSH into GPU instance (or use web terminal)

# Navigate to VibeVoiceTTS and activate venv
cd /workspace/VibeVoiceTTS
source .venv/bin/activate

# Start the Gradio server
cd demo
python gradio_demo.py --model_path ../models/VibeVoice-Large --share
```

**Output will show:**
```
Running on local URL: http://0.0.0.0:7860
Running on public URL: https://xxxxx.gradio.live  <-- Copy this
```

### GPU API Endpoints

The VibeVoice Gradio server exposes:

| Endpoint | Purpose |
|----------|---------|
| `/generate_with_audio_reference` | Clone voice from reference audio |
| `/generate_podcast_wrapper` | Generate with built-in speakers |

### Connecting Local to GPU

1. Copy the `https://xxxxx.gradio.live` URL from GPU output
2. Set in local `.env`:
   ```
   VIBEVOICE_ENDPOINT=https://xxxxx.gradio.live
   ```
3. Restart local app: `bash scripts/local_run.sh`

---

## Local Development

### Key Files

```
src/
├── avatarvoice_ui/
│   └── app.py              # Gradio UI (port 7861)
├── voicematch/
│   ├── api.py              # Voice matching logic
│   ├── config.py           # Environment config
│   ├── gemini_analyzer.py  # Gemini Vision integration
│   └── database.py         # SQLite voice database
└── vibevoice_client/
    └── client.py           # GPU TTS client
```

### Data Flow

```
1. User uploads image
2. Gemini Vision analyzes → demographics
3. VoiceMatch queries SQLite → matching actors
4. User selects actor + enters text
5. VibeVoice client calls GPU → generated audio
6. Audio returned to UI for playback
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | Yes | Google Gemini API key |
| `VIBEVOICE_ENDPOINT` | Yes | GPU's Gradio public URL |
| `DATA_DIR` | No | Data directory (default: `.`) |
| `DATABASE_PATH` | No | SQLite path (default: `./data/voice_database.sqlite`) |

---

## Common Issues

### "No sample found" after voice match
- **Cause**: Path resolution issue
- **Fix**: Ensure `DATA_DIR` is `.` or omitted in `.env`

### TTS generation fails
- **Cause**: GPU endpoint not reachable
- **Fix**:
  1. Check GPU is running (see "Starting the GPU Server" above)
  2. Verify URL in `VIBEVOICE_ENDPOINT`
  3. Gradio live URLs expire after ~72 hours

### GPU won't deploy (ArchitectDock)
- **Cause**: Volume already attached to another instance
- **Fix**: Stop other instances using the volume first

### Model not loading on GPU
- **Cause**: Model files missing from volume
- **Fix**: Ensure `/workspace/VibeVoiceTTS/models/VibeVoice-Large` exists

---

## Testing Without GPU

The local system can partially function without GPU:

| Feature | Works Without GPU |
|---------|-------------------|
| Image analysis (Gemini) | Yes |
| Voice matching | Yes |
| Voice preview (samples) | Yes |
| Speech generation | No |

To test locally without GPU, leave `VIBEVOICE_ENDPOINT` empty. Everything except "Generate Speech" will work.

---

## Deployment Notes

### For Production

1. **GPU**: Use persistent GPU instance with static endpoint
2. **Local**: Can be dockerized or run directly
3. **Railway**: API can be deployed to Railway (see `railway.toml`)

### Current Setup

- GPU: ArchitectDock with saved image/volume
- Local: Clone repo, run scripts
- Data: Included in repo (~600MB audio files)
