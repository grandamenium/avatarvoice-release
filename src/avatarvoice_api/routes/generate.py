"""Audio generation routes."""

import base64
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from vibevoice_client import VibeVoiceClient, GenerationConfig, VoiceReference
from vibevoice_client.exceptions import GenerationError, ValidationError, ConnectionError
from vibevoice_client.models import EmotionType

from voicematch import VoiceMatchAPI

from ..dependencies import get_voicematch_api, get_vibevoice_client
from ..models.requests import GenerateAudioRequest
from ..models.responses import GenerationResponse

router = APIRouter(prefix="/generate", tags=["Generation"])


def _map_emotion(emotion: Optional[str]) -> Optional[EmotionType]:
    """Map emotion string to EmotionType enum."""
    if not emotion:
        return None

    emotion_map = {
        "neutral": EmotionType.NEUTRAL,
        "happy": EmotionType.HAPPY,
        "sad": EmotionType.SAD,
        "angry": EmotionType.ANGRY,
        "anger": EmotionType.ANGRY,
        "fear": EmotionType.FEAR,
        "disgust": EmotionType.DISGUST,
        "surprise": EmotionType.SURPRISE,
    }

    return emotion_map.get(emotion.lower())


@router.post("/audio", response_model=GenerationResponse)
async def generate_audio(
    request: GenerateAudioRequest,
    tts_client: VibeVoiceClient = Depends(get_vibevoice_client),
    voicematch_api: VoiceMatchAPI = Depends(get_voicematch_api),
) -> GenerationResponse:
    """Generate speech audio from text.

    Supports voice cloning from a reference audio file or matched actor.
    """
    try:
        # Build voice reference
        voice_ref = None

        if request.actor_id:
            # Get voice sample from matched actor
            sample_path = voicematch_api.get_voice_sample(
                actor_id=request.actor_id,
                emotion=request.emotion or "NEU",
            )
            if sample_path:
                voice_ref = VoiceReference(audio_path=sample_path)

        elif request.voice_reference_base64:
            # Decode base64 audio
            audio_data = base64.b64decode(request.voice_reference_base64)
            voice_ref = VoiceReference(audio_bytes=audio_data)

        elif request.voice_reference_path:
            # Use file path
            path = Path(request.voice_reference_path)
            if path.exists():
                voice_ref = VoiceReference(audio_path=path)

        # Build generation config
        config = GenerationConfig(
            text=request.text,
            voice_reference=voice_ref,
            cfg_scale=request.cfg_scale,
            inference_steps=request.inference_steps,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            speed=request.speed,
            emotion=_map_emotion(request.emotion),
            emotion_intensity=request.emotion_intensity,
            sample_rate=request.sample_rate,
            seed=request.seed,
        )

        # Generate audio
        result = await tts_client.generate(config)

        # Encode audio to base64
        audio_base64 = None
        if result.audio_bytes:
            audio_base64 = base64.b64encode(result.audio_bytes).decode()

        return GenerationResponse(
            audio_base64=audio_base64,
            sample_rate=result.sample_rate,
            duration_seconds=result.duration_seconds,
            format="wav",
        )

    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except ConnectionError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"TTS service unavailable: {e}",
        )
    except GenerationError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {e}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {e}",
        )


@router.post("/audio/stream")
async def generate_audio_stream(
    request: GenerateAudioRequest,
    tts_client: VibeVoiceClient = Depends(get_vibevoice_client),
    voicematch_api: VoiceMatchAPI = Depends(get_voicematch_api),
):
    """Generate speech audio with streaming response.

    Returns audio data as a stream for real-time playback.
    """
    try:
        # Build voice reference
        voice_ref = None

        if request.actor_id:
            sample_path = voicematch_api.get_voice_sample(
                actor_id=request.actor_id,
                emotion=request.emotion or "NEU",
            )
            if sample_path:
                voice_ref = VoiceReference(audio_path=sample_path)

        # Build generation config
        config = GenerationConfig(
            text=request.text,
            voice_reference=voice_ref,
            cfg_scale=request.cfg_scale,
            inference_steps=request.inference_steps,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            speed=request.speed,
            emotion=_map_emotion(request.emotion),
            emotion_intensity=request.emotion_intensity,
            sample_rate=request.sample_rate,
            seed=request.seed,
            stream=True,
        )

        async def audio_stream():
            async for chunk in tts_client.generate_stream(config):
                yield chunk.audio_bytes

        return StreamingResponse(
            audio_stream(),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=generated.wav",
            },
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Streaming failed: {e}",
        )
