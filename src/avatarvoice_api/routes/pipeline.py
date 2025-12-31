"""Pipeline route for end-to-end AvatarVoice generation."""

import base64
import json
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from voicematch.api import VoiceMatchAPI
from vibevoice_client import VibeVoiceClient

from ..dependencies import get_voicematch_api, get_vibevoice_client
from ..config import get_settings, Settings
from ..services.pipeline_orchestrator import PipelineOrchestrator

# Import prompt_optimizer conditionally
try:
    from voicematch.prompt_optimizer import PromptOptimizer
    HAS_PROMPT_OPTIMIZER = True
except ImportError:
    HAS_PROMPT_OPTIMIZER = False
    PromptOptimizer = None


router = APIRouter(prefix="/pipeline", tags=["pipeline"])


class PipelineRequest(BaseModel):
    """Request model for pipeline generation."""

    media_base64: str = Field(
        ...,
        description="Base64 encoded image or video",
    )
    mime_type: str = Field(
        ...,
        description="MIME type of the media (image/jpeg, image/png, video/mp4)",
    )
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Text to synthesize",
    )
    cfg_scale: float = Field(
        default=2.0,
        ge=0.1,
        le=10.0,
        description="Voice consistency (1.0-2.0 recommended)",
    )
    optimize_prompt: bool = Field(
        default=True,
        description="Whether to optimize text for TTS",
    )
    stream: bool = Field(
        default=True,
        description="Stream progress updates via SSE",
    )


class PipelineResponse(BaseModel):
    """Response model for non-streaming pipeline generation."""

    success: bool = Field(..., description="Whether generation succeeded")
    audio_base64: Optional[str] = Field(None, description="Base64 encoded audio")
    duration: Optional[float] = Field(None, description="Audio duration in seconds")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Generation metadata")
    error: Optional[str] = Field(None, description="Error message if failed")
    step: Optional[str] = Field(None, description="Step where error occurred")


def get_prompt_optimizer(settings: Settings = Depends(get_settings)) -> Optional["PromptOptimizer"]:
    """Get prompt optimizer if available."""
    if not HAS_PROMPT_OPTIMIZER:
        return None
    try:
        return PromptOptimizer(api_key=settings.gemini_api_key)
    except Exception:
        return None


def get_pipeline_orchestrator(
    voicematch_api: VoiceMatchAPI = Depends(get_voicematch_api),
    vibevoice_client: VibeVoiceClient = Depends(get_vibevoice_client),
    settings: Settings = Depends(get_settings),
) -> PipelineOrchestrator:
    """Get pipeline orchestrator with all dependencies."""
    optimizer = get_prompt_optimizer(settings)
    return PipelineOrchestrator(
        voicematch_api=voicematch_api,
        vibevoice_client=vibevoice_client,
        prompt_optimizer=optimizer,
    )


async def stream_generator(
    orchestrator: PipelineOrchestrator,
    media_bytes: bytes,
    mime_type: str,
    text: str,
    cfg_scale: float,
    optimize_prompt: bool,
):
    """Generate SSE events from pipeline.

    Args:
        orchestrator: Pipeline orchestrator instance.
        media_bytes: Raw media bytes.
        mime_type: MIME type of the media.
        text: Text to synthesize.
        cfg_scale: CFG scale parameter.
        optimize_prompt: Whether to optimize the prompt.

    Yields:
        SSE events with progress updates and final result.
    """
    async for update in orchestrator.generate_streaming(
        media_bytes=media_bytes,
        mime_type=mime_type,
        text=text,
        cfg_scale=cfg_scale,
        optimize_prompt=optimize_prompt,
    ):
        step = update.get("step", "")
        if step == "complete":
            event_type = "complete"
        elif step == "error":
            event_type = "error"
        else:
            event_type = "progress"

        yield {
            "event": event_type,
            "data": json.dumps(update),
        }


@router.post("/generate")
async def generate_pipeline(
    request: PipelineRequest,
    orchestrator: PipelineOrchestrator = Depends(get_pipeline_orchestrator),
):
    """Full end-to-end pipeline for avatar voice generation.

    This endpoint:
    1. Analyzes the provided image/video for demographics
    2. Matches to appropriate voice actors
    3. Optionally optimizes the text for TTS
    4. Generates speech using VibeVoice

    Args:
        request: Pipeline request with media and text.
        orchestrator: Injected pipeline orchestrator.

    Returns:
        If stream=true: SSE stream with progress updates
        If stream=false: JSON response with audio data
    """
    try:
        media_bytes = base64.b64decode(request.media_base64)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid base64 encoding: {str(e)}",
        )

    if request.stream:
        return EventSourceResponse(
            stream_generator(
                orchestrator=orchestrator,
                media_bytes=media_bytes,
                mime_type=request.mime_type,
                text=request.text,
                cfg_scale=request.cfg_scale,
                optimize_prompt=request.optimize_prompt,
            ),
            media_type="text/event-stream",
        )
    else:
        try:
            result = await orchestrator.generate(
                media_bytes=media_bytes,
                mime_type=request.mime_type,
                text=request.text,
                cfg_scale=request.cfg_scale,
                optimize_prompt=request.optimize_prompt,
            )

            return PipelineResponse(
                success=True,
                audio_base64=base64.b64encode(result.audio_bytes).decode(),
                duration=result.duration,
                metadata={
                    "voice_actor": result.voice_actor,
                    "emotion": result.emotion,
                    "original_text": result.original_text,
                    "optimized_text": result.optimized_text,
                    "demographics": result.demographics,
                },
            )
        except ValueError as e:
            return PipelineResponse(
                success=False,
                error=str(e),
            )
        except Exception as e:
            return PipelineResponse(
                success=False,
                error=f"Pipeline failed: {str(e)}",
            )
