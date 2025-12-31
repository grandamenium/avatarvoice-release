"""Request models for AvatarVoice API."""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class AnalyzeImageRequest(BaseModel):
    """Request to analyze an avatar image."""

    model_config = ConfigDict(extra="forbid")

    # Image can be provided as base64 or URL
    image_base64: Optional[str] = Field(
        None, description="Base64-encoded image data"
    )
    image_url: Optional[str] = Field(
        None, description="URL to image file"
    )
    mime_type: str = Field(
        default="image/jpeg",
        description="MIME type of the image"
    )


class FindMatchesRequest(BaseModel):
    """Request to find matching voices for demographics."""

    model_config = ConfigDict(extra="forbid")

    # Demographics (can come from analysis or manual input)
    estimated_age: int = Field(..., ge=0, le=120)
    age_range: tuple[int, int] = Field(...)
    gender: str = Field(...)
    race: str = Field(...)
    ethnicity: Optional[str] = Field(None)
    emotion: str = Field(default="neutral")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)

    # Match options
    limit: int = Field(default=5, ge=1, le=50)
    emotion_filter: Optional[str] = Field(
        None, description="Filter by CREMA-D emotion code"
    )


class ListVoicesRequest(BaseModel):
    """Request to list available voices with filters."""

    model_config = ConfigDict(extra="forbid")

    gender: Optional[str] = Field(None, description="Filter by gender")
    race: Optional[str] = Field(None, description="Filter by race")
    age_min: Optional[int] = Field(None, ge=0)
    age_max: Optional[int] = Field(None, ge=0)
    limit: int = Field(default=50, ge=1, le=100)
    offset: int = Field(default=0, ge=0)


class GenerateAudioRequest(BaseModel):
    """Request to generate TTS audio."""

    model_config = ConfigDict(extra="forbid")

    # Required fields
    text: str = Field(..., min_length=1, max_length=5000)

    # Voice reference
    voice_reference_path: Optional[str] = Field(
        None, description="Path to reference audio file"
    )
    voice_reference_base64: Optional[str] = Field(
        None, description="Base64-encoded reference audio"
    )
    actor_id: Optional[str] = Field(
        None, description="Actor ID to use as voice reference"
    )

    # Generation parameters
    cfg_scale: float = Field(default=2.0, ge=0.1, le=10.0)
    inference_steps: int = Field(default=32, ge=1, le=100)
    temperature: float = Field(default=0.8, ge=0.1, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0, le=1000)
    speed: float = Field(default=1.0, ge=0.5, le=2.0)

    # Emotion control
    emotion: Optional[str] = Field(None, description="Target emotion")
    emotion_intensity: float = Field(default=1.0, ge=0.0, le=2.0)

    # Audio settings
    sample_rate: int = Field(default=24000)
    seed: Optional[int] = Field(None, ge=0)
