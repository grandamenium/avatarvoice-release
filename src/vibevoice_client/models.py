"""Pydantic models for VibeVoice client."""

from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class EmotionType(str, Enum):
    """Emotion types for voice generation."""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEAR = "fear"
    DISGUST = "disgust"
    SURPRISE = "surprise"


class StreamingStatus(str, Enum):
    """Status of streaming generation."""
    PENDING = "pending"
    PROCESSING = "processing"
    STREAMING = "streaming"
    COMPLETE = "complete"
    ERROR = "error"


class VoiceReference(BaseModel):
    """Voice reference for TTS generation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    audio_path: Optional[Path] = Field(None, description="Path to reference audio file")
    audio_bytes: Optional[bytes] = Field(None, description="Raw audio bytes")
    audio_url: Optional[str] = Field(None, description="URL to reference audio")
    speaker_id: Optional[str] = Field(None, description="Pre-defined speaker ID")

    def to_gradio_input(self) -> Union[str, bytes, None]:
        """Convert to Gradio-compatible input."""
        if self.audio_path and self.audio_path.exists():
            return str(self.audio_path)
        if self.audio_bytes:
            return self.audio_bytes
        if self.audio_url:
            return self.audio_url
        return self.speaker_id


class GenerationConfig(BaseModel):
    """Configuration for TTS generation."""

    model_config = ConfigDict(extra="forbid")

    # Text input
    text: str = Field(..., min_length=1, max_length=5000, description="Text to synthesize")

    # Voice reference
    voice_reference: Optional[VoiceReference] = Field(
        None, description="Voice reference for cloning"
    )

    # Generation parameters
    cfg_scale: float = Field(
        default=2.0,
        ge=0.1,
        le=10.0,
        description="Classifier-free guidance scale"
    )
    inference_steps: int = Field(
        default=32,
        ge=1,
        le=100,
        description="Number of diffusion inference steps"
    )
    temperature: float = Field(
        default=0.8,
        ge=0.1,
        le=2.0,
        description="Sampling temperature"
    )
    top_p: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Top-p (nucleus) sampling"
    )
    top_k: int = Field(
        default=50,
        ge=0,
        le=1000,
        description="Top-k sampling"
    )

    # Emotion control
    emotion: Optional[EmotionType] = Field(
        None,
        description="Target emotion for generation"
    )
    emotion_intensity: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Emotion intensity multiplier"
    )

    # Audio settings
    sample_rate: int = Field(
        default=24000,
        description="Output sample rate in Hz"
    )
    speed: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Speech speed multiplier"
    )

    # Advanced options
    seed: Optional[int] = Field(
        None,
        ge=0,
        description="Random seed for reproducibility"
    )
    stream: bool = Field(
        default=False,
        description="Enable streaming generation"
    )


class GenerationResult(BaseModel):
    """Result of TTS generation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    audio_path: Optional[Path] = Field(None, description="Path to generated audio file")
    audio_bytes: Optional[bytes] = Field(None, description="Raw audio bytes")
    sample_rate: int = Field(default=24000, description="Audio sample rate")
    duration_seconds: float = Field(default=0.0, description="Audio duration in seconds")
    config_used: Optional[GenerationConfig] = Field(
        None, description="Config used for generation"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional generation metadata"
    )

    @property
    def has_audio(self) -> bool:
        """Check if result contains audio."""
        return self.audio_path is not None or self.audio_bytes is not None


class StreamingChunk(BaseModel):
    """A chunk of streaming audio data."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    audio_bytes: bytes = Field(..., description="Audio chunk data")
    chunk_index: int = Field(..., ge=0, description="Index of this chunk")
    is_final: bool = Field(default=False, description="Whether this is the last chunk")
    sample_rate: int = Field(default=24000, description="Audio sample rate")


class StreamingState(BaseModel):
    """State of streaming generation."""

    status: StreamingStatus = Field(
        default=StreamingStatus.PENDING,
        description="Current streaming status"
    )
    chunks_received: int = Field(default=0, ge=0, description="Number of chunks received")
    total_duration: float = Field(default=0.0, description="Total audio duration so far")
    error_message: Optional[str] = Field(None, description="Error message if failed")
