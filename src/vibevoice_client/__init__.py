"""VibeVoice TTS Client - Python client for VibeVoice Gradio API."""

from .client import VibeVoiceClient
from .exceptions import (
    VibeVoiceError,
    ConnectionError,
    GenerationError,
    ValidationError,
    TimeoutError,
)
from .models import (
    GenerationConfig,
    GenerationResult,
    VoiceReference,
    StreamingStatus,
)

__version__ = "0.1.0"

__all__ = [
    # Main client
    "VibeVoiceClient",
    # Exceptions
    "VibeVoiceError",
    "ConnectionError",
    "GenerationError",
    "ValidationError",
    "TimeoutError",
    # Models
    "GenerationConfig",
    "GenerationResult",
    "VoiceReference",
    "StreamingStatus",
]
