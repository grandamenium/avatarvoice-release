"""VoiceMatch - Voice matching pipeline for AI avatars."""

from .api import VoiceMatchAPI
from .exceptions import (
    VoiceMatchError,
    ImageAnalysisError,
    VoiceNotFoundError,
    DatabaseError,
    ConfigurationError,
    UnsupportedFormatError,
    APIConnectionError,
)
from .models import (
    Gender,
    Race,
    Emotion,
    AvatarAnalysisResponse,
    ActorResponse,
    MatchResultResponse,
    VoiceListResponse,
    EMOTION_CODES,
    EMOTION_TO_CODE,
)

__version__ = "0.1.0"

__all__ = [
    # Main API
    "VoiceMatchAPI",
    # Exceptions
    "VoiceMatchError",
    "ImageAnalysisError",
    "VoiceNotFoundError",
    "DatabaseError",
    "ConfigurationError",
    "UnsupportedFormatError",
    "APIConnectionError",
    # Models
    "Gender",
    "Race",
    "Emotion",
    "AvatarAnalysisResponse",
    "ActorResponse",
    "MatchResultResponse",
    "VoiceListResponse",
    "EMOTION_CODES",
    "EMOTION_TO_CODE",
]
