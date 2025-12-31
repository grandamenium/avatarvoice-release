"""Pydantic models for AvatarVoice API."""

from .requests import (
    AnalyzeImageRequest,
    FindMatchesRequest,
    ListVoicesRequest,
    GenerateAudioRequest,
)
from .responses import (
    HealthResponse,
    AnalysisResponse,
    MatchResponse,
    VoiceListResponse,
    GenerationResponse,
    ErrorResponse,
)

__all__ = [
    # Requests
    "AnalyzeImageRequest",
    "FindMatchesRequest",
    "ListVoicesRequest",
    "GenerateAudioRequest",
    # Responses
    "HealthResponse",
    "AnalysisResponse",
    "MatchResponse",
    "VoiceListResponse",
    "GenerationResponse",
    "ErrorResponse",
]
