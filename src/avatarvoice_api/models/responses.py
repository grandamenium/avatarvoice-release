"""Response models for AvatarVoice API."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(default="healthy")
    version: str = Field(...)
    services: Dict[str, bool] = Field(default_factory=dict)


class AnalysisResponse(BaseModel):
    """Response from image analysis."""

    model_config = ConfigDict(from_attributes=True)

    estimated_age: int
    age_range: tuple[int, int]
    gender: str
    race: str
    ethnicity: Optional[str] = None
    emotion: str = "neutral"
    confidence: float


class ActorInfo(BaseModel):
    """Voice actor information."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    age: int
    sex: str
    race: str
    ethnicity: Optional[str] = None
    sample_count: int = 0


class MatchInfo(BaseModel):
    """Voice match information."""

    model_config = ConfigDict(from_attributes=True)

    actor: ActorInfo
    score: float
    match_details: Dict[str, Any] = Field(default_factory=dict)


class MatchResponse(BaseModel):
    """Response with voice matches."""

    matches: List[MatchInfo] = Field(default_factory=list)
    total: int = 0


class VoiceListResponse(BaseModel):
    """Response with voice listing."""

    actors: List[ActorInfo] = Field(default_factory=list)
    total: int = 0
    limit: int = 50
    offset: int = 0


class GenerationResponse(BaseModel):
    """Response from audio generation."""

    audio_base64: Optional[str] = Field(
        None, description="Base64-encoded audio data"
    )
    audio_url: Optional[str] = Field(
        None, description="URL to download audio"
    )
    sample_rate: int = 24000
    duration_seconds: float = 0.0
    format: str = "wav"


class ErrorResponse(BaseModel):
    """Error response."""

    error: str
    detail: Optional[str] = None
    code: Optional[str] = None
