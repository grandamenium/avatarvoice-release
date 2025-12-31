"""Pydantic models for VoiceMatch API."""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field


class Gender(str, Enum):
    """Gender enumeration."""
    MALE = "male"
    FEMALE = "female"
    AMBIGUOUS = "ambiguous"


class Race(str, Enum):
    """Race enumeration matching CREMA-D dataset categories."""
    AFRICAN_AMERICAN = "african_american"
    ASIAN = "asian"
    CAUCASIAN = "caucasian"
    HISPANIC = "hispanic"
    MIXED = "mixed"
    AMBIGUOUS = "ambiguous"


class Emotion(str, Enum):
    """Emotion enumeration matching CREMA-D dataset."""
    ANGER = "anger"
    DISGUST = "disgust"
    FEAR = "fear"
    HAPPY = "happy"
    NEUTRAL = "neutral"
    SAD = "sad"
    AMBIGUOUS = "ambiguous"


# CREMA-D emotion codes
EMOTION_CODES = {
    "ANG": Emotion.ANGER,
    "DIS": Emotion.DISGUST,
    "FEA": Emotion.FEAR,
    "HAP": Emotion.HAPPY,
    "NEU": Emotion.NEUTRAL,
    "SAD": Emotion.SAD,
}

EMOTION_TO_CODE = {v: k for k, v in EMOTION_CODES.items()}


class AvatarAnalysisResponse(BaseModel):
    """API response for image analysis."""

    model_config = ConfigDict(use_enum_values=True)

    estimated_age: int = Field(..., ge=0, le=120, description="Estimated age of the person")
    age_range: Tuple[int, int] = Field(..., description="Age confidence interval (min, max)")
    gender: Gender = Field(..., description="Detected gender")
    race: Race = Field(..., description="Detected race/ethnicity")
    ethnicity: Optional[str] = Field(None, description="Additional ethnicity notes")
    emotion: Emotion = Field(default=Emotion.NEUTRAL, description="Detected emotional state")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Analysis confidence score")
    raw_response: Optional[Dict[str, Any]] = Field(None, description="Raw Gemini API response")


class ActorResponse(BaseModel):
    """API response for voice actor."""

    model_config = ConfigDict(from_attributes=True)

    id: str = Field(..., description="Actor identifier from CREMA-D")
    age: int = Field(..., ge=0, description="Actor's age")
    sex: str = Field(..., description="Actor's sex (Male/Female)")
    race: str = Field(..., description="Actor's race")
    ethnicity: Optional[str] = Field(None, description="Actor's ethnicity")
    sample_count: int = Field(default=0, ge=0, description="Number of audio samples available")


class MatchResultResponse(BaseModel):
    """API response for voice match."""

    model_config = ConfigDict(from_attributes=True)

    actor: ActorResponse = Field(..., description="Matched voice actor")
    score: float = Field(..., ge=0.0, le=1.0, description="Match score")
    match_details: Dict[str, Any] = Field(default_factory=dict, description="Breakdown of scoring")


class VoiceListResponse(BaseModel):
    """API response for voice listing."""

    actors: List[ActorResponse] = Field(default_factory=list, description="List of voice actors")
    total: int = Field(default=0, ge=0, description="Total count (for pagination)")
    limit: int = Field(default=50, ge=1, description="Results per page")
    offset: int = Field(default=0, ge=0, description="Pagination offset")


class AnalyzeRequest(BaseModel):
    """Request model for image analysis."""

    model_config = ConfigDict(extra="forbid")

    mime_type: str = Field(..., description="MIME type of the image")


class FindMatchesRequest(BaseModel):
    """Request model for finding voice matches."""

    model_config = ConfigDict(extra="forbid")

    analysis: AvatarAnalysisResponse = Field(..., description="Demographics analysis")
    limit: int = Field(default=5, ge=1, le=50, description="Max results to return")
    emotion_filter: Optional[str] = Field(None, description="CREMA-D emotion code filter")


class VoiceListRequest(BaseModel):
    """Request model for listing voices."""

    model_config = ConfigDict(extra="forbid")

    gender: Optional[str] = Field(None, description="Filter by gender (Male/Female)")
    race: Optional[str] = Field(None, description="Filter by race")
    age_min: Optional[int] = Field(None, ge=0, description="Minimum age filter")
    age_max: Optional[int] = Field(None, ge=0, description="Maximum age filter")
    limit: int = Field(default=50, ge=1, le=100, description="Results per page")
    offset: int = Field(default=0, ge=0, description="Pagination offset")
