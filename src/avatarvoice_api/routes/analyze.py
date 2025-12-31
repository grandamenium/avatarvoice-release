"""Image analysis routes."""

import base64

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status

from voicematch import VoiceMatchAPI
from voicematch.exceptions import ImageAnalysisError, UnsupportedFormatError

from ..dependencies import get_voicematch_api
from ..models.requests import AnalyzeImageRequest, FindMatchesRequest
from ..models.responses import AnalysisResponse, MatchResponse, MatchInfo, ActorInfo

router = APIRouter(prefix="/analyze", tags=["Analysis"])


@router.post("/image", response_model=AnalysisResponse)
async def analyze_image(
    file: UploadFile = File(...),
    api: VoiceMatchAPI = Depends(get_voicematch_api),
) -> AnalysisResponse:
    """Analyze an uploaded avatar image for demographics.

    Returns estimated age, gender, race, emotion, and confidence score.
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type: {file.content_type}. Expected image.",
        )

    try:
        # Read file content
        content = await file.read()

        # Analyze using VoiceMatch API
        result = await api.analyze_image_bytes(content, file.content_type)

        return AnalysisResponse(
            estimated_age=result.estimated_age,
            age_range=result.age_range,
            gender=result.gender,
            race=result.race,
            ethnicity=result.ethnicity,
            emotion=result.emotion,
            confidence=result.confidence,
        )

    except UnsupportedFormatError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except ImageAnalysisError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {e}",
        )


@router.post("/image/base64", response_model=AnalysisResponse)
async def analyze_image_base64(
    request: AnalyzeImageRequest,
    api: VoiceMatchAPI = Depends(get_voicematch_api),
) -> AnalysisResponse:
    """Analyze a base64-encoded avatar image for demographics."""
    if not request.image_base64:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="image_base64 is required",
        )

    try:
        # Decode base64
        image_data = base64.b64decode(request.image_base64)

        # Analyze
        result = await api.analyze_image_bytes(image_data, request.mime_type)

        return AnalysisResponse(
            estimated_age=result.estimated_age,
            age_range=result.age_range,
            gender=result.gender,
            race=result.race,
            ethnicity=result.ethnicity,
            emotion=result.emotion,
            confidence=result.confidence,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {e}",
        )


@router.post("/matches", response_model=MatchResponse)
async def find_matches(
    request: FindMatchesRequest,
    api: VoiceMatchAPI = Depends(get_voicematch_api),
) -> MatchResponse:
    """Find voice actors matching the provided demographics."""
    from voicematch.models import AvatarAnalysisResponse, Gender, Race, Emotion

    try:
        # Build analysis response from request
        analysis = AvatarAnalysisResponse(
            estimated_age=request.estimated_age,
            age_range=request.age_range,
            gender=Gender(request.gender),
            race=Race(request.race),
            ethnicity=request.ethnicity,
            emotion=Emotion(request.emotion),
            confidence=request.confidence,
        )

        # Find matches
        matches = api.find_matches(
            analysis,
            limit=request.limit,
            emotion_filter=request.emotion_filter,
        )

        # Convert to response format
        match_infos = []
        for m in matches:
            match_infos.append(MatchInfo(
                actor=ActorInfo(
                    id=m.actor.id,
                    age=m.actor.age,
                    sex=m.actor.sex,
                    race=m.actor.race,
                    ethnicity=m.actor.ethnicity,
                    sample_count=m.actor.sample_count,
                ),
                score=m.score,
                match_details=m.match_details,
            ))

        return MatchResponse(
            matches=match_infos,
            total=len(match_infos),
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Match finding failed: {e}",
        )
