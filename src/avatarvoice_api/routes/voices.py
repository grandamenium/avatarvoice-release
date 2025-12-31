"""Voice listing and sample routes."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse

from voicematch import VoiceMatchAPI
from voicematch.exceptions import VoiceNotFoundError

from ..dependencies import get_voicematch_api
from ..models.responses import VoiceListResponse, ActorInfo

router = APIRouter(prefix="/voices", tags=["Voices"])


@router.get("", response_model=VoiceListResponse)
async def list_voices(
    gender: Optional[str] = Query(None, description="Filter by gender (Male/Female)"),
    race: Optional[str] = Query(None, description="Filter by race"),
    age_min: Optional[int] = Query(None, ge=0, description="Minimum age"),
    age_max: Optional[int] = Query(None, ge=0, description="Maximum age"),
    limit: int = Query(50, ge=1, le=100, description="Results per page"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    api: VoiceMatchAPI = Depends(get_voicematch_api),
) -> VoiceListResponse:
    """List available voice actors with optional filtering."""
    actors, total = api.list_voices(
        gender=gender,
        race=race,
        age_min=age_min,
        age_max=age_max,
        limit=limit,
        offset=offset,
    )

    return VoiceListResponse(
        actors=[
            ActorInfo(
                id=a.id,
                age=a.age,
                sex=a.sex,
                race=a.race,
                ethnicity=a.ethnicity,
                sample_count=a.sample_count,
            )
            for a in actors
        ],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{actor_id}", response_model=ActorInfo)
async def get_actor(
    actor_id: str,
    api: VoiceMatchAPI = Depends(get_voicematch_api),
) -> ActorInfo:
    """Get details for a specific voice actor."""
    actor = api.get_actor_details(actor_id)

    if actor is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Actor not found: {actor_id}",
        )

    return ActorInfo(
        id=actor.id,
        age=actor.age,
        sex=actor.sex,
        race=actor.race,
        ethnicity=actor.ethnicity,
        sample_count=actor.sample_count,
    )


@router.get("/{actor_id}/sample")
async def get_voice_sample(
    actor_id: str,
    emotion: str = Query("NEU", description="Emotion code (ANG, DIS, FEA, HAP, NEU, SAD)"),
    duration: float = Query(10.0, ge=1.0, le=60.0, description="Target duration in seconds"),
    api: VoiceMatchAPI = Depends(get_voicematch_api),
) -> FileResponse:
    """Get a voice sample for an actor.

    Returns an audio file with the actor's voice.
    """
    try:
        sample_path = api.get_voice_sample(
            actor_id=actor_id,
            emotion=emotion,
            duration_seconds=duration,
        )

        if sample_path is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No voice sample available for actor {actor_id}",
            )

        return FileResponse(
            path=sample_path,
            media_type="audio/wav",
            filename=f"{actor_id}_{emotion}_sample.wav",
        )

    except VoiceNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Actor not found: {actor_id}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get sample: {e}",
        )
