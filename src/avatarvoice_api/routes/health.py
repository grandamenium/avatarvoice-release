"""Health check routes."""

from fastapi import APIRouter, Depends

from ..config import Settings, get_settings
from ..dependencies import APIState
from ..models.responses import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    """Check API health and service availability."""
    APIState.initialize(settings)

    services = {
        "voicematch": APIState._voicematch_api is not None,
        "vibevoice": APIState._vibevoice_client is not None,
    }

    # Check VibeVoice endpoint health
    if APIState._vibevoice_client:
        try:
            services["vibevoice_endpoint"] = await APIState._vibevoice_client.health_check()
        except Exception:
            services["vibevoice_endpoint"] = False

    return HealthResponse(
        status="healthy" if all(services.values()) else "degraded",
        version=settings.api_version,
        services=services,
    )


@router.get("/")
async def root() -> dict:
    """Root endpoint."""
    return {
        "name": "AvatarVoice API",
        "version": "0.1.0",
        "docs": "/docs",
    }
