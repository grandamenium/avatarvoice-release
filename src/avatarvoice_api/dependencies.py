"""FastAPI dependencies for AvatarVoice API."""

from pathlib import Path

from fastapi import Depends, HTTPException, status

from voicematch import VoiceMatchAPI
from vibevoice_client import VibeVoiceClient

from .config import Settings, get_settings


class APIState:
    """Singleton state holder for API dependencies."""

    _voicematch_api: VoiceMatchAPI = None
    _vibevoice_client: VibeVoiceClient = None
    _initialized: bool = False

    @classmethod
    def initialize(cls, settings: Settings) -> None:
        """Initialize API state with settings."""
        if cls._initialized:
            return

        # Initialize VoiceMatch API
        try:
            from voicematch.config import Config as VMConfig
            VMConfig.reset()  # Reset singleton

            config = VMConfig(
                gemini_api_key=settings.gemini_api_key or "",
                data_dir=Path(settings.data_dir),
                output_dir=Path(settings.output_dir),
                database_path=Path(settings.database_path),
            )
            cls._voicematch_api = VoiceMatchAPI(config=config)
        except Exception as e:
            # Log error but don't fail - API might be used without Gemini
            print(f"Warning: VoiceMatch API initialization failed: {e}")
            cls._voicematch_api = None

        # Initialize VibeVoice client
        cls._vibevoice_client = VibeVoiceClient(
            endpoint=settings.vibevoice_endpoint,
            timeout=settings.vibevoice_timeout,
        )

        cls._initialized = True

    @classmethod
    def get_voicematch_api(cls) -> VoiceMatchAPI:
        """Get VoiceMatch API instance."""
        if cls._voicematch_api is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="VoiceMatch API not available. Check GEMINI_API_KEY configuration.",
            )
        return cls._voicematch_api

    @classmethod
    def get_vibevoice_client(cls) -> VibeVoiceClient:
        """Get VibeVoice client instance."""
        if cls._vibevoice_client is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="VibeVoice client not available.",
            )
        return cls._vibevoice_client

    @classmethod
    def cleanup(cls) -> None:
        """Cleanup resources."""
        if cls._voicematch_api:
            cls._voicematch_api.close()
            cls._voicematch_api = None
        cls._initialized = False


def get_voicematch_api(
    settings: Settings = Depends(get_settings),
) -> VoiceMatchAPI:
    """Dependency to get VoiceMatch API instance."""
    APIState.initialize(settings)
    return APIState.get_voicematch_api()


def get_vibevoice_client(
    settings: Settings = Depends(get_settings),
) -> VibeVoiceClient:
    """Dependency to get VibeVoice client instance."""
    APIState.initialize(settings)
    return APIState.get_vibevoice_client()
