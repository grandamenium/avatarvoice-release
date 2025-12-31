"""Configuration for AvatarVoice API."""

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment."""

    # API Settings
    api_title: str = "AvatarVoice API"
    api_version: str = "0.1.0"
    api_description: str = "API for avatar voice matching and TTS generation"
    debug: bool = False

    # CORS Settings
    cors_origins: list[str] = ["*"]
    cors_allow_credentials: bool = True

    # Gemini API
    gemini_api_key: Optional[str] = None

    # Database
    database_path: str = "./data/voice_database.sqlite"

    # Audio Storage
    data_dir: str = "./data/crema_d"
    output_dir: str = "./output"

    # VibeVoice TTS Settings
    vibevoice_endpoint: str = "http://localhost:7860"
    vibevoice_timeout: float = 120.0

    # Rate Limiting
    rate_limit_requests: int = 100
    rate_limit_period: int = 60  # seconds

    class Config:
        """Pydantic settings config."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
