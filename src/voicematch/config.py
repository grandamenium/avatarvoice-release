"""Configuration management for VoiceMatch pipeline."""

from dataclasses import dataclass
from pathlib import Path
import os
from typing import ClassVar, Optional

from dotenv import load_dotenv


class ConfigError(Exception):
    """Raised when configuration is invalid."""

    pass


@dataclass
class Config:
    """Application configuration."""

    gemini_api_key: str
    data_dir: Path
    output_dir: Path
    database_path: Path

    # Singleton instance
    _instance: ClassVar[Optional["Config"]] = None

    @classmethod
    def load(cls, env_path: Optional[Path] = None) -> "Config":
        """Load configuration from environment.

        Args:
            env_path: Optional path to .env file. If not provided,
                     searches in current directory and parent directories.

        Returns:
            Config instance with loaded values.

        Raises:
            ConfigError: If required configuration is missing or invalid.
        """
        if cls._instance is not None:
            return cls._instance

        # Load .env file
        if env_path:
            load_dotenv(env_path)
        else:
            load_dotenv()

        # Get required values
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if gemini_api_key is None:
            raise ConfigError("GEMINI_API_KEY environment variable is required")

        if not gemini_api_key.strip():
            raise ConfigError("GEMINI_API_KEY cannot be empty")

        # Get optional values with defaults
        # NOTE: Database stores full paths like "data/crema_d/AudioWAV/..."
        # so data_dir should be project root, not "./data/crema_d"
        data_dir_str = os.getenv("DATA_DIR", ".")
        output_dir_str = os.getenv("OUTPUT_DIR", "./output")
        database_path_str = os.getenv("DATABASE_PATH", "./data/voice_database.sqlite")

        # Resolve to absolute paths
        data_dir = Path(data_dir_str).resolve()
        output_dir = Path(output_dir_str).resolve()
        database_path = Path(database_path_str).resolve()

        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        cls._instance = cls(
            gemini_api_key=gemini_api_key,
            data_dir=data_dir,
            output_dir=output_dir,
            database_path=database_path,
        )

        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset singleton instance (useful for testing)."""
        cls._instance = None
