"""Unit tests for configuration management."""

import pytest

from voicematch.config import Config, ConfigError


class TestConfigLoading:
    """Tests for Config.load() functionality."""

    def test_config_loads_from_env_variables(self, mock_env):
        """Test that config loads values from environment variables."""
        config = Config.load()

        assert config.gemini_api_key == "test_api_key_12345"
        assert config.data_dir.is_absolute()
        assert config.output_dir.is_absolute()
        assert config.database_path.is_absolute()

    def test_config_raises_on_missing_api_key(self, temp_dir, monkeypatch):
        """Test that ConfigError is raised when GEMINI_API_KEY is missing."""
        from unittest.mock import patch

        Config.reset()

        # Clear the environment variable
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        # Mock load_dotenv to prevent it from loading the .env file
        with patch("voicematch.config.load_dotenv"):
            with pytest.raises(ConfigError, match="GEMINI_API_KEY"):
                Config.load()

    def test_config_raises_on_empty_api_key(self, temp_dir, monkeypatch):
        """Test that ConfigError is raised when GEMINI_API_KEY is empty."""
        Config.reset()

        monkeypatch.setenv("GEMINI_API_KEY", "")

        with pytest.raises(ConfigError, match="cannot be empty"):
            Config.load()

    def test_config_uses_default_data_dir(self, temp_dir, monkeypatch):
        """Test that config uses default DATA_DIR when not specified."""
        Config.reset()

        monkeypatch.setenv("GEMINI_API_KEY", "test_key")
        monkeypatch.delenv("DATA_DIR", raising=False)

        config = Config.load()

        # Should contain 'data/crema_d' somewhere in the path
        assert "crema_d" in str(config.data_dir)

    def test_config_resolves_absolute_paths(self, mock_env):
        """Test that all paths are resolved to absolute paths."""
        config = Config.load()

        assert config.data_dir.is_absolute()
        assert config.output_dir.is_absolute()
        assert config.database_path.is_absolute()


class TestConfigSingleton:
    """Tests for Config singleton pattern."""

    def test_config_singleton_pattern(self, mock_env):
        """Test that Config.load() returns the same instance."""
        config1 = Config.load()
        config2 = Config.load()

        assert config1 is config2

    def test_config_reset_clears_singleton(self, mock_env):
        """Test that Config.reset() clears the singleton instance."""
        config1 = Config.load()
        Config.reset()
        config2 = Config.load()

        # After reset, should be a new instance (but with same values)
        assert config1 is not config2


class TestConfigEdgeCases:
    """Edge case tests for configuration."""

    def test_config_creates_output_dir_if_missing(self, temp_dir, monkeypatch):
        """Test that output directory is created if it doesn't exist."""
        Config.reset()

        output_path = temp_dir / "new_output_dir"
        assert not output_path.exists()

        monkeypatch.setenv("GEMINI_API_KEY", "test_key")
        monkeypatch.setenv("OUTPUT_DIR", str(output_path))

        config = Config.load()

        # Use resolve() to handle macOS symlinks (/var -> /private/var)
        assert output_path.resolve().exists()
        assert config.output_dir == output_path.resolve()

    def test_config_handles_whitespace_in_api_key(self, temp_dir, monkeypatch):
        """Test that API key with only whitespace is rejected."""
        Config.reset()

        monkeypatch.setenv("GEMINI_API_KEY", "   ")

        with pytest.raises(ConfigError, match="cannot be empty"):
            Config.load()
