"""Unit tests for AvatarVoice API routes."""

import pytest
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
import base64

from fastapi.testclient import TestClient

from avatarvoice_api.main import app
from avatarvoice_api.dependencies import APIState
from avatarvoice_api.config import Settings


@pytest.fixture(autouse=True)
def reset_api_state():
    """Reset API state before each test."""
    APIState._voicematch_api = None
    APIState._vibevoice_client = None
    APIState._initialized = False
    yield
    APIState.cleanup()


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_settings(monkeypatch, temp_dir, test_database):
    """Mock settings for testing."""
    settings = Settings(
        gemini_api_key="test_key",
        database_path=str(test_database),
        data_dir=str(temp_dir),
        output_dir=str(temp_dir / "output"),
    )

    def get_test_settings():
        return settings

    monkeypatch.setattr("avatarvoice_api.dependencies.get_settings", get_test_settings)
    monkeypatch.setattr("avatarvoice_api.routes.health.get_settings", get_test_settings)

    return settings


class TestHealthRoutes:
    """Tests for health check endpoints."""

    def test_root_endpoint(self, client):
        """Should return API info."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "AvatarVoice API"
        assert "version" in data

    def test_health_check(self, client, mock_settings):
        """Should return health status."""
        # Mock the VoiceMatch API
        mock_voicematch = Mock()
        mock_vibevoice = Mock()
        mock_vibevoice.health_check = AsyncMock(return_value=False)

        APIState._voicematch_api = mock_voicematch
        APIState._vibevoice_client = mock_vibevoice
        APIState._initialized = True

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "services" in data


class TestAnalyzeRoutes:
    """Tests for image analysis endpoints."""

    @pytest.fixture
    def mock_voicematch_api(self):
        """Create mock VoiceMatch API."""
        mock_api = Mock()

        async def mock_analyze(data, mime_type):
            from voicematch.models import AvatarAnalysisResponse, Gender, Race, Emotion
            return AvatarAnalysisResponse(
                estimated_age=30,
                age_range=(25, 35),
                gender=Gender.MALE,
                race=Race.CAUCASIAN,
                ethnicity=None,
                emotion=Emotion.NEUTRAL,
                confidence=0.85,
            )

        mock_api.analyze_image_bytes = mock_analyze
        return mock_api

    def test_analyze_image_upload(self, client, mock_settings, mock_voicematch_api):
        """Should analyze uploaded image."""
        APIState._voicematch_api = mock_voicematch_api
        APIState._vibevoice_client = Mock()
        APIState._initialized = True

        # Create a fake image file
        image_content = b"\xff\xd8\xff\xe0\x00\x10JFIF"  # JPEG header

        response = client.post(
            "/analyze/image",
            files={"file": ("test.jpg", image_content, "image/jpeg")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["estimated_age"] == 30
        assert data["gender"] == "male"

    def test_analyze_image_base64(self, client, mock_settings, mock_voicematch_api):
        """Should analyze base64-encoded image."""
        APIState._voicematch_api = mock_voicematch_api
        APIState._vibevoice_client = Mock()
        APIState._initialized = True

        image_content = b"\xff\xd8\xff\xe0\x00\x10JFIF"
        encoded = base64.b64encode(image_content).decode()

        response = client.post(
            "/analyze/image/base64",
            json={
                "image_base64": encoded,
                "mime_type": "image/jpeg",
            },
        )

        assert response.status_code == 200

    def test_analyze_matches(self, client, mock_settings, mock_voicematch_api):
        """Should find voice matches."""
        from voicematch.models import MatchResultResponse, ActorResponse

        mock_voicematch_api.find_matches = Mock(return_value=[
            MatchResultResponse(
                actor=ActorResponse(
                    id="1001",
                    age=28,
                    sex="Male",
                    race="African American",
                    sample_count=10,
                ),
                score=0.95,
                match_details={},
            )
        ])

        APIState._voicematch_api = mock_voicematch_api
        APIState._vibevoice_client = Mock()
        APIState._initialized = True

        response = client.post(
            "/analyze/matches",
            json={
                "estimated_age": 30,
                "age_range": [25, 35],
                "gender": "male",
                "race": "caucasian",
                "emotion": "neutral",
                "confidence": 0.85,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["matches"]) == 1


class TestVoicesRoutes:
    """Tests for voice listing endpoints."""

    @pytest.fixture
    def mock_voicematch_api(self):
        """Create mock VoiceMatch API."""
        from voicematch.models import ActorResponse

        mock_api = Mock()
        mock_api.list_voices = Mock(return_value=(
            [
                ActorResponse(id="1001", age=28, sex="Male", race="African American", sample_count=10),
                ActorResponse(id="1002", age=34, sex="Female", race="Caucasian", sample_count=8),
            ],
            2,
        ))
        mock_api.get_actor_details = Mock(return_value=ActorResponse(
            id="1001", age=28, sex="Male", race="African American", sample_count=10
        ))
        return mock_api

    def test_list_voices(self, client, mock_settings, mock_voicematch_api):
        """Should list available voices."""
        APIState._voicematch_api = mock_voicematch_api
        APIState._vibevoice_client = Mock()
        APIState._initialized = True

        response = client.get("/voices")

        assert response.status_code == 200
        data = response.json()
        assert len(data["actors"]) == 2
        assert data["total"] == 2

    def test_list_voices_with_filter(self, client, mock_settings, mock_voicematch_api):
        """Should filter voices by gender."""
        APIState._voicematch_api = mock_voicematch_api
        APIState._vibevoice_client = Mock()
        APIState._initialized = True

        response = client.get("/voices?gender=Male")

        assert response.status_code == 200
        mock_voicematch_api.list_voices.assert_called_with(
            gender="Male",
            race=None,
            age_min=None,
            age_max=None,
            limit=50,
            offset=0,
        )

    def test_get_actor(self, client, mock_settings, mock_voicematch_api):
        """Should get actor details."""
        APIState._voicematch_api = mock_voicematch_api
        APIState._vibevoice_client = Mock()
        APIState._initialized = True

        response = client.get("/voices/1001")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "1001"
        assert data["age"] == 28

    def test_get_actor_not_found(self, client, mock_settings, mock_voicematch_api):
        """Should return 404 for unknown actor."""
        mock_voicematch_api.get_actor_details = Mock(return_value=None)

        APIState._voicematch_api = mock_voicematch_api
        APIState._vibevoice_client = Mock()
        APIState._initialized = True

        response = client.get("/voices/unknown")

        assert response.status_code == 404


class TestGenerateRoutes:
    """Tests for audio generation endpoints."""

    @pytest.fixture
    def mock_clients(self):
        """Create mock clients."""
        from vibevoice_client import GenerationResult

        mock_voicematch = Mock()
        mock_voicematch.get_voice_sample = Mock(return_value=None)

        mock_vibevoice = Mock()

        async def mock_generate(config):
            return GenerationResult(
                audio_bytes=b"fake audio data",
                sample_rate=24000,
                duration_seconds=1.5,
            )

        mock_vibevoice.generate = mock_generate

        return mock_voicematch, mock_vibevoice

    def test_generate_audio(self, client, mock_settings, mock_clients):
        """Should generate audio from text."""
        mock_voicematch, mock_vibevoice = mock_clients

        APIState._voicematch_api = mock_voicematch
        APIState._vibevoice_client = mock_vibevoice
        APIState._initialized = True

        response = client.post(
            "/generate/audio",
            json={
                "text": "Hello, this is a test.",
                "cfg_scale": 2.0,
                "inference_steps": 32,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["audio_base64"] is not None
        assert data["sample_rate"] == 24000

    def test_generate_audio_with_actor(self, client, mock_settings, mock_clients, temp_dir):
        """Should generate audio using actor voice reference."""
        mock_voicematch, mock_vibevoice = mock_clients

        # Create a fake audio file
        sample_path = temp_dir / "sample.wav"
        sample_path.write_bytes(b"RIFF" + b"\x00" * 100)
        mock_voicematch.get_voice_sample = Mock(return_value=sample_path)

        APIState._voicematch_api = mock_voicematch
        APIState._vibevoice_client = mock_vibevoice
        APIState._initialized = True

        response = client.post(
            "/generate/audio",
            json={
                "text": "Hello, this is a test.",
                "actor_id": "1001",
            },
        )

        assert response.status_code == 200

    def test_generate_audio_validation_error(self, client, mock_settings, mock_clients):
        """Should return 400 for empty text."""
        mock_voicematch, mock_vibevoice = mock_clients

        APIState._voicematch_api = mock_voicematch
        APIState._vibevoice_client = mock_vibevoice
        APIState._initialized = True

        response = client.post(
            "/generate/audio",
            json={
                "text": "",  # Empty text should fail validation
            },
        )

        assert response.status_code == 422  # Validation error


class TestModels:
    """Tests for API models."""

    def test_generation_request_defaults(self):
        """Should have sensible defaults."""
        from avatarvoice_api.models.requests import GenerateAudioRequest

        request = GenerateAudioRequest(text="Test text")

        assert request.cfg_scale == 2.0
        assert request.inference_steps == 32
        assert request.temperature == 0.8
        assert request.speed == 1.0

    def test_health_response(self):
        """Should create health response."""
        from avatarvoice_api.models.responses import HealthResponse

        response = HealthResponse(
            status="healthy",
            version="0.1.0",
            services={"voicematch": True, "vibevoice": True},
        )

        assert response.status == "healthy"
        assert response.services["voicematch"] is True


class TestConfig:
    """Tests for API configuration."""

    def test_default_settings(self):
        """Should have default values."""
        settings = Settings()

        assert settings.api_title == "AvatarVoice API"
        assert settings.cors_origins == ["*"]
        assert settings.vibevoice_timeout == 120.0

    def test_custom_settings(self, monkeypatch):
        """Should load from environment."""
        monkeypatch.setenv("GEMINI_API_KEY", "test_key_123")
        monkeypatch.setenv("VIBEVOICE_ENDPOINT", "http://custom:8080")

        settings = Settings()

        assert settings.gemini_api_key == "test_key_123"
        assert settings.vibevoice_endpoint == "http://custom:8080"
