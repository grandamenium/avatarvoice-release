"""Integration tests for the full AvatarVoice pipeline.

These tests verify that all components work together correctly.
Set INTEGRATION_TEST=1 to run these tests.
"""

import os
import pytest
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

# Skip if not running integration tests
pytestmark = pytest.mark.skipif(
    os.getenv("INTEGRATION_TEST") != "1",
    reason="Integration tests disabled. Set INTEGRATION_TEST=1 to run.",
)


@pytest.fixture
def test_image(temp_dir):
    """Create a test image file."""
    # Create a minimal JPEG image
    image_path = temp_dir / "test_avatar.jpg"
    # Minimal JPEG data
    jpeg_data = bytes([
        0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46,
        0x49, 0x46, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01,
        0x00, 0x01, 0x00, 0x00, 0xFF, 0xD9
    ])
    image_path.write_bytes(jpeg_data)
    return image_path


class TestVoiceMatchAPIIntegration:
    """Integration tests for VoiceMatch API."""

    @pytest.fixture
    def api(self, temp_dir, test_database, mock_env):
        """Create VoiceMatch API with test configuration."""
        from voicematch.config import Config
        from voicematch.api import VoiceMatchAPI

        Config.reset()

        (temp_dir / "output").mkdir(exist_ok=True)

        config = Config(
            gemini_api_key="test_key",
            data_dir=temp_dir,
            output_dir=temp_dir / "output",
            database_path=test_database,
        )

        return VoiceMatchAPI(config=config)

    def test_list_and_get_voices(self, api):
        """Should list voices and get individual actor details."""
        # List voices
        actors, total = api.list_voices(limit=10)

        assert total > 0
        assert len(actors) > 0

        # Get individual actor
        first_actor = actors[0]
        details = api.get_actor_details(first_actor.id)

        assert details is not None
        assert details.id == first_actor.id

    def test_find_matches_workflow(self, api):
        """Should find matches for demographics."""
        from voicematch.models import AvatarAnalysisResponse, Gender, Race, Emotion

        # Create analysis
        analysis = AvatarAnalysisResponse(
            estimated_age=30,
            age_range=(25, 35),
            gender=Gender.MALE,
            race=Race.CAUCASIAN,
            emotion=Emotion.NEUTRAL,
            confidence=0.85,
        )

        # Find matches
        matches = api.find_matches(analysis, limit=5)

        assert len(matches) > 0
        assert matches[0].score <= 1.0
        assert matches[0].score >= 0.0

        # Verify sorting (highest score first)
        scores = [m.score for m in matches]
        assert scores == sorted(scores, reverse=True)

    def test_filter_by_demographics(self, api):
        """Should filter voices by various demographics."""
        # Filter by gender
        males, male_total = api.list_voices(gender="Male")
        females, female_total = api.list_voices(gender="Female")

        for actor in males:
            assert actor.sex == "Male"

        for actor in females:
            assert actor.sex == "Female"

        # Filter by age range
        young, _ = api.list_voices(age_min=20, age_max=30)

        for actor in young:
            assert 20 <= actor.age <= 30


class TestVibeVoiceClientIntegration:
    """Integration tests for VibeVoice client."""

    @pytest.fixture
    def client(self):
        """Create VibeVoice client."""
        from vibevoice_client import VibeVoiceClient
        return VibeVoiceClient(endpoint="http://localhost:7860")

    @pytest.mark.asyncio
    async def test_client_initialization(self, client):
        """Should initialize client correctly."""
        assert client.endpoint == "http://localhost:7860"
        assert client.timeout == 120.0

    def test_generation_config_validation(self):
        """Should validate generation config."""
        from vibevoice_client import GenerationConfig

        # Valid config
        config = GenerationConfig(
            text="Hello world",
            cfg_scale=2.0,
            inference_steps=32,
        )
        assert config.text == "Hello world"

        # Invalid config (empty text)
        with pytest.raises(Exception):
            GenerationConfig(text="")


class TestFastAPIIntegration:
    """Integration tests for FastAPI backend."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient
        from avatarvoice_api.main import app
        return TestClient(app)

    def test_api_root(self, client):
        """Should return API info at root."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data

    def test_health_endpoint(self, client, mock_env, temp_dir, test_database):
        """Should return health status."""
        from avatarvoice_api.dependencies import APIState
        from avatarvoice_api.config import Settings

        # Setup mock state
        APIState._voicematch_api = Mock()
        APIState._vibevoice_client = Mock()
        APIState._vibevoice_client.health_check = AsyncMock(return_value=False)
        APIState._initialized = True

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "services" in data

        # Cleanup
        APIState.cleanup()


class TestOrchestratorIntegration:
    """Integration tests for the orchestrator service."""

    @pytest.fixture
    def orchestrator(self, temp_dir, test_database, mock_env):
        """Create orchestrator with test services."""
        from voicematch.config import Config
        from voicematch.api import VoiceMatchAPI
        from vibevoice_client import VibeVoiceClient
        from avatarvoice_api.services.orchestrator import AvatarVoiceOrchestrator

        Config.reset()

        (temp_dir / "output").mkdir(exist_ok=True)

        config = Config(
            gemini_api_key="test_key",
            data_dir=temp_dir,
            output_dir=temp_dir / "output",
            database_path=test_database,
        )

        voicematch = VoiceMatchAPI(config=config)
        vibevoice = VibeVoiceClient()

        return AvatarVoiceOrchestrator(voicematch, vibevoice)

    def test_find_matching_voices(self, orchestrator):
        """Should find matching voices through orchestrator."""
        from voicematch.models import AvatarAnalysisResponse, Gender, Race, Emotion

        analysis = AvatarAnalysisResponse(
            estimated_age=35,
            age_range=(30, 40),
            gender=Gender.FEMALE,
            race=Race.AFRICAN_AMERICAN,
            emotion=Emotion.HAPPY,
            confidence=0.9,
        )

        matches = orchestrator.find_matching_voices(analysis, limit=3)

        # Should return matches (or empty list if no exact match)
        assert isinstance(matches, list)

    def test_emotion_mapping(self, orchestrator):
        """Should correctly map emotions to codes."""
        assert orchestrator._emotion_to_code("happy") == "HAP"
        assert orchestrator._emotion_to_code("sad") == "SAD"
        assert orchestrator._emotion_to_code("angry") == "ANG"
        assert orchestrator._emotion_to_code("neutral") == "NEU"
        assert orchestrator._emotion_to_code("unknown") == "NEU"  # Default


class TestEndToEndWorkflow:
    """End-to-end workflow tests."""

    @pytest.fixture
    def setup(self, temp_dir, test_database, mock_env):
        """Setup test environment."""
        from voicematch.config import Config
        from voicematch.api import VoiceMatchAPI

        Config.reset()

        (temp_dir / "output").mkdir(exist_ok=True)

        config = Config(
            gemini_api_key="test_key",
            data_dir=temp_dir,
            output_dir=temp_dir / "output",
            database_path=test_database,
        )

        api = VoiceMatchAPI(config=config)

        return {
            "api": api,
            "temp_dir": temp_dir,
        }

    def test_complete_voice_matching_flow(self, setup):
        """Test complete flow: demographics -> match -> voice sample."""
        api = setup["api"]

        from voicematch.models import AvatarAnalysisResponse, Gender, Race, Emotion

        # Step 1: Create demographics (simulating image analysis)
        analysis = AvatarAnalysisResponse(
            estimated_age=28,
            age_range=(25, 32),
            gender=Gender.MALE,
            race=Race.AFRICAN_AMERICAN,
            emotion=Emotion.NEUTRAL,
            confidence=0.85,
        )

        # Step 2: Find matching voices
        matches = api.find_matches(analysis, limit=5)

        assert len(matches) > 0, "Should find at least one match"

        # Step 3: Get best match details
        best_match = matches[0]
        actor_details = api.get_actor_details(best_match.actor.id)

        assert actor_details is not None
        assert actor_details.sample_count >= 0

        # Step 4: Attempt to get voice sample (may be None if files don't exist)
        sample_path = api.get_voice_sample(
            actor_id=best_match.actor.id,
            emotion="NEU",
        )

        # Sample may be None if audio files aren't present
        # This is expected in test environment
        assert sample_path is None or isinstance(sample_path, Path)

    def test_pagination_works(self, setup):
        """Test that pagination works correctly."""
        api = setup["api"]

        # Get first page
        page1, total1 = api.list_voices(limit=2, offset=0)

        # Get second page
        page2, total2 = api.list_voices(limit=2, offset=2)

        # Totals should be the same
        assert total1 == total2

        # Pages should have different actors
        if len(page1) > 0 and len(page2) > 0:
            ids1 = {a.id for a in page1}
            ids2 = {a.id for a in page2}
            assert ids1.isdisjoint(ids2), "Pages should have different actors"
