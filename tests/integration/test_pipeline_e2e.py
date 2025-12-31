"""End-to-end integration tests for pipeline API.

These tests require the full stack to be running and configured.
Run with: pytest tests/integration/test_pipeline_e2e.py -v

Set INTEGRATION_TEST=1 to enable these tests.
"""

import base64
import json
import os
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient


# Skip all tests if INTEGRATION_TEST is not set
pytestmark = pytest.mark.skipif(
    os.getenv("INTEGRATION_TEST") != "1",
    reason="Integration tests disabled. Set INTEGRATION_TEST=1 to enable."
)


@pytest.fixture
def test_image_bytes():
    """Create a minimal valid JPEG image for testing."""
    # Minimal JPEG (1x1 red pixel)
    jpeg_data = bytes([
        0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01,
        0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43,
        0x00, 0x08, 0x06, 0x06, 0x07, 0x06, 0x05, 0x08, 0x07, 0x07, 0x07, 0x09,
        0x09, 0x08, 0x0A, 0x0C, 0x14, 0x0D, 0x0C, 0x0B, 0x0B, 0x0C, 0x19, 0x12,
        0x13, 0x0F, 0x14, 0x1D, 0x1A, 0x1F, 0x1E, 0x1D, 0x1A, 0x1C, 0x1C, 0x20,
        0x24, 0x2E, 0x27, 0x20, 0x22, 0x2C, 0x23, 0x1C, 0x1C, 0x28, 0x37, 0x29,
        0x2C, 0x30, 0x31, 0x34, 0x34, 0x34, 0x1F, 0x27, 0x39, 0x3D, 0x38, 0x32,
        0x3C, 0x2E, 0x33, 0x34, 0x32, 0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x01,
        0x00, 0x01, 0x01, 0x01, 0x11, 0x00, 0xFF, 0xC4, 0x00, 0x1F, 0x00, 0x00,
        0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
        0x09, 0x0A, 0x0B, 0xFF, 0xC4, 0x00, 0xB5, 0x10, 0x00, 0x02, 0x01, 0x03,
        0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00, 0x00, 0x01, 0x7D,
        0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06,
        0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08,
        0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0, 0x24, 0x33, 0x62, 0x72,
        0x82, 0x09, 0x0A, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28,
        0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45,
        0x46, 0x47, 0x48, 0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
        0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x73, 0x74, 0x75,
        0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
        0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3,
        0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6,
        0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9,
        0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2,
        0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1, 0xF2, 0xF3, 0xF4,
        0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA, 0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01,
        0x00, 0x00, 0x3F, 0x00, 0xFB, 0xD3, 0x28, 0xA0, 0x02, 0x80, 0xFF, 0xD9
    ])
    return jpeg_data


@pytest.fixture
def test_image_base64(test_image_bytes):
    """Get base64 encoded test image."""
    return base64.b64encode(test_image_bytes).decode()


class TestPipelineE2EMocked:
    """End-to-end tests with mocked external services."""

    @pytest.fixture
    def mock_voicematch_api(self):
        """Create a mock VoiceMatchAPI."""
        mock = MagicMock()
        mock.analyze_image_bytes = AsyncMock()
        mock.find_matches = MagicMock()
        mock.get_voice_sample = MagicMock()
        return mock

    @pytest.fixture
    def mock_vibevoice_client(self):
        """Create a mock VibeVoiceClient."""
        mock = MagicMock()
        mock.generate = AsyncMock()
        return mock

    @pytest.fixture
    def mock_analysis_response(self):
        """Create a mock analysis response."""
        mock = MagicMock()
        mock.estimated_age = 35
        mock.gender = "male"
        mock.race = "caucasian"
        mock.emotion = "happy"
        return mock

    @pytest.fixture
    def mock_match_result(self):
        """Create a mock match result."""
        mock = MagicMock()
        mock.actor = MagicMock()
        mock.actor.id = "1050"
        mock.score = 0.95
        return mock

    @pytest.fixture
    def mock_generation_result(self):
        """Create a mock generation result."""
        mock = MagicMock()
        mock.audio_bytes = b"fake audio bytes for testing"
        mock.duration_seconds = 3.5
        return mock

    @pytest.fixture
    def app_with_mocks(
        self,
        mock_voicematch_api,
        mock_vibevoice_client,
        mock_analysis_response,
        mock_match_result,
        mock_generation_result,
        temp_dir,
    ):
        """Create app with mocked dependencies."""
        from avatarvoice_api.main import create_app
        from avatarvoice_api.routes.pipeline import get_pipeline_orchestrator
        from avatarvoice_api.services.pipeline_orchestrator import PipelineOrchestrator

        # Setup mocks
        mock_voicematch_api.analyze_image_bytes.return_value = mock_analysis_response
        mock_voicematch_api.find_matches.return_value = [mock_match_result]
        sample_path = temp_dir / "sample.wav"
        sample_path.write_bytes(b"fake wav data")
        mock_voicematch_api.get_voice_sample.return_value = sample_path
        mock_vibevoice_client.generate.return_value = mock_generation_result

        app = create_app()

        # Create a mock orchestrator with our mocks
        mock_orchestrator = PipelineOrchestrator(
            voicematch_api=mock_voicematch_api,
            vibevoice_client=mock_vibevoice_client,
            prompt_optimizer=None,
        )

        # Override the dependency
        app.dependency_overrides[get_pipeline_orchestrator] = lambda: mock_orchestrator

        yield app

        # Cleanup
        app.dependency_overrides.clear()

    def test_non_streaming_generation(
        self,
        app_with_mocks,
        test_image_base64,
    ):
        """Test full pipeline with non-streaming response."""
        with TestClient(app_with_mocks) as client:
            response = client.post(
                "/pipeline/generate",
                json={
                    "media_base64": test_image_base64,
                    "mime_type": "image/jpeg",
                    "text": "Hello, this is a test.",
                    "cfg_scale": 2.0,
                    "optimize_prompt": False,
                    "stream": False,
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["audio_base64"] is not None
        assert data["duration"] == 3.5
        assert "metadata" in data
        assert data["metadata"]["voice_actor"] == "1050"

    def test_streaming_generation(
        self,
        app_with_mocks,
        test_image_base64,
    ):
        """Test full pipeline with streaming SSE response."""
        with TestClient(app_with_mocks) as client:
            # Note: TestClient doesn't properly handle SSE, so we just verify
            # the endpoint accepts streaming requests
            response = client.post(
                "/pipeline/generate",
                json={
                    "media_base64": test_image_base64,
                    "mime_type": "image/jpeg",
                    "text": "Hello, this is a test.",
                    "stream": True,
                },
            )

        # SSE returns 200 with text/event-stream content type
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")

    def test_invalid_base64(self, app_with_mocks):
        """Test error handling for invalid base64."""
        with TestClient(app_with_mocks) as client:
            response = client.post(
                "/pipeline/generate",
                json={
                    "media_base64": "not-valid-base64!!!",
                    "mime_type": "image/jpeg",
                    "text": "Hello",
                    "stream": False,
                },
            )

        assert response.status_code == 400
        assert "Invalid base64" in response.json()["detail"]

    def test_missing_required_fields(self, app_with_mocks):
        """Test error handling for missing required fields."""
        with TestClient(app_with_mocks) as client:
            response = client.post(
                "/pipeline/generate",
                json={
                    "media_base64": "SGVsbG8=",
                    # Missing mime_type and text
                },
            )

        assert response.status_code == 422  # Validation error


class TestPipelineWithOptimization:
    """Tests for pipeline with prompt optimization enabled."""

    @pytest.fixture
    def mock_voicematch_api(self):
        """Create a mock VoiceMatchAPI."""
        mock = MagicMock()
        mock.analyze_image_bytes = AsyncMock()
        mock.find_matches = MagicMock()
        mock.get_voice_sample = MagicMock()
        return mock

    @pytest.fixture
    def mock_vibevoice_client(self):
        """Create a mock VibeVoiceClient."""
        mock = MagicMock()
        mock.generate = AsyncMock()
        return mock

    @pytest.fixture
    def mock_prompt_optimizer(self):
        """Create a mock PromptOptimizer."""
        mock = MagicMock()
        mock.optimize = AsyncMock(return_value="Optimized text!")
        return mock

    @pytest.fixture
    def mock_analysis_response(self):
        """Create a mock analysis response."""
        mock = MagicMock()
        mock.estimated_age = 35
        mock.gender = "male"
        mock.race = "caucasian"
        mock.emotion = "happy"
        return mock

    @pytest.fixture
    def mock_match_result(self):
        """Create a mock match result."""
        mock = MagicMock()
        mock.actor = MagicMock()
        mock.actor.id = "1050"
        mock.score = 0.95
        return mock

    @pytest.fixture
    def mock_generation_result(self):
        """Create a mock generation result."""
        mock = MagicMock()
        mock.audio_bytes = b"fake audio bytes"
        mock.duration_seconds = 3.5
        return mock

    @pytest.mark.asyncio
    async def test_optimization_applied(
        self,
        mock_voicematch_api,
        mock_vibevoice_client,
        mock_prompt_optimizer,
        mock_analysis_response,
        mock_match_result,
        mock_generation_result,
        temp_dir,
    ):
        """Test that prompt optimization is applied when enabled."""
        from avatarvoice_api.services.pipeline_orchestrator import PipelineOrchestrator

        # Setup mocks
        mock_voicematch_api.analyze_image_bytes.return_value = mock_analysis_response
        mock_voicematch_api.find_matches.return_value = [mock_match_result]
        sample_path = temp_dir / "sample.wav"
        sample_path.write_bytes(b"fake wav data")
        mock_voicematch_api.get_voice_sample.return_value = sample_path
        mock_vibevoice_client.generate.return_value = mock_generation_result

        orchestrator = PipelineOrchestrator(
            voicematch_api=mock_voicematch_api,
            vibevoice_client=mock_vibevoice_client,
            prompt_optimizer=mock_prompt_optimizer,
        )

        result = await orchestrator.generate(
            media_bytes=b"fake image",
            mime_type="image/jpeg",
            text="Original text",
            optimize_prompt=True,
        )

        # Verify optimizer was called
        mock_prompt_optimizer.optimize.assert_called_once()

        # Verify optimized text was used
        assert result.optimized_text == "Optimized text!"

    @pytest.mark.asyncio
    async def test_optimization_skipped_when_disabled(
        self,
        mock_voicematch_api,
        mock_vibevoice_client,
        mock_prompt_optimizer,
        mock_analysis_response,
        mock_match_result,
        mock_generation_result,
        temp_dir,
    ):
        """Test that optimization is skipped when disabled."""
        from avatarvoice_api.services.pipeline_orchestrator import PipelineOrchestrator

        # Setup mocks
        mock_voicematch_api.analyze_image_bytes.return_value = mock_analysis_response
        mock_voicematch_api.find_matches.return_value = [mock_match_result]
        sample_path = temp_dir / "sample.wav"
        sample_path.write_bytes(b"fake wav data")
        mock_voicematch_api.get_voice_sample.return_value = sample_path
        mock_vibevoice_client.generate.return_value = mock_generation_result

        orchestrator = PipelineOrchestrator(
            voicematch_api=mock_voicematch_api,
            vibevoice_client=mock_vibevoice_client,
            prompt_optimizer=mock_prompt_optimizer,
        )

        result = await orchestrator.generate(
            media_bytes=b"fake image",
            mime_type="image/jpeg",
            text="Original text",
            optimize_prompt=False,
        )

        # Verify optimizer was NOT called
        mock_prompt_optimizer.optimize.assert_not_called()

        # Verify original text was used
        assert result.optimized_text is None


class TestPipelineErrorHandling:
    """Tests for pipeline error handling."""

    @pytest.fixture
    def mock_voicematch_api(self):
        """Create a mock VoiceMatchAPI."""
        mock = MagicMock()
        mock.analyze_image_bytes = AsyncMock()
        mock.find_matches = MagicMock()
        mock.get_voice_sample = MagicMock()
        return mock

    @pytest.fixture
    def mock_vibevoice_client(self):
        """Create a mock VibeVoiceClient."""
        mock = MagicMock()
        mock.generate = AsyncMock()
        return mock

    @pytest.mark.asyncio
    async def test_analysis_error_handling(
        self,
        mock_voicematch_api,
        mock_vibevoice_client,
    ):
        """Test error handling when image analysis fails."""
        from avatarvoice_api.services.pipeline_orchestrator import PipelineOrchestrator

        mock_voicematch_api.analyze_image_bytes = AsyncMock(
            side_effect=Exception("Gemini API error")
        )

        orchestrator = PipelineOrchestrator(
            voicematch_api=mock_voicematch_api,
            vibevoice_client=mock_vibevoice_client,
        )

        with pytest.raises(ValueError) as exc_info:
            await orchestrator.generate(
                media_bytes=b"fake image",
                mime_type="image/jpeg",
                text="Hello",
            )

        assert "Image analysis failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_no_matches_error_handling(
        self,
        mock_voicematch_api,
        mock_vibevoice_client,
    ):
        """Test error handling when no voice matches are found."""
        from avatarvoice_api.services.pipeline_orchestrator import PipelineOrchestrator

        mock_analysis = MagicMock()
        mock_analysis.estimated_age = 35
        mock_analysis.gender = "male"
        mock_analysis.race = "caucasian"
        mock_analysis.emotion = "happy"
        mock_voicematch_api.analyze_image_bytes.return_value = mock_analysis
        mock_voicematch_api.find_matches.return_value = []  # No matches

        orchestrator = PipelineOrchestrator(
            voicematch_api=mock_voicematch_api,
            vibevoice_client=mock_vibevoice_client,
        )

        with pytest.raises(ValueError) as exc_info:
            await orchestrator.generate(
                media_bytes=b"fake image",
                mime_type="image/jpeg",
                text="Hello",
            )

        assert "No matching voice actors" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_tts_error_handling(
        self,
        mock_voicematch_api,
        mock_vibevoice_client,
        temp_dir,
    ):
        """Test error handling when TTS generation fails."""
        from avatarvoice_api.services.pipeline_orchestrator import PipelineOrchestrator

        mock_analysis = MagicMock()
        mock_analysis.estimated_age = 35
        mock_analysis.gender = "male"
        mock_analysis.race = "caucasian"
        mock_analysis.emotion = "happy"
        mock_voicematch_api.analyze_image_bytes.return_value = mock_analysis

        mock_match = MagicMock()
        mock_match.actor = MagicMock()
        mock_match.actor.id = "1050"
        mock_voicematch_api.find_matches.return_value = [mock_match]

        sample_path = temp_dir / "sample.wav"
        sample_path.write_bytes(b"fake wav data")
        mock_voicematch_api.get_voice_sample.return_value = sample_path

        mock_vibevoice_client.generate = AsyncMock(
            side_effect=Exception("TTS service unavailable")
        )

        orchestrator = PipelineOrchestrator(
            voicematch_api=mock_voicematch_api,
            vibevoice_client=mock_vibevoice_client,
        )

        with pytest.raises(ValueError) as exc_info:
            await orchestrator.generate(
                media_bytes=b"fake image",
                mime_type="image/jpeg",
                text="Hello",
            )

        assert "Speech generation failed" in str(exc_info.value)


# ============================================================================
# Additional E2E Edge Case Tests
# ============================================================================

class TestPipelineInvalidImageData:
    """E2E tests for invalid image data scenarios."""

    @pytest.fixture
    def mock_voicematch_api(self):
        """Create a mock VoiceMatchAPI."""
        mock = MagicMock()
        mock.analyze_image_bytes = AsyncMock()
        mock.find_matches = MagicMock()
        mock.get_voice_sample = MagicMock()
        return mock

    @pytest.fixture
    def mock_vibevoice_client(self):
        """Create a mock VibeVoiceClient."""
        mock = MagicMock()
        mock.generate = AsyncMock()
        return mock

    @pytest.fixture
    def app_with_analysis_error(
        self,
        mock_voicematch_api,
        mock_vibevoice_client,
    ):
        """Create app with analysis error configured."""
        from avatarvoice_api.main import create_app
        from avatarvoice_api.routes.pipeline import get_pipeline_orchestrator
        from avatarvoice_api.services.pipeline_orchestrator import PipelineOrchestrator
        from voicematch.exceptions import ImageAnalysisError

        mock_voicematch_api.analyze_image_bytes.side_effect = ImageAnalysisError(
            "Not a valid image"
        )

        app = create_app()
        mock_orchestrator = PipelineOrchestrator(
            voicematch_api=mock_voicematch_api,
            vibevoice_client=mock_vibevoice_client,
        )
        app.dependency_overrides[get_pipeline_orchestrator] = lambda: mock_orchestrator

        yield app
        app.dependency_overrides.clear()

    def test_invalid_image_returns_error(self, app_with_analysis_error, test_image_base64):
        """Test that invalid image data returns proper error."""
        with TestClient(app_with_analysis_error) as client:
            response = client.post(
                "/pipeline/generate",
                json={
                    "media_base64": test_image_base64,
                    "mime_type": "image/jpeg",
                    "text": "Hello",
                    "stream": False,
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "error" in data


class TestPipelineUnsupportedMimeTypes:
    """E2E tests for unsupported MIME types."""

    @pytest.fixture
    def mock_voicematch_api(self):
        """Create a mock VoiceMatchAPI."""
        mock = MagicMock()
        mock.analyze_image_bytes = AsyncMock()
        mock.find_matches = MagicMock()
        mock.get_voice_sample = MagicMock()
        return mock

    @pytest.fixture
    def mock_vibevoice_client(self):
        """Create a mock VibeVoiceClient."""
        mock = MagicMock()
        mock.generate = AsyncMock()
        return mock

    @pytest.fixture
    def app_with_format_error(
        self,
        mock_voicematch_api,
        mock_vibevoice_client,
    ):
        """Create app with unsupported format error."""
        from avatarvoice_api.main import create_app
        from avatarvoice_api.routes.pipeline import get_pipeline_orchestrator
        from avatarvoice_api.services.pipeline_orchestrator import PipelineOrchestrator
        from voicematch.exceptions import UnsupportedFormatError

        mock_voicematch_api.analyze_image_bytes.side_effect = UnsupportedFormatError(
            "Unsupported format"
        )

        app = create_app()
        mock_orchestrator = PipelineOrchestrator(
            voicematch_api=mock_voicematch_api,
            vibevoice_client=mock_vibevoice_client,
        )
        app.dependency_overrides[get_pipeline_orchestrator] = lambda: mock_orchestrator

        yield app
        app.dependency_overrides.clear()

    def test_unsupported_mime_returns_error(self, app_with_format_error):
        """Test that unsupported MIME type returns proper error."""
        with TestClient(app_with_format_error) as client:
            response = client.post(
                "/pipeline/generate",
                json={
                    "media_base64": "SGVsbG8=",  # Just "Hello" in base64
                    "mime_type": "video/avi",
                    "text": "Hello",
                    "stream": False,
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False


class TestPipelineTextValidation:
    """E2E tests for text validation."""

    @pytest.fixture
    def mock_voicematch_api(self):
        """Create a mock VoiceMatchAPI."""
        mock = MagicMock()
        mock.analyze_image_bytes = AsyncMock()
        mock.find_matches = MagicMock()
        mock.get_voice_sample = MagicMock()
        return mock

    @pytest.fixture
    def mock_vibevoice_client(self):
        """Create a mock VibeVoiceClient."""
        mock = MagicMock()
        mock.generate = AsyncMock()
        return mock

    @pytest.fixture
    def mock_analysis_response(self):
        """Create a mock analysis response."""
        mock = MagicMock()
        mock.estimated_age = 35
        mock.gender = "male"
        mock.race = "caucasian"
        mock.emotion = "neutral"
        return mock

    @pytest.fixture
    def mock_match_result(self):
        """Create a mock match result."""
        mock = MagicMock()
        mock.actor = MagicMock()
        mock.actor.id = "1050"
        return mock

    @pytest.fixture
    def app_with_text_validation_error(
        self,
        mock_voicematch_api,
        mock_vibevoice_client,
        mock_analysis_response,
        mock_match_result,
        temp_dir,
    ):
        """Create app that will fail on empty text."""
        from avatarvoice_api.main import create_app
        from avatarvoice_api.routes.pipeline import get_pipeline_orchestrator
        from avatarvoice_api.services.pipeline_orchestrator import PipelineOrchestrator
        from vibevoice_client.exceptions import ValidationError as VVValidationError

        mock_voicematch_api.analyze_image_bytes.return_value = mock_analysis_response
        mock_voicematch_api.find_matches.return_value = [mock_match_result]
        sample_path = temp_dir / "sample.wav"
        sample_path.write_bytes(b"fake wav")
        mock_voicematch_api.get_voice_sample.return_value = sample_path
        mock_vibevoice_client.generate.side_effect = VVValidationError("Text cannot be empty")

        app = create_app()
        mock_orchestrator = PipelineOrchestrator(
            voicematch_api=mock_voicematch_api,
            vibevoice_client=mock_vibevoice_client,
        )
        app.dependency_overrides[get_pipeline_orchestrator] = lambda: mock_orchestrator

        yield app
        app.dependency_overrides.clear()

    def test_empty_text_validation_error(self, app_with_text_validation_error, test_image_base64):
        """Test that empty text is rejected by Pydantic validation."""
        with TestClient(app_with_text_validation_error) as client:
            response = client.post(
                "/pipeline/generate",
                json={
                    "media_base64": test_image_base64,
                    "mime_type": "image/jpeg",
                    "text": "",  # Empty text
                    "stream": False,
                },
            )

        # Pydantic validation should reject this
        assert response.status_code == 422

    def test_text_too_long_validation_error(self, app_with_text_validation_error, test_image_base64):
        """Test that text over 5000 chars is rejected."""
        with TestClient(app_with_text_validation_error) as client:
            response = client.post(
                "/pipeline/generate",
                json={
                    "media_base64": test_image_base64,
                    "mime_type": "image/jpeg",
                    "text": "A" * 5001,  # Too long
                    "stream": False,
                },
            )

        # Pydantic validation should reject this
        assert response.status_code == 422


class TestPipelineVibeVoiceFailures:
    """E2E tests for VibeVoice endpoint failures."""

    @pytest.fixture
    def mock_voicematch_api(self):
        """Create a mock VoiceMatchAPI."""
        mock = MagicMock()
        mock.analyze_image_bytes = AsyncMock()
        mock.find_matches = MagicMock()
        mock.get_voice_sample = MagicMock()
        return mock

    @pytest.fixture
    def mock_vibevoice_client(self):
        """Create a mock VibeVoiceClient."""
        mock = MagicMock()
        mock.generate = AsyncMock()
        return mock

    @pytest.fixture
    def mock_analysis_response(self):
        """Create a mock analysis response."""
        mock = MagicMock()
        mock.estimated_age = 35
        mock.gender = "male"
        mock.race = "caucasian"
        mock.emotion = "neutral"
        return mock

    @pytest.fixture
    def mock_match_result(self):
        """Create a mock match result."""
        mock = MagicMock()
        mock.actor = MagicMock()
        mock.actor.id = "1050"
        return mock

    @pytest.fixture
    def app_with_vibevoice_connection_error(
        self,
        mock_voicematch_api,
        mock_vibevoice_client,
        mock_analysis_response,
        mock_match_result,
        temp_dir,
    ):
        """Create app with VibeVoice connection error."""
        from avatarvoice_api.main import create_app
        from avatarvoice_api.routes.pipeline import get_pipeline_orchestrator
        from avatarvoice_api.services.pipeline_orchestrator import PipelineOrchestrator
        from vibevoice_client.exceptions import ConnectionError as VVConnectionError

        mock_voicematch_api.analyze_image_bytes.return_value = mock_analysis_response
        mock_voicematch_api.find_matches.return_value = [mock_match_result]
        sample_path = temp_dir / "sample.wav"
        sample_path.write_bytes(b"fake wav")
        mock_voicematch_api.get_voice_sample.return_value = sample_path
        mock_vibevoice_client.generate.side_effect = VVConnectionError("Connection refused")

        app = create_app()
        mock_orchestrator = PipelineOrchestrator(
            voicematch_api=mock_voicematch_api,
            vibevoice_client=mock_vibevoice_client,
        )
        app.dependency_overrides[get_pipeline_orchestrator] = lambda: mock_orchestrator

        yield app
        app.dependency_overrides.clear()

    def test_vibevoice_connection_error_handled(
        self, app_with_vibevoice_connection_error, test_image_base64
    ):
        """Test that VibeVoice connection error returns proper error response."""
        with TestClient(app_with_vibevoice_connection_error) as client:
            response = client.post(
                "/pipeline/generate",
                json={
                    "media_base64": test_image_base64,
                    "mime_type": "image/jpeg",
                    "text": "Hello",
                    "stream": False,
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "error" in data

    @pytest.fixture
    def app_with_vibevoice_timeout(
        self,
        mock_voicematch_api,
        mock_vibevoice_client,
        mock_analysis_response,
        mock_match_result,
        temp_dir,
    ):
        """Create app with VibeVoice timeout error."""
        from avatarvoice_api.main import create_app
        from avatarvoice_api.routes.pipeline import get_pipeline_orchestrator
        from avatarvoice_api.services.pipeline_orchestrator import PipelineOrchestrator
        from vibevoice_client.exceptions import TimeoutError as VVTimeoutError

        mock_voicematch_api.analyze_image_bytes.return_value = mock_analysis_response
        mock_voicematch_api.find_matches.return_value = [mock_match_result]
        sample_path = temp_dir / "sample.wav"
        sample_path.write_bytes(b"fake wav")
        mock_voicematch_api.get_voice_sample.return_value = sample_path
        mock_vibevoice_client.generate.side_effect = VVTimeoutError("Request timed out")

        app = create_app()
        mock_orchestrator = PipelineOrchestrator(
            voicematch_api=mock_voicematch_api,
            vibevoice_client=mock_vibevoice_client,
        )
        app.dependency_overrides[get_pipeline_orchestrator] = lambda: mock_orchestrator

        yield app
        app.dependency_overrides.clear()

    def test_vibevoice_timeout_handled(
        self, app_with_vibevoice_timeout, test_image_base64
    ):
        """Test that VibeVoice timeout returns proper error response."""
        with TestClient(app_with_vibevoice_timeout) as client:
            response = client.post(
                "/pipeline/generate",
                json={
                    "media_base64": test_image_base64,
                    "mime_type": "image/jpeg",
                    "text": "Hello",
                    "stream": False,
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False


class TestPipelineGeminiAPIFailures:
    """E2E tests for Gemini API failures."""

    @pytest.fixture
    def mock_voicematch_api(self):
        """Create a mock VoiceMatchAPI."""
        mock = MagicMock()
        mock.analyze_image_bytes = AsyncMock()
        return mock

    @pytest.fixture
    def mock_vibevoice_client(self):
        """Create a mock VibeVoiceClient."""
        mock = MagicMock()
        mock.generate = AsyncMock()
        return mock

    @pytest.fixture
    def app_with_gemini_rate_limit(
        self,
        mock_voicematch_api,
        mock_vibevoice_client,
    ):
        """Create app with Gemini rate limit error."""
        from avatarvoice_api.main import create_app
        from avatarvoice_api.routes.pipeline import get_pipeline_orchestrator
        from avatarvoice_api.services.pipeline_orchestrator import PipelineOrchestrator
        from voicematch.exceptions import ImageAnalysisError

        mock_voicematch_api.analyze_image_bytes.side_effect = ImageAnalysisError(
            "Rate limit exceeded"
        )

        app = create_app()
        mock_orchestrator = PipelineOrchestrator(
            voicematch_api=mock_voicematch_api,
            vibevoice_client=mock_vibevoice_client,
        )
        app.dependency_overrides[get_pipeline_orchestrator] = lambda: mock_orchestrator

        yield app
        app.dependency_overrides.clear()

    def test_gemini_rate_limit_handled(
        self, app_with_gemini_rate_limit, test_image_base64
    ):
        """Test that Gemini rate limit returns proper error response."""
        with TestClient(app_with_gemini_rate_limit) as client:
            response = client.post(
                "/pipeline/generate",
                json={
                    "media_base64": test_image_base64,
                    "mime_type": "image/jpeg",
                    "text": "Hello",
                    "stream": False,
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "rate limit" in data.get("error", "").lower()


class TestPipelineCfgScaleParameter:
    """E2E tests for cfg_scale parameter."""

    @pytest.fixture
    def mock_voicematch_api(self):
        """Create a mock VoiceMatchAPI."""
        mock = MagicMock()
        mock.analyze_image_bytes = AsyncMock()
        mock.find_matches = MagicMock()
        mock.get_voice_sample = MagicMock()
        return mock

    @pytest.fixture
    def mock_vibevoice_client(self):
        """Create a mock VibeVoiceClient."""
        mock = MagicMock()
        mock.generate = AsyncMock()
        return mock

    @pytest.fixture
    def mock_analysis_response(self):
        """Create a mock analysis response."""
        mock = MagicMock()
        mock.estimated_age = 35
        mock.gender = "male"
        mock.race = "caucasian"
        mock.emotion = "neutral"
        return mock

    @pytest.fixture
    def mock_match_result(self):
        """Create a mock match result."""
        mock = MagicMock()
        mock.actor = MagicMock()
        mock.actor.id = "1050"
        return mock

    @pytest.fixture
    def mock_generation_result(self):
        """Create a mock generation result."""
        mock = MagicMock()
        mock.audio_bytes = b"fake audio"
        mock.duration_seconds = 2.5
        return mock

    @pytest.fixture
    def app_with_cfg_scale_tracking(
        self,
        mock_voicematch_api,
        mock_vibevoice_client,
        mock_analysis_response,
        mock_match_result,
        mock_generation_result,
        temp_dir,
    ):
        """Create app that tracks cfg_scale parameter."""
        from avatarvoice_api.main import create_app
        from avatarvoice_api.routes.pipeline import get_pipeline_orchestrator
        from avatarvoice_api.services.pipeline_orchestrator import PipelineOrchestrator

        mock_voicematch_api.analyze_image_bytes.return_value = mock_analysis_response
        mock_voicematch_api.find_matches.return_value = [mock_match_result]
        sample_path = temp_dir / "sample.wav"
        sample_path.write_bytes(b"fake wav")
        mock_voicematch_api.get_voice_sample.return_value = sample_path
        mock_vibevoice_client.generate.return_value = mock_generation_result

        app = create_app()
        mock_orchestrator = PipelineOrchestrator(
            voicematch_api=mock_voicematch_api,
            vibevoice_client=mock_vibevoice_client,
        )
        app.dependency_overrides[get_pipeline_orchestrator] = lambda: mock_orchestrator

        yield app, mock_vibevoice_client
        app.dependency_overrides.clear()

    @pytest.mark.parametrize("cfg_scale", [0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    def test_valid_cfg_scale_values(
        self, app_with_cfg_scale_tracking, test_image_base64, cfg_scale
    ):
        """Test various valid cfg_scale values."""
        app, mock_vibevoice = app_with_cfg_scale_tracking
        with TestClient(app) as client:
            response = client.post(
                "/pipeline/generate",
                json={
                    "media_base64": test_image_base64,
                    "mime_type": "image/jpeg",
                    "text": "Hello",
                    "cfg_scale": cfg_scale,
                    "stream": False,
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_cfg_scale_too_low_rejected(
        self, app_with_cfg_scale_tracking, test_image_base64
    ):
        """Test that cfg_scale below 0.1 is rejected."""
        app, _ = app_with_cfg_scale_tracking
        with TestClient(app) as client:
            response = client.post(
                "/pipeline/generate",
                json={
                    "media_base64": test_image_base64,
                    "mime_type": "image/jpeg",
                    "text": "Hello",
                    "cfg_scale": 0.01,  # Below minimum
                    "stream": False,
                },
            )

        assert response.status_code == 422

    def test_cfg_scale_too_high_rejected(
        self, app_with_cfg_scale_tracking, test_image_base64
    ):
        """Test that cfg_scale above 10.0 is rejected."""
        app, _ = app_with_cfg_scale_tracking
        with TestClient(app) as client:
            response = client.post(
                "/pipeline/generate",
                json={
                    "media_base64": test_image_base64,
                    "mime_type": "image/jpeg",
                    "text": "Hello",
                    "cfg_scale": 15.0,  # Above maximum
                    "stream": False,
                },
            )

        assert response.status_code == 422


class TestPipelineOptimizePrompt:
    """E2E tests for optimize_prompt parameter."""

    @pytest.fixture
    def mock_voicematch_api(self):
        """Create a mock VoiceMatchAPI."""
        mock = MagicMock()
        mock.analyze_image_bytes = AsyncMock()
        mock.find_matches = MagicMock()
        mock.get_voice_sample = MagicMock()
        return mock

    @pytest.fixture
    def mock_vibevoice_client(self):
        """Create a mock VibeVoiceClient."""
        mock = MagicMock()
        mock.generate = AsyncMock()
        return mock

    @pytest.fixture
    def mock_prompt_optimizer(self):
        """Create a mock PromptOptimizer."""
        mock = MagicMock()
        mock.optimize = AsyncMock(return_value="Optimized text!")
        return mock

    @pytest.fixture
    def mock_analysis_response(self):
        """Create a mock analysis response."""
        mock = MagicMock()
        mock.estimated_age = 35
        mock.gender = "male"
        mock.race = "caucasian"
        mock.emotion = "neutral"
        return mock

    @pytest.fixture
    def mock_match_result(self):
        """Create a mock match result."""
        mock = MagicMock()
        mock.actor = MagicMock()
        mock.actor.id = "1050"
        return mock

    @pytest.fixture
    def mock_generation_result(self):
        """Create a mock generation result."""
        mock = MagicMock()
        mock.audio_bytes = b"fake audio"
        mock.duration_seconds = 2.5
        return mock

    @pytest.fixture
    def app_with_optimizer(
        self,
        mock_voicematch_api,
        mock_vibevoice_client,
        mock_prompt_optimizer,
        mock_analysis_response,
        mock_match_result,
        mock_generation_result,
        temp_dir,
    ):
        """Create app with optimizer enabled."""
        from avatarvoice_api.main import create_app
        from avatarvoice_api.routes.pipeline import get_pipeline_orchestrator
        from avatarvoice_api.services.pipeline_orchestrator import PipelineOrchestrator

        mock_voicematch_api.analyze_image_bytes.return_value = mock_analysis_response
        mock_voicematch_api.find_matches.return_value = [mock_match_result]
        sample_path = temp_dir / "sample.wav"
        sample_path.write_bytes(b"fake wav")
        mock_voicematch_api.get_voice_sample.return_value = sample_path
        mock_vibevoice_client.generate.return_value = mock_generation_result

        app = create_app()
        mock_orchestrator = PipelineOrchestrator(
            voicematch_api=mock_voicematch_api,
            vibevoice_client=mock_vibevoice_client,
            prompt_optimizer=mock_prompt_optimizer,
        )
        app.dependency_overrides[get_pipeline_orchestrator] = lambda: mock_orchestrator

        yield app, mock_prompt_optimizer
        app.dependency_overrides.clear()

    def test_optimize_prompt_true_calls_optimizer(
        self, app_with_optimizer, test_image_base64
    ):
        """Test that optimize_prompt=True calls the optimizer."""
        app, mock_optimizer = app_with_optimizer
        with TestClient(app) as client:
            response = client.post(
                "/pipeline/generate",
                json={
                    "media_base64": test_image_base64,
                    "mime_type": "image/jpeg",
                    "text": "Hello",
                    "optimize_prompt": True,
                    "stream": False,
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        mock_optimizer.optimize.assert_called_once()

    def test_optimize_prompt_false_skips_optimizer(
        self, app_with_optimizer, test_image_base64
    ):
        """Test that optimize_prompt=False skips the optimizer."""
        app, mock_optimizer = app_with_optimizer
        with TestClient(app) as client:
            response = client.post(
                "/pipeline/generate",
                json={
                    "media_base64": test_image_base64,
                    "mime_type": "image/jpeg",
                    "text": "Hello",
                    "optimize_prompt": False,
                    "stream": False,
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        mock_optimizer.optimize.assert_not_called()


class TestPipelineStreamingBehavior:
    """E2E tests for streaming vs non-streaming behavior."""

    @pytest.fixture
    def mock_voicematch_api(self):
        """Create a mock VoiceMatchAPI."""
        mock = MagicMock()
        mock.analyze_image_bytes = AsyncMock()
        mock.find_matches = MagicMock()
        mock.get_voice_sample = MagicMock()
        return mock

    @pytest.fixture
    def mock_vibevoice_client(self):
        """Create a mock VibeVoiceClient."""
        mock = MagicMock()
        mock.generate = AsyncMock()
        return mock

    @pytest.fixture
    def mock_analysis_response(self):
        """Create a mock analysis response."""
        mock = MagicMock()
        mock.estimated_age = 35
        mock.gender = "male"
        mock.race = "caucasian"
        mock.emotion = "neutral"
        return mock

    @pytest.fixture
    def mock_match_result(self):
        """Create a mock match result."""
        mock = MagicMock()
        mock.actor = MagicMock()
        mock.actor.id = "1050"
        return mock

    @pytest.fixture
    def mock_generation_result(self):
        """Create a mock generation result."""
        mock = MagicMock()
        mock.audio_bytes = b"fake audio bytes"
        mock.duration_seconds = 2.5
        return mock

    @pytest.fixture
    def working_app(
        self,
        mock_voicematch_api,
        mock_vibevoice_client,
        mock_analysis_response,
        mock_match_result,
        mock_generation_result,
        temp_dir,
    ):
        """Create a working app."""
        from avatarvoice_api.main import create_app
        from avatarvoice_api.routes.pipeline import get_pipeline_orchestrator
        from avatarvoice_api.services.pipeline_orchestrator import PipelineOrchestrator

        mock_voicematch_api.analyze_image_bytes.return_value = mock_analysis_response
        mock_voicematch_api.find_matches.return_value = [mock_match_result]
        sample_path = temp_dir / "sample.wav"
        sample_path.write_bytes(b"fake wav")
        mock_voicematch_api.get_voice_sample.return_value = sample_path
        mock_vibevoice_client.generate.return_value = mock_generation_result

        app = create_app()
        mock_orchestrator = PipelineOrchestrator(
            voicematch_api=mock_voicematch_api,
            vibevoice_client=mock_vibevoice_client,
        )
        app.dependency_overrides[get_pipeline_orchestrator] = lambda: mock_orchestrator

        yield app
        app.dependency_overrides.clear()

    def test_non_streaming_returns_json(self, working_app, test_image_base64):
        """Test that non-streaming returns JSON response."""
        with TestClient(working_app) as client:
            response = client.post(
                "/pipeline/generate",
                json={
                    "media_base64": test_image_base64,
                    "mime_type": "image/jpeg",
                    "text": "Hello",
                    "stream": False,
                },
            )

        assert response.status_code == 200
        assert "application/json" in response.headers.get("content-type", "")
        data = response.json()
        assert "success" in data
        assert "audio_base64" in data

    def test_streaming_returns_sse(self, working_app, test_image_base64):
        """Test that streaming returns SSE response."""
        with TestClient(working_app) as client:
            response = client.post(
                "/pipeline/generate",
                json={
                    "media_base64": test_image_base64,
                    "mime_type": "image/jpeg",
                    "text": "Hello",
                    "stream": True,
                },
            )

        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")

    def test_default_is_streaming(self, working_app, test_image_base64):
        """Test that default behavior is streaming."""
        with TestClient(working_app) as client:
            response = client.post(
                "/pipeline/generate",
                json={
                    "media_base64": test_image_base64,
                    "mime_type": "image/jpeg",
                    "text": "Hello",
                    # Not specifying stream - should default to True
                },
            )

        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")
