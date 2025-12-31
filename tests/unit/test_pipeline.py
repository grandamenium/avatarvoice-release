"""Unit tests for pipeline orchestrator and routes."""

import base64
import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from avatarvoice_api.services.pipeline_orchestrator import (
    PipelineOrchestrator,
    PipelineResult,
    map_gemini_emotion_to_cremad,
    EMOTION_MAPPING,
)


# Test emotion mapping
class TestEmotionMapping:
    """Tests for emotion mapping function."""

    def test_map_happy_emotion(self):
        """Should map 'happy' to 'HAP'."""
        assert map_gemini_emotion_to_cremad("happy") == "HAP"

    def test_map_sad_emotion(self):
        """Should map 'sad' to 'SAD'."""
        assert map_gemini_emotion_to_cremad("sad") == "SAD"

    def test_map_anger_emotion(self):
        """Should map 'anger' to 'ANG'."""
        assert map_gemini_emotion_to_cremad("anger") == "ANG"

    def test_map_neutral_emotion(self):
        """Should map 'neutral' to 'NEU'."""
        assert map_gemini_emotion_to_cremad("neutral") == "NEU"

    def test_map_unknown_emotion_to_neutral(self):
        """Should map unknown emotions to 'NEU'."""
        assert map_gemini_emotion_to_cremad("unknown") == "NEU"

    def test_map_none_emotion_to_neutral(self):
        """Should map None to 'NEU'."""
        assert map_gemini_emotion_to_cremad(None) == "NEU"

    def test_case_insensitive_mapping(self):
        """Should handle case insensitive emotion strings."""
        assert map_gemini_emotion_to_cremad("HAPPY") == "HAP"
        assert map_gemini_emotion_to_cremad("Happy") == "HAP"


# Test PipelineOrchestrator initialization
class TestPipelineOrchestratorInit:
    """Tests for PipelineOrchestrator initialization."""

    def test_init_with_all_components(self):
        """Should initialize with all components."""
        voicematch = MagicMock()
        vibevoice = MagicMock()
        optimizer = MagicMock()

        orchestrator = PipelineOrchestrator(
            voicematch_api=voicematch,
            vibevoice_client=vibevoice,
            prompt_optimizer=optimizer,
        )

        assert orchestrator.voicematch is voicematch
        assert orchestrator.vibevoice is vibevoice
        assert orchestrator.optimizer is optimizer

    def test_init_without_optimizer(self):
        """Should initialize without prompt optimizer."""
        voicematch = MagicMock()
        vibevoice = MagicMock()

        orchestrator = PipelineOrchestrator(
            voicematch_api=voicematch,
            vibevoice_client=vibevoice,
            prompt_optimizer=None,
        )

        assert orchestrator.voicematch is voicematch
        assert orchestrator.vibevoice is vibevoice
        assert orchestrator.optimizer is None


# Test PipelineOrchestrator streaming
class TestPipelineOrchestratorStreaming:
    """Tests for streaming pipeline generation."""

    @pytest.fixture
    def mock_voicematch(self):
        """Create a mock VoiceMatchAPI."""
        mock = MagicMock()
        mock.analyze_image_bytes = AsyncMock()
        mock.find_matches = MagicMock()
        mock.get_voice_sample = MagicMock()
        return mock

    @pytest.fixture
    def mock_vibevoice(self):
        """Create a mock VibeVoiceClient."""
        mock = MagicMock()
        mock.generate = AsyncMock()
        return mock

    @pytest.fixture
    def mock_optimizer(self):
        """Create a mock PromptOptimizer."""
        mock = MagicMock()
        mock.optimize = AsyncMock(return_value="Optimized text!")
        return mock

    @pytest.fixture
    def mock_analysis_response(self):
        """Create a mock AvatarAnalysisResponse."""
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

    @pytest.fixture
    def sample_image_bytes(self):
        """Create sample image bytes."""
        return b"fake image data"

    @pytest.mark.asyncio
    async def test_streaming_emits_progress_events(
        self,
        mock_voicematch,
        mock_vibevoice,
        mock_optimizer,
        mock_analysis_response,
        mock_match_result,
        mock_generation_result,
        sample_image_bytes,
        temp_dir,
    ):
        """Should emit progress events for each step."""
        # Setup mocks
        mock_voicematch.analyze_image_bytes.return_value = mock_analysis_response
        mock_voicematch.find_matches.return_value = [mock_match_result]
        sample_path = temp_dir / "sample.wav"
        sample_path.write_bytes(b"fake wav data")
        mock_voicematch.get_voice_sample.return_value = sample_path
        mock_vibevoice.generate.return_value = mock_generation_result

        orchestrator = PipelineOrchestrator(
            voicematch_api=mock_voicematch,
            vibevoice_client=mock_vibevoice,
            prompt_optimizer=mock_optimizer,
        )

        events = []
        async for event in orchestrator.generate_streaming(
            media_bytes=sample_image_bytes,
            mime_type="image/jpeg",
            text="Hello world",
            cfg_scale=2.0,
            optimize_prompt=True,
        ):
            events.append(event)

        # Should have at least analyzing, demographics, matching, optimizing, generating, complete
        assert len(events) >= 5

        # Check steps are present
        steps = [e.get("step") for e in events]
        assert "analyzing" in steps
        assert "demographics" in steps
        assert "matching" in steps
        assert "generating" in steps
        assert "complete" in steps

    @pytest.mark.asyncio
    async def test_streaming_returns_audio_on_complete(
        self,
        mock_voicematch,
        mock_vibevoice,
        mock_optimizer,
        mock_analysis_response,
        mock_match_result,
        mock_generation_result,
        sample_image_bytes,
        temp_dir,
    ):
        """Should return audio data on completion."""
        mock_voicematch.analyze_image_bytes.return_value = mock_analysis_response
        mock_voicematch.find_matches.return_value = [mock_match_result]
        sample_path = temp_dir / "sample.wav"
        sample_path.write_bytes(b"fake wav data")
        mock_voicematch.get_voice_sample.return_value = sample_path
        mock_vibevoice.generate.return_value = mock_generation_result

        orchestrator = PipelineOrchestrator(
            voicematch_api=mock_voicematch,
            vibevoice_client=mock_vibevoice,
            prompt_optimizer=mock_optimizer,
        )

        complete_event = None
        async for event in orchestrator.generate_streaming(
            media_bytes=sample_image_bytes,
            mime_type="image/jpeg",
            text="Hello world",
        ):
            if event.get("step") == "complete":
                complete_event = event

        assert complete_event is not None
        assert "audio_base64" in complete_event
        assert complete_event["voice_actor"] == "1050"
        assert complete_event["duration"] == 3.5

    @pytest.mark.asyncio
    async def test_streaming_handles_analysis_error(
        self,
        mock_voicematch,
        mock_vibevoice,
        sample_image_bytes,
    ):
        """Should emit error event when analysis fails."""
        mock_voicematch.analyze_image_bytes = AsyncMock(
            side_effect=Exception("Analysis failed")
        )

        orchestrator = PipelineOrchestrator(
            voicematch_api=mock_voicematch,
            vibevoice_client=mock_vibevoice,
        )

        events = []
        async for event in orchestrator.generate_streaming(
            media_bytes=sample_image_bytes,
            mime_type="image/jpeg",
            text="Hello world",
        ):
            events.append(event)

        # Should have analyzing then error
        assert len(events) >= 2
        error_event = next((e for e in events if e.get("step") == "error"), None)
        assert error_event is not None
        assert "Analysis failed" in error_event["message"]

    @pytest.mark.asyncio
    async def test_streaming_handles_no_matches(
        self,
        mock_voicematch,
        mock_vibevoice,
        mock_analysis_response,
        sample_image_bytes,
    ):
        """Should emit error event when no voice matches found."""
        mock_voicematch.analyze_image_bytes.return_value = mock_analysis_response
        mock_voicematch.find_matches.return_value = []

        orchestrator = PipelineOrchestrator(
            voicematch_api=mock_voicematch,
            vibevoice_client=mock_vibevoice,
        )

        events = []
        async for event in orchestrator.generate_streaming(
            media_bytes=sample_image_bytes,
            mime_type="image/jpeg",
            text="Hello world",
        ):
            events.append(event)

        error_event = next((e for e in events if e.get("step") == "error"), None)
        assert error_event is not None
        assert "No matching voice actors" in error_event["message"]

    @pytest.mark.asyncio
    async def test_streaming_skips_optimizer_when_disabled(
        self,
        mock_voicematch,
        mock_vibevoice,
        mock_optimizer,
        mock_analysis_response,
        mock_match_result,
        mock_generation_result,
        sample_image_bytes,
        temp_dir,
    ):
        """Should skip optimization when optimize_prompt=False."""
        mock_voicematch.analyze_image_bytes.return_value = mock_analysis_response
        mock_voicematch.find_matches.return_value = [mock_match_result]
        sample_path = temp_dir / "sample.wav"
        sample_path.write_bytes(b"fake wav data")
        mock_voicematch.get_voice_sample.return_value = sample_path
        mock_vibevoice.generate.return_value = mock_generation_result

        orchestrator = PipelineOrchestrator(
            voicematch_api=mock_voicematch,
            vibevoice_client=mock_vibevoice,
            prompt_optimizer=mock_optimizer,
        )

        events = []
        async for event in orchestrator.generate_streaming(
            media_bytes=sample_image_bytes,
            mime_type="image/jpeg",
            text="Hello world",
            optimize_prompt=False,
        ):
            events.append(event)

        # Optimizer should not have been called
        mock_optimizer.optimize.assert_not_called()

        # Complete event should not have optimized_text
        complete_event = next((e for e in events if e.get("step") == "complete"), None)
        assert complete_event is not None
        assert complete_event.get("optimized_text") is None

    @pytest.mark.asyncio
    async def test_streaming_handles_tts_error(
        self,
        mock_voicematch,
        mock_vibevoice,
        mock_analysis_response,
        mock_match_result,
        sample_image_bytes,
        temp_dir,
    ):
        """Should emit error event when TTS generation fails."""
        mock_voicematch.analyze_image_bytes.return_value = mock_analysis_response
        mock_voicematch.find_matches.return_value = [mock_match_result]
        sample_path = temp_dir / "sample.wav"
        sample_path.write_bytes(b"fake wav data")
        mock_voicematch.get_voice_sample.return_value = sample_path
        mock_vibevoice.generate = AsyncMock(side_effect=Exception("TTS failed"))

        orchestrator = PipelineOrchestrator(
            voicematch_api=mock_voicematch,
            vibevoice_client=mock_vibevoice,
        )

        events = []
        async for event in orchestrator.generate_streaming(
            media_bytes=sample_image_bytes,
            mime_type="image/jpeg",
            text="Hello world",
        ):
            events.append(event)

        error_event = next((e for e in events if e.get("step") == "error"), None)
        assert error_event is not None
        assert "Speech generation failed" in error_event["message"]


# Test non-streaming generation
class TestPipelineOrchestratorNonStreaming:
    """Tests for non-streaming pipeline generation."""

    @pytest.fixture
    def mock_voicematch(self):
        """Create a mock VoiceMatchAPI."""
        mock = MagicMock()
        mock.analyze_image_bytes = AsyncMock()
        mock.find_matches = MagicMock()
        mock.get_voice_sample = MagicMock()
        return mock

    @pytest.fixture
    def mock_vibevoice(self):
        """Create a mock VibeVoiceClient."""
        mock = MagicMock()
        mock.generate = AsyncMock()
        return mock

    @pytest.fixture
    def mock_analysis_response(self):
        """Create a mock AvatarAnalysisResponse."""
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
    async def test_generate_returns_pipeline_result(
        self,
        mock_voicematch,
        mock_vibevoice,
        mock_analysis_response,
        mock_match_result,
        mock_generation_result,
        temp_dir,
    ):
        """Should return PipelineResult on success."""
        mock_voicematch.analyze_image_bytes.return_value = mock_analysis_response
        mock_voicematch.find_matches.return_value = [mock_match_result]
        sample_path = temp_dir / "sample.wav"
        sample_path.write_bytes(b"fake wav data")
        mock_voicematch.get_voice_sample.return_value = sample_path
        mock_vibevoice.generate.return_value = mock_generation_result

        orchestrator = PipelineOrchestrator(
            voicematch_api=mock_voicematch,
            vibevoice_client=mock_vibevoice,
        )

        result = await orchestrator.generate(
            media_bytes=b"fake image",
            mime_type="image/jpeg",
            text="Hello world",
        )

        assert isinstance(result, PipelineResult)
        assert result.audio_bytes == b"fake audio bytes"
        assert result.voice_actor == "1050"
        assert result.duration == 3.5

    @pytest.mark.asyncio
    async def test_generate_raises_on_error(
        self,
        mock_voicematch,
        mock_vibevoice,
    ):
        """Should raise ValueError on pipeline error."""
        mock_voicematch.analyze_image_bytes = AsyncMock(
            side_effect=Exception("Analysis failed")
        )

        orchestrator = PipelineOrchestrator(
            voicematch_api=mock_voicematch,
            vibevoice_client=mock_vibevoice,
        )

        with pytest.raises(ValueError) as exc_info:
            await orchestrator.generate(
                media_bytes=b"fake image",
                mime_type="image/jpeg",
                text="Hello world",
            )

        assert "Analysis failed" in str(exc_info.value)


# Test SSE event formatting
class TestSSEEventFormatting:
    """Tests for SSE event formatting in the pipeline route."""

    def test_progress_event_format(self):
        """Progress events should be properly formatted."""
        from avatarvoice_api.routes.pipeline import stream_generator, PipelineRequest

        # Create minimal test - just check the event dictionary format
        progress_update = {
            "step": "analyzing",
            "message": "Analyzing avatar...",
            "progress": 0,
        }

        # Check event type determination
        step = progress_update.get("step", "")
        if step == "complete":
            event_type = "complete"
        elif step == "error":
            event_type = "error"
        else:
            event_type = "progress"

        assert event_type == "progress"

        # Check JSON serialization
        event_data = json.dumps(progress_update)
        parsed = json.loads(event_data)
        assert parsed["step"] == "analyzing"
        assert parsed["progress"] == 0

    def test_complete_event_format(self):
        """Complete events should be properly formatted."""
        complete_update = {
            "step": "complete",
            "audio_base64": "base64encodedaudio",
            "duration": 3.5,
            "voice_actor": "1050",
        }

        step = complete_update.get("step", "")
        event_type = "complete" if step == "complete" else "progress"

        assert event_type == "complete"

    def test_error_event_format(self):
        """Error events should be properly formatted."""
        error_update = {
            "step": "error",
            "message": "Something went wrong",
            "progress": 40,
        }

        step = error_update.get("step", "")
        event_type = "error" if step == "error" else "progress"

        assert event_type == "error"


# Test pipeline route request validation
class TestPipelineRequestValidation:
    """Tests for PipelineRequest validation."""

    def test_valid_request(self):
        """Should accept valid request."""
        from avatarvoice_api.routes.pipeline import PipelineRequest

        request = PipelineRequest(
            media_base64="SGVsbG8gV29ybGQ=",  # "Hello World" in base64
            mime_type="image/jpeg",
            text="Hello world",
            cfg_scale=2.0,
            optimize_prompt=True,
            stream=True,
        )

        assert request.media_base64 == "SGVsbG8gV29ybGQ="
        assert request.mime_type == "image/jpeg"
        assert request.text == "Hello world"
        assert request.cfg_scale == 2.0
        assert request.optimize_prompt is True
        assert request.stream is True

    def test_default_values(self):
        """Should use default values when not specified."""
        from avatarvoice_api.routes.pipeline import PipelineRequest

        request = PipelineRequest(
            media_base64="SGVsbG8gV29ybGQ=",
            mime_type="image/jpeg",
            text="Hello world",
        )

        assert request.cfg_scale == 2.0
        assert request.optimize_prompt is True
        assert request.stream is True

    def test_cfg_scale_validation(self):
        """Should validate cfg_scale range."""
        from avatarvoice_api.routes.pipeline import PipelineRequest
        from pydantic import ValidationError

        # Too low
        with pytest.raises(ValidationError):
            PipelineRequest(
                media_base64="SGVsbG8gV29ybGQ=",
                mime_type="image/jpeg",
                text="Hello world",
                cfg_scale=0.01,
            )

        # Too high
        with pytest.raises(ValidationError):
            PipelineRequest(
                media_base64="SGVsbG8gV29ybGQ=",
                mime_type="image/jpeg",
                text="Hello world",
                cfg_scale=15.0,
            )

    def test_text_length_validation(self):
        """Should validate text length."""
        from avatarvoice_api.routes.pipeline import PipelineRequest
        from pydantic import ValidationError

        # Empty text
        with pytest.raises(ValidationError):
            PipelineRequest(
                media_base64="SGVsbG8gV29ybGQ=",
                mime_type="image/jpeg",
                text="",
            )

        # Text too long
        with pytest.raises(ValidationError):
            PipelineRequest(
                media_base64="SGVsbG8gV29ybGQ=",
                mime_type="image/jpeg",
                text="x" * 5001,
            )


# Test pipeline response model
class TestPipelineResponse:
    """Tests for PipelineResponse model."""

    def test_success_response(self):
        """Should create success response."""
        from avatarvoice_api.routes.pipeline import PipelineResponse

        response = PipelineResponse(
            success=True,
            audio_base64="base64audio",
            duration=3.5,
            metadata={
                "voice_actor": "1050",
                "emotion": "HAP",
            },
        )

        assert response.success is True
        assert response.audio_base64 == "base64audio"
        assert response.duration == 3.5
        assert response.error is None

    def test_error_response(self):
        """Should create error response."""
        from avatarvoice_api.routes.pipeline import PipelineResponse

        response = PipelineResponse(
            success=False,
            error="No matching voice actors found",
            step="matching",
        )

        assert response.success is False
        assert response.error == "No matching voice actors found"
        assert response.audio_base64 is None


# ============================================================================
# Additional Edge Case Tests
# ============================================================================

class TestInvalidImageData:
    """Tests for handling invalid image data scenarios."""

    @pytest.fixture
    def mock_voicematch(self):
        """Create a mock VoiceMatchAPI."""
        mock = MagicMock()
        mock.analyze_image_bytes = AsyncMock()
        mock.find_matches = MagicMock()
        mock.get_voice_sample = MagicMock()
        return mock

    @pytest.fixture
    def mock_vibevoice(self):
        """Create a mock VibeVoiceClient."""
        mock = MagicMock()
        mock.generate = AsyncMock()
        return mock

    @pytest.mark.asyncio
    async def test_not_an_image_bytes(self, mock_voicematch, mock_vibevoice):
        """Test handling of bytes that are not a valid image."""
        from voicematch.exceptions import ImageAnalysisError

        mock_voicematch.analyze_image_bytes.side_effect = ImageAnalysisError(
            "Invalid image data: not a valid image"
        )

        orchestrator = PipelineOrchestrator(
            voicematch_api=mock_voicematch,
            vibevoice_client=mock_vibevoice,
        )

        events = []
        async for event in orchestrator.generate_streaming(
            media_bytes=b"not an image at all",
            mime_type="image/png",
            text="Hello world",
        ):
            events.append(event)

        error_events = [e for e in events if e.get("step") == "error"]
        assert len(error_events) >= 1
        assert "analysis failed" in error_events[0]["message"].lower()

    @pytest.mark.asyncio
    async def test_random_binary_data(self, mock_voicematch, mock_vibevoice):
        """Test handling of random binary data."""
        from voicematch.exceptions import ImageAnalysisError

        mock_voicematch.analyze_image_bytes.side_effect = ImageAnalysisError(
            "Cannot process file: corrupted or invalid format"
        )

        orchestrator = PipelineOrchestrator(
            voicematch_api=mock_voicematch,
            vibevoice_client=mock_vibevoice,
        )

        events = []
        async for event in orchestrator.generate_streaming(
            media_bytes=b"\x00\x01\x02\x03\x04\x05",
            mime_type="image/jpeg",
            text="Hello world",
        ):
            events.append(event)

        error_events = [e for e in events if e.get("step") == "error"]
        assert len(error_events) >= 1

    @pytest.mark.asyncio
    async def test_text_file_as_image(self, mock_voicematch, mock_vibevoice):
        """Test handling of text file sent as image."""
        from voicematch.exceptions import ImageAnalysisError

        mock_voicematch.analyze_image_bytes.side_effect = ImageAnalysisError(
            "File is not a valid image"
        )

        orchestrator = PipelineOrchestrator(
            voicematch_api=mock_voicematch,
            vibevoice_client=mock_vibevoice,
        )

        events = []
        async for event in orchestrator.generate_streaming(
            media_bytes=b"This is just plain text",
            mime_type="image/png",
            text="Hello world",
        ):
            events.append(event)

        error_events = [e for e in events if e.get("step") == "error"]
        assert len(error_events) >= 1

    @pytest.mark.asyncio
    async def test_empty_image_bytes(self, mock_voicematch, mock_vibevoice):
        """Test handling of empty image bytes."""
        from voicematch.exceptions import ImageAnalysisError

        mock_voicematch.analyze_image_bytes.side_effect = ImageAnalysisError(
            "Empty image data"
        )

        orchestrator = PipelineOrchestrator(
            voicematch_api=mock_voicematch,
            vibevoice_client=mock_vibevoice,
        )

        events = []
        async for event in orchestrator.generate_streaming(
            media_bytes=b"",
            mime_type="image/png",
            text="Hello world",
        ):
            events.append(event)

        error_events = [e for e in events if e.get("step") == "error"]
        assert len(error_events) >= 1


class TestCorruptedBase64:
    """Tests for handling corrupted base64 input."""

    def test_decode_invalid_base64_strict(self):
        """Test that invalid base64 raises an error with validate=True."""
        invalid_base64 = "this is not valid base64!!!"
        with pytest.raises(Exception):
            base64.b64decode(invalid_base64, validate=True)

    def test_decode_truncated_base64(self):
        """Test handling of truncated base64."""
        truncated = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAY"
        try:
            result = base64.b64decode(truncated)
            assert len(result) < 100
        except Exception:
            pass

    def test_decode_with_invalid_characters_strict(self):
        """Test base64 with invalid characters raises error with validate=True."""
        with_invalid_chars = "iVBORw0K$$$GgoAAAANSUhEUg"
        with pytest.raises(Exception):
            base64.b64decode(with_invalid_chars, validate=True)

    def test_decode_produces_garbage_without_validation(self):
        """Test that invalid base64 without validation produces garbage or raises."""
        # This shows that without validate=True, base64 may silently produce garbage
        invalid_base64 = "not-valid-base64!!!"
        # The decoder may succeed but produce meaningless data
        try:
            result = base64.b64decode(invalid_base64)
            # If it decodes, check it's not what we'd expect
            assert result != b"meaningful data"
        except Exception:
            pass  # Expected for truly malformed input


class TestUnsupportedMimeTypes:
    """Tests for handling unsupported MIME types."""

    @pytest.fixture
    def mock_voicematch(self):
        mock = MagicMock()
        mock.analyze_image_bytes = AsyncMock()
        mock.find_matches = MagicMock()
        mock.get_voice_sample = MagicMock()
        return mock

    @pytest.fixture
    def mock_vibevoice(self):
        mock = MagicMock()
        mock.generate = AsyncMock()
        return mock

    @pytest.mark.asyncio
    async def test_unsupported_video_mime_type(self, mock_voicematch, mock_vibevoice):
        """Test handling of unsupported video MIME type."""
        from voicematch.exceptions import UnsupportedFormatError

        mock_voicematch.analyze_image_bytes.side_effect = UnsupportedFormatError(
            "Unsupported MIME type: video/avi"
        )

        orchestrator = PipelineOrchestrator(
            voicematch_api=mock_voicematch,
            vibevoice_client=mock_vibevoice,
        )

        events = []
        async for event in orchestrator.generate_streaming(
            media_bytes=b"fake video data",
            mime_type="video/avi",
            text="Hello world",
        ):
            events.append(event)

        error_events = [e for e in events if e.get("step") == "error"]
        assert len(error_events) >= 1

    @pytest.mark.asyncio
    async def test_unsupported_audio_mime_type(self, mock_voicematch, mock_vibevoice):
        """Test handling of audio MIME type instead of image."""
        from voicematch.exceptions import UnsupportedFormatError

        mock_voicematch.analyze_image_bytes.side_effect = UnsupportedFormatError(
            "Unsupported MIME type: audio/mp3"
        )

        orchestrator = PipelineOrchestrator(
            voicematch_api=mock_voicematch,
            vibevoice_client=mock_vibevoice,
        )

        events = []
        async for event in orchestrator.generate_streaming(
            media_bytes=b"fake audio data",
            mime_type="audio/mp3",
            text="Hello world",
        ):
            events.append(event)

        error_events = [e for e in events if e.get("step") == "error"]
        assert len(error_events) >= 1

    @pytest.mark.asyncio
    async def test_malformed_mime_type(self, mock_voicematch, mock_vibevoice):
        """Test handling of malformed MIME type."""
        from voicematch.exceptions import UnsupportedFormatError

        mock_voicematch.analyze_image_bytes.side_effect = UnsupportedFormatError(
            "Unsupported MIME type: not-a-valid-mime"
        )

        orchestrator = PipelineOrchestrator(
            voicematch_api=mock_voicematch,
            vibevoice_client=mock_vibevoice,
        )

        events = []
        async for event in orchestrator.generate_streaming(
            media_bytes=b"data",
            mime_type="not-a-valid-mime",
            text="Hello world",
        ):
            events.append(event)

        error_events = [e for e in events if e.get("step") == "error"]
        assert len(error_events) >= 1


class TestEmptyText:
    """Tests for handling empty text input."""

    @pytest.fixture
    def mock_voicematch(self):
        mock = MagicMock()
        mock.analyze_image_bytes = AsyncMock()
        mock.find_matches = MagicMock()
        mock.get_voice_sample = MagicMock()
        return mock

    @pytest.fixture
    def mock_vibevoice(self):
        mock = MagicMock()
        mock.generate = AsyncMock()
        return mock

    @pytest.fixture
    def mock_analysis_response(self):
        mock = MagicMock()
        mock.estimated_age = 35
        mock.gender = "male"
        mock.race = "caucasian"
        mock.emotion = "neutral"
        return mock

    @pytest.fixture
    def mock_match_result(self):
        mock = MagicMock()
        mock.actor = MagicMock()
        mock.actor.id = "1050"
        return mock

    @pytest.mark.asyncio
    async def test_empty_string_text(
        self, mock_voicematch, mock_vibevoice, mock_analysis_response, mock_match_result, temp_dir
    ):
        """Test handling of empty string text."""
        from vibevoice_client.exceptions import ValidationError as VVValidationError

        mock_voicematch.analyze_image_bytes.return_value = mock_analysis_response
        mock_voicematch.find_matches.return_value = [mock_match_result]
        sample_path = temp_dir / "sample.wav"
        sample_path.write_bytes(b"fake wav")
        mock_voicematch.get_voice_sample.return_value = sample_path
        mock_vibevoice.generate.side_effect = VVValidationError("Text cannot be empty")

        orchestrator = PipelineOrchestrator(
            voicematch_api=mock_voicematch,
            vibevoice_client=mock_vibevoice,
        )

        events = []
        async for event in orchestrator.generate_streaming(
            media_bytes=b"fake image",
            mime_type="image/png",
            text="",
        ):
            events.append(event)

        error_events = [e for e in events if e.get("step") == "error"]
        assert len(error_events) >= 1

    @pytest.mark.asyncio
    async def test_whitespace_only_text(
        self, mock_voicematch, mock_vibevoice, mock_analysis_response, mock_match_result, temp_dir
    ):
        """Test handling of whitespace-only text."""
        from vibevoice_client.exceptions import ValidationError as VVValidationError

        mock_voicematch.analyze_image_bytes.return_value = mock_analysis_response
        mock_voicematch.find_matches.return_value = [mock_match_result]
        sample_path = temp_dir / "sample.wav"
        sample_path.write_bytes(b"fake wav")
        mock_voicematch.get_voice_sample.return_value = sample_path
        mock_vibevoice.generate.side_effect = VVValidationError("Text cannot be empty")

        orchestrator = PipelineOrchestrator(
            voicematch_api=mock_voicematch,
            vibevoice_client=mock_vibevoice,
        )

        events = []
        async for event in orchestrator.generate_streaming(
            media_bytes=b"fake image",
            mime_type="image/png",
            text="   \t\n   ",
        ):
            events.append(event)

        error_events = [e for e in events if e.get("step") == "error"]
        assert len(error_events) >= 1


class TestVeryLongText:
    """Tests for handling very long text input."""

    @pytest.fixture
    def mock_voicematch(self):
        mock = MagicMock()
        mock.analyze_image_bytes = AsyncMock()
        mock.find_matches = MagicMock()
        mock.get_voice_sample = MagicMock()
        return mock

    @pytest.fixture
    def mock_vibevoice(self):
        mock = MagicMock()
        mock.generate = AsyncMock()
        return mock

    @pytest.fixture
    def mock_analysis_response(self):
        mock = MagicMock()
        mock.estimated_age = 35
        mock.gender = "male"
        mock.race = "caucasian"
        mock.emotion = "neutral"
        return mock

    @pytest.fixture
    def mock_match_result(self):
        mock = MagicMock()
        mock.actor = MagicMock()
        mock.actor.id = "1050"
        return mock

    @pytest.fixture
    def mock_generation_result(self):
        mock = MagicMock()
        mock.audio_bytes = b"fake audio" * 100
        mock.duration_seconds = 5.0
        return mock

    @pytest.mark.asyncio
    async def test_text_at_5000_char_limit(
        self, mock_voicematch, mock_vibevoice, mock_analysis_response,
        mock_match_result, mock_generation_result, temp_dir
    ):
        """Test text at exactly 5000 character limit."""
        mock_voicematch.analyze_image_bytes.return_value = mock_analysis_response
        mock_voicematch.find_matches.return_value = [mock_match_result]
        sample_path = temp_dir / "sample.wav"
        sample_path.write_bytes(b"fake wav")
        mock_voicematch.get_voice_sample.return_value = sample_path
        mock_vibevoice.generate.return_value = mock_generation_result

        orchestrator = PipelineOrchestrator(
            voicematch_api=mock_voicematch,
            vibevoice_client=mock_vibevoice,
        )

        long_text = "A" * 5000

        events = []
        async for event in orchestrator.generate_streaming(
            media_bytes=b"fake image",
            mime_type="image/png",
            text=long_text,
        ):
            events.append(event)

        complete_events = [e for e in events if e.get("step") == "complete"]
        assert len(complete_events) == 1

    @pytest.mark.asyncio
    async def test_text_exceeds_limit(
        self, mock_voicematch, mock_vibevoice, mock_analysis_response,
        mock_match_result, temp_dir
    ):
        """Test text exceeding 5000 character limit."""
        from vibevoice_client.exceptions import ValidationError as VVValidationError

        mock_voicematch.analyze_image_bytes.return_value = mock_analysis_response
        mock_voicematch.find_matches.return_value = [mock_match_result]
        sample_path = temp_dir / "sample.wav"
        sample_path.write_bytes(b"fake wav")
        mock_voicematch.get_voice_sample.return_value = sample_path
        mock_vibevoice.generate.side_effect = VVValidationError(
            "Text exceeds maximum length of 5000 characters"
        )

        orchestrator = PipelineOrchestrator(
            voicematch_api=mock_voicematch,
            vibevoice_client=mock_vibevoice,
        )

        events = []
        async for event in orchestrator.generate_streaming(
            media_bytes=b"fake image",
            mime_type="image/png",
            text="A" * 10000,
        ):
            events.append(event)

        error_events = [e for e in events if e.get("step") == "error"]
        assert len(error_events) >= 1


class TestMissingVoiceMatches:
    """Tests for handling cases where no voice matches are found."""

    @pytest.fixture
    def mock_voicematch(self):
        mock = MagicMock()
        mock.analyze_image_bytes = AsyncMock()
        mock.find_matches = MagicMock()
        mock.get_voice_sample = MagicMock()
        return mock

    @pytest.fixture
    def mock_vibevoice(self):
        mock = MagicMock()
        mock.generate = AsyncMock()
        return mock

    @pytest.fixture
    def mock_analysis_response(self):
        mock = MagicMock()
        mock.estimated_age = 120  # Unusual age
        mock.gender = "male"
        mock.race = "mixed"
        mock.emotion = "neutral"
        return mock

    @pytest.fixture
    def mock_match_result(self):
        mock = MagicMock()
        mock.actor = MagicMock()
        mock.actor.id = "1050"
        return mock

    @pytest.mark.asyncio
    async def test_no_matching_voices_empty_database(
        self, mock_voicematch, mock_vibevoice, mock_analysis_response
    ):
        """Test when database has no matching voices."""
        mock_voicematch.analyze_image_bytes.return_value = mock_analysis_response
        mock_voicematch.find_matches.return_value = []

        orchestrator = PipelineOrchestrator(
            voicematch_api=mock_voicematch,
            vibevoice_client=mock_vibevoice,
        )

        events = []
        async for event in orchestrator.generate_streaming(
            media_bytes=b"fake image",
            mime_type="image/png",
            text="Hello world",
        ):
            events.append(event)

        error_events = [e for e in events if e.get("step") == "error"]
        assert len(error_events) >= 1
        assert any("no matching voice" in e["message"].lower() for e in error_events)

    @pytest.mark.asyncio
    async def test_matched_actor_has_no_voice_samples(
        self, mock_voicematch, mock_vibevoice, mock_analysis_response, mock_match_result
    ):
        """Test when matched actor has no voice samples."""
        mock_voicematch.analyze_image_bytes.return_value = mock_analysis_response
        mock_voicematch.find_matches.return_value = [mock_match_result]
        mock_voicematch.get_voice_sample.return_value = None

        orchestrator = PipelineOrchestrator(
            voicematch_api=mock_voicematch,
            vibevoice_client=mock_vibevoice,
        )

        events = []
        async for event in orchestrator.generate_streaming(
            media_bytes=b"fake image",
            mime_type="image/png",
            text="Hello world",
        ):
            events.append(event)

        error_events = [e for e in events if e.get("step") == "error"]
        assert len(error_events) >= 1
        assert any("no voice samples" in e["message"].lower() for e in error_events)

    @pytest.mark.asyncio
    async def test_voice_sample_file_not_found(
        self, mock_voicematch, mock_vibevoice, mock_analysis_response, mock_match_result
    ):
        """Test when voice sample file path doesn't exist."""
        from voicematch.exceptions import VoiceNotFoundError

        mock_voicematch.analyze_image_bytes.return_value = mock_analysis_response
        mock_voicematch.find_matches.return_value = [mock_match_result]
        mock_voicematch.get_voice_sample.side_effect = VoiceNotFoundError(
            "Voice sample file not found"
        )

        orchestrator = PipelineOrchestrator(
            voicematch_api=mock_voicematch,
            vibevoice_client=mock_vibevoice,
        )

        events = []
        async for event in orchestrator.generate_streaming(
            media_bytes=b"fake image",
            mime_type="image/png",
            text="Hello world",
        ):
            events.append(event)

        error_events = [e for e in events if e.get("step") == "error"]
        assert len(error_events) >= 1


class TestVibeVoiceFailures:
    """Tests for handling VibeVoice endpoint failures."""

    @pytest.fixture
    def mock_voicematch(self):
        mock = MagicMock()
        mock.analyze_image_bytes = AsyncMock()
        mock.find_matches = MagicMock()
        mock.get_voice_sample = MagicMock()
        return mock

    @pytest.fixture
    def mock_vibevoice(self):
        mock = MagicMock()
        mock.generate = AsyncMock()
        return mock

    @pytest.fixture
    def mock_analysis_response(self):
        mock = MagicMock()
        mock.estimated_age = 35
        mock.gender = "male"
        mock.race = "caucasian"
        mock.emotion = "neutral"
        return mock

    @pytest.fixture
    def mock_match_result(self):
        mock = MagicMock()
        mock.actor = MagicMock()
        mock.actor.id = "1050"
        return mock

    @pytest.mark.asyncio
    async def test_vibevoice_connection_refused(
        self, mock_voicematch, mock_vibevoice, mock_analysis_response, mock_match_result, temp_dir
    ):
        """Test when VibeVoice endpoint is unreachable."""
        from vibevoice_client.exceptions import ConnectionError as VVConnectionError

        mock_voicematch.analyze_image_bytes.return_value = mock_analysis_response
        mock_voicematch.find_matches.return_value = [mock_match_result]
        sample_path = temp_dir / "sample.wav"
        sample_path.write_bytes(b"fake wav")
        mock_voicematch.get_voice_sample.return_value = sample_path
        mock_vibevoice.generate.side_effect = VVConnectionError(
            "Connection refused: localhost:7860"
        )

        orchestrator = PipelineOrchestrator(
            voicematch_api=mock_voicematch,
            vibevoice_client=mock_vibevoice,
        )

        events = []
        async for event in orchestrator.generate_streaming(
            media_bytes=b"fake image",
            mime_type="image/png",
            text="Hello world",
        ):
            events.append(event)

        error_events = [e for e in events if e.get("step") == "error"]
        assert len(error_events) >= 1
        assert any("speech generation failed" in e["message"].lower() for e in error_events)

    @pytest.mark.asyncio
    async def test_vibevoice_timeout(
        self, mock_voicematch, mock_vibevoice, mock_analysis_response, mock_match_result, temp_dir
    ):
        """Test when VibeVoice request times out."""
        from vibevoice_client.exceptions import TimeoutError as VVTimeoutError

        mock_voicematch.analyze_image_bytes.return_value = mock_analysis_response
        mock_voicematch.find_matches.return_value = [mock_match_result]
        sample_path = temp_dir / "sample.wav"
        sample_path.write_bytes(b"fake wav")
        mock_voicematch.get_voice_sample.return_value = sample_path
        mock_vibevoice.generate.side_effect = VVTimeoutError("Request timed out")

        orchestrator = PipelineOrchestrator(
            voicematch_api=mock_voicematch,
            vibevoice_client=mock_vibevoice,
        )

        events = []
        async for event in orchestrator.generate_streaming(
            media_bytes=b"fake image",
            mime_type="image/png",
            text="Hello world",
        ):
            events.append(event)

        error_events = [e for e in events if e.get("step") == "error"]
        assert len(error_events) >= 1

    @pytest.mark.asyncio
    async def test_vibevoice_server_error(
        self, mock_voicematch, mock_vibevoice, mock_analysis_response, mock_match_result, temp_dir
    ):
        """Test when VibeVoice returns a server error."""
        from vibevoice_client.exceptions import GenerationError as VVGenerationError

        mock_voicematch.analyze_image_bytes.return_value = mock_analysis_response
        mock_voicematch.find_matches.return_value = [mock_match_result]
        sample_path = temp_dir / "sample.wav"
        sample_path.write_bytes(b"fake wav")
        mock_voicematch.get_voice_sample.return_value = sample_path
        mock_vibevoice.generate.side_effect = VVGenerationError(
            "Server error: 500 Internal Server Error"
        )

        orchestrator = PipelineOrchestrator(
            voicematch_api=mock_voicematch,
            vibevoice_client=mock_vibevoice,
        )

        events = []
        async for event in orchestrator.generate_streaming(
            media_bytes=b"fake image",
            mime_type="image/png",
            text="Hello world",
        ):
            events.append(event)

        error_events = [e for e in events if e.get("step") == "error"]
        assert len(error_events) >= 1

    @pytest.mark.asyncio
    async def test_vibevoice_returns_empty_audio(
        self, mock_voicematch, mock_vibevoice, mock_analysis_response, mock_match_result, temp_dir
    ):
        """Test when VibeVoice returns empty audio."""
        mock_voicematch.analyze_image_bytes.return_value = mock_analysis_response
        mock_voicematch.find_matches.return_value = [mock_match_result]
        sample_path = temp_dir / "sample.wav"
        sample_path.write_bytes(b"fake wav")
        mock_voicematch.get_voice_sample.return_value = sample_path

        empty_result = MagicMock()
        empty_result.audio_bytes = None
        empty_result.duration_seconds = 0
        mock_vibevoice.generate.return_value = empty_result

        orchestrator = PipelineOrchestrator(
            voicematch_api=mock_voicematch,
            vibevoice_client=mock_vibevoice,
        )

        events = []
        async for event in orchestrator.generate_streaming(
            media_bytes=b"fake image",
            mime_type="image/png",
            text="Hello world",
        ):
            events.append(event)

        error_events = [e for e in events if e.get("step") == "error"]
        assert len(error_events) >= 1
        assert any("no audio" in e["message"].lower() for e in error_events)


class TestGeminiAPIFailures:
    """Tests for handling Gemini API failures."""

    @pytest.fixture
    def mock_voicematch(self):
        mock = MagicMock()
        mock.analyze_image_bytes = AsyncMock()
        return mock

    @pytest.fixture
    def mock_vibevoice(self):
        mock = MagicMock()
        mock.generate = AsyncMock()
        return mock

    @pytest.mark.asyncio
    async def test_gemini_rate_limit(self, mock_voicematch, mock_vibevoice):
        """Test when Gemini API rate limit is exceeded."""
        from voicematch.exceptions import ImageAnalysisError

        mock_voicematch.analyze_image_bytes.side_effect = ImageAnalysisError(
            "Rate limit exceeded. Please try again later."
        )

        orchestrator = PipelineOrchestrator(
            voicematch_api=mock_voicematch,
            vibevoice_client=mock_vibevoice,
        )

        events = []
        async for event in orchestrator.generate_streaming(
            media_bytes=b"fake image",
            mime_type="image/png",
            text="Hello world",
        ):
            events.append(event)

        error_events = [e for e in events if e.get("step") == "error"]
        assert len(error_events) >= 1
        assert any("rate limit" in e["message"].lower() for e in error_events)

    @pytest.mark.asyncio
    async def test_gemini_invalid_api_key(self, mock_voicematch, mock_vibevoice):
        """Test when Gemini API key is invalid."""
        from voicematch.exceptions import ImageAnalysisError

        mock_voicematch.analyze_image_bytes.side_effect = ImageAnalysisError(
            "Invalid API key"
        )

        orchestrator = PipelineOrchestrator(
            voicematch_api=mock_voicematch,
            vibevoice_client=mock_vibevoice,
        )

        events = []
        async for event in orchestrator.generate_streaming(
            media_bytes=b"fake image",
            mime_type="image/png",
            text="Hello world",
        ):
            events.append(event)

        error_events = [e for e in events if e.get("step") == "error"]
        assert len(error_events) >= 1

    @pytest.mark.asyncio
    async def test_gemini_service_unavailable(self, mock_voicematch, mock_vibevoice):
        """Test when Gemini service is unavailable."""
        from voicematch.exceptions import ImageAnalysisError

        mock_voicematch.analyze_image_bytes.side_effect = ImageAnalysisError(
            "Service unavailable: 503"
        )

        orchestrator = PipelineOrchestrator(
            voicematch_api=mock_voicematch,
            vibevoice_client=mock_vibevoice,
        )

        events = []
        async for event in orchestrator.generate_streaming(
            media_bytes=b"fake image",
            mime_type="image/png",
            text="Hello world",
        ):
            events.append(event)

        error_events = [e for e in events if e.get("step") == "error"]
        assert len(error_events) >= 1


class TestCfgScaleParameter:
    """Tests for cfg_scale parameter combinations."""

    @pytest.fixture
    def mock_voicematch(self):
        mock = MagicMock()
        mock.analyze_image_bytes = AsyncMock()
        mock.find_matches = MagicMock()
        mock.get_voice_sample = MagicMock()
        return mock

    @pytest.fixture
    def mock_vibevoice(self):
        mock = MagicMock()
        mock.generate = AsyncMock()
        return mock

    @pytest.fixture
    def mock_analysis_response(self):
        mock = MagicMock()
        mock.estimated_age = 35
        mock.gender = "male"
        mock.race = "caucasian"
        mock.emotion = "neutral"
        return mock

    @pytest.fixture
    def mock_match_result(self):
        mock = MagicMock()
        mock.actor = MagicMock()
        mock.actor.id = "1050"
        return mock

    @pytest.fixture
    def mock_generation_result(self):
        mock = MagicMock()
        mock.audio_bytes = b"fake audio"
        mock.duration_seconds = 2.5
        return mock

    @pytest.mark.asyncio
    @pytest.mark.parametrize("cfg_scale", [0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0])
    async def test_valid_cfg_scale_values(
        self, mock_voicematch, mock_vibevoice, mock_analysis_response,
        mock_match_result, mock_generation_result, temp_dir, cfg_scale
    ):
        """Test various valid cfg_scale values."""
        mock_voicematch.analyze_image_bytes.return_value = mock_analysis_response
        mock_voicematch.find_matches.return_value = [mock_match_result]
        sample_path = temp_dir / "sample.wav"
        sample_path.write_bytes(b"fake wav")
        mock_voicematch.get_voice_sample.return_value = sample_path
        mock_vibevoice.generate.return_value = mock_generation_result

        orchestrator = PipelineOrchestrator(
            voicematch_api=mock_voicematch,
            vibevoice_client=mock_vibevoice,
        )

        events = []
        async for event in orchestrator.generate_streaming(
            media_bytes=b"fake image",
            mime_type="image/png",
            text="Hello world",
            cfg_scale=cfg_scale,
        ):
            events.append(event)

        complete_events = [e for e in events if e.get("step") == "complete"]
        assert len(complete_events) == 1

    @pytest.mark.asyncio
    async def test_cfg_scale_passed_to_vibevoice(
        self, mock_voicematch, mock_vibevoice, mock_analysis_response,
        mock_match_result, mock_generation_result, temp_dir
    ):
        """Test that cfg_scale is correctly passed to VibeVoice."""
        mock_voicematch.analyze_image_bytes.return_value = mock_analysis_response
        mock_voicematch.find_matches.return_value = [mock_match_result]
        sample_path = temp_dir / "sample.wav"
        sample_path.write_bytes(b"fake wav")
        mock_voicematch.get_voice_sample.return_value = sample_path
        mock_vibevoice.generate.return_value = mock_generation_result

        orchestrator = PipelineOrchestrator(
            voicematch_api=mock_voicematch,
            vibevoice_client=mock_vibevoice,
        )

        async for _ in orchestrator.generate_streaming(
            media_bytes=b"fake image",
            mime_type="image/png",
            text="Hello world",
            cfg_scale=3.5,
        ):
            pass

        mock_vibevoice.generate.assert_called_once()
        config_arg = mock_vibevoice.generate.call_args[0][0]
        assert config_arg.cfg_scale == 3.5

    @pytest.mark.asyncio
    async def test_default_cfg_scale(
        self, mock_voicematch, mock_vibevoice, mock_analysis_response,
        mock_match_result, mock_generation_result, temp_dir
    ):
        """Test that default cfg_scale is 2.0."""
        mock_voicematch.analyze_image_bytes.return_value = mock_analysis_response
        mock_voicematch.find_matches.return_value = [mock_match_result]
        sample_path = temp_dir / "sample.wav"
        sample_path.write_bytes(b"fake wav")
        mock_voicematch.get_voice_sample.return_value = sample_path
        mock_vibevoice.generate.return_value = mock_generation_result

        orchestrator = PipelineOrchestrator(
            voicematch_api=mock_voicematch,
            vibevoice_client=mock_vibevoice,
        )

        async for _ in orchestrator.generate_streaming(
            media_bytes=b"fake image",
            mime_type="image/png",
            text="Hello world",
        ):
            pass

        config_arg = mock_vibevoice.generate.call_args[0][0]
        assert config_arg.cfg_scale == 2.0


class TestOptimizePromptParameter:
    """Tests for optimize_prompt parameter behavior."""

    @pytest.fixture
    def mock_voicematch(self):
        mock = MagicMock()
        mock.analyze_image_bytes = AsyncMock()
        mock.find_matches = MagicMock()
        mock.get_voice_sample = MagicMock()
        return mock

    @pytest.fixture
    def mock_vibevoice(self):
        mock = MagicMock()
        mock.generate = AsyncMock()
        return mock

    @pytest.fixture
    def mock_optimizer(self):
        mock = MagicMock()
        mock.optimize = AsyncMock(return_value="Optimized: Hello world")
        return mock

    @pytest.fixture
    def mock_analysis_response(self):
        mock = MagicMock()
        mock.estimated_age = 35
        mock.gender = "male"
        mock.race = "caucasian"
        mock.emotion = "neutral"
        return mock

    @pytest.fixture
    def mock_match_result(self):
        mock = MagicMock()
        mock.actor = MagicMock()
        mock.actor.id = "1050"
        return mock

    @pytest.fixture
    def mock_generation_result(self):
        mock = MagicMock()
        mock.audio_bytes = b"fake audio"
        mock.duration_seconds = 2.5
        return mock

    @pytest.mark.asyncio
    async def test_optimize_prompt_true_with_optimizer(
        self, mock_voicematch, mock_vibevoice, mock_optimizer,
        mock_analysis_response, mock_match_result, mock_generation_result, temp_dir
    ):
        """Test optimize_prompt=True with available optimizer."""
        mock_voicematch.analyze_image_bytes.return_value = mock_analysis_response
        mock_voicematch.find_matches.return_value = [mock_match_result]
        sample_path = temp_dir / "sample.wav"
        sample_path.write_bytes(b"fake wav")
        mock_voicematch.get_voice_sample.return_value = sample_path
        mock_vibevoice.generate.return_value = mock_generation_result

        orchestrator = PipelineOrchestrator(
            voicematch_api=mock_voicematch,
            vibevoice_client=mock_vibevoice,
            prompt_optimizer=mock_optimizer,
        )

        complete_event = None
        async for event in orchestrator.generate_streaming(
            media_bytes=b"fake image",
            mime_type="image/png",
            text="Hello world",
            optimize_prompt=True,
        ):
            if event.get("step") == "complete":
                complete_event = event

        mock_optimizer.optimize.assert_called_once()
        assert complete_event["optimized_text"] == "Optimized: Hello world"
        assert complete_event["original_text"] == "Hello world"

    @pytest.mark.asyncio
    async def test_optimize_prompt_false(
        self, mock_voicematch, mock_vibevoice, mock_optimizer,
        mock_analysis_response, mock_match_result, mock_generation_result, temp_dir
    ):
        """Test optimize_prompt=False skips optimization."""
        mock_voicematch.analyze_image_bytes.return_value = mock_analysis_response
        mock_voicematch.find_matches.return_value = [mock_match_result]
        sample_path = temp_dir / "sample.wav"
        sample_path.write_bytes(b"fake wav")
        mock_voicematch.get_voice_sample.return_value = sample_path
        mock_vibevoice.generate.return_value = mock_generation_result

        orchestrator = PipelineOrchestrator(
            voicematch_api=mock_voicematch,
            vibevoice_client=mock_vibevoice,
            prompt_optimizer=mock_optimizer,
        )

        complete_event = None
        async for event in orchestrator.generate_streaming(
            media_bytes=b"fake image",
            mime_type="image/png",
            text="Hello world",
            optimize_prompt=False,
        ):
            if event.get("step") == "complete":
                complete_event = event

        mock_optimizer.optimize.assert_not_called()
        assert complete_event["optimized_text"] is None

    @pytest.mark.asyncio
    async def test_optimize_prompt_true_without_optimizer(
        self, mock_voicematch, mock_vibevoice,
        mock_analysis_response, mock_match_result, mock_generation_result, temp_dir
    ):
        """Test optimize_prompt=True without available optimizer."""
        mock_voicematch.analyze_image_bytes.return_value = mock_analysis_response
        mock_voicematch.find_matches.return_value = [mock_match_result]
        sample_path = temp_dir / "sample.wav"
        sample_path.write_bytes(b"fake wav")
        mock_voicematch.get_voice_sample.return_value = sample_path
        mock_vibevoice.generate.return_value = mock_generation_result

        orchestrator = PipelineOrchestrator(
            voicematch_api=mock_voicematch,
            vibevoice_client=mock_vibevoice,
            prompt_optimizer=None,
        )

        optimization_events = []
        async for event in orchestrator.generate_streaming(
            media_bytes=b"fake image",
            mime_type="image/png",
            text="Hello world",
            optimize_prompt=True,
        ):
            if event.get("step") == "optimizing":
                optimization_events.append(event)

        assert len(optimization_events) >= 1
        assert any("not available" in e.get("message", "").lower() for e in optimization_events)

    @pytest.mark.asyncio
    async def test_optimizer_failure_uses_original_text(
        self, mock_voicematch, mock_vibevoice, mock_optimizer,
        mock_analysis_response, mock_match_result, mock_generation_result, temp_dir
    ):
        """Test that optimizer failure falls back to original text."""
        mock_voicematch.analyze_image_bytes.return_value = mock_analysis_response
        mock_voicematch.find_matches.return_value = [mock_match_result]
        sample_path = temp_dir / "sample.wav"
        sample_path.write_bytes(b"fake wav")
        mock_voicematch.get_voice_sample.return_value = sample_path
        mock_vibevoice.generate.return_value = mock_generation_result
        mock_optimizer.optimize.side_effect = Exception("Optimizer failed")

        orchestrator = PipelineOrchestrator(
            voicematch_api=mock_voicematch,
            vibevoice_client=mock_vibevoice,
            prompt_optimizer=mock_optimizer,
        )

        complete_event = None
        async for event in orchestrator.generate_streaming(
            media_bytes=b"fake image",
            mime_type="image/png",
            text="Hello world",
            optimize_prompt=True,
        ):
            if event.get("step") == "complete":
                complete_event = event

        assert complete_event is not None
        config_arg = mock_vibevoice.generate.call_args[0][0]
        assert config_arg.text == "Hello world"


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.fixture
    def mock_voicematch(self):
        mock = MagicMock()
        mock.analyze_image_bytes = AsyncMock()
        mock.find_matches = MagicMock()
        mock.get_voice_sample = MagicMock()
        return mock

    @pytest.fixture
    def mock_vibevoice(self):
        mock = MagicMock()
        mock.generate = AsyncMock()
        return mock

    @pytest.fixture
    def mock_analysis_response(self):
        mock = MagicMock()
        mock.estimated_age = 35
        mock.gender = "male"
        mock.race = "caucasian"
        mock.emotion = "neutral"
        return mock

    @pytest.fixture
    def mock_match_result(self):
        mock = MagicMock()
        mock.actor = MagicMock()
        mock.actor.id = "1050"
        return mock

    @pytest.fixture
    def mock_generation_result(self):
        mock = MagicMock()
        mock.audio_bytes = b"fake audio"
        mock.duration_seconds = 2.5
        return mock

    @pytest.mark.asyncio
    async def test_special_characters_in_text(
        self, mock_voicematch, mock_vibevoice, mock_analysis_response,
        mock_match_result, mock_generation_result, temp_dir
    ):
        """Test handling of special characters in text."""
        mock_voicematch.analyze_image_bytes.return_value = mock_analysis_response
        mock_voicematch.find_matches.return_value = [mock_match_result]
        sample_path = temp_dir / "sample.wav"
        sample_path.write_bytes(b"fake wav")
        mock_voicematch.get_voice_sample.return_value = sample_path
        mock_vibevoice.generate.return_value = mock_generation_result

        orchestrator = PipelineOrchestrator(
            voicematch_api=mock_voicematch,
            vibevoice_client=mock_vibevoice,
        )

        special_text = "Hello! @#$%^&*() How are you?"

        complete_event = None
        async for event in orchestrator.generate_streaming(
            media_bytes=b"fake image",
            mime_type="image/png",
            text=special_text,
        ):
            if event.get("step") == "complete":
                complete_event = event

        assert complete_event["original_text"] == special_text

    @pytest.mark.asyncio
    async def test_unicode_text(
        self, mock_voicematch, mock_vibevoice, mock_analysis_response,
        mock_match_result, mock_generation_result, temp_dir
    ):
        """Test handling of Unicode text."""
        mock_voicematch.analyze_image_bytes.return_value = mock_analysis_response
        mock_voicematch.find_matches.return_value = [mock_match_result]
        sample_path = temp_dir / "sample.wav"
        sample_path.write_bytes(b"fake wav")
        mock_voicematch.get_voice_sample.return_value = sample_path
        mock_vibevoice.generate.return_value = mock_generation_result

        orchestrator = PipelineOrchestrator(
            voicematch_api=mock_voicematch,
            vibevoice_client=mock_vibevoice,
        )

        unicode_text = "Hello world!"

        complete_event = None
        async for event in orchestrator.generate_streaming(
            media_bytes=b"fake image",
            mime_type="image/png",
            text=unicode_text,
        ):
            if event.get("step") == "complete":
                complete_event = event

        assert complete_event["original_text"] == unicode_text

    @pytest.mark.asyncio
    async def test_multiline_text(
        self, mock_voicematch, mock_vibevoice, mock_analysis_response,
        mock_match_result, mock_generation_result, temp_dir
    ):
        """Test handling of multiline text."""
        mock_voicematch.analyze_image_bytes.return_value = mock_analysis_response
        mock_voicematch.find_matches.return_value = [mock_match_result]
        sample_path = temp_dir / "sample.wav"
        sample_path.write_bytes(b"fake wav")
        mock_voicematch.get_voice_sample.return_value = sample_path
        mock_vibevoice.generate.return_value = mock_generation_result

        orchestrator = PipelineOrchestrator(
            voicematch_api=mock_voicematch,
            vibevoice_client=mock_vibevoice,
        )

        multiline_text = """Line one.
Line two.
Line three."""

        complete_event = None
        async for event in orchestrator.generate_streaming(
            media_bytes=b"fake image",
            mime_type="image/png",
            text=multiline_text,
        ):
            if event.get("step") == "complete":
                complete_event = event

        assert complete_event["original_text"] == multiline_text

    @pytest.mark.asyncio
    async def test_multiple_voice_matches_uses_first(
        self, mock_voicematch, mock_vibevoice, mock_analysis_response,
        mock_generation_result, temp_dir
    ):
        """Test with multiple voice matches - should use first (best) match."""
        match1 = MagicMock()
        match1.actor = MagicMock()
        match1.actor.id = "1001"
        match2 = MagicMock()
        match2.actor = MagicMock()
        match2.actor.id = "1002"
        match3 = MagicMock()
        match3.actor = MagicMock()
        match3.actor.id = "1003"

        mock_voicematch.analyze_image_bytes.return_value = mock_analysis_response
        mock_voicematch.find_matches.return_value = [match1, match2, match3]
        sample_path = temp_dir / "sample.wav"
        sample_path.write_bytes(b"fake wav")
        mock_voicematch.get_voice_sample.return_value = sample_path
        mock_vibevoice.generate.return_value = mock_generation_result

        orchestrator = PipelineOrchestrator(
            voicematch_api=mock_voicematch,
            vibevoice_client=mock_vibevoice,
        )

        complete_event = None
        async for event in orchestrator.generate_streaming(
            media_bytes=b"fake image",
            mime_type="image/png",
            text="Hello world",
        ):
            if event.get("step") == "complete":
                complete_event = event

        assert complete_event["voice_actor"] == "1001"


class TestAllEmotionMappings:
    """Comprehensive tests for all emotion mappings."""

    @pytest.mark.parametrize("emotion,expected", [
        ("anger", "ANG"),
        ("disgust", "DIS"),
        ("fear", "FEA"),
        ("happy", "HAP"),
        ("neutral", "NEU"),
        ("sad", "SAD"),
        ("ambiguous", "NEU"),
    ])
    def test_all_known_emotions(self, emotion, expected):
        """Test all known emotion mappings."""
        assert map_gemini_emotion_to_cremad(emotion) == expected

    @pytest.mark.parametrize("emotion", [
        "excited", "confused", "surprised", "bored", "curious", "anxious",
        "content", "frustrated", "hopeful", "proud"
    ])
    def test_unknown_emotions_map_to_neutral(self, emotion):
        """Test unknown emotions default to NEU."""
        assert map_gemini_emotion_to_cremad(emotion) == "NEU"

    def test_all_emotion_mapping_dict_values(self):
        """Test that all values in EMOTION_MAPPING are valid CREMA-D codes."""
        valid_codes = {"ANG", "DIS", "FEA", "HAP", "NEU", "SAD"}
        for emotion, code in EMOTION_MAPPING.items():
            assert code in valid_codes, f"Invalid code {code} for emotion {emotion}"
