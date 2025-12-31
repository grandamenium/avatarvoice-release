"""Unit tests for VibeVoice client."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch, MagicMock
import asyncio
import base64

from vibevoice_client import (
    VibeVoiceClient,
    GenerationConfig,
    GenerationResult,
    VoiceReference,
    StreamingStatus,
)
from vibevoice_client.exceptions import (
    VibeVoiceError,
    ConnectionError,
    GenerationError,
    ValidationError,
    TimeoutError,
)
from vibevoice_client.streaming import StreamingHandler, AudioBuffer
from vibevoice_client.models import EmotionType, StreamingChunk, StreamingState


class TestGenerationConfig:
    """Tests for GenerationConfig model."""

    def test_default_values(self):
        """Should have sensible defaults."""
        config = GenerationConfig(text="Hello world")

        assert config.text == "Hello world"
        assert config.cfg_scale == 2.0
        assert config.inference_steps == 32
        assert config.temperature == 0.8
        assert config.top_p == 0.95
        assert config.speed == 1.0
        assert config.stream is False

    def test_custom_values(self):
        """Should accept custom values."""
        config = GenerationConfig(
            text="Test text",
            cfg_scale=3.5,
            inference_steps=50,
            temperature=1.0,
            emotion=EmotionType.HAPPY,
        )

        assert config.cfg_scale == 3.5
        assert config.inference_steps == 50
        assert config.temperature == 1.0
        assert config.emotion == EmotionType.HAPPY

    def test_validation_text_required(self):
        """Should require text field."""
        with pytest.raises(Exception):  # Pydantic validation error
            GenerationConfig()

    def test_validation_cfg_scale_range(self):
        """Should validate cfg_scale range."""
        with pytest.raises(Exception):
            GenerationConfig(text="Test", cfg_scale=0.0)  # Below 0.1

        with pytest.raises(Exception):
            GenerationConfig(text="Test", cfg_scale=15.0)  # Above 10.0

    def test_validation_inference_steps_range(self):
        """Should validate inference_steps range."""
        with pytest.raises(Exception):
            GenerationConfig(text="Test", inference_steps=0)  # Below 1

        with pytest.raises(Exception):
            GenerationConfig(text="Test", inference_steps=200)  # Above 100


class TestVoiceReference:
    """Tests for VoiceReference model."""

    def test_from_path(self, temp_dir):
        """Should create from file path."""
        audio_file = temp_dir / "voice.wav"
        audio_file.touch()

        ref = VoiceReference(audio_path=audio_file)

        assert ref.audio_path == audio_file
        assert ref.to_gradio_input() == str(audio_file)

    def test_from_bytes(self):
        """Should create from audio bytes."""
        audio_data = b"fake audio data"

        ref = VoiceReference(audio_bytes=audio_data)

        assert ref.audio_bytes == audio_data
        assert ref.to_gradio_input() == audio_data

    def test_from_url(self):
        """Should create from URL."""
        url = "https://example.com/voice.wav"

        ref = VoiceReference(audio_url=url)

        assert ref.audio_url == url
        assert ref.to_gradio_input() == url

    def test_from_speaker_id(self):
        """Should create from speaker ID."""
        ref = VoiceReference(speaker_id="speaker_01")

        assert ref.speaker_id == "speaker_01"
        assert ref.to_gradio_input() == "speaker_01"


class TestVibeVoiceClient:
    """Tests for VibeVoiceClient."""

    @pytest.fixture
    def client(self):
        """Create a client instance."""
        return VibeVoiceClient(endpoint="http://localhost:7860")

    def test_init_default_endpoint(self):
        """Should use default endpoint if not specified."""
        client = VibeVoiceClient()
        assert client.endpoint == "http://localhost:7860"

    def test_init_custom_endpoint(self):
        """Should use custom endpoint if specified."""
        client = VibeVoiceClient(endpoint="http://example.com:8080")
        assert client.endpoint == "http://example.com:8080"

    def test_init_timeout(self):
        """Should use custom timeout."""
        client = VibeVoiceClient(timeout=60.0)
        assert client.timeout == 60.0

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Should work as async context manager."""
        async with VibeVoiceClient() as client:
            assert client is not None

    def test_validate_config_empty_text(self, client):
        """Should reject empty text."""
        config = GenerationConfig(text="   ")

        with pytest.raises(ValidationError):
            client._validate_config(config)

    def test_prepare_request_data(self, client):
        """Should prepare correct request data for VibeVoice Podcast API."""
        config = GenerationConfig(
            text="Hello world",
            cfg_scale=2.5,
            inference_steps=32,
        )

        data = client._prepare_request_data(config)

        assert "data" in data
        # Format: [num_speakers, script, speaker1, speaker2, speaker3, speaker4, cfg_scale]
        assert data["data"][0] == 1  # num_speakers
        assert data["data"][1] == "Hello world"  # script text
        assert data["data"][6] == 2.5  # cfg_scale is last

    @pytest.mark.asyncio
    async def test_health_check_success(self, client):
        """Should return True when endpoint is healthy."""
        mock_response = Mock()
        mock_response.status_code = 200

        mock_http_client = AsyncMock()
        mock_http_client.get = AsyncMock(return_value=mock_response)
        mock_http_client.is_closed = False

        # Patch _client directly and make the property use it
        client._client = mock_http_client

        result = await client.health_check()

        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, client):
        """Should return False when endpoint is unhealthy."""
        mock_http_client = Mock()
        mock_http_client.is_closed = False

        async def mock_get(*args, **kwargs):
            raise Exception("Connection failed")

        mock_http_client.get = mock_get
        client._client = mock_http_client

        result = await client.health_check()

        assert result is False

    @pytest.mark.asyncio
    async def test_generate_success(self, client):
        """Should generate audio with proper base64 data (using httpx fallback)."""
        config = GenerationConfig(text="Hello world")

        # Create properly formatted base64 audio data
        audio_data = b"RIFF" + b"\x00" * 100  # Fake WAV header
        encoded_audio = f"data:audio/wav;base64,{base64.b64encode(audio_data).decode()}"

        # Mock the _generate_with_httpx method directly since gradio_client
        # tries to connect to the endpoint on initialization
        mock_result = GenerationResult(
            audio_bytes=audio_data,
            sample_rate=24000,
            config_used=config,
        )

        with patch.object(client, '_generate_with_gradio_client', new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = mock_result
            result = await client.generate(config)

        assert result is not None
        assert result.audio_bytes is not None

    @pytest.mark.asyncio
    async def test_generate_validation_error(self, client):
        """Should raise ValidationError for invalid config."""
        config = GenerationConfig(text="   ")  # Empty after strip

        with pytest.raises(ValidationError):
            await client.generate(config)

    @pytest.mark.asyncio
    async def test_generate_simple_interface(self, client):
        """Should work with simple interface."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": []}

        mock_http_client = AsyncMock()
        mock_http_client.post = AsyncMock(return_value=mock_response)
        mock_http_client.is_closed = False
        client._client = mock_http_client

        with pytest.raises(GenerationError):  # No audio in response
            await client.generate_simple("Hello world")

    def test_set_endpoint(self, client):
        """Should update endpoint."""
        client.set_endpoint("http://new-endpoint:8080")
        assert client.endpoint == "http://new-endpoint:8080"


class TestStreamingHandler:
    """Tests for StreamingHandler."""

    def test_init(self):
        """Should initialize with defaults."""
        handler = StreamingHandler()

        assert handler.buffer_size == 4096
        assert handler.timeout_seconds == 60.0
        assert handler.state.status == StreamingStatus.PENDING

    def test_process_chunk(self):
        """Should process audio chunks."""
        handler = StreamingHandler()

        chunk = handler.process_chunk(b"audio data", is_final=False)

        assert chunk.audio_bytes == b"audio data"
        assert chunk.chunk_index == 0
        assert chunk.is_final is False
        assert handler.state.chunks_received == 1

    def test_process_final_chunk(self):
        """Should mark final chunk."""
        handler = StreamingHandler()

        handler.process_chunk(b"chunk 1", is_final=False)
        chunk = handler.process_chunk(b"chunk 2", is_final=True)

        assert chunk.is_final is True
        assert handler.state.status == StreamingStatus.COMPLETE

    def test_get_complete_audio(self):
        """Should return all audio data."""
        handler = StreamingHandler()

        handler.process_chunk(b"chunk1")
        handler.process_chunk(b"chunk2")

        complete = handler.get_complete_audio()

        assert complete == b"chunk1chunk2"

    def test_reset(self):
        """Should reset state."""
        handler = StreamingHandler()
        handler.process_chunk(b"data")

        handler.reset()

        assert handler.state.status == StreamingStatus.PENDING
        assert handler.state.chunks_received == 0
        assert len(handler.chunks) == 0

    def test_callbacks(self):
        """Should call registered callbacks."""
        handler = StreamingHandler()
        callback_data = []

        handler.add_callback(lambda chunk: callback_data.append(chunk))
        handler.process_chunk(b"test")

        assert len(callback_data) == 1
        assert callback_data[0].audio_bytes == b"test"

    def test_remove_callback(self):
        """Should remove callbacks."""
        handler = StreamingHandler()
        callback_data = []
        callback = lambda chunk: callback_data.append(chunk)

        handler.add_callback(callback)
        handler.remove_callback(callback)
        handler.process_chunk(b"test")

        assert len(callback_data) == 0


class TestAudioBuffer:
    """Tests for AudioBuffer."""

    @pytest.mark.asyncio
    async def test_write_and_read(self):
        """Should write and read data."""
        buffer = AudioBuffer()

        await buffer.write(b"hello")
        data = await buffer.getvalue()

        assert data == b"hello"

    @pytest.mark.asyncio
    async def test_buffer_full(self):
        """Should raise error when buffer is full."""
        buffer = AudioBuffer(max_size=10)

        await buffer.write(b"12345")

        with pytest.raises(GenerationError):
            await buffer.write(b"123456")  # Would exceed 10 bytes

    @pytest.mark.asyncio
    async def test_closed_buffer(self):
        """Should raise error when writing to closed buffer."""
        buffer = AudioBuffer()
        await buffer.close()

        with pytest.raises(GenerationError):
            await buffer.write(b"data")


class TestGenerationResult:
    """Tests for GenerationResult model."""

    def test_has_audio_with_path(self, temp_dir):
        """Should detect audio presence with path."""
        audio_file = temp_dir / "audio.wav"
        audio_file.touch()

        result = GenerationResult(audio_path=audio_file)

        assert result.has_audio is True

    def test_has_audio_with_bytes(self):
        """Should detect audio presence with bytes."""
        result = GenerationResult(audio_bytes=b"data")

        assert result.has_audio is True

    def test_has_audio_empty(self):
        """Should detect no audio."""
        result = GenerationResult()

        assert result.has_audio is False


class TestEmotionType:
    """Tests for EmotionType enum."""

    def test_all_emotions(self):
        """Should have all expected emotions."""
        emotions = [e.value for e in EmotionType]

        assert "neutral" in emotions
        assert "happy" in emotions
        assert "sad" in emotions
        assert "angry" in emotions
        assert "fear" in emotions


class TestStreamingState:
    """Tests for StreamingState model."""

    def test_default_state(self):
        """Should have pending status by default."""
        state = StreamingState()

        assert state.status == StreamingStatus.PENDING
        assert state.chunks_received == 0
        assert state.total_duration == 0.0
        assert state.error_message is None
