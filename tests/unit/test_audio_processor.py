"""Unit tests for audio processing."""

import pytest
import wave
import struct
import math

from voicematch.audio_processor import AudioProcessor, AudioConfig


@pytest.fixture
def multiple_audio_files(temp_dir):
    """Generate multiple sample WAV files."""
    files = []
    for i in range(5):
        filepath = temp_dir / f"clip_{i}.wav"
        sample_rate = 44100
        duration = 3.0  # 3 seconds each
        frequency = 440 + i * 50  # Different frequencies

        num_samples = int(sample_rate * duration)

        with wave.open(str(filepath), "w") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)

            for j in range(num_samples):
                sample = int(16000 * math.sin(2 * math.pi * frequency * j / sample_rate))
                wav_file.writeframes(struct.pack("<h", sample))

        files.append(filepath)

    return files


class TestAudioProcessor:
    """Tests for AudioProcessor class."""

    def test_concatenate_reaches_target_duration(self, multiple_audio_files, temp_dir):
        """Test that concatenation reaches the target duration."""
        processor = AudioProcessor(AudioConfig(target_duration_ms=10000))  # 10 seconds
        output_path = temp_dir / "output.wav"

        result = processor.concatenate_clips(multiple_audio_files, output_path)

        assert result.exists()

        # Check output duration
        with wave.open(str(result), "r") as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            duration_ms = (frames / rate) * 1000

        # Should be close to target (within 1 second tolerance for crossfades)
        assert 9000 <= duration_ms <= 11000

    def test_output_is_valid_wav(self, multiple_audio_files, temp_dir):
        """Test that output is a valid WAV file."""
        processor = AudioProcessor(AudioConfig(target_duration_ms=5000))
        output_path = temp_dir / "output.wav"

        result = processor.concatenate_clips(multiple_audio_files, output_path)

        # Should be readable as WAV
        with wave.open(str(result), "r") as wav_file:
            assert wav_file.getnchannels() >= 1
            assert wav_file.getsampwidth() >= 1
            assert wav_file.getframerate() > 0
            assert wav_file.getnframes() > 0

    def test_handles_single_long_clip(self, temp_dir):
        """Test handling of a single clip longer than target."""
        # Create a 10-second clip
        filepath = temp_dir / "long_clip.wav"
        sample_rate = 44100
        duration = 10.0

        num_samples = int(sample_rate * duration)
        with wave.open(str(filepath), "w") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            for i in range(num_samples):
                sample = int(16000 * math.sin(2 * math.pi * 440 * i / sample_rate))
                wav_file.writeframes(struct.pack("<h", sample))

        processor = AudioProcessor(AudioConfig(target_duration_ms=5000))
        output_path = temp_dir / "output.wav"

        result = processor.concatenate_clips([filepath], output_path)

        assert result.exists()

        # Should be truncated to target duration
        with wave.open(str(result), "r") as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            duration_ms = (frames / rate) * 1000

        assert duration_ms <= 6000  # Allow some tolerance

    def test_handles_insufficient_clips(self, temp_dir):
        """Test handling when clips don't reach target duration."""
        # Create a single short clip
        filepath = temp_dir / "short_clip.wav"
        sample_rate = 44100
        duration = 1.0  # Only 1 second

        num_samples = int(sample_rate * duration)
        with wave.open(str(filepath), "w") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            for i in range(num_samples):
                sample = int(16000 * math.sin(2 * math.pi * 440 * i / sample_rate))
                wav_file.writeframes(struct.pack("<h", sample))

        processor = AudioProcessor(AudioConfig(target_duration_ms=30000))
        output_path = temp_dir / "output.wav"

        # Should still produce output, even if short
        result = processor.concatenate_clips([filepath], output_path)

        assert result.exists()

    def test_get_clip_duration(self, sample_audio_file):
        """Test getting duration of a single clip."""
        processor = AudioProcessor()

        duration = processor.get_clip_duration(sample_audio_file)

        # Our sample audio is 1 second
        assert 900 <= duration <= 1100  # Allow tolerance


class TestAudioConfig:
    """Tests for AudioConfig dataclass."""

    def test_default_config_values(self):
        """Test default configuration values."""
        config = AudioConfig()

        assert config.target_duration_ms == 30000
        assert config.crossfade_ms == 100
        assert config.normalize is True
        assert config.target_dbfs == -20.0
        assert config.sample_rate == 24000  # VibeVoice expects 24kHz
        assert config.channels == 1

    def test_custom_config_values(self):
        """Test custom configuration values."""
        config = AudioConfig(
            target_duration_ms=15000,
            crossfade_ms=50,
            normalize=False,
        )

        assert config.target_duration_ms == 15000
        assert config.crossfade_ms == 50
        assert config.normalize is False


class TestAudioEdgeCases:
    """Edge case tests for audio processing."""

    def test_handles_empty_clip_list(self, temp_dir):
        """Test handling of empty clip list."""
        processor = AudioProcessor()
        output_path = temp_dir / "output.wav"

        # Should handle gracefully (raise or return None)
        result = processor.concatenate_clips([], output_path)

        # Either no file created or minimal output
        assert result is None or not result.exists()

    def test_handles_missing_clip_file(self, temp_dir):
        """Test handling of missing clip file."""
        processor = AudioProcessor()
        output_path = temp_dir / "output.wav"

        non_existent = [temp_dir / "does_not_exist.wav"]

        # Should handle missing files gracefully
        result = processor.concatenate_clips(non_existent, output_path)
        assert result is None or not result.exists()
