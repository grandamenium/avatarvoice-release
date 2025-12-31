"""Tests to verify that test fixtures work correctly."""

import wave
import sqlite3


class TestFixtures:
    """Tests for test fixture validity."""

    def test_fixtures_are_available(self, temp_dir, sample_audio_file, mock_gemini_response):
        """Test that all fixtures are available and working."""
        assert temp_dir.exists()
        assert sample_audio_file.exists()
        assert mock_gemini_response is not None

    def test_temp_dir_is_cleaned_up(self, temp_dir):
        """Test that temp_dir provides a valid temporary directory."""
        # Create a file in temp_dir
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")
        assert test_file.exists()
        # Note: Cleanup verification would need a different approach
        # as the fixture cleans up after the test

    def test_sample_audio_is_valid_wav(self, sample_audio_file):
        """Test that sample audio file is a valid WAV file."""
        with wave.open(str(sample_audio_file), "r") as wav_file:
            assert wav_file.getnchannels() == 1  # Mono
            assert wav_file.getsampwidth() == 2  # 16-bit
            assert wav_file.getframerate() == 44100  # 44.1 kHz
            assert wav_file.getnframes() > 0  # Has frames

    def test_test_database_has_correct_schema(self, test_database):
        """Test that test database has the correct schema."""
        conn = sqlite3.connect(str(test_database))
        cursor = conn.cursor()

        # Check actors table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='actors'")
        assert cursor.fetchone() is not None

        # Check audio_clips table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='audio_clips'")
        assert cursor.fetchone() is not None

        # Check actors have data
        cursor.execute("SELECT COUNT(*) FROM actors")
        assert cursor.fetchone()[0] == 5

        # Check audio_clips have data
        cursor.execute("SELECT COUNT(*) FROM audio_clips")
        assert cursor.fetchone()[0] == 5

        conn.close()

    def test_mock_gemini_response_has_expected_fields(self, mock_gemini_response):
        """Test that mock Gemini response has all expected fields."""
        expected_fields = [
            "estimated_age",
            "age_range_min",
            "age_range_max",
            "gender",
            "race",
            "ethnicity_notes",
            "confidence",
        ]
        for field in expected_fields:
            assert field in mock_gemini_response
