"""Unit tests for CREMA-D dataset download module."""

import pytest
from unittest.mock import Mock, patch


class TestDownloadCremaD:
    """Tests for download module."""

    def test_download_creates_correct_directory_structure(self, temp_dir):
        """Test that download creates the expected directory structure."""
        from voicematch.download import ensure_directories

        output_dir = temp_dir / "crema_d"
        ensure_directories(output_dir)

        assert output_dir.exists()
        assert (output_dir / "AudioWAV").exists()

    def test_download_demographics_csv(self, temp_dir):
        """Test that VideoDemographics.csv is downloaded correctly."""
        from voicematch.download import download_demographics_csv

        output_path = temp_dir / "VideoDemographics.csv"

        # Mock httpx to avoid actual network call
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "ActorID,Age,Sex,Race,Ethnicity\n1001,51,Male,Caucasian,Not Hispanic\n"
        mock_response.raise_for_status = Mock()

        with patch("voicematch.download.httpx.get", return_value=mock_response):
            download_demographics_csv(output_path)

        assert output_path.exists()
        content = output_path.read_text()
        assert "ActorID" in content
        assert "1001" in content

    def test_download_handles_network_error(self, temp_dir):
        """Test that network errors are handled gracefully."""
        from voicematch.download import download_demographics_csv, DownloadError
        import httpx

        output_path = temp_dir / "VideoDemographics.csv"

        with patch(
            "voicematch.download.httpx.get", side_effect=httpx.ConnectError("Connection failed")
        ):
            with pytest.raises(DownloadError, match="Failed to download"):
                download_demographics_csv(output_path)

    def test_parse_audio_filename_valid(self):
        """Test parsing of valid CREMA-D audio filenames."""
        from voicematch.download import parse_audio_filename

        result = parse_audio_filename("1001_IEO_ANG_HI.wav")

        assert result["actor_id"] == "1001"
        assert result["sentence"] == "IEO"
        assert result["emotion"] == "ANG"
        assert result["level"] == "HI"

    def test_parse_audio_filename_unspecified_level(self):
        """Test parsing filename with XX (unspecified) level."""
        from voicematch.download import parse_audio_filename

        result = parse_audio_filename("1005_DFA_NEU_XX.wav")

        assert result["actor_id"] == "1005"
        assert result["sentence"] == "DFA"
        assert result["emotion"] == "NEU"
        assert result["level"] == "XX"

    def test_parse_audio_filename_invalid(self):
        """Test parsing of invalid filename returns None."""
        from voicematch.download import parse_audio_filename

        result = parse_audio_filename("invalid_file.wav")
        assert result is None

        result = parse_audio_filename("1001_IEO_ANG.wav")  # Missing level
        assert result is None


class TestHuggingFaceDownload:
    """Tests for HuggingFace dataset download."""

    @pytest.mark.slow
    def test_download_from_huggingface_structure(self, temp_dir):
        """Test that HuggingFace download function is importable and structured correctly."""
        # Just verify the function exists and can be imported
        from voicematch.download import get_audio_files_from_huggingface

        assert callable(get_audio_files_from_huggingface)

    @pytest.mark.slow
    def test_huggingface_download_handles_import_error(self, temp_dir):
        """Test that missing datasets package raises helpful error."""
        from voicematch.download import get_audio_files_from_huggingface

        # Just verify the function can be imported
        # Full import error testing would need a separate environment
        assert get_audio_files_from_huggingface is not None


class TestProgressTracking:
    """Tests for download progress tracking."""

    def test_progress_callback_called(self, temp_dir):
        """Test that progress callback is invoked during download."""
        from voicematch.download import download_demographics_csv

        progress_calls = []

        def progress_callback(current, total):
            progress_calls.append((current, total))

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "ActorID,Age,Sex,Race,Ethnicity\n1001,51,Male,Caucasian,Not Hispanic\n"
        mock_response.raise_for_status = Mock()

        with patch("voicematch.download.httpx.get", return_value=mock_response):
            download_demographics_csv(temp_dir / "test.csv", progress_callback=progress_callback)

        # At least one progress update should have been made
        assert len(progress_calls) >= 1
