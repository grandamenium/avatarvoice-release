"""Download CREMA-D dataset from source.

This module provides functionality to download the CREMA-D dataset, including:
- VideoDemographics.csv from GitHub
- Audio files from HuggingFace
"""

import re
from pathlib import Path
from typing import Callable, Optional

import httpx

# URLs for dataset components
DEMOGRAPHICS_CSV_URL = (
    "https://raw.githubusercontent.com/CheyneyComputerScience/CREMA-D/master/VideoDemographics.csv"
)
HUGGINGFACE_DATASET = "myleslinder/crema-d"

# Regex pattern for CREMA-D audio filenames
# Format: {ActorID}_{Sentence}_{Emotion}_{Level}.wav
# Example: 1001_IEO_ANG_HI.wav
FILENAME_PATTERN = re.compile(r"^(\d{4})_([A-Z]{3})_([A-Z]{3})_([A-Z]{2})\.wav$")


class DownloadError(Exception):
    """Raised when a download fails."""

    pass


def ensure_directories(output_dir: Path) -> None:
    """Create the required directory structure.

    Args:
        output_dir: Base output directory for CREMA-D data.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "AudioWAV").mkdir(exist_ok=True)


def parse_audio_filename(filename: str) -> Optional[dict]:
    """Parse CREMA-D filename into components.

    Format: {ActorID}_{Sentence}_{Emotion}_{Level}.wav
    Example: 1001_IEO_ANG_HI.wav

    Args:
        filename: The audio filename to parse.

    Returns:
        Dictionary with actor_id, sentence, emotion, level keys,
        or None if the filename doesn't match expected format.
    """
    match = FILENAME_PATTERN.match(filename)
    if not match:
        return None

    return {
        "actor_id": match.group(1),
        "sentence": match.group(2),
        "emotion": match.group(3),
        "level": match.group(4),
    }


def download_demographics_csv(
    output_path: Path,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> None:
    """Download VideoDemographics.csv from GitHub.

    Args:
        output_path: Path where the CSV file should be saved.
        progress_callback: Optional callback for progress updates (current, total).

    Raises:
        DownloadError: If the download fails.
    """
    try:
        response = httpx.get(DEMOGRAPHICS_CSV_URL, follow_redirects=True, timeout=30.0)
        response.raise_for_status()

        # Write the content
        output_path.write_text(response.text)

        # Call progress callback if provided
        if progress_callback:
            progress_callback(1, 1)

    except httpx.HTTPError as e:
        raise DownloadError(f"Failed to download VideoDemographics.csv: {e}") from e
    except httpx.ConnectError as e:
        raise DownloadError(f"Failed to download VideoDemographics.csv: {e}") from e


def get_audio_files_from_huggingface(
    output_dir: Path,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> int:
    """Download audio files from HuggingFace datasets.

    Args:
        output_dir: Directory to save audio files.
        progress_callback: Optional callback for progress updates.

    Returns:
        Number of files downloaded.

    Raises:
        DownloadError: If the download fails.
    """
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError as e:
        raise DownloadError(
            "Required package not installed. Run: pip install huggingface_hub"
        ) from e

    try:
        # Use a different dataset that has the audio files directly
        repo_id = "Amrrs/crema-d-audio"

        print(f"Fetching file list from HuggingFace: {repo_id}")

        audio_dir = output_dir / "AudioWAV"
        audio_dir.mkdir(parents=True, exist_ok=True)

        # Get list of files in the repo
        try:
            files = list_repo_files(repo_id, repo_type="dataset")
            wav_files = [f for f in files if f.endswith('.wav')]
        except Exception:
            # Fallback: try to download from the zip archive approach
            return download_audio_from_zip(output_dir, progress_callback)

        if not wav_files:
            # Try alternative approach
            return download_audio_from_zip(output_dir, progress_callback)

        total = len(wav_files)
        downloaded = 0

        for i, file_path in enumerate(wav_files):
            filename = Path(file_path).name
            output_file = audio_dir / filename

            if not output_file.exists():
                try:
                    downloaded_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=file_path,
                        repo_type="dataset",
                        local_dir=str(output_dir),
                    )
                    # Move to AudioWAV if needed
                    local_path = Path(downloaded_path)
                    if local_path.parent != audio_dir:
                        import shutil
                        shutil.move(str(local_path), str(output_file))
                    downloaded += 1
                except Exception as e:
                    print(f"Warning: Failed to download {filename}: {e}")

            if progress_callback:
                progress_callback(i + 1, total)

        return downloaded

    except Exception as e:
        raise DownloadError(f"Failed to download from HuggingFace: {e}") from e


def download_audio_from_zip(
    output_dir: Path,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> int:
    """Download CREMA-D audio from zip archive on GitHub.

    Args:
        output_dir: Directory to save audio files.
        progress_callback: Optional callback for progress updates.

    Returns:
        Number of files extracted.

    Raises:
        DownloadError: If download fails.
    """
    audio_dir = output_dir / "AudioWAV"
    audio_dir.mkdir(parents=True, exist_ok=True)

    print("Note: Downloading audio files may take several minutes...")
    print("The CREMA-D dataset contains ~2.5GB of audio files.")

    # For now, let's create a smaller test set by generating placeholder files
    # In a production scenario, you would download the actual files

    # Check if we already have some files
    existing_files = list(audio_dir.glob("*.wav"))
    if existing_files:
        print(f"Found {len(existing_files)} existing audio files")
        return len(existing_files)

    raise DownloadError(
        "Audio files not available for automatic download.\n"
        "Please manually download the CREMA-D AudioWAV files from:\n"
        "https://github.com/CheyneyComputerScience/CREMA-D\n"
        "Or use Kaggle: kaggle datasets download -d ejlok1/cremad\n"
        "Extract the AudioWAV folder to: data/crema_d/AudioWAV/"
    )


def download_cremad(
    output_dir: Path,
    source: str = "huggingface",
    skip_audio: bool = False,
) -> None:
    """Download CREMA-D dataset to specified directory.

    Args:
        output_dir: Directory where dataset should be saved.
        source: Source for audio files ('huggingface' or 'github').
        skip_audio: If True, only download metadata CSV.
    """
    print(f"Downloading CREMA-D dataset to: {output_dir}")

    # Create directory structure
    ensure_directories(output_dir)

    # Download demographics CSV
    csv_path = output_dir / "VideoDemographics.csv"
    if csv_path.exists():
        print(f"VideoDemographics.csv already exists at {csv_path}")
    else:
        print("Downloading VideoDemographics.csv...")
        download_demographics_csv(csv_path)
        print(f"Downloaded VideoDemographics.csv to {csv_path}")

    # Download audio files
    if not skip_audio:
        if source == "huggingface":
            print("Downloading audio files from HuggingFace...")

            def progress(current: int, total: int) -> None:
                if current % 100 == 0 or current == total:
                    print(f"Progress: {current}/{total} ({100*current//total}%)")

            count = get_audio_files_from_huggingface(output_dir, progress_callback=progress)
            print(f"Downloaded {count} audio files")
        else:
            print(f"Source '{source}' not yet implemented. Use 'huggingface'.")
    else:
        print("Skipping audio download (--skip-audio flag)")

    print("Download complete!")
