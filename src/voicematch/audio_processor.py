"""Audio processing for voice clip concatenation."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from pydub import AudioSegment


@dataclass
class AudioConfig:
    """Configuration for audio processing."""

    target_duration_ms: int = 30000  # 30 seconds
    crossfade_ms: int = 100
    normalize: bool = True
    target_dbfs: float = -20.0  # Target loudness
    sample_rate: int = 24000  # VibeVoice expects 24kHz
    channels: int = 1  # Mono


class AudioProcessor:
    """Processes and concatenates audio clips."""

    def __init__(self, config: Optional[AudioConfig] = None):
        """Initialize the audio processor.

        Args:
            config: Audio processing configuration. Uses defaults if not provided.
        """
        self.config = config or AudioConfig()

    def concatenate_clips(
        self,
        clip_paths: List[Path],
        output_path: Path,
    ) -> Optional[Path]:
        """Concatenate clips to target duration and save.

        Args:
            clip_paths: List of paths to audio clips.
            output_path: Path for output file.

        Returns:
            Path to output file, or None if concatenation failed.
        """
        if not clip_paths:
            return None

        # Load all clips
        clips = []
        for path in clip_paths:
            clip = self._load_clip(path)
            if clip is not None:
                clips.append(clip)

        if not clips:
            return None

        # Select clips to reach target duration
        selected = self._select_clips_for_duration(clips, self.config.target_duration_ms)

        if not selected:
            return None

        # Concatenate with crossfade
        result = selected[0]
        for clip in selected[1:]:
            if self.config.crossfade_ms > 0:
                result = result.append(clip, crossfade=self.config.crossfade_ms)
            else:
                result = result + clip

        # Trim to target duration if longer
        if len(result) > self.config.target_duration_ms:
            result = result[: self.config.target_duration_ms]

        # Normalize if configured
        if self.config.normalize:
            result = self._normalize_audio(result)

        # Convert to target sample rate and channels
        result = result.set_frame_rate(self.config.sample_rate)
        result = result.set_channels(self.config.channels)

        # Export
        result.export(str(output_path), format="wav")

        return output_path

    def _load_clip(self, path: Path) -> Optional[AudioSegment]:
        """Load and optionally normalize a single clip.

        Args:
            path: Path to audio file.

        Returns:
            AudioSegment or None if loading failed.
        """
        if not path.exists():
            return None

        try:
            return AudioSegment.from_file(str(path))
        except Exception:
            return None

    def _normalize_audio(self, audio: AudioSegment) -> AudioSegment:
        """Normalize audio to target loudness.

        Args:
            audio: AudioSegment to normalize.

        Returns:
            Normalized AudioSegment.
        """
        change_in_dbfs = self.config.target_dbfs - audio.dBFS
        return audio.apply_gain(change_in_dbfs)

    def _select_clips_for_duration(
        self,
        clips: List[AudioSegment],
        target_ms: int,
    ) -> List[AudioSegment]:
        """Select and order clips to reach target duration.

        Args:
            clips: Available audio clips.
            target_ms: Target duration in milliseconds.

        Returns:
            List of clips to concatenate.
        """
        selected = []
        total_duration = 0

        # Account for crossfade overlap
        crossfade_overlap = self.config.crossfade_ms

        for clip in clips:
            if total_duration >= target_ms:
                break

            selected.append(clip)

            if len(selected) == 1:
                total_duration = len(clip)
            else:
                # Add duration minus crossfade overlap
                total_duration += len(clip) - crossfade_overlap

        # If still short, loop clips
        if total_duration < target_ms and clips:
            while total_duration < target_ms:
                for clip in clips:
                    if total_duration >= target_ms:
                        break
                    selected.append(clip)
                    total_duration += len(clip) - crossfade_overlap

        return selected

    def get_clip_duration(self, path: Path) -> int:
        """Get duration of a clip in milliseconds.

        Args:
            path: Path to audio file.

        Returns:
            Duration in milliseconds, or 0 if file cannot be read.
        """
        clip = self._load_clip(path)
        if clip is None:
            return 0
        return len(clip)
