#!/usr/bin/env python3
"""Download CREMA-D dataset from source.

Usage:
    python scripts/download_cremad.py
    python scripts/download_cremad.py --source huggingface
    python scripts/download_cremad.py --output-dir ./data/crema_d
    python scripts/download_cremad.py --skip-audio
"""

import argparse
import sys
from pathlib import Path

from voicematch.download import DownloadError, download_cremad


def main() -> None:
    """Entry point for the download script."""
    parser = argparse.ArgumentParser(description="Download CREMA-D dataset")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/crema_d"),
        help="Output directory for dataset (default: data/crema_d)",
    )
    parser.add_argument(
        "--source",
        choices=["huggingface"],
        default="huggingface",
        help="Source for audio files (default: huggingface)",
    )
    parser.add_argument(
        "--skip-audio",
        action="store_true",
        help="Only download metadata CSV, skip audio files",
    )

    args = parser.parse_args()

    try:
        download_cremad(args.output_dir, args.source, args.skip_audio)
    except DownloadError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
