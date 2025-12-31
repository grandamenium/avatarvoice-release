#!/usr/bin/env python3
"""Build SQLite database from CREMA-D CSV files.

This script reads the VideoDemographics.csv and audio files to create
a SQLite database with actor demographics and audio clip metadata.

Usage:
    python scripts/build_database.py
    python scripts/build_database.py --data-dir ./data/crema_d
    python scripts/build_database.py --db-path ./data/voice_database.sqlite
"""

import argparse
import csv
import sys
from pathlib import Path

from pydub import AudioSegment

from voicematch.database import VoiceDatabase, Actor, AudioClip, DatabaseError
from voicematch.download import parse_audio_filename


def get_audio_duration_ms(filepath: Path) -> int:
    """Get duration of an audio file in milliseconds.

    Args:
        filepath: Path to audio file.

    Returns:
        Duration in milliseconds, or 0 if file cannot be read.
    """
    try:
        audio = AudioSegment.from_wav(str(filepath))
        return len(audio)
    except Exception:
        return 0


def build_database(
    data_dir: Path,
    db_path: Path,
    skip_audio_scan: bool = False,
) -> dict:
    """Build the voice database from CREMA-D data.

    Args:
        data_dir: Directory containing CREMA-D data.
        db_path: Path for output SQLite database.
        skip_audio_scan: If True, don't scan for audio files (useful for testing).

    Returns:
        Dictionary with statistics about the build.

    Raises:
        FileNotFoundError: If required files are missing.
        DatabaseError: If database operations fail.
    """
    csv_path = data_dir / "VideoDemographics.csv"
    audio_dir = data_dir / "AudioWAV"

    if not csv_path.exists():
        raise FileNotFoundError(f"Demographics CSV not found: {csv_path}")

    stats = {
        "actors_inserted": 0,
        "clips_inserted": 0,
        "clips_skipped": 0,
        "errors": [],
    }

    # Ensure output directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Create database
    db = VoiceDatabase(db_path)
    db.connect()
    db.create_schema()

    # Read and insert actors from CSV
    print(f"Reading actors from {csv_path}...")
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                actor = Actor(
                    id=row["ActorID"].strip(),
                    age=int(row["Age"]),
                    sex=row["Sex"].strip(),
                    race=row["Race"].strip(),
                    ethnicity=row["Ethnicity"].strip(),
                )
                db.insert_actor(actor)
                stats["actors_inserted"] += 1
            except (KeyError, ValueError) as e:
                stats["errors"].append(f"Error parsing actor row: {e}")

    print(f"Inserted {stats['actors_inserted']} actors")

    # Scan audio files and insert clips
    if not skip_audio_scan and audio_dir.exists():
        print(f"Scanning audio files in {audio_dir}...")
        audio_files = list(audio_dir.glob("*.wav"))
        total = len(audio_files)

        for i, audio_file in enumerate(audio_files):
            parsed = parse_audio_filename(audio_file.name)
            if parsed is None:
                stats["clips_skipped"] += 1
                continue

            # Get duration
            duration_ms = get_audio_duration_ms(audio_file)

            clip = AudioClip(
                id=0,  # Auto-assigned
                actor_id=parsed["actor_id"],
                filepath=audio_file,
                sentence=parsed["sentence"],
                emotion=parsed["emotion"],
                level=parsed["level"],
                duration_ms=duration_ms,
            )

            try:
                db.insert_audio_clip(clip)
                stats["clips_inserted"] += 1
            except DatabaseError as e:
                stats["errors"].append(f"Error inserting clip {audio_file.name}: {e}")
                stats["clips_skipped"] += 1

            if (i + 1) % 500 == 0 or i + 1 == total:
                print(f"Progress: {i + 1}/{total} clips processed")

        print(f"Inserted {stats['clips_inserted']} audio clips")
        if stats["clips_skipped"]:
            print(f"Skipped {stats['clips_skipped']} files (invalid format or errors)")
    else:
        print("Skipping audio scan (no audio directory or skip flag)")

    db.close()

    if stats["errors"]:
        print(f"\nWarnings ({len(stats['errors'])}):")
        for err in stats["errors"][:5]:
            print(f"  - {err}")
        if len(stats["errors"]) > 5:
            print(f"  ... and {len(stats['errors']) - 5} more")

    return stats


def main() -> None:
    """Entry point for the build script."""
    parser = argparse.ArgumentParser(description="Build voice database from CREMA-D data")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/crema_d"),
        help="Directory containing CREMA-D data (default: data/crema_d)",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("data/voice_database.sqlite"),
        help="Output database path (default: data/voice_database.sqlite)",
    )
    parser.add_argument(
        "--skip-audio",
        action="store_true",
        help="Skip scanning audio files (only import actors)",
    )

    args = parser.parse_args()

    try:
        print(f"Building database at {args.db_path}")
        print(f"Data directory: {args.data_dir}")
        print()

        stats = build_database(args.data_dir, args.db_path, args.skip_audio)

        print()
        print("Build complete!")
        print(f"  Actors: {stats['actors_inserted']}")
        print(f"  Clips: {stats['clips_inserted']}")

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except DatabaseError as e:
        print(f"Database error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
