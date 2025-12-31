#!/bin/bash
# Package CREMA-D data for transfer
# Creates a compressed archive of the voice database

set -e

echo "=========================================="
echo "  Packaging Voice Data for Transfer"
echo "=========================================="

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Check data exists
if [ ! -d "data/crema_d/AudioWAV" ] || [ ! -f "data/voice_database.sqlite" ]; then
    echo "Error: Data files not found."
    echo "Expected:"
    echo "  - data/crema_d/AudioWAV/ (audio files)"
    echo "  - data/voice_database.sqlite"
    exit 1
fi

# Count files
AUDIO_COUNT=$(ls -1 data/crema_d/AudioWAV/*.wav 2>/dev/null | wc -l | tr -d ' ')
DB_SIZE=$(du -h data/voice_database.sqlite | cut -f1)

echo ""
echo "Data to package:"
echo "  - $AUDIO_COUNT audio files"
echo "  - Database: $DB_SIZE"
echo ""

# Create archive
OUTPUT_FILE="avatarvoice_data.tar.gz"
echo "Creating archive: $OUTPUT_FILE"
echo "This may take a few minutes..."

tar -czf "$OUTPUT_FILE" \
    data/voice_database.sqlite \
    data/crema_d/AudioWAV \
    data/crema_d/VideoDemographics.csv 2>/dev/null || \
tar -czf "$OUTPUT_FILE" \
    data/voice_database.sqlite \
    data/crema_d/AudioWAV

ARCHIVE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)

echo ""
echo "=========================================="
echo "  Package Complete!"
echo "=========================================="
echo ""
echo "Created: $OUTPUT_FILE ($ARCHIVE_SIZE)"
echo ""
echo "To transfer to another machine:"
echo "  scp $OUTPUT_FILE user@remote:/path/to/project/"
echo ""
echo "To extract on the destination:"
echo "  tar -xzf $OUTPUT_FILE"
echo ""
