#!/bin/bash
# Test provider.py on example documents in a specified subfolder.
# Run from the warrint root directory: ./scripts/test_provider.sh <subfolder>
# Example: ./scripts/test_provider.sh unmodified_scanned
# Use --all to run on all test subfolders

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

DIMENSIONS_CSV=examples/dimensions.csv

FILES=(
  mtd-12024mj00012-1112024
  wied-221mj00530SCD
  casdc-323mj02081wvg-060823
)

ALL_SUBFOLDERS=(
  modified_ocrmistake_digital
  modified_ocrmistake_scanned
  modified_replacement_digital
  modified_replacement_scanned
  unmodified_scanned
)

run_subfolder() {
  local SUBFOLDER="$1"
  local INPUT_DIR="examples/pdfs/$SUBFOLDER"
  local OUTPUT_DIR="examples/pdfs/$SUBFOLDER"

  if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Directory $INPUT_DIR does not exist"
    return 1
  fi

  echo "Testing provider.py on $SUBFOLDER documents"
  echo "===================================================="
  echo ""

  for f in "${FILES[@]}"; do
    echo "=== Testing $f ==="

    if [ ! -f "$INPUT_DIR/$f.pdf" ]; then
      echo "  SKIP: Input file not found: $INPUT_DIR/$f.pdf"
      echo ""
      continue
    fi

    if [ ! -f "$INPUT_DIR/${f}_ocr.csv" ]; then
      echo "  SKIP: OCR file not found: $INPUT_DIR/${f}_ocr.csv"
      echo ""
      continue
    fi

    python3 provider.py \
      --input-pdf "$INPUT_DIR/$f.pdf" \
      --output-pdf "$OUTPUT_DIR/${f}_annotated.pdf" \
      --use-intermediate-csv-path "$INPUT_DIR/${f}_ocr.csv" \
      --dimensions-csv-path "$DIMENSIONS_CSV" \
      --seal-csv-path examples/seal_detections.csv \
      && echo "  SUCCESS: Output saved to $OUTPUT_DIR/${f}_annotated.pdf" \
      || echo "  FAILED: See error above"

    echo ""
  done

  echo "===================================================="
  echo "Test complete. Check $OUTPUT_DIR/ for results."
  echo "===================================================="
}

# Check for arguments
if [ -z "$1" ]; then
  echo "Usage: $0 <subfolder>"
  echo "       $0 --all"
  echo ""
  echo "Example: $0 unmodified_scanned"
  echo ""
  echo "Available subfolders:"
  ls -d examples/pdfs/*/ 2>/dev/null | sed 's|examples/pdfs/||g' | sed 's|/||g' | sed 's/^/  /'
  exit 1
fi

if [ "$1" == "--all" ]; then
  echo "Running tests on all subfolders..."
  echo ""
  for subfolder in "${ALL_SUBFOLDERS[@]}"; do
    run_subfolder "$subfolder"
  done
  echo "All tests complete!"
else
  run_subfolder "$1"
fi
