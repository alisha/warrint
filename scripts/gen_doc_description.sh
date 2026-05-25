#!/bin/bash
# Generate document descriptions (optar-encoded pages) for the 3 example PDFs.
# Run from the warrint root directory: ./scripts/gen_doc_description.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

PRIVATE_KEY=examples/example_private_key.pem
PUBLIC_KEY=examples/example_public_key.pem
DIMENSIONS_CSV=examples/dimensions.csv
INPUT_DIR=examples/pdfs/original
OUTPUT_DIR=examples/pdfs/original_with_doc_description

# Generate keys if they don't exist
if [ ! -f "$PRIVATE_KEY" ] || [ ! -f "$PUBLIC_KEY" ]; then
  echo "Generating key pair..."
  python3 generate_keys.py --private-key-path "$PRIVATE_KEY" --public-key-path "$PUBLIC_KEY"
fi

mkdir -p "$OUTPUT_DIR"

FILES=(
  wied-221mj00530SCD
  mtd-12024mj00012-1112024
  casdc-323mj02081wvg-060823
)

for f in "${FILES[@]}"; do
  echo "=== Processing $f ==="
  python3 court.py \
    --input-pdf "$INPUT_DIR/$f.pdf" \
    --output-pdf "$OUTPUT_DIR/$f.pdf" \
    --private-key-path "$PRIVATE_KEY" \
    --public-key-path "$PUBLIC_KEY" \
    --use-intermediate-csv-path "$INPUT_DIR/${f}_ocr.csv" \
    --dimensions-csv-path "$DIMENSIONS_CSV"
  echo ""
done

echo "Done. Output PDFs saved to $OUTPUT_DIR/"
