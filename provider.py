#!/usr/bin/env python3
import setup_paths  # noqa: F401 - must be first

import argparse
import base64
import os
import tempfile

import pandas as pd
import PyPDF2

from data_encoding import (
  load_dimensions_from_csv,
  decompress_lzma,
  decode_binary_format,
  verify_signature,
)
from detect_differences import find_diffs_and_create_annotated_pdf
from ocr import ocr_document
from optar_lib import decode as optar_decode


def extract_last_page(pdf_path, output_path):
  """Extract the last page of a PDF to a new file."""
  with open(pdf_path, 'rb') as f:
    reader = PyPDF2.PdfReader(f)
    writer = PyPDF2.PdfWriter()
    writer.add_page(reader.pages[-1])
    with open(output_path, 'wb') as out:
      writer.write(out)


def extract_all_but_last_page(pdf_path, output_path):
  """Extract all pages except the last one to a new file."""
  with open(pdf_path, 'rb') as f:
    reader = PyPDF2.PdfReader(f)
    writer = PyPDF2.PdfWriter()
    for page in reader.pages[:-1]:
      writer.add_page(page)
    with open(output_path, 'wb') as out:
      writer.write(out)


def decode_optar_page(input_pdf, debug=False):
  """Extract and decode the optar data from the last page."""
  with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
    tmp_path = tmp.name

  try:
    extract_last_page(input_pdf, tmp_path)
    if debug:
      print(f"Extracted last page to {tmp_path}")

    # Optar decode
    base64_data = optar_decode(tmp_path, debug=debug)
    if debug:
      print(f"Decoded base64 length: {len(base64_data)}")

    # Decompress
    compressed_data = base64.b64decode(base64_data)
    binary_data = decompress_lzma(compressed_data)
    if debug:
      print(f"Decompressed data length: {len(binary_data)}")

    # Decode binary format
    sig_bytes, pubkey_bytes, dimensions, ocr_df = decode_binary_format(binary_data)
    if debug:
      print(f"Signature length: {len(sig_bytes)}")
      print(f"Public key length: {len(pubkey_bytes)}")
      print(f"Dimensions: {len(dimensions)} pages")
      print(f"DataFrame rows: {len(ocr_df)}")

    # Verify signature
    valid = verify_signature(ocr_df, sig_bytes, pubkey_bytes)
    if debug:
      print(f"Signature valid: {valid}")

    return sig_bytes, pubkey_bytes, dimensions, ocr_df, valid

  finally:
    os.unlink(tmp_path)


def ocr_current_document(
  input_pdf,
  project_id=None,
  processor_id=None,
  use_intermediate_csv_path=None,
  save_intermediate_csv_path=None,
  dimensions_csv_path=None,
  debug=False,
):
  """OCR the current document (all pages except the last optar page).

  Returns:
    Tuple of (ocr_df, dimensions) where dimensions is list of (width, height) tuples
  """
  if use_intermediate_csv_path is not None:
    if dimensions_csv_path is None:
      raise ValueError("--dimensions-csv-path is required when using --use-intermediate-csv-path")
    if debug:
      print(f"Loading OCR from {use_intermediate_csv_path}")
    ocr_df = pd.read_csv(use_intermediate_csv_path)
    dimensions = load_dimensions_from_csv(dimensions_csv_path, use_intermediate_csv_path)
    if debug:
      print(f"Loaded {len(ocr_df)} OCR tokens")
      print(f"Loaded dimensions: {len(dimensions)} pages")
    return ocr_df, dimensions

  if project_id is None or processor_id is None:
    raise ValueError("--project-id and --processor-id are required when not using --use-intermediate-csv-path")

  # Extract all but the last page
  with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
    tmp_path = tmp.name

  try:
    extract_all_but_last_page(input_pdf, tmp_path)
    if debug:
      print(f"Extracted document (without optar page) to {tmp_path}")

    # Run OCR
    ocr_df, dimensions = ocr_document(project_id, processor_id, tmp_path)
    if debug:
      print(f"OCR produced {len(ocr_df)} tokens")
      print(f"Dimensions: {dimensions}")

    if save_intermediate_csv_path is not None:
      ocr_df.to_csv(save_intermediate_csv_path, index=False)
      if debug:
        print(f"Saved OCR to {save_intermediate_csv_path}")

    return ocr_df, dimensions

  finally:
    os.unlink(tmp_path)


def main(
  input_pdf,
  output_pdf,
  project_id=None,
  processor_id=None,
  use_intermediate_csv_path=None,
  save_intermediate_csv_path=None,
  dimensions_csv_path=None,
  seal_csv_path=None,
  debug=False,
):
  # --- Step 1: Decode optar data from last page ---
  sig, pubkey, original_dimensions, original_ocr_df, valid = decode_optar_page(input_pdf, debug=debug)

  if not valid:
    print("WARNING: Signature verification failed! The encoded data may have been tampered with.")
    exit(1)

  print(f"Signature valid: {valid}")
  print(f"Recovered {len(original_ocr_df)} OCR tokens across {original_ocr_df['page'].nunique()} pages")

  # --- Step 2: OCR the current document ---
  current_ocr_df, current_dimensions = ocr_current_document(
    input_pdf,
    project_id=project_id,
    processor_id=processor_id,
    use_intermediate_csv_path=use_intermediate_csv_path,
    save_intermediate_csv_path=save_intermediate_csv_path,
    dimensions_csv_path=dimensions_csv_path,
    debug=debug,
  )

  if debug:
    print(f"Current OCR: {len(current_ocr_df)} tokens across {current_ocr_df['page'].nunique()} pages")

  # --- Step 3: Load seal CSV if provided ---
  if seal_csv_path is not None:
    seals = pd.read_csv(seal_csv_path)
    if debug:
      print(f"Loaded {len(seals)} seal entries from {seal_csv_path}")
  else:
    seals = pd.DataFrame(columns=['pdf_name', 'page_num', 'page_width', 'page_height',
                                   'bbox_xmin', 'bbox_ymin', 'bbox_xmax', 'bbox_ymax'])

  # --- Step 4: Find diffs and create annotated PDF ---
  find_diffs_and_create_annotated_pdf(
    original_ocr_df=original_ocr_df,
    new_ocr_df=current_ocr_df,
    pdf_path=input_pdf,
    original_dimensions=original_dimensions,
    new_dimensions=current_dimensions,
    output_filename=output_pdf,
    seals=seals,
    top_filter=0.05,
  )


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Decode optar data from the last page of a PDF and compare with current OCR")
  parser.add_argument("--input-pdf", required=True, help="Input PDF file with optar-encoded last page")
  parser.add_argument("--output-pdf", required=True, help="Output annotated PDF")
  parser.add_argument("--project-id", help="Google Cloud project ID (required unless using --use-intermediate-csv-path)")
  parser.add_argument("--processor-id", help="Document AI processor ID (required unless using --use-intermediate-csv-path)")
  parser.add_argument("--use-intermediate-csv-path", help="Use existing OCR CSV instead of running OCR")
  parser.add_argument("--dimensions-csv-path", help="CSV with page dimensions (required with --use-intermediate-csv-path)")
  parser.add_argument("--seal-csv-path", help="Path to seal_detections.csv for seal filtering")
  parser.add_argument("--save-intermediate-csv-path", help="Save OCR results to this CSV path")
  parser.add_argument("--debug", action="store_true", help="Enable debug output")

  args = parser.parse_args()

  main(
    input_pdf=args.input_pdf,
    output_pdf=args.output_pdf,
    project_id=args.project_id,
    processor_id=args.processor_id,
    use_intermediate_csv_path=args.use_intermediate_csv_path,
    dimensions_csv_path=args.dimensions_csv_path,
    seal_csv_path=args.seal_csv_path,
    save_intermediate_csv_path=args.save_intermediate_csv_path,
    debug=args.debug,
  )
