#!/usr/bin/env python3
import setup_paths  # noqa: F401 - must be first

import argparse
import base64
import os
import tempfile

import pandas as pd
import PyPDF2
from Cryptodome.PublicKey import ECC

from data_encoding import (
  load_dimensions_from_csv,
  sign_dataframe,
  encode_binary_format,
  compress_data,
)
from ocr import ocr_document
from optar_lib import encode as optar_encode


def main(
  input_pdf,
  output_pdf,
  private_key_path,
  public_key_path,
  project_id=None,
  processor_id=None,
  intermediate_output_path=None,
  save_intermediate_csv_path=None,
  use_intermediate_csv_path=None,
  dimensions_csv_path=None,
  debug=False,
):
  # --- Step 1: OCR the document ---
  if use_intermediate_csv_path is None:
    if project_id is None or processor_id is None:
      raise ValueError("project_id and processor_id are required when not using use_intermediate_csv_path")
    ocr_df, dimensions = ocr_document(project_id, processor_id, input_pdf, intermediate_output_path)
    if save_intermediate_csv_path is not None:
      ocr_df.to_csv(save_intermediate_csv_path, index=False)
  else:
    if dimensions_csv_path is None:
      raise ValueError("--dimensions-csv-path is required when using --use-intermediate-csv-path")
    ocr_df = pd.read_csv(use_intermediate_csv_path)
    dimensions = load_dimensions_from_csv(dimensions_csv_path, use_intermediate_csv_path)
    if debug:
      print(f"Loaded OCR from {use_intermediate_csv_path}: {len(ocr_df)} rows")
      print(f"Loaded dimensions: {len(dimensions)} pages")

  # --- Step 2: Get signature and public key ---
  sig = sign_dataframe(ocr_df, private_key_path)
  if debug:
    print(f"Signature length: {len(sig)} bytes")
    print(f"Signature (hex): {sig.hex()[:32]}...")

  with open(public_key_path, 'rb') as f:
    pub_key_data = f.read()
  pub_key = ECC.import_key(pub_key_data).export_key(format='raw')
  if debug:
    print(f"Public key length: {len(pub_key)} bytes")
    print(f"Public key (hex): {pub_key.hex()[:32]}...")

  # --- Step 3: Encode to binary format ---
  binary_data = encode_binary_format(ocr_df, sig, pub_key, dimensions)
  if debug:
    print(f"Binary data length: {len(binary_data)} bytes")
    print(f"Compressed data length: {len(compress_data(binary_data))}")

  # --- Step 4: Compress, encode to optar, and append to original PDF ---
  compressed_data = compress_data(binary_data)
  base64_data = base64.b64encode(compressed_data).decode('ascii')

  with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
    optar_tmp_path = tmp.name

  try:
    optar_encode(base64_data, optar_tmp_path)
    if debug:
      print(f"Optar page encoded to {optar_tmp_path}")

    # Merge original PDF + optar page(s) into output
    writer = PyPDF2.PdfWriter()
    with open(input_pdf, 'rb') as f:
      reader = PyPDF2.PdfReader(f)
      for page in reader.pages:
        writer.add_page(page)
    with open(optar_tmp_path, 'rb') as f:
      optar_reader = PyPDF2.PdfReader(f)
      for page in optar_reader.pages:
        writer.add_page(page)
    with open(output_pdf, 'wb') as f:
      writer.write(f)

    if debug:
      print(f"Saved {writer.pages.__len__()} pages to {output_pdf}")
  finally:
    os.unlink(optar_tmp_path)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Process a PDF and encode OCR data with optar")
  parser.add_argument("--input-pdf", required=True, help="Input PDF file to process")
  parser.add_argument("--output-pdf", required=True, help="Output PDF with optar-encoded data")
  parser.add_argument("--private-key-path", required=True, help="Path to Ed25519 private key")
  parser.add_argument("--public-key-path", required=True, help="Path to Ed25519 public key")
  parser.add_argument("--project-id", help="Google Cloud project ID (required unless --use-intermediate-csv-path)")
  parser.add_argument("--processor-id", help="Document AI processor ID (required unless --use-intermediate-csv-path)")
  parser.add_argument("--intermediate-output-path", help="Path for intermediate OCR output")
  parser.add_argument("--save-intermediate-csv-path", help="Save OCR results to this CSV path")
  parser.add_argument("--use-intermediate-csv-path", help="Use existing CSV instead of running OCR")
  parser.add_argument("--dimensions-csv-path", help="CSV with page dimensions (required with --use-intermediate-csv-path)")
  parser.add_argument("--debug", action="store_true", help="Enable debug output")

  args = parser.parse_args()

  main(
    input_pdf=args.input_pdf,
    output_pdf=args.output_pdf,
    private_key_path=args.private_key_path,
    public_key_path=args.public_key_path,
    project_id=args.project_id,
    processor_id=args.processor_id,
    intermediate_output_path=args.intermediate_output_path,
    save_intermediate_csv_path=args.save_intermediate_csv_path,
    use_intermediate_csv_path=args.use_intermediate_csv_path,
    dimensions_csv_path=args.dimensions_csv_path,
    debug=args.debug,
  )
