"""
Optar library for encoding and decoding paper-archivable barcodes.

This module provides two functions:
  - encode(base64_data) -> PDF bytes
  - decode(pdf_path) -> base64 string
"""

import base64
import glob
import os
import shutil
import subprocess
import tempfile

# Fixed format specification for all encode/decode operations
FORMAT_SPEC = "0-21-26-24-3-1-2-24"

# Path to optar executables (in ../optar/ relative to this file)
_OPTAR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'optar')

# US Letter page dimensions in points (72 points = 1 inch)
# Full page: 612 x 792 pts (8.5" x 11")
# Printable area with 0.25" margins: 576 x 756 pts
PAGE_WIDTH_PTS = 612
PAGE_HEIGHT_PTS = 792
PRINTABLE_WIDTH_PTS = 576
PRINTABLE_HEIGHT_PTS = 756


def _get_executable(name):
  """Get the path to an optar executable."""
  path = os.path.join(_OPTAR_DIR, name)
  if not os.path.exists(path):
    raise FileNotFoundError(f"Executable '{name}' not found at {path}")
  return path


def encode(base64_data: str, pdf_path: str) -> None:
  """
  Encode base64 data to an Optar PDF.

  Args:
    base64_data: Base64-encoded binary data to encode.
    pdf_path: Path where the output PDF will be saved.

  Raises:
    ValueError: If base64_data is invalid.
    RuntimeError: If encoding fails.
  """
  # Decode base64 to binary
  try:
    binary_data = base64.b64decode(base64_data)
  except Exception as e:
    raise ValueError(f"Invalid base64 data: {e}")

  # Create temporary directory for intermediate files
  temp_dir = tempfile.mkdtemp(prefix="optar_encode_")

  try:
    # Write binary data to temp file
    input_path = os.path.join(temp_dir, "input.bin")
    with open(input_path, "wb") as f:
      f.write(binary_data)

    # Run optar to create PGM files
    optar_exe = _get_executable("optar")
    base_name = os.path.join(temp_dir, "output")

    result = subprocess.run(
      [optar_exe, input_path, base_name],
      capture_output=True,
      text=True,
    )
    if result.returncode != 0:
      raise RuntimeError(f"optar failed: {result.stderr}")

    # Find generated PGM files
    pgm_files = sorted(glob.glob(f"{base_name}_*.pgm"))
    if not pgm_files:
      raise RuntimeError("optar did not create any PGM files")

    # Scale PGM files to fill US Letter page at 600 DPI
    # This ensures each optar pixel maps to enough physical pixels when printed/scanned
    # Use nearest-neighbor interpolation (-filter point) to preserve sharp edges
    dpi = 600
    page_width_px = int(PAGE_WIDTH_PTS / 72 * dpi)  # 5100 pixels
    page_height_px = int(PAGE_HEIGHT_PTS / 72 * dpi)  # 6600 pixels
    printable_width_px = int(PRINTABLE_WIDTH_PTS / 72 * dpi)  # 4800 pixels
    printable_height_px = int(PRINTABLE_HEIGHT_PTS / 72 * dpi)  # 6300 pixels

    scaled_files = []
    for pgm_file in pgm_files:
      scaled_path = pgm_file.replace(".pgm", "_scaled.png")
      result = subprocess.run(
        [
          "convert", pgm_file,
          "-filter", "point",  # Nearest-neighbor for crisp scaling
          "-resize", f"{printable_width_px}x{printable_height_px}",
          "-gravity", "center",
          "-background", "white",
          "-extent", f"{page_width_px}x{page_height_px}",
          scaled_path,
        ],
        capture_output=True,
        text=True,
      )
      if result.returncode != 0:
        raise RuntimeError(f"PGM scaling failed: {result.stderr}")
      scaled_files.append(scaled_path)

    # Convert scaled images to single PDF at 600 DPI
    temp_pdf_path = os.path.join(temp_dir, "output.pdf")
    result = subprocess.run(
      ["convert", "-density", str(dpi), "-quality", "100"] + scaled_files + [temp_pdf_path],
      capture_output=True,
      text=True,
    )
    if result.returncode != 0:
      raise RuntimeError(f"PDF conversion failed: {result.stderr}")

    # Copy PDF to final destination
    shutil.copy2(temp_pdf_path, pdf_path)

  finally:
    shutil.rmtree(temp_dir, ignore_errors=True)


def decode(pdf_path: str, debug: bool = False) -> str:
  """
  Decode an Optar PDF to base64 data.

  Args:
    pdf_path: Path to the PDF file containing Optar-encoded pages.
    debug: If True, print diagnostic info (unoptar stderr, data sizes).

  Returns:
    Base64-encoded string of the decoded binary data.

  Raises:
    FileNotFoundError: If pdf_path does not exist.
    RuntimeError: If decoding fails.
  """
  if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found: {pdf_path}")

  # Create temporary directory for intermediate files
  temp_dir = tempfile.mkdtemp(prefix="optar_decode_")

  try:
    # Convert PDF to PNG at 600 DPI
    result = subprocess.run(
      [
        "convert",
        "-density", "600",
        "-depth", "8",
        pdf_path,
        os.path.join(temp_dir, "page.png"),
      ],
      capture_output=True,
      text=True,
    )
    if result.returncode != 0:
      raise RuntimeError(f"PDF to PNG conversion failed: {result.stderr}")

    # Find generated PNG files
    png_files = sorted(glob.glob(os.path.join(temp_dir, "page*.png")))
    if not png_files:
      raise RuntimeError("No PNG files created from PDF")

    if debug:
      print(f"Generated {len(png_files)} PNG file(s) from PDF")
      for png_file in png_files:
        size = os.path.getsize(png_file)
        print(f"  {os.path.basename(png_file)}: {size} bytes")

    # Rename files to match unoptar's expected pattern (scan_0001.png, etc.)
    scan_dir = os.path.join(temp_dir, "scans")
    os.makedirs(scan_dir)

    for i, png_file in enumerate(png_files, start=1):
      new_name = f"scan_{i:04d}.png"
      shutil.copy2(png_file, os.path.join(scan_dir, new_name))

    # Run unoptar to decode
    unoptar_exe = _get_executable("unoptar")
    output_path = os.path.join(temp_dir, "output.bin")

    with open(output_path, "wb") as outf:
      result = subprocess.run(
        [unoptar_exe, FORMAT_SPEC, os.path.join(scan_dir, "scan")],
        stdout=outf,
        stderr=subprocess.PIPE,
      )

    if debug:
      print(f"unoptar stderr:\n{result.stderr.decode('utf-8', errors='replace')}")

    if result.returncode != 0:
      raise RuntimeError(f"unoptar failed: {result.stderr.decode('utf-8', errors='replace')}")

    # Read decoded data
    with open(output_path, "rb") as f:
      data = f.read()

    # Strip trailing null bytes (padding added by optar)
    raw_len = len(data)
    while data and data[-1] == 0:
      data = data[:-1]
    stripped_len = len(data)

    if debug:
      print(f"Raw unoptar output: {raw_len} bytes")
      print(f"Trailing null bytes stripped: {raw_len - stripped_len}")
      print(f"Data after stripping: {stripped_len} bytes")

    # Encode as base64 and return
    return base64.b64encode(data).decode("ascii")

  finally:
    shutil.rmtree(temp_dir, ignore_errors=True)
