"""Data encoding utilities for signing and binary serialization."""
import lzma
import os
import struct

import pandas as pd
from Cryptodome.Hash import SHA512
from Cryptodome.PublicKey import ECC
from Cryptodome.Signature import eddsa


def load_dimensions_from_csv(dimensions_csv_path, ocr_csv_path):
  """Load dimensions for a specific OCR file from dimensions.csv."""
  dims_df = pd.read_csv(dimensions_csv_path)
  # Filter by filename (try exact match, then basename match)
  file_dims = dims_df[dims_df['filename'] == ocr_csv_path]
  if len(file_dims) == 0:
    file_dims = dims_df[dims_df['filename'].str.endswith(os.path.basename(ocr_csv_path))]
  if len(file_dims) == 0:
    raise ValueError(f"No dimensions found for {ocr_csv_path} in {dimensions_csv_path}")
  file_dims = file_dims.sort_values('page')
  return [(int(row['width']), int(row['height'])) for _, row in file_dims.iterrows()]


def quantize_width(value):
  """Quantize a width to 8-bit based on dimension mapping."""
  min_val, max_val = (1, 1137)
  if max_val == min_val:
    return 0
  normalized = value - min_val
  scaled = normalized * 255.0 / (max_val - min_val)
  quantized = int(round(scaled))
  quantized = max(0, min(255, quantized))
  return quantized


def dequantize_width(quantized_value):
  """Dequantize an 8-bit width value back to original scale."""
  min_val, max_val = (1, 1137)
  if max_val == min_val:
    return min_val
  scaled = quantized_value * (max_val - min_val) / 255.0
  return min_val + scaled


def _normalize_df_for_signing(df):
  """Normalize DataFrame by applying quantize/dequantize to width_pixels.

  This ensures signatures match after encode/decode round-trip.
  """
  df = df.copy()
  df['width_pixels'] = df['width_pixels'].apply(lambda w: dequantize_width(quantize_width(w)))
  return df


def sign_dataframe(df, priv_path):
  """Sign a DataFrame's CSV representation using Ed25519.

  Note: width_pixels is normalized (quantize/dequantize) before signing
  to ensure signatures verify after decode.
  """
  normalized_df = _normalize_df_for_signing(df)
  h = SHA512.new(normalized_df.to_csv(index=False).encode())
  with open(priv_path, "rb") as f:
    key = ECC.import_key(f.read())
  signer = eddsa.new(key, 'rfc8032')
  return signer.sign(h)


def verify_signature(df, sig_bytes, pubkey_bytes):
  """Verify a signature on a DataFrame.

  Args:
    df: DataFrame with OCR data (e.g., from decode_binary_format)
    sig_bytes: Signature bytes
    pubkey_bytes: Raw Ed25519 public key bytes (32 bytes)

  Returns:
    True if signature is valid, False otherwise
  """
  h = SHA512.new(df.to_csv(index=False).encode())
  pub = eddsa.import_public_key(pubkey_bytes)
  verifier = eddsa.new(pub, 'rfc8032')
  try:
    verifier.verify(h, sig_bytes)
    return True
  except ValueError:
    return False


def encode_binary_format(ocr_df, sig_bytes, pubkey_bytes, dimensions):
  """Encode OCR data with signature into binary format.

  Args:
    ocr_df: DataFrame with OCR data
    sig_bytes: Signature bytes
    pubkey_bytes: Public key bytes
    dimensions: List of (width, height) tuples for each page
  """
  buffer = bytearray()

  # Signature
  buffer.append(len(sig_bytes))
  buffer.extend(sig_bytes)

  # Public key
  buffer.append(len(pubkey_bytes))
  buffer.extend(pubkey_bytes)

  # Page dimensions (column-oriented for compression)
  num_pages = len(dimensions)
  buffer.append(num_pages)
  # All widths first
  for w, h in dimensions:
    buffer.extend(struct.pack('!H', w))
  # Then all heights
  for w, h in dimensions:
    buffer.extend(struct.pack('!H', h))

  # CSV rows
  for page_num in sorted(ocr_df['page'].unique()):
    page_rows = ocr_df[ocr_df['page'] == page_num]
    buffer.extend(struct.pack('!H', len(page_rows)))  # row count for this page

    # Store all text first
    for _, row in page_rows.iterrows():
      text_bytes = row['text'].encode('utf-8')
      text_len = min(255, len(text_bytes))
      buffer.append(text_len)
      buffer.extend(text_bytes[:text_len])

    # Store all coordinates grouped by dimension
    # All left values
    for _, row in page_rows.iterrows():
      buffer.extend(struct.pack('!H', int(row['left_pixels'])))

    # All top values
    for _, row in page_rows.iterrows():
      buffer.extend(struct.pack('!H', int(row['top_pixels'])))

    # All width values (quantized)
    for _, row in page_rows.iterrows():
      buffer.extend(struct.pack('!B', quantize_width(row['width_pixels'])))

    # All height values
    for _, row in page_rows.iterrows():
      buffer.extend(struct.pack('!B', row['height_pixels']))

  return bytes(buffer)


def decode_binary_format(data):
  """Decode binary format back to signature, public key, dimensions, and OCR DataFrame.

  Args:
    data: Binary data produced by encode_binary_format

  Returns:
    Tuple of (sig_bytes, pubkey_bytes, dimensions, ocr_df)
    where dimensions is a list of (width, height) tuples
    and ocr_df has columns: text, left_pixels, top_pixels, width_pixels, height_pixels, page
  """
  offset = 0

  # Signature
  sig_len = data[offset]
  offset += 1
  sig_bytes = data[offset:offset + sig_len]
  offset += sig_len

  # Public key
  pubkey_len = data[offset]
  offset += 1
  pubkey_bytes = data[offset:offset + pubkey_len]
  offset += pubkey_len

  # Page dimensions (column-oriented)
  num_pages = data[offset]
  offset += 1
  widths = []
  for _ in range(num_pages):
    w = struct.unpack('!H', data[offset:offset + 2])[0]
    offset += 2
    widths.append(w)
  heights = []
  for _ in range(num_pages):
    h = struct.unpack('!H', data[offset:offset + 2])[0]
    offset += 2
    heights.append(h)
  dimensions = list(zip(widths, heights))

  # Read pages until we exhaust the buffer
  rows = []
  page_num = 0

  while offset < len(data):
    # Row count for this page
    row_count = struct.unpack('!H', data[offset:offset + 2])[0]
    offset += 2

    # Read all text
    texts = []
    for _ in range(row_count):
      text_len = data[offset]
      offset += 1
      text = data[offset:offset + text_len].decode('utf-8')
      offset += text_len
      texts.append(text)

    # Read all left values
    lefts = []
    for _ in range(row_count):
      left = struct.unpack('!H', data[offset:offset + 2])[0]
      offset += 2
      lefts.append(left)

    # Read all top values
    tops = []
    for _ in range(row_count):
      top = struct.unpack('!H', data[offset:offset + 2])[0]
      offset += 2
      tops.append(top)

    # Read all width values (quantized)
    widths = []
    for _ in range(row_count):
      width_q = data[offset]
      offset += 1
      widths.append(dequantize_width(width_q))

    # Read all height values
    heights = []
    for _ in range(row_count):
      height = data[offset]
      offset += 1
      heights.append(height)

    # Build rows for this page
    for i in range(row_count):
      rows.append({
        'text': texts[i],
        'left_pixels': lefts[i],
        'top_pixels': tops[i],
        'width_pixels': widths[i],
        'height_pixels': heights[i],
        'page': page_num,
      })

    page_num += 1

  ocr_df = pd.DataFrame(rows)
  return sig_bytes, pubkey_bytes, dimensions, ocr_df


def compress_data(binary_data):
  return lzma.compress(binary_data)

def decompress_lzma(binary):
  return lzma.decompress(binary)
