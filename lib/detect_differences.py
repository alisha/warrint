import io
import os
import string
import tempfile

import cv2
import fitz
import Levenshtein
import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial import distance

def set_up_for_edit_dist(ground_truth, new_df, case_insensitive=False):
  gt_setup = ground_truth.copy()
  new_setup = new_df.copy()

  to_replace = {
    '"': '"',
    '"': '"',
    ''': '\'',
    ''': '\'',
    '–': '-',
    '—': '-',
    '─': '-',
    '☐': '',
    '□': '',
    '☑': '',
    '✔': '',
    '✓': '',
    '_': '',
    '•': '',
    '“': '"',
    '”': '"',
  }

  for key, value in to_replace.items():
    gt_setup['text'] = gt_setup['text'].str.replace(key, value)
    new_setup['text'] = new_setup['text'].str.replace(key, value)

  gt_setup['text'] = gt_setup['text'].str.strip()
  new_setup['text'] = new_setup['text'].str.strip()
  gt_setup['text'] = gt_setup['text'].str.translate(str.maketrans('', '', string.punctuation))
  new_setup['text'] = new_setup['text'].str.translate(str.maketrans('', '', string.punctuation))
  gt_setup = gt_setup.drop(gt_setup[gt_setup['text'] == ''].index)
  new_setup = new_setup.drop(new_setup[new_setup['text'] == ''].index)
  gt_setup = gt_setup.dropna(subset=['text'])
  new_setup = new_setup.dropna(subset=['text'])

  if case_insensitive:
    gt_setup['text'] = gt_setup['text'].str.lower()
    new_setup['text'] = new_setup['text'].str.lower()

  return (gt_setup, new_setup)

# BoundingBox is a tuple of (left, top, width, height)
BoundingBox = tuple[int, int, int, int]

def compute_box(df, idx, left_col='left', top_col='top', width_col='width', height_col='height') -> BoundingBox:
  left = df.iloc[idx, df.columns.get_loc(left_col)]
  top = df.iloc[idx, df.columns.get_loc(top_col)]
  width = df.iloc[idx, df.columns.get_loc(width_col)]
  height = df.iloc[idx, df.columns.get_loc(height_col)]
  return (left, top, width, height)

def get_center_points(df, use_pixels=False):
  if use_pixels:
    df['center_x'] = df['left_pixels'] + df['width_pixels'] / 2
    df['center_y'] = df['top_pixels'] + df['height_pixels'] / 2
  else:
    df['center_x'] = df['left'] + df['width'] / 2
    df['center_y'] = df['top'] + df['height'] / 2
  return df

def find_correspondences(wordsA, wordsB):
  ptsA, ptsB = [], []
  for _, row_a in wordsA.iterrows():
    word = row_a['text']
    coordA = (row_a['center_x'], row_a['center_y'])
    matching_rows_b = wordsB[wordsB['text'] == word]
    if not matching_rows_b.empty:
      distances = matching_rows_b.apply(
          lambda row_b: distance.euclidean(coordA, (row_b['center_x'], row_b['center_y'])),
          axis=1
      )
      closest_index = distances.idxmin()
      coordB = (matching_rows_b.loc[closest_index, 'center_x'],
                matching_rows_b.loc[closest_index, 'center_y'])
      ptsA.append(coordA)
      ptsB.append(coordB)
  return ptsA, ptsB

def align_images_ocr_features(wordsA, wordsB, image):
  wordsA = get_center_points(wordsA, use_pixels=True)
  wordsB = get_center_points(wordsB, use_pixels=True)
  ptsA, ptsB = find_correspondences(wordsA, wordsB)
  if len(ptsA) < 4 or len(ptsB) < 4:
    print(f"ERROR: find_correspondences returned lengths {len(ptsA)} and {len(ptsB)}")
  ptsA_np = np.array(ptsA)
  ptsB_np = np.array(ptsB)
  (H, mask) = cv2.findHomography(ptsA_np, ptsB_np, method=cv2.RANSAC)
  if H is None:
    print(f"ERROR: Could not compute homography matrix for ptsA (len {len(ptsA_np)}) and ptsB (len {len(ptsB_np)})")
  return H

def project(df, original_shape, H, x_col='center_x', y_col='center_y', use_pixels=False):
  if use_pixels:
    points = np.stack([df[x_col], df[y_col], np.ones(len(df))], axis=-1)
  else:
    points = np.stack([df[x_col] * original_shape[1], df[y_col] * original_shape[0], np.ones(len(df))], axis=-1)
  projected_points = np.dot(points, H.T)
  df[f"{x_col}_projected"] = projected_points[:, 0] / projected_points[:, 2]
  df[f"{y_col}_projected"] = projected_points[:, 1] / projected_points[:, 2]
  return df

def pdf_page_to_image(pdf_path, page_number, output_path):
  try:
    doc = fitz.open(pdf_path)
    if page_number < 1 or page_number > len(doc):
      print(f"Error: Page {page_number} out of range for {pdf_path}")
      return None
    page = doc[page_number - 1]
    pix = page.get_pixmap(dpi=300)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    img.save(output_path, format="PNG")
    doc.close()
    return np.array(img)
  except Exception as e:
    print(f"Error processing {pdf_path}: {e}")
    return None

def detect_skew_angle(image):
  """Detect the skew angle of a document image using Hough line transform."""
  if len(image.shape) == 3:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  else:
    gray = image.copy()

  edges = cv2.Canny(gray, 50, 150, apertureSize=3)
  lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

  if lines is None or len(lines) == 0:
    return 0.0

  angles = []
  for line in lines:
    x1, y1, x2, y2 = line[0]
    if x2 - x1 == 0:
      continue
    angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
    if abs(angle) < 45:
      angles.append(angle)

  if not angles:
    return 0.0

  return np.median(angles)


def deskew_image(image, angle):
  """Rotate an image to correct for skew."""
  if abs(angle) < 0.1:
    return image

  h, w = image.shape[:2]
  center = (w // 2, h // 2)
  M = cv2.getRotationMatrix2D(center, angle, 1.0)

  cos = np.abs(M[0, 0])
  sin = np.abs(M[0, 1])
  new_w = int((h * sin) + (w * cos))
  new_h = int((h * cos) + (w * sin))

  M[0, 2] += (new_w / 2) - center[0]
  M[1, 2] += (new_h / 2) - center[1]

  rotated = cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
  return rotated


def deskew_ocr_coordinates(df, angle, original_shape, new_shape):
  """Transform OCR bounding box coordinates after deskewing."""
  if abs(angle) < 0.1:
    return df

  df = df.copy()
  h, w = original_shape[:2]
  new_h, new_w = new_shape[:2]
  center = (w / 2, h / 2)

  theta = np.radians(angle)
  cos_t = np.cos(theta)
  sin_t = np.sin(theta)

  offset_x = (new_w / 2) - center[0]
  offset_y = (new_h / 2) - center[1]

  cx = df['left_pixels'] + df['width_pixels'] / 2
  cy = df['top_pixels'] + df['height_pixels'] / 2

  new_cx = (cx - center[0]) * cos_t + (cy - center[1]) * sin_t + center[0] + offset_x
  new_cy = -(cx - center[0]) * sin_t + (cy - center[1]) * cos_t + center[1] + offset_y

  df['left_pixels'] = new_cx - df['width_pixels'] / 2
  df['top_pixels'] = new_cy - df['height_pixels'] / 2

  return df


def ignore_bboxes(df, bounding_box_list):
  """
  Filters out rows from a DataFrame if the bounding box for the row overlaps with a bounding box in bounding_box_list

  Args:
    df (pd.DataFrame): The Pandas DataFrame containing the OCR output for the document.
    bounding_box_list (list): A list of tuples (left, top, right, bottom) representing the bounding boxes to ignore. Assumes coordinates are NOT normalized

  Returns:
    pd.DataFrame: The filtered DataFrame.
  """
  df_filtered = df.copy()
  tok_left   = df_filtered['left_pixels']
  tok_top    = df_filtered['top_pixels']
  tok_right  = df_filtered['left_pixels'] + df_filtered['width_pixels']
  tok_bottom = df_filtered['top_pixels'] + df_filtered['height_pixels']

  keep_mask = pd.Series(True, index=df_filtered.index)

  for (box_left, box_top, box_right, box_bottom) in bounding_box_list:
    # 1D interval intersections
    overlaps_x = (tok_left <= box_right) & (tok_right >= box_left)
    overlaps_y = (tok_top <= box_bottom) & (tok_bottom >= box_top)

    # 2D rectangle intersection
    overlaps_box = overlaps_x & overlaps_y

    # drop overlaps
    keep_mask &= ~overlaps_box

  return df_filtered[keep_mask]

def sort_tokens_into_lines(df, top_col, left_col, height_col, opts={}):
  ignore_punctuation = opts.get('ignore_punctuation', False)
  # Allow passing y_space directly, or compute from median height
  y_space = opts.get('y_space', None)
  if y_space is None:
    y_space_of_token_height = opts.get('y_space_of_token_height', 0.4)
    if y_space_of_token_height is not None:
      y_space = df[height_col].median() * y_space_of_token_height

  df.dropna(subset='text', inplace=True)
  df_sorted = df.sort_values(by=[top_col, left_col])
  df_with_lines = df_sorted.copy()
  df_with_lines['line'] = None

  # line_boundaries is an array of tuples (top, bottom, right)
  line_boundaries = [(df_sorted.iloc[0][top_col], df_sorted.iloc[0][top_col] + df_sorted.iloc[0][height_col])]
  current_line_idx = 0
  for idx, row in df_sorted.iterrows():
    if ignore_punctuation and all([x in string.punctuation for x in row['text'].strip()]):
      continue
    
    bottom = row[top_col] + row[height_col]
    if row[top_col] > max(line_boundaries[current_line_idx][0], line_boundaries[current_line_idx][1] - y_space):
      # Create new line
      current_line_idx += 1
      line_boundaries.append((row[top_col], bottom))
      df_with_lines.loc[idx, 'line'] = current_line_idx
    else:
      # Add to current line
      line_boundaries[current_line_idx] = (min(line_boundaries[current_line_idx][0], row[top_col]), max(line_boundaries[current_line_idx][1], bottom))
      df_with_lines.loc[idx, 'line'] = current_line_idx

  return df_with_lines.sort_values(by=['line', left_col])

def extract_text_differences(scanned_df, reference_df, img, pdf_page=None, opts=None, use_pixels=False):

  scanned_df = scanned_df.copy()
  reference_df = reference_df.copy()
  scanned_df.reset_index(drop=True, inplace=True)
  reference_df.reset_index(drop=True, inplace=True)

  reference_text = ''.join(reference_df['text'].values)
  scanned_text = ''.join(scanned_df['text'].values)
  scanned_char_to_line = [idx for idx, token in scanned_df['text'].items() for _ in range(len(token))]
  reference_char_to_line = [idx for idx, token in reference_df['text'].items() for _ in range(len(token))]

  scanned_tokens_with_boxes = set()
  reference_tokens_with_boxes = set()
  replace_tokens_with_boxes = set()

  opcodes = Levenshtein.opcodes(reference_text, scanned_text)
  for (tag, i1, i2, j1, j2) in opcodes:
    if tag == 'equal':
      continue
    if tag == 'insert':
      scanned_tokens = set(scanned_char_to_line[j1:j2])
      for token_idx in scanned_tokens:
        scanned_tokens_with_boxes.add(token_idx)
    elif tag == 'delete':
      reference_tokens = set(reference_char_to_line[i1:i2])
      for token_idx in reference_tokens:
        reference_tokens_with_boxes.add(token_idx)
    elif tag == 'replace':
      reference_tokens = set(reference_char_to_line[i1:i2])
      scanned_tokens = set(scanned_char_to_line[j1:j2])
      replace_tokens_with_boxes.add((tuple(reference_tokens), tuple(scanned_tokens)))

  for (ref_token_idxs, scan_token_idxs) in list(replace_tokens_with_boxes):
    for token_idx in list(ref_token_idxs):
      reference_tokens_with_boxes.discard(token_idx)
    for token_idx in list(scan_token_idxs):
      scanned_tokens_with_boxes.discard(token_idx)

  interactive_boxes = []

  reference_left_col = 'left_projected' if not use_pixels else 'left_pixels_projected'
  reference_top_col = 'top_projected' if not use_pixels else 'top_pixels_projected'
  reference_height_col = 'height_scaled'
  reference_width_col = 'width_scaled'

  for token_idx in reference_tokens_with_boxes:
    box = compute_box(reference_df, token_idx,
                      left_col=reference_left_col, top_col=reference_top_col,
                      height_col=reference_height_col, width_col=reference_width_col)
    text = reference_df.iloc[token_idx]['text'] if token_idx < len(reference_df) else ""
    interactive_boxes.append({
      'box': box,
      'text': f"Deleted text: {text}",
      'type': 'reference',
      'token_idx': token_idx
    })

  scanned_left_col = 'left_pixels'
  scanned_top_col = 'top_pixels'
  scanned_height_col = 'height_pixels'
  scanned_width_col = 'width_pixels'

  for token_idx in scanned_tokens_with_boxes:
    box = compute_box(scanned_df, token_idx,
                      left_col=scanned_left_col, top_col=scanned_top_col,
                      height_col=scanned_height_col, width_col=scanned_width_col)
    text = scanned_df.iloc[token_idx]['text'] if token_idx < len(scanned_df) else ""
    interactive_boxes.append({
      'box': box,
      'text': f"Inserted text: {text}",
      'type': 'scanned',
      'token_idx': token_idx
    })

  def biggest_box(box1, box2):
    return (
      min(box1[0], box2[0]),
      min(box1[1], box2[1]),
      max(box1[0] + box1[2], box2[0] + box2[2]) - min(box1[0], box2[0]),
      max(box1[1] + box1[3], box2[1] + box2[3]) - min(box1[1], box2[1]),
    )

  for (ref_token_idxs, scan_token_idxs) in list(replace_tokens_with_boxes):
    ref_token_idxs = list(ref_token_idxs)
    scan_token_idxs = list(scan_token_idxs)
    ref_text = ""
    for tok in ref_token_idxs:
      ref_text += f" {reference_df.iloc[tok]['text']}"
    max_box = compute_box(scanned_df, scan_token_idxs[0],
                          left_col=scanned_left_col, top_col=scanned_top_col,
                          height_col=scanned_height_col, width_col=scanned_width_col)
    scan_text = scanned_df.iloc[scan_token_idxs[0]]['text']
    for tok in scan_token_idxs[1:]:
      scan_box = compute_box(scanned_df, tok,
                             left_col=scanned_left_col, top_col=scanned_top_col,
                             height_col=scanned_height_col, width_col=scanned_width_col)
      max_box = biggest_box(max_box, scan_box)
      scan_text += f" {scanned_df.iloc[tok]['text']}"
    interactive_boxes.append({
      'box': max_box,
      'text': f"Original text:{ref_text}\nNew text: {scan_text}",
      'type': 'replace',
    })

  interactive_boxes.sort(key=lambda x: (x['box'][1], x['box'][0]))

  return {
    'img': img,
    'interactive_boxes': interactive_boxes,
    'page_number': pdf_page
  }

def create_annotated_page(doc, page_data, top_filter=None, highlight_extra_boxes=None):
  """
  Add a single annotated page to an existing PyMuPDF document.

  Parameters:
  - doc: fitz.Document object (open PDF document)
  - page_data: Dictionary from extract_text_differences() containing:
      - 'img': image array
      - 'interactive_boxes': list of annotation boxes
      - 'page_number': page number
  - top_filter: Float (0-1) representing the fraction of the page top to filter out (default None).
            E.g., 0.25 filters out boxes in the top 25% of the page.
  - highlight_extra_boxes: List of extra boxes (with absolute pixel coordinates) to highlight as excluded (default None)

  Returns:
  - The page object that was added
  """
  img = page_data['img']
  interactive_boxes = page_data['interactive_boxes']
  pdf_page = page_data['page_number']

  # Convert image for PyMuPDF
  if isinstance(img, np.ndarray):
    if len(img.shape) == 3:
      img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
      img_pil = Image.fromarray(img)
  else:
    img_pil = img

  # Get image dimensions
  img_width, img_height = img_pil.size

  # US Letter size: 8.5" x 11" = 612 x 792 points
  page_width = 612
  page_height = 792

  # Calculate scale to fit image as large as possible while maintaining aspect ratio
  scale_x = page_width / img_width
  scale_y = page_height / img_height
  scale = min(scale_x, scale_y)  # Use smaller scale to fit both dimensions

  scaled_width = img_width * scale
  scaled_height = img_height * scale

  # Center the image on the page
  img_x = (page_width - scaled_width) / 2
  img_y = (page_height - scaled_height) / 2

  # Create US Letter page
  page = doc.new_page(width=page_width, height=page_height)

  # Convert image to bytes for insertion
  img_buffer = io.BytesIO()
  img_pil.save(img_buffer, format='PNG')
  img_bytes = img_buffer.getvalue()

  # Insert image (centered)
  img_rect = fitz.Rect(img_x, img_y, img_x + scaled_width, img_y + scaled_height)
  page.insert_image(img_rect, stream=img_bytes)

  # Add interactive annotations
  for i, box_info in enumerate(interactive_boxes):
    box = box_info['box']
    text = box_info['text']
    box_type = box_info['type']

    # Calculate annotation rectangle
    # Scale coordinates and adjust for image position
    x = img_x + (box[0] * scale)
    y = img_y + (box[1] * scale)
    width = box[2] * scale
    height = box[3] * scale

    if top_filter is not None and y < page_height * top_filter:
      # Skip boxes in the top portion of the page
      continue

    # Make sure box is not empty
    if width <= 0 or height <= 0:
      continue

    # Create rectangle for the annotation
    annot_rect = fitz.Rect(x, y, x + width, y + height)
    # print(annot_rect)

    # Determine color based on annotation type
    if box_type == 'replace':
      # Calculate difference percentage for replace annotations
      color = calculate_diff_color(text)
    else:
      # Insert and delete remain red
      color = (1, 0, 0)  # Red

    # METHOD 1: Square/Rectangle annotation (most visible in Preview)
    square = page.add_rect_annot(annot_rect)
    square.set_info(title=f"WarrInt: Box for Text Difference {i+1}")
    square.set_colors(stroke=color, fill=color)
    square.set_opacity(0.3)
    square.set_border(width=2)
    square.update()

    # METHOD 2: Text annotation (note icon) - always visible in Preview
    # Add a small note icon at the corner
    note_point = fitz.Point(x + width - 8, y + 8)
    text_annot = page.add_text_annot(note_point, text)
    text_annot.set_info(
      title=f"WarrInt: Text Difference {i+1}",
      content=text
    )
    text_annot.set_colors(stroke=color)
    text_annot.set_flags(fitz.PDF_ANNOT_IS_PRINT)  # Ensure it prints
    text_annot.update()

  if top_filter is not None:
    # Create a shaded box around filtered top portion of page
    filter_end_y = page_height * top_filter
    top_rect = fitz.Rect(0, 0, page_width, filter_end_y)
    rect = page.add_rect_annot(top_rect)
    rect.set_info(title="WarrInt: Not Showing Boxes in this Region")
    rect.set_colors(stroke=(0,0,1), fill=(0,0,1))
    rect.set_opacity(0.3)
    rect.set_border(width=2)
    rect.update()

  if highlight_extra_boxes is not None:
    for box in highlight_extra_boxes:
      x = img_x + (box[0] * scale)
      y = img_y + (box[1] * scale)
      x_max = img_x + (box[2] * scale)
      y_max = img_y + (box[3] * scale)
      scaled_box = fitz.Rect(x, y, x_max, y_max)
      rect = page.add_rect_annot(scaled_box)
      rect.set_info(title="WarrInt: Not Showing Boxes in this Region")
      rect.set_colors(stroke=(0,0,1), fill=(0,0,1))
      rect.set_opacity(0.3)
      rect.update()

  return page

def filter_boxes_by_region(interactive_boxes, img_shape, top_filter=None):
  """
  Filter out boxes in the top or bottom regions of the page.

  Args:
      interactive_boxes: List of box dictionaries with 'box' key containing (left, top, width, height, confidence)
      img_shape: Tuple of (height, width) or (height, width, channels) of the image
      top_filter: Float (0-1) representing fraction of page top to filter out

  Returns:
      List of boxes that pass the filter
  """
  if filter is None and top_filter is None:
    return interactive_boxes

  # Get image dimensions
  img_height = img_shape[0]
  img_width = img_shape[1]

  # US Letter size in points
  page_width = 612
  page_height = 792

  # Calculate scale (same logic as create_annotated_page)
  scale_x = page_width / img_width
  scale_y = page_height / img_height
  scale = min(scale_x, scale_y)

  # Calculate image offset (centered on page)
  scaled_height = img_height * scale
  img_y = (page_height - scaled_height) / 2

  filtered_boxes = []
  for box_info in interactive_boxes:
    box = box_info['box']
    # Calculate y position in page coordinates
    y = img_y + (box[1] * scale)

    # Apply filters
    if top_filter is not None and y < page_height * top_filter:
      continue

    filtered_boxes.append(box_info)

  return filtered_boxes


def calculate_diff_color(text):
  """
  Calculate color for replace annotations based on text difference.
  Returns RGB tuple from yellow (1, 1, 0) to red (1, 0, 0).

  Parameters:
  - text: The annotation text containing "Original text:" and "New text:"

  Returns:
  - RGB tuple (r, g, b) where values are 0-1
  """
  try:
    # Parse the text to extract original and new text
    if "Original text:" in text and "New text:" in text:
      parts = text.split("\n")
      original_text = ""
      new_text = ""

      for part in parts:
        if part.startswith("Original text:"):
          original_text = part.replace("Original text:", "").strip()
        elif part.startswith("New text:"):
          new_text = part.replace("New text:", "").strip()

      # Calculate Levenshtein distance
      distance = Levenshtein.distance(original_text, new_text)
      max_len = max(len(original_text), len(new_text))

      if max_len == 0:
        # No text, default to yellow
        diff_percentage = 0
      else:
        # Percentage of characters that are different
        diff_percentage = distance / max_len

      # Clamp between 0 and 1
      diff_percentage = min(1.0, max(0.0, diff_percentage))

      # Interpolate from yellow (1, 1, 0) to red (1, 0, 0)
      # As diff_percentage increases, green component decreases from 1 to 0
      r = 1.0
      g = 1.0 - diff_percentage  # Goes from 1 (yellow) to 0 (red)
      b = 0.0

      return (r, g, b)
    else:
      # Fallback to red if parsing fails
      return (1, 0, 0)

  except Exception as e:
    # If anything goes wrong, default to red
    print(f"Warning: Could not calculate diff color: {e}")
    return (1, 0, 0)


def get_seal(seal_df, pdf_name, page, page_width, page_height):
  s = seal_df[seal_df['pdf_name'] == pdf_name].iloc[0]
  if (s['page_num'] - 1) == page:
    sw = s['page_width']
    sh = s['page_height']
    seal = (s['bbox_xmin'] * page_width / sw, s['bbox_ymin'] * page_height / sh, s['bbox_xmax'] * page_width / sw, s['bbox_ymax'] * page_height / sh)
    return seal
  else:
    return None
  
def find_diffs_and_create_annotated_pdf(original_ocr_df, new_ocr_df, pdf_path, original_dimensions, new_dimensions, output_filename, seals, usability_optimizations=True, top_filter=None, deskew=True, case_insensitive=False):
  output_doc = fitz.open()

  # Make sure page lengths are the same
  assert original_ocr_df['page'].max() == new_ocr_df['page'].max()
  assert len(original_dimensions) == (original_ocr_df['page'].max() + 1)
  assert len(new_dimensions) == len(original_dimensions)

  for page in range(len(original_dimensions)):
    reference_words = original_ocr_df[original_ocr_df['page'] == page].copy()
    modified_words = new_ocr_df[new_ocr_df['page'] == page].copy()
    dimensions = original_dimensions[page]

    # Create temporary file for image
    with tempfile.NamedTemporaryFile(suffix=".png") as temp_image_file:
      image = pdf_page_to_image(pdf_path, page + 1, temp_image_file.name)

      # Scale provider's document using new (provider-side) dimensions
      scale_x = image.shape[1] / new_dimensions[page][0]  # width
      scale_y = image.shape[0] / new_dimensions[page][1]  # height

      modified_words['top_pixels'] = round(modified_words['top_pixels'] * scale_y)
      modified_words['height_pixels'] = round(modified_words['height_pixels'] * scale_y)
      modified_words['width_pixels'] = round(modified_words['width_pixels'] * scale_x)
      modified_words['left_pixels'] = round(modified_words['left_pixels'] * scale_x)

      # Scale factors for reference (original) dimensions - compute BEFORE deskewing
      scale_x_original = image.shape[1] / original_dimensions[page][0]
      scale_y_original = image.shape[0] / original_dimensions[page][1]

      # Deskew if enabled
      if deskew:
        original_shape = image.shape
        skew_angle = detect_skew_angle(image)
        if abs(skew_angle) >= 0.1:
          image = deskew_image(image, skew_angle)
          modified_words = deskew_ocr_coordinates(modified_words, skew_angle, original_shape, image.shape)

      # Align images and get homography matrix
      H_ocr = align_images_ocr_features(modified_words, reference_words, image)

      # Get seal if applicable
      scan_seal = None
      scanned_seal = None
      ref_seal = None
      pdf_name = os.path.basename(pdf_path)
      if usability_optimizations and pdf_name in seals['pdf_name'].values:
        ref_seal = get_seal(seals, pdf_name, page, original_dimensions[page][0], original_dimensions[page][1])
        if ref_seal is not None:
          reference_words = ignore_bboxes(reference_words, [ref_seal])

      reference_projected = project(reference_words, image.shape, np.linalg.inv(H_ocr), use_pixels=True)
      reference_projected = project(reference_projected, image.shape, np.linalg.inv(H_ocr), x_col='left_pixels', y_col='top_pixels', use_pixels=True)
      reference_projected['top_pixels_projected'] = round(reference_projected['top_pixels_projected'])
      reference_projected['left_pixels_projected'] = round(reference_projected['left_pixels_projected'])
      reference_projected['height_scaled'] = round(reference_projected['height_pixels'] * scale_y_original)
      reference_projected['width_scaled'] = round(reference_projected['width_pixels'] * scale_x_original)

      # Project reference seal to scanned document
      if ref_seal is not None:
        points = np.stack([[ref_seal[0], ref_seal[2]], [ref_seal[1], ref_seal[3]], [1, 1]], axis=-1)
        projected_seal = np.dot(points, np.linalg.inv(H_ocr).T)
        scanned_seal = (projected_seal[0][0] / projected_seal[0][2], projected_seal[0][1] / projected_seal[0][2], projected_seal[1][0] / projected_seal[1][2], projected_seal[1][1] / projected_seal[1][2])
        modified_words = ignore_bboxes(modified_words, [scanned_seal])
        scan_seal = [scanned_seal]

      reference_projected_setup, provider_side_df_setup = set_up_for_edit_dist(reference_projected, modified_words, case_insensitive)
      reference_projected_setup.dropna(subset=['text'], inplace=True)
      provider_side_df_setup.dropna(subset=['text'], inplace=True)

      opts = {'ignore_punctuation': True, 'y_space_of_token_height': 0.4}
      provider_side_df_with_lines = sort_tokens_into_lines(provider_side_df_setup, 'top_pixels', 'left_pixels', 'height_pixels', opts=opts).dropna(subset='text')
      reference_df_with_lines = sort_tokens_into_lines(reference_projected_setup, 'top_pixels_projected', 'left_pixels_projected', 'height_scaled', opts=opts).dropna(subset='text')

      provider_side_df_with_lines = provider_side_df_with_lines[~provider_side_df_with_lines['line'].isna()].sort_values(by=['line', 'left_pixels'])
      reference_df_with_lines = reference_df_with_lines[~reference_df_with_lines['line'].isna()].sort_values(by=['line', 'left_pixels_projected'])

      page_data = extract_text_differences(
        scanned_df=provider_side_df_with_lines,
        reference_df=reference_df_with_lines,
        img=image,
        pdf_page=page,
        opts={},
        use_pixels=True
      )

      create_annotated_page(output_doc, page_data, top_filter=top_filter, highlight_extra_boxes=scan_seal)

  # Save annotated PDF
  output_doc.save(output_filename, garbage=4, deflate=True, clean=True, pretty=True)
