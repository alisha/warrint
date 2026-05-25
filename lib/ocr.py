"""OCR utilities using Google Document AI."""
import copy
import json
from typing import Optional

import pandas as pd
import PyPDF2
from google.api_core.client_options import ClientOptions
from google.cloud import documentai
from google.cloud import documentai_v1

# Hardcoded configuration
_LOCATION = "us"
_PROCESSOR_VERSION = "pretrained-ocr-v2.1-2024-08-07"
_MIME_TYPE = "application/pdf"
_PROCESS_OPTIONS = documentai.ProcessOptions(
  ocr_config=documentai.OcrConfig(
    enable_image_quality_scores=True,
    hints=documentai.OcrConfig.Hints(language_hints=["en"]),
  )
)


def _process_document(
  project_id: str,
  processor_id: str,
  file_path: str,
) -> documentai.Document:
  client = documentai.DocumentProcessorServiceClient(
    client_options=ClientOptions(
      api_endpoint=f"{_LOCATION}-documentai.googleapis.com"
    )
  )

  name = client.processor_version_path(
    project_id, _LOCATION, processor_id, _PROCESSOR_VERSION
  )

  with open(file_path, "rb") as image:
    image_content = image.read()

  request = documentai.ProcessRequest(
    name=name,
    raw_document=documentai.RawDocument(content=image_content, mime_type=_MIME_TYPE),
    process_options=_PROCESS_OPTIONS,
  )

  result = client.process_document(request=request)
  return result.document


def _split_document_and_process(
  project_id: str,
  processor_id: str,
  file_path: str,
  output_file_path: Optional[str] = None,
) -> documentai.Document:

  with open(file_path, 'rb') as f:
    pdfReader = PyPDF2.PdfReader(f)
    num_pages = len(pdfReader.pages)

    if num_pages <= 15:
      return _process_document(project_id, processor_id, file_path)

    results = []
    for i in range(0, num_pages, 15):
      pdfWriter = PyPDF2.PdfWriter()
      start = i
      end = min(i + 15, num_pages)
      for j in range(start, end):
        pdfWriter.add_page(pdfReader.pages[j])
      output_filename = file_path.split(".")[0] + f"_part_{i}.pdf"
      with open(output_filename, "wb") as f:
        pdfWriter.write(f)

      results.append(_process_document(project_id, processor_id, output_filename))

    merged_results = copy.deepcopy(results[0])

    for i in range(0, len(results)):
      if output_file_path is not None:
        with open(output_file_path.split(".")[0] + f"_part_{i}.json", "w") as f:
          json.dump(documentai_v1.Document.to_json(results[i]), f)

    for i in range(1, len(results)):
      for page in results[i].pages:
        for block in page.blocks:
          for text_segment in block.layout.text_anchor.text_segments:
            old_val = text_segment.start_index
            text_segment.start_index += len(merged_results.text)
            assert(text_segment.start_index == (old_val + len(merged_results.text)))
            text_segment.end_index += len(merged_results.text)

        for token in page.tokens:
          for text_segment in token.layout.text_anchor.text_segments:
            text_segment.start_index += len(merged_results.text)
            text_segment.end_index += len(merged_results.text)

        for line in page.lines:
          for text_segment in line.layout.text_anchor.text_segments:
            text_segment.start_index += len(merged_results.text)
            text_segment.end_index += len(merged_results.text)

      merged_results.text += results[i].text
      page.page_number += len(merged_results.pages)

      merged_results.pages.extend(results[i].pages)

    return merged_results


def _ocr_df_for_one_page(page: documentai.Document.Page, text):
  words = []
  text_start_idx = []
  text_end_idx = []
  left_pixels = []
  top_pixels = []
  width_pixels = []
  height_pixels = []
  line_idxs = []
  line_text_idxs = []

  for line in page.lines:
    line_text_idxs.append(line.layout.text_anchor.text_segments)

  tokens = page.tokens
  for token in tokens:
    layout = token.layout
    vertices = layout.bounding_poly.normalized_vertices
    if len(vertices) == 0:
      continue
    pixel_vertices = layout.bounding_poly.vertices
    if len(pixel_vertices) == 0:
      continue

    pixel_x_coords = [v.x for v in pixel_vertices]
    pixel_y_coords = [v.y for v in pixel_vertices]
    left_pixels.append(min(pixel_x_coords))
    top_pixels.append(min(pixel_y_coords))
    width_pixels.append(max(pixel_x_coords) - min(pixel_x_coords))
    height_pixels.append(max(pixel_y_coords) - min(pixel_y_coords))

    start = int(layout.text_anchor.text_segments[0].start_index)
    text_start_idx.append(start)
    end = int(layout.text_anchor.text_segments[0].end_index)
    text_end_idx.append(end)
    words.append(text[start:end])

    line_idx = None
    for i, line_segments in enumerate(line_text_idxs):
      for segment in line_segments:
        if start >= segment.start_index and end <= segment.end_index:
          line_idx = i
          break
    if line_idx == None:
      print(f"Couldn't find line index for token {text[start:end]}")
    else:
      line_idxs.append(line_idx)

  page_df = pd.DataFrame.from_dict({
    'text': words,
    'text_start_idx': text_start_idx,
    'text_end_idx': text_end_idx,
    'line_idx': line_idxs,
    'left_pixels': left_pixels,
    'top_pixels': top_pixels,
    'width_pixels': width_pixels,
    'height_pixels': height_pixels
  })
  return page_df


def _get_page_dimensions(document: documentai.Document):
  widths = []
  heights = []
  pages = []
  for idx, page in enumerate(document.pages):
    dimension = page.dimension
    widths.append(dimension.width)
    heights.append(dimension.height)
    pages.append(idx)
  return (pages, widths, heights)


def _ocr_df_for_all_pages(document: documentai.Document):
  page_dfs = []
  for page_num, page in enumerate(document.pages):
    df = _ocr_df_for_one_page(page, document.text)
    df['page'] = page_num
    page_dfs.append(df)

  all_dfs = pd.concat(page_dfs)
  return all_dfs


def ocr_document(project_id: str, processor_id: str, input_pdf: str, output_path: str = None):
  """OCR a PDF document and return a DataFrame with token positions and page dimensions.

  Args:
    project_id: Google Cloud project ID
    processor_id: Document AI processor ID
    input_pdf: Path to input PDF file
    output_path: Optional path for intermediate output files

  Returns:
    Tuple of (ocr_df, dimensions) where:
      - ocr_df has columns: text, left_pixels, top_pixels, width_pixels, height_pixels, page
      - dimensions is a list of (width, height) tuples for each page
  """
  doc = _split_document_and_process(project_id, processor_id, input_pdf, output_path)
  ocr_df = _ocr_df_for_all_pages(doc)
  ocr_df = ocr_df[["text", "left_pixels", "top_pixels", "width_pixels", "height_pixels", "page"]]

  # Get page dimensions
  pages, widths, heights = _get_page_dimensions(doc)
  dimensions = list(zip(widths, heights))

  return ocr_df, dimensions
