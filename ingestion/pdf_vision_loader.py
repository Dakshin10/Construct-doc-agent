"""
ingestion/pdf_vision_loader.py
─────────────────────────────
PDF loader and vision preprocessor using PyMuPDF (fitz).

Replaces Google Document AI. Converts PDF pages into high-resolution images
for Gemini Vision processing and extracts text layout blocks.

Usage
-----
  from ingestion.pdf_vision_loader import load_pdf_for_vision

  doc_data = load_pdf_for_vision("drawings/Foundational_diagram.pdf")
  # doc_data: {
  #     "pdf_path": str,
  #     "page_images": [{"page_number": int, "image_path": str, "image_bytes": bytes}],
  #     "full_text": str,
  #     "text_blocks": [{"text": str, "page": int}]
  # }
"""

from __future__ import annotations

import os
from pathlib import Path
import pymupdf as fitz  # PyMuPDF


def load_pdf_for_vision(
    pdf_path: str | Path,
    output_dir: str | Path | None = None,
    dpi: int = 200,
) -> dict:
    """Load a PDF, extract layout text, and render pages to PNG images for Gemini Vision.

    Parameters
    ----------
    pdf_path:
        Path to the target PDF file.
    output_dir:
        Directory to save rendered page images. Defaults to a 'rendered_pages'
        subfolder next to the PDF or in current working directory.
    dpi:
        Resolution for image rendering (default 200 DPI for high vision clarity).

    Returns
    -------
    dict with keys:
      - "pdf_path": str
      - "page_images": list[dict] with page_number, image_path, image_bytes
      - "full_text": str
      - "text_blocks": list[dict]
    """
    path = Path(pdf_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    if output_dir is None:
        out_path = path.parent / "rendered_pages"
    else:
        out_path = Path(output_dir).resolve()

    out_path.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(path))
    page_images: list[dict] = []
    text_blocks: list[dict] = []
    full_text_parts: list[str] = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        page_num = page_index + 1

        # Render page to PNG image
        pix = page.get_pixmap(dpi=dpi)
        img_filename = f"{path.stem}_page_{page_num}.png"
        img_path = out_path / img_filename
        pix.save(str(img_path))

        with open(img_path, "rb") as fh:
            img_bytes = fh.read()

        page_images.append(
            {
                "page_number": page_num,
                "image_path": str(img_path),
                "image_bytes": img_bytes,
                "width": pix.width,
                "height": pix.height,
            }
        )

        # Extract text blocks
        page_text = page.get_text("text")
        full_text_parts.append(f"--- PAGE {page_num} ---\n{page_text}")

        for block in page.get_text("blocks"):
            block_text = block[4].strip() if len(block) > 4 else ""
            if block_text:
                text_blocks.append({"text": block_text, "page": page_num})

    doc.close()

    return {
        "pdf_path": str(path),
        "page_images": page_images,
        "full_text": "\n\n".join(full_text_parts),
        "text_blocks": text_blocks,
    }
