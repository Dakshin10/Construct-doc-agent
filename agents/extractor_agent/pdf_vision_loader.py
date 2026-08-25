"""
agents/extractor_agent/pdf_vision_loader.py
─────────────────────────────────────────────
PDF loader and vision preprocessor using PyMuPDF (fitz).

Converts PDF pages into high-resolution images for Gemini Vision processing
and extracts text layout blocks with per-span detail.

Changes from original:
  - page_rotation added per page_images entry.
  - spans key added per page_images entry (font, size, dir, bbox, text per span).
  - image_bytes is NOT loaded eagerly — use get_image_bytes(page_info) instead.
    This prevents memory ballooning on multi-page drawings.

Usage
-----
  from agents.extractor_agent.pdf_vision_loader import load_pdf_for_vision, get_image_bytes

  doc_data = load_pdf_for_vision("drawings/Foundational_diagram.pdf")
  # doc_data: {
  #     "pdf_path": str,
  #     "page_images": [
  #         {
  #           "page_number": int,
  #           "image_path": str,       # use get_image_bytes() to load lazily
  #           "page_rotation": int,    # page.rotation (0, 90, 180, 270)
  #           "spans": [dict],         # per-span: font, size, dir, bbox, text
  #           "width": int, "height": int, "page_rect": list
  #         }
  #     ],
  #     "full_text": str,
  #     "text_blocks": [{"text": str, "page": int, "bbox": list, "spans": list}],
  #     "words": [{"bbox": list, "text": str, "page": int}]
  # }
"""

from __future__ import annotations

import os
from pathlib import Path
import pymupdf as fitz  # PyMuPDF


def load_pdf_for_vision(
    pdf_path: str | Path,
    output_dir: str | Path | None = None,
    dpi: int = 300,
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
        Resolution for image rendering (default 300 DPI for high vision clarity).

    Returns
    -------
    dict with keys:
      - "pdf_path": str
      - "page_images": list[dict] with page_number, image_path, image_bytes, width, height
      - "full_text": str
      - "text_blocks": list[dict]
      - "words": list[dict] with x0, y0, x1, y1, text, page
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
    words: list[dict] = []
    full_text_parts: list[str] = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        page_num = page_index + 1

        # Render page to PNG image
        pix = page.get_pixmap(dpi=dpi)
        img_filename = f"{path.stem}_page_{page_num}.png"
        img_path = out_path / img_filename
        pix.save(str(img_path))
        # image_bytes NOT loaded eagerly — use get_image_bytes(page_info) on demand

        # Extract per-span detail for this page
        page_spans: list[dict] = []
        page_dict = page.get_text("dict")
        for blk in page_dict.get("blocks", []):
            if "lines" not in blk:
                continue
            for ln in blk["lines"]:
                ldir = tuple(round(v) for v in ln.get("dir", (1, 0)))
                for sp in ln["spans"]:
                    page_spans.append({
                        "text":  sp.get("text", "").strip(),
                        "font":  sp.get("font", ""),
                        "size":  sp.get("size", 0.0),
                        "dir":   ldir,
                        "bbox":  list(sp.get("bbox", [0, 0, 0, 0])),
                    })

        page_images.append(
            {
                "page_number":   page_num,
                "image_path":    str(img_path),
                "page_rotation": page.rotation,       # 0 / 90 / 180 / 270
                "spans":         page_spans,           # per-span detail (lazy image_bytes avoidance)
                "width":         pix.width,
                "height":        pix.height,
                "page_rect":     [page.rect.x0, page.rect.y0, page.rect.x1, page.rect.y1],
            }
        )

        # Extract text blocks & page text
        page_text = page.get_text("text")
        full_text_parts.append(f"--- PAGE {page_num} ---\n{page_text}")

        for block in page.get_text("blocks"):
            block_text = block[4].strip() if len(block) > 4 else ""
            if block_text:
                # Include span detail from the already-walked page_dict
                block_bbox = [block[0], block[1], block[2], block[3]]
                block_spans = [
                    sp for sp in page_spans
                    if block_bbox[0] - 1 <= sp["bbox"][0]
                    and sp["bbox"][2] <= block_bbox[2] + 1
                    and block_bbox[1] - 1 <= sp["bbox"][1]
                    and sp["bbox"][3] <= block_bbox[3] + 1
                ]
                text_blocks.append({
                    "text":  block_text,
                    "page":  page_num,
                    "bbox":  block_bbox,
                    "spans": block_spans,   # per-span detail for title_block.py
                })

        for w in page.get_text("words"):
            # w: (x0, y0, x1, y1, "word", block_no, line_no, word_no)
            words.append({
                "bbox": [w[0], w[1], w[2], w[3]],
                "text": w[4],
                "page": page_num,
            })

    doc.close()

    return {
        "pdf_path":   str(path),
        "page_images": page_images,
        "full_text":  "\n\n".join(full_text_parts),
        "text_blocks": text_blocks,
        "words":      words,
    }


def get_image_bytes(page_info: dict) -> bytes:
    """
    Lazy image bytes loader. Reads the rendered PNG from disk on demand.
    Use this instead of page_info["image_bytes"] (which is no longer stored eagerly).

    Parameters
    ----------
    page_info:
        One entry from doc_data["page_images"].

    Returns
    -------
    bytes of the PNG image, or b'' if the path is missing.
    """
    img_path = page_info.get("image_path", "")
    if img_path:
        try:
            with open(img_path, "rb") as fh:
                return fh.read()
        except OSError:
            pass
    return b""
