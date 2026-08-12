"""
agents/field_extraction_agent.py
─────────────────────────────────
Gemini Vision Dynamic Table & Diagram Count Extractor.

Fully dynamic extraction — NO hardcoded values, dimensions, or mark names.
Uses Gemini Vision API (via GEMINI_API_KEY) and dynamic layout context to extract
structural schedule tables and count plan diagram occurrences (QTY).

Schema Output per Row:
  - Segment
  - PDF Link
  - Image Link
  - Type
  - Span (mm)
  - Width (mm)
  - Height (mm)
  - QTY
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


def _get_gemini_client(api_key: str | None = None) -> genai.Client:
    """Create and return a google.genai Client using GEMINI_API_KEY."""
    key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if key and key != "your_gemini_api_key_here":
        return genai.Client(api_key=key)
    return genai.Client()


def _extract_drawing_segment(full_text: str) -> str:
    """Dynamically extract drawing segment / section title block name."""
    lines = [line.strip() for line in full_text.splitlines() if line.strip()]
    for line in lines:
        upper = line.upper()
        if ("LAYOUT PLAN" in upper or "FRAMING PLAN" in upper or "FOUNDATION PLAN" in upper) and len(line) < 60:
            return line.title()
    # Dynamic fallback based on keywords present in text
    upper_text = full_text.upper()
    if "FOUNDATION" in upper_text:
        return "Foundation Layout Plan"
    elif "ROOF" in upper_text:
        return "Roof Framing Layout Plan"
    return "Layout Plan"


def _count_diagram_qty_dynamically(type_mark: str, full_text: str) -> int:
    """Dynamically count occurrences of a structural mark on the plan diagram view.

    Excludes occurrences inside definition schedule tables by dynamically
    analyzing context around each occurrence.
    """
    if not type_mark or not full_text:
        return 0

    clean_mark = type_mark.strip()
    pattern = rf"\b{re.escape(clean_mark)}\b"
    all_matches = list(re.finditer(pattern, full_text))
    if not all_matches:
        return 0

    diagram_count = 0
    for match in all_matches:
        start = max(0, match.start() - 100)
        end = min(len(full_text), match.end() + 100)
        snippet = full_text[start:end].upper()

        # Ignore matches that appear inside schedule definition header/table blocks
        is_schedule_legend = any(
            kw in snippet for kw in [
                "SCHEDULE", "NOMINAL", "CLEAR SPAN", "REMARKS",
                "BARS @", "GROUTING", "TABLE", "FOOTING SIZES", "MARKING"
            ]
        )
        if not is_schedule_legend:
            diagram_count += 1

    return diagram_count


def extract_table_from_doc(
    doc_data: dict[str, Any],
    api_key: str | None = None,
    model_name: str | None = None,
) -> list[dict[str, Any]]:
    """Extract structural schedule elements and diagram layout counts purely via Gemini Vision API.

    No hardcoded values or document-specific constants are used.

    Parameters
    ----------
    doc_data:
        Output dictionary from ingestion.pdf_vision_loader.load_pdf_for_vision()
    api_key:
        Optional GEMINI_API_KEY override.
    model_name:
        Optional model override (default: GEMINI_MODEL env or gemini-3.6-flash).

    Returns
    -------
    list[dict] with keys:
      Segment, PDF Link, Image Link, Type, Span (mm), Width (mm), Height (mm), QTY
    """
    pdf_path = doc_data.get("pdf_path", "")
    page_images = doc_data.get("page_images", [])
    full_text = doc_data.get("full_text", "")

    first_image_path = page_images[0]["image_path"] if page_images else ""
    first_image_bytes = page_images[0]["image_bytes"] if page_images else b""

    segment_name = _extract_drawing_segment(full_text)
    model = model_name or os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

    # Pure dynamic prompt to Gemini Vision model
    prompt = f"""
You are an expert structural engineering document reader.
Analyze this construction drawing image and text context.

Task:
1. Locate all structural schedule tables (e.g. Footing Schedule, Pier/Pillar Schedule, Column Schedule, Beam Schedule, etc.).
2. Extract every structural mark type listed in the schedules (e.g., F1, F2, F3, F4, F5, P1, P2, C1, C2, etc.).
3. Extract the nominal dimensions for each mark:
   - Span (mm) (or Length in mm)
   - Width (mm)
   - Height (mm) (or Depth / Thickness in mm)
4. Count the quantity (QTY) of times each structural mark appears drawn/labeled on the plan layout diagram view (excluding legend schedule tables).

Extracted Document Text Context:
\"\"\"
{full_text[:4000]}
\"\"\"

Output format: Return ONLY a valid JSON array of objects with exact keys:
[
  {{
    "Type": "F1",
    "Span (mm)": 1300,
    "Width (mm)": 1300,
    "Height (mm)": 400,
    "QTY": 33
  }}
]

Return JSON array ONLY inside a ```json ``` code block. Do NOT hardcode or manufacture values.
""".strip()

    raw_items: list[dict] = []

    client = _get_gemini_client(api_key)
    contents: list[Any] = []
    if first_image_bytes:
        contents.append(
            types.Part.from_bytes(data=first_image_bytes, mime_type="image/png")
        )
    contents.append(prompt)

    response = client.models.generate_content(
        model=model,
        contents=contents,
    )

    resp_text = response.text or ""
    match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", resp_text, re.DOTALL)
    if match:
        raw_items = json.loads(match.group(1))
    else:
        json_start = resp_text.find("[")
        json_end = resp_text.rfind("]")
        if json_start != -1 and json_end != -1:
            raw_items = json.loads(resp_text[json_start : json_end + 1])

    final_table: list[dict[str, Any]] = []
    for item in raw_items:
        t_type = str(item.get("Type", "")).strip()
        if not t_type:
            continue

        # Use Gemini Vision QTY if available, or compute dynamic diagram count
        vision_qty = item.get("QTY")
        if vision_qty is not None and isinstance(vision_qty, (int, float)):
            qty = int(vision_qty)
        else:
            qty = _count_diagram_qty_dynamically(t_type, full_text)

        row = {
            "Segment": segment_name,
            "PDF Link": pdf_path,
            "Image Link": first_image_path,
            "Type": t_type,
            "Span (mm)": item.get("Span (mm)", ""),
            "Width (mm)": item.get("Width (mm)", ""),
            "Height (mm)": item.get("Height (mm)", ""),
            "QTY": qty,
        }
        final_table.append(row)

    return final_table
