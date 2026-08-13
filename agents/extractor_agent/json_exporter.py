"""
agents/json_exporter.py
───────────────────────
Converter module for exporting extracted structural drawing table elements
into formatted JSON format with exactly the requested 8 fields:
  Segment, PDF Link, Image Link, Type, Span (mm), Width (mm), Height (mm), QTY
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

ALLOWED_FIELDS = (
    "Segment",
    "PDF Link",
    "Image Link",
    "Type",
    "Span (mm)",
    "Width (mm)",
    "Height (mm)",
    "QTY",
)


def convert_extraction_to_json_payload(
    table_rows: Sequence[dict[str, Any]],
    vector_schedules: dict[str, Any] | None = None,
    qc_report: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Build a clean JSON payload containing only the 8 specified fields per element row.

    Fields included:
      - Segment
      - PDF Link
      - Image Link
      - Type
      - Span (mm)
      - Width (mm)
      - Height (mm)
      - QTY
    """
    payload: list[dict[str, Any]] = []

    for row in table_rows:
        filtered_row = {field: row.get(field, "") for field in ALLOWED_FIELDS}
        payload.append(filtered_row)

    return payload


def convert_to_json_string(
    table_rows: Sequence[dict[str, Any]],
    vector_schedules: dict[str, Any] | None = None,
    qc_report: dict[str, Any] | None = None,
    indent: int = 2,
) -> str:
    """Convert extracted structural table data into a formatted JSON string."""
    payload = convert_extraction_to_json_payload(table_rows, vector_schedules, qc_report)
    return json.dumps(payload, indent=indent, ensure_ascii=False)


def save_json_file(
    table_rows: Sequence[dict[str, Any]],
    json_path: str | Path,
    vector_schedules: dict[str, Any] | None = None,
    qc_report: dict[str, Any] | None = None,
    indent: int = 2,
) -> Path:
    """Save extracted structural table data into a JSON file at json_path."""
    out_path = Path(json_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    json_str = convert_to_json_string(
        table_rows=table_rows,
        vector_schedules=vector_schedules,
        qc_report=qc_report,
        indent=indent,
    )

    out_path.write_text(json_str, encoding="utf-8")
    print(f"[json] Saved JSON output to {out_path}")
    return out_path
