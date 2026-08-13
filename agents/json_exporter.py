"""
agents/json_exporter.py
───────────────────────
Separate converter module for exporting extracted structural drawing elements,
dataclass payloads (ElementTypeAttr, RebarLayer), grid references, and QC report
into formatted JSON output.
"""

from __future__ import annotations

import json
from dataclasses import is_dataclass, asdict
from pathlib import Path
from typing import Any, Sequence


def _serialize_dataclass(obj: Any) -> Any:
    """Recursively convert dataclass objects to JSON-serializable dicts."""
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, list):
        return [_serialize_dataclass(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _serialize_dataclass(v) for k, v in obj.items()}
    return obj


def convert_extraction_to_json_payload(
    table_rows: Sequence[dict[str, Any]],
    vector_schedules: dict[str, Any] | None = None,
    qc_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build a comprehensive, structured JSON payload from table rows, vector schedule attributes,
    and quality control report data.

    Parameters
    ----------
    table_rows:
        List of dicts returned from extract_table_from_doc()
    vector_schedules:
        Optional mapping of mark -> ElementTypeAttr objects or dicts
    qc_report:
        Optional quality control report dictionary

    Returns
    -------
    dict representing the root JSON structure
    """
    elements_list: list[dict[str, Any]] = []

    for row in table_rows:
        mark = row.get("Type", "")
        elem_payload: dict[str, Any] = {
            "type": mark,
            "quantity": row.get("QTY", 0),
            "span_mm": row.get("Span (mm)"),
            "width_mm": row.get("Width (mm)"),
            "height_mm": row.get("Height (mm)"),
            "reinforcement": row.get("Reinforcement", ""),
            "sample_grid_references": row.get("Sample Grid References", ""),
        }

        # Embed detailed dataclass attributes if available
        if vector_schedules and mark in vector_schedules:
            attr_obj = vector_schedules[mark]
            attr_dict = _serialize_dataclass(attr_obj)
            if isinstance(attr_dict, dict):
                elem_payload["structured_attributes"] = attr_dict

        elements_list.append(elem_payload)

    # Extract high-level metadata from first row if present
    first_row = table_rows[0] if table_rows else {}
    metadata = {
        "segment": first_row.get("Segment", ""),
        "pdf_path": first_row.get("PDF Link", ""),
        "image_path": first_row.get("Image Link", ""),
        "total_elements_extracted": len(elements_list),
    }

    payload = {
        "metadata": metadata,
        "elements": elements_list,
    }

    if qc_report:
        payload["qc_report"] = _serialize_dataclass(qc_report)

    return payload


def convert_to_json_string(
    table_rows: Sequence[dict[str, Any]],
    vector_schedules: dict[str, Any] | None = None,
    qc_report: dict[str, Any] | None = None,
    indent: int = 2,
) -> str:
    """Convert extracted structural data into a formatted JSON string."""
    payload = convert_extraction_to_json_payload(table_rows, vector_schedules, qc_report)
    return json.dumps(payload, indent=indent, ensure_ascii=False)


def save_json_file(
    table_rows: Sequence[dict[str, Any]],
    json_path: str | Path,
    vector_schedules: dict[str, Any] | None = None,
    qc_report: dict[str, Any] | None = None,
    indent: int = 2,
) -> Path:
    """Save extracted structural data into a JSON file at json_path."""
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
