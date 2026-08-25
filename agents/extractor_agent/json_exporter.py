"""
agents/extractor_agent/json_exporter.py
─────────────────────────────────────────
Flat CSV and JSON export for the normalised 36-column schema.

Primary functions (new schema):
  write_flat_csv(rows, path)   — writes output_table.csv
  write_flat_json(rows, path)  — writes output_table.json

Compatibility shim (old schema — used by tools.py / database agent path):
  save_json_file(table_rows, json_path, ...)

IMPORTANT: CSV_HEADER is imported from csv_schema — never re-typed here.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Sequence

from agents.extractor_agent.csv_schema import CSV_HEADER, ELEMENT_CSV_HEADER, FlatRow

logger = logging.getLogger(__name__)


# ── Flat CSV / JSON writers (new schema) ─────────────────────────────────────

def write_flat_csv(rows: list[FlatRow], path: Path | str) -> Path:
    """
    Write a list of FlatRow objects to an element-level flat CSV file (ONE ROW PER ELEMENT).
    Uses 22-column ELEMENT_CSV_HEADER schema.

    Encoding: UTF-8, \n line endings, RFC4180 quoting.
    Handles file locking gracefully by retrying or falling back if open in Excel/editor.
    """
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    seen_elem_ids: set[str] = set()
    element_rows: list[FlatRow] = []
    for r in rows:
        eid = r.drawing_element_id or r.element_mark
        if eid and eid not in seen_elem_ids:
            seen_elem_ids.add(eid)
            element_rows.append(r)

    target_path = out
    for attempt in range(3):
        try:
            with open(target_path, "w", encoding="utf-8", newline="\n") as fh:
                writer = csv.DictWriter(
                    fh,
                    fieldnames=ELEMENT_CSV_HEADER,
                    extrasaction="raise",   # schema drift → immediate error
                    quoting=csv.QUOTE_MINIMAL,
                    lineterminator="\n",
                )
                writer.writeheader()
                for row in element_rows:
                    writer.writerow(row.to_element_dict())
            logger.info("[csv] Wrote %d element row(s) to %s", len(element_rows), target_path)
            print(f"[csv] Saved flat CSV ({len(element_rows)} element rows, {len(ELEMENT_CSV_HEADER)} cols) -> {target_path}")
            return target_path
        except PermissionError:
            time.sleep(0.5)

    # If target file is locked by user's editor or Excel, write to unlocked fallback
    fallback_path = out.with_name(out.stem + "_output" + out.suffix)
    with open(fallback_path, "w", encoding="utf-8", newline="\n") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=ELEMENT_CSV_HEADER,
            extrasaction="raise",
            quoting=csv.QUOTE_MINIMAL,
            lineterminator="\n",
        )
        writer.writeheader()
        for row in element_rows:
            writer.writerow(row.to_element_dict())
    print(f"[csv] Note: '{out.name}' was locked by an active editor. Saved element CSV to -> {fallback_path}")
    return fallback_path


def write_flat_json(rows: list[FlatRow], path: Path | str) -> Path:
    """
    Write a list of FlatRow objects to a JSON file.
    Encoding: UTF-8, \n line endings, indent=2.
    """
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    target_path = out
    for attempt in range(3):
        try:
            with open(target_path, "w", encoding="utf-8", newline="\n") as fh:
                json.dump([r.to_dict() for r in rows], fh, indent=2, ensure_ascii=False)
                fh.write("\n")
            logger.info("[json] Wrote %d rows to %s", len(rows), target_path)
            print(f"[json] Saved flat JSON ({len(rows)} rows) -> {target_path}")
            return target_path
        except PermissionError:
            time.sleep(0.5)

    fallback_path = out.with_name(out.stem + "_output" + out.suffix)
    with open(fallback_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump([r.to_dict() for r in rows], fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print(f"[json] Note: '{out.name}' was locked by an active editor. Saved flat JSON to -> {fallback_path}")
    return fallback_path


# ── Compatibility shim (old schema — tools.py / database agent path) ─────────

# These fields are what the old pipeline expected
_OLD_FIELDS = (
    "Type", "Category", "Length (mm)", "Width (mm)", "Depth (mm)",
    "Primary reinforcement", "Secondary reinforcement / notes",
    "Plan instances", "Segment", "PDF Link", "Image Link", "QTY",
)


def convert_extraction_to_json_payload(
    table_rows: Sequence[dict[str, Any]],
    vector_schedules: Any = None,
    qc_report: Any = None,
) -> list[dict[str, Any]]:
    """Build the old-schema JSON payload (for database agent)."""
    return [{field: row.get(field, "") for field in _OLD_FIELDS} for row in table_rows]


def convert_to_json_string(
    table_rows: Sequence[dict[str, Any]],
    vector_schedules: Any = None,
    qc_report: Any = None,
    indent: int = 2,
) -> str:
    payload = convert_extraction_to_json_payload(table_rows, vector_schedules, qc_report)
    return json.dumps(payload, indent=indent, ensure_ascii=False)


def save_json_file(
    table_rows: Sequence[dict[str, Any]],
    json_path:  str | Path,
    vector_schedules: Any = None,
    qc_report:  Any = None,
    indent: int = 2,
) -> Path:
    """Compatibility shim: save old-schema table_rows to JSON."""
    out = Path(json_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    json_str = convert_to_json_string(table_rows, vector_schedules, qc_report, indent)
    out.write_text(json_str, encoding="utf-8")
    print(f"[json] Saved JSON output to {out}")
    return out
