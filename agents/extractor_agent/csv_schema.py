"""
agents/extractor_agent/csv_schema.py
─────────────────────────────────────
Single source of truth for the 36-column flat CSV output contract.

Four logical tables denormalised into one row grain:
  projectDetails → drawing_input_master → drawing_elements → reinforcement_details

One row per (drawing_element × reinforcement_layer).
An element with no reinforcement emits exactly 1 row with all reinf_* columns empty.
Element-level columns repeat on every layer row — downstream consumers group by drawing_element_id.

NEVER re-type CSV_HEADER anywhere else. Import it from here.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, fields
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ── StrEnum-compatible base (Python 3.9+ compat) ─────────────────────────────
class _StrEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class LayerCode(_StrEnum):
    BLL        = "BLL"        # bottom-lower layer
    BUL        = "BUL"        # bottom-upper layer
    TLL        = "TLL"        # top-lower layer
    TUL        = "TUL"        # top-upper layer
    VERT       = "VERT"       # vertical bars
    TIES       = "TIES"       # ties / stirrups
    CONT_TOP   = "CONT_TOP"   # continuous bars at top
    CONT_BOT   = "CONT_BOT"   # continuous bars at bottom
    MESH       = "MESH"       # welded wire mesh
    DOWEL      = "DOWEL"      # dowels
    UNSPECIFIED = "UNSPECIFIED"


class Direction(_StrEnum):
    EW           = "EW"
    LONGITUDINAL = "LONGITUDINAL"
    TRANSVERSE   = "TRANSVERSE"
    VERTICAL     = "VERTICAL"
    HORIZONTAL   = "HORIZONTAL"
    DIAGONAL     = "DIAGONAL"
    UNSPECIFIED  = "UNSPECIFIED"


class SourceView(_StrEnum):
    SCHEDULE     = "SCHEDULE"
    PLAN         = "PLAN"
    SECTION      = "SECTION"
    DETAIL       = "DETAIL"
    GENERAL_NOTE = "GENERAL_NOTE"


class ExtractionMethod(_StrEnum):
    VECTOR_TEXT   = "VECTOR_TEXT"
    TABLE_PARSE   = "TABLE_PARSE"
    VISION        = "VISION"
    VISION_FALLBACK = "VISION_FALLBACK"


class DrawingType(_StrEnum):
    PLAN      = "PLAN"
    SECTION   = "SECTION"
    ELEVATION = "ELEVATION"
    SCHEDULE  = "SCHEDULE"
    DETAIL    = "DETAIL"
    MIXED     = "MIXED"


class JobType(_StrEnum):
    NEW_CONSTRUCTION = "NEW_CONSTRUCTION"
    ADDITION         = "ADDITION"
    RENOVATION       = "RENOVATION"
    UNKNOWN          = "UNKNOWN"


# ── Headers ───────────────────────────────────────────────────────────────────
# 36-column flat header for JSON / internal table
CSV_HEADER: tuple[str, ...] = (
    "project_number_id", "project_name", "architect_name", "engineer_name",
    "customer_name", "job_type", "estimator_name", "current_status",
    "drawing_id", "drawing_number", "drawing_title", "drawing_type_name",
    "drawing_description", "drawing_date", "document_location",
    "drawing_element_id", "element_mark", "element_name", "element_shape",
    "span_mm", "width_mm", "height_mm", "qty",
    "reinforcement_id", "layer_code", "bar_size", "bar_count", "spacing_mm",
    "direction", "bar_length_mm", "cover_mm",
    "source_view", "source_page", "extraction_method", "confidence", "raw_text",
)

assert len(CSV_HEADER) == 36, f"CSV_HEADER must have exactly 36 fields, got {len(CSV_HEADER)}"

# 22-column element-level header for slimmed CSV output (ONE ROW PER ELEMENT)
ELEMENT_CSV_HEADER: tuple[str, ...] = (
    "project_number_id", "project_name", "architect_name", "engineer_name",
    "customer_name", "job_type", "estimator_name", "current_status",
    "drawing_id", "drawing_number", "drawing_title", "drawing_type_name",
    "drawing_description", "document_location",
    "drawing_element_id", "element_mark", "element_name", "element_shape",
    "span_mm", "width_mm", "height_mm", "qty",
)

assert len(ELEMENT_CSV_HEADER) == 22, f"ELEMENT_CSV_HEADER must have exactly 22 fields, got {len(ELEMENT_CSV_HEADER)}"

# Validation helpers
_MM_FIELDS = frozenset({"span_mm", "width_mm", "height_mm", "bar_count",
                         "spacing_mm", "bar_length_mm", "cover_mm"})
_BAR_SIZE_PAT  = re.compile(r"^(10|15|20|25|30|35|45|55)M$")
_MESH_SIZE_PAT = re.compile(r"^\d+x\d+-MW[\d.]+/MW[\d.]+$")


# ── FlatRow dataclass ─────────────────────────────────────────────────────────
@dataclass
class FlatRow:
    """
    One output row. None → '' on export via to_dict().
    Field order matches CSV_HEADER exactly.
    """
    # projectDetails
    project_number_id: Optional[str]   = None
    project_name:      Optional[str]   = None
    architect_name:    Optional[str]   = None
    engineer_name:     Optional[str]   = None
    customer_name:     Optional[str]   = None
    job_type:          Optional[str]   = None
    estimator_name:    Optional[str]   = None
    current_status:    Optional[str]   = None
    # drawing_input_master
    drawing_id:          Optional[str] = None
    drawing_number:      Optional[str] = None
    drawing_title:       Optional[str] = None
    drawing_type_name:   Optional[str] = None
    drawing_description: Optional[str] = None
    drawing_date:        Optional[str] = None
    document_location:   Optional[str] = None
    # drawing_elements
    drawing_element_id: Optional[str] = None
    element_mark:       Optional[str] = None
    element_name:       Optional[str] = None
    element_shape:      Optional[str] = None
    span_mm:            Optional[int] = None
    width_mm:           Optional[int] = None
    height_mm:          Optional[int] = None
    qty:                Optional[int] = None
    # reinforcement_details
    reinforcement_id: Optional[str] = None
    layer_code:       Optional[str] = None
    bar_size:         Optional[str] = None
    bar_count:        Optional[int] = None
    spacing_mm:       Optional[int] = None
    direction:        Optional[str] = None
    bar_length_mm:    Optional[int] = None
    cover_mm:         Optional[int] = None
    # provenance
    source_view:       Optional[str]   = None
    source_page:       Optional[int]   = None
    extraction_method: Optional[str]   = None
    confidence:        Optional[float] = None
    raw_text:          Optional[str]   = None

    def to_dict(self) -> dict[str, str]:
        """Render to 36-column JSON/full dict. None → '', confidence formatted to 2dp."""
        result: dict[str, str] = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if val is None:
                result[f.name] = ""
            elif f.name == "confidence" and isinstance(val, float):
                result[f.name] = f"{val:.2f}"
            else:
                result[f.name] = str(val)
        return result

    def to_element_dict(self) -> dict[str, str]:
        """Render 22 element-level fields to CSV-ready dict."""
        result: dict[str, str] = {}
        for col in ELEMENT_CSV_HEADER:
            val = getattr(self, col, None)
            if val is None:
                result[col] = ""
            else:
                result[col] = str(val)
        return result


# ── Validators ────────────────────────────────────────────────────────────────
def validate_rows(
    rows: list[FlatRow],
    doc_data: dict[str, Any] | None = None,
    log_path: Path | None = None,
) -> list[str]:
    """
    Run §7 validations on the full list of FlatRow objects.
    Returns list of warning strings.
    Writes all warnings to log_path (default: extraction_warnings.log in cwd).
    Does NOT silently correct anything.
    """
    warnings_out: list[str] = []
    log_path = log_path or Path("extraction_warnings.log")

    def warn(msg: str) -> None:
        warnings_out.append(msg)
        logger.warning(msg)

    # 1. Field count per row
    for i, row in enumerate(rows):
        d = row.to_dict()
        if len(d) != len(CSV_HEADER):
            warn(f"[V1] Row {i}: field count {len(d)} != {len(CSV_HEADER)}")
        for col in CSV_HEADER:
            if col not in d:
                warn(f"[V1] Row {i}: missing column '{col}'")

    # 2a. drawing_element_id unique per (element_mark, drawing_id)
    elem_id_map: dict[tuple, str] = {}
    for i, row in enumerate(rows):
        if row.drawing_element_id and row.element_mark and row.drawing_id:
            key = (row.element_mark, row.drawing_id)
            seen = elem_id_map.get(key)
            if seen and seen != row.drawing_element_id:
                warn(f"[V2a] Row {i}: drawing_element_id conflict for {key}: "
                     f"'{row.drawing_element_id}' vs '{seen}'")
            elem_id_map[key] = row.drawing_element_id

    # 2b. reinforcement_id globally unique
    reinf_seen: set[str] = set()
    for i, row in enumerate(rows):
        if row.reinforcement_id:
            if row.reinforcement_id in reinf_seen:
                warn(f"[V2b] Row {i}: duplicate reinforcement_id '{row.reinforcement_id}'")
            reinf_seen.add(row.reinforcement_id)

    # 3. Every reinforcement_id prefix resolves to a known drawing_element_id
    all_elem_ids = {row.drawing_element_id for row in rows if row.drawing_element_id}
    for i, row in enumerate(rows):
        if row.reinforcement_id:
            prefix = re.sub(r"-R\d{2}$", "", row.reinforcement_id)
            if prefix not in all_elem_ids:
                warn(f"[V3] Row {i}: reinforcement_id '{row.reinforcement_id}' "
                     f"prefix '{prefix}' not in drawing_element_ids")

    # 4. bar_size format
    for i, row in enumerate(rows):
        if row.bar_size:
            if not (_BAR_SIZE_PAT.match(row.bar_size) or _MESH_SIZE_PAT.match(row.bar_size)):
                warn(f"[V4] Row {i} (mark={row.element_mark}): "
                     f"bar_size '{row.bar_size}' not a valid CSA or mesh designation")

    # 5. bar_count XOR spacing_mm on non-empty reinforcement rows
    for i, row in enumerate(rows):
        has_count   = row.bar_count   is not None
        has_spacing = row.spacing_mm  is not None
        if (row.layer_code or row.bar_size) and has_count and has_spacing:
            warn(f"[V5] Row {i} (mark={row.element_mark}): both bar_count={row.bar_count} "
                 f"and spacing_mm={row.spacing_mm} set — must be mutually exclusive")

    # 6. *_mm positive integers 1..200000
    for i, row in enumerate(rows):
        for col in _MM_FIELDS:
            val = getattr(row, col)
            if val is not None:
                if not isinstance(val, int) or not (1 <= val <= 200_000):
                    warn(f"[V6] Row {i} (mark={row.element_mark}): "
                         f"{col}={val!r} not a positive int in 1..200000")

    # 7. qty positive int or None — never 0, never float
    for i, row in enumerate(rows):
        if row.qty is not None:
            if not isinstance(row.qty, int) or row.qty <= 0:
                warn(f"[V7] Row {i} (mark={row.element_mark}): "
                     f"qty={row.qty!r} must be a positive integer or empty")

    # 8. Every row has raw_text and source_view
    for i, row in enumerate(rows):
        if not row.raw_text:
            warn(f"[V8] Row {i} (mark={row.element_mark}): raw_text is empty — audit trail broken")
        if not row.source_view:
            warn(f"[V8] Row {i} (mark={row.element_mark}): source_view is empty")

    # 9. Per-mark qty totals (deduplicate by drawing_element_id before summing)
    qty_by_elem: dict[str, int] = {}
    for row in rows:
        if row.drawing_element_id and row.qty is not None:
            qty_by_elem[row.drawing_element_id] = row.qty
    if qty_by_elem:
        qty_summary = "\n".join(
            f"  {eid}: qty={qty}" for eid, qty in sorted(qty_by_elem.items())
        )
        warnings_out.append(f"[QTY TOTALS — verify against manual count]\n{qty_summary}")

    # 10. Count rows at confidence < 0.60
    low_conf = sum(1 for r in rows if r.confidence is not None and r.confidence < 0.60)
    if low_conf:
        warn(f"[V10] {low_conf} row(s) have confidence < 0.60 — "
             "consider Vision fallback for low-quality regions")

    # 11. Page Text Layer Traceability — every populated numeric field MUST appear in raw_text or page text layer
    page_text = ""
    if doc_data and "text_blocks" in doc_data:
        page_text = " ".join(b.get("text", "") for b in doc_data["text_blocks"]).upper()

    for i, row in enumerate(rows):
        raw = (row.raw_text or "").upper()
        search_ctx = f"{raw} {page_text}" if page_text else raw
        for col in _MM_FIELDS:
            val = getattr(row, col)
            if val is not None:
                val_str = str(val)
                pat = re.compile(r"(?:^|[^\d])" + re.escape(val_str) + r"(?:[^\d]|$)")
                if not pat.search(search_ctx):
                    warn(f"[V11] Row {i} (mark={row.element_mark}): "
                         f"{col}={val} not found in text layer context — clearing field")
                    setattr(row, col, None)

    # 12. F5 Generated Prose Check — raw_text must NOT contain generated prose
    _GEN_PROSE_PAT = re.compile(r"\(scheduled|per plan|dimensions per", re.IGNORECASE)
    for i, row in enumerate(rows):
        if row.raw_text and _GEN_PROSE_PAT.search(row.raw_text):
            warn(f"[V12] Row {i} (mark={row.element_mark}): raw_text contains generated prose '{row.raw_text}' — resetting to element mark")
            row.raw_text = row.element_mark or ""

    # Write log file
    if warnings_out:
        try:
            with open(log_path, "w", encoding="utf-8", newline="\n") as fh:
                fh.write("\n".join(warnings_out) + "\n")
        except OSError as e:
            logger.error("Could not write %s: %s", log_path, e)

    return warnings_out
