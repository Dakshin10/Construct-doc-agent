"""
tests/test_extraction_pipeline.py
───────────────────────────────────
Pytest regression tests for the structural extraction pipeline.

Updated to use the new extract_flat_rows() / FlatRow schema.
extract_table_from_doc() is still tested as a backward-compat shim.

Verifies:
1. 2D spatial callout pairing (50 pairs + 6 orphan columns).
2. Non-tautological structural invariants (columns==baseplates==56, pads==piers==50).
3. Pairing closure invariant.
4. Grid snap parameters derived at runtime.
5. F3 — scheduled but unused: qty is empty/None (not 0) in new schema.
6. QTY ground-truth for F1..BP2 in new FlatRow schema.
7. Backward-compat shim (extract_table_from_doc) still returns expected shape.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from agents.extractor_agent.field_extraction_agent import (
    extract_flat_rows,
    extract_table_from_doc,
    _count_diagram_quantities_with_pairing,
)
from agents.extractor_agent.pdf_vision_loader import load_pdf_for_vision
from agents.extractor_agent.csv_schema import FlatRow, CSV_HEADER

PDF_PATH = Path(r"C:\Users\daksh\OneDrive - MinoriLabs\Construction_ai_assitant\Foundational_diagram.pdf")


@pytest.fixture(scope="module")
def doc_data():
    if not PDF_PATH.exists():
        pytest.skip(f"Test PDF not found at {PDF_PATH}")
    return load_pdf_for_vision(PDF_PATH)


# ── Spatial pairing & invariants ──────────────────────────────────────────────

def test_spatial_callout_pairing_and_invariants(doc_data):
    diagram_counts, grid_refs_map, qc_report = _count_diagram_quantities_with_pairing(
        PDF_PATH, doc_data
    )

    # 1. Exact pairing totals
    assert qc_report["spatially_paired_callouts"] == 50, \
        "Expected 50 spatially paired footing+column callouts"
    assert qc_report["orphan_columns"] == 6, \
        "Expected 6 orphan columns bearing on continuous wall footing"

    # §0.1 FIX: orphan_footings now credited to final_counts, not silently dropped
    # On this drawing orphan_footings should still be 0 (all pair successfully)
    assert qc_report["orphan_footings"] == 0, \
        "Orphan footings must be 0 on this drawing"

    # 2. Independent structural invariants
    assert qc_report["total_pads"]        == 50
    assert qc_report["total_piers"]       == 50
    assert qc_report["total_columns"]     == 56
    assert qc_report["total_base_plates"] == 56
    assert qc_report["pairing_closure_ok"] is True

    # 3. Grid snap parameters derived at runtime (not hardcoded)
    assert qc_report["grid_snap_tol_x"] > 0.0
    assert qc_report["grid_snap_tol_y"] > 0.0


# ── FlatRow QTY ground truth ──────────────────────────────────────────────────

def test_element_qty_ground_truth_flat_rows(doc_data):
    """Verify QTY counts using the new extract_flat_rows() → FlatRow schema."""
    rows = extract_flat_rows(doc_data)

    # Deduplicate by (element_mark, drawing_element_id) to get one qty per element
    qty_by_mark: dict[str, int | None] = {}
    for row in rows:
        mark = row.element_mark
        if mark and mark not in qty_by_mark:
            qty_by_mark[mark] = row.qty

    expected = {
        "F1": 33,
        "F2": 6,
        # F3: §0.2 — scheduled but not on plan → qty is None/empty (NOT 0)
        "F4": 6,
        "F5": 5,
        "P1": 44,
        "P2": 6,
        "C1": 35,
        "BP1": 45,
        "C2": 21,
        "BP2": 11,
    }

    for mark, expected_qty in expected.items():
        actual = qty_by_mark.get(mark)
        assert actual == expected_qty, \
            f"Mark {mark}: expected qty={expected_qty}, got qty={actual}"


def test_f3_qty_is_empty_not_zero(doc_data):
    """
    §0.2: F3 is scheduled but never appears on the plan.
    In the new schema qty must be None (empty), never 0.
    0 means an actual measured zero; unknown means empty.
    """
    rows = extract_flat_rows(doc_data)
    f3_rows = [r for r in rows if r.element_mark == "F3"]

    if not f3_rows:
        pytest.skip("F3 not discovered (mark not in schedule or plan)")

    for r in f3_rows:
        assert r.qty is None, (
            f"F3 qty must be None (not on plan), got qty={r.qty!r}. "
            "Empty means 'not counted'; 0 means measured zero."
        )


def test_flatrow_schema_fields(doc_data):
    """Every extracted FlatRow must have all 36 CSV_HEADER fields in to_dict()."""
    rows = extract_flat_rows(doc_data)
    assert rows, "No rows extracted"
    for i, row in enumerate(rows[:5]):  # spot-check first 5
        d = row.to_dict()
        assert len(d) == 36, f"Row {i}: expected 36 fields, got {len(d)}"
        for col in CSV_HEADER:
            assert col in d, f"Row {i}: missing column '{col}'"


def test_flatrow_raw_text_always_present(doc_data):
    """Every row must have non-empty raw_text (audit trail)."""
    rows = extract_flat_rows(doc_data)
    for i, row in enumerate(rows):
        assert row.raw_text, (
            f"Row {i} (mark={row.element_mark}): raw_text is empty — audit trail broken"
        )


def test_flatrow_source_view_always_present(doc_data):
    """Every row must have a source_view."""
    rows = extract_flat_rows(doc_data)
    for i, row in enumerate(rows):
        assert row.source_view, (
            f"Row {i} (mark={row.element_mark}): source_view is empty"
        )


# ── Backward-compat shim ──────────────────────────────────────────────────────

def test_extract_table_from_doc_shim(doc_data):
    """
    extract_table_from_doc() shim must return non-empty list of dicts.
    Used by tools.py / database agent path via main.py.
    §0.3: Height (mm) must be numeric-only (no 'per section', 'wall thk.').
    """
    table_rows = extract_table_from_doc(doc_data)
    assert isinstance(table_rows, list)
    assert len(table_rows) > 0

    non_numeric_found = []
    for r in table_rows:
        h = str(r.get("Height (mm)", ""))
        # §0.3: must be empty or parseable as a number
        if h and not h.replace(".", "").isdigit():
            non_numeric_found.append((r.get("Type"), h))

    assert not non_numeric_found, (
        f"§0.3 REGRESSION: non-numeric Height (mm) values found: {non_numeric_found}"
    )
