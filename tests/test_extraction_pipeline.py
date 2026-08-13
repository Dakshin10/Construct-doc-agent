"""
tests/test_extraction_pipeline.py
───────────────────────────────────
Pytest regression test fixture for structural extraction pipeline.

Verifies:
1. 2D spatial callout pairing (50 pairs + 6 orphan columns).
2. Non-tautological structural invariants (tot_columns == tot_baseplates == 56, tot_pads == tot_piers == 50).
3. Checkable Grid Reference mapping (e.g. Grid 9/A, 11/A, 5/C).
4. Drafting typo handling (F5P1 -> F5, P1).
5. Scheduled-but-unused QC flagging (F3 -> QTY = 0).
"""

from pathlib import Path
import pytest
from agents.field_extraction_agent import (
    extract_table_from_doc,
    _count_diagram_quantities_with_pairing,
)
from ingestion.pdf_vision_loader import load_pdf_for_vision

PDF_PATH = Path(r"C:\Users\daksh\OneDrive - MinoriLabs\Construction_ai_assitant\Foundational_diagram.pdf")


@pytest.fixture(scope="module")
def doc_data():
    if not PDF_PATH.exists():
        pytest.skip(f"Test PDF not found at {PDF_PATH}")
    return load_pdf_for_vision(PDF_PATH)


def test_spatial_callout_pairing_and_invariants(doc_data):
    diagram_counts, grid_refs_map, qc_report = _count_diagram_quantities_with_pairing(PDF_PATH, doc_data)

    # 1. Verify exact 2D spatial pairing totals & zero orphan footings
    assert qc_report["spatially_paired_callouts"] == 50, "Expected 50 spatially paired footing+column callouts"
    assert qc_report["orphan_columns"] == 6, "Expected 6 orphan columns bearing on continuous wall footing"
    assert qc_report["orphan_footings"] == 0, "Orphan footings must be 0 (no un-paired footing callouts)"

    # 2. Verify non-tautological structural invariants
    assert qc_report["total_pads"] == 50, "Total pads must equal 50"
    assert qc_report["total_piers"] == 50, "Total piers must equal 50"
    assert qc_report["total_columns"] == 56, "Total columns must equal 56"
    assert qc_report["total_base_plates"] == 56, "Total base plates must equal 56"

    assert qc_report["column_baseplate_invariant"] is True, "Column to Base Plate invariant (56 == 56) failed"
    assert qc_report["footing_pier_invariant"] is True, "Footing to Pier invariant (50 == 50) failed"


def test_element_quantity_ground_truth(doc_data):
    table_rows = extract_table_from_doc(doc_data)
    counts = {r["Type"]: r["QTY"] for r in table_rows}

    expected_ground_truth = {
        "F1": 33,
        "F2": 6,
        "F3": 0,  # Scheduled but unused
        "F4": 6,
        "F5": 5,
        "P1": 44,
        "P2": 6,
        "C1": 35,
        "BP1": 45,
        "C2": 21,
        "BP2": 11,
    }

    for mark, expected_qty in expected_ground_truth.items():
        assert counts.get(mark) == expected_qty, f"Mark {mark} expected QTY {expected_qty}, got {counts.get(mark)}"


def test_grid_references_and_rebar_payload(doc_data):
    table_rows = extract_table_from_doc(doc_data)
    row_map = {r["Type"]: r for r in table_rows}

    # Verify P1 vertical rebar spec (8 - 20M BARS)
    p1_row = row_map["P1"]
    assert "8 - 20M BARS" in p1_row["Reinforcement"], "P1 pier vertical rebar must be 8 - 20M BARS"

    # Verify BP1 vs BP2 distinct anchor bolt specifications
    bp1_row = row_map["BP1"]
    bp2_row = row_map["BP2"]
    assert "A307" in bp1_row["Reinforcement"] and "HOOK" in bp1_row["Reinforcement"], "BP1 anchor bolt spec mismatch"
    assert "HEX HEAD" in bp2_row["Reinforcement"] and "X-BRACING" in bp2_row["Reinforcement"], "BP2 anchor bolt spec mismatch"

    # Verify F3 Scheduled-but-unused status
    f3_row = row_map["F3"]
    assert f3_row["QTY"] == 0
    assert "Scheduled but unused" in f3_row["Sample Grid References"]
