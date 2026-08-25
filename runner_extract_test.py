"""
runner_extract_test.py
──────────────────────
CLI runner for flat-CSV structural extraction.

Produces:
  output_table.csv  — 36-column normalised flat CSV (always written)
  output_table.json — JSON mirror of the same rows (always written)
  extraction_warnings.log — validation warnings

Does NOT invoke the database agent. send_to_database_agent is wired in
agents/extractor_agent/agent.py but is never imported or reached here.

Usage
-----
  python runner_extract_test.py --pdf "C:\\path\\to\\drawing.pdf"

  python runner_extract_test.py \\
    --pdf path/to/drawing.pdf \\
    --csv path/to/output.csv  \\
    --json path/to/output.json \\
    --env .env
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import os
from dotenv import load_dotenv
load_dotenv()
if os.environ.get("GEMINI_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

# ── IMPORTANT: never import from agents/database_agent/ here ─────────────────
from agents.extractor_agent.pdf_vision_loader import load_pdf_for_vision
from agents.extractor_agent.field_extraction_agent import extract_flat_rows
from agents.extractor_agent.csv_schema import CSV_HEADER, FlatRow
from agents.extractor_agent.json_exporter import write_flat_csv, write_flat_json


# ── Console summary table ─────────────────────────────────────────────────────

def _print_summary(rows: list[FlatRow]) -> None:
    """Print a compact terminal summary of extracted elements."""
    if not rows:
        print("[Notice] No rows extracted.")
        return

    # Deduplicate by drawing_element_id for the element-level summary
    seen_elements: set[str] = set()
    elem_rows: list[FlatRow] = []
    for r in rows:
        eid = r.drawing_element_id or ""
        if eid and eid not in seen_elements:
            seen_elements.add(eid)
            elem_rows.append(r)

    print(f"\n{'-'*72}")
    print(f"  Extraction Summary  --  {len(rows)} row(s) / {len(elem_rows)} element(s)")
    print(f"{'-'*72}")
    header = f"  {'Mark':<8}  {'Name':<26}  {'L':<6}  {'W':<6}  {'H':<6}  {'QTY':<5}  {'Conf':>5}"
    print(header)
    print(f"  {'-'*67}")

    for r in elem_rows:
        mark  = (r.element_mark  or "")[:8]
        name  = (r.element_name  or "")[:26]
        span  = str(r.span_mm   or "")[:6]
        width = str(r.width_mm  or "")[:6]
        hgt   = str(r.height_mm or "")[:6]
        qty   = str(r.qty       or "")[:5]
        conf  = f"{r.confidence:.2f}" if r.confidence is not None else ""
        print(f"  {mark:<8}  {name:<26}  {span:<6}  {width:<6}  {hgt:<6}  {qty:<5}  {conf:>5}")

    # Reinforcement layer summary
    reinf_count = sum(1 for r in rows if r.layer_code)
    print(f"{'-'*72}")
    print(f"  Reinforcement layers extracted: {reinf_count}")
    low_conf = sum(1 for r in rows if r.confidence is not None and r.confidence < 0.60)
    if low_conf:
        print(f"  !  {low_conf} row(s) below confidence 0.60 — see extraction_warnings.log")
    print(f"{'-'*72}\n")


# ── Argument parsing ──────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract structural drawing schedules to flat 36-column CSV. "
            "Does NOT invoke the database agent."
        )
    )
    parser.add_argument(
        "--pdf",
        required=True,
        metavar="PATH",
        help="Path to the structural drawing PDF file.",
    )
    parser.add_argument(
        "--csv",
        nargs="?",
        const="output_table.csv",
        default="output_table.csv",
        metavar="PATH",
        help="Output CSV path (default: output_table.csv).",
    )
    parser.add_argument(
        "--json",
        nargs="?",
        const="output_table.json",
        default="output_table.json",
        metavar="PATH",
        help="Output JSON path (default: output_table.json).",
    )
    parser.add_argument(
        "--env",
        default=".env",
        metavar="PATH",
        help="Path to .env file (default: .env in current directory).",
    )
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    env_path = Path(args.env)
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)
        print(f"[env] Loaded environment from {env_path.resolve()}")

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"ERROR: PDF file not found: {pdf_path.resolve()}", file=sys.stderr)
        sys.exit(1)

    print(f"\n[processing] PDF: {pdf_path.resolve()}")
    print(f"             Output CSV  -> {Path(args.csv).resolve()}")
    print(f"             Output JSON -> {Path(args.json).resolve()}")

    # Step 1: Render PDF + extract text
    print("\nStep 1/2 — Rendering PDF pages & extracting text context...")
    doc_data  = load_pdf_for_vision(pdf_path)
    page_count = len(doc_data.get("page_images", []))
    print(f"           Rendered {page_count} page(s).")

    # Step 2: Extract flat rows (no database agent involved)
    print("Step 2/2 — Running extraction pipeline...")
    rows = extract_flat_rows(doc_data)
    print(f"           Extracted {len(rows)} row(s).")

    # Print summary
    _print_summary(rows)

    # Always write CSV and JSON
    csv_path  = Path(args.csv)
    json_path = Path(args.json)
    write_flat_csv(rows,  csv_path)
    write_flat_json(rows, json_path)

    warn_log = Path("extraction_warnings.log")
    if warn_log.exists() and warn_log.stat().st_size > 0:
        print(f"\n[warnings] See {warn_log.resolve()} for validation issues.")


if __name__ == "__main__":
    main()
