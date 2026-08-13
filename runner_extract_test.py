"""
runner_extract_test.py
──────────────────────
CLI runner for Gemini Vision structural table extraction & element count agent.

Usage
-----
  python runner_extract_test.py --pdf "C:\\Users\\daksh\\OneDrive - MinoriLabs\\Construction_ai_assitant\\Foundational_diagram.pdf"

  # Optional: specify output CSV path
  python runner_extract_test.py \\
    --pdf path/to/drawing.pdf \\
    --csv path/to/output.csv \\
    --env .env

Output
------
  Prints an 8-column formatted table (Segment | PDF Link | Image Link | Type | Span | Width | Height | QTY) to stdout.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from dotenv import load_dotenv

from ingestion.pdf_vision_loader import load_pdf_for_vision
from agents.field_extraction_agent import extract_table_from_doc


def _print_results_table(table_rows: list[dict]) -> None:
    """Print extracted rows in a formatted tabular grid."""
    if not table_rows:
        print("[Notice] No structural table rows extracted.")
        return

    try:
        from tabulate import tabulate

        headers = ["Segment", "PDF Link", "Image Link", "Type", "Span (mm)", "Width (mm)", "Height (mm)", "QTY", "Reinforcement", "Sample Grid References"]
        rows = [
            [
                r.get("Segment", ""),
                r.get("PDF Link", ""),
                r.get("Image Link", ""),
                r.get("Type", ""),
                r.get("Span (mm)", ""),
                r.get("Width (mm)", ""),
                r.get("Height (mm)", ""),
                r.get("QTY", 0),
                r.get("Reinforcement", ""),
                r.get("Sample Grid References", ""),
            ]
            for r in table_rows
        ]
        print("\n" + tabulate(rows, headers=headers, tablefmt="grid") + "\n")
    except ImportError:
        # Simple fallback text table formatting
        headers = ["Segment", "PDF Link", "Image Link", "Type", "Span (mm)", "Width (mm)", "Height (mm)", "QTY", "Reinforcement", "Sample Grid References"]
        print("\t".join(headers))
        for r in table_rows:
            print(
                f"{r.get('Segment')}\t{r.get('PDF Link')}\t{r.get('Image Link')}\t"
                f"{r.get('Type')}\t{r.get('Span (mm)')}\t{r.get('Width (mm)')}\t"
                f"{r.get('Height (mm)')}\t{r.get('QTY')}\t{r.get('Reinforcement')}\t{r.get('Sample Grid References')}"
            )


def _save_csv(table_rows: list[dict], csv_path: Path) -> None:
    """Save extracted rows to a CSV file."""
    if not table_rows:
        return

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["Segment", "PDF Link", "Image Link", "Type", "Span (mm)", "Width (mm)", "Height (mm)", "QTY", "Reinforcement", "Sample Grid References"]

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in table_rows:
            writer.writerow(r)

    print(f"[csv] Saved table output to {csv_path.resolve()}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract structural table schedules & diagram counts using Gemini Vision API."
    )
    parser.add_argument(
        "--pdf",
        required=True,
        metavar="PATH",
        help="Path to the PDF diagram file to process.",
    )
    parser.add_argument(
        "--csv",
        metavar="PATH",
        help="Optional path to save CSV output table.",
    )
    parser.add_argument(
        "--env",
        default=".env",
        metavar="PATH",
        help="Path to .env file (default: .env in current directory).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Load environment variables
    env_path = Path(args.env)
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)
        print(f"[env] Loaded environment from {env_path.resolve()}")

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"ERROR: PDF file not found: {pdf_path.resolve()}", file=sys.stderr)
        sys.exit(1)

    print(f"\n[processing] PDF: {pdf_path.resolve()}")

    # Step 1: Vision PDF loading and image rendering
    print("Step 1/2 — Rendering PDF pages & extracting text context...")
    doc_data = load_pdf_for_vision(pdf_path)
    page_count = len(doc_data.get("page_images", []))
    print(f"       Rendered {page_count} page(s) to PNG images.")

    # Step 2: Gemini Vision Table & QTY Extraction
    print("Step 2/2 — Running Gemini Vision table extraction & element count agent...")
    table_rows = extract_table_from_doc(doc_data)

    # Print Table
    _print_results_table(table_rows)

    # Save CSV if requested
    if args.csv:
        _save_csv(table_rows, Path(args.csv))


if __name__ == "__main__":
    main()
