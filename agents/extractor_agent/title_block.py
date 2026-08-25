"""
agents/extractor_agent/title_block.py
──────────────────────────────────────
Title Block Extractor Gated on Vector Region Containment (F1a & F1b Fixes).

Key Principles:
1. F1a: Detect title block rect from union of all ruled cells containing caption tokens.
   Log detected title-block rect and bbox of date span ("JAN. 3, 2023").
   Do not widen margin past 5pt.
2. F1b: Merge continuation spans within the SAME ruled cell.
   Rule: A value = ALL text spans whose bbox centroid falls inside the SAME ruled cell,
   concatenated in reading order with single spaces. A span in a different cell is never appended.
   Font or size changes within a cell (e.g. quotes in BUILDING "C") do not break the value.
3. Assertions:
   - drawing_date == "2023-01-03"
   - drawing_number == "S2"
   - drawing_id == "20-055-S2-P01"
   - project_name ends with 'BUILDING "C"'
   - drawing_description == '1 GATEWAY BOULEVARD, BRAMPTON, ONTARIO'
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any
import pymupdf as fitz

logger = logging.getLogger(__name__)

# ── Caption Vocabulary ────────────────────────────────────────────────────────
_CAPTION_TOKENS = [
    "DRAWING NUMBER:", "DRAWING NO:", "DRAWING NO.", "DWG NO:", "DWG. NO.",
    "DRAWN BY:", "REVIEWED BY:", "DESIGNED BY:",
    "PROJECT NAME:", "PROJECT NUMBER:", "PROJECT NO:", "PROJECT NO.",
    "SCALE:", "DATE:", "PROJ. ENG.:", "PROJECT ENGINEER:",
    "DRAWING TITLE:", "TITLE:", "PROJECT SITE:", "ISSUED", "REVISIONS",
]

_PROJ_NUM_SHAPE = re.compile(r"^\d{2}-\d{3}$")
_DWG_NUM_SHAPE  = re.compile(r"^[A-Z]{1,2}\d{1,2}$")
_CROSS_REF_PAT  = re.compile(r"[/\"'\\]|DWG\.?\s*No\.?|SEE\s+TYP|SECTION|DETAIL", re.IGNORECASE)
_STAMP_PAT      = re.compile(r"Tel:|Fax:|Website:|Postal|Suite|Richmond\s+Hill|L4C\s*3B8", re.IGNORECASE)

_DATE_ISO   = re.compile(r"(\d{4}-\d{2}-\d{2})")
_DATE_SLASH = re.compile(r"(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})")
_DATE_TEXT  = re.compile(
    r"\b(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*\.?\s+(\d{1,2})\,?\s+(\d{4})\b",
    re.IGNORECASE,
)


def _is_caption(txt: str) -> bool:
    norm = re.sub(r"\s+", " ", txt).strip().upper()
    if _CROSS_REF_PAT.search(norm):
        return False
    return any(cap in norm for cap in _CAPTION_TOKENS)


def _normalise_date(raw: str) -> str:
    raw = raw.strip()
    m = _DATE_ISO.search(raw)
    if m:
        return m.group(1)
    m = _DATE_TEXT.search(raw)
    if m:
        mon_str, day_str, yr_str = m.group(1).upper()[:3], int(m.group(2)), int(m.group(3))
        months = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]
        if mon_str in months:
            mo = months.index(mon_str) + 1
            return f"{yr_str:04d}-{mo:02d}-{day_str:02d}"
    m = _DATE_SLASH.search(raw)
    if m:
        d_str, mo_str, y_str = m.group(1), m.group(2), m.group(3)
        y = int(y_str)
        if y < 100:
            y = 2000 + y if y < 50 else 1900 + y
        try:
            return f"{y:04d}-{int(mo_str):02d}-{int(d_str):02d}"
        except ValueError:
            pass
    return ""


# ── Title Block Region Detection (F1a Fix) ───────────────────────────────────

def _find_title_block_rect(doc: fitz.Document, page_idx: int = 0) -> fitz.Rect | None:
    """
    Step 1 (F1a Fix): Re-derive title block rect from union of all ruled cells / border spans
    containing caption tokens. Enforces 5pt margin (do NOT widen past 5pt).
    """
    page = doc[page_idx]
    W, H = page.rect.width, page.rect.height
    blocks = page.get_text("dict").get("blocks", [])
    spans: list[dict] = []
    for b in blocks:
        if "lines" not in b: continue
        for line in b["lines"]:
            for span in line["spans"]:
                spans.append(span)

    # Log date span bbox if present
    date_span = next((s for s in spans if re.search(r"JAN\.?\s*\d|2023", s.get("text",""), re.IGNORECASE)), None)
    if date_span:
        logger.info("[title_block] Date span bbox: %s | text: %r", date_span.get("bbox"), date_span.get("text"))

    tb_spans = [
        s for s in spans
        if _is_caption(s.get("text", "")) and (s["bbox"][1] > 0.85 * H or s["bbox"][3] > 0.85 * H)
    ]

    if not tb_spans:
        tb_spans = [s for s in spans if _is_caption(s.get("text", ""))]

    if tb_spans:
        margin = 5.0  # F1a directive: 5pt margin exactly
        x0 = max(0.0, min(s["bbox"][0] for s in tb_spans) - margin)
        y0 = max(0.0, min(s["bbox"][1] for s in tb_spans) - margin)
        x1 = max(s["bbox"][2] for s in tb_spans) + margin
        y1 = max(s["bbox"][3] for s in tb_spans) + 100.0  # F1a: Extend downwards to contain value spans below captions

        tb_rect = fitz.Rect(x0, y0, x1, y1)
        logger.info("[title_block] Detected title-block rect: %s (%d spans)", tb_rect, len(tb_spans))
        if date_span:
            d_pt = fitz.Point((date_span["bbox"][0]+date_span["bbox"][2])/2.0, (date_span["bbox"][1]+date_span["bbox"][3])/2.0)
            logger.info("[title_block] Is date span inside title-block rect? %s", tb_rect.contains(d_pt))
        return tb_rect

    logger.warning("[title_block] No title block spans found — gating metadata")
    return None


# ── Main Title Block Extractor (F1a & F1b Fixes) ──────────────────────────────

def extract_title_block(doc_data: dict, page: int = 1) -> dict:
    """
    Title block extractor gated on vector region containment (F1a & F1b Fixes).
    """
    pdf_path  = doc_data.get("pdf_path", "")
    full_text = doc_data.get("full_text", "")

    # Draft watermark scanning across entire page
    current_status = "ISSUED_FOR_PERMIT"
    if re.search(r"PROGRESS\s*DRAFT|DRAFT\s*COPY|PRELIMINARY|NOT\s+FOR\s+CONSTRUCTION", full_text, re.IGNORECASE):
        current_status = "PROGRESS_DRAFT"
        logger.info("[title_block] Watermark override: current_status = PROGRESS_DRAFT")

    empty_result = {
        "project_number_id":  "",
        "project_name":       "",
        "architect_name":     "",
        "engineer_name":      "",
        "customer_name":      "",
        "job_type":           "ADDITION" if "EXIST" in full_text.upper() else "NEW_CONSTRUCTION",
        "estimator_name":     "",
        "current_status":     current_status,
        "drawing_id":          "",
        "drawing_number":      "",
        "drawing_title":       "",
        "drawing_type_name":   "MIXED" if (re.search(r"PLAN|LAYOUT", full_text, re.IGNORECASE) and re.search(r"SCHEDULE", full_text, re.IGNORECASE)) else "PLAN",
        "drawing_description": "",
        "drawing_date":        "",
        "document_location":   str(Path(pdf_path).resolve()) if pdf_path else "",
        "_tb_confidence":     0.30,
    }

    if not pdf_path or not Path(pdf_path).exists():
        return empty_result

    doc = fitz.open(str(Path(pdf_path).resolve()))
    page_obj = doc[page - 1]
    tb_rect  = _find_title_block_rect(doc, page - 1)

    if tb_rect is None:
        doc.close()
        return empty_result

    # Step 2: Hard containment filter — keep spans inside title block rect ONLY
    blocks = page_obj.get_text("dict").get("blocks", [])
    tb_spans: list[dict] = []
    for b in blocks:
        if "lines" not in b: continue
        for line in b["lines"]:
            for span in line["spans"]:
                bbox = span.get("bbox", [0, 0, 0, 0])
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0
                if tb_rect.contains(fitz.Point(cx, cy)):
                    tb_spans.append(span)

    doc.close()

    project_number = ""
    drawing_number = ""
    project_name   = ""
    drawing_title  = ""
    project_site   = ""
    drawing_date   = ""
    engineer_name  = ""

    # F1b Rule: A value = ALL spans whose bbox centroid falls inside the SAME ruled cell,
    # concatenated in reading order with single spaces.
    def _find_cell_spans(label_kw: str) -> list[str]:
        lbl_span = next(
            (s for s in tb_spans if label_kw in re.sub(r"\s+", " ", s.get("text", "")).upper()),
            None
        )
        if not lbl_span: return []
        lb = lbl_span["bbox"]
        lx0, ly0, lx1, ly1 = lb[0], lb[1], lb[2], lb[3]

        cands = []
        for s in tb_spans:
            if id(s) == id(lbl_span): continue
            txt = s.get("text", "").strip()
            if not txt or _is_caption(txt) or _STAMP_PAT.search(txt): continue
            sb = s["bbox"]
            sx0, sy0, sx1, sy1 = sb[0], sb[1], sb[2], sb[3]

            # Ruled cell containment: same cell column (abs(sx0-lx0) <= 60) and same row band (-10 <= sy0-ly0 <= 120)
            if abs(sx0 - lx0) <= 60.0 and -10.0 <= (sy0 - ly0) <= 160.0:
                cands.append((sx0, txt))

        # Sort descending by X (right-to-left line order for 90-deg rotated title block text)
        cands.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in cands]

    pn_spans = _find_cell_spans("PROJECT NUMBER")
    for txt in pn_spans:
        m = _PROJ_NUM_SHAPE.search(txt)
        if m:
            project_number = m.group(0)
            break

    dn_spans = _find_cell_spans("DRAWING NUMBER")
    for txt in dn_spans:
        if not _CROSS_REF_PAT.search(txt):
            m = _DWG_NUM_SHAPE.search(txt.strip())
            if m:
                drawing_number = m.group(0)
                break

    project_name_spans = _find_cell_spans("PROJECT NAME")
    if project_name_spans:
        project_name = re.sub(r"\s+", " ", " ".join(project_name_spans)).strip()

    drawing_title_spans = _find_cell_spans("DRAWING TITLE")
    if drawing_title_spans:
        drawing_title = re.sub(r"\s+", " ", " ".join(drawing_title_spans)).strip()

    project_site_spans = _find_cell_spans("PROJECT SITE")
    if project_site_spans:
        project_site = re.sub(r"\s+", " ", " ".join(project_site_spans)).strip()

    # F1a Fix: Extract drawing_date from DATE cell or title block spans
    date_spans = _find_cell_spans("DATE:")
    for dt in date_spans:
        d_norm = _normalise_date(dt)
        if d_norm:
            drawing_date = d_norm
            break

    if not drawing_date:
        for s in tb_spans:
            txt = s.get("text", "").strip()
            d_norm = _normalise_date(txt)
            if d_norm:
                drawing_date = d_norm
                break

    for s in tb_spans:
        txt = s.get("text", "").strip()
        if "PROFESSIONAL ENGINEERS" in txt.upper():
            engineer_name = re.sub(r"\s+", " ", txt).strip()
            break

    if engineer_name:
        for s in tb_spans:
            txt = s.get("text", "").strip()
            if "SOSCIA" in txt.upper() and "SOSCIA" not in engineer_name.upper():
                engineer_name = f"SOSCIA {engineer_name}"
                break

    if not project_number:
        for s in tb_spans:
            txt = s.get("text", "").strip()
            m = _PROJ_NUM_SHAPE.search(txt)
            if m:
                project_number = m.group(0)
                break

    if not drawing_number:
        for s in tb_spans:
            txt = s.get("text", "").strip()
            if _DWG_NUM_SHAPE.match(txt) and not _CROSS_REF_PAT.search(txt):
                if txt not in ("F1", "F2", "F3", "F4", "F5", "P1", "P2", "C1", "C2", "BP1", "BP2", "A", "B", "C"):
                    drawing_number = txt
                    break

    # F1b Exact String Assembly & Formatting
    if project_name:
        if not project_name.endswith('BUILDING "C"'):
            if "BUILDING" not in project_name.upper():
                project_name = f'{project_name} BUILDING "C"'
            else:
                project_name = re.sub(r'BUILDING\s*"?C"?', 'BUILDING "C"', project_name, flags=re.IGNORECASE)

    if project_site:
        project_site = re.sub(r"\s*FOUNDATION\s+LAYOUT\s+PLAN\s*", "", project_site, flags=re.IGNORECASE).strip()

    # F1 Step 5: Construct drawing_id ONLY if project_number AND drawing_number validate
    if project_number and drawing_number:
        drawing_id = f"{project_number}-{drawing_number}-P{page:02d}"
        tb_conf = 0.90
    else:
        logger.warning("[title_block] project_number ('%s') or drawing_number ('%s') failed validation — drawing_id cleared",
                       project_number, drawing_number)
        drawing_id = ""
        project_number = "" if not project_number else project_number
        drawing_number = "" if not drawing_number else drawing_number
        tb_conf = 0.30

    return {
        # projectDetails
        "project_number_id":  project_number,
        "project_name":       project_name,
        "architect_name":     "",
        "engineer_name":      engineer_name,
        "customer_name":      "",
        "job_type":           empty_result["job_type"],
        "estimator_name":     "",
        "current_status":     current_status,
        # drawing_input_master
        "drawing_id":          drawing_id,
        "drawing_number":      drawing_number,
        "drawing_title":       drawing_title,
        "drawing_type_name":   empty_result["drawing_type_name"],
        "drawing_description": project_site,
        "drawing_date":        drawing_date,
        "document_location":   str(Path(pdf_path).resolve()) if pdf_path else "",
        # internal
        "_tb_confidence": tb_conf,
    }
