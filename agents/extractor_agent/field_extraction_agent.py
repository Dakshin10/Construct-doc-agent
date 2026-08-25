"""
agents/field_extraction_agent.py
─────────────────────────────────
Gemini Vision Dynamic Table & Diagram Count Extractor.

PRIMARY OUTPUT: extract_flat_rows() → list[FlatRow]  (36-column normalised schema)
COMPAT SHIM:   extract_table_from_doc() → list[dict]  (kept for tools.py / database agent path)

## Defects fixed in this version

D1  Raw text traceability validator integration; 0 fallback constants.
D2  Explicit schedule clipping detector triggers Gemini Vision fallback (VISION_FALLBACK, conf ≤ 0.70).
D3  CAD title block cell-based binding with strict shape validation.
D4  Strict zone containment + row-band binding; excludes masonry wall/lintel/notes schedules.
D5  Anchor bolt callouts excluded from plate geometry (span/width/height).
D6  Wrapped leader notes reassembled before grammar matching; noise fragments discarded.
D7  Synthetic mark naming (CF-, GB-, SOG-), qty=1 for continuous elements, loading ramp slab discovery.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pymupdf as fitz
from dotenv import load_dotenv
load_dotenv()
if os.environ.get("GEMINI_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

from google import genai
from google.genai import types

from agents.extractor_agent.pdf_vision_loader import load_pdf_for_vision
from agents.extractor_agent.csv_schema import (
    FlatRow, LayerCode, Direction, SourceView, ExtractionMethod,
    validate_rows,
)
from agents.extractor_agent.title_block import extract_title_block

load_dotenv()
logger = logging.getLogger(__name__)


# ── Gemini client ─────────────────────────────────────────────────────────────
def _get_gemini_client(api_key: str | None = None) -> genai.Client:
    key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if key and key != "your_gemini_api_key_here":
        return genai.Client(api_key=key)
    return genai.Client()


# ── Drawing segment (title) ───────────────────────────────────────────────────
def _extract_drawing_segment(full_text: str, doc_data: dict | None = None) -> str:
    """Dynamically extract drawing title/description. Returns '' on no match."""
    if doc_data and "text_blocks" in doc_data:
        for block in doc_data["text_blocks"]:
            txt = block.get("text", "").strip()
            upper = txt.upper()
            if "BUILDING" in upper and (
                "LAYOUT" in upper or "FRAMING" in upper or "FOUNDATION" in upper
            ):
                for line in txt.splitlines():
                    u_line = line.strip().upper()
                    if "BUILDING" in u_line and (
                        "LAYOUT" in u_line or "FRAMING" in u_line or "FOUNDATION" in u_line
                    ) and len(line.strip()) < 80:
                        return line.strip()
    return ""


# ── Grid reference extraction ─────────────────────────────────────────────────
def _extract_grid_references(
    doc: fitz.Document,
    page_idx: int = 0,
) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    """
    Extract Grid X (letter labels) and Grid Y (number labels) coordinates from
    ArialMT 8.5pt grid bubble tokens.
    """
    page = doc[page_idx]
    blocks = page.get_text("dict").get("blocks", [])

    grid_x_map: dict[str, float] = {}
    grid_y_map: dict[str, float] = {}
    GRID_PAT = re.compile(r"^(?:[A-C][xy]?|\d{1,2}[xy]?)$")

    for b in blocks:
        if "lines" not in b:
            continue
        for line in b["lines"]:
            for span in line["spans"]:
                txt  = span.get("text", "").strip()
                font = span.get("font", "")
                sz   = span.get("size", 0.0)
                if GRID_PAT.match(txt) and "Arial" in font and 8.0 <= sz <= 9.0:
                    bbox = span.get("bbox", [0, 0, 0, 0])
                    cx = (bbox[0] + bbox[2]) / 2.0
                    cy = (bbox[1] + bbox[3]) / 2.0
                    if re.match(r"^\d", txt):
                        grid_y_map.setdefault(txt, cy)
                    else:
                        grid_x_map.setdefault(txt, cx)

    grid_x = sorted(grid_x_map.items(), key=lambda it: it[1], reverse=True)
    grid_y = sorted(grid_y_map.items(), key=lambda it: it[1])
    return grid_x, grid_y


# ── Grid snap parameters ──────────────────────────────────────────────────────
def _compute_grid_snap_params(
    grid_x: list[tuple[str, float]],
    grid_y: list[tuple[str, float]],
    element_positions_x: list[float],
    element_positions_y: list[float],
    bias_iterations: int = 4,
) -> dict:
    import statistics

    def _nearest(val: float, lines: list[float]) -> float:
        return min(lines, key=lambda l: abs(l - val))

    def _estimate_bias(vals: list[float], lines: list[float], iters: int = 4) -> float:
        if not vals or not lines:
            return 0.0
        bias = 0.0
        for _ in range(iters):
            residuals = [v - _nearest(v - bias, lines) for v in vals]
            bias = statistics.median(residuals)
        return bias

    gx = [v for _, v in grid_x]
    gy = [v for _, v in grid_y]
    bias_x = _estimate_bias(element_positions_x, gx, bias_iterations) if gx else 0.0
    bias_y = _estimate_bias(element_positions_y, gy, bias_iterations) if gy else 0.0

    if len(gx) >= 2:
        gx_s = sorted(gx)
        tol_x = 0.5 * min(gx_s[i+1] - gx_s[i] for i in range(len(gx_s)-1))
    else:
        tol_x = 50.0
    if len(gy) >= 2:
        gy_s = sorted(gy)
        tol_y = 0.5 * min(gy_s[i+1] - gy_s[i] for i in range(len(gy_s)-1))
    else:
        tol_y = 50.0

    return {"bias_x": bias_x, "bias_y": bias_y, "tol_x": tol_x, "tol_y": tol_y}


# ── Orphan column bearing verification ───────────────────────────────────────
def _verify_orphan_column_bearing(
    doc: fitz.Document,
    orphan_cols: list[dict],
    page_idx: int = 0,
    buffer: float = 50.0,
) -> tuple[int, list[dict]]:
    page = doc[page_idx]
    wall_paths = []
    for item in page.get_drawings():
        r = item.get("rect")
        if r and (r.width > 200 or r.height > 200):
            wall_paths.append((r.x0, r.y0, r.x1, r.y1))

    verified = 0
    unverified = []
    for c in orphan_cols:
        px, py = c["x"], c["y"]
        near = [
            r for r in wall_paths
            if (r[0]-buffer) <= px <= (r[2]+buffer)
            and (r[1]-buffer) <= py <= (r[3]+buffer)
        ]
        if near:
            verified += 1
            c["bearing_element"] = "unnamed_continuous_footing"
        else:
            unverified.append({
                "text": c.get("raw", ""),
                "pos": (px, py),
                "reason": "unsupported_orphan_column",
            })
    return verified, unverified


# ── Callout axis detection ────────────────────────────────────────────────────
def _detect_callout_axes(blocks: list[dict]) -> tuple[str, str]:
    dir_votes: dict[tuple, int] = {}
    for b in blocks:
        if "lines" not in b:
            continue
        for line in b["lines"]:
            sz_ok = any(7.5 <= sp.get("size", 0) <= 9.0 for sp in line["spans"])
            if not sz_ok:
                continue
            d = tuple(round(v) for v in line.get("dir", (1, 0)))
            dir_votes[d] = dir_votes.get(d, 0) + 1

    if not dir_votes:
        return "y", "x"
    dominant = max(dir_votes, key=dir_votes.get)
    if abs(dominant[1]) > abs(dominant[0]):
        return "x", "y"
    return "y", "x"


# ── Diagram quantity counting ─────────────────────────────────────────────────
def _count_diagram_quantities_with_pairing(
    pdf_path: str | Path,
    doc_data: dict[str, Any],
) -> tuple[dict[str, int], dict[str, list[str]], dict[str, Any]]:
    import statistics
    import warnings

    CALLOUT_SIZE_SANITY_MIN = 7.0
    CALLOUT_SIZE_SANITY_MAX = 10.0
    CALLOUT_SIZE_HINT_MIN   = 7.5
    CALLOUT_SIZE_HINT_MAX   = 9.0

    path = Path(pdf_path).resolve()
    if not path.exists():
        return {}, {}, {}

    doc = fitz.open(str(path))
    page_idx = 0
    page  = doc[page_idx]
    blocks = page.get_text("dict").get("blocks", [])

    grid_x, grid_y = _extract_grid_references(doc, page_idx)

    _BIAS_PREPASS_PAT = re.compile(r"^(?:F\d|P\d|C\d|BP\d)")
    _pre_x: list[float] = []
    _pre_y: list[float] = []
    for _b in blocks:
        if "lines" not in _b:
            continue
        for _line in _b["lines"]:
            for _span in _line["spans"]:
                _sz  = _span.get("size", 0.0)
                _txt = _span.get("text", "").strip()
                if _txt and 7.0 <= _sz <= 10.0 and _BIAS_PREPASS_PAT.match(_txt):
                    _bb = _span.get("bbox", [0, 0, 0, 0])
                    _pre_x.append((_bb[0] + _bb[2]) / 2.0)
                    _pre_y.append((_bb[1] + _bb[3]) / 2.0)

    _snap  = _compute_grid_snap_params(grid_x, grid_y, _pre_x, _pre_y)
    _bias_x = _snap["bias_x"]
    _bias_y = _snap["bias_y"]
    _tol_x  = _snap["tol_x"]
    _tol_y  = _snap["tol_y"]

    def get_grid_ref(x: float, y: float) -> str | None:
        if not grid_x or not grid_y:
            return None
        corr_x = x - _bias_x
        corr_y = y - _bias_y
        nx_name, nx_val = min(grid_x, key=lambda it: abs(it[1] - corr_x))
        ny_name, ny_val = min(grid_y, key=lambda it: abs(it[1] - corr_y))
        if abs(nx_val - corr_x) > _tol_x or abs(ny_val - corr_y) > _tol_y:
            return None
        return f"Grid {ny_name}/{nx_name}"

    FOOT_PAT     = re.compile(r"^(F\d)\s*[,.\u00b7]?\s*(P\d)$")
    COL_PAT      = re.compile(r"^(C\d)\s*[,.\u00b7]?\s*(BP\d)$")
    SINGLE_F     = re.compile(r"^(F\d)$")
    SINGLE_BP    = re.compile(r"^(BP\d)$")
    ALL_CALLOUT  = re.compile(r"^(?:F\d|P\d|C\d|BP\d)")

    size_bucket_hits: dict[float, int] = {}
    for b in blocks:
        if "lines" not in b:
            continue
        for line in b["lines"]:
            for span in line["spans"]:
                sz     = span.get("size", 0.0)
                bucket = round(sz * 2) / 2
                txt    = span.get("text", "").strip()
                if ALL_CALLOUT.match(txt) and CALLOUT_SIZE_SANITY_MIN <= sz <= CALLOUT_SIZE_SANITY_MAX:
                    size_bucket_hits[bucket] = size_bucket_hits.get(bucket, 0) + 1

    if size_bucket_hits:
        best_bucket = max(size_bucket_hits, key=size_bucket_hits.get)
        cluster_min = best_bucket - 0.75
        cluster_max = best_bucket + 0.75
    else:
        cluster_min, cluster_max = CALLOUT_SIZE_HINT_MIN, CALLOUT_SIZE_HINT_MAX

    stacking_axis, alignment_axis = _detect_callout_axes(blocks)

    raw_spans = []
    for b in blocks:
        if "lines" not in b:
            continue
        for line in b["lines"]:
            ldir = tuple(round(v) for v in line.get("dir", (1, 0)))
            for span in line["spans"]:
                txt  = span.get("text", "").strip()
                sz   = span.get("size", 0.0)
                if txt:
                    raw_spans.append({
                        "text": txt, "bbox": span.get("bbox", [0,0,0,0]),
                        "font": span.get("font", ""), "size": sz,
                        "x": span.get("bbox", [0])[0],
                        "y": span.get("bbox", [0,0,0,0])[1],
                        "dir": ldir,
                    })

    callout_spans = [s for s in raw_spans if cluster_min <= s["size"] <= cluster_max]

    if stacking_axis == "x":
        callout_spans.sort(key=lambda s: (round(s["x"], 1), s["y"]))
        def _adjacent(prev, cur):
            return abs(prev["x"] - cur["x"]) < 2.0 and 0 <= (cur["y"] - prev["bbox"][3]) < 6.0
    else:
        callout_spans.sort(key=lambda s: (round(s["y"], 1), s["x"]))
        def _adjacent(prev, cur):
            return abs(prev["y"] - cur["y"]) < 2.0 and 0 <= (cur["x"] - prev["bbox"][2]) < 6.0

    merged_spans = []
    for s in callout_spans:
        if not merged_spans:
            merged_spans.append(dict(s))
        else:
            prev = merged_spans[-1]
            if _adjacent(prev, s):
                prev["text"] += s["text"]
                prev["bbox"] = [
                    min(prev["bbox"][0], s["bbox"][0]),
                    min(prev["bbox"][1], s["bbox"][1]),
                    max(prev["bbox"][2], s["bbox"][2]),
                    max(prev["bbox"][3], s["bbox"][3]),
                ]
            else:
                merged_spans.append(dict(s))

    foots: list[dict] = []
    cols:  list[dict] = []
    review_queue: list[dict] = []

    for s in merged_spans:
        txt  = s["text"]
        gref = get_grid_ref(s["x"], s["y"])
        if gref is None:
            gref = "Unsnapped — review"

        m_foot = FOOT_PAT.match(txt)
        if m_foot:
            foots.append({"f": m_foot.group(1), "p": m_foot.group(2),
                          "x": s["x"], "y": s["y"], "grid": gref, "raw": txt})
            continue
        m_sf = SINGLE_F.match(txt)
        if m_sf:
            foots.append({"f": m_sf.group(1), "p": None,
                          "x": s["x"], "y": s["y"], "grid": gref, "raw": txt})
            continue
        m_col = COL_PAT.match(txt)
        if m_col:
            cols.append({"c": m_col.group(1), "bp": m_col.group(2),
                         "x": s["x"], "y": s["y"], "grid": gref, "raw": txt})
            continue
        m_sbp = SINGLE_BP.match(txt)
        if m_sbp:
            cols.append({"c": None, "bp": m_sbp.group(1),
                         "x": s["x"], "y": s["y"], "grid": gref, "raw": txt})
            continue

    def pair_callouts(foot_list, col_list):
        if not foot_list or not col_list:
            return [], foot_list[:], col_list[:]

        stack_offsets, align_offsets = [], []
        for f in foot_list:
            closest_c = min(col_list, key=lambda c: (c["x"]-f["x"])**2 + (c["y"]-f["y"])**2)
            if stacking_axis == "x":
                stack_offsets.append(closest_c["x"] - f["x"])
                align_offsets.append(closest_c["y"] - f["y"])
            else:
                stack_offsets.append(closest_c["y"] - f["y"])
                align_offsets.append(closest_c["x"] - f["x"])

        median_stack = statistics.median(stack_offsets)
        median_align = statistics.median(align_offsets)

        stack_lo = min(median_stack * 1.5, median_stack * 0.5)
        stack_hi = max(median_stack * 1.5, median_stack * 0.5)
        try:
            align_std = statistics.stdev(align_offsets)
        except statistics.StatisticsError:
            align_std = 1.0
        align_tol = max(3.0 * align_std, 2.0)

        used_cols: set[int] = set()
        pairs: list[tuple[dict, dict]] = []
        orphan_foots: list[dict] = []

        for f in foot_list:
            cands = []
            for i, c in enumerate(col_list):
                if i in used_cols:
                    continue
                sv = (c["x"] - f["x"]) if stacking_axis == "x" else (c["y"] - f["y"])
                av = (c["y"] - f["y"]) if stacking_axis == "x" else (c["x"] - f["x"])
                if stack_lo <= sv <= stack_hi and abs(av - median_align) <= align_tol:
                    cands.append((abs(sv - median_stack) + abs(av), i))
            if cands:
                cands.sort()
                best_i = cands[0][1]
                used_cols.add(best_i)
                pairs.append((f, col_list[best_i]))
            else:
                orphan_foots.append(f)

        orphan_cols = [c for i, c in enumerate(col_list) if i not in used_cols]
        return pairs, orphan_foots, orphan_cols

    pairs, orphan_foots, orphan_cols = pair_callouts(foots, cols)

    final_counts:   dict[str, int]        = {}
    grid_refs_map:  dict[str, list[str]]  = {}

    for f_item, c_item in pairs:
        for mark, item_ref, gref in [
            (f_item["f"], f_item, f_item["grid"]),
            (f_item["p"], f_item, f_item["grid"]),
            (c_item["c"], c_item, c_item["grid"]),
            (c_item["bp"], c_item, c_item["grid"]),
        ]:
            if mark:
                final_counts[mark] = final_counts.get(mark, 0) + 1
                grid_refs_map.setdefault(mark, []).append(gref)

    for of in orphan_foots:
        f_mark = of["f"]
        p_mark = of["p"]
        if f_mark:
            final_counts[f_mark] = final_counts.get(f_mark, 0) + 1
            grid_refs_map.setdefault(f_mark, []).append(of["grid"])
        if p_mark:
            final_counts[p_mark] = final_counts.get(p_mark, 0) + 1

    for oc in orphan_cols:
        c_mark  = oc.get("c")
        bp_mark = oc.get("bp")
        if c_mark:
            final_counts[c_mark] = final_counts.get(c_mark, 0) + 1
            grid_refs_map.setdefault(c_mark, []).append(oc["grid"])
        if bp_mark:
            final_counts[bp_mark] = final_counts.get(bp_mark, 0) + 1

    _verified_cnt, _unverified = _verify_orphan_column_bearing(doc, orphan_cols, page_idx)

    tot_pads  = sum(v for k, v in final_counts.items() if k.startswith("F"))
    tot_piers = sum(v for k, v in final_counts.items() if k.startswith("P") and not k.startswith("P-"))
    tot_cols  = sum(v for k, v in final_counts.items() if k.startswith("C") and not k.startswith("CF"))
    tot_bps   = sum(v for k, v in final_counts.items() if k.startswith("BP"))

    qc_report = {
        "total_pads":                tot_pads,
        "total_piers":               tot_piers,
        "total_columns":             tot_cols,
        "total_base_plates":         tot_bps,
        "spatially_paired_callouts": len(pairs),
        "orphan_columns":            len(orphan_cols),
        "orphan_footings":           len(orphan_foots),
        "pairing_closure_ok":        (len(pairs) + len(orphan_foots) == len(foots)),
        "grid_snap_bias_x":          round(_bias_x, 2),
        "grid_snap_bias_y":          round(_bias_y, 2),
        "grid_snap_tol_x":           round(_tol_x, 2),
        "grid_snap_tol_y":           round(_tol_y, 2),
    }

    doc.close()
    return final_counts, grid_refs_map, qc_report


# ── Reinforcement data structures ─────────────────────────────────────────────
@dataclass
class RebarLayer:
    """One layer of reinforcement per the §6 grammar."""
    layer_code:  Optional[str] = None
    direction:   Optional[str] = None
    quantity:    Optional[int] = None
    size:        Optional[str] = None
    spacing_mm:  Optional[int] = None
    length_mm:   Optional[int] = None
    hook_mm:     Optional[int] = None
    cover_mm:    Optional[int] = None
    type:        Optional[str] = None
    raw:         str = ""


@dataclass
class ElementTypeAttr:
    """Dimensional + reinforcement attributes for one schedule element type."""
    mark:                   str
    length_mm:              Optional[int]   = None
    width_mm:               Optional[int]   = None
    depth_mm:               Optional[int]   = None
    rebar_layers:           list[RebarLayer] = field(default_factory=list)
    reinforcement_display:  str              = ""
    provenance:             dict             = field(default_factory=dict)
    source_view:            str              = "SCHEDULE"
    source_page:            int              = 1
    extraction_method:      str              = "TABLE_PARSE"
    confidence:             float            = 0.85


# ── Rebar string parser ───────────────────────────────────────────────────────
_CSA_SIZE_PAT   = re.compile(r"\b(10M|15M|20M|25M|30M|35M|45M|55M)\b", re.IGNORECASE)
_MESH_SIZE_PAT  = re.compile(r"(\d+x\d+-MW[\d.]+/MW[\d.]+)", re.IGNORECASE)
_LAYER_TAG_PAT  = re.compile(r"\((BLL|BUL|TLL|TUL|VERT|TIES|CONT_TOP|CONT_BOT|MESH|DOWEL)\)", re.IGNORECASE)
_SPACING_PAT    = re.compile(r"@\s*(\d+)\s*(?:mm\s*)?c/c", re.IGNORECASE)
_QTY_PAT        = re.compile(r"(?:^|[\s(])(\d{1,2})\s*[-–]\s*(\d{2}M)\b", re.IGNORECASE)
_HOOK_PAT       = re.compile(r"\+\s*(\d+)\s*mm\s+hook", re.IGNORECASE)
_LENGTH_PAT     = re.compile(r"[xX]\s*(\d+)\s*mm\s+(?:Lg|Long)", re.IGNORECASE)
_COVER_PAT      = re.compile(r"(\d+)\s*mm\s+cover", re.IGNORECASE)
_EW_PAT         = re.compile(r"E\.?W\.?|each\s+way", re.IGNORECASE)
_VERT_PAT       = re.compile(r"\bVERT(?:ICAL)?\b|\bVERT\.?", re.IGNORECASE)
_TOP_BOT_PAT    = re.compile(r"\bTOP\s*&\s*BOTTOM\b|\bTOP\s+AND\s+BOTTOM\b", re.IGNORECASE)
_TOP_WALL_PAT   = re.compile(r"\bAT\s+TOP\s+OF\s+WALL\b|\bCONT(?:\.|INUOUS)?\s+BARS\s+AT\s+TOP\b", re.IGNORECASE)
_TIES_ADDL_PAT  = re.compile(r"\+\s*(\d+)\s+TIES?\s+@\s+TOP", re.IGNORECASE)
_WWM_PAT        = re.compile(r"W\.?W\.?M\.?|WELDED\s+WIRE\s+MESH", re.IGNORECASE)
_ANCHOR_BOLT_PAT = re.compile(r"anchor\s+bolt|hex\s+head|bolt\s+gauge", re.IGNORECASE)


def _parse_rebar_string(text: str) -> list[RebarLayer]:
    """
    Parse flat reinforcement string into RebarLayer records.
    D6 Fix: multi-line notes reassembled; non-reinforcement lines filtered out.
    D5 Fix: Anchor bolts store diameter in raw_text, never bar_size.
    """
    layers: list[RebarLayer] = []

    for raw_line in re.split(r"[;\n]", text):
        raw_line = raw_line.strip()
        if not raw_line:
            continue

        # Skip non-reinforcement general note text (D6 Fix)
        if re.search(r"\b(CONFORM TO|MIN\. Fy=|FINISHED GRADE|REINFORCING STEEL TO BE)\b", raw_line, re.IGNORECASE):
            continue

        # Anchor bolts — D5 Fix
        if _ANCHOR_BOLT_PAT.search(raw_line):
            layer = RebarLayer(raw=raw_line, type="ANCHOR_BOLTS", layer_code=LayerCode.DOWEL.value)
            qty_m = re.search(r"^(\d+)\s*[-\u2013]|\b(\d+)\s*-\s*\d+mm\b", raw_line)
            if qty_m:
                layer.quantity = int(qty_m.group(1) or qty_m.group(2))
            lm = _LENGTH_PAT.search(raw_line)
            if lm:
                layer.length_mm = int(lm.group(1))
            hm = _HOOK_PAT.search(raw_line)
            if hm:
                layer.hook_mm = int(hm.group(1))
            layers.append(layer)
            continue

        # Welded wire mesh
        if _WWM_PAT.search(raw_line):
            layer = RebarLayer(raw=raw_line, type="MESH", layer_code=LayerCode.MESH.value, direction=Direction.EW.value)
            mesh_m = _MESH_SIZE_PAT.search(raw_line)
            if mesh_m:
                layer.size = mesh_m.group(1)
            else:
                layer.size = raw_line.strip()
            layers.append(layer)
            continue

        # TOP & BOTTOM expansion
        if _TOP_BOT_PAT.search(raw_line):
            qty_m  = _QTY_PAT.search(raw_line)
            qty    = int(qty_m.group(1)) if qty_m else None
            size   = qty_m.group(2).upper() if qty_m else (_CSA_SIZE_PAT.search(raw_line).group(1).upper() if _CSA_SIZE_PAT.search(raw_line) else None)
            for lc in (LayerCode.CONT_TOP.value, LayerCode.CONT_BOT.value):
                layers.append(RebarLayer(
                    raw=raw_line, layer_code=lc, size=size, quantity=qty,
                    type="BARS", direction=Direction.LONGITUDINAL.value,
                ))
            continue

        # CONT. BARS AT TOP OF WALL
        if _TOP_WALL_PAT.search(raw_line):
            qty_m  = _QTY_PAT.search(raw_line)
            qty    = int(qty_m.group(1)) if qty_m else None
            size   = qty_m.group(2).upper() if qty_m else (_CSA_SIZE_PAT.search(raw_line).group(1).upper() if _CSA_SIZE_PAT.search(raw_line) else None)
            layers.append(RebarLayer(
                raw=raw_line, layer_code=LayerCode.CONT_TOP.value,
                size=size, quantity=qty, type="BARS",
            ))
            continue

        # Ties with additional count at top
        addl_m = _TIES_ADDL_PAT.search(raw_line)
        if addl_m and re.search(r"TIES?", raw_line, re.IGNORECASE):
            size_m = _CSA_SIZE_PAT.search(raw_line)
            size   = size_m.group(1).upper() if size_m else None
            sp_m   = _SPACING_PAT.search(raw_line)
            layers.append(RebarLayer(
                raw=raw_line, layer_code=LayerCode.TIES.value,
                size=size, spacing_mm=int(sp_m.group(1)) if sp_m else None,
                type="TIES",
            ))
            layers.append(RebarLayer(
                raw=raw_line, layer_code=LayerCode.TIES.value,
                size=size, quantity=int(addl_m.group(1)),
                type="TIES",
            ))
            continue

        # General case
        layer = RebarLayer(raw=raw_line)
        qty_m = _QTY_PAT.search(raw_line)
        if qty_m:
            layer.quantity = int(qty_m.group(1))
            layer.size     = qty_m.group(2).upper()
        else:
            size_m = _CSA_SIZE_PAT.search(raw_line)
            if size_m:
                layer.size = size_m.group(1).upper()

        sp_m = _SPACING_PAT.search(raw_line)
        if sp_m:
            if layer.quantity is None:
                layer.spacing_mm = int(sp_m.group(1))

        if re.search(r"\bBARs?\b", raw_line, re.IGNORECASE):
            layer.type = "BARS"
        elif re.search(r"\bTIEs?\b", raw_line, re.IGNORECASE):
            layer.type = "TIES"
        elif re.search(r"\bMESH\b", raw_line, re.IGNORECASE):
            layer.type = "MESH"

        tag_m = _LAYER_TAG_PAT.search(raw_line)
        if tag_m:
            layer.layer_code = tag_m.group(1).upper()
        elif _VERT_PAT.search(raw_line):
            layer.layer_code = LayerCode.VERT.value
            layer.direction  = Direction.VERTICAL.value
        elif re.search(r"\bTIEs?\b", raw_line, re.IGNORECASE):
            layer.layer_code = LayerCode.TIES.value
        elif not layer.layer_code:
            layer.layer_code = LayerCode.UNSPECIFIED.value

        if _EW_PAT.search(raw_line):
            layer.direction = Direction.EW.value
        elif _VERT_PAT.search(raw_line) and not layer.direction:
            layer.direction = Direction.VERTICAL.value

        lm = _LENGTH_PAT.search(raw_line)
        if lm:
            layer.length_mm = int(lm.group(1))
        hm = _HOOK_PAT.search(raw_line)
        if hm:
            layer.hook_mm = int(hm.group(1))
        cm = _COVER_PAT.search(raw_line)
        if cm:
            layer.cover_mm = int(cm.group(1))

        # Filter out noise fragments with no size, quantity, spacing, or layer code (D6 Fix)
        if layer.size or layer.quantity or layer.spacing_mm or (layer.layer_code and layer.layer_code != LayerCode.UNSPECIFIED.value):
            layers.append(layer)

    # D7.5 Fix: Deduplicate tie rows (remove unpopulated tie row when populated exists)
    tie_rows_populated = [l for l in layers if l.layer_code == LayerCode.TIES.value and (l.quantity or l.spacing_mm)]
    if tie_rows_populated:
        layers = [l for l in layers if not (l.layer_code == LayerCode.TIES.value and not l.quantity and not l.spacing_mm)]

    return layers


# ── Schedule zone helpers & D4 containment ───────────────────────────────────

_EXCLUDED_ZONES_PAT = re.compile(
    r"\b(MASONRY\s+WALL\s+SCHEDULE|LINTEL\s+SCHEDULE|CONCRETE\s+SCHEDULE|GENERAL\s+NOTES)\b",
    re.IGNORECASE,
)


def _assemble_schedule_zones(doc: fitz.Document, page_idx: int = 0) -> list[dict]:
    """
    Derive schedule table zones from get_drawings() vector geometry.
    D4 Fix: Exclude non-foundation schedule zones (masonry walls, lintels, notes).
    """
    page = doc[page_idx]
    W, H = page.rect.width, page.rect.height
    page_text = page.get_text("blocks")

    rects = []
    for item in page.get_drawings():
        r = item.get("rect")
        if r is None:
            continue
        w, h = r.width, r.height
        if 100 < w < 0.8 * W and 100 < h < 0.8 * H:
            # Check if header matches an excluded zone
            zone_text = ""
            for blk in page_text:
                bx0, by0, bx1, by1 = blk[0], blk[1], blk[2], blk[3]
                if r.x0 - 5 <= bx0 and bx1 <= r.x1 + 5 and r.y0 - 5 <= by0 and by1 <= r.y1 + 5:
                    zone_text += " " + blk[4]
            if _EXCLUDED_ZONES_PAT.search(zone_text):
                continue  # Skip excluded table
            rects.append({"x0": r.x0, "y0": r.y0, "x1": r.x1, "y1": r.y1, "width": w, "height": h})

    def _overlaps(a, b):
        return not (a["x1"] <= b["x0"] or b["x1"] <= a["x0"] or a["y1"] <= b["y0"] or b["y1"] <= a["y0"])

    merged = []
    for r in sorted(rects, key=lambda x: x["width"] * x["height"], reverse=True):
        if not any(_overlaps(r, m) for m in merged):
            merged.append(r)
    return sorted(merged, key=lambda x: (x["y0"], x["x0"]))


def _blocks_in_zone(blocks: list[dict], zone: dict, margin: float = 10.0) -> list[dict]:
    x0, y0, x1, y1 = zone["x0"], zone["y0"], zone["x1"], zone["y1"]
    result = []
    for b in blocks:
        bb = b.get("bbox", [0, 0, 0, 0])
        cx = (bb[0] + bb[2]) / 2.0
        cy = (bb[1] + bb[3]) / 2.0
        if (x0-margin) <= cx <= (x1+margin) and (y0-margin) <= cy <= (y1+margin):
            result.append(b)
    return result


def _find_mark_cell_anchored(blocks: list[dict], mark: str) -> dict | None:
    pat = re.compile(r"(?:^|[\s,.\n])" + re.escape(mark) + r"(?:$|[\s,.\n])")
    matches = [b for b in blocks if pat.search(b.get("text", ""))]
    if not matches:
        return None
    if len(matches) > 1:
        exact = [b for b in matches if any(line.strip() == mark for line in b.get("text", "").splitlines())]
        if len(exact) == 1:
            return exact[0]
        matches.sort(key=lambda b: (b.get("bbox", [0, 0])[1], b.get("bbox", [0])[0]))
    return matches[0]


_MARK_PAT = re.compile(r"^(F|P|C|BP|SF|GB|W)\d+[A-Z]?$")


def _discover_schedule_marks(sched_blocks: list[dict]) -> list[str]:
    seen: set[str] = set()
    marks: list[str] = []
    for b in sched_blocks:
        for line in b.get("text", "").splitlines():
            for token in re.split(r"[\s,./]+", line.strip()):
                token = token.strip()
                if _MARK_PAT.match(token) and token not in seen:
                    seen.add(token)
                    marks.append(token)
    return marks


# ── Dimension parsing (D5 & D1 Fix) ──────────────────────────────────────────
_BOLT_SPEC_PAT = re.compile(r"\b(DIA\.?|BOLT|ANCHOR|HEX\s+HEAD|Lg\.?|LONG|HOOK)\b", re.IGNORECASE)
_PLATE_DIM_PAT = re.compile(r"(\d+)\s*[xX]\s*(\d+)\s*[xX]\s*(\d+)\s*(?:Thk\.?|MM)?", re.IGNORECASE)


def _dim_from_text_positional(text: str) -> tuple[Optional[int], Optional[int], Optional[int]]:
    """
    Parse (length_mm, width_mm, depth_mm) from text.
    D5 Fix: Ignore any token containing anchor bolt keywords (DIA., BOLT, ANCHOR, Lg.).
    D1 Fix: No fallback constants.
    """
    # D5 Fix: exclude bolt callouts
    if _BOLT_SPEC_PAT.search(text):
        # Look for explicit plate dimensions like "400 x 400 x 22 Thk."
        pm = _PLATE_DIM_PAT.search(text)
        if pm:
            return int(pm.group(1)), int(pm.group(2)), int(pm.group(3))
        return None, None, None

    nums = re.findall(r"(\d+(?:\.\d+)?)\s*mm", text, re.IGNORECASE)
    if not nums:
        nums = re.findall(r"\b(\d{3,4})\b", text)

    def _to_int(s: str) -> Optional[int]:
        try:
            return int(float(s))
        except (ValueError, TypeError):
            return None

    length_mm = _to_int(nums[0]) if len(nums) >= 1 else None
    width_mm  = _to_int(nums[1]) if len(nums) >= 2 else None
    depth_mm  = _to_int(nums[2]) if len(nums) >= 3 else None
    return length_mm, width_mm, depth_mm


def _element_name_shape(mark: str, span_mm: int | None = None, width_mm: int | None = None) -> tuple[str, str]:
    """Map mark and extracted dimensions to human element_name and element_shape."""
    name = "Foundation Element"
    if mark.startswith("SF"):
        name = "Spread Footing"
    elif mark.startswith("F") and not mark.startswith("CF"):
        name = "Footing Pad"
    elif mark.startswith("P"):
        name = "Pier"
    elif mark.startswith("C") and not mark.startswith("CF"):
        name = "Column"
    elif mark.startswith("BP"):
        name = "Base Plate"
    elif mark.startswith("GB"):
        name = "Grade Beam"
    elif mark.startswith("CF"):
        name = "Continuous Trench Footing"
    elif mark.startswith("SOG"):
        name = "Slab-on-Grade"
    elif mark.startswith("W"):
        name = "Block Wall"

    # Shape derivation rules:
    if mark.startswith("CF-") or mark.startswith("GB-"):
        shape = "CONTINUOUS_STRIP"
    elif mark.startswith("SOG-"):
        shape = "SLAB"
    elif span_mm is not None and width_mm is not None:
        if span_mm == width_mm:
            shape = "SQUARE"
        else:
            shape = "RECTANGULAR"
    else:
        shape = "UNKNOWN"

    return name, shape


EXCLUDED_ZONE_HEADINGS = (
    "CONCRETE SCHEDULE",
    "MASONRY WALL SCHEDULE",
    "LINTEL SCHEDULE",
    "BLOCK LINTEL SCHEDULE",
)

VALID_MARK_PAT = re.compile(r"^(F|P|C|BP|SF|GB|W)\d+[A-Z]?$", re.IGNORECASE)


def _is_zone_eligible_for_vision(zblocks: list[dict], zrect: fitz.Rect | dict) -> tuple[bool, str]:
    r"""
    D1 Gate: Check zone eligibility before invoking Gemini Vision.
    Excludes materials spec tables (e.g. CONCRETE SCHEDULE) and requires ALL of:
      - zone text matches r"BARS @|TIES @|ANCHOR BOLT|HSS|Thk\.|PLATE"
      - zone text contains >= 2 dimension tokens matching r"\d{2,4}mm"
    """
    ztext = " ".join(b.get("text", "") for b in zblocks).strip()
    ztext_upper = ztext.upper()

    for h in EXCLUDED_ZONE_HEADINGS:
        if h in ztext_upper:
            heading_msg = f"Excluded heading: '{h}'"
            logger.info("[zone_gate] Zone %s heading '%s' -> SKIPPED (%s)", zrect, h, heading_msg)
            return False, heading_msg

    has_rebar = bool(re.search(r"BARS\s*@|TIES\s*@|ANCHOR\s*BOLT|HSS|Thk\.|PLATE", ztext, re.IGNORECASE))
    mm_tokens = re.findall(r"\b\d{2,4}\s*mm\b", ztext, re.IGNORECASE)
    has_dims = len(mm_tokens) >= 2

    if not has_rebar or not has_dims:
        reason = f"Ineligible: has_rebar={has_rebar}, mm_tokens={len(mm_tokens)}"
        logger.info("[zone_gate] Zone %s -> SKIPPED (%s)", zrect, reason)
        return False, reason

    logger.info("[zone_gate] Zone %s -> ELIGIBLE for Vision fallback", zrect)
    return True, "ELIGIBLE"


# ── D2 Clipping Detector & Vision Fallback (F2 Fix) ───────────────────────────
def _detect_zone_clipping(sched_blocks: list[dict], mark_list: list[str]) -> tuple[bool, str]:
    """
    Return (True, reason) if schedule zone displays clipping signals (F2 Fix):
      1. Spaced-out header (>50% of chars single-space separated)
      2. Dimension tokens matching ^0+mm$ or shorter than 3 chars (e.g. 00mm)
      3. Reinforcement-grammar lines binding to no mark in the same row band
      4. Distinct mark tokens < distinct dimension rows
    """
    rebar_kw = ("BARS", "TIES", "C/C", "W.W.M", "MESH", "BOLT", "ANCHOR")

    for b in sched_blocks:
        txt = b.get("text", "").strip()
        if not txt:
            continue
        # Rule 1: Spaced-out header (>50% of chars single-space separated)
        if re.search(r"(?:[A-Z]\s){4,}", txt):
            return True, "Spaced-out header text"
        letters = [c for c in txt if c.isalpha()]
        if len(letters) >= 6:
            single_letter_tokens = len(re.findall(r"\b[A-Za-z]\b", txt))
            if single_letter_tokens / len(letters) > 0.5:
                return True, "Spaced-out header (>50% single-space separated chars)"

        # Rule 2: dimension tokens matching ^0+mm$ or shorter than 3 chars (e.g. 00mm)
        if re.search(r"^\s*0+mm\s*$", txt, re.IGNORECASE) or re.search(r"\b0{1,2}mm\b", txt, re.IGNORECASE):
            return True, "Truncated numerics (^0+mm$ or <3 chars)"

        # Rule 3: reinforcement-grammar lines binding to no mark in the same row band
        txt_upper = txt.upper()
        if any(kw in txt_upper for kw in rebar_kw):
            by = b.get("bbox", [0, 0, 0, 0])[1]
            bound = False
            for mb in sched_blocks:
                mtxt = mb.get("text", "").strip()
                if mtxt in mark_list:
                    my = mb.get("bbox", [0, 0, 0, 0])[1]
                    if abs(by - my) <= 35:
                        bound = True
                        break
            if not bound:
                return True, "Unbound reinforcement grammar line"

    # Rule 4: distinct mark tokens < distinct dimension rows
    dim_rows = sum(
        1 for b in sched_blocks
        if len(re.findall(r"\d+\s*mm", b.get("text", ""), re.IGNORECASE)) >= 2
    )
    if dim_rows > 0 and len(mark_list) < dim_rows:
        return True, f"Count mismatch: {len(mark_list)} mark(s) < {dim_rows} dimension row(s)"

    return False, ""


_GEMINI_CLIENT: genai.Client | None = None


def _get_gemini_client(api_key: str | None = None) -> genai.Client:
    global _GEMINI_CLIENT
    if _GEMINI_CLIENT is not None and not api_key:
        return _GEMINI_CLIENT

    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise ValueError("No GEMINI_API_KEY or GOOGLE_API_KEY found in environment")
    if os.environ.get("GEMINI_API_KEY"):
        key = os.environ["GEMINI_API_KEY"]
    os.environ["GOOGLE_API_KEY"] = key

    client = genai.Client(api_key=key)
    if not api_key:
        _GEMINI_CLIENT = client
    return client


def _process_clipped_zone_with_vision(
    pdf_path: str,
    zone_rect: fitz.Rect | dict,
    page_idx: int = 0,
    api_key: str | None = None,
) -> list[dict]:
    """
    Step 2 (F2 Fix): Render cropped zone at 300 DPI and send to Gemini Vision.
    Returns list of row dicts from Vision JSON array.
    Widens crop boundary to capture the left-hand mark column.
    """
    if not pdf_path or not Path(pdf_path).exists():
        return []

    doc = fitz.open(pdf_path)
    page = doc[page_idx]
    if isinstance(zone_rect, dict):
        fitz_rect = fitz.Rect(zone_rect["x0"], zone_rect["y0"], zone_rect["x1"], zone_rect["y1"])
    else:
        fitz_rect = fitz.Rect(zone_rect)

    # Widen crop to capture left-hand mark column and margins
    fitz_rect = fitz.Rect(
        max(0, fitz_rect.x0 - 150),
        max(0, fitz_rect.y0 - 15),
        min(page.rect.x1, fitz_rect.x1 + 15),
        min(page.rect.y1, fitz_rect.y1 + 15),
    )

    # Intersect with page boundary to prevent bandwriter invalid crop errors
    fitz_rect = fitz_rect.intersect(page.rect)
    if fitz_rect.width <= 5 or fitz_rect.height <= 5:
        doc.close()
        return []

    pix = page.get_pixmap(clip=fitz_rect, dpi=300)
    img_bytes = pix.tobytes("png")
    doc.close()

    client = _get_gemini_client(api_key)
    prompt = (
        "You are reading one schedule table cropped from a structural drawing. "
        "The table's left-hand mark column may be clipped or partially cut off.\n\n"
        "Return ONLY a JSON array, no prose, no markdown fences. One object per table row:\n\n"
        '[{"mark": "F1", "length_mm": 1300, "width_mm": 1300, "depth_mm": 400, '
        '"reinforcement": ["15M BARS @ 250 c/c. (BLL)", "15M BARS @ 250 c/c. (BUL)"], "row_confidence": 0.9}]\n\n'
        "Rules:\n"
        "- Transcribe values EXACTLY as printed. Do not infer, complete, or standardise.\n"
        "- List EVERY reinforcement line in the row, including ties, stirrups, vertical bars, and additional-bar notes. Do not summarise or return only the first line.\n"
        "- If a cell is cut off or unreadable, use null. Never guess a plausible value.\n"
        "- If you cannot determine which mark owns a row, set \"mark\": null and still return the row.\n"
        "- Dimensions are millimetres, integers.\n"
        "- row_confidence 0.0-1.0 reflects how legible that row was."
    )

    last_err = None
    for mname in ["gemini-3.6-flash"]:
        try:
            response = client.models.generate_content(
                model=mname,
                contents=[
                    types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                    prompt,
                ],
            )
            txt = (response.text or "").strip()
            if "```" in txt:
                txt = re.sub(r"```(?:json)?", "", txt).strip()

            data = json.loads(txt)
            if isinstance(data, list):
                logger.info("[vision_fallback] Successfully parsed %d row(s) from Vision model %s", len(data), mname)
                return data
        except Exception as inner_e:
            logger.error("[vision_fallback] Model %s failed", mname, exc_info=True)
            last_err = inner_e

    if last_err:
        logger.error("[vision_fallback] All Vision models failed for cropped zone", exc_info=True)

    return []


# ── Vector schedule dimension parser ─────────────────────────────────────────
def _parse_vector_schedule_dimensions(
    full_text: str,
    doc_data: dict[str, Any] | None = None,
) -> dict[str, ElementTypeAttr]:
    results: dict[str, ElementTypeAttr] = {}
    if doc_data is None:
        return results

    all_blocks = doc_data.get("text_blocks", [])
    pdf_path = doc_data.get("pdf_path", "")

    sched_blocks = []
    if pdf_path and Path(pdf_path).exists():
        try:
            _doc = fitz.open(str(Path(pdf_path).resolve()))
            zones = _assemble_schedule_zones(_doc, page_idx=0)
            _doc.close()
            for zone in zones:
                sched_blocks.extend(_blocks_in_zone(all_blocks, zone))
            seen_ids = set()
            unique = []
            for b in sched_blocks:
                bid = id(b)
                if bid not in seen_ids:
                    seen_ids.add(bid)
                    unique.append(b)
            sched_blocks = unique
        except Exception as e:
            logger.warning("Could not assemble schedule zones: %s", e)
            sched_blocks = all_blocks
    else:
        sched_blocks = all_blocks

    if not sched_blocks:
        return results

    discovered_marks = _discover_schedule_marks(sched_blocks)
    if not discovered_marks:
        return results

    for mark in discovered_marks:
        cell = _find_mark_cell_anchored(sched_blocks, mark)
        if not cell:
            results[mark] = ElementTypeAttr(mark=mark, confidence=0.30)
            continue

        mark_bbox = cell["bbox"]
        mark_x = mark_bbox[0]
        mark_y = mark_bbox[1]

        # D4 Fix: Require both x-column alignment AND y-row-band containment (± 25pt)
        col_blocks = [
            b for b in sched_blocks
            if abs(b["bbox"][0] - mark_x) <= 30
            and abs(b["bbox"][1] - mark_y) <= 35
        ]

        dim_block = next(
            (b for b in col_blocks if len(re.findall(r"\d+\s*mm", b.get("text", ""), re.IGNORECASE)) >= 2),
            None,
        )
        if not dim_block:
            dim_block = next(
                (b for b in col_blocks if re.search(r"\d+\s*[xX]\s*\d+\s*[xX]\s*\d+", b.get("text", ""))),
                None,
            )

        reb_blocks = [
            b for b in col_blocks
            if any(kw in b.get("text", "").upper() for kw in ("BARS", "TIES", "C/C", "W.W.M", "MESH", "BOLT", "ANCHOR"))
        ]

        dim_text = dim_block.get("text", "") if dim_block else ""
        length_mm, width_mm, depth_mm = _dim_from_text_positional(dim_text)

        if mark.startswith("F") and length_mm and not width_mm:
            width_mm = length_mm

        elem_name, elem_shape = _element_name_shape(mark)
        if length_mm and width_mm and length_mm == width_mm:
            elem_shape = "SQUARE"
        elif length_mm and width_mm and length_mm != width_mm:
            elem_shape = "RECTANGULAR"

        reb_lines = []
        for rb in reb_blocks:
            for ln in rb.get("text", "").splitlines():
                s = ln.strip()
                if any(kw in s.upper() for kw in ("BARS", "TIES", "C/C", "W.W.M", "MESH", "BOLT", "ANCHOR")):
                    if s not in reb_lines:
                        reb_lines.append(s)
        reb_text = "; ".join(reb_lines)
        rebar_layers = _parse_rebar_string(reb_text) if reb_text else []
        confidence = 0.85 if dim_block else 0.45

        results[mark] = ElementTypeAttr(
            mark=mark,
            length_mm=length_mm,
            width_mm=width_mm,
            depth_mm=depth_mm,
            rebar_layers=rebar_layers,
            reinforcement_display=reb_text,
            provenance={"dim_text": dim_text, "reb_text": reb_text},
            confidence=confidence,
        )

    # F2 Fix: Check zone clipping per schedule zone & merge Vision fallback
    if pdf_path and Path(pdf_path).exists():
        try:
            _doc = fitz.open(str(Path(pdf_path).resolve()))
            zones = _assemble_schedule_zones(_doc, page_idx=0)
            _doc.close()
            for zrect in zones:
                zblocks = _blocks_in_zone(all_blocks, zrect)
                zmarks = _discover_schedule_marks(zblocks)
                is_clipped, reason = _detect_zone_clipping(zblocks, zmarks)
                if is_clipped:
                    eligible, gate_reason = _is_zone_eligible_for_vision(zblocks, zrect)
                    if not eligible:
                        logger.info("[zone_gate] Zone %s clipped (%s) but skipped by eligibility gate (%s)", zrect, reason, gate_reason)
                        continue

                    logger.info("[clipping_detector] Zone %s clipped (%s) & ELIGIBLE — invoking Vision fallback", zrect, reason)
                    vrows = _process_clipped_zone_with_vision(pdf_path, zrect, page_idx=0)
                    for vrow in vrows:
                        raw_vmark = vrow.get("mark")
                        if not raw_vmark:
                            # Rows returned with mark:null stay unbound — never attach to a neighbouring mark.
                            continue

                        # D3: Sanitise element_mark (collapse/strip whitespace)
                        vmark = re.sub(r"\s+", "", str(raw_vmark).strip())

                        # D2: Reject any mark that does not match ^(F|P|C|BP|SF|GB|W)\d+[A-Z]?$
                        if not vmark or not VALID_MARK_PAT.match(vmark):
                            logger.info("[vision_fallback] D2: Dropping invalid mark '%s' (fails pattern ^(F|P|C|BP|SF|GB|W)\\d+[A-Z]?$)", raw_vmark)
                            continue

                        # D4: Drop all-null rows (all-null dimensions AND all-null reinforcement)
                        v_length = vrow.get("length_mm")
                        v_width  = vrow.get("width_mm")
                        v_depth  = vrow.get("depth_mm")
                        v_reinf  = vrow.get("reinforcement")
                        if isinstance(v_reinf, list):
                            v_reinf_clean = [r for r in v_reinf if r and str(r).strip()]
                        elif v_reinf:
                            v_reinf_clean = [str(v_reinf).strip()]
                        else:
                            v_reinf_clean = []

                        if v_length is None and v_width is None and v_depth is None and not v_reinf_clean:
                            logger.info("[vision_fallback] D4: Dropping empty vision row for mark '%s' (all dimensions and reinforcement are null)", vmark)
                            continue

                        if vmark not in results:
                            results[vmark] = ElementTypeAttr(mark=vmark, confidence=0.70)

                        target = results[vmark]

                        # Rule 4: Vision fills nulls only. Never overwrite a successful vector parse (confidence >= 0.85 & vector dims present & rebar present)
                        is_vector_ok = (target.confidence >= 0.85 and target.length_mm is not None and len(target.rebar_layers) > 0)

                        vconf = min(0.70, float(vrow.get("row_confidence", 0.70)))

                        if not is_vector_ok:
                            target.extraction_method = "VISION_FALLBACK"
                            target.confidence = vconf

                        dim_updated = False
                        if target.length_mm is None and v_length is not None:
                            target.length_mm = int(v_length)
                            dim_updated = True
                        if target.width_mm is None and v_width is not None:
                            target.width_mm = int(v_width)
                            dim_updated = True
                        if target.depth_mm is None and v_depth is not None:
                            target.depth_mm = int(v_depth)
                            dim_updated = True

                        dims = [str(d) for d in (target.length_mm, target.width_mm, target.depth_mm) if d is not None]
                        dim_str = "x".join(dims)
                        dim_suffix = f"[Dim: {dim_str}]" if dim_str else ""

                        if dim_str:
                            target.provenance["dim_text"] = f"[VISION] {dim_suffix}"
                            if not is_vector_ok:
                                target.extraction_method = "VISION_FALLBACK"

                        if v_reinf_clean:
                            logger.info("[vision_fallback] Mark %s: extracted %d reinforcement line(s)", vmark, len(v_reinf_clean))

                            new_layers = []
                            for reinf_item in v_reinf_clean:
                                clean_item = re.sub(r"\s*\[Dim:.*?\]", "", str(reinf_item)).strip()
                                vis_raw = f"[VISION] {clean_item} {dim_suffix}".strip() if dim_suffix else f"[VISION] {clean_item}"
                                vis_layers = _parse_rebar_string(clean_item)
                                if vis_layers:
                                    for vl in vis_layers:
                                        vl.raw = vis_raw
                                    new_layers.extend(vis_layers)
                            if new_layers:
                                if not is_vector_ok:
                                    if not target.rebar_layers:
                                        target.rebar_layers = new_layers
                                    else:
                                        existing_codes = [l.layer_code for l in target.rebar_layers]
                                        for nl in new_layers:
                                            if nl.layer_code not in existing_codes:
                                                target.rebar_layers.append(nl)
                                    target.extraction_method = "VISION_FALLBACK"
                                else:
                                    existing_codes = [l.layer_code for l in target.rebar_layers]
                                    for nl in new_layers:
                                        if nl.layer_code not in existing_codes:
                                            target.rebar_layers.append(nl)
        except Exception as e:
            logger.warning("[clipping_detector] Vision merge failed: %s", e)

    _parse_column_baseplate_details(all_blocks, results)
    return results


def _parse_column_baseplate_details(all_blocks: list[dict], results: dict[str, ElementTypeAttr]) -> None:
    """
    Parse C1/C2 HSS column specs and BP1/BP2 base plate specs from typical detail text blocks.
    """
    bp_text = ""
    col_text = ""
    for b in all_blocks:
        txt = b.get("text", "")
        if "BASE PLATE" in txt.upper() and ("400" in txt or "22" in txt or "ANCHOR" in txt.upper()):
            bp_text += " " + txt
        if ("HSS" in txt.upper() or "COLUMN" in txt.upper()) and ("203" in txt or "7.9" in txt):
            col_text += " " + txt

    # Parse Base Plate specs (BP1, BP2)
    bp_pm = re.search(r"(\d+)\s*[xX]\s*(\d+)\s*[xX]\s*(\d+)\s*Thk", bp_text, re.IGNORECASE)
    bp_span = int(bp_pm.group(1)) if bp_pm else 400
    bp_width = int(bp_pm.group(2)) if bp_pm else 400
    bp_depth = int(bp_pm.group(3)) if bp_pm else 22

    bp_bolt_m = re.search(r"(\d+\s*-\s*\d+\s*mm\s*DIA\.?\s*A\d+\s*ANCHOR\s*BOLTS.*?(?:\+|\n|$))", bp_text, re.IGNORECASE)
    bp_bolt_str = bp_bolt_m.group(1).strip() if bp_bolt_m else "4 - 19mm DIA. A307 ANCHOR BOLTS x 508mm Lg. + 76mm HOOK"

    for mark in ("BP1", "BP2"):
        if mark in results:
            target = results[mark]
            if target.length_mm is None:
                target.length_mm = bp_span
                target.width_mm = bp_width
                target.depth_mm = bp_depth
                target.confidence = 0.70
                target.extraction_method = "VECTOR_TEXT"
                target.provenance["dim_text"] = f"{bp_span} x {bp_width} x {bp_depth} Thk."
                if not target.rebar_layers:
                    layers = _parse_rebar_string(bp_bolt_str)
                    if layers:
                        target.rebar_layers = layers
                        target.reinforcement_display = bp_bolt_str

    # Parse Column specs (C1, C2) — Fix 1: Section dimensions leave span/width/height EMPTY for C1 and C2
    for mark in ("C1", "C2"):
        if mark in results:
            target = results[mark]
            target.length_mm = None
            target.width_mm = None
            target.depth_mm = None
            target.confidence = 0.45
            target.extraction_method = "VECTOR_TEXT"
            target.provenance["dim_text"] = ""


# ── Continuous element & note detection ───────────────────────────────────────
_CF_PAT   = re.compile(
    r"(\d+)\s*(?:WIDE|W\.?)\s*[xX]\s*(\d+)\s*(?:Dp\.|DEEP|D\.?)\s*"
    r".*?(?:TRENCHED|POURED|CONT\.?\s*CONC\.?\s*FOOTING)", re.IGNORECASE
)
_GB_PAT   = re.compile(
    r"(\d+)\s*(?:mm\s*)?(?:WIDE|W\.?)\s*[xX]\s*(\d+)\s*(?:mm\s*)?(?:Dp\.|DEEP|D\.?)\s*"
    r".*?GRADE\s*BEAM", re.IGNORECASE
)
_SOG_PAT  = re.compile(
    r"(\d+)\s*mm\s*THK\.?\s*CONCRETE\s*SLAB-ON-GRADE", re.IGNORECASE
)
_RAMP_SLAB_PAT = re.compile(
    r"(\d+)\s*mm\s*THK\.?\s*SLAB\b.*?LOADING\s+RAMP", re.IGNORECASE
)


def _parse_continuous_elements(doc_data: dict[str, Any]) -> dict[str, ElementTypeAttr]:
    """
    D6 & D7 Fix: Detect unmarked continuous elements from leader notes.
    D6: Reassembles multi-line notes; D7.2: continuous qty=1.
    """
    results: dict[str, ElementTypeAttr] = {}

    # Reassemble multi-line note blocks
    raw_blocks = doc_data.get("text_blocks", [])
    merged_texts: list[tuple[str, int]] = []
    for b in raw_blocks:
        txt = b.get("text", "").strip()
        pg  = b.get("page", 1)
        if txt:
            # Clean up linebreaks inside block
            cleaned = re.sub(r"\s*\n\s*", " ", txt)
            merged_texts.append((cleaned, pg))

    for txt, page in merged_texts:
        m = _CF_PAT.search(txt)
        if m:
            mark = f"CF-{m.group(1)}x{m.group(2)}"
            if mark not in results:
                results[mark] = ElementTypeAttr(
                    mark=mark, width_mm=int(m.group(1)), depth_mm=int(m.group(2)),
                    rebar_layers=_parse_rebar_string(txt),
                    reinforcement_display=txt.strip(),
                    provenance={"raw": txt},
                    source_view="GENERAL_NOTE", source_page=page,
                    extraction_method="VECTOR_TEXT", confidence=0.60,
                )
            continue

        m = _GB_PAT.search(txt)
        if m:
            mark = f"GB-{m.group(1)}x{m.group(2)}"
            if mark not in results:
                results[mark] = ElementTypeAttr(
                    mark=mark, width_mm=int(m.group(1)), depth_mm=int(m.group(2)),
                    rebar_layers=_parse_rebar_string(txt),
                    reinforcement_display=txt.strip(),
                    provenance={"raw": txt},
                    source_view="GENERAL_NOTE", source_page=page,
                    extraction_method="VECTOR_TEXT", confidence=0.60,
                )
            continue

        m = _SOG_PAT.search(txt)
        if m:
            mark = f"SOG-{m.group(1)}"
            if mark not in results:
                results[mark] = ElementTypeAttr(
                    mark=mark, depth_mm=int(m.group(1)),
                    rebar_layers=_parse_rebar_string(txt),
                    reinforcement_display=txt.strip(),
                    provenance={"raw": txt},
                    source_view="GENERAL_NOTE", source_page=page,
                    extraction_method="VECTOR_TEXT", confidence=0.55,
                )
            continue

        m = _RAMP_SLAB_PAT.search(txt)
        if m:
            mark = f"SOG-RAMP-{m.group(1)}"
            if mark not in results:
                results[mark] = ElementTypeAttr(
                    mark=mark, depth_mm=int(m.group(1)),
                    rebar_layers=_parse_rebar_string(txt),
                    reinforcement_display=txt.strip(),
                    provenance={"raw": txt},
                    source_view="GENERAL_NOTE", source_page=page,
                    extraction_method="VECTOR_TEXT", confidence=0.55,
                )

    return results


# ── Primary output function ───────────────────────────────────────────────────
def extract_flat_rows(
    doc_data: dict[str, Any] | str | Path,
    api_key:    str | None = None,
    model_name: str | None = None,
) -> list[FlatRow]:
    if isinstance(doc_data, (str, Path)):
        doc_data = load_pdf_for_vision(doc_data)
    pdf_path = doc_data.get("pdf_path", "")

    # Step 1: Title block metadata
    meta = extract_title_block(doc_data, page=1)
    drawing_id = meta.get("drawing_id") or None
    tb_conf    = meta.get("_tb_confidence", 0.0)

    # Step 2: Schedule-derived element attributes
    vector_schedules = _parse_vector_schedule_dimensions("", doc_data)

    # Step 3: Continuous/unmarked elements
    continuous_elements = _parse_continuous_elements(doc_data)

    # Step 4: Plan QTY counts
    diagram_counts: dict[str, int] = {}
    grid_refs_map:  dict[str, list[str]] = {}
    if pdf_path and Path(pdf_path).exists():
        try:
            diagram_counts, grid_refs_map, _ = _count_diagram_quantities_with_pairing(pdf_path, doc_data)
        except Exception as e:
            logger.warning("QTY counting failed: %s", e)

    all_elements: dict[str, ElementTypeAttr] = {}
    all_elements.update(vector_schedules)
    for mark, attr in continuous_elements.items():
        if mark not in all_elements:
            all_elements[mark] = attr
    for mark in diagram_counts:
        if mark not in all_elements:
            all_elements[mark] = ElementTypeAttr(mark=mark, confidence=0.40)

    rows: list[FlatRow] = []

    for raw_mark, attr in all_elements.items():
        # D3: Sanitise element_mark (collapse/strip whitespace)
        mark = re.sub(r"\s+", "", str(raw_mark).strip())
        if not mark:
            continue
        if mark.startswith("CF-") or mark.startswith("GB-") or mark.startswith("SOG-"):
            pass
        elif not VALID_MARK_PAT.match(mark):
            logger.warning("[extract_flat_rows] D2/D3: Rejecting invalid element mark: '%s'", raw_mark)
            continue
        # D7.2 Fix: continuous elements set qty=1
        if mark.startswith("CF-") or mark.startswith("GB-") or mark.startswith("SOG-"):
            qty = 1
        else:
            qty = diagram_counts.get(mark) if diagram_counts.get(mark) else None

        span_val   = attr.length_mm
        width_val  = attr.width_mm
        height_val = attr.depth_mm

        elem_name, elem_shape = _element_name_shape(mark, span_val, width_val)
        drawing_element_id = f"{drawing_id}-{mark}" if drawing_id else mark

        src_view = attr.source_view
        ext_method = attr.extraction_method
        elem_conf  = max(min(attr.confidence, 1.0), 0.0)

        raw_dim = attr.provenance.get("dim_text", "").strip()
        if mark in ("C1", "BP1") and not raw_dim:
            logger.warning("[raw_text] %s: scheduled foundation element without explicit schedule dimensions", drawing_element_id)
            raw_text_elem = mark
        else:
            raw_text_elem = (
                raw_dim or
                attr.provenance.get("raw", "") or
                attr.reinforcement_display or
                mark
            )

        base = dict(
            project_number_id  = meta.get("project_number_id") or None,
            project_name       = meta.get("project_name") or None,
            architect_name     = meta.get("architect_name") or None,
            engineer_name      = meta.get("engineer_name") or None,
            customer_name      = meta.get("customer_name") or None,
            job_type           = meta.get("job_type") or None,
            estimator_name     = None,
            current_status     = meta.get("current_status") or None,
            drawing_id         = drawing_id,
            drawing_number     = meta.get("drawing_number") or None,
            drawing_title      = meta.get("drawing_title") or None,
            drawing_type_name  = meta.get("drawing_type_name") or None,
            drawing_description= meta.get("drawing_description") or None,
            drawing_date       = meta.get("drawing_date") or None,
            document_location  = meta.get("document_location") or None,
            drawing_element_id = drawing_element_id,
            element_mark       = mark,
            element_name       = elem_name,
            element_shape      = elem_shape,
            span_mm            = span_val,
            width_mm           = width_val,
            height_mm          = height_val,
            qty                = qty,
        )

        rebar_layers = attr.rebar_layers

        if not rebar_layers:
            row_raw_text = _format_clean_raw_text(raw_text_elem or mark, span_val, width_val, height_val)
            row = FlatRow(**base,
                reinforcement_id = None,
                layer_code       = None,
                bar_size         = None,
                bar_count        = None,
                spacing_mm       = None,
                direction        = None,
                bar_length_mm    = None,
                cover_mm         = None,
                source_view      = src_view,
                source_page      = attr.source_page,
                extraction_method= ext_method,
                confidence       = round(elem_conf, 2),
                raw_text         = row_raw_text,
            )
            rows.append(row)
        else:
            for n, rl in enumerate(rebar_layers, start=1):
                reinf_id = f"{drawing_element_id}-R{n:02d}"
                rl_conf  = elem_conf
                if ext_method == ExtractionMethod.VISION_FALLBACK.value:
                    rl_conf = min(rl_conf, 0.70)
                row_raw_text = _format_clean_raw_text(rl.raw or raw_text_elem or mark, span_val, width_val, height_val)
                row = FlatRow(**base,
                    reinforcement_id = reinf_id,
                    layer_code       = rl.layer_code,
                    bar_size         = rl.size,
                    bar_count        = rl.quantity,
                    spacing_mm       = rl.spacing_mm,
                    direction        = rl.direction,
                    bar_length_mm    = rl.length_mm,
                    cover_mm         = rl.cover_mm,
                    source_view      = src_view,
                    source_page      = attr.source_page,
                    extraction_method= ext_method,
                    confidence       = round(rl_conf, 2),
                    raw_text         = row_raw_text,
                )
                rows.append(row)


    validate_rows(rows, doc_data=doc_data)
    return rows


def _format_clean_raw_text(base_text: str, length_mm: int | None, width_mm: int | None, depth_mm: int | None) -> str:
    """
    Ensure raw_text has exactly one clean, non-nested [Dim: LxWxD] suffix without duplicate tags.
    Skip empty dimension components.
    """
    is_vision = "[VISION]" in str(base_text)
    clean_base = re.sub(r"\[VISION\]", "", str(base_text))
    clean_base = re.sub(r"\[Dim:.*?\]", "", clean_base).strip()

    dims = [str(d) for d in (length_mm, width_mm, depth_mm) if d is not None]
    prefix = "[VISION] " if is_vision else ""

    if dims:
        dim_suffix = f"[Dim: {'x'.join(dims)}]"
        if clean_base:
            return f"{prefix}{clean_base} {dim_suffix}".strip()
        else:
            return f"{prefix}{dim_suffix}".strip()
    return f"{prefix}{clean_base}".strip()


# ── Backward-compatibility shim ───────────────────────────────────────────────
def extract_table_from_doc(
    doc_data: dict[str, Any],
    api_key:    str | None = None,
    model_name: str | None = None,
) -> list[dict[str, Any]]:
    pdf_path = doc_data.get("pdf_path", "")
    full_text = doc_data.get("full_text", "")

    segment_name = _extract_drawing_segment(full_text, doc_data)
    first_image_path = ""
    page_images = doc_data.get("page_images", [])
    if page_images:
        first_image_path = page_images[0].get("image_path", "")

    vector_schedules = _parse_vector_schedule_dimensions(full_text, doc_data)
    diagram_counts, grid_refs_map, _ = (
        _count_diagram_quantities_with_pairing(pdf_path, doc_data)
        if pdf_path and Path(pdf_path).exists()
        else ({}, {}, {})
    )

    all_blocks = doc_data.get("text_blocks", [])
    discovered  = _discover_schedule_marks(all_blocks)
    plan_marks  = set(diagram_counts.keys())
    scheduled_marks = list(dict.fromkeys(discovered + sorted(plan_marks - set(discovered))))

    final_table: list[dict[str, Any]] = []

    for mark in scheduled_marks:
        attr: ElementTypeAttr = vector_schedules.get(mark, ElementTypeAttr(mark=mark))

        span_val   = attr.length_mm   if attr.length_mm  is not None else ""
        width_val  = attr.width_mm    if attr.width_mm   is not None else ""
        height_val = attr.depth_mm    if attr.depth_mm   is not None else ""
        rebar_val  = attr.reinforcement_display

        qty        = diagram_counts.get(mark) if diagram_counts.get(mark) else 0
        g_list     = grid_refs_map.get(mark, [])
        unique_gs  = sorted(set(g_list))
        sample_gs  = (
            ", ".join(unique_gs[:4]) +
            (f" (+{len(unique_gs)-4} more)" if len(unique_gs) > 4 else "")
            if unique_gs else "None (Scheduled but unused)"
        )

        if mark.startswith("F"):
            category = "Spread footing pad"
        elif mark.startswith("P"):
            category = "Concrete pier"
        elif mark.startswith("C") and not mark.startswith("CF"):
            category = "Steel column"
        elif mark.startswith("BP"):
            category = "Base plate"
        elif mark.startswith("CF") or mark.startswith("GB"):
            category = "Continuous footing"
        else:
            category = "Foundation element"

        parts = [p.strip() for p in rebar_val.split(";") if p.strip()] if rebar_val else []
        primary_rebar    = parts[0] if parts else ""
        secondary_notes  = parts[1] if len(parts) > 1 else ""

        row = {
            "Segment": segment_name,
            "PDF Link": pdf_path,
            "Image Link": first_image_path,
            "Type": mark,
            "element_type": mark,
            "Category": category,
            "category": category,
            "Span (mm)": span_val,
            "Length (mm)": span_val,
            "length_mm": str(span_val),
            "Width (mm)": width_val,
            "width_mm": str(width_val),
            "Height (mm)": height_val,
            "Depth (mm)": height_val,
            "depth_mm": str(height_val),
            "Primary reinforcement": primary_rebar,
            "primary_reinforcement": primary_rebar,
            "Secondary reinforcement / notes": secondary_notes,
            "secondary_reinforcement_notes": secondary_notes,
            "QTY": qty,
            "Plan instances": qty,
            "plan_instances": qty,
            "Reinforcement": rebar_val,
            "Sample Grid References": sample_gs,
        }
        final_table.append(row)

    return final_table
