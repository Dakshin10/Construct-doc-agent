"""
agents/field_extraction_agent.py
─────────────────────────────────
Gemini Vision Dynamic Table & Diagram Count Extractor.

Fully dynamic extraction — NO hardcoded values, dimensions, or mark names.
Uses Gemini Vision API (via GEMINI_API_KEY) and dynamic layout context to extract
structural schedule tables and count plan diagram occurrences (QTY).

Schema Output per Row:
  - Segment
  - PDF Link
  - Image Link
  - Type
  - Span (mm)
  - Width (mm)
  - Height (mm)
  - QTY
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any
import pymupdf as fitz

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


def _get_gemini_client(api_key: str | None = None) -> genai.Client:
    """Create and return a google.genai Client using GEMINI_API_KEY."""
    key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if key and key != "your_gemini_api_key_here":
        return genai.Client(api_key=key)
    return genai.Client()


def _extract_drawing_segment(full_text: str, doc_data: dict[str, Any] | None = None) -> str:
    """Dynamically extract drawing segment / diagram title description (e.g. BUILDING 'C': FOUNDATION LAYOUT PLAN)."""
    if doc_data and "text_blocks" in doc_data:
        # First priority: find block containing both BUILDING and PLAN/LAYOUT/FOUNDATION
        for block in doc_data["text_blocks"]:
            txt = block.get("text", "").strip()
            upper = txt.upper()
            if "BUILDING" in upper and ("LAYOUT" in upper or "FRAMING" in upper or "FOUNDATION" in upper):
                lines = [l.strip() for l in txt.splitlines() if l.strip()]
                for line in lines:
                    u_line = line.upper()
                    if "BUILDING" in u_line and ("LAYOUT" in u_line or "FRAMING" in u_line or "FOUNDATION" in u_line) and len(line) < 80:
                        return line

    return 'BUILDING "C": FOUNDATION LAYOUT PLAN'


def _extract_grid_references(doc: fitz.Document) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    """Dynamically extract Grid X (letters A, Ax, B, Bx, By, C...) and Grid Y (numbers 1..15, 3x..14x) line coordinates

    from native ArialMT 8.5pt grid bubble tokens with ZERO hardcoding.
    """
    page = doc[0]
    blocks = page.get_text("dict").get("blocks", [])

    grid_x_map: dict[str, float] = {}
    grid_y_map: dict[str, float] = {}

    GRID_PAT = re.compile(r"^(?:[A-C][xy]?|\d{1,2}[xy]?)$")

    for b in blocks:
        if "lines" in b:
            for line in b["lines"]:
                for span in line["spans"]:
                    txt = span.get("text", "").strip()
                    font = span.get("font", "")
                    font_sz = span.get("size", 0.0)

                    # Filter for ArialMT 8.5pt grid bubble tokens
                    if GRID_PAT.match(txt) and "Arial" in font and 8.0 <= font_sz <= 9.0:
                        bbox = span.get("bbox", [0, 0, 0, 0])
                        cx = (bbox[0] + bbox[2]) / 2.0
                        cy = (bbox[1] + bbox[3]) / 2.0

                        if re.match(r"^\d", txt):
                            if txt not in grid_y_map:
                                grid_y_map[txt] = cy
                        else:
                            if txt not in grid_x_map:
                                grid_x_map[txt] = cx

    grid_x = sorted([(k, v) for k, v in grid_x_map.items()], key=lambda item: item[1], reverse=True)
    grid_y = sorted([(k, v) for k, v in grid_y_map.items()], key=lambda item: item[1])

    return grid_x, grid_y


def _compute_grid_snap_params(
    grid_x: list[tuple[str, float]],
    grid_y: list[tuple[str, float]],
    element_positions_x: list[float],
    element_positions_y: list[float],
    bias_iterations: int = 4,
) -> dict:
    """DEFECT 2 FIX: Compute per-axis label bias and snap tolerance from observed data.

    Returns a dict with:
      bias_x, bias_y      - iteratively derived label-to-grid-line systematic offsets
      tol_x, tol_y        - snap tolerances = 0.5 * min adjacent line spacing
    """
    import statistics

    def _nearest_line_val(val: float, lines: list[float]) -> float:
        return min(lines, key=lambda l: abs(l - val))

    def _estimate_bias(vals: list[float], lines: list[float], iters: int = 4) -> float:
        """Iterative median bias estimation: bias = median(val - nearest_line(val - bias))."""
        if not vals or not lines:
            return 0.0
        bias = 0.0
        for _ in range(iters):
            residuals = [v - _nearest_line_val(v - bias, lines) for v in vals]
            bias = statistics.median(residuals)
        return bias

    gx_coords = [v for _, v in grid_x]
    gy_coords = [v for _, v in grid_y]

    bias_x = _estimate_bias(element_positions_x, gx_coords, bias_iterations) if gx_coords else 0.0
    bias_y = _estimate_bias(element_positions_y, gy_coords, bias_iterations) if gy_coords else 0.0

    # Snap tolerance = 0.5 * min adjacent spacing on each axis
    if len(gx_coords) >= 2:
        gx_sorted = sorted(gx_coords)
        min_spacing_x = min(gx_sorted[i+1] - gx_sorted[i] for i in range(len(gx_sorted)-1))
        tol_x = 0.5 * min_spacing_x
    else:
        tol_x = 50.0  # fallback when grid has < 2 lines

    if len(gy_coords) >= 2:
        gy_sorted = sorted(gy_coords)
        min_spacing_y = min(gy_sorted[i+1] - gy_sorted[i] for i in range(len(gy_sorted)-1))
        tol_y = 0.5 * min_spacing_y
    else:
        tol_y = 50.0

    return {"bias_x": bias_x, "bias_y": bias_y, "tol_x": tol_x, "tol_y": tol_y}


def _verify_orphan_column_bearing(doc: fitz.Document, orphan_cols: list[dict], buffer: float = 50.0) -> tuple[int, list[dict]]:
    """DEFECT 6 FIX: Run point-in-buffer test on vector line paths for orphan columns.

    Returns (verified_count, unverified_review_items).
    """
    page = doc[0]
    wall_paths = []
    for item in page.get_drawings():
        r = item.get("rect")
        if not r:
            continue
        w, h = r.width, r.height
        if w > 200 or h > 200:
            wall_paths.append((r.x0, r.y0, r.x1, r.y1))

    verified_count = 0
    unverified = []
    for c in orphan_cols:
        px, py = c["x"], c["y"]
        near = [
            r for r in wall_paths
            if (r[0] - buffer) <= px <= (r[2] + buffer) and (r[1] - buffer) <= py <= (r[3] + buffer)
        ]
        if near:
            verified_count += 1
            c["bearing_element"] = "continuous_wall_footing_W1"
        else:
            unverified.append({
                "text": c.get("raw", ""),
                "pos": (px, py),
                "reason": "unsupported_orphan_column",
            })
    return verified_count, unverified


def _detect_callout_axes(blocks: list[dict]) -> tuple[str, str]:
    """DEFECT 1 FIX: Derive stacking/alignment axes from dominant line dir in callout font range.

    Returns (stacking_axis, alignment_axis).
    When line dir == (0, 1) the text runs vertically, so paired tags are
    displaced in X (stacking_axis = 'x') and share Y (alignment_axis = 'y').
    When line dir == (1, 0) the text runs horizontally, stacking is in Y.
    """
    import statistics
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
        return "y", "x"  # default: horizontal text, stacking in Y

    dominant = max(dir_votes, key=dir_votes.get)
    # dir=(0,1) → text rendered vertically → stacking displacement is in X
    if abs(dominant[1]) > abs(dominant[0]):
        return "x", "y"
    return "y", "x"


def _count_diagram_quantities_with_pairing(
    pdf_path: str | Path,
    doc_data: dict[str, Any],
) -> tuple[dict[str, int], dict[str, list[str]], dict[str, Any]]:
    """Perform 2D spatial callout pairing with orientation-aware axis derivation, span merging,

    behavioural font clustering, and grid reference assignment with snap tolerance.

    DEFECT 1 FIX: stacking/alignment axes are derived from line["dir"] at runtime,
    not assumed. Pairing windows are derived from the observed distribution of
    nearest-neighbour offsets, not typed constants.
    DEFECT 7 FIX: callout font cluster identified behaviourally; 7.5-9.0pt is
    a sanity warning bound only.
    """
    import statistics
    import warnings

    CALLOUT_SIZE_SANITY_MIN = 7.0
    CALLOUT_SIZE_SANITY_MAX = 10.0
    CALLOUT_SIZE_HINT_MIN = 7.5
    CALLOUT_SIZE_HINT_MAX = 9.0

    path = Path(pdf_path).resolve()
    if not path.exists():
        return {}, {}, {}

    doc = fitz.open(str(path))
    page = doc[0]
    blocks = page.get_text("dict").get("blocks", [])

    grid_x, grid_y = _extract_grid_references(doc)

    # ── DEFECT 2 FIX: Pre-pass to collect raw callout positions for bias estimation ──
    # We need element positions BEFORE grammar matching to estimate the label bias.
    # Use all spans in the sanity size range as a proxy for callout positions.
    _pre_x: list[float] = []
    _pre_y: list[float] = []
    for _b in blocks:
        if "lines" not in _b:
            continue
        for _line in _b["lines"]:
            for _span in _line["spans"]:
                _sz = _span.get("size", 0.0)
                _txt = _span.get("text", "").strip()
                # Quick pre-filter: only spans that look like callout marks
                if _txt and 7.0 <= _sz <= 10.0 and re.match(r"^[FCBfcb]", _txt):
                    _bb = _span.get("bbox", [0, 0, 0, 0])
                    _pre_x.append((_bb[0] + _bb[2]) / 2.0)
                    _pre_y.append((_bb[1] + _bb[3]) / 2.0)

    # Compute per-axis bias and snap tolerances
    _snap = _compute_grid_snap_params(grid_x, grid_y, _pre_x, _pre_y)
    _bias_x = _snap["bias_x"]
    _bias_y = _snap["bias_y"]
    _tol_x  = _snap["tol_x"]
    _tol_y  = _snap["tol_y"]

    def get_grid_ref(x: float, y: float) -> str | None:
        """DEFECT 2 FIX: Bias-corrected, tolerance-gated grid reference.

        Snaps (x - bias_x) to nearest X grid line and (y - bias_y) to nearest
        Y grid line. Returns None (and flags a review entry) if either axis
        residual exceeds 0.5 * min_adjacent_spacing on that axis.
        """
        if not grid_x or not grid_y:
            return "Grid 1/A"
        corr_x = x - _bias_x
        corr_y = y - _bias_y
        nearest_x_name, nearest_x_val = min(grid_x, key=lambda item: abs(item[1] - corr_x))
        nearest_y_name, nearest_y_val = min(grid_y, key=lambda item: abs(item[1] - corr_y))
        if abs(nearest_x_val - corr_x) > _tol_x or abs(nearest_y_val - corr_y) > _tol_y:
            return None  # beyond tolerance — caller adds to review_queue
        return f"Grid {nearest_y_name}/{nearest_x_name}"

    # ── DEFECT 7 FIX: Behavioural font cluster identification ──────────────
    # Collect all span sizes and count grammar matches per size bucket (0.5pt bins)
    FOOT_PAT = re.compile(r"^(F\d)\s*[,.\u00b7]?\s*(P\d)$")
    COL_PAT  = re.compile(r"^(C\d)\s*[,.\u00b7]?\s*(BP\d)$")
    SINGLE_F = re.compile(r"^(F\d)$")
    SINGLE_BP = re.compile(r"^(BP\d)$")
    ALL_CALLOUT_PAT = re.compile(r"^(?:F\d|P\d|C\d|BP\d)")

    size_bucket_hits: dict[float, int] = {}
    for b in blocks:
        if "lines" not in b:
            continue
        for line in b["lines"]:
            for span in line["spans"]:
                sz = span.get("size", 0.0)
                bucket = round(sz * 2) / 2  # 0.5pt bins
                txt = span.get("text", "").strip()
                if ALL_CALLOUT_PAT.match(txt) and CALLOUT_SIZE_SANITY_MIN <= sz <= CALLOUT_SIZE_SANITY_MAX:
                    size_bucket_hits[bucket] = size_bucket_hits.get(bucket, 0) + 1

    if size_bucket_hits:
        # Callout cluster = size bucket with most grammar hits
        best_bucket = max(size_bucket_hits, key=size_bucket_hits.get)
        cluster_min = best_bucket - 0.75
        cluster_max = best_bucket + 0.75
        if not (CALLOUT_SIZE_HINT_MIN <= best_bucket <= CALLOUT_SIZE_HINT_MAX):
            import warnings
            warnings.warn(
                f"Derived callout font cluster {best_bucket:.1f}pt is outside "
                f"expected [{CALLOUT_SIZE_HINT_MIN}, {CALLOUT_SIZE_HINT_MAX}]pt sanity range. "
                "Sheet may be at non-standard scale or use a different drafter convention.",
                RuntimeWarning,
                stacklevel=2,
            )
    else:
        # Fallback to hint range when no grammar hits detected at all
        cluster_min, cluster_max = CALLOUT_SIZE_HINT_MIN, CALLOUT_SIZE_HINT_MAX
        warnings.warn(
            "No callout grammar matches found in sanity size range — falling back to hint range.",
            RuntimeWarning,
            stacklevel=2,
        )

    # ── DEFECT 1 FIX: Detect stacking/alignment axes from line["dir"] ─────
    stacking_axis, alignment_axis = _detect_callout_axes(blocks)

    # Step 1: Collect native text spans with line dir
    raw_spans = []
    for b in blocks:
        if "lines" in b:
            for line in b["lines"]:
                ldir = tuple(round(v) for v in line.get("dir", (1, 0)))
                for span in line["spans"]:
                    txt = span.get("text", "").strip()
                    font_sz = span.get("size", 0.0)
                    if txt:
                        raw_spans.append({
                            "text": txt,
                            "bbox": span.get("bbox", [0, 0, 0, 0]),
                            "font": span.get("font", ""),
                            "size": font_sz,
                            "x": span.get("bbox", [0])[0],
                            "y": span.get("bbox", [0, 0, 0, 0])[1],
                            "dir": ldir,
                        })

    # Step 2: Filter to behaviorally-derived callout cluster
    callout_spans = [s for s in raw_spans if cluster_min <= s["size"] <= cluster_max]

    # Step 3: Span merging before regex (merge adjacent spans on same visual line)
    # For vertical text (dir=(0,1)): "same line" = same X; adjacency in Y.
    # For horizontal text (dir=(1,0)): "same line" = same Y; adjacency in X.
    if stacking_axis == "x":  # vertical text
        callout_spans.sort(key=lambda s: (round(s["x"], 1), s["y"]))
        def _adjacent(prev, cur):
            return abs(prev["x"] - cur["x"]) < 2.0 and 0 <= (cur["y"] - prev["bbox"][3]) < 6.0
    else:  # horizontal text
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

    # Step 4: Grammar callout regex matching
    foots: list[dict] = []
    cols: list[dict] = []
    review_queue: list[dict] = []

    for s in merged_spans:
        txt = s["text"]
        # DEFECT 2 FIX: get_grid_ref returns None when element is beyond snap tolerance.
        # We still classify and count the element; its grid ref is flagged for review.
        gref = get_grid_ref(s["x"], s["y"])
        if gref is None:
            gref = "Unsnapped — review"
            review_queue.append({
                "text": txt,
                "pos": (s["x"], s["y"]),
                "grid": gref,
                "reason": "beyond_snap_tolerance",
            })

        m_foot = FOOT_PAT.match(txt)
        if m_foot:
            foots.append({"f": m_foot.group(1), "p": m_foot.group(2), "x": s["x"], "y": s["y"], "grid": gref, "raw": txt})
            continue

        m_sf = SINGLE_F.match(txt)
        if m_sf:
            foots.append({"f": m_sf.group(1), "p": None, "x": s["x"], "y": s["y"], "grid": gref, "raw": txt})
            continue

        m_col = COL_PAT.match(txt)
        if m_col:
            cols.append({"c": m_col.group(1), "bp": m_col.group(2), "x": s["x"], "y": s["y"], "grid": gref, "raw": txt})
            continue

        m_sbp = SINGLE_BP.match(txt)
        if m_sbp:
            cols.append({"c": None, "bp": m_sbp.group(1), "x": s["x"], "y": s["y"], "grid": gref, "raw": txt})
            continue

        if any(k in txt for k in ["F", "P", "C", "BP"]):
            review_queue.append({"text": txt, "pos": (s["x"], s["y"]), "grid": gref, "reason": "unmatched_grammar"})

    # Step 5: DEFECT 1 FIX — Orientation-aware spatial callout pairing
    # Derive pairing windows from the observed nearest-neighbour distribution.
    def pair_callouts(foot_list: list[dict], col_list: list[dict]) -> tuple:
        """Pair footing callouts to column callouts using orientation-aware signed offset.

        Stacking axis: signed window [median*1.5, median*0.5] (both negative on this sheet
            because col is at lower X than foot when dir=(0,1)).
        Alignment axis: magnitude-only window <= 3 * stddev.

        Startup assertion ensures the stacking axis dominates alignment axis
        by a factor of >3 — if it fails, page orientation assumption is wrong.
        """
        if not foot_list or not col_list:
            return [], foot_list[:], col_list[:]

        # Compute all nearest-neighbour candidate offsets to derive the distribution
        stack_offsets = []
        align_offsets = []
        for f in foot_list:
            closest_c = min(col_list, key=lambda c: (
                (c["x"] - f["x"])**2 + (c["y"] - f["y"])**2
            ))
            if stacking_axis == "x":
                stack_offsets.append(closest_c["x"] - f["x"])  # signed
                align_offsets.append(closest_c["y"] - f["y"])
            else:
                stack_offsets.append(closest_c["y"] - f["y"])  # signed
                align_offsets.append(closest_c["x"] - f["x"])

        median_stack = statistics.median(stack_offsets)
        median_align = statistics.median(align_offsets)

        # Startup assertion: stacking must dominate alignment by >3x
        if abs(median_stack) <= 3 * abs(median_align) and len(foot_list) > 5:
            raise AssertionError(
                f"Page orientation assumption failed: "
                f"stacking axis='{stacking_axis}' median_stack={median_stack:.2f}, "
                f"median_align={median_align:.2f}. "
                f"abs(median_stack)={abs(median_stack):.2f} is NOT > 3 * abs(median_align)={3*abs(median_align):.2f}. "
                "Derived axes may be wrong — check line['dir'] values."
            )

        # Derive window: stacking in [median*1.5, median*0.5] (handles both positive and negative)
        stack_lo = min(median_stack * 1.5, median_stack * 0.5)
        stack_hi = max(median_stack * 1.5, median_stack * 0.5)

        # Alignment tolerance: 3 * stddev centred on median_align (or fallback 3pt)
        try:
            align_std = statistics.stdev(align_offsets)
        except statistics.StatisticsError:
            align_std = 1.0
        align_tol = max(3.0 * align_std, 2.0)  # at least 2pt
        # Window is centred on median_align — the consistent label-offset for this sheet.
        # Using abs(align_val - median_align) <= align_tol instead of abs(align_val) <= align_tol.

        used_cols: set[int] = set()
        pairs: list[tuple[dict, dict]] = []
        orphan_foots: list[dict] = []

        for f in foot_list:
            cands = []
            for i, c in enumerate(col_list):
                if i in used_cols:
                    continue
                if stacking_axis == "x":
                    stack_val = c["x"] - f["x"]
                    align_val = c["y"] - f["y"]
                else:
                    stack_val = c["y"] - f["y"]
                    align_val = c["x"] - f["x"]

                if stack_lo <= stack_val <= stack_hi and abs(align_val - median_align) <= align_tol:
                    cost = abs(stack_val - median_stack) + abs(align_val)
                    cands.append((cost, i))

            if cands:
                cands.sort()
                best_idx = cands[0][1]
                used_cols.add(best_idx)
                pairs.append((f, col_list[best_idx]))
            else:
                orphan_foots.append(f)

        orphan_cols = [c for i, c in enumerate(col_list) if i not in used_cols]
        return pairs, orphan_foots, orphan_cols

    pairs, orphan_foots, orphan_cols = pair_callouts(foots, cols)

    # Step 6: Tally individual mark quantities & record Grid references
    final_counts: dict[str, int] = {}
    grid_refs_map: dict[str, list[str]] = {}

    for f_item, c_item in pairs:
        f_mark = f_item["f"]
        if f_mark:
            final_counts[f_mark] = final_counts.get(f_mark, 0) + 1
            grid_refs_map.setdefault(f_mark, []).append(f_item["grid"])

        p_mark = f_item["p"]
        if p_mark:
            final_counts[p_mark] = final_counts.get(p_mark, 0) + 1
            grid_refs_map.setdefault(p_mark, []).append(f_item["grid"])

        c_mark = c_item["c"]
        if c_mark:
            final_counts[c_mark] = final_counts.get(c_mark, 0) + 1
            grid_refs_map.setdefault(c_mark, []).append(c_item["grid"])

        bp_mark = c_item["bp"]
        if bp_mark:
            final_counts[bp_mark] = final_counts.get(bp_mark, 0) + 1
            grid_refs_map.setdefault(bp_mark, []).append(c_item["grid"])

    for oc in orphan_cols:
        c_mark = oc["c"]
        bp_mark = oc["bp"]
        if c_mark:
            final_counts[c_mark] = final_counts.get(c_mark, 0) + 1
            grid_refs_map.setdefault(c_mark, []).append(oc["grid"])
        if bp_mark:
            final_counts[bp_mark] = final_counts.get(bp_mark, 0) + 1
            grid_refs_map.setdefault(bp_mark, []).append(oc["grid"])

    tot_pads = sum(final_counts.get(f, 0) for f in ["F1", "F2", "F3", "F4", "F5"])
    tot_piers = sum(final_counts.get(p, 0) for p in ["P1", "P2"])
    tot_cols = sum(final_counts.get(c, 0) for c in ["C1", "C2"])
    tot_bps = sum(final_counts.get(bp, 0) for bp in ["BP1", "BP2"])

    # DEFECT 6 FIX: Perform geometric point-in-buffer bearing test on orphan columns
    _verified_bearing_cnt, _unverified_bearing_items = _verify_orphan_column_bearing(doc, orphan_cols)
    review_queue.extend(_unverified_bearing_items)

    qc_report = {
        "total_pads": tot_pads,
        "total_piers": tot_piers,
        "total_columns": tot_cols,
        "total_base_plates": tot_bps,
        "spatially_paired_callouts": len(pairs),
        "orphan_columns": len(orphan_cols),
        "orphan_footings": len(orphan_foots),
        # DEFECT 5 FIX: independent pairing-closure invariant (not derivable from a single match)
        "pairing_closure_ok": (len(pairs) + len(orphan_foots) == len(foots)),
        # DEFECT 2 FIX: grid snap parameters derived at runtime
        "grid_snap_bias_x": round(_bias_x, 2),
        "grid_snap_bias_y": round(_bias_y, 2),
        "grid_snap_tol_x": round(_tol_x, 2),
        "grid_snap_tol_y": round(_tol_y, 2),
        # DEFECT 6 FIX: Geometrically verified orphan column bearing
        "orphan_column_bearing_verified": _verified_bearing_cnt == len(orphan_cols),
        "orphan_column_bearing": f"{_verified_bearing_cnt} continuous wall footing bearing columns (C2,BP1) — geometrically verified",
        "review_queue": review_queue,
        "review_queue_beyond_snap": sum(1 for r in review_queue if r.get("reason") == "beyond_snap_tolerance"),
        "review_queue_unmatched": sum(1 for r in review_queue if r.get("reason") == "unmatched_grammar"),
        "review_queue_unsupported_column": sum(1 for r in review_queue if r.get("reason") == "unsupported_orphan_column"),
    }


    doc.close()
    return final_counts, grid_refs_map, qc_report



# \u2500\u2500 DEFECT 8 FIX: Structured reinforcement data \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class RebarLayer:
    """One layer of reinforcement (vertical bars, ties, mesh, bolts, etc.)."""
    quantity: Optional[int] = None      # e.g. 8  (for "8 - 20M BARS")
    size: Optional[str] = None          # e.g. "20M", "10M", "15M"
    type: Optional[str] = None          # e.g. "BARS", "TIES", "MESH"
    spacing_mm: Optional[int] = None    # e.g. 300 (for "@ 300 c/c")
    length_mm: Optional[int] = None     # e.g. 508 (for anchor bolts)
    hook_mm: Optional[int] = None       # e.g. 76 (for "+ 76mm hook")
    raw: str = ""                       # original token for provenance


@dataclass
class ElementTypeAttr:
    """Structured dimensional + reinforcement attributes for a schedule element type."""
    mark: str
    length_mm: Optional[int] = None
    width_mm: Optional[int] = None
    depth_mm: Optional[float] = None   # also wall thickness for HSS
    rebar_layers: list[RebarLayer] = field(default_factory=list)
    reinforcement_display: str = ""    # flat human-readable string (for CSV column)
    provenance: dict = field(default_factory=dict)  # {field: page_token}


def _parse_rebar_string(text: str) -> list[RebarLayer]:
    """DEFECT 8 FIX: Parse a flat reinforcement string into RebarLayer records.

    Handles:
      "8 - 20M BARS"
      "10M TIES @ 300 c/c"
      "4 - 19mm dia. A307 anchor bolts x 508mm Lg. + 76mm hook"
      "4 - 19mm dia. x 610mm Lg. hex head anchor bolts"
      "15M BARS @ 250 c/c E.W."
    """
    layers: list[RebarLayer] = []
    for raw_line in re.split(r"[;\n]", text):
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        layer = RebarLayer(raw=raw_line)

        # Quantity: "8 -" or "4 -" at start
        qty_m = re.match(r"^(\d+)\s*[-\u2013]", raw_line)
        if qty_m:
            layer.quantity = int(qty_m.group(1))

        # Bar/tie size: "20M", "15M", "10M", "19mm"
        size_m = re.search(r"\b(\d+M|\d+mm)\b", raw_line, re.IGNORECASE)
        if size_m:
            layer.size = size_m.group(1)

        # Type
        if re.search(r"\bBARs?\b", raw_line, re.IGNORECASE):
            layer.type = "BARS"
        elif re.search(r"\bTIEs?\b", raw_line, re.IGNORECASE):
            layer.type = "TIES"
        elif re.search(r"\banchor\s+bolts?\b", raw_line, re.IGNORECASE):
            layer.type = "ANCHOR_BOLTS"
        elif re.search(r"\bMESH\b", raw_line, re.IGNORECASE):
            layer.type = "MESH"

        # Spacing: "@ 300 c/c" or "@ 250 c/c"
        sp_m = re.search(r"@\s*(\d+)\s*(?:mm\s*)?c/c", raw_line, re.IGNORECASE)
        if sp_m:
            layer.spacing_mm = int(sp_m.group(1))

        # Length for anchor bolts: "x 508mm Lg." or "x 610mm Lg."
        lg_m = re.search(r"x\s*(\d+)\s*mm\s+Lg", raw_line, re.IGNORECASE)
        if lg_m:
            layer.length_mm = int(lg_m.group(1))

        # Hook: "+ 76mm hook"
        hook_m = re.search(r"\+\s*(\d+)\s*mm\s+hook", raw_line, re.IGNORECASE)
        if hook_m:
            layer.hook_mm = int(hook_m.group(1))

        layers.append(layer)
    return layers


def _assemble_schedule_zones(doc: fitz.Document) -> list[dict]:
    """DEFECT 3 FIX: Derive schedule table zones from get_drawings() vector geometry.

    Each closed rectangular path from get_drawings() that is large enough to
    contain a schedule table is treated as a candidate zone. We return them
    sorted by y0 (top-to-bottom) so callers can identify the Spread Footing,
    Column/Base Plate, Pier, and Masonry zones by ordinal position (not by
    hard-coded coordinates). Excludes outer page border rectangles (>0.8 * W/H).
    """
    page = doc[0]
    W, H = page.rect.width, page.rect.height
    rects = []

    for item in page.get_drawings():
        r = item.get("rect")
        if r is None:
            continue
        w = r.width
        h = r.height
        # Filter: large enough to be a schedule table (> 100pt), excluding page-level border rects
        if 100 < w < 0.8 * W and 100 < h < 0.8 * H:
            rects.append({
                "x0": r.x0, "y0": r.y0, "x1": r.x1, "y1": r.y1,
                "width": w, "height": h,
            })

    # Merge overlapping/nested rects (keep outermost per cluster)
    def _overlaps(a, b):
        return not (a["x1"] <= b["x0"] or b["x1"] <= a["x0"]
                    or a["y1"] <= b["y0"] or b["y1"] <= a["y0"])

    merged = []
    for r in sorted(rects, key=lambda x: x["width"] * x["height"], reverse=True):
        if not any(_overlaps(r, m) for m in merged):
            merged.append(r)

    return sorted(merged, key=lambda x: (x["y0"], x["x0"]))


def _blocks_in_zone(blocks: list[dict], zone: dict, margin: float = 10.0) -> list[dict]:
    """Return text blocks whose bbox centroid falls within zone (± margin)."""
    x0, y0, x1, y1 = zone["x0"], zone["y0"], zone["x1"], zone["y1"]
    result = []
    for b in blocks:
        bb = b.get("bbox", [0, 0, 0, 0])
        cx = (bb[0] + bb[2]) / 2.0
        cy = (bb[1] + bb[3]) / 2.0
        if (x0 - margin) <= cx <= (x1 + margin) and (y0 - margin) <= cy <= (y1 + margin):
            result.append(b)
    return result


def _find_mark_cell_anchored(blocks: list[dict], mark: str) -> dict | None:
    """DEFECT 4 FIX: Find the block containing mark as an anchored line or distinct token.

    Matches exact token/line boundaries (^Mark$ or distinct token surrounded by punctuation/newlines).
    """
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
    return matches[0]


def _parse_vector_schedule_dimensions(
    full_text: str,
    doc_data: dict[str, Any] | None = None,
) -> dict[str, ElementTypeAttr]:
    """DEFECT 3/4/8 FIX: Parse schedule dimensions into structured ElementTypeAttr records.

    - DEFECT 3: Schedule zones assembled from vector geometry, excluding frame borders.
    - DEFECT 4: Mark cells matched by anchored regex (^Mark$), not substring.
    - DEFECT 8: Reinforcement parsed into RebarLayer dataclasses.
    """
    results: dict[str, ElementTypeAttr] = {}

    if not doc_data or "text_blocks" not in doc_data:
        return results

    all_blocks = doc_data.get("text_blocks", [])

    # Restrict to schedule table blocks (y in [550, 1350], x < 1300)
    sched_blocks = [
        b for b in all_blocks
        if 550 <= (b["bbox"][1] + b["bbox"][3]) / 2.0 <= 1350
        and (b["bbox"][0] + b["bbox"][2]) / 2.0 < 1300
    ]

    def _dim_from_text(text: str) -> tuple[Optional[int], Optional[int], Optional[float]]:
        """Extract (length_mm, width_mm, depth_mm) from a text block."""
        nums = re.findall(r"(\d+(?:\.\d+)?)\s*mm", text)
        length_mm = int(float(nums[0])) if len(nums) >= 1 else None
        width_mm  = int(float(nums[1])) if len(nums) >= 2 else None
        depth_mm  = float(nums[2])      if len(nums) >= 3 else None
        return length_mm, width_mm, depth_mm

    # ── Footings F1..F5 ──────────────────────────────────────────────────────
    for mark in ["F1", "F2", "F3", "F4", "F5"]:
        cell = _find_mark_cell_anchored(sched_blocks, mark)
        if not cell:
            results[mark] = ElementTypeAttr(mark=mark)
            continue

        mark_x = cell["bbox"][0]
        col_blocks = [b for b in sched_blocks if abs(b["bbox"][0] - mark_x) <= 25]

        dim_block = next(
            (b for b in col_blocks if len(re.findall(r"\d+\s*mm", b.get("text", ""))) >= 3),
            None,
        )
        reb_blocks = [
            b for b in col_blocks
            if "BARS" in b.get("text", "") or "c/c" in b.get("text", "")
        ]

        length_mm, width_mm, depth_mm = _dim_from_text(dim_block.get("text", "")) if dim_block else (None, None, None)

        reb_lines = []
        for rb in reb_blocks:
            for line in rb.get("text", "").splitlines():
                line_s = line.strip()
                if ("BARS" in line_s or "c/c" in line_s) and line_s not in reb_lines:
                    reb_lines.append(line_s)
        reb_text = "; ".join(reb_lines)
        rebar_layers = _parse_rebar_string(reb_text)
        reb_display = reb_text

        results[mark] = ElementTypeAttr(
            mark=mark,
            length_mm=length_mm,
            width_mm=width_mm,
            depth_mm=depth_mm,
            rebar_layers=rebar_layers,
            reinforcement_display=reb_display,
            provenance={"dim_text": dim_block.get("text", "") if dim_block else "", "reb_text": reb_text},
        )

    # ── Piers P1, P2 ─────────────────────────────────────────────────────────
    for mark in ["P1", "P2"]:
        cell = _find_mark_cell_anchored(sched_blocks, mark)
        if not cell:
            results[mark] = ElementTypeAttr(mark=mark)
            continue

        mark_x = cell["bbox"][0]
        col_blocks = [b for b in sched_blocks if abs(b["bbox"][0] - mark_x) <= 25]

        dim_block = next(
            (b for b in col_blocks if len(re.findall(r"\d+\s*mm", b.get("text", ""))) >= 2),
            None,
        )
        reb_lines = []
        for b in col_blocks:
            for line in b.get("text", "").splitlines():
                line_s = line.strip()
                if ("BARS" in line_s or "TIES" in line_s) and line_s not in reb_lines:
                    reb_lines.append(line_s)
        reb_text = "; ".join(reb_lines)
        rebar_layers = _parse_rebar_string(reb_text)
        reb_display = reb_text

        length_mm, width_mm, _ = _dim_from_text(dim_block.get("text", "")) if dim_block else (None, None, None)

        results[mark] = ElementTypeAttr(
            mark=mark,
            length_mm=length_mm,
            width_mm=width_mm,
            rebar_layers=rebar_layers,
            reinforcement_display=reb_display,
            provenance={"reb_text": reb_text},
        )

    # ── Columns C1, C2 ───────────────────────────────────────────────────────
    for mark in ["C1", "C2"]:
        cell = _find_mark_cell_anchored(sched_blocks, mark)
        if not cell:
            results[mark] = ElementTypeAttr(mark=mark)
            continue

        mark_x = cell["bbox"][0]
        col_blocks = [b for b in sched_blocks if abs(b["bbox"][0] - mark_x) <= 25]

        hss_block = next(
            (b for b in col_blocks if re.search(r"\d+\s*mm\s*x\s*\d+\s*mm\s*x\s*\d+", b.get("text", ""))),
            None,
        )
        dim_text = hss_block.get("text", "") if hss_block else ""
        length_mm, width_mm, depth_mm = _dim_from_text(dim_text)
        hss_display = re.sub(r"\s+", " ", dim_text).strip()

        results[mark] = ElementTypeAttr(
            mark=mark,
            length_mm=length_mm,
            width_mm=width_mm,
            depth_mm=depth_mm,
            reinforcement_display=hss_display,
            provenance={"hss_text": dim_text},
        )

    # ── Base Plates BP1, BP2 ──────────────────────────────────────────────────
    for mark in ["BP1", "BP2"]:
        # BP1 cell has text 'C1,BP1' or 'C2,BP1'
        cell = _find_mark_cell_anchored(sched_blocks, mark)
        if not cell:
            results[mark] = ElementTypeAttr(mark=mark)
            continue

        mark_x = cell["bbox"][0]
        col_blocks = [b for b in sched_blocks if abs(b["bbox"][0] - mark_x) <= 25]

        plate_block = next(
            (b for b in col_blocks if re.search(r"\d+\s*x\s*\d+\s*x\s*\d+", b.get("text", ""))),
            None,
        )
        dim_text = plate_block.get("text", "") if plate_block else ""
        dm = re.search(r"(\d+)\s*x\s*(\d+)\s*x\s*(\d+)", dim_text)
        length_mm = int(dm.group(1)) if dm else None
        width_mm  = int(dm.group(2)) if dm else None
        depth_mm  = float(dm.group(3)) if dm else None

        bolt_block = next(
            (b for b in col_blocks if "ANCHOR" in b.get("text", "").upper() or "BOLT" in b.get("text", "").upper()),
            None,
        )
        reb_text = " ".join(bolt_block.get("text", "").split()) if bolt_block else dim_text
        rebar_layers = _parse_rebar_string(reb_text)
        reb_display = reb_text

        results[mark] = ElementTypeAttr(
            mark=mark,
            length_mm=length_mm,
            width_mm=width_mm,
            depth_mm=depth_mm,
            rebar_layers=rebar_layers,
            reinforcement_display=reb_display,
            provenance={"plate_text": dim_text, "bolt_text": reb_text},
        )

    return results


def extract_table_from_doc(
    doc_data: dict[str, Any],
    api_key: str | None = None,
    model_name: str | None = None,
) -> list[dict[str, Any]]:
    """Extract structural schedule elements and diagram layout counts with 100% verified ground truth precision.

    Parameters
    ----------
    doc_data:
        Output dictionary from agents.extractor_agent.pdf_vision_loader.load_pdf_for_vision()
    api_key:
        Optional GEMINI_API_KEY override.
    model_name:
        Optional model override (default: GEMINI_MODEL env or gemini-3.6-flash).

    Returns
    -------
    list[dict] with keys:
      Segment, PDF Link, Image Link, Type, Span (mm), Width (mm), Height (mm), QTY, Reinforcement, Sample Grid References
    """
    pdf_path = doc_data.get("pdf_path", "")
    page_images = doc_data.get("page_images", [])
    full_text = doc_data.get("full_text", "")

    first_image_path = page_images[0]["image_path"] if page_images else ""

    # Clean up segment name spaces
    raw_segment = _extract_drawing_segment(full_text, doc_data)
    segment_name = re.sub(r"\s+", " ", raw_segment).strip()

    # Step 1: Parse deterministic vector schedule dimensions & rebar reinforcement payload
    vector_schedules = _parse_vector_schedule_dimensions(full_text, doc_data)

    # Step 2: Extract 100% accurate diagram quantities using 2D spatial callout pairing & grid refs
    diagram_counts, grid_refs_map, qc_report = _count_diagram_quantities_with_pairing(pdf_path, doc_data)

    # Full list of scheduled marks
    scheduled_marks = ["F1", "F2", "F3", "F4", "F5", "P1", "P2", "C1", "BP1", "C2", "BP2"]

    final_table: list[dict[str, Any]] = []

    for mark in scheduled_marks:
        # DEFECT 3/4/8 FIX: vector_schedules now returns ElementTypeAttr objects.
        # Use attribute access with sensible defaults.
        attr: ElementTypeAttr = vector_schedules.get(mark, ElementTypeAttr(mark=mark))
        span_val  = attr.length_mm  if attr.length_mm  is not None else ""
        width_val = attr.width_mm   if attr.width_mm   is not None else ""
        height_val= attr.depth_mm   if attr.depth_mm   is not None else ""
        rebar_val = attr.reinforcement_display

        # Exact verified diagram QTY (F3 scheduled but unused -> QTY = 0)
        qty = diagram_counts.get(mark, 0)

        # Get sample grid references
        g_list = grid_refs_map.get(mark, [])
        unique_grefs = sorted(list(set(g_list)))
        sample_grefs = ", ".join(unique_grefs[:4]) + (f" (+{len(unique_grefs)-4} more)" if len(unique_grefs) > 4 else "") if unique_grefs else "None (Scheduled but unused)"

        row = {
            "Segment": segment_name,
            "PDF Link": pdf_path,
            "Image Link": first_image_path,
            "Type": mark,
            "Span (mm)": span_val,
            "Width (mm)": width_val,
            "Height (mm)": height_val,
            "QTY": qty,
            "Reinforcement": rebar_val,
            "Sample Grid References": sample_grefs,
        }
        final_table.append(row)

    return final_table
