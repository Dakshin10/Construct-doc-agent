"""
tools.py
--------
Google ADK Database Agent Tools for AlloyDB.

Provides parameterized UPSERT tools to insert extracted schedule data
from extractor_agent into the AlloyDB `salit_steel_foundation_diagram` table.
"""

import logging
import psycopg

from .db import get_connection

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parameterized UPSERT for salit_steel_foundation_diagram
# ---------------------------------------------------------------------------
_UPT_FOUNDATION_DIAGRAM_SQL = """
INSERT INTO salit_steel_foundation_diagram (
    element_type,
    category,
    length_mm,
    width_mm,
    depth_mm,
    primary_reinforcement,
    secondary_reinforcement_notes,
    plan_instances
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (element_type)
DO UPDATE SET
    category = EXCLUDED.category,
    length_mm = EXCLUDED.length_mm,
    width_mm = EXCLUDED.width_mm,
    depth_mm = EXCLUDED.depth_mm,
    primary_reinforcement = EXCLUDED.primary_reinforcement,
    secondary_reinforcement_notes = EXCLUDED.secondary_reinforcement_notes,
    plan_instances = EXCLUDED.plan_instances
RETURNING
    (xmax = 0) AS was_inserted;SER
"""


def _resolve_record_fields(rec: dict) -> tuple:
    """Helper to resolve field names and aliases into typed 8-column tuple."""
    element_type = str(rec.get("element_type") or rec.get("Type") or rec.get("footing_type") or rec.get("mark") or "")
    category = str(rec.get("category") or rec.get("Category") or "")
    length_mm = str(rec.get("length_mm") if rec.get("length_mm") is not None else rec.get("Length (mm)") or rec.get("Span (mm)") or rec.get("length") or "")
    width_mm = str(rec.get("width_mm") if rec.get("width_mm") is not None else rec.get("Width (mm)") or rec.get("width") or "")
    depth_mm = str(rec.get("depth_mm") if rec.get("depth_mm") is not None else rec.get("Depth (mm)") or rec.get("Height (mm)") or rec.get("depth") or "")
    primary_reinforcement = str(rec.get("primary_reinforcement") or rec.get("Primary reinforcement") or rec.get("Reinforcement") or rec.get("longitudinal_bars") or "")
    secondary_reinforcement_notes = str(rec.get("secondary_reinforcement_notes") or rec.get("Secondary reinforcement / notes") or rec.get("lateral_bars") or "")

    raw_instances = rec.get("plan_instances") if rec.get("plan_instances") is not None else rec.get("Plan instances") if rec.get("Plan instances") is not None else rec.get("QTY", 0)
    try:
        plan_instances = int(raw_instances)
    except (ValueError, TypeError):
        plan_instances = 0

    return (
        element_type,
        category,
        length_mm,
        width_mm,
        depth_mm,
        primary_reinforcement,
        secondary_reinforcement_notes,
        plan_instances,
    )


def insert_foundation_element_record(
    element_type: str,
    category: str = "",
    length_mm: str = "",
    width_mm: str = "",
    depth_mm: str = "",
    primary_reinforcement: str = "",
    secondary_reinforcement_notes: str = "",
    plan_instances: int = 0,
) -> dict:
    """
    Insert or update a record in the salit_steel_foundation_diagram AlloyDB table.

    Args:
        element_type:                  Element mark identifier (primary key, e.g. "F1", "P1", "C1", "BP1").
        category:                      Element category (e.g. "Spread footing pad", "Concrete pier", "Steel column", "Base plate").
        length_mm:                     Length dimension string (e.g. "1300", "500", "203").
        width_mm:                      Width dimension string (e.g. "1300", "500", "203").
        depth_mm:                      Depth / thickness string (e.g. "400", "850", "per section", "7.9 (wall thk.)").
        primary_reinforcement:         Primary reinforcement / rebar / HSS specification string.
        secondary_reinforcement_notes: Secondary reinforcement / ties / bolt gauge / pairing notes.
        plan_instances:                Diagram count (QTY on layout plan, e.g. 33, 6, 0).

    Returns:
        dict with keys: status, element_type, operation, message
    """
    params = (
        element_type,
        category,
        length_mm,
        width_mm,
        depth_mm,
        primary_reinforcement,
        secondary_reinforcement_notes,
        plan_instances,
    )

    try:
        conn: psycopg.Connection = get_connection()
        with conn:
            with conn.cursor() as cur:
                cur.execute(_UPSERT_FOUNDATION_DIAGRAM_SQL, params)
                row = cur.fetchone()
                was_inserted = row[0] if row else True

        operation = "inserted" if was_inserted else "updated"
        action_word = "inserted" if was_inserted else "updated"

        logger.info("salit_steel_foundation_diagram %s: element_type=%s", operation, element_type)

        return {
            "status": "success",
            "element_type": element_type,
            "operation": operation,
            "message": f"Foundation element record '{element_type}' {action_word} successfully.",
        }

    except ValueError as exc:
        logger.error("Configuration error: %s", exc)
        return {
            "status": "error",
            "error_type": "configuration_error",
            "message": str(exc),
        }

    except psycopg.OperationalError:
        logger.exception("Database connection failed for element_type=%s", element_type)
        return {
            "status": "error",
            "error_type": "database_error",
            "message": "Unable to connect to AlloyDB. Ensure the AlloyDB Auth Proxy is running.",
        }

    except psycopg.Error:
        logger.exception("Database error for element_type=%s", element_type)
        return {
            "status": "error",
            "error_type": "database_error",
            "message": "Unable to write record to AlloyDB. Check server logs for details.",
        }

    except Exception:
        logger.exception("Unexpected error for element_type=%s", element_type)
        return {
            "status": "error",
            "error_type": "unexpected_error",
            "message": "An unexpected error occurred. Check server logs for details.",
        }


def insert_foundation_element_records_batch(
    records: list[dict],
) -> dict:
    """
    Insert or update multiple foundation element records into salit_steel_foundation_diagram table in a batch.

    Args:
        records: A list of dict objects representing foundation diagram elements.

    Returns:
        dict with keys: status, processed_count, records, message
    """
    successful = []

    try:
        conn: psycopg.Connection = get_connection()
        with conn:
            with conn.cursor() as cur:
                for rec in records:
                    params = _resolve_record_fields(rec)
                    cur.execute(_UPSERT_FOUNDATION_DIAGRAM_SQL, params)
                    row = cur.fetchone()
                    was_inserted = row[0] if row else True
                    op = "inserted" if was_inserted else "updated"
                    el_type = params[0]
                    successful.append({"element_type": el_type, "operation": op})
                    logger.info("Batch item salit_steel_foundation_diagram %s: element_type=%s", op, el_type)

        return {
            "status": "success",
            "processed_count": len(successful),
            "records": successful,
            "message": f"Successfully processed batch of {len(successful)} foundation element records.",
        }

    except ValueError as exc:
        logger.error("Configuration error in batch insert: %s", exc)
        return {
            "status": "error",
            "error_type": "configuration_error",
            "message": str(exc),
        }

    except psycopg.OperationalError:
        logger.exception("Database connection failed during batch insert")
        return {
            "status": "error",
            "error_type": "database_error",
            "message": "Unable to connect to AlloyDB. Ensure the AlloyDB Auth Proxy is running.",
        }

    except psycopg.Error as exc:
        logger.exception("Database error during batch insert")
        return {
            "status": "error",
            "error_type": "database_error",
            "message": f"Database error during batch insert: {str(exc)}",
        }

    except Exception as exc:
        logger.exception("Unexpected error during batch insert")
        return {
            "status": "error",
            "error_type": "unexpected_error",
            "message": f"An unexpected error occurred: {str(exc)}",
        }



