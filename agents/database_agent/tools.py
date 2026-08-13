"""
tools.py
--------
Google ADK Database Agent Tools for AlloyDB.

Provides parameterized UPSERT tools to insert extracted schedule data
from extractor_agent into the AlloyDB spread_footing_pad_schedule table.
"""

import logging
import psycopg

from .db import get_connection

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parameterized UPSERT for structural_footing_schedule
# ---------------------------------------------------------------------------
_UPSERT_FOOTING_SQL = """
INSERT INTO structural_footing_schedule (
    footing_type,
    length,
    width,
    depth,
    longitudinal_bars,
    lateral_bars
)
VALUES (%s, %s, %s, %s, %s, %s)
ON CONFLICT (footing_type)
DO UPDATE SET
    length = EXCLUDED.length,
    width = EXCLUDED.width,
    depth = EXCLUDED.depth,
    longitudinal_bars = EXCLUDED.longitudinal_bars,
    lateral_bars = EXCLUDED.lateral_bars
RETURNING
    (xmax = 0) AS was_inserted;
"""


def insert_spread_footing_record(
    footing_type: str,
    length: str,
    width: str,
    depth: str,
    longitudinal_bars: str,
    lateral_bars: str,
) -> dict:
    """
    Insert or update a record in the spread_footing_pad_schedule AlloyDB table.

    This tool performs a parameterized UPSERT:
    - If footing_type does not exist → INSERT (new record)
    - If footing_type already exists → UPDATE all dimensions and bar specs

    Args:
        footing_type:      Identifier for the footing (primary key, e.g. "F1").
        length:            Length dimension string (e.g. "1300mm").
        width:             Width dimension string (e.g. "1300mm").
        depth:             Depth dimension string (e.g. "400mm").
        longitudinal_bars: Specification string for longitudinal bars.
        lateral_bars:      Specification string for lateral bars.

    Returns:
        dict with keys: status, footing_type, operation, message
    """
    params = (
        footing_type,
        length,
        width,
        depth,
        longitudinal_bars,
        lateral_bars,
    )

    try:
        conn: psycopg.Connection = get_connection()
        with conn:
            with conn.cursor() as cur:
                cur.execute(_UPSERT_FOOTING_SQL, params)
                row = cur.fetchone()
                was_inserted = row[0] if row else True

        operation = "inserted" if was_inserted else "updated"
        action_word = "inserted" if was_inserted else "updated"

        logger.info("spread_footing_pad_schedule %s: footing_type=%s", operation, footing_type)

        return {
            "status": "success",
            "footing_type": footing_type,
            "operation": operation,
            "message": f"Spread footing record '{footing_type}' {action_word} successfully.",
        }

    except ValueError as exc:
        logger.error("Configuration error: %s", exc)
        return {
            "status": "error",
            "error_type": "configuration_error",
            "message": str(exc),
        }

    except psycopg.OperationalError:
        logger.exception("Database connection failed for footing_type=%s", footing_type)
        return {
            "status": "error",
            "error_type": "database_error",
            "message": "Unable to connect to AlloyDB. Ensure the AlloyDB Auth Proxy is running.",
        }

    except psycopg.Error:
        logger.exception("Database error for footing_type=%s", footing_type)
        return {
            "status": "error",
            "error_type": "database_error",
            "message": "Unable to write record to AlloyDB. Check server logs for details.",
        }

    except Exception:
        logger.exception("Unexpected error for footing_type=%s", footing_type)
        return {
            "status": "error",
            "error_type": "unexpected_error",
            "message": "An unexpected error occurred. Check server logs for details.",
        }


def insert_spread_footing_records_batch(
    records: list[dict],
) -> dict:
    """
    Insert or update multiple spread footing records into the spread_footing_pad_schedule table in a single batch operation.

    Args:
        records: A list of dict objects, each containing:
                 - footing_type: str
                 - length: str
                 - width: str
                 - depth: str
                 - longitudinal_bars: str
                 - lateral_bars: str

    Returns:
        dict with keys: status, processed_count, successful_records, message
    """
    successful = []
    failed = []

    try:
        conn: psycopg.Connection = get_connection()
        with conn:
            with conn.cursor() as cur:
                for rec in records:
                    params = (
                        rec.get("footing_type"),
                        rec.get("length"),
                        rec.get("width"),
                        rec.get("depth"),
                        rec.get("longitudinal_bars"),
                        rec.get("lateral_bars"),
                    )
                    cur.execute(_UPSERT_FOOTING_SQL, params)
                    row = cur.fetchone()
                    was_inserted = row[0] if row else True
                    op = "inserted" if was_inserted else "updated"
                    successful.append({"footing_type": rec.get("footing_type"), "operation": op})
                    logger.info("Batch item spread_footing_pad_schedule %s: footing_type=%s", op, rec.get("footing_type"))

        return {
            "status": "success",
            "processed_count": len(successful),
            "records": successful,
            "message": f"Successfully processed batch of {len(successful)} spread footing records.",
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


