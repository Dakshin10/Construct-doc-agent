"""
app/tools.py
────────────
Google ADK tools for Reader Agent in beta_salitsteels.

Tools provided:
1. extract_pdf_drawing_data: Vision & PDF text layout extractor tool.
2. send_to_database_agent: Transmits JSON payload to alloydb_adk_agent in database_agent.
3. commit_and_push_to_main: Stages, commits, and pushes output_table.json to git origin/main.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from agents.extractor_agent.pdf_vision_loader import load_pdf_for_vision
from agents.extractor_agent.field_extraction_agent import (
    extract_table_from_doc,
    _parse_vector_schedule_dimensions,
    _count_diagram_quantities_with_pairing,
)
from agents.extractor_agent.json_exporter import save_json_file

logger = logging.getLogger(__name__)


def extract_pdf_drawing_data(pdf_path: str) -> dict[str, Any]:
    """
    Extract structural schedule tables and diagram element quantities from a PDF file.

    Saves the extracted results to output_table.json and formats a database payload for
    the AlloyDB spread_footing_pad_schedule table.

    Args:
        pdf_path: Path to the structural drawing PDF file.

    Returns:
        dict containing extraction status, json_path, total_rows, and db_payload.
    """
    pdf_file = Path(pdf_path).resolve()
    if not pdf_file.exists():
        return {
            "status": "error",
            "message": f"PDF file not found: {pdf_file}",
        }

    try:
        logger.info("Step 1: Rendering PDF and extracting text context for %s", pdf_file)
        doc_data = load_pdf_for_vision(pdf_file)

        logger.info("Step 2: Extracting structural table rows and diagram counts...")
        table_rows = extract_table_from_doc(doc_data)
        vector_schedules = _parse_vector_schedule_dimensions("", doc_data)
        _, _, qc_report = _count_diagram_quantities_with_pairing(pdf_file, doc_data)

        # Save standard output_table.json
        json_target = Path("output_table.json").resolve()
        save_json_file(
            table_rows=table_rows,
            json_path=json_target,
            vector_schedules=vector_schedules,
            qc_report=qc_report,
        )

        # Build salit_steel_foundation_diagram DB payload with all 11 element rows
        db_payload = []
        for r in table_rows:
            db_payload.append({
                "element_type": r.get("Type", ""),
                "Type": r.get("Type", ""),
                "category": r.get("Category", ""),
                "Category": r.get("Category", ""),
                "length_mm": r.get("Length (mm)", ""),
                "Length (mm)": r.get("Length (mm)", ""),
                "width_mm": r.get("Width (mm)", ""),
                "Width (mm)": r.get("Width (mm)", ""),
                "depth_mm": r.get("Depth (mm)", ""),
                "Depth (mm)": r.get("Depth (mm)", ""),
                "primary_reinforcement": r.get("Primary reinforcement", ""),
                "Primary reinforcement": r.get("Primary reinforcement", ""),
                "secondary_reinforcement_notes": r.get("Secondary reinforcement / notes", ""),
                "Secondary reinforcement / notes": r.get("Secondary reinforcement / notes", ""),
                "plan_instances": r.get("Plan instances", 0),
                "Plan instances": r.get("Plan instances", 0),
            })

        return {
            "status": "success",
            "pdf_path": str(pdf_file),
            "json_path": str(json_target),
            "total_extracted_rows": len(table_rows),
            "db_payload": db_payload,
            "message": f"Successfully extracted {len(table_rows)} rows and generated DB payload with {len(db_payload)} footing records.",
        }
    except Exception as exc:
        logger.exception("Error extracting PDF drawing data: %s", exc)
        return {
            "status": "error",
            "message": f"Extraction failed: {str(exc)}",
        }


async def _run_database_agent_async(payload_json: str) -> str:
    from google.adk.runners import InMemoryRunner
    from google.genai import types as genai_types
    from agents.database_agent.agent import root_agent as db_agent

    parsed = json.loads(payload_json)
    runner = InMemoryRunner(agent=db_agent, app_name="alloydb_insert_agent")
    session = await runner.session_service.create_session(
        app_name="alloydb_insert_agent",
        user_id="cli_user",
    )

    user_message = genai_types.Content(
        role="user",
        parts=[genai_types.Part(text=json.dumps(parsed, indent=2))],
    )

    final_response = None
    async for event in runner.run_async(
        user_id="cli_user",
        session_id=session.id,
        new_message=user_message,
    ):
        if event.is_final_response() and event.content and event.content.parts:
            final_response = event.content.parts[0].text

    return final_response or "(No response returned from database agent)"


def _parse_json_or_literal(raw: Any) -> Any:
    if isinstance(raw, (list, dict)):
        return raw
    if not isinstance(raw, str):
        return raw
    raw = raw.strip()
    try:
        return json.loads(raw)
    except Exception:
        try:
            return json.loads(raw.replace("'", '"'))
        except Exception:
            import ast
            return ast.literal_eval(raw)


async def send_to_database_agent(payload_json: str) -> dict[str, Any]:
    """
    Transmit extracted JSON payload to the AlloyDB Database Agent.

    Args:
        payload_json: JSON string or list representing the array of footing objects or schedule record.

    Returns:
        dict containing execution status and output from database agent.
    """
    try:
        # Validate and parse payload into standard python object
        parsed = _parse_json_or_literal(payload_json)
        logger.info("Invoking Database Agent in-process with payload of %d item(s)...", len(parsed) if isinstance(parsed, list) else 1)

        db_output = await _run_database_agent_async(json.dumps(parsed))

        return {
            "status": "success",
            "database_agent_output": db_output,
            "message": "Database agent executed successfully.",
        }

    except Exception as exc:
        logger.exception("Error invoking database agent: %s", exc)
        return {
            "status": "error",
            "message": f"Failed to transmit payload to database agent: {str(exc)}",
        }


def commit_and_push_to_main(commit_message: str = "Update Reader ADK Workflow and output_table.json") -> dict[str, Any]:
    """
    Stage, commit, and push updated output_table.json and workflow code to origin/main of beta_salitsteels.

    Args:
        commit_message: Optional custom Git commit message.

    Returns:
        dict containing status and git output details.
    """
    repo_dir = Path(".").resolve()
    try:
        logger.info("Git staging files in %s", repo_dir)

        # 1. Git add agents, main.py, output_table.json
        add_res = subprocess.run(["git", "add", "agents/", "main.py", "output_table.json"], cwd=str(repo_dir), capture_output=True, text=True)
        if add_res.returncode != 0:
            return {"status": "error", "message": f"git add failed: {add_res.stderr}"}

        # 2. Check if there are changes to commit
        status_res = subprocess.run(["git", "status", "--porcelain"], cwd=str(repo_dir), capture_output=True, text=True)
        if not status_res.stdout.strip():
            return {
                "status": "success",
                "message": "No changes to commit. Working directory clean.",
                "git_output": "Already up to date.",
            }

        # 3. Git commit
        commit_res = subprocess.run(["git", "commit", "-m", commit_message], cwd=str(repo_dir), capture_output=True, text=True)
        if commit_res.returncode != 0:
            return {"status": "error", "message": f"git commit failed: {commit_res.stderr}"}

        # 4. Git push
        push_res = subprocess.run(["git", "push", "origin", "main"], cwd=str(repo_dir), capture_output=True, text=True)
        if push_res.returncode != 0:
            return {"status": "error", "message": f"git push failed: {push_res.stderr}"}

        return {
            "status": "success",
            "commit_message": commit_message,
            "git_output": push_res.stdout or push_res.stderr or "Pushed to origin/main successfully.",
            "message": "Committed and pushed to main branch successfully.",
        }

    except Exception as exc:
        logger.exception("Error executing git commit and push: %s", exc)
        return {
            "status": "error",
            "message": f"Git commit/push failed: {str(exc)}",
        }
