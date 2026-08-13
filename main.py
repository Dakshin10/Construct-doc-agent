"""
main.py
───────
Programmatic entry point for the Reader Agent to Database Agent ADK Workflow.

Usage (CLI):
    python main.py --pdf "C:\\Users\\daksh\\OneDrive - MinoriLabs\\Construction_ai_assitant\\Foundational_diagram.pdf"

    # Or simple positional arg:
    python main.py "path/to/diagram.pdf"

This runner uses the ADK InMemoryRunner to execute the reader-to-database agent workflow.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def run_workflow(pdf_path: str) -> None:
    """Run the Reader-to-Database ADK workflow with the given PDF file path."""
    from google.adk.runners import InMemoryRunner
    from google.genai import types as genai_types
    from app.agent import root_agent

    abs_pdf_path = str(Path(pdf_path).resolve())

    print("\n" + "=" * 70)
    print(f"STARTING READER-TO-DATABASE WORKFLOW FOR PDF:")
    print(f"Path: {abs_pdf_path}")
    print("=" * 70 + "\n")

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            runner = InMemoryRunner(agent=root_agent, app_name="reader_to_database_workflow")
            session = await runner.session_service.create_session(
                app_name="reader_to_database_workflow",
                user_id="cli_user",
            )

            prompt = (
                f"Please execute the workflow for PDF: '{abs_pdf_path}'. "
                f"Step 1: Extract PDF drawing data. "
                f"Step 2: Send extracted JSON payload to database agent. "
                f"Step 3: Commit and push output_table.json to main branch."
            )

            user_message = genai_types.Content(
                role="user",
                parts=[genai_types.Part(text=prompt)],
            )

            logger.info("Executing ADK workflow (Attempt %d/%d)...", attempt, max_retries)

            final_response = None
            async for event in runner.run_async(
                user_id="cli_user",
                session_id=session.id,
                new_message=user_message,
            ):
                if event.is_final_response() and event.content and event.content.parts:
                    final_response = event.content.parts[0].text

            print("\n" + "=" * 70)
            print("WORKFLOW EXECUTION COMPLETE")
            print("=" * 70)
            print("AGENT RESPONSE SUMMARY:")
            print(final_response or "(No response returned)")
            print("=" * 70 + "\n")
            break

        except Exception as exc:
            err_str = str(exc)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                wait_time = 35
                if attempt < max_retries:
                    print(f"\n[WARNING] Gemini API rate limit (429). Waiting {wait_time}s before retry ({attempt}/{max_retries})...")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"\n[ERROR] Rate limit quota exceeded after {max_retries} attempts.")
                    raise
            else:
                raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Reader Agent to Database Agent ADK workflow on a structural drawing PDF."
    )
    parser.add_argument(
        "--pdf",
        nargs="?",
        default="C:\\Users\\daksh\\OneDrive - MinoriLabs\\Construction_ai_assitant\\Foundational_diagram.pdf",
        metavar="PATH",
        help="Path to the PDF diagram file to process.",
    )
    return parser.parse_args()


def main() -> None:
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        pdf_path = sys.argv[1]
    else:
        args = _parse_args()
        pdf_path = args.pdf

    asyncio.run(run_workflow(pdf_path))


if __name__ == "__main__":
    main()
