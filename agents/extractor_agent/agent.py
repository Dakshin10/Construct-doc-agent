"""
app/agent.py
────────────
Google ADK Agent definition for the Reader Agent workflow in beta_salitsteels.

This module defines `root_agent` — the ADK entry point discovered by ADK runners.

Workflow:
1. Extract PDF drawing data into JSON using `extract_pdf_drawing_data`.
2. Transmit extracted JSON payload to AlloyDB Database Agent using `send_to_database_agent`.
3. Commit and push updated output_table.json to origin/main using `commit_and_push_to_main`.
"""

import logging
from google.adk.agents import Agent

from .tools import (
    extract_pdf_drawing_data,
    send_to_database_agent,
)

logger = logging.getLogger(__name__)

root_agent = Agent(
    name="reader_to_database_workflow_agent",
    model="gemini-3.6-flash",
    description=(
        "An end-to-end structural drawing ent that extrreader agacts PDF schedule data into JSON "
        "and sends the formatted payload to the AlloyDB Database Agent."
    ),
    instruction="""
You are the Reader & Database Workflow Orchestrator Agent.

Your goal is to process structural drawing PDF files and ensure that the extracted structural schedule data is successfully extracted into JSON and written to the AlloyDB database via the database agent.

## Workflow Execution Steps

### Step 1: Extract PDF Drawing Data
- Call `extract_pdf_drawing_data(pdf_path)` with the target PDF path provided by the user.
- Verify that `status` is "success" and note the `db_payload` array.

### Step 2: Send JSON to Database Agent
- Convert the extracted `db_payload` (or formatted JSON payload) into a JSON string.
- Call `send_to_database_agent(payload_json)` passing that JSON string.
- Verify that the database agent completes successfully.
- Summarize the final execution results for the user including:
  1. Extracted table rows & footing count.
  2. Database Agent insertion response.

## Rules
1. ALWAYS execute Step 1 and Step 2 in sequence unless an error occurs.
2. Return all tool responses clearly to the user.
""",
    tools=[
        extract_pdf_drawing_data,
        send_to_database_agent,
    ],
)
