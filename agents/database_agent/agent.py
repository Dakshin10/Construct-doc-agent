"""
agent.py
--------
Google ADK 2.x root agent definition.

This module defines `root_agent` — the ADK entry point.
ADK discovers this variable when running:
    adk run alloydb_adk_agent/app
    adk web

Architecture:
    JSON input
        → root_agent (Gemini LLM)
        → insert_master_record OR insert_spread_footing_record tool
        → AlloyDB database tables
        → structured result returned to user
"""

import logging

from google.adk.agents import Agent

from .tools import (
    insert_foundation_element_record,
    insert_foundation_element_records_batch,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Root agent — the single entry point for ADK
# ---------------------------------------------------------------------------
root_agent = Agent(
    name="alloydb_insert_agent",
    model="gemini-flash-lite-latest",
    description=(
        "A database insertion agent that receives structured extracted foundation diagram JSON data and "
        "writes it into AlloyDB `salit_steel_foundation_diagram` table."
    ),
    instruction="""
You are a database insertion agent for AlloyDB PostgreSQL tables.

Your ONLY job is to receive structured extracted foundation diagram JSON data (single object or list of objects) from extractor_agent and insert them into the `salit_steel_foundation_diagram` table using tools.

## Supported Tools

### Salit Steel Foundation Diagram (`salit_steel_foundation_diagram`)
If the input contains foundation element data (Type/element_type, Category, Length/length_mm, Width/width_mm, Depth/depth_mm, Primary reinforcement, Secondary reinforcement / notes, Plan instances/QTY):
- If the input is a LIST / ARRAY of foundation element records:
  - Call `insert_foundation_element_records_batch` ONCE passing the entire array as `records`.
- If the input is a SINGLE foundation element object:
  - Call `insert_foundation_element_record` with the individual fields.

## Rules
1. For multiple records, ALWAYS use `insert_foundation_element_records_batch` in a SINGLE tool call to preserve quota.
2. Return all tool execution results clearly.
3. Do NOT generate SQL directly.
4. Do NOT skip tool calls.
""",
    tools=[
        insert_foundation_element_record,
        insert_foundation_element_records_batch,
    ],
)



