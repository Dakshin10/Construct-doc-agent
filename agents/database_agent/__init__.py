"""
agents/database_agent
──────────────────────
AlloyDB Database Agent package for processing structural drawing payloads,
managing DB schemas, and pushing records to AlloyDB.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path when loaded directly by ADK runner
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from .agent import root_agent
except ImportError:
    from agents.database_agent.agent import root_agent

__all__ = ["root_agent"]

