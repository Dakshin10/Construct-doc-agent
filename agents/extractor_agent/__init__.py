"""
agents/extractor_agent
──────────────────────
Extractor Agent package for vision PDF processing, schedule table parsing,
rebar reinforcement extraction, and JSON output generation.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path when loaded directly by ADK runner
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from .agent import root_agent
    from .field_extraction_agent import extract_table_from_doc
    from .json_exporter import save_json_file, convert_to_json_string
    from .pdf_vision_loader import load_pdf_for_vision
except ImportError:
    from agents.extractor_agent.agent import root_agent
    from agents.extractor_agent.field_extraction_agent import extract_table_from_doc
    from agents.extractor_agent.json_exporter import save_json_file, convert_to_json_string
    from agents.extractor_agent.pdf_vision_loader import load_pdf_for_vision

__all__ = [
    "root_agent",
    "extract_table_from_doc",
    "save_json_file",
    "convert_to_json_string",
    "load_pdf_for_vision",
]

