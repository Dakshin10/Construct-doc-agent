import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from .extractor_agent import field_extraction_agent, json_exporter
except ImportError:
    from agents.extractor_agent import field_extraction_agent, json_exporter

__all__ = ["extractor_agent", "database_agent", "field_extraction_agent", "json_exporter"]

