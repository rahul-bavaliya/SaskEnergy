"""
src/__init__.py
---------------
Public API for the ai_estimate_tool source package.

Exposes the key classes and functions so main.py
stays clean and readable.
"""

from src.models          import Project, NewProjectBrief
from src.loader          import load_all_projects, load_new_project_brief
from src.embedder        import ingest_projects
from src.retriever       import retrieve
from src.governance      import run_governance_checks
from src.context_builder import build_context
from src.synthesizer     import synthesize
from src.reporter        import assemble_report, save_report

__all__ = [
    "Project",
    "NewProjectBrief",
    "load_all_projects",
    "load_new_project_brief",
    "ingest_projects",
    "retrieve",
    "run_governance_checks",
    "build_context",
    "synthesize",
    "assemble_report",
    "save_report",
]