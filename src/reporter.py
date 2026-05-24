"""
reporter.py
-----------
Assembles the final human-readable estimating support report.

Combines:
  - New project metadata header
  - List of comparable projects retrieved (with similarity scores)
  - Governance warnings (if any)
  - LLM analysis (grounded, cited)

Also handles saving the report to the output/ folder.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from src.models import Project, NewProjectBrief

OUTPUT_DIR = Path(__file__).parent.parent / "output"


def assemble_report(
    new_project: NewProjectBrief,
    retrieved:   list[tuple[Project, float]],
    warnings:    list[str],
    llm_output:  str,
) -> str:
    """
    Combine all sections into the final report string.

    Structure:
      1. Header — new project details
      2. Comparable projects retrieved (with scores)
      3. Governance warnings (if any)
      4. AI analysis (LLM output, grounded on retrieved data)
      5. Footer
    """
    lines = [
        "=" * 72,
        "  AI-ASSISTED ESTIMATE SUPPORT REPORT",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 72,
        f"  New Project : {new_project.name}",
        f"  Type        : {new_project.project_type}",
        f"  Size        : {new_project.square_feet:,} sq ft",
        "",
        "  COMPARABLE PROJECTS RETRIEVED:",
    ]

    for rank, (p, score) in enumerate(retrieved, 1):
        actuals = (
            f"${p.final_actual_cost:,.0f} actual"
            if p.final_actual_cost else "actuals incomplete"
        )
        lines.append(
            f"    {rank}. {p.project_id} — {p.project_name[:52]}\n"
            f"         Similarity: {score:.3f} | {p.square_feet:,} sqft | {actuals}"
        )

    if warnings:
        lines += ["", "  GOVERNANCE WARNINGS:"]
        for w in warnings:
            lines.append(f"    {w}")

    lines += [
        "",
        "─" * 72,
        "  AI ANALYSIS",
        "  (grounded strictly on retrieved project data — citations required)",
        "─" * 72,
        "",
        llm_output,
        "",
        "=" * 72,
    ]

    return "\n".join(lines)


def save_report(report: str) -> Path:
    """
    Save the report to output/ with a timestamped filename.
    Returns the path where it was saved.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    output_path = OUTPUT_DIR / f"estimate_report_{timestamp}.txt"
    output_path.write_text(report, encoding="utf-8")
    return output_path