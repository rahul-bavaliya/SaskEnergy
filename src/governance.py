"""
Governance checks for retrieved projects.
Checks performed:
  1. Low similarity score     — retrieved projects may not be comparable
  2. Incomplete actuals       — P-005 has no final cost (reconciliation pending)
  3. Size mismatch > 35%      — cost per sqft benchmarks need adjustment
  4. Old project (pre-2022)   — raw figures may understate current pricing
"""

from __future__ import annotations

from src.models import Project, NewProjectBrief

# Thresholds — centralised here so they're easy to tune
SIMILARITY_THRESHOLD   = 0.80   # below this → low confidence warning
SIZE_MISMATCH_THRESHOLD = 0.35  # above this → size mismatch warning
ESCALATION_YEAR        = 2017   # projects older than this → escalation warning


def run_governance_checks(
    new_project: NewProjectBrief,
    retrieved:   list[tuple[Project, float]],
) -> list[str]:
    """
    Run all governance checks on the retrieved project set.

    Returns a list of warning strings. Empty list = no issues found.
    Warnings are included in the report and shown to the estimator.

    The LLM is also instructed (via system prompt in synthesizer.py)
    to respect these flags — e.g. not to use P-005 for benchmarking.
    """
    warnings = []

    # ── Guard: nothing retrieved ──
    if not retrieved:
        warnings.append(
            "⚠ CRITICAL: No comparable projects retrieved. "
            "Output is unreliable. Manual review required."
        )
        return warnings

    # ── Check 1: Low similarity score ──
    top_score = retrieved[0][1]
    if top_score <= SIMILARITY_THRESHOLD:
        warnings.append(
            f"⚠ LOW CONFIDENCE: Best match similarity score is {top_score:.2f} "
            f"(threshold: {SIMILARITY_THRESHOLD}). Retrieved projects may not be "
            "closely comparable. Treat cost benchmarks as directional only."
        )

    for project, score in retrieved:

        # ── Check 2: Incomplete actuals ──
        if project.final_actual_cost is None:
            warnings.append(
                f"⚠ INCOMPLETE DATA: {project.project_id} ({project.project_name}) "
                "has no final actual cost — reconciliation still in progress. "
                "Do not use for cost benchmarking. Lessons learned still applicable."
            )

        # ── Check 3: Size mismatch ──
        size_diff_pct = (
            abs(project.square_feet - new_project.square_feet)
            / new_project.square_feet
        )
        if size_diff_pct > SIZE_MISMATCH_THRESHOLD:
            warnings.append(
                f"⚠ SIZE MISMATCH: {project.project_id} is {project.square_feet:,} sqft "
                f"vs new project {new_project.square_feet:,} sqft "
                f"({size_diff_pct * 100:.0f}% difference). "
                "Use $/sqft benchmarks rather than total cost comparisons."
            )

        # ── Check 4: Cost escalation risk ──
        if project.year < ESCALATION_YEAR:
            warnings.append(
                f"⚠ COST ESCALATION: {project.project_id} completed in {project.year}. "
                "Raw cost figures may understate current pricing due to material "
                "and labour escalation since completion. Apply an escalation factor."
            )

    return warnings