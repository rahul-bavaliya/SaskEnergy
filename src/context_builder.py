"""
context_builder.py
------------------
Formats retrieved historical projects into a structured context block
that gets injected into the LLM prompt.

This is the critical anti-hallucination mechanism in RAG.

Key principle:
  The LLM is only allowed to answer from this context block.
  It cannot draw on its training data for project-specific figures.
  Every number, lesson, and risk flag in the LLM's response must
  be traceable back to a real project in this context.

What we include per project:
  - Structured metrics: cost, variance, sqft rate, schedule
  - Estimate vs actual breakdown by category (from CSVs)
  - Change order details with drivers and avoidability flags
  - Full scope summary (from markdown)
  - Full lessons learned (from markdown)
"""

from __future__ import annotations

from src.models import Project


def build_context(retrieved: list[tuple[Project, float]]) -> str:
    """
    Format the top-K retrieved projects into one structured text block.

    This block becomes the LLM's sole knowledge source for the response.
    Including structured data (cost tables, COs) alongside unstructured
    text (scope, lessons) gives the LLM everything it needs to produce
    a rich, cited, grounded estimating report.
    """
    blocks = []

    for rank, (p, score) in enumerate(retrieved, 1):

        # ── Estimate vs Actual cost table ──
        all_cats = sorted(
            set(list(p.estimate_line_items.keys()) + list(p.actual_line_items.keys()))
        )
        cost_table = ""
        for cat in all_cats:
            est = p.estimate_line_items.get(cat)
            act = p.actual_line_items.get(cat)
            est_str = f"${est:>10,.0f}" if est is not None else "         N/A"
            act_str = f"${act:>10,.0f}" if act is not None else "         N/A"
            if est is not None and act is not None:
                var = act - est
                var_str = f"  ({'+' if var >= 0 else ''}{var:,.0f})"
            else:
                var_str = ""
            cost_table += f"    {cat:<30} Est:{est_str}  Act:{act_str}{var_str}\n"

        # ── Change order summary ──
        co_lines = ""
        for co in p.change_orders:
            co_lines += (
                f"    {co['change_order_id']}: {co['description']} "
                f"(${float(co['amount']):,.0f}) "
                f"| Driver: {co['driver']} "
                f"| Avoidable: {co['avoidable']}\n"
            )

        # ── Actuals summary line ──
        if p.final_actual_cost:
            actuals_line = (
                f"Actual Cost       : ${p.final_actual_cost:,.0f}  "
                f"({p.cost_variance_pct:+.1f}% vs estimate)  |  "
                f"${p.actual_cost_per_sqft:.2f}/sqft actual"
            )
        else:
            actuals_line = (
                "Actual Cost       : INCOMPLETE — reconciliation in progress. "
                "Do not use for cost benchmarking."
            )

        block = f"""
━━━ COMPARABLE PROJECT {rank}  (semantic similarity: {score:.3f}) ━━━
    ID                : {p.project_id}
    Name              : {p.project_name}
    Type              : {p.project_type}
    Building Type     : {p.building_type}
    Region            : {p.region}
    Year Completed    : {p.year}
    Size              : {p.square_feet:,} sq ft
    Complexity        : {p.complexity}
    Delivery Model    : {p.delivery_model}
    Occupied Site     : {p.occupied_site}
    Original Estimate : ${p.original_estimate:,.0f}  |  ${p.estimate_cost_per_sqft:.2f}/sqft estimated
    {actuals_line}
    Total Change Orders: ${p.total_co_amount:,.0f}
    Project Notes     : {p.project_notes}

    ESTIMATE vs ACTUAL BY CATEGORY:
    {cost_table}
    CHANGE ORDERS:
    {co_lines if co_lines else "    None recorded."}
    SCOPE SUMMARY:
    {p.scope_summary}

    LESSONS LEARNED:
    {p.lessons_learned}
""".strip()

        blocks.append(block)

    return "\n\n".join(blocks)