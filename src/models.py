"""
models.py
---------
Data models for the AI-Assisted Estimate Support Tool.

Contains:
  - Project      : one historical construction project (structured + unstructured)
  - NewProjectBrief : the new project the estimator needs to price
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Project:
    """
    Represents one historical construction project.

    Structured fields are loaded from CSVs.
    Unstructured fields (scope_summary, lessons_learned) come from markdown files.
    Both are combined into a single embedding vector for semantic search.
    """
    # ── from projects.csv ──
    project_id:         str
    project_name:       str
    year:               int
    region:             str
    building_type:      str
    project_type:       str
    square_feet:        int
    complexity:         str
    delivery_model:     str
    original_estimate:  float
    final_actual_cost:  Optional[float]   # None = reconciliation incomplete (e.g. P-005)
    schedule_months:    float
    occupied_site:      str               # "Yes" / "No" / "Partial"
    project_notes:      str

    # ── from estimate_line_items.csv ──
    estimate_line_items: dict = field(default_factory=dict)   # {category: amount}

    # ── from actual_costs.csv ──
    actual_line_items:   dict = field(default_factory=dict)   # {category: amount}

    # ── from change_orders.csv ──
    change_orders:       list[dict] = field(default_factory=list)

    # ── from project_documents/*.md ──
    scope_summary:       str = ""
    lessons_learned:     str = ""

    # ── populated at ingest time by embedder.py ──
    embedding:           list[float] = field(default_factory=list)

    # ── computed properties ──
    @property
    def cost_variance_pct(self) -> Optional[float]:
        """% over/under budget. Positive = over budget."""
        if self.final_actual_cost is None or self.original_estimate == 0:
            return None
        return round(
            (self.final_actual_cost - self.original_estimate)
            / self.original_estimate * 100, 1
        )

    @property
    def actual_cost_per_sqft(self) -> Optional[float]:
        if self.final_actual_cost and self.square_feet:
            return round(self.final_actual_cost / self.square_feet, 2)
        return None

    @property
    def estimate_cost_per_sqft(self) -> float:
        if self.original_estimate and self.square_feet:
            return round(self.original_estimate / self.square_feet, 2)
        return 0.0

    @property
    def total_co_amount(self) -> float:
        return sum(float(co.get("amount", 0)) for co in self.change_orders)

    def to_text_for_embedding(self) -> str:
        """
        Combine all fields into one string for embedding.

        Including structured fields (type, size, region) inside the text means
        the semantic search naturally surfaces matching projects even without
        hard filters. This is the core design choice for hybrid RAG.
        """
        co_text = " ".join(
            f"{co['change_order_id']}: {co['description']} (driver: {co['driver']})"
            for co in self.change_orders
        )
        estimate_text = " ".join(
            f"{cat}: ${amt:,.0f}" for cat, amt in self.estimate_line_items.items()
        )
        return (
            f"Project: {self.project_name}. "
            f"Type: {self.project_type}. "
            f"Building type: {self.building_type}. "
            f"Region: {self.region}. "
            f"Size: {self.square_feet:,} sq ft. "
            f"Year: {self.year}. "
            f"Complexity: {self.complexity}. "
            f"Occupied site: {self.occupied_site}. "
            f"Estimate line items: {estimate_text}. "
            f"Scope: {self.scope_summary} "
            f"Lessons learned: {self.lessons_learned} "
            f"Change orders: {co_text}"
        )


@dataclass
class NewProjectBrief:
    """The estimator's input — the project they need to estimate."""
    name:         str
    square_feet:  int
    project_type: str
    description:  str       # full scope text from markdown

    def to_query_text(self) -> str:
        return (
            f"Project: {self.name}. "
            f"Type: {self.project_type}. "
            f"Size: {self.square_feet:,} sq ft. "
            f"Scope: {self.description}"
        )