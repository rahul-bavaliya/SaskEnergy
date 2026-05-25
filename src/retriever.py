"""

Hybrid retrieval pipeline: structured filter + semantic search.

Two-step approach:
  Step 1 — Structured filter (SQL-style WHERE clause)
            Narrows candidates by metadata: type, size, year, exclusions.
            Fast and deterministic.

  Step 2 — Semantic search (cosine similarity on embeddings)
            Ranks filtered candidates by meaning similarity to the query.
            Finds relevant projects even when exact words don't match.
"""

from __future__ import annotations

from typing import Optional

from src.models import Project, NewProjectBrief
from src.embedder import get_embedding, cosine_similarity


def structured_filter(
    projects:     list[Project],
    project_type: Optional[str]       = None,
    min_sqft:     Optional[int]       = None,
    max_sqft:     Optional[int]       = None,
    min_year:     Optional[int]       = None,
    exclude_ids:  Optional[list[str]] = None,
) -> list[Project]:
    """
    Step 1 — Hard metadata filters.

    Uses generous tolerances (±50% size, last 8 years) so we don't
    over-filter. Semantic ranking in Step 2 will refine quality.

    P-006 (cosmetic refresh) and P-003 (ground-up shell) are excluded
    by ID because their cost profiles are fundamentally different from
    a retail tenant build-out — even if their text vocabulary overlaps.
    """
    result = projects
    if project_type:
        result = [
            p for p in result
            if project_type.lower() in p.project_type.lower()
            or p.project_type.lower() in project_type.lower()
        ]
    if min_sqft is not None:
        result = [p for p in result if p.square_feet >= min_sqft]
    if max_sqft is not None:
        result = [p for p in result if p.square_feet <= max_sqft]
    if min_year is not None:
        result = [p for p in result if p.year >= min_year]
    if exclude_ids:
        result = [p for p in result if p.project_id not in exclude_ids]

    return result


def semantic_search(
    query_embedding:    list[float],
    candidate_projects: list[Project],
    # top_k:              int = 3,
) -> list[tuple[Project, float]]:
    """
    Step 2 — Semantic similarity ranking.

    Computes cosine similarity between the query embedding and every
    candidate project's embedding. Returns the top-K most similar.

    Score interpretation:
      > 0.90  — very close match
      0.80–0.90 — good match
      0.70–0.80 — moderate, use with caution
      < 0.70  — weak match, governance warning triggered
    """
    scored = [
        (p, cosine_similarity(query_embedding, p.embedding))
        for p in candidate_projects
        if p.embedding
    ]

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def retrieve(
    new_project:  NewProjectBrief,
    all_projects: list[Project],
    # top_k:        int = 3,
) -> list[tuple[Project, float]]:
    """
    Full hybrid retrieval pipeline.

    Orchestrates:
      1. Structured filter  — narrow candidates by metadata
      2. Embed the query    — convert new project brief to vector
      3. Semantic search    — rank filtered candidates by meaning

    Fallback: if the strict filter leaves fewer than 2 candidates,
    we widen to the full corpus (minus hard exclusions) so the LLM
    always has enough context to work with.

    Returns: list of (Project, similarity_score) tuples, best first.
    """
    print(f"[Retriever] Processing: {new_project.name}")
    print(f"  Size: {new_project.square_feet:,} sqft | Type: {new_project.project_type}")

    # Step 1 — Structured filter
    candidates = structured_filter(
        all_projects,
        project_type = new_project.project_type,
        min_sqft     = int(new_project.square_feet * 0.5),   # ±50% size tolerance
        max_sqft     = int(new_project.square_feet * 1.5),   # ±150% size tolerance
        min_year     = min(project.year for project in all_projects),
        exclude_ids  = ["P-006", "P-003"],  # cosmetic refresh & ground-up shell
    )
    print(f"  Structured filter: {len(all_projects)} → {len(candidates)} candidates")

    # Fallback: widen if too few candidates remain
    if len(candidates) < 2:
        print("** Too few candidates — widening to full corpus (minus exclusions)")
        candidates = [
            p for p in all_projects
            if p.project_id not in ["P-006", "P-003"]
        ]

    # Step 2 — Embed the new project query
    query_embedding = get_embedding(new_project.to_query_text())

    # Step 3 — Semantic ranking
    results = semantic_search(query_embedding, candidates)

    print(f"  Top {len(results)} matches:\n")
    for rank, (p, score) in enumerate(results, 1):
        actuals = (
            f"${p.final_actual_cost:,.0f} actual"
            if p.final_actual_cost else "actuals incomplete"
        )
        print(f"    {rank}. {p.project_id} — {p.project_name[:50]}")
        print(f"       Similarity: {score:.3f} | {p.square_feet:,} sqft | {actuals}")
    print()

    return results