"""
main.py
-------
Entry point for the AI-Assisted Estimate Support Tool.

Orchestrates the full pipeline in 7 clean steps:
  1. Load     — read all CSVs and markdown files
  2. Ingest   — generate embeddings for all historical projects
  3. Brief    — load the new project brief
  4. Retrieve — hybrid search: structured filter + semantic ranking
  5. Govern   — run pre-flight governance checks
  6. Synthesize — grounded LLM analysis
  7. Report   — assemble and save the final report
"""

from dotenv import load_dotenv
load_dotenv()   # load NVIDIA_API_KEY from .env before any imports

from src import (
    load_all_projects,
    load_new_project_brief,
    ingest_projects,
    retrieve,
    run_governance_checks,
    build_context,
    synthesize,
    assemble_report,
    save_report,
)


def main():
    print("\n" + "=" * 60)
    print("  AI-Assisted Estimate Support Tool")
    print("  SaskEnergy Assessment")
    print("=" * 60 + "\n")

    # ── Step 1: Load all projects data ──
    projects = load_all_projects()

    # ── Step 2: Embed all projects ──
    projects = ingest_projects(projects)

    # ── Step 3: Load new project brief ──
    new_project = load_new_project_brief()
    print(f"[Main] New project: {new_project.name} | {new_project.square_feet:,} sqft\n")

    # ── Step 4: Hybrid retrieval ──
    retrieved = retrieve(new_project, projects)

    # ── Step 5: Governance checks ──
    warnings = run_governance_checks(new_project, retrieved)
    if warnings:
        print("[Governance] Warnings flagged:")
        for w in warnings:
            print(f"  {w}")
        print()

    # ── Step 6: Build context and synthesize ──
    context    = build_context(retrieved)
    llm_output = synthesize(new_project, context)

    # ── Step 7: Assemble and save report ──
    report      = assemble_report(new_project, retrieved, warnings, llm_output)
    output_path = save_report(report)

    # ── Print to console ──
    print(report)
    print(f"\n[Done] Report saved → {output_path}")


if __name__ == "__main__":
    main()