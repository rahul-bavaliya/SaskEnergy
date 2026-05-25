"""

Loads all data from real files into Project objects.

Responsibilities:
  - Read all 4 CSVs from structured_data/
  - Join them by project_id
  - Read markdown files from project_documents/
  - Read the new project brief from new_project/ using LLM extraction


No embedding happens here — that lives in embedder.py.
"""
from __future__ import annotations


import csv
import json
import os
import re
from pathlib import Path
from typing import Optional

from openai import OpenAI
import httpx

from src.models import Project, NewProjectBrief

# ── LLM client (reuses same API key as synthesizer) ──
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),

    # Optional:
    # verify=False helps if your company network intercepts SSL
    # REMOVE in production if not needed
    http_client=httpx.Client(verify=False))

CHAT_MODEL = "google/gemma-3n-e2b-it"  

# ── Folder paths (relative to project root) ──
BASE_DIR        = Path(__file__).parent.parent
STRUCTURED_DIR  = BASE_DIR / "structured_data"
DOCUMENTS_DIR   = BASE_DIR / "project_documents"
NEW_PROJECT_DIR = BASE_DIR / "new_project"


def _load_csv(path: Path) -> list[dict]:
    """Generic CSV loader — returns list of row dicts."""
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_markdown(path: Path) -> str:
    """
    Load a markdown file as plain text.
    Strips heading markers (#) so they don't pollute embeddings.
    """
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    return text.strip()


# ─────────────────────────────────────────────
# LLM-BASED FIELD EXTRACTION
# ─────────────────────────────────────────────

EXTRACTION_SYSTEM_PROMPT = """
You are a data extraction assistant. Your job is to read a construction
project brief and extract structured information from it.

You must respond with ONLY a valid JSON object — no explanation, no markdown
fences, no preamble. Just the raw JSON.

Extract these fields:

{
  "name": "Full project name or title from the document heading",
  "square_feet": 18000,
  "project_type": "One of: Tenant build-out | Tenant improvement | Renovation | Cosmetic refresh | Ground-up shell | Office conversion | Medical office | Other",
  "region": "City or region if mentioned, else null",
  "description": "A clean full summary of the project including all the key scope items mentioned in the document"
}

Rules:
- square_feet must be an integer (no commas, no units)
- If square footage is not mentioned, use 0
- Pick the closest matching project_type from the allowed list
- description should capture each and every small details about the project
- If a field cannot be determined, use null
"""


def extract_brief_with_llm(raw_text: str) -> dict:
    """
    Send the raw markdown brief to the LLM and get back structured JSON.

    Why LLM extraction instead of regex?
    - Handles any phrasing: "approx 18,000 sqft", "eighteen thousand square feet",
      "±18K sf" — regex breaks on variations, LLM understands them all.
    - Extracts semantic fields like project_type and description that
      regex cannot reliably determine.
    - Works on any future project brief without code changes.
    - Returns a clean, typed dict ready to build NewProjectBrief from.
    """
    print("[Loader] Extracting project brief fields using LLM...")

    response = client.chat.completions.create(
        model    = CHAT_MODEL,
        messages = [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user",   "content": f"Extract fields from this project brief:\n\n{raw_text}"},
        ],
        temperature = 0.0,    # fully deterministic — this is structured extraction
        max_tokens  = 1200,
    )

    raw_response = response.choices[0].message.content.strip()

    # Strip markdown fences if the LLM accidentally adds them
    clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_response, flags=re.DOTALL).strip()

    try:
        extracted = json.loads(clean)
    except json.JSONDecodeError as e:
        print(f"  ⚠ [Loader] LLM returned invalid JSON: {e}")
        print(f"  Raw response: {raw_response[:200]}")
        # Fallback to safe defaults
        extracted = {
            "name":         "New Project",
            "square_feet":  0,
            "project_type": "Tenant build-out",
            "region":       None,
            "description":  raw_text[:500],
        }

    return extracted


def load_new_project_brief() -> NewProjectBrief:
    """
    Load and parse the new project brief from new_project/*.md.

    Uses the LLM to extract:
      - name         : project title from the document heading
      - square_feet  : numeric area in square feet
      - project_type : classified from allowed type list
      - region       : city/region if mentioned
      - description  : clean 2-3 sentence scope summary

    Advantage over regex: works on any phrasing, any format, any future
    brief without changing a single line of code.
    """
    brief_path = NEW_PROJECT_DIR / "example_new_project_scope.md"
    raw_text   = brief_path.read_text(encoding="utf-8")

    # ── LLM extracts all fields ──
    extracted = extract_brief_with_llm(raw_text)

    name         = extracted.get("name")         or "New Project"
    square_feet  = int(extracted.get("square_feet") or 0)
    project_type = extracted.get("project_type") or "Tenant build-out"
    description  = extracted.get("description")  or raw_text

    print(f"[Loader] New project brief extracted:")
    print(f"  Name         : {name}")
    print(f"  Square Feet  : {square_feet:,} sqft")
    print(f"  Project Type : {project_type}")
    if extracted.get("region"):
        print(f"  Region       : {extracted['region']}")
    print(f"  Description  : {description}")
    print("\n")

    return NewProjectBrief(
        name         = name,
        square_feet  = square_feet,
        project_type = project_type,
        description  = description,
    )

# ─────────────────────────────────────────────
# HISTORICAL PROJECT LOADER
# ─────────────────────────────────────────────

def load_all_projects() -> list[Project]:
    """
    Build the complete Project list by joining all data sources:

      projects.csv              → base metadata
      estimate_line_items.csv   → per-category estimates
      actual_costs.csv          → per-category actuals
      change_orders.csv         → change order records
      P-XXX_scope_summary.md    → unstructured scope text
      P-XXX_lessons_learned.md  → unstructured lessons text

    Returns a list of fully populated Project objects,
    ready for embedding and retrieval.
    """
    # ── Load all CSVs ──
    project_rows  = _load_csv(STRUCTURED_DIR / "projects.csv")
    estimate_rows = _load_csv(STRUCTURED_DIR / "estimate_line_items.csv")
    actual_rows   = _load_csv(STRUCTURED_DIR / "actual_costs.csv")
    co_rows       = _load_csv(STRUCTURED_DIR / "change_orders.csv")

    # ── Index supporting tables by project_id ──
    estimates_by_project: dict[str, dict] = {}
    for row in estimate_rows:
        pid = row["project_id"]
        estimates_by_project.setdefault(pid, {})
        estimates_by_project[pid][row["category"]] = float(row["estimate_amount"])

    actuals_by_project: dict[str, dict] = {}
    for row in actual_rows:
        pid = row["project_id"]
        actuals_by_project.setdefault(pid, {})
        actuals_by_project[pid][row["actual_category"]] = float(row["actual_amount"])

    cos_by_project: dict[str, list] = {}
    for row in co_rows:
        pid = row["project_id"]
        cos_by_project.setdefault(pid, [])
        cos_by_project[pid].append(row)

    # ── Build Project objects ──
    projects = []
    for row in project_rows:
        pid = row["project_id"]

        # P-005 has no final_actual_cost — handle gracefully
        actual_cost_raw = row["final_actual_cost"].strip()
        final_actual_cost: Optional[float] = (
            float(actual_cost_raw) if actual_cost_raw else None
        )

        # Load markdown files (graceful fallback if file missing)
        scope_path   = DOCUMENTS_DIR / f"{pid}_scope_summary.md"
        lessons_path = DOCUMENTS_DIR / f"{pid}_lessons_learned.md"
        scope_text   = _load_markdown(scope_path)   if scope_path.exists()   else ""
        lessons_text = _load_markdown(lessons_path) if lessons_path.exists() else ""

        project = Project(
            project_id          = pid,
            project_name        = row["project_name"],
            year                = int(row["year"]),
            region              = row["region"],
            building_type       = row["building_type"],
            project_type        = row["project_type"],
            square_feet         = int(row["square_feet"]),
            complexity          = row["complexity"],
            delivery_model      = row["delivery_model"],
            original_estimate   = float(row["original_estimate"]),
            final_actual_cost   = final_actual_cost,
            schedule_months     = float(row["schedule_months"]),
            occupied_site       = row["occupied_site"],
            project_notes       = row["notes"],
            estimate_line_items = estimates_by_project.get(pid, {}),
            actual_line_items   = actuals_by_project.get(pid, {}),
            change_orders       = cos_by_project.get(pid, []),
            scope_summary       = scope_text,
            lessons_learned     = lessons_text,
        )
        projects.append(project)

    print(f"[Loader] Loaded {len(projects)} historical projects.")
    return projects