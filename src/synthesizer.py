"""
LLM synthesis layer — the 'G' in RAG (Retrieval-Augmented Generation).

Responsibilities:
  - Define the system prompt (grounding rules, output structure)
  - Call the LLM with retrieved context + new project brief
  - Return the grounded, cited estimating analysis

Key design decisions:
  - temperature=0.2 for consistent, factual output
  - System prompt explicitly forbids using training data for project figures
  - LLM must cite project IDs for every specific claim
  - LLM must always end with the human review statement
  - Context block is the ONLY allowed knowledge source
"""

from __future__ import annotations

import os
from openai import OpenAI
import httpx
from src.models import NewProjectBrief

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),

    # Optional:
    # verify=False helps if your company network intercepts SSL
    # REMOVE in production if not needed
    http_client=httpx.Client(verify=False))

CHAT_MODEL = "google/gemma-3n-e2b-it"    # swap to gpt-4o for higher quality output
                              # swap to Azure OpenAI for enterprise deployment

SYSTEM_PROMPT = """
You are an AI assistant supporting commercial construction estimators.

You help estimators understand comparable historical projects and prepare
better early-stage cost estimates for retail build-out projects.

CRITICAL RULES — follow these exactly:
1. ONLY use information from the COMPARABLE PROJECTS context provided.
   Never invent project-specific cost figures from your training data.
2. Always cite the project ID (e.g. P-001) when referencing a specific project.
3. If a project shows "INCOMPLETE" actuals, do NOT use it for cost benchmarking.
   You may reference its lessons learned and scope observations only.
4. Clearly flag any data gaps, low-confidence areas, or caveats.
5. Be structured, concise, and directly useful to a working estimator.
   Use clear numbered headings. Avoid generic filler sentences.
6. End every response with this exact statement:

⚠ HUMAN REVIEW REQUIRED — This AI analysis supports estimating decisions.
It does not replace professional judgment, site verification, or quantity takeoff.
"""

OUTPUT_SECTIONS = """
Please provide a structured estimating support report with these six sections:

1. COMPARABLE PROJECT SUMMARY
   Which retrieved projects are most relevant and why. Note any that are
   less comparable (e.g. different delivery constraints, building type).

2. COST BENCHMARKS
   Based on comparable actuals: estimated cost range in total $ and $/sqft.
   Show your reasoning. Clearly flag any projects excluded from benchmarking.

3. KEY COST DRIVERS
   What categories typically drive cost for this project type, based on
   the estimate vs actual category breakdowns in the comparable projects.

4. RISK FLAGS & CHANGE ORDER PATTERNS
   What change orders appeared in comparable projects?
   Which were avoidable? What specific risks apply to this new project?

5. LESSONS LEARNED
   The most actionable lessons from comparable projects, directly
   applicable to this new project brief. Be specific, not generic.

6. CONFIDENCE & DATA GAPS
   What assumptions or gaps should the estimator verify before
   finalising the estimate? Be explicit about data limitations.
"""


def synthesize(new_project: NewProjectBrief, context: str) -> str:
    """
       The LLM is grounded by:
      1. System prompt rules (no hallucination, always cite)
      2. Context block that contains ONLY real retrieved project data
      3. Low temperature (0.2) for consistent, factual output

    Returns the full LLM response as a string.
    """
    user_prompt = f"""
   NEW PROJECT REQUIRING ESTIMATE:
   Name        : {new_project.name}
   Type        : {new_project.project_type}
   Size        : {new_project.square_feet:,} sq ft
   Description : {new_project.description}

   COMPARABLE HISTORICAL PROJECTS (your only allowed knowledge source):
   {context}

   {OUTPUT_SECTIONS}
   """

    print("[Synthesizer] Calling LLM for grounded analysis...")
    response = client.chat.completions.create(
        model       = CHAT_MODEL,
        messages    = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        temperature = 0.2,
        max_tokens  = 1800,
    )
    return response.choices[0].message.content