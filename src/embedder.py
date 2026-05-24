"""
embedder.py
-----------
Handles all embedding operations.

Responsibilities:
  - Generate embedding vectors via OpenAI API
  - Compute cosine similarity between vectors
  - Run ingestion: embed all historical projects at startup

Why embeddings?
  Text is converted into a list of numbers (vector) that captures
  its *meaning*. Similar meanings produce similar vectors.
  This enables semantic search — finding relevant projects even
  when the exact words don't match.
"""
from __future__ import annotations

import math
# pyrefly: ignore [missing-import]
from openai import OpenAI
import os

from src.models import Project
import httpx

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),

    # Optional:
    # verify=False helps if your company network intercepts SSL
    # REMOVE in production if not needed
    http_client=httpx.Client(verify=False))

EMBED_MODEL = "nvidia/nv-embed-v1"   # 1536-dim; swap to Azure ada-002 if needed


def get_embedding(text: str) -> list[float]:
    """
    Convert text → embedding vector using OpenAI's model.

    The vector captures semantic meaning, not just keywords.
    Example: "point-of-sale rough-ins" and "POS infrastructure"
    will produce similar vectors even though the words differ.

    Azure swap (one line):
      Use AzureOpenAI client instead of OpenAI client.
      Everything else stays identical.
    """
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=text.replace("\n", " "),
    )
    return response.data[0].embedding


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Measure semantic similarity between two vectors.
    Range: 0.0 (unrelated) to 1.0 (identical meaning).

    Why cosine and not Euclidean distance?
    Cosine is magnitude-independent — a short and a long document
    about the same topic score equally. Euclidean would penalise
    shorter documents unfairly.

    Production note: Azure AI Search computes this natively at scale.
    We implement it here to keep every step visible and explainable.
    """
    dot   = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a * a for a in vec_a))
    mag_b = math.sqrt(sum(b * b for b in vec_b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def ingest_projects(projects: list[Project]) -> list[Project]:
    """
    Generate and store an embedding vector for every historical project.

    Each project's embedding captures its full profile:
    type, size, region, scope, lessons learned, change orders.

    In production: runs once on data update, persists vectors
    to Azure AI Search or pgvector. Here we store in-memory
    for clarity and portability.
    """
    print(f"\n[Embedder] Generating embeddings for {len(projects)} projects...")
    for p in projects:
        text = p.to_text_for_embedding()
        p.embedding = get_embedding(text)
        status = "✓" if p.final_actual_cost else "⚠ No actuals"
        print(f"  {p.project_id} — {p.project_name[:55]} [{status}]")
    print("[Embedder] Complete.\n")
    return projects