# AI-Assisted Estimate Support Tool

## 1. Problem Understanding

A construction company project knowledge across structured records and unstructured documents — estimates, actuals, change orders, scope summaries, and lessons learned.

The goal is to explore whether AI can help estimators find relevant historical projects, understand cost drivers, surface lessons learned, and support better early-stage decisions.

Three things stood out when reading the data:

- **Incomplete data must be handled explicitly.** P-005 has no final actual cost. It must be restricted to lessons learned only — never cost benchmarking.
- **Change orders tell the real story.** The largest cost surprises in P-001, P-002, and P-004 came from late electrical discovery, underestimated access constraints, and specialty systems.

---

## 2. Proposed Solution

**Hybrid RAG (Retrieval-Augmented Generation) pipeline** combining structured database filtering with semantic vector search, feeding retrieved context to an LLM for a grounded estimating report.

**Core principle: the LLM never invents data.** It answers only from retrieved project records. Every figure is traceable to a real project.

```
example_new_project_scope(.md)
        │
        ▼
[ Hybrid Retrieval ]
  Structured Filter (type, sqft, year)  +  Semantic Vector Search (cosine similarity)
        │
        ▼
[ LLM Extraction from New Project Requirement ]  →  name, square_feet, project_type, description
        │
        ▼
[ Governance Checks ]  →  low confidence · incomplete actuals · size mismatch · escalation
        │
        ▼
[ LLM Synthesis ]  →  grounded on retrieved context · cites project IDs · human review required
        │
        ▼
[ Estimator Report ]  →  cost benchmarks · risk flags · lessons learned · confidence notes
```

---

## 4. Key Technical Decisions

| Decision           | Choice                   |
| ------------------ | ------------------------ |
| Embedding model    | `nvidia/nv-embed-v1`     |
| Similarity measure | Cosine similarity        |
| LLM                | `google/gemma-3n-e2b-it` |
| Brief parsing      | LLM JSON extraction      |

---

## 5. Risks and Controls

| Risk                      | Control                                                                                                                                  |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| **Hallucination**         | LLM restricted to retrieved context only. All claims must cite a project ID.                                                             |
| **Low-quality retrieval** | Governance layer flags similarity scores below 0.80 before output is shown.                                                              |
| **Incomplete data**       | P-005 auto-detected and flagged — excluded from cost benchmarking, lessons learned only.                                                 |
| **Cost escalation**       | Projects completed before 2022 flagged with escalation warning in the report.                                                            |
| **Overreliance**          | Mandatory human review statement on every report. Tool supports judgment — never replaces it.                                            |
| **Data privacy**          | Production deploys on Azure AI Search within enterprise boundary. Replaced with Azure OpenAI endpoint — no data leaves the organisation. |

---

## 6. Project Structure

```
ai_estimate_tool/
├── main.py                  ← Entry point
├── requirements.txt
├── .env                     ← OPENAI_API_KEY (never committed)
├── src/
│   ├── models.py            ← Project and NewProjectBrief dataclasses
│   ├── loader.py            ← CSV + markdown loading, LLM brief extraction
│   ├── embedder.py          ← Embedding generation and cosine similarity
│   ├── retriever.py         ← Structured filter + semantic search
│   ├── governance.py        ← Pre-flight checks and warnings
│   ├── context_builder.py   ← Formats retrieved data for LLM
│   ├── synthesizer.py       ← Grounded LLM call
│   └── reporter.py          ← Report assembly and file output
├── structured_data/         ← 4 CSV files
├── project_documents/       ← 12 markdown files
├── new_project/             ← New project brief
└── output/                  ← Timestamped reports saved here
```

---

## 7. Running the Tool

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate          # Mac/Linux
venv\Scripts\activate             # Windows

# Install dependencies
pip install -r requirements.txt

# Add API key
echo "NVIDIA_API_KEY=sk-**********" > .env

# Run
python main.py
```

Report is printed to console and saved to `output/estimate_report_YYYYMMDD_HHMMSS.txt`.

---

## 8. Tools Used

| Tool                                                           | Purpose                                  |
| -------------------------------------------------------------- | ---------------------------------------- |
| Python 3.11                                                    | Core language                            |
| Nvidia `google/gemma-3n-e2b-it`                                | LLM synthesis and brief field extraction |
| Nvidia `nvidia/nv-embed-v1`                                    | Embedding generation                     |
| `python-dotenv`                                                | Secure API key management                |
| Standard library only (`csv`, `json`, `re`, `pathlib`, `math`) | No heavy dependencies                    |

---
