# FinGuard Orchestrator

Enterprise agentic wealth management compliance & AML audit engine.

Ingests batch transaction logs (SWIFT/ISO 20022) and regulatory corpora, audits them
against a semantic compliance knowledge base with a stateful LangGraph agent, and emits
structured Suspicious Activity Reports backed by verifiable citations.

Full design: [`docs/Karthik Project 1 Blueprint.pdf`](docs/).

## Stack

- **Python 3.11**, managed with [`uv`](https://docs.astral.sh/uv/)
- **LangGraph / LangChain** — stateful multi-agent graph (extraction → AML audit → critic → generation)
- **ChromaDB** — vector store for regulatory + ledger grounding
- **Redis** — vector semantic cache
- **FastAPI** — async serving layer
- **Streamlit** — auditor cockpit UI
- **LangSmith / DeepEval / RAGAS** — tracing and evaluation

## Layout

```
src/
  ingestion/   # PDF parsing, semantic chunking, ChromaDB loading
  graph/       # LangGraph state, nodes, edges
  api/         # FastAPI service
  ui/          # Streamlit cockpit
  utils/       # SWIFT log generator, semantic cache
tests/         # pytest + DeepEval eval suite
data/          # datasets (gitignored)
docs/          # project blueprint
```

## Getting started

```bash
# install uv (once) — macOS
brew install uv
# other platforms: curl -LsSf https://astral.sh/uv/install.sh | sh

uv sync
cp .env.example .env   # then fill in keys
```

## Roadmap

- [ ] Phase 1 — Ingestion & semantic grounding
- [ ] Phase 2 — LangGraph agent core
- [ ] Phase 3 — Observability, evals, cost optimization
- [ ] Phase 4 — Streamlit cockpit & Docker packaging
