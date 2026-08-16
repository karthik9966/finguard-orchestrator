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
  ingestion/   # dataset acquisition, PDF parsing, semantic chunking, ChromaDB loading
  graph/       # LangGraph state, nodes, edges
  api/         # FastAPI service
  ui/          # Streamlit cockpit
  utils/       # SWIFT log generator, semantic cache
tests/         # pytest + DeepEval eval suite
data/          # MANIFEST.json + document map tracked; raw/ and processed/ gitignored
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

## Data

Two corpora, acquired by script. Neither is committed — `data/raw/` and `data/processed/`
are gitignored — but `data/MANIFEST.json` records the source URL, sha256, size, licence and
retrieval date of every artifact, so the ~1 GB is reproducible from a few KB of tracked JSON.

```bash
uv run python -m src.ingestion.download          # fetch both corpora, write the manifest
uv run python -m src.ingestion.download --check  # verify what's on disk against the manifest
uv run python -m src.ingestion.obliqa_map        # map ObliQA DocumentIDs to document titles
uv run python -m src.utils.pdf_generator         # render SWIFT MT103 monthly logs
```

Nothing here needs credentials. SAML-D is a public Kaggle dataset, and `kagglehub` falls back
to an unauthenticated client when no token is configured. If Kaggle ever refuses the anonymous
download (rate limiting, or a licence you must accept in the browser first), create a token at
kaggle.com → Settings → API → *Create New Token*, save it as `~/.kaggle/kaggle.json`, and
re-run — the script picks it up automatically.

**A. Transaction ledger — [SAML-D](https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml)**
9,504,852 transactions × 12 features, Oct 2022 – Aug 2023, labelled with 28 typologies
(11 normal / 17 suspicious) at a 0.10% suspicious base rate.

`src/utils/pdf_generator.py` turns it into the auditor's actual input: monthly *"Private
Banking Institutional Transaction Logs"* of SWIFT MT103 messages (blocks 1/2/3/4 — BIC
headers, `:20:` reference, `:32A:` value date/currency/amount, `:50K:` ordering customer,
memo lines). It selects whole laundering *cases* rather than isolated flagged rows, and how a
case is found depends on the typology's shape:

- **anchored** — one account concentrates the run (Structuring is a fan-in of many senders
  into one collector; Smurfing is one sender repeating sub-threshold deposits)
- **chained** — the pattern is a path, `A → B → C → A`, walked through the transaction graph.
  Cycle's 382 flagged edges span 382 distinct senders *and* 382 distinct receivers, so no
  account concentrates and anchoring finds nothing.
- **single wire** — the pattern *is* one transaction (Over-Invoicing at £1.1M–12.6M); the
  signal is the implausible amount, not a repeated shape.

So each log contains a pattern that can actually be detected. Ground truth is written to
`data/processed/ledger_labels.csv`, never into the documents.

**B. Regulatory knowledge base** — chunked into ChromaDB by §3.3/§3.4:

| Source | Content |
|---|---|
| [ObliQA / ADGM](https://github.com/RegNLP/ObliQADataset) | 40 ADGM rulebooks, 13,732 passages, ~876k words. DocumentID 1 is the AML Rulebook. |
| FINRA Rule 3310 / 3110 | Rule text, scraped — FINRA publishes its rulebook as HTML only. |
| FINRA Regulatory Notice 19-18 | 104 money laundering red flags for broker-dealers (PDF). |
| FinCEN alerts (×3) | Shell company, real estate and sanctions-evasion red flags (PDF). |

ObliQA carries no document titles and its files are not in DocumentID order, so
`src/ingestion/obliqa_map.py` recovers the mapping by matching passage text and writes
`data/obliqa_document_map.json` — without it every citation would read "Document 1, §1.1.1".

> SAML-D is licensed **CC BY-NC-SA 4.0** (non-commercial, attribution required):
> B. Oztas, D. Cetinkaya, F. Adedoyin, M. Budka, H. Dogan and G. Aksu, "Enhancing Anti-Money
> Laundering: Development of a Synthetic Transaction Monitoring Dataset," *2023 IEEE ICEBE*,
> pp. 47–54, doi:10.1109/ICEBE59045.2023.00028

## Roadmap

- [ ] Phase 1 — Ingestion & semantic grounding
  - [x] §3.1 Environment setup
  - [x] §3.2 Dataset acquisition & SWIFT log generation
  - [ ] §3.3 Semantic chunking (cosine distance)
  - [ ] §3.4 ChromaDB loading & metadata
- [ ] Phase 2 — LangGraph agent core
- [ ] Phase 3 — Observability, evals, cost optimization
- [ ] Phase 4 — Streamlit cockpit & Docker packaging
