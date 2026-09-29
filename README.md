# FinGuard Orchestrator

An agentic AML compliance audit engine. It takes a month of transactions, finds the patterns US AML
regulation cares about, and produces a report in which **every finding carries the clause it rests
on** — checkable by an examiner, and by construction unable to cite a rule it was not shown.

```
                    ┌─ detection ── arithmetic. finds shapes, judges nothing.
batch ── ingest ────┤                no candidates → $0.0000, the model is never built
                    └─ per candidate: retrieve → ground → critique ⟲ → report → store
                                                  ▲            │
                                                  └─ a thin finding goes back for more law
```

Two halves on opposite sides of the model. **Detection is arithmetic** — five windowed typology
detectors that find shapes and call none of them suspicious. **Grounding is judgement** — the model is
shown one shape and the law that applies to it, and may conclude only what that law supports. A
deterministic gate enforces the second half: a finding may cite only clauses that were actually
retrieved, checked in Python *before* any model is asked its opinion.

- **Jurisdiction:** US only — 31 USC §5324, 31 CFR 1010.311 / 1010.410 / 1020.210 / 1020.320, FFIEC
  manual + Appendix F, FINRA, FinCEN. 731 chunks, 20 sources.
- **Typologies:** `structuring · fan_in · fan_out · cycle · scatter_gather`
- **Data:** synthetic throughout — SAML-D, re-domiciled as a US institution.

## Documentation

| | |
|---|---|
| [docs/DESIGN.md](docs/DESIGN.md) | as built, and the four deviations from the design set |
| [docs/HLD.md](docs/HLD.md) | zones, the three journeys, privacy, observability |
| [docs/LLD.md](docs/LLD.md) | node by node, the contracts, the error taxonomy |
| [docs/CONSTANTS.md](docs/CONSTANTS.md) | every tunable number beside the measurement that chose it |
| [docs/TEST_DESIGN.md](docs/TEST_DESIGN.md) | three tiers, six golden corpora, two CI gates |
| [docs/TEST_RESULTS.md](docs/TEST_RESULTS.md) | **what it scores, including what fails** |
| [docs/CHANGELOG.md](docs/CHANGELOG.md) | what changed and why, with the reversals kept |

The `.docx` files in `docs/` are the design set this was built against; the markdown above are their
as-built companions.

---

## Quickstart

```bash
# 1. install
curl -LsSf https://astral.sh/uv/install.sh | sh     # or: brew install uv
uv sync
cp .env.example .env                                # then set LLM_API_KEY

# 2. build the knowledge base (once; ~10 min, mostly embedding)
uv run finguard-download                            # both corpora + data/MANIFEST.json (sha256)
uv run finguard-chunk --rules                       # → data/processed/chunks/*.jsonl
uv run finguard-store --rules                       # → the `rule_chunks` collection

# 3. make some transactions to audit
uv run finguard-ledger --profile dev                # 3 batches + a clean control

# 4. audit one
uv run finguard-audit --batch data/processed/ledger/2023-05_private_banking_log.pdf
```

Step 4 on the clean control costs **$0.0000** and makes zero model calls — detection finds nothing, so
the model client is never constructed. A batch with patterns in it costs roughly **$0.02 per
candidate**.

## Commands

| command | | needs a key |
|---|---|---|
| `finguard-download` | fetch both corpora, write the manifest | no |
| `finguard-chunk --rules` | tier-aware chunking → `chunks/*.jsonl` | only `--backend openai` |
| `finguard-store --rules` | embed and upsert → `rule_chunks` | no |
| `finguard-store --rule-stats` | chunk counts by tier and authority | no |
| `finguard-store --query "…"` | run a retrieval and print the hits | no |
| `finguard-ledger --profile dev` | render the SWIFT MT103 monthly logs | no |
| `finguard-parse` | MT103 batch → wires, with per-batch stats | no |
| `finguard-benchmark` | recall@k against ObliQA's 2,786 gold questions | no |
| `finguard-map` | ObliQA DocumentIDs → document titles | no |
| **`finguard-audit --batch <path>`** | **batch → `ComplianceReport`** | **yes** |

`finguard-audit` also draws its own graph without running anything:

```bash
uv run finguard-audit --ascii        # terminal diagram, fully offline
uv run finguard-audit --mermaid      # mermaid source
uv run finguard-audit --png g.png    # renders via mermaid.ink (node names only leave the machine)
```

Everything is also reachable the long way — `uv run python -m src.graph.run` is exactly
`uv run finguard-audit` — which is what to use from a checkout you have not synced.

## The service and the cockpit

```bash
export API_AUTH_TOKEN=$(openssl rand -hex 32)
uv run uvicorn src.api.main:app --reload            # :8000
uv run streamlit run src/ui/cockpit.py              # :8501, talks to the API over HTTP
```

| endpoint | |
|---|---|
| `POST /audits` | 202 + `job_id`; `?wait=true` for one round trip; `?force=true` to re-audit |
| `GET /audits/{job_id}` | status, and the report once there is one |
| `GET /audits` | every job on record, newest first |
| `GET /reports?period=YYYY-MM` | stored reports, by month |
| `GET /reports/{id}` | the frozen report joined to its findings' **current** review statuses |
| `GET /reports/{id}/filed` | the same report exactly as filed, with no review applied |
| `GET /reports/{id}/validation` | the ingestion record — what was *not* screened |
| `POST /findings/{id}/review` | `clear` / `escalate` / `approve` → append-only history |
| `GET /health` | 503 unless the collection is genuinely queryable |

Every endpoint but `/health` needs `Authorization: Bearer $API_AUTH_TOKEN`. **A service with no token
configured refuses them all with 503** rather than serving them open — a forgotten token must not be
the same thing as a public AML API. `/health` is exempt because a load balancer cannot carry a secret,
and it returns counts rather than any report content.

Runs are serialised through one worker: two concurrent audits contend for one vector store and one
rate limit, and the failure mode is both getting slower and one hitting a 429. **The same batch posted
twice does not run twice** — dedup is on the sha256 of the uploaded bytes.

Verified against a real uvicorn on the clean control:

```
GET  /health                    → 200  731 vectors                 (no token needed)
GET  /reports                   → 401  WWW-Authenticate: Bearer
POST /audits                    → 202  run-fd6706b71605 · 500 transactions
GET  /audits/run-fd6706b71605   → running · complete · rating none, clean, $0.0000
POST /audits (same bytes)       → 202  same job · deduplicated=true · 1 job on record

-- uvicorn stopped and restarted on the same database --

GET  /audits/run-fd6706b71605   → complete, report rep-run-fd6706b71605
```

The `job_id` **is** the `run_id`, and the report is `rep-{run_id}` — so a Langfuse trace, a job row and
a stored report are one run rather than three id schemes to join.

## Docker

```bash
export LLM_API_KEY=sk-... API_AUTH_TOKEN=$(openssl rand -hex 32)
docker compose up -d api ui          # the engine on :8000 and the cockpit on :8501
docker compose up -d                 # everything, including the Langfuse stack on :3000
```

`api` and `ui` refuse to start without both variables; every `LANGFUSE_*` has a default, because
tracing being unconfigured must mean *off* rather than *broken*. Reports and the Chroma corpus sit on
named volumes so they survive `--force-recreate`.

The image is **3.13 GB**, measured. Where it goes: `torch` (CPU build) 656 MB, `pyarrow`/`scipy`/
`transformers` 140/122/114 MB, the baked MiniLM + TinyBERT models 93 MB. The CPU-torch swap earns its
place by removing roughly a gigabyte of CUDA runtime the lockfile otherwise resolves; the models are
baked in because a container that downloads a model on first use is a container whose first audit fails
behind a firewall. `chroma_db` is deliberately **not** copied — it is an artefact of `finguard-store`,
not source — and mounts **not** read-only, because Chroma is SQLite underneath and opens a journal even
to read.

## Observability

```bash
docker compose up -d langfuse
export LANGFUSE_TRACING=true LANGFUSE_HOST=http://localhost:3000
export LANGFUSE_PUBLIC_KEY=pk-lf-... LANGFUSE_SECRET_KEY=sk-lf-...
```

Self-hosted, because traces here carry reasoning over transaction data. Every payload passes through
`redact` then `trim` on the client, so it covers every span the SDK emits rather than the ones a caller
remembered to sanitise. Measured: **~137 KB → 9.2 KB per run, zero account numbers, zero counterparty
names**, with the parsed ledger replaced by a count. Tracing is optional — with no keys every function
is a no-op, and a run with Langfuse unreachable completes normally.

## Tests and evaluation

```bash
uv run pytest tests/ -q --cov                        # 489 tests, free, no key, no network
uv run python -m eval.run --tier deterministic       # ~30s, free
uv run python -m eval.run --tier live --batches 1    # ~8 min, ~$0.45
```

The suite needs no API key at all, and the PR gate **fails** if one is present — every test that would
reach a model injects a stub, so a real call should fail loudly rather than quietly bill.

Headline numbers, with the failures included:

| | target | measured |
|---|---|---|
| recall, detector level | ≥ 0.90 | **0.827** ✗ — the whole gap is `structuring` at 5/15, diagnosed |
| rules-only baseline | — | 0.907 recall, at **77.6%** of the batch alerted vs our 19.1% |
| faithfulness | 1.00 hard gate | **1.00** (110 checks, 0 violations) |
| context precision hit@1 | ≥ 0.90 | **0.60** ✗ (hit@3 0.80), up from 0.22 |
| prompt injection resisted | 1.00 | **5/5** |
| clean batch | 0 candidates, 0 calls | **$0.0000** |
| branch coverage, audit path | ≥ 85% | **85.6%** |

[docs/TEST_RESULTS.md](docs/TEST_RESULTS.md) has the diagnoses, the cost table, and the seven known
gaps. The three failing metrics are left failing on purpose: a first baseline's job is to be true, and
a threshold moved to fit the number it measures stops measuring anything.

## Layout

```
src/
  config.py          Settings (env) + Config (config.yaml).  numbers never in code
  models.py          every contract: TransactionRecord, Candidate, RuleChunk, Finding, …
  ingestion/         corpus acquisition, tier-aware chunking, the vector store, batch ingestion
  detection/         five typology detectors + the reconciler.  no model
  retrieval/         TierAwareRetriever, the cross-encoder
  graph/             the five nodes, the graph, the run orchestrator, the three prompts
  store/            the results store: immutable reports, mutable statuses, append-only reviews
  api/ · ui/         FastAPI, and a Streamlit cockpit that is a client of it
  observability/     Langfuse, with redaction on the client
eval/                golden datasets, the metric runners, the rules-only baseline
tests/               489 tests
```

Nothing under `src/` imports anything from `eval/` — that is what stops an evaluation fixture becoming
production behaviour.

## Caveats, stated rather than buried

- **The synthetic data is synthetic.** SAML-D models laundering typologies, not US thresholds, which is
  exactly why `structuring` recall is 5/15 on held-out data. See
  [docs/TEST_RESULTS.md](docs/TEST_RESULTS.md).
- **The corpus has no cycle-specific red flag.** All 479 illustrative chunks, searched for
  round-trip language: nothing. A cycle candidate grounds on obligations alone, which is correct
  behaviour and a corpus gap.
- **Neither CI workflow has been executed** — no runner here, and the nightly needs secrets.
- **`docker compose` is configuration-validated, not launched** — no Docker daemon on the build
  machine.
- Pseudonymisation is not anonymisation: `ACCT-` tokens are stable so a finding can be traced back to
  the ledger, which means they are reversible by anyone holding both the token and a candidate list.
  `REDACTION_PEPPER` breaks that at the cost of cross-run comparability.
