# High-Level Design

**As-built**, 2026-09-08. Every figure here is measured on this repository, not projected.
Companion documents: [LLD.md](LLD.md) for contracts, [DESIGN.md](DESIGN.md) for the short version.

---

## 1 · Problem

An ADGM-regulated private bank receives a monthly ledger of SWIFT MT103 payment messages. A
compliance analyst must decide which payments warrant a Suspicious Activity Report, and cite the
regulatory obligation that requires it. Doing that by hand over 220 messages is slow; doing it by
asking a language model to read all 220 is both expensive and unaccountable — the resulting claims
cannot be traced back to any specific rule.

**Two constraints shape everything downstream.**

*A miss is a regulatory failure; a false alarm costs about five analyst-minutes.* The system is
therefore tuned for recall, and its second stage exists to narrow the resulting over-selection.

*A citation that cannot be resolved to a stored clause is worthless* — worse than worthless, since
it looks like authority. Traceability is a hard requirement, not a feature.

## 2 · Shape of the solution

```
 batch PDF (220 MT103 messages)
        │
        ▼
  ┌───────────────────────────── deterministic, $0.00 ─────────────────────────────┐
  │  parse      regex state machine        →  220 typed Wire records               │
  │  detect     four shape primitives      →  ~1-16 Candidates (geometry only)     │
  │  route      candidates == [] ?          →  END at $0.00, with a written report │
  │  audit      shape → obligation queries  →  ≤24 clauses from ChromaDB           │
  └────────────────────────────────────────┬───────────────────────────────────────┘
                                           ▼
  ┌────────────────────────────── judgement, ~$0.10 ──────────────────────────────┐
  │  draft      gpt-4o          findings connecting patterns to clauses            │
  │  critic     Python veto, then gpt-4o support score                             │
  │             └── thin? → back to audit with a reformulated query (max 2)        │
  │  generate   gpt-4o bound to ComplianceReport, then Python repairs 3 fields     │
  └───────────────────────────────────────────────────────────────────────────────┘
```

Four of the seven nodes never reach a model. A batch that produces no candidates terminates at
**$0.0000** with a written negative result — silence would be indistinguishable from a crash.

## 3 · Why a graph

`parse → … → generate` is a straight line that a `for` loop expresses perfectly well. The edge
that justifies LangGraph is **`critic → audit`**: a critic that can only approve or reject is a
filter; one that can reformulate the question and send execution back to retrieval is the
difference between a report that says *"no clause covers this"* and one that goes and finds the
clause.

It returns to **retrieval**, not to drafting, deliberately — a thin finding is usually missing law
rather than bad prose, and re-drafting the same material cannot fix that.

## 4 · Component view

| component | responsibility | model? |
|---|---|---|
| `ingestion/` | 46 documents → 12,273 chunks → ChromaDB, tiered by AML relevance | local embeddings |
| `utils/swift_parser` | MT103 → `Wire`. Refuses rather than guesses | no |
| `utils/detectors` | four geometric primitives → `Candidate` | no |
| `graph/prompts` | shape → obligation-shaped queries; all prompt text | no |
| `graph/nodes` | the seven node functions and two routers | 3 calls |
| `graph/rerank` | cross-encoder reordering of each query's 15 hits | local, 3 MB |
| `graph/cost` | per-node token and dollar accounting | no |
| `utils/cache` | Redis cache for retrieved clauses | local embeddings |
| `ui/cockpit` · `api/main` | the two front doors | — |

## 5 · Data

**Transactions** — SAML-D (Kaggle), rendered into synthetic MT103 logs by `pdf_generator.py`,
which also writes `ledger_labels.csv`: the answer key naming every planted laundering wire and its
typology. Four batches of 220. One (2023-05) is a deliberate clean control.

**Regulations** — ObliQA (40 ADGM documents) plus FINRA/FinCEN advisories. Semantically chunked,
embedded with `all-MiniLM-L6-v2`, stored in one ChromaDB collection with a relevance tier.

Both corpora are gitignored and reproducible from `data/MANIFEST.json`.

## 6 · Three cross-cutting decisions

**Tiering, not filtering by document.** All 40 ObliQA documents are indexed; 2.9% of passages are
AML-bearing and the rest are the distractor set that makes Context Precision measurable at all.
Tier 1 opens by default, tier 2 when a candidate has a cross-border leg — a deterministic rule,
because the alternative is putting a 46-document inventory into every prompt.

**No transaction vectors.** Measured on 220 messages, laundering and clean wires separate by
+0.029 cosine — noise, since ~55 of ~65 tokens are boilerplate. Parsed wires belong in a table
queried with pandas, not in a vector store.

**The router fires on `candidates == []`.** The blueprint's predicate ("contains a cross-border
wire") can never fire: cross-border is 9.77% of SAML-D, so a fully domestic 220-wire batch has
probability 1.5 × 10⁻¹⁰. Routing on an empty candidate list is the decision that actually saves
money, and it is evaluated *after* two free nodes.

## 7 · Trust boundaries

The model is untrusted for anything checkable. Three mechanisms enforce that:

| mechanism | catches | where |
|---|---|---|
| **citation veto** — cited clause absent from retrieval ⇒ `score = 0.0` | fabricated law | `critic_node` |
| **evidence repair** — `flagged_wires`, `source_document_hashes` recomputed | wrong identifiers in a filing | `generate_node` |
| **rating cap** — High requires confidence ≥ 0.9 | over-claiming on a thin finding | `generate_node` |

Each exists because the failure was **observed**, not anticipated. The model returned account
numbers where wire references belong; returned an empty hash list beside a live citation; and
rated a clean batch High. On 2026-09-08 the veto fired on a live May run against two genuinely
fabricated citations.

## 8 · Deployment

Three interfaces over one engine — none contains audit logic; all call `build_graph()`:

- **CLI** — `finguard-audit`, plus eight sibling commands for the ingestion pipeline
- **Streamlit cockpit** — upload, live node-by-node tracker, report, citations drawer, telemetry
- **FastAPI** — `POST /audit` returns an id and works in the background; a 40-second synchronous
  request is a timeout waiting for a proxy to find it

Packaged as a 3.13 GB multi-stage image (CPU-only torch, models baked in, `HF_HUB_OFFLINE=1`), with
ChromaDB mounted rather than copied. `docker-compose.yml` brings the API and Redis up together.

## 9 · Observability

LangSmith traces every node — not only model calls, since LangGraph compiles each node into a
Runnable. A run emits 16 spans (13 `chain`, 3 `llm`) under one `audit_id` that is also the API's
resource id. Run-level metadata carries identity; per-call metadata carries node, loop number and
clause count, which is what makes *"which context caused the loop"* answerable.

**One consequence, stated plainly:** tracing uploads full node inputs and outputs — 137 KB per run,
including all 220 parsed wires with names and account numbers, most of which no model ever sees.
Fine for a synthetic ledger, a real decision before pointing it at live payment data.

Cost is measured locally as well, via a callback (three of four model calls use structured output,
which discards the token counts), so spend is visible with tracing off.

## 10 · Measured characteristics

| | |
|---|---|
| parsing | **880/880** wires, every field matching the answer key |
| detection | **100% recall** (52/52), 32% precision, 164/660 wires swept |
| retrieval | hit@15 79.2%; reranking lifts hit@1 45.2% → 55.6% |
| cost | **$0.047–$0.178** per batch, median $0.093; $0.0000 on the free path |
| latency | ~45s cold, ~24s warm cache |
| tests | **233**, ~40s, no API key, no network |

## 11 · Known limits

- **Risk rating does not reliably separate clean from dirty batches.** Deferred; a failing test.
- **17.2% of gold questions have no correct clause in the top 15.** A retrieval ceiling that
  reranking cannot lift — it reorders, it cannot add.
- **Detector precision 32%.** Deliberate, given the recall trade.
- **Single-instance API.** The audit registry is an in-process dict.
- **`ainvoke` over synchronous nodes** — correct, non-blocking, but not true async.
