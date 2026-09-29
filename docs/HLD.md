# HLD as built

Companion to `FinGuard_HLD.docx`. What the architecture is, where the boundaries fell, and which
HLD claims are enforced by code rather than by intention.

---

## §1.1 Scope — what is enforced, not merely stated

| in scope | where it lives | enforced by |
|---|---|---|
| 1a. Knowledge-base ingestion | `src/ingestion/` | `finguard-download` → `finguard-chunk --rules` → `finguard-store --rules`; a manifest with sha256 per source |
| 1b. Transaction-batch ingestion | `src/ingestion/batch.py` | `TransactionBatchIngestor` → `(records, ValidationReport)` |
| 2. Rule KB + retrieval | `src/retrieval/` | two tiers, two mechanisms — id lookup and semantic search |
| 3. Agentic reasoning core | `src/graph/` | five LangGraph nodes, per-candidate loop |
| 4. Structured report generation | `src/graph/nodes.py` | `ReportGenerationNode`, **no model call** |
| 5. Serving layer | `src/api/`, `src/ui/` | bearer-authenticated FastAPI; Streamlit over HTTP |
| 6. Results store | `src/store/results.py` | SQLite/Postgres by URL; immutable reports |
| Observability | `src/observability/` | Langfuse, self-hosted, redaction on the client |
| (Offline) eval harness incl. rules-only baseline | `eval/` | *not* importable from `src/` |

The last row is a boundary worth naming: **nothing under `src/` imports anything from `eval/`.** That
is the property that stops an evaluation fixture from quietly becoming production behaviour.

Out of scope, and enforced:

- **Non-US law** — `RuleChunk.jurisdiction` rejects anything but `US` on the model itself. ObliQA's
  ADGM corpus is fenced into `obliqa_benchmark`, never retrieved from at runtime, kept only because
  its 2,786 labelled questions are the sole ground truth this project has for retrieval quality.
- **Cross-month memory** — no state crosses a batch, which is what bounds every detector's window by
  construction rather than by a check.
- **Excluded typologies** — `OUT_OF_SCOPE_TYPOLOGIES` keeps them in the ledgers as *unflagged*
  context, so precision is measurable against activity that genuinely looks odd, and never plants
  them as cases: a detector is not expected to find them and seeding one would dilute recall.

## §2 Architecture — three zones plus observability

```
   ┌── Serving & Cockpit ─────────────────────────────────────────────┐
   │  FastAPI (bearer)        Streamlit cockpit ── httpx ──┐          │
   │      │  single worker, one job at a time              │          │
   └──────┼──────────────────────────────────────────────────┼─────────┘
          │                                                 │
   ┌──────▼── Reasoning Core ──────────────────┐   ┌─────────▼─────────┐
   │  detection → retrieval → grounding →      │   │  Results store    │
   │  critique ⟲ (per candidate) → report      │   │  reports (frozen) │
   └──────┬────────────────────────┬───────────┘   │  findings.status  │
          │                        │               │  reviews (append) │
   ┌──────▼── Ingestion ──────┐  ┌─▼─ Grounding ──┐│  jobs             │
   │  MT103 → TransactionRecord│  │ Chroma         │└───────────────────┘
   │  light-model rescue ×1    │  │ rule_chunks    │
   │  quarantine the rest      │  │ MiniLM+FlashRank│
   └───────────────────────────┘  └────────────────┘
                    └── Langfuse (self-hosted), redacted ──┘
```

The build-once rulebook path (`download → chunk → store`) is separate from the per-run transaction
path, and they meet only at retrieval. That separation is why re-indexing the corpus does not mean
rebuilding the image, and why `chroma_db` mounts rather than being copied in.

## §2.2 The three journeys, as built

**Journey 1 — monthly audit, human in the loop.** An analyst uploads in the cockpit; the cockpit
POSTs to the API; the API queues one job on a single worker; the cockpit polls. The report renders
with each finding's own clauses, and clear / escalate / approve write to the store. Verified live end
to end on the clean control at $0.0000.

**Journey 2 — automated API run.** `POST /audits` with a bearer token → 202 + `job_id`, or
`?wait=true` for one round trip. See [DESIGN.md](DESIGN.md) deviation 3 for why both exist.

**Journey 3 — audit-defence lookup.** `GET /reports?period=YYYY-MM` then `GET /reports/{id}`. No
re-analysis: the stored report carries the rule *as cited at the time*. The distinction that makes
this work is described under §5 below.

## §4 Model tiering

Two model roles, named separately so the reservation is real rather than aspirational:

| role | setting | used for |
|---|---|---|
| reasoning | `REASONING_MODEL` (default `gpt-4o`) | grounding, critique |
| light | `LIGHT_MODEL` (default `gpt-4o-mini`) | the extraction rescue, once per refused message |

Everything else runs in-environment: MiniLM embeddings, Chroma, FlashRank's cross-encoder. A clean
month therefore costs **$0.0000** — not "almost nothing", zero, because the model client is never
constructed.

## §5 Privacy

The HLD's posture is a hosted frontier model under zero-data-retention, with identifiers stripped
before any external call. As built, there is exactly one redaction function and it guards both exits:

1. the grounding context, before it reaches the model, and
2. the observability payload, before it reaches Langfuse.

One function for both, so they cannot drift. Accounts are **pseudonymised** rather than anonymised —
`ACCT-` plus eight hex characters, stable per account — because a fan-in narrative that cannot say
"these eleven senders all paid the same account" is useless and an analyst has to be able to map a
finding back to the ledger. Determinism is bought with reversibility, and `REDACTION_PEPPER` breaks
the correlation at the cost of cross-run comparability; on synthetic data the default is the right
trade, and stating it is better than implying an anonymity guarantee that is not there.

**Memos are scrubbed, not dropped.** A memo line is the one realistic prompt-injection vector a
payment message has, so removing it would make the injection fixtures vacuous. The text reaches the
prompt as inert data with identifier-shaped substrings masked, and the grounding system prompt names
it as untrusted.

Transaction references and amounts survive deliberately. A trace or a report that cannot say *which*
transactions a finding covers, or for how much, is not usable as an audit record.

## §6 Observability

Langfuse, self-hosted, so traces carrying reasoning over transaction data stay in-environment.

**The defect this closed was measured.** The pre-migration system traced to a hosted project and
uploaded ~137 KB per run — every parsed wire with its counterparty names, account numbers and
addresses, of which about 21 ever reached a model. Every span records its inputs and outputs and
`parse` was a node, so the whole book went up as a side effect of instrumenting the graph.

The fix is one line, set on the client rather than at call sites:

```python
mask = lambda data: trim(redact(data))
```

`redact` decides what may leave at all; `trim` decides how much is worth sending, because
pseudonymised bulk is still bulk. Measured on the clean control, captured from the OpenTelemetry
exporter rather than estimated: **~137 KB → 9.2 KB across 5 spans, zero account numbers, zero
counterparty names**, and the parsed ledger replaced by `[500 record(s) — omitted from the trace]`.

Tagging follows §6: run and batch id, period, record count and client tier at run level; node,
candidate index and `loop:N` per span; and one `critique` score **per finding** rather than per run,
so four confident findings and one the review could not stand behind do not average into a single
reassuring number.

Tracing is entirely optional. With no keys configured every function is a no-op — verified live: a run
with tracing on and Langfuse unreachable completes normally, reports its target honestly, and costs
$0.0000. An audit must not fail because an observability stack is down.

## §3 Caching — deliberately absent

The pre-migration system cached retrieved clauses in Redis and it measurably worked (audit node 44.0s
→ 0.0s warm, 7/7 hits, context byte-identical). It is gone, because the new design removed the
conditions that made it pay: a monthly batch has no high-frequency duplicate-query volume, and
per-candidate retrieval makes each query more specific than the batch-level ones that used to repeat.
Recorded rather than deleted silently — if profiling shows repeated work, the measurement to restore
is in [CHANGELOG.md](CHANGELOG.md).

## §5 (results store) — the decision that shapes Journey 3

`reports.report_json` is written once and never updated. `findings.status` is exactly what review
changes. Those two facts are in tension, and the resolution is that **reads return a join**:

| endpoint | returns |
|---|---|
| `GET /reports/{id}` | the frozen report re-hydrated with each finding's *current* status and review trail |
| `GET /reports/{id}/filed` | the same report exactly as the engine produced it |

A regulator asking what the system concluded in June must get an answer later human review cannot
have edited; an analyst asking where the work stands must get the current state. Reading
`report_json` alone would show every finding as `pending_review` for ever, however much review had
happened — which is the bug the join exists to prevent. Reviews are append-only, because a status
column answers "where is this now" and nothing about how it got there.

## Deployment

`docker compose up -d api ui` for the engine and the cockpit; `docker compose up -d` adds the
Langfuse stack. Six services for Langfuse rather than two — v3 split storage across Postgres,
ClickHouse, MinIO and Redis, and the SDK here speaks the OpenTelemetry endpoint only v3+ serves.
`api` and `ui` refuse to start without `LLM_API_KEY` and `API_AUTH_TOKEN`; every `LANGFUSE_*`
variable has a default, because tracing being unconfigured must mean *off* rather than *broken*.

The compose file is configuration-validated and has **not been run**: it needs a Docker daemon that
was not available on the machine this was built on.
