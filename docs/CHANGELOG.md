# Changelog

Implementation log for FinGuard Orchestrator, 2026-08-13 → 2026-09-08. 17 commits, four phases.

This is a record of **what changed and why**, and the "why" is usually a measurement or a defect
rather than a preference. Where a decision reversed an earlier one, both are kept — a changelog
that only shows the winning branch hides the reason the winner won.

---

## Phase 4 — Cockpit, service, packaging, cache

### `3f9a5a3` · 2026-09-08 · Retrieval cache (§9.3)
Redis cache for retrieved clauses. **Audit node 44.0s → 0.0s warm**, 7/7 hits, context
byte-identical. Exact-key matching for the ten fixed templates, semantic (≥0.95) for the command
bar.

*Reversal:* §9.3 had been **deliberately skipped** in Phase 3 as an architecture demonstration
rather than an economy. Two Phase-4 changes made the case: the cockpit turned those seconds into
something a person watches, and the command bar gave the pipeline its first free-text query.

**Reports are deliberately not cached.** A June narrative names 3 real accounts and 11 amounts, so
reusing one across batches would put wrong identifiers into a filing. The blueprint's single
similarity threshold is split into *semantic for clauses, never for findings*.

Defects found while building it:
- stats were **overwritten, not accumulated** across the refinement pass — a run serving 7/8 from
  cache reported `0/1 (cold)`
- the cache intercepts *before* `nodes.retrieve`, so seven graph tests silently used real cached
  clauses on a machine with Redis running; fixed with an autouse `cache_off` fixture
- re-learning an unreachable Redis per call burned the connect timeout each time — **suite 30s →
  96s**; the verdict is now memoised per process
- "seconds saved" derived from the current run reported **0.0s on a fully warm run**, at the moment
  it saved most; elapsed time is now stored *in* the entry

**Measured and left alone:** at 0.95, MiniLM scores genuine paraphrases ~0.65. Only near-identical
rewordings match, so the exact path delivers the entire win. The threshold stays at the
blueprint's value rather than being tuned toward hits on a path that is not where the time goes.

### `7effac5` · 2026-09-05 · Dockerfile and containerization
Multi-stage build, **3.13 GB**, full audit verified inside the container. An earlier estimate of
~1.2 GB was wrong by 2.6× — the Linux CPU torch wheel is 656 MB, not the ~200 MB inferred from
macOS. `.dockerignore` takes the build context from 1.7 GB to 2.4 MB.

Three defects that only the first real build could find, because every local run has more
installed than a container does:

1. **`flashrank` was a dev dependency** but `nodes.py` imports it at module scope with
   `USE_RERANKER` on. Any `--no-dev` install died at import. Moved to runtime deps.
2. **The CPU-torch swap was silently undone** — it ran before the final `uv sync`, which restored
   the locked CUDA build. And even ordered correctly it did nothing, because the CPU and CUDA
   wheels share a version number, so `uv pip install` saw the requirement satisfied.
3. **The embedding cache could not be written** — hard-coded to `PROJECT_ROOT/data/…`, which the
   non-root container user cannot create. Now `EMBEDDING_CACHE_DIR`.

Also corrected: the ChromaDB mount must **not** be `:ro`. SQLite opens a journal even to read, and
`/health` correctly refused to serve.

### `61c7506` · 2026-09-05 · Streamlit cockpit and FastAPI service
§6's five sections, plus `POST /audit` returning 202 with an id.

Two seams first: `stream_batch()` (LangGraph streams *deltas*, so the generator assembles state and
yields `("__final__", state)` last) and `store.by_id()` — the drawer must show the clause the model
*actually saw*, so re-searching would defeat the audit trail.

**The command bar resolved a real conflict.** §6.2 wants free text; Decision 3 removed free-text
queries because a template ranks the correct clause 5th where a narrative ranks it 315th. Resolved
by asking the typed question *alongside* the templates with reserved seats. It earns its place —
on June it took the run from 2 critic passes at 0.50 to **1 pass at 0.75, $0.1364 → $0.0820**.

Also fixed here: a live run hit gpt-4o's **16,384-token output ceiling** in `generate` and lost the
whole run after four paid calls had succeeded. `GENERATE_MAX_TOKENS = 4096` plus a Python
`fallback_report()` that never asserts High and says it was a fallback.

**Finding:** the slowest node is the free one — `audit` spends ~22s embedding and reranking locally
while the three paid calls together take ~24s.

---

## Phase 3 — Observability, evaluation, optimization

### `284ef1f` · 2026-09-04 · DeepEval suite (§8)
`tests/eval_suite.py`, marker-gated so `pytest tests/` stays free and key-less. **Faithfulness
1.000 on every batch; Context Precision the weak metric** (0.547 on June).

Two departures from the blueprint: the gold set is generated from `ledger_labels.csv` rather than
hand-written — §8.2's example cites `FINRA Rule 3310(a)`, which cannot be grounded because FINRA
publishes no rule PDFs — and capture is separated from scoring, so a prompt change is judged
against the same frozen evidence.

Added `test_a_clean_batch_is_not_reported_as_a_finding`, which no LLM judge can make: each report
is individually plausible and only the answer key knows better. Written as a **failing test**
rather than a paragraph.

Three scoring runs were spent reading `tenacity.RetryError` as rate limiting before unwrapping it
to `insufficient_quota` — an exhausted account. The suite now skips on that rather than recording a
quality regression that never happened.

### `b22d0f6` · 2026-09-04 · FlashRank reranking (§9.4)
Adopted: hit@1 45.2% → 55.6%, hit@4 65.2% → 72.9%, hit@15 unchanged (it reorders, it cannot add).

**The blueprint's 15→4 prune was measured and rejected.** Tracking every clause the live reports
cited, the worst reaches rank 17 even after reranking — a 4-clause context would discard a clause a
real report grounded a finding on. `MAX_CONTEXT_CLAUSES` stays at 24; the win taken is ordering
quality, not token savings.

Per-query reranking beat a joined query: joined drops cited clauses from rank 3→7 and 4→12.

Also caught: the suite had acquired a **network dependency**, passing only because FlashRank's
model was cached in `/tmp`.

### `8fdb9ab` · 2026-09-04 · Cost instrumentation
Per-node tokens and dollars on every run. Token counts cannot be read off the response —
`with_structured_output` discards the `AIMessage` — so a callback watches the raw generation and
attributes spend by the `node:` tag tracing already stamps.

Measured: **$0.047–$0.178 per batch, $0.0000 on the free path.** The spread is not batch size (all
220 wires) but whether the critic loops: 3 calls vs 5.

`@dataclass(eq=False)` — a plain `@dataclass` nulls `__hash__` and LangChain merges callbacks
through `set(handlers)`; that crashed a live run mid-flight, after money was spent.

**§9.2 needed no code.** `route_after_detect` already *is* the hierarchical cost router. The
blueprint's trigger ("contains a cross-border wire") would escalate every batch — a fully domestic
220-wire batch has probability 1.5 × 10⁻¹⁰.

### `255672d`, `e121780` · 2026-09-04 · LangSmith tracing (§7)
Run-tagging and per-call metadata.

**The blueprint's §7.2 example cannot work as written** — it reads `batch_wire_count` from state at
`invoke()` time, before `parse` has run, so it is always 0. Metadata is therefore split: identity
at run level, batch-derived numbers per model call. That split is what makes §7.2's own question —
*which context caused the critic to loop* — answerable.

Also found: the tracing variables were reaching the process **by accident**, through an import
chain that exists for another reason. `graph.py` now loads `.env` explicitly.

**Correction recorded:** tracing uploads *full node inputs and outputs*, not just prompts — 137 KB
per run including all 220 parsed wires with names and account numbers, most of which no model sees.
An earlier README claim of "prompts leave the machine" understated it.

---

## Phase 2 — The LangGraph agent

### `20b25f8` · 2026-09-03 · Retrieval merge, fallback route, risk cap
Three defects and one missing route, all found by questioning the shipped code rather than by a
test.

**The merge had two defects.** `setdefault` stored whichever distance a clause was *first* found
with — 7 of 93 clauses held a worse score than they had earned. And distances are not comparable
across queries (June's seven best hits span 0.3433–0.4827), so pooling them ranked *how easy the
question was* above *how good the answer is*. Replaced with **reciprocal rank fusion**: the cited
clause moved from rank 20 to 10. Round-robin (23) and min-max normalisation (42) were both tested
and are worse than the bug.

*Self-inflicted, then fixed:* RRF accumulates, so after seven queries no single refinement list can
compete — the loop admitted **1 new clause of 15**, nearly inert. `REFINEMENT_RESERVE = 5` seats
the critic's query rather than ranking it.

**The third fallback route was missing.** `audit → draft` was unconditional, so an empty retrieval
would reach gpt-4o with a blank regulations block. Now raises.

**The risk rating was anti-correlated with truth.** Live runs put clean May at *High, file a SAR*
and July (23 laundering wires) at *Low*. `HIGH_RISK_CONFIDENCE = 0.9` caps High below the critic's
top band. This is a mitigation, not a fix — the defect remains open.

Also: `parse_amount`'s docstring was corrected. It claimed the `"." in raw` branch prevents a
misparse; it does not — every string it rejects is already caught by the comma count or by
`Decimal`. It earns its place only by naming the format in the refusal.

### `cf40dee` · 2026-08-29 · The agent
Seven nodes, two conditional edges, one cycle. Parser verified **880/880**; detectors at **100%
recall, 32% precision**.

Design decisions taken here and unchanged since: geometry-only candidates (Python may observe, only
a clause may conclude); obligation-shaped query templates; the Python citation veto; evidence
fields recomputed after the model returns them.

Calibration found by running it: at 93 clauses in context the model cited **nothing** and the
critic scored 0.00 — hence `MAX_CONTEXT_CLAUSES = 24`, chosen because `14.2.3.Guidance.1.` sat at
rank 20. The draft prompt then over-steered into refusal and needed rules 3-5 added.

---

## Phase 1 — Ingestion and grounding

### `bc6668e`, `e931dc5` · 2026-08-17/18 · ChromaDB and the ingestion panel
12,273 chunks, 46 documents, one collection, tiered by AML relevance.

**All 40 ObliQA documents are indexed, not just the AML-bearing ones.** 2.9% of passages are
AML-bearing; the rest are the distractor set that makes Context Precision measurable at all.

**No transaction vectors.** Laundering and clean wires separate by +0.029 cosine — noise, since ~55
of ~65 tokens are boilerplate. Parsed wires belong in a table.

### `b8f44f8` · 2026-08-16 · Semantic chunking
Cosine-boundary chunking. Retrieval against the 2,786-question gold set: **hit@1 45.2%, hit@15
79.2%, MRR 0.552**. OpenAI embeddings measured better (+3.6 hit@15); adoption deferred.

**The finding that shaped Phase 2:** phrasing decides retrieval. Same facts, rank of the correct
clause out of 12,273 — raw JSON **11,268**, narrative **315**, obligation-shaped question **5**.

### `c496b79` · 2026-08-15 · Dataset acquisition and SWIFT log generation
SAML-D → synthetic MT103 logs, with `ledger_labels.csv` as the answer key.

**FINRA publishes no machine-readable rule PDFs** — established here, and it invalidates the
blueprint's evaluation example later.

Defect found during Phase 2 and fixed retroactively: `select_cases` capped *total* context but
nothing capped it **per anchor**, so August was 85% one account (184 wires, 181 on one day) and
July 52%. `CONTEXT_PER_ANCHOR = 12` brought the busiest account to 7-9%. Two of three batches were
unusable until then.

### `1220d6a`, `585d619`, `ee2c619` · 2026-08-13/14 · Skeleton
uv workspace, Python 3.11 pinned. **`ragas` dropped**: 0.4.3 imports
`langchain_community.chat_models.vertexai`, removed in langchain-community 0.4.x. DeepEval covers
the three blueprint metrics.

---

## Recurring lessons

**Live runs find what tests cannot.** The 16,384-token ceiling, the anti-correlated risk rating,
the unhashable callback, the fabricated citations, and every container defect were found by
running the thing, not by the 233 tests.

**A dependency present locally is invisible until something installs less than you do.** FlashRank's
`/tmp` cache, the dev-only `flashrank`, and Redis's presence during a test run all hid the same
class of bug.

**The blueprint was right about goals and wrong about several mechanisms** — the §7.2 metadata
example, the §9.2 router predicate, the §9.4 15→4 prune, the §8.2 FINRA gold set, and §9.3's single
cache threshold. Each deviation is recorded with the measurement that forced it.
