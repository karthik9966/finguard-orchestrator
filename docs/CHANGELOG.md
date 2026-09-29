# Changelog

Implementation log for FinGuard Orchestrator.

- **v1 build** — 2026-08-13 → 2026-09-08, 17 commits, four phases. Below, from "Phase 4 — Cockpit".
- **v2 migration** — 2026-09-27 → 2026-09-29, 16 commits, ten phases. Below the graph section.
- **v2 graph engine** — 2026-09-29, branch `migrate/v2-graph`, against the `_v2.docx` design set.
  Immediately below this line.

This is a record of **what changed and why**, and the "why" is usually a measurement or a defect
rather than a preference. Where a decision reversed an earlier one, both are kept — a changelog
that only shows the winning branch hides the reason the winner won. The v1 section is left intact
for that reason: much of what v2 replaced, it replaced for a measured cause, and the measurement
lives in the entry that introduced the thing.

---

# v2 graph — nine patterns over one money-flow graph

Against `FinGuard_PRD_v2`, `HLD_v2`, `LLD_v2` and `Eval_Design_v2`, whose change is one thing: the
transaction graph becomes the detection substrate, and coverage grows from five detectors to nine
patterns. The plan followed the documents; the data corrected them in six places, all recorded in
[DESIGN.md](DESIGN.md) under "Deviations".

## P2 · `0daa265` — keep the whole payment type

`swift_parser` kept the first word of `:72:/INS/<type>`, so `CASH DEPOSIT` and `CASH WITHDRAWAL` both
became `CASH`. LLD v2 designs deposit-send around SAML-D having no deposit/withdrawal split; SAML-D has
225,206 deposits and 300,477 withdrawals. **The document was describing our parser.** Fixed at the
source; `TransactionRecord.payment_kind` normalises in one place.

## P3 · `2cd478b` — the GraphEngine, a `graph_build` node, and a redaction gap

`BatchGraph` gains bounded structural queries — `counterparties`, `hub_nodes`, `multi_hop_layers`,
`bipartite_blocks`, `subgraph` — and `GraphBuildNode` builds it once, ahead of detection. A 14,280-edge
near-complete graph runs every query in under a second.

*Found on the way:* a cycle's `route` and a scatter-gather's `sink`/`intermediaries` had gone into the
grounding prompt as **raw account numbers** since v1. Redaction dispatches on field name and those names
were never listed. The subgraph's `source`/`target`/`nodes` would have widened the hole; all covered now.

## P1 + P4 · `2620f57` — nine patterns, four detectors, and a golden set that is held out

Landed together because the contract cannot be green without the detectors: `precedence_order` must
name every pattern, and `detect_all` refuses a precedence entry no detector registers.

Detector findings, each from data rather than from the documents:

- **Deposit-send's discriminator is the amount, not the timing.** 66% of clean depositors send
  *something* within three days; a send matching the deposit within 5%, about 4%.
- **Structuring groups by beneficiary too.** Every SAML-D Structuring cluster is many parties paying
  one account; originator grouping could see none, and fan-in had been covering for it.
- **Layered fan leaves are windowed per branch.** Windowing the whole structure let one busy
  branch's month of unrelated traffic push a real funnel out of range.
- **Gather-scatter's conservation band is wide** (0.5–2.0) because SAML-D hubs run 0.80–4.06.

Corpus findings — the larger half of this commit:

- **v1's held-out corpus was not held out.** Both profiles took the largest clusters of the same
  months; all 84 of v1's dev-planted rows were planted in eval too, and 3 of 75 golden instances were
  tuning clusters (verified by regenerating v1's corpora from `a374ac5`). Clusters are now partitioned
  by a hash of the anchor account: **0 rows** shared.
- **Scatter-Gather was planted as its scatter leg.** Anchored selection took only the edges touching
  one account. A `COMPONENT` shape plants whole structures.
- **Month boundaries cut structures in half.** Most planted "Gather-Scatter" instances were only their
  scatter side. Structural clusters are planted only when whole inside a month.
- **Two patterns cannot reach fifteen.** Whole + held out leaves scatter_gather 6 and gather_scatter
  12. *Decision (project owner, 2026-09-29):* take every instance, record the shortfall on each record,
  report the denominator — rather than plant half-shapes or tune on golden clusters.
- **Option 1** as a data policy: eval takes every threshold-aligned structuring and deposit-send
  cluster (~21 structuring exist in all of SAML-D), dev takes none. Smurfing leaves scope.

*Reversal:* the first v2 tuning pass began sweeping on three dev instances per pattern — too few to
tell a threshold from noise. Dev was widened to seven months × two clusters (93 instances) and every v2
number was swept there instead.

Measured on eval: pattern-level recall **0.827 → 0.959**; structuring **5/15 → 15/15**; alert volume
**19.1% → 9.2%**; named recall 0.886; scatter_gather 5/6 fails the new per-pattern gate; context
precision hit@1 **0.60 → 0.33**, all of the drop in the four new queries, characterised not tuned.

## P5 · `9d68e21` — the matched structure, three ways

`evidence.describe()` for the report summary (still no model call), `edge_lines()` for the grounding
prompt (redacted, capped at 30), `to_dot()` for the cockpit's `st.graphviz_chart`. `SCHEMA_VERSION`
2.0 → 2.1, additive.

## P6 · `616d153` — graph-engine correctness as a PR-gate step

It already ran in the full suite; Eval Design v2 names it a hard gate, so it gets its own line.

## P7 — docs as built

[HLD.md](HLD.md) and [LLD.md](LLD.md) rewritten for v2 from their v1 versions, keeping what still holds;
DESIGN, CONSTANTS, TEST_DESIGN, TEST_RESULTS and the datasets README updated. TEST_RESULTS marks every
live-tier number as v1's: **the live tier has not been run on v2.**

---

# v2 — the design migration

Ten phases against a hand-written design set (`AML- PRD.docx`, `FinGuard_HLD.docx`,
`FinGuard_LLD.docx`, `FinGuard_Eval_Design.docx`) that superseded the v1 blueprint. The governing
invariant: **the suite is green and the system runs at the end of every phase.** Phases 2–4 were
purely additive and every deletion was deferred to Phase 5, which is what made that invariant
keepable while two sets of contracts coexisted.

As-built documentation: [DESIGN.md](DESIGN.md) · [HLD.md](HLD.md) · [LLD.md](LLD.md) ·
[CONSTANTS.md](CONSTANTS.md) · [TEST_DESIGN.md](TEST_DESIGN.md) · [TEST_RESULTS.md](TEST_RESULTS.md).

## Phase 0 · `23fa5db` — config spine, contracts, redaction, PR gate

`config.yaml` + `Settings`/`Config`: numbers never in code, secrets never in the file. Every Phase-0
contract written before its consumer. `redaction.py` with one function guarding both exits.

**The PR gate lands here rather than in Phase 8, where the plan put it.** The migration's promise is
a green suite at the end of every phase, and a promise whose only enforcement arrives eight phases
later is not enforced — it is remembered.

## Phase 1 · `be22a5f` `dc1b4d4` `28ca4ce` — the citable US corpus

731 chunks, 20 US sources, tier and authority classified separately because tier says what kind of
document the text came from and authority says whether it **binds**.

Four chunking defects, each found on real regulation: **31 CFR 1010.311 was silently absent** (a
single unlettered paragraph, so the CTR obligation and the $10,000 threshold were not indexed at
all); one bullet swallowed 50,000 characters while reporting 86% coverage; 14 Appendix F indicators
were filed under a page footer; 13 more under wrapped prose reading as a heading.

*Reversal:* the `--rerank` flag that `test_rerank.py`'s skip message told people to run **had never
existed**. Built, and it reproduces 45.2% → 55.6% hit@1 exactly.

ObliQA fenced into its own collection: ADGM law is out of scope as *citable* law, but its 2,786
labelled questions are the only retrieval ground truth this project has.

## Phase 2 · `b4d2b72` — US ledgers, `TransactionRecord`, batch ingestion

Re-domiciled the synthetic ledgers as a US institution. **Amounts are relabelled, not converted** —
converting at an FX rate would lift a cluster sitting just under 10,000 straight over the threshold
and stop it being structuring at all.

*Reversal:* `PRIORITY_TYPOLOGIES` named the wrong things. The inherited list seeded Deposit-Send,
Gather-Scatter and Layered_Fan_In — all excluded by PRD §2 — while plain `Fan_In` and `Fan_Out`, two
of the five detectors, were absent entirely. Regenerating with the old list produced ledgers in which
three of five detectors had nothing to find.

Suite 40s → 120s from six tests each parsing the 10k batch; a session-scoped cache took it to 62s.

## Phase 3 · `d660dfe` — five windowed typologies, reconciled

`structuring · fan_in · fan_out · cycle · scatter_gather`, replacing four geometric primitives.

**`window_days` 7 → 14 on measurement**: 85% recall → 96% → **99%**, for one extra candidate. SAML-D
plants its clusters across 9–10 days, so 7 could not hold one.

**`detection_confidence` is anti-correlated with planted wires** (incidental 0.562, planted 0.395).
Recorded because it settles a design question: nothing may gate on it.

*Defect:* the cycle DFS rejected its own closing edge. The `visited` set, ported from `find_paths`,
blocked returning to the origin — so a directed cycle could never close.

## Phase 4 · `ffefe9c` — tier-aware retrieval

Tier 1 by curated id, Tier 2 by search + rerank. Both failure modes non-fatal, because raising on a
config gap would let it fail a whole run.

## Phase 5 · `d808ef9` — per-candidate reasoning core, and the deletions

The two halves meet. **The self-check loop becomes per candidate**: one thin finding no longer sends
every candidate back through retrieval, and one fabricated citation no longer vetoes the run.

The High-risk bar moved out of the critic into report generation, and report generation lost its
model call entirely — the pre-migration version had a model write the filing and then repaired three
of its fields, and its ratings were *anti-correlated with the truth* (clean May came back High
recommending a SAR; July, with 23 laundering patterns, came back Low).

Measured: clean control **$0.0000**, June dev batch $0.0908–$0.1242 → **$0.018–$0.025 per candidate**.

Deleted: `utils/detectors.py` and its 26 tests, the old six-node graph and its 55 tests, the reverse
`Wire` adapter, `fallback_report`, the batch-level RRF pool, and every module-level config shim —
that last now *enforced*, because `NAME = get_config().x.y` freezes at import and makes the file look
live while being dead.

## Phase 6 · `4ef3e05` `06d5cc6` `ff8bef8` — store, API, cockpit

**The decision this phase turns on:** `reports.report_json` is immutable and `findings.status` is not,
so reads return a *join*. A regulator asking what the system concluded in June must get an answer
later review cannot have edited; an analyst must get the current state. Reading `report_json` alone
would show every finding as `pending_review` for ever.

Bearer auth where **unconfigured is closed, not open** (503 + a loud startup log). Job state moved
from a process dict into the database, because the dict could not answer `GET /audits/{id}` after a
restart. Batch-hash dedup so a client's impatient retry does not pay for a second audit.

The cockpit became a client of the same API a bank would call — removing two execution paths, gaining
the single worker's serialisation, and making it impossible for a Streamlit rerun to bill money.

*Two defects:* an `asyncio.Queue` at module scope bound itself to the first event loop it saw, so
every job after the first app shutdown sat at `running` for ever (suite 338s → 6s once fixed). And
`finding_id` was not unique across reports — `candidate_id` is deliberately stable across runs, so
`f-{candidate_id}` collided on the second audit of a month, which is what `?force=true` does.

*Deleted:* `src/ui/ingestion_panel.py`, which read the pre-migration `regulations` collection and
reported **12,273 ADGM chunks** as the active corpus while the engine cited 731 US ones.

## Phase 7 · `0d5be5d` — Langfuse, with redaction on the way out

Closes a measured defect: the pre-migration system uploaded **137,261 characters per run** to a hosted
project, including every wire's counterparty names and account numbers, of which ~21 ever reached a
model. `mask = lambda data: trim(redact(data))`, set on the client so it covers every span the SDK
emits. **~137 KB → 9.2 KB, zero account numbers, zero names**, captured from the OpenTelemetry
exporter rather than estimated.

*Two redaction fixes the probe forced:* an exported `Decimal` came out as the literal `"<Decimal>"`,
useless in a trace about sub-threshold structuring; and `"2023-06-14"` read as an 8-digit run with
hyphen separators and was masked — so a finding's explanation said "three transfers on [REDACTED]".

Six services for Langfuse rather than the plan's two: v3 split storage three ways and the SDK speaks
an endpoint only v3+ serves. Pinning an older server the client cannot talk to would have been the
smaller diff and the wrong answer.

## Phase 8 · `7dfff09` `7ba1ccc` `990998a` — golden datasets, runners, the first honest numbers

Six corpora, 75 labelled patterns at 15 per typology, in a corpus **deliberately not** the one the
detectors were tuned on. *Caught in the act:* the first eval build ran `--append` into the dev corpus,
which would have silently turned the recorded 99% recall into a number measured on different data.

The rules-only baseline is what makes recall mean anything: it scores **0.907 by alerting on 77.6% of
the batch**, against our 0.827 on 19.1%.

**Three metrics fail and are left failing** with named causes — see [TEST_RESULTS.md](TEST_RESULTS.md).
Structuring recall is 5/15 because the planted amounts (median $2,323) never reach the $8,000 band; it
is a mismatch of premises, since §5324 structuring means amounts *chosen* to evade a threshold.

*Defects the harness found:* **context precision was 0.22** because the indicator search used
obligation-shaped queries — Phase 1's measurement had not transferred, because obligations are now
fetched by id and the only corpus still searched is written as descriptions of behaviour (0.22 → 0.60).
**Faithfulness was 0.9811** — two findings whose narrative named a clause their own citation list
omitted; nothing fabricated, but a report that cites what it does not list is inconsistent (now 1.00
over 110 checks). And **a non-UTF-8 batch lost all 500 messages** to a `UnicodeDecodeError`.

*And three defects in my own datasets*, corrected: four of ten query records specified attributes no
detector emits; one labelled answer was simply wrong, the retriever's first result being definitionally
better; and the set was ten records where only six queries are distinguishable.

*Correction · `ebfac70`:* Phase 8c overwrote Phase 0's `pr.yml` without reading it, losing the offline
environment guarantee, an explicit credential-leak check, a pinned interpreter and the skip report.
Restored verbatim and extended.

## Phase 9 — docs as built

This section, plus [DESIGN.md](DESIGN.md), [HLD.md](HLD.md), [LLD.md](LLD.md),
[CONSTANTS.md](CONSTANTS.md), [TEST_DESIGN.md](TEST_DESIGN.md) and
[TEST_RESULTS.md](TEST_RESULTS.md); the four documented deviations from the design set; and every
number in `config.yaml` beside the measurement that chose it.

---

# v1 — the original build

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
