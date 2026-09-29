# Constants index

Every tunable number in `config.yaml`, beside the measurement that chose it.

The rule this table enforces is LLD §8's: **numbers never in code, secrets never in the file.** The
rule it *documents* is stricter — a number here without evidence beside it is a number nobody can
change safely, because the next person cannot tell a measured threshold from a guess that survived.
Where a value was inherited from the pre-migration system and then measured, both are shown: a table
that only records the winning value hides the reason it won.

"Measured on" matters, and there are two dev corpora in this file's history:

- **v1 rows** were measured on the v1 dev corpus (`data/processed/ledger/`, four 500-message
  batches). They were not re-tuned for v2 and are kept with their original evidence.
- **v2 rows** were measured on the v2 dev corpus: seven 1,500-message months drawn from the *dev half*
  of SAML-D's clusters (see `partition_of` in `pdf_generator.py`), 93 planted instances of all nine
  patterns, structural clusters only when whole inside a month.

The golden corpus (`data/processed/eval_ledger/`) is held out and is what `eval/` reports against —
see [TEST_RESULTS.md](TEST_RESULTS.md). In v1 it was less held out than claimed: every dev cluster was
also planted in it, and 3 of 75 golden instances were tuning clusters. v2's partition makes the overlap
zero by construction.

In the v2 tables, **named** recall is a candidate *of that pattern* covering the instance; **any** is
any candidate covering it. The 10,000-message batch stayed at ~308 candidates and the clean control at
**0** through every v2 change below.

---

## Retrieval — LLD §2.4

| key | value | evidence |
|---|---|---|
| `k_indicators` | 15 | The blueprint figure, and the one every recorded hit@k number was measured at (hit@15 = 79.2% on ObliQA's 2,786 labelled questions). Changing it invalidates the comparability of the retrieval benchmark. |
| `rerank_top_n` | 5 | The pre-migration system carried 24 clauses because **one** pool was shared by every candidate in the batch. Per-candidate retrieval does not need that, and a cross-encoder's precision is concentrated at the top. Provisional: Phase 8's context-precision measurement is what should set it, and at 5 the correct clause is inside the window for 4 of 5 scored queries (hit@3 = 0.80). |
| `multi_query_rrf` | `false` | LLD §2.4 specifies one query per candidate. The pre-migration system issued 2–4 and fused them with reciprocal rank fusion, which measurably beat every alternative merge — the cited clause moved from rank 20 to 10 — but that win was on *semantic discovery of obligations*, which the curated Tier-1 map replaces. Kept behind the flag rather than deleted. |
| `rrf_k` | 60 | The constant from the original RRF paper. Read only when `multi_query_rrf` is true. |

## Detection — LLD §2.6

| key | value | evidence |
|---|---|---|
| `window_days` | 14 | Was 7, inherited from the path detector's `MAX_PATH_GAP_DAYS`, with a note that applying a window to fan-in and fan-out was "a behaviour change to watch in the recall harness". The harness watched, and 7 was too tight — SAML-D plants its clusters across 9–10 days. Measured across the four dev batches: **7d → 85% recall / 19 candidates · 10d → 96% / 20 · 14d → 99% / 20.** Fourteen points of recall for one extra candidate, which also established that the *window* was the binding constraint and the amount band was not (widening the band from 0.2 to 0.7 moved recall not at all at 7 days). |
| `structuring.thresholds` | `[10000, 3000]` | Not tuned — statutory. $10,000 is the CTR filing threshold (31 CFR 1010.311); $3,000 is the funds-transfer recordkeeping threshold (31 CFR 1010.410(e)). |
| `structuring.band_fraction` | 0.2 | **A recorded deviation from LLD §8**, which names a single absolute `band`. One absolute value cannot serve both thresholds: an absolute 2,000 makes the $3,000 band `[1000, 3000)`, which catches most ordinary payments. A fraction gives `[8000, 10000)` and `[2400, 3000)`. |
| `structuring.min_count` | 3 | Fewer than three sub-threshold transfers is not a pattern. |
| `structuring.min_aggregate_multiple` | 1.0 | The group must total at least the threshold, or there was nothing to evade. |
| *(structuring sides)* | originator, then beneficiary | v2. Not a number, but a behaviour a reader would otherwise assume: every SAML-D Structuring cluster is receiver-anchored, so grouping by originator alone saw none of them. Originator runs are claimed first; `side` records which. |
| `fan_in.min_sources` | 4 | Was 3, inherited from `MIN_CLUSTER_WIRES` on the grounds that 4 "lost Layered_Fan_Out entirely" — a typology PRD §2 now excludes, so that reason expired. Measured across the dev batches and the 10,000-message batch: **3 → 99% recall / 20 dev candidates / 729 on the 10k batch · 4 → 99% / 19 / 304 · 5 → 96% / 17 / 119 · 6 → 96% / 17 / 41.** Four keeps recall whole and more than halves volume at scale; going further buys a much shorter list for three points of recall, and in AML a miss is a regulatory failure while a false alarm costs an analyst minutes. |
| `fan_out.min_targets` | 4 | Same curve, same reasoning. |
| `cycle.min_hops` | 3 | Two accounts paying each other is a relationship, not a ring. |
| `cycle.max_length` | 25 | Bounds the DFS. Without it the walk on 10,000 messages does not finish. |
| `cycle.path_overlap` | 0.6 | Above this share of overlapping transaction references, a chain is a retelling of one already reported. Without it a single ring produced **11** near-duplicate chains; it found **2** after. Plain subset dedup misses this because branching chains are not subsets of each other. |
| `cycle.min_retained_fraction` | 0.5 | Funds must return roughly intact for a ring to read as laundering rather than coincidence. SAML-D's cycles decay 10–20% per hop. |
| `scatter_gather.min_fan` | 3 | One → many → one needs at least three intermediaries to be a shape rather than a relay. |
| `gather_scatter.min_in` / `min_out` | 3 / 3 | v2. Whole planted hubs had 3–6 counterparties a side. **4/4 → 1/5 named · 3/3 → 1/5 · 3/3 with the wider band below → 3/5 named, 5/5 any.** |
| `gather_scatter.min_conservation` / `max_conservation` | 0.5 / 2.0 | v2. Out/in on whole planted hubs ranged **0.80 to 4.06** — SAML-D does not model conservation tightly — so the band only rules out a hub that plainly kept the money or paid from elsewhere. 0.7–1.3 was the shape-derived starting value and held recall at 1/5. |
| `gather_scatter.window_days` | 21 | v2. SAML-D hubs run 10–17 days end to end; the shared 14-day window cannot hold one. |
| `deposit_send.window_hours` | 72 | v2. Over SAML-D's labelled pairs the send follows the deposit in a median 1.8 days (75th pct 2.6). 168 h moved dev recall not at all (11/14 either way). |
| `deposit_send.amount_tolerance` | 0.05 | v2. **The discriminator** — timing alone fires on 66% of clean depositors within 3 days. First-following-send over all SAML-D: **1% → 26% of laundering deposits / 1.0% of clean depositors · 2% → 47% / 1.7% · 5% → 74% / 4.2% · 5% at 7 d → 74% / 8.8%.** 10% moved dev recall not at all. *Measured over the whole dataset, before the partition existed — the one v2 number whose evidence touched golden clusters; the dev sweep that confirmed it did not.* |
| `deposit_send.send_kinds` | cross_border, ach, wire, cheque | v2. What counts as the "send". Card payments are excluded: a debit-card purchase after a deposit is spending, not moving. |
| `deposit_send.min_pairs` | 1 | v2. SAML-D spreads a hub's ~6 pairs over ~250 days, so a month usually holds exactly one. |
| `layered_fan.min_branches` | 2 | v2. SAML-D funnels have ~4 intermediaries over a whole cluster, fewer within one month. **3 → 2/6 on the first v2 dev draft; 2 → 6/9 named on the whole-cluster dev corpus.** |
| `layered_fan.min_leaves_per_branch` | 2 | v2. A branch with one leaf is a relay, not a collector. |
| `layered_fan.min_total_leaves` | 4 | v2. **5 → 6/9 named · 4 → 7/9**, no added candidates on the 10k batch. |
| `layered_fan.window_days` | 28 | v2. Planted funnels span 14–23 days; applied per branch hand-off, not to the whole structure. |
| `bipartite.min_src` / `min_dst` | 2 / 3 | v2. SAML-D Bipartite is K(2,7); a stacked layer can be 2×3. **min_dst 4 → 13/14 named · 3 → 14/14.** |
| `bipartite.min_density` | 0.8 | v2. The largest SAML-D Bipartite clusters are complete K(2,7) blocks; 0.8 tolerates a missing payment in a small block. Not swept. |
| `bipartite.window_days` | 21 | v2. Plain blocks span ~13 days, stacked ~18. |
| `confidence_weights` | 0.4 / 0.2 / 0.2 / 0.2 | Tightness, member count, window compactness, aggregate ratio; they sum to 1.0. Tightness is measured on the threshold-band subset and never the whole group: one legitimate large wire moved a cluster's coefficient of variation by **43×**, which is the recorded weakness of the pre-migration scoring. **Note the measured limit of the whole score:** `detection_confidence` is *anti-correlated* with planted wires (incidental 0.562, planted 0.395), which is why nothing gates on it. |
| `precedence_order` | structuring, deposit_send, cycle, bipartite, layered_fan, gather_scatter, scatter_gather, fan_in, fan_out | Load-bearing rather than hygiene. `fan_out` fires on the first leg of every `scatter_gather`, and in v2 the fans fire inside every structure — a layered funnel's collectors, a gather-scatter's halves, a bipartite block's senders — so the more specific shape must claim the transactions first. Not confidence-ordered: half a pattern looks tighter than the whole of it. |

## Graph engine — LLD v2 §2.5

Guardrails, not semantics: how much of a dense graph one query may touch before it truncates (with a
log line) rather than blowing up.

| key | value | evidence |
|---|---|---|
| `max_traversal_depth` | 3 | No SAML-D layered or stacked cluster is deeper than 2 — all 138 measured, 113 at depth 2 and 25 at depth 1 — so 3 leaves one hop of headroom. |
| `max_frontier` | 200 | A funnel's layer is 4–10 accounts in SAML-D; a level of hundreds is a payment processor. |
| `max_block_senders` | 500 | The bipartite pair search is quadratic in it. |
| `max_block_side` | 50 | A receiver with more payers than this is a hub, and a hub links every sender to every other. A 120-account, 14,280-edge near-complete stress graph runs every query in under a second at these values. |

## Reasoning core — LLD §5

| key | value | evidence |
|---|---|---|
| `confidence_threshold` | 0.75 | A draft that is supported but may be thin. Below it the retrieval question is reformulated. |
| `max_loops` | 2 | The give-up point. Phase 1 measured that **17.2%** of ObliQA's gold questions have no correct clause in the top 15 at all, so a third attempt usually spends money on a clause that is not in the collection. |
| `high_risk_min_confidence` | 0.9 | High means *file a SAR*, so it needs the top band rather than merely enough score to stop the loop. Forced by measurement: across the four pre-migration batches the model's own rating was **anti-correlated with the truth** — clean May (0 laundering) came back High recommending a SAR, July (23 laundering) came back Low. Applied in `ReportGenerationNode`, never in the critic, whose prompt forbids it from changing a fact. |
| `schema_retries` | 3 | LLD §6 `SCHEMA_PARSE_FAILURE`. A model that returned prose where a schema was asked for usually returns the schema when shown the validation error. |
| `llm_max_attempts` | 3 | Tenacity-style attempts on 429/5xx/timeout. Passed to the client as `max_retries`, so a timeout is not re-prompted as a schema error. |
| `evidence_edges_in_prompt` | 30 | v2. Edges of the matched subgraph shown to the grounding model. The largest golden instance has 24 transactions (a stacked bipartite), so the cap bounds tokens on a pathological candidate without trimming a real one; the prompt counts what it left out. |
| `llm_timeout_seconds` | 60 | Explicit because the default OpenAI client has no ceiling on how long it will wait, and a node that hangs is indistinguishable from a node that is working. |
| *(no candidate cap)* | — | Deliberately absent. Measured on the June dev batch (500 messages, 5 candidates), twice: **16 calls / 37,684 tokens / $0.1242 → $0.0248 per candidate**, and **12 calls / 27,051 tokens / $0.0908 → $0.0182**. A range rather than a number because the loop is what varies: at temperature 0 the critic still scored the same drafts differently across runs, and each extra pass is two more calls. Confirmed on the golden corpus at **$0.0186–$0.0204 per candidate**. A cap by `detection_confidence` is ruled out on the evidence above. |

## Persistence — LLD §3.2, §6

| key | value | evidence |
|---|---|---|
| `write_attempts` | 3 | `RESULTS_STORE_WRITE_FAILURE` is retryable infrastructure, not a bad report. By step 8 the run is already paid for — five candidates of grounding and review — so discarding the result because the disk was busy for 300 ms would be the most expensive possible response. |
| `write_backoff_seconds` | 0.5 | Exponential from here. After the attempts are exhausted the report is **held in memory** and the error surfaced, so a caller can re-save rather than re-run. |

## Chunking — LLD §2.1

| key | value | evidence |
|---|---|---|
| `min_chars` | 200 | Below this a chunk is usually a stub that retrieves badly on its own. |
| `max_chars` | 2000 | Above this the reranker carries too much irrelevant text into the prompt. |
| `percentile` | 95.0 | The cosine-boundary threshold, computed per document. |
| `min_sentences_for_percentile` | 6 | Below this a percentile is not a statistic, it is noise. |
| `min_passage_chars` | 40 | Below this a passage is a bare heading, not a clause. |
| `context_prefix_below` | 200 | Short chunks get a `{title} — {clause}: ` prefix so they stay citable. |

## Ingestion — LLD §2.2

| key | value | evidence |
|---|---|---|
| `upsert_batch` | 1000 | Chroma write batching. |
| `llm_fallback_attempts` | 1 | LLD §6 `INGEST_ROW_MALFORMED` gives exactly one light-model attempt. **One, not two, on purpose:** a model that could not read a malformed message the first time will usually produce something *plausible* on the second, and a plausible account number in a filing is worse than a refusal. |

---

## What is deliberately *not* in this file

`config.yaml` also carries two curated structures rather than tunables:

- **`pattern_to_obligations`** — the Tier-1 map, 3–5 `(source_id, section_ref)` pairs per typology.
  The four v2 patterns rest on the shared set LLD v2 §5.2 specifies — § 1020.320(a) and (a)(2)(iii),
  plus § 1010.410(e) — and deposit-send adds § 1010.311, because its cash leg is what engages the CTR.
  Curated as pairs and never as chunk ids, because `chunk_id = hash(source_id, section_ref, version)`
  and a literal id goes stale on a re-chunk with no error. `tests/test_obligation_map.py` asserts
  every entry resolves; per LLD §6 an `OBLIGATION_MAP_MISS` disables grounding for a whole typology,
  so a wrong entry is silent and expensive.
- **`source_topics`** — 20 sources' `topic_tags`, inherited by every chunk cut from them. Curated
  rather than derived: the manifest's `note` was written as prose for a human, and tags inferred from
  it would be inconsistent in exactly the way a filter cannot tolerate. A tag that looks like a
  typology must be one — `config.py` rejects `faninn` at startup rather than letting it narrow a
  retrieval to nothing.

Secrets and endpoints live in the environment (`.env.example` is the template), never here. The
split is enforced by a test: no module may read a tunable at import time, because
`NAME = get_config().x.y` at module level freezes at import and makes the file look live while being
dead.
