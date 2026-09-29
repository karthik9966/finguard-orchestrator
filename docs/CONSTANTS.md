# Constants index

Every tunable number in `config.yaml`, beside the measurement that chose it.

The rule this table enforces is LLD §8's: **numbers never in code, secrets never in the file.** The
rule it *documents* is stricter — a number here without evidence beside it is a number nobody can
change safely, because the next person cannot tell a measured threshold from a guess that survived.
Where a value was inherited from the pre-migration system and then measured, both are shown: a table
that only records the winning value hides the reason it won.

"Measured on" matters. The detection and retrieval numbers were chosen against the **dev corpus**
(`data/processed/ledger/`, four batches). The golden corpus (`data/processed/eval_ledger/`, eleven
batches) is held out and is what `eval/` reports against — see [TEST_RESULTS.md](TEST_RESULTS.md),
where the held-out numbers are lower than the tuning ones and that gap is the point.

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
| `fan_in.min_sources` | 4 | Was 3, inherited from `MIN_CLUSTER_WIRES` on the grounds that 4 "lost Layered_Fan_Out entirely" — a typology PRD §2 now excludes, so that reason expired. Measured across the dev batches and the 10,000-message batch: **3 → 99% recall / 20 dev candidates / 729 on the 10k batch · 4 → 99% / 19 / 304 · 5 → 96% / 17 / 119 · 6 → 96% / 17 / 41.** Four keeps recall whole and more than halves volume at scale; going further buys a much shorter list for three points of recall, and in AML a miss is a regulatory failure while a false alarm costs an analyst minutes. |
| `fan_out.min_targets` | 4 | Same curve, same reasoning. |
| `cycle.min_hops` | 3 | Two accounts paying each other is a relationship, not a ring. |
| `cycle.max_length` | 25 | Bounds the DFS. Without it the walk on 10,000 messages does not finish. |
| `cycle.path_overlap` | 0.6 | Above this share of overlapping transaction references, a chain is a retelling of one already reported. Without it a single ring produced **11** near-duplicate chains; it found **2** after. Plain subset dedup misses this because branching chains are not subsets of each other. |
| `cycle.min_retained_fraction` | 0.5 | Funds must return roughly intact for a ring to read as laundering rather than coincidence. SAML-D's cycles decay 10–20% per hop. |
| `scatter_gather.min_fan` | 3 | One → many → one needs at least three intermediaries to be a shape rather than a relay. |
| `confidence_weights` | 0.4 / 0.2 / 0.2 / 0.2 | Tightness, member count, window compactness, aggregate ratio; they sum to 1.0. Tightness is measured on the threshold-band subset and never the whole group: one legitimate large wire moved a cluster's coefficient of variation by **43×**, which is the recorded weakness of the pre-migration scoring. **Note the measured limit of the whole score:** `detection_confidence` is *anti-correlated* with planted wires (incidental 0.562, planted 0.395), which is why nothing gates on it. |
| `precedence_order` | structuring, cycle, scatter_gather, fan_in, fan_out | Load-bearing rather than hygiene. `fan_out` fires on the first leg of every `scatter_gather` and on several typologies PRD §2 excludes, so the more specific shape must claim the transactions first. |

## Reasoning core — LLD §5

| key | value | evidence |
|---|---|---|
| `confidence_threshold` | 0.75 | A draft that is supported but may be thin. Below it the retrieval question is reformulated. |
| `max_loops` | 2 | The give-up point. Phase 1 measured that **17.2%** of ObliQA's gold questions have no correct clause in the top 15 at all, so a third attempt usually spends money on a clause that is not in the collection. |
| `high_risk_min_confidence` | 0.9 | High means *file a SAR*, so it needs the top band rather than merely enough score to stop the loop. Forced by measurement: across the four pre-migration batches the model's own rating was **anti-correlated with the truth** — clean May (0 laundering) came back High recommending a SAR, July (23 laundering) came back Low. Applied in `ReportGenerationNode`, never in the critic, whose prompt forbids it from changing a fact. |
| `schema_retries` | 3 | LLD §6 `SCHEMA_PARSE_FAILURE`. A model that returned prose where a schema was asked for usually returns the schema when shown the validation error. |
| `llm_max_attempts` | 3 | Tenacity-style attempts on 429/5xx/timeout. Passed to the client as `max_retries`, so a timeout is not re-prompted as a schema error. |
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

- **`pattern_to_obligations`** — the Tier-1 map, 4–5 `(source_id, section_ref)` pairs per typology.
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
