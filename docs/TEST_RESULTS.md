# Test results

What the system scores, including what fails. Measured on the golden corpus, which is **not** the corpus
the detectors were tuned on — see [TEST_DESIGN.md](TEST_DESIGN.md).

Reproduce:

```bash
uv run pytest tests/ -q --cov --cov-report=term          # 489 tests, free
uv run python -m eval.run --tier deterministic           # ~30s, free
uv run python -m eval.run --tier live --batches 1        # ~8 min, ~$0.45
```

---

## Summary

| KPI | target | measured | |
|---|---|---|---|
| Recall (detector level) | ≥ 0.90 | **0.827** | ✗ |
| Recall (system reports it) | ≥ 0.90 | **0.667** | ✗ |
| Faithfulness | 1.00 hard gate | **1.00** (110 checks, 0 violations) | ✓ |
| Schema conformance | 100% | **100%** | ✓ |
| Context precision (hit@1) | ≥ 0.90 | **0.60** (hit@3 0.80) | ✗ |
| Triage: TPs ranked high/medium | ≥ 90% | **100%** | ✓ |
| Prompt injection resisted | 1.00 | **5/5** | ✓ |
| Narrative quality | ≥ 0.85 advisory | **1.00** | ✓ |
| Clean batch | 0 candidates, 0 calls, $0.0000 | **all three** | ✓ |
| Malformed inputs handled | 10/10 | **10/10** | ✓ |
| Branch coverage (audit path) | ≥ 85% | **85.6%** | ✓ |
| 10,000-message batch | detection inside 5 min | **seconds** | ✓ |

Three metrics fail. They are left failing with named causes rather than tuned until green: a first
baseline's job is to be true, and a threshold moved to fit the number it measures stops measuring
anything.

## Recall, and the comparator that makes it mean something

```
ours      0.827 recall  at  19.1% of the batch alerted
baseline  0.907 recall  at  77.6% of the batch alerted
```

The rules-only baseline beats us on recall by alerting on four times as much. That is the trade the whole
system exists to make, and reporting recall without the volume beside it would be reporting half of it.

| pattern | ours | baseline |
|---|---|---|
| cycle | 15/15 | 15/15 |
| fan_in | 14/15 | 15/15 |
| fan_out | 14/15 | 15/15 |
| scatter_gather | 14/15 | 15/15 |
| **structuring** | **5/15** | 8/15 |

**The whole gap is one detector, and it is diagnosed.** The 15 planted structuring clusters have amounts
spanning **$1,035–$5,811, median $2,323**, against bands of `[8000, 10000)` and `[2400, 3000)` with
`min_count: 3`. The $10,000 band catches nothing — no planted amount reaches $8,000 — and the $2,400 band
catches only the one or two transactions per cluster that land inside it, below the minimum.

That is a **mismatch of premises rather than obviously a bug.** Structuring under 31 USC §5324 means
amounts *chosen* to stay under a reporting threshold, so a cluster spread across $1,035–$5,811 is not
structuring however suspicious it is. SAML-D's `Structuring` and `Smurfing` labels mean only "split into
many small amounts" and model no threshold at all.

What *is* a real gap is what falls between the detectors: **one account making many modest deposits over
a fortnight is caught by neither** — structuring wants the amounts banded near a threshold, fan-in wants
four or more *distinct* senders. Three options, none taken:

1. Give structuring an aggregate rule — *n* transfers totalling over a threshold inside the window,
   regardless of band. Catches these, and will cost precision on ordinary business.
2. Widen `band_fraction`. Cheapest and worst: Phase 3 already measured that no band width separates
   structuring from clean traffic (clean median $6,220, 73% under $10,000).
3. Report recall per pattern with this caveat and leave the detector matching the statute.

**System recall (0.667) is the same root cause**, measured on one batch where both misses are structuring.
Worth stating plainly: on a single batch of nine instances each miss is 11 points, and two live runs of
the *same* batch differed — 0.778 then 0.667 — on model non-determinism at temperature 0. A recall figure
over one batch is a weak claim, which is why the batch cap is reported beside the number.

## Faithfulness: 1.00, and it found a defect getting there

Three checks per finding, over every filed report: every cited id resolves to a real chunk; the narrative
names no chunk id the finding does not cite; the narrative names no transaction outside the candidate.

**First live run: 0.9811 over 106 checks.** Two findings whose narrative named a chunk id their own
citation list omitted. Nothing was fabricated — the clause was real and had been in the retrieval bundle,
so the critic was right to pass it — but a report that cites something it does not list is internally
inconsistent, and §6.4's citations drawer could not resolve what the prose pointed at.

The repair lists what the model used rather than editing its prose: an id named in the narrative and
present in the bundle is added to the finding's citations. **Re-measured live: 1.00 over 110 checks, zero
violations.** Two tests hold both halves — a narrative-only id is listed, and a narrative-only id that was
*never retrieved* is still a veto, so the repair cannot become a way in.

## Context precision: 0.60 hit@1, 0.80 hit@3

Up from **0.22**, which was a genuine defect: the Tier-2 indicator search was using obligation-shaped
queries. See [LLD.md](LLD.md) §2.4 for why Phase 1's measurement did not transfer. Fixing the register
moved CQ-001 from absent-from-the-top-5 to rank 1.

The two remaining misses, both characterised:

- **CQ-002** — structuring at the $3,000 recordkeeping threshold returns the $10,000 answer. The threshold
  reaches the query only as an appended clause, and that is not enough to separate two chunks whose
  difference *is* the threshold. Putting the amount inside the template body is the obvious fix and is
  deliberately not done: it would be tuning against the single record that measures it.
- **CQ-003** — fan-in ranks *"deposits to various accounts that are purportedly unrelated"* above
  *"multiple accounts used to collect and funnel funds to a small number of beneficiaries"*. The first
  describes dispersal *across* accounts, the second collection *into* one. I believe the label is right
  and the retriever wrong; it is the closest call in the set.

hit@3 is reported because `rerank_top_n` is 5: hit@3 of 0.80 is what the model actually sees.

## Cost

| | |
|---|---|
| clean month (500 records, 0 candidates) | **$0.0000**, 0 model calls |
| per candidate | **$0.0186 – $0.0248** |
| one golden batch (1,200 records, 23 candidates) | $0.43 – $0.47 |
| projected: 10,000-message batch, 304 candidates | ~$5.50 – $7.50 |

The range on per-candidate cost is the self-check loop: at temperature 0 the critic still scored the same
drafts differently across runs, and each extra pass is two more model calls. There is deliberately **no
candidate cap** — and specifically not one by `detection_confidence`, which was measured as
*anti-correlated* with planted wires (incidental 0.562, planted 0.395), so it would drop the real findings
first.

## Tier 3 — adversarial, 23 tests, all passing

| scenario | expected | result |
|---|---|---|
| Empty RAG | needs_review with a reason; **no model call at all** | ✓ |
| Indicator miss only | proceeds on obligations; finding still filed | ✓ |
| LLM timeout | that candidate halts; neighbours' findings and evidence intact | ✓ |
| Timeout ≠ schema error | not re-prompted | ✓ |
| Malformed inputs (10) | each degrades as its record says; no exception escapes | ✓ |
| Garbage reaching detectors | impossible — everything surviving ingestion is a valid record | ✓ |
| Clean batch | 0 candidates, 0 calls, $0.0000, valid empty report | ✓ |
| Injected memo (5) | a complied-with draft is vetoed **before** the critic model is built | ✓ |
| Memo redaction | attack text present as data; account numbers and emails masked | ✓ |

The injection tests assert the *system's* defence rather than the model's judgement: whatever a compliant
model would do, a memo cannot put a citation into a finding, because the gate admits only ids that were in
the bundle. Whether a live model's risk level moves is the separate Tier-2 question — and it scored 5/5,
each case against its own clean-memo control so any difference is attributable to the injection.

## Privacy, measured

| | before | after |
|---|---|---|
| trace payload per run | ~137 KB | **9.2 KB** across 5 spans |
| raw account numbers in traces | 500 records' worth | **0** |
| counterparty names | all of them | **0** |
| the parsed ledger | uploaded in full | `[500 record(s) — omitted]` |

Asserted against the batch's *real* contents: accounts, names and memos are read out of the ledger and
searched for in the emitted spans, so the test cannot pass by checking values the batch does not contain.

## Retrieval benchmark (ObliQA, 2,786 labelled questions)

Kept from Phase 1 as a regression floor for the reranker. ADGM law, held in a separate collection and
never retrieved from at runtime.

| | hit@1 | hit@4 | hit@8 | hit@15 |
|---|---|---|---|---|
| embedding only | 45.2% | 65.2% | 73.2% | 79.2% |
| + cross-encoder | **55.6%** | **72.9%** | **77.6%** | 79.2% |

The unchanged last column is the point rather than a disappointment: a reranker reorders, it cannot add.
17.2% of questions have no correct clause in the top 15 at all, and nothing short of better retrieval
changes that.

## Known gaps

1. **Structuring recall**, above. The decision is deferred with three options and the measurements behind
   each.
2. **The corpus has no cycle-specific red flag.** Searching all 479 illustrative chunks for
   circular / round-trip / returns-to-origin language returns nothing, so a cycle candidate grounds on
   obligations alone. CQ-006 expects exactly that and is excluded from the precision denominator with that
   reason stated. This is a corpus gap to close, not a scoring convenience.
3. **Context precision below target**, with both misses characterised above.
4. **Neither CI workflow has run.** No runner here; the nightly needs secrets.
5. **`docker compose` is configuration-validated but not launched** — no Docker daemon on this machine.
6. **The cockpit's file-uploader widget is the one path `AppTest` cannot reach.** Everything either side
   of it is tested, and submit-through-the-client is verified against a live server.
7. **System recall is measured on one batch.** The nightly job defaults to more.
