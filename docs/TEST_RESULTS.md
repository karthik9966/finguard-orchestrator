# Test results — v2

What the system scores, including what fails. Measured on the golden corpus, which is **not** the corpus
the detectors were tuned on — and in v2, provably so: see [TEST_DESIGN.md](TEST_DESIGN.md).

Reproduce:

```bash
uv run pytest tests/ -q --cov --cov-report=term          # 539 tests, free
uv run python -m eval.run --tier deterministic           # ~60s, free
uv run python -m eval.run --tier live --batches 1        # not yet re-run for v2 -- see below
```

**Two kinds of number below.** Everything the deterministic tier measures was re-measured for v2 on
2026-09-29. The live tier — faithfulness, triage, injection, narrative quality, cost — **has not been
re-run since v1**; those rows are v1's figures, marked as such, and they are a claim about the v1 system
until the v2 live run replaces them. v2 grounds more candidate shapes and shows the model a subgraph, so
none of them can be assumed to carry over.

---

## Summary

| KPI | target | v2 measured | v1 | |
|---|---|---|---|---|
| Recall, pattern level | ≥ 0.90 | **0.959** (118/123) | 0.827 | ✓ |
| Recall, weakest pattern | ≥ 0.90 each | **0.833** — scatter_gather 5/6 | — | ✗ |
| Recall, named pattern | reported | **0.886** (109/123) | — | |
| Alert volume (share of batch) | reported | **9.2%** (baseline 73.5%) | 19.1% | |
| Context precision (hit@1) | ≥ 0.90 | **0.33** (hit@3 0.67) | 0.60 | ✗ |
| Schema conformance | 100% | **100%** (26,786 objects) | 100% | ✓ |
| Clean batch | 0 candidates, 0 calls, $0.0000 | **0 candidates** | all three | ✓ |
| Malformed inputs handled | 10/10 | **10/10** | 10/10 | ✓ |
| Graph-engine correctness | 100% hard gate | **19/19** in `test_graph_engine.py` | — | ✓ |
| Branch coverage (audit path) | ≥ 85% | **86.8%** | 85.6% | ✓ |
| 10,000-message batch | detection inside 5 min | **5.3 s**, 308 candidates | seconds, 304 | ✓ |
| Faithfulness | 1.00 hard gate | *not re-run* | 1.00 | — |
| Triage: TPs ranked high/medium | ≥ 90% | *not re-run* | 100% | — |
| Prompt injection resisted | 1.00 | *not re-run* (system defence: 5/5 in Tier 3, stubbed) | 5/5 | — |
| Narrative quality | ≥ 0.85 advisory | *not re-run* | 1.00 | — |

Two deterministic metrics fail. As in v1, they are left failing with named causes rather than tuned until
green — and here the reason is sharper than principle: the only data left to tune them on is the golden
set.

## Recall, and the comparator that makes it mean something

```
ours      0.959 recall  at   9.2% of the batch alerted
baseline  1.000 recall  at  73.5% of the batch alerted
```

The rules-only baseline catches everything by alerting on three-quarters of the ledger. The trade is
the one the system exists to make, and v2 made it better on both axes: recall up from 0.827, volume down
from 19.1%.

| pattern | pattern level | named | baseline | note |
|---|---|---|---|---|
| structuring | **15/15** | 15/15 | 15/15 | v1: 5/15 |
| fan_in | 14/15 | 13/15 | 15/15 | |
| fan_out | 14/15 | 14/15 | 15/15 | |
| cycle | 15/15 | 15/15 | 15/15 | |
| scatter_gather | **5/6** | 5/6 | 6/6 | supply-limited: 6 is every whole held-out instance SAML-D has |
| gather_scatter | 11/12 | **7/12** | 12/12 | supply-limited: 12 |
| deposit_send | 15/15 | 13/15 | 15/15 | |
| layered_fan | 14/15 | 12/15 | 15/15 | |
| bipartite | 15/15 | 15/15 | 15/15 | |

**Structuring, the v1 headline gap, is closed — by two changes, neither of them a threshold.** v1
diagnosed its 5/15 as a premise mismatch: SAML-D's "Structuring" label means *split into small amounts*,
with no threshold in it, while §5324 means *kept under* one. PRD v2's Option 1 resolves that on the data
side — gold instances are the clusters that do hug $10,000 — and a second finding resolved it on the
detector side: every SAML-D Structuring cluster is many parties paying **one receiving account**, and the
detector only grouped by originator. It now groups by both. Option 1 changes what is being measured, so
15/15 and v1's 5/15 are not the same question; the v1 figure stays in the table for that reason.

**The per-pattern gate fails on one miss in six.** scatter_gather's denominator is six because that is
every whole, held-out scatter-gather SAML-D contains; a single miss is 16.7 points. The miss
(`LP-scatter_gather-002`) is one to investigate, but no threshold should move to recover a sixth of a
six-instance set. Read this row as "5 of 6", not as "83%".

**Named recall is where the v2 detectors are weak, and it is honest to say which.** gather_scatter is
found by name 7 times in 12; the other four are covered by `fan_in` or `fan_out`, reporting half the
shape. The dev sweep only reached 3/5 by name, and the curve is in [CONSTANTS.md](CONSTANTS.md) —
SAML-D hubs do not conserve money tightly (out/in from 0.80 to 4.06), which is what the detector's
conservation band leans on. deposit_send's two named misses are covered by other detectors, as are two of layered_fan's three;
the third layered instance (`LP-layered_fan-014`) is missed outright.

## Context precision: 0.33 hit@1, 0.67 hit@3

The five v1 queries score exactly as in v1 (3/5 hit@1; CQ-002 and CQ-003 characterised in the v1
section below). The drop is the four v2 queries, **0/4 at rank 1**:

| query | correct at | what outranked it |
|---|---|---|
| CQ-007 gather_scatter | absent from top 5 | both distractors — each describes *half* the shape (collect-and-funnel; many beneficiaries from one company) |
| CQ-008 deposit_send | rank 2 | a FINRA red flag; the corpus has **no cash-specific** deposit-then-wire flag, so the correct answer is the checks-and-money-orders version |
| CQ-009 layered_fan | rank 2 | Other Transactions ¶ 2 — CQ-003's fan-in answer |
| CQ-010 bipartite | absent from top 5 | its distractor ¶ 15 (one person, several accounts) at rank 1 |

Characterised and not fixed, for v1's reason: rewording a template until its own measuring record passes
tunes the instrument to the reading. CQ-007 is the instructive one — a query describing a pass-through
retrieves each half of a pass-through, which suggests the behaviour-register template for a two-sided
shape needs to say what connects the halves.

## The v1 sections below are v1's measurements

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

1. **The live tier has not been run on v2.** Faithfulness, triage, injection resistance, narrative
   quality and per-candidate cost are v1's numbers. It needs a key and costs money — more than v1's
   ~$0.45 per batch, because a golden batch is now 2,400 messages rather than 1,200.
2. **Per-pattern recall below 0.90 for scatter_gather** (5/6), on a denominator SAML-D cannot enlarge.
3. **Context precision** 0.33 hit@1, with all four v2 misses characterised above.
4. **gather_scatter named recall** 7/12 — the detector is the weakest of the nine by name.
5. **Benign lookalikes are never executed.** 24 authored records, schema- and reasoning-checked, but no
   runner grades their risk band; the triage metric grades true positives only.
6. **The corpus has no cycle-specific red flag, and no cash-specific deposit-then-wire flag.** CQ-006
   expects no indicator; CQ-008's correct answer names checks and money orders.
7. **`deposit_send.amount_tolerance` was sized on all of SAML-D**, before the dev/eval partition existed,
   so its evidence touched golden clusters. The dev-only sweep that followed did not move it.
8. **Neither CI workflow has run.** No runner here; the nightly needs secrets.
9. **`docker compose` is configuration-validated but not launched** — no Docker daemon on this machine.
10. **The cockpit's file-uploader widget is the one path `AppTest` cannot reach.**
