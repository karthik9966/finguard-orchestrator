# Test & Evaluation Suite Results

**Run 2026-09-08**, all measurements taken in one sitting against commit `3f9a5a3`
(*add caching at retrieval*). Design in [TEST_DESIGN.md](TEST_DESIGN.md).

---

## 1 · Summary

| | result |
|---|---|
| Free suite | **233 passed**, 42.7s, no API key, no network |
| Free suite with Redis stopped | **233 passed**, 39.1s — identical |
| Evaluation suite | 13 passed, **3 failed** (16 gpt-4o judgements) |
| Live audits | 4 batches, all produced valid reports |
| Container | built, 3.13 GB, full audit completed inside it |

The three eval failures are real signal, not flakes. They are discussed in §4 and §5.

## 2 · Free suite

```
233 passed in 42.72s
```

| file | tests | | file | tests |
|---|---|---|---|---|
| `test_graph.py` | 49 | | `test_store.py` | 16 |
| `test_swift_parser.py` | 39 | | `test_cost.py` | 14 |
| `test_chunker.py` | 27 | | `test_pdf_generator.py` | 14 |
| `test_detectors.py` | 25 | | `test_acquisition.py` | 12 |
| `test_cache.py` | 23 | | `test_api.py` | 11 |
| | | | `test_rerank.py` | 3 |

**The offline guarantee was verified both ways.** With the Redis container stopped, the same 233
tests pass in 39.1s. That is the contract for §9.3: a cache that can break an audit is worse than
no cache.

## 3 · Component measurements

**Parsing** — 880/880 wires across four batches, every field matching `ledger_labels.csv`; PDF and
TXT renderings parse identically; 0 refusals.

**Detection** — 100% recall (52/52 planted laundering wires), 32% precision, 164/660 wires swept.

**Retrieval** — against ObliQA's 2,786 labelled questions:

| | embedding | + FlashRank |
|---|---|---|
| hit@1 | 45.2% | **55.6%** |
| hit@4 | 65.2% | **72.9%** |
| hit@8 | 73.2% | **77.6%** |
| hit@15 | 79.2% | 79.2% |

hit@15 is unchanged by design — a reranker reorders and cannot add. The 17.2% of questions with no
correct clause in the top 15 is the ceiling.

**Cache** — August batch, audit node only:

| | time | hits |
|---|---|---|
| cold | **44.0s** | 0/7 |
| warm | **0.0s** | **7/7, 21.0s saved** |

Retrieved context byte-identical between the two. On a full run the warm figure is 7/8 — the miss
is the critic's reformulated query, which is model-written and new every time.

## 4 · Live audits — 2026-09-08

| batch | truth | rating | confidence | passes | cost |
|---|---|---|---|---|---|
| 2023-05 *(clean control)* | **0** laundering | **Low** | 0.00 | 2 | $0.0839 |
| 2023-06 | 21 | Medium | 0.50 | 2 | $0.1231 |
| 2023-07 | 23 | Medium | 0.50 | 2 | $0.1768 |
| 2023-08 | 8 | Medium | 0.75 | 2 | $0.0962 |

### The May result needs reading carefully

May is the clean control and came back **Low** — the correct answer, and the first time it has.
**That is not the risk-rating defect being fixed.** Confidence was **0.00**, which is the citation
veto's score, and the reservations say why:

> - Cites 'FINRA Regulatory Notice 19-18 part 2', which is not among the retrieved clauses.
> - Cites 'FinCEN Alert FIN-2023-Alert002 (commercial real estate) part 21', which is not among
>   the retrieved clauses.

**The draft fabricated two citations and the Python gate caught both.** The right rating arrived
by way of a failure, not by calibration. Two things are simultaneously true and both worth
recording:

- **The safety machinery works in production.** This is the first observed live firing of the
  citation veto. `applicable_regulations` on the filed report contains only
  `AML Rulebook 14.2.3.Guidance.1.` — the one clause genuinely retrieved. The fabrications were
  stripped by `generate_node`'s evidence repair and recorded as reservations, exactly as designed.
- **The rating defect is unchanged.** June, July and August all read Medium regardless of carrying
  21, 23 and 8 laundering wires. The failing test stays failing.

## 5 · Evaluation suite

16 gpt-4o judgements, 138s.

| batch | Faithfulness | Answer Relevancy | Context Precision |
|---|---|---|---|
| 2023-05 | **1.000** | 0.765 ✗ | 0.646 ✗ |
| 2023-06 | **1.000** | 0.968 | 0.547 ✗ |
| 2023-07 | **1.000** | 1.000 | 0.887 |
| 2023-08 | **1.000** | 1.000 | 0.975 |
| threshold | 0.85 | 0.80 | 0.70 |

**Faithfulness is 1.000 on every batch.** The reports invent nothing — measured, not asserted.
Notably this holds on May *even though its draft fabricated two citations*: the veto and the
evidence repair removed them before the report existed, so the artifact being judged was clean.
The layered defence is doing exactly what it was built for.

**Context Precision is the weak metric**, and it is the same finding the reranker experiment
reached independently. The judge's reasoning on June:

> *"relevant nodes are present, but not consistently ranked higher than irrelevant nodes"*

and on May:

> *"the fourth node, which focuses on BSA reporting for large-dollar cash transactions…"*

— a US Bank Secrecy Act clause ranked above ADGM material on an ADGM batch. That is a retrieval
ordering problem, not a writing problem, which is precisely the distinction this metric exists to
draw.

**May's Answer Relevancy of 0.765** is arguably the metric working correctly against a report that
should not have been written at all: on a batch with nothing to find, discussion of a legitimate
£337,217 consultancy fee reads as partly irrelevant, because it is.

## 6 · Cost

| batch | calls | tokens | cost |
|---|---|---|---|
| no candidates | 0 | 0 | **$0.0000** |
| 2023-05 | 5 | — | $0.0839 |
| 2023-08 | 5 | — | $0.0962 |
| 2023-06 | 5 | — | $0.1231 |
| 2023-07 | 5 | — | $0.1768 |

Across 12 runs recorded over the project: **median $0.093, mean $0.100, range $0.047–$0.178.** The
spread is not batch size — all four batches hold exactly 220 wires. It is driven by whether the
critic accepts the first draft (3 calls vs 5) and by how many candidates fire.

Evaluation adds roughly **$0.30** per full scoring run.

## 7 · Container

```
docker build -t finguard .        → 3.13 GB
GET  /health                      → {"status":"ok","vectors":12273,"backend":"minilm"}
POST /audit  (2023-05)            → 202, 220 wires
GET  /audit/…                     → complete after ~40s
                                  → risk Medium · confidence 0.50 · 5 calls · $0.0935
```

An earlier estimate of ~1.2 GB was wrong by 2.6×: the Linux CPU torch wheel is 656 MB, not the
~200 MB inferred from macOS. The CPU swap still earns its place — it removes roughly a gigabyte of
`nvidia-*` CUDA runtime, confirmed by the build printing `torch 2.14.0+cpu cuda None`.

## 8 · Open, and unchanged by this run

1. **Risk rating does not separate clean from dirty batches.** May's Low came from a veto.
2. **Context Precision below threshold on two of four batches** — retrieval ordering.
3. **17.2% retrieval ceiling** — no correct clause in the top 15; not addressable by reranking.
4. **`escalate()` never exercised live.**
5. **Detector precision 32%** — deliberate, given 100% recall.
