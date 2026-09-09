# Test & Evaluation Suite Design

**As-built**, 2026-09-08. Results in [TEST_RESULTS.md](TEST_RESULTS.md).

---

## 1 · Two suites, because there are two kinds of question

**Assertable questions have right answers.** Does `5669,49` parse to 5669.49? Does a fabricated
citation get vetoed? Does the merge keep the best distance? These are ordinary tests — 233 of
them, ~40 seconds, **no API key and no network**.

**Judgement questions have no right answer.** Is this finding well reasoned? Does the report
address the audit that was requested? These need an LLM judge, cost money per case, and live in a
separate marker-gated suite.

```bash
uv run pytest tests/                    # 233 tests, free, no key
uv run pytest tests/eval_suite.py -m eval   # ~16 gpt-4o judgements, ~$0.30
```

`addopts = "-m 'not eval'"` in `pyproject.toml` keeps the paid suite out of the default run. That
separation is load-bearing: the free suite is the one that runs on every change, so anything that
makes it need a key or a network has broken its purpose, not just a test.

## 2 · The offline guarantee, and how it has been broken

Three times, a change quietly made the free suite depend on something external. Each is now
prevented by a fixture rather than by discipline:

| what happened | how it hid | the guard |
|---|---|---|
| FlashRank downloads a 3 MB model on first use | it caches to `/tmp`, so a machine that had run the pipeline once passed | autouse `reranker_off` in `test_graph.py` |
| the retrieval cache intercepts *before* `nodes.retrieve` | seven tests got real cached clauses where they had stubbed a retrieval — only on a machine with Redis running | autouse `cache_off` |
| `flashrank` was declared a **dev** dependency but imported at module scope | every local run has dev deps; only `--no-dev` (i.e. the container) failed | moved to runtime deps; caught by the first real Docker build |

The lesson each time was the same: **a dependency that is present locally is invisible until
something installs less than you do.**

## 3 · What the free suite covers

| file | tests | what it pins |
|---|---|---|
| `test_swift_parser.py` | 39 | comma-decimal, continuation lines, refusal, 880/880 against the answer key |
| `test_graph.py` | 49 | node contracts, the citation veto, routing, the cycle, RRF, evidence repair |
| `test_chunker.py` | 27 | semantic boundaries |
| `test_detectors.py` | 25 | recall against `ledger_labels.csv`; each primitive |
| `test_cache.py` | 23 | exact/semantic matching, TTL, degradation |
| `test_store.py` | 16 | build, tiering, `by_id`, backend mismatch |
| `test_cost.py` | 14 | price matching, per-node attribution, unpriced models |
| `test_pdf_generator.py` | 14 | MT103 rendering, label integrity |
| `test_acquisition.py` | 12 | manifest, checksums |
| `test_api.py` | 11 | 202/400/415/404/503, background execution, temp-file cleanup |
| `test_rerank.py` | 3 | promotion, nothing added or lost — skipped if the model is absent |

### Three testing patterns worth copying

**Ground truth, not self-consistency.** `ledger_labels.csv` names every planted laundering wire, so
detector recall is measured against what was actually planted rather than against the detector's
own output.

**Guard the specific bug, not the happy path.**

```python
def test_the_naive_reading_would_be_a_hundredfold_error():
    assert parse_amount("5810,46") == Decimal("5810.46")
    assert float("5810,46".replace(",", "")) == 581046.0, "the trap is still a trap"
```

The second assertion fails if the trap ever stops being a trap — at which point the test is
obsolete and should say so.

**Every stub is a real object.** `StubModel` records prompts *and* configs, so a test can assert
that the two drafts of a looping run carry different `loop:` tags. Stubs that only return values
cannot verify observability.

## 4 · The evaluation suite

Four cases, one per batch, each a real captured run scored on three metrics (§8.1):

| metric | question | catches |
|---|---|---|
| **Faithfulness** | does every claim rest on the retrieved clauses? | hallucination |
| **Answer Relevancy** | does the report address the audit requested? | drift |
| **Contextual Precision** | did retrieval rank the useful clauses above the noise? | a *retrieval* failure disguised as a writing failure |

The third is the valuable one: a thin report has two possible causes needing opposite fixes — the
model wrote badly, or the model never received the right law — and they are indistinguishable from
the output alone.

Thresholds: 0.85 / 0.80 / 0.70. The first two are the blueprint's; **0.70 is a starting line, not a
measured one**, and the honest run is what should eventually set it.

### Two deliberate departures from the blueprint

**The gold set is real.** §8.2 proposes hand-written scenarios; its example cites `FINRA Rule
3310(a)`, which cannot be grounded here at all, because FINRA publishes no rule PDFs and the ADGM
AML Rulebook is what was indexed. Instead each reference answer is generated from
`ledger_labels.csv`:

> *"This batch contains 21 laundering wires out of 220: 10 exhibiting structuring, 8 smurfing,
> 3 deposit-send."*

**Capture is separate from scoring.** `src/graph/evalset.py` runs the pipeline and freezes input,
output, retrieval context and reference answer to `eval_cases.json`; the suite scores that file.
Re-running the pipeline to test a prompt change would vary *two* things — the report and the
clauses it saw. Freezing the evidence isolates the variable.

### The assertion no judge can make

```python
def test_a_clean_batch_is_not_reported_as_a_finding(case):
    if case["ground_truth"]["laundering_wires"] == 0:
        assert case["run"]["risk_rating"] == "Low"
```

No LLM judge catches a wrongly-rated clean batch: each report is individually plausible,
internally consistent, and scores 1.000 on Faithfulness. Only the answer key knows better. It is
written as a **failing test rather than a paragraph** because a known defect described in prose
gets forgotten, and one that turns the run red does not.

## 5 · Retrieval benchmarking

Separate from both suites, because it measures the corpus rather than the code:
`src/ingestion/benchmark.py` scores hit@k / recall@k / MRR against ObliQA's **2,786 labelled
questions**. This is where the reranker was decided — and where the ceiling was established:
17.2% of questions have no correct clause in the top 15, which no reranker can lift.

## 6 · Live verification

Some properties only appear against a real model, and each of these was a defect found that way,
not in a test:

- gpt-4o running to its 16,384-token output ceiling and killing a run after four paid calls
- the risk rating being anti-correlated with ground truth
- the citation veto firing on genuinely fabricated citations
- an unhashable callback crashing mid-run
- a container failing on a dev-only dependency, a re-installed CUDA torch, and a read-only mount

The protocol is one run per batch after any change to prompts, retrieval or the graph, recording
candidates, queries, clauses, critic passes, confidence, rating, citations and cost.

## 7 · What is not tested

Stated so the gaps are choices rather than oversights:

- **`escalate()` against a live model** — no generated batch contains a malformed message
- **Concurrency** — the API registry is a single-process dict; no test covers two workers
- **The Streamlit UI itself** — its data paths are exercised, its rendering is not
- **Load** — no throughput or sustained-run testing
- **Adversarial input** — a deliberately hostile PDF is refused by `parse_batch`, but prompt
  injection through wire memo fields is unexplored
