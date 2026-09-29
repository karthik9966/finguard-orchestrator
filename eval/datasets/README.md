# Golden datasets

The six corpora of Evaluation Design v2 §3. Sizes are deliberately modest — small enough that every
label can be checked by a person, which is the only thing that makes them *golden*.

| dataset | records | origin | measures |
|---|---|---|---|
| `labeled_patterns.json` | 123 (15 × 7; gather_scatter 12, scatter_gather 6) | **derived** | recall ≥ 0.90 per pattern, triage, narrative quality |
| `clean_batch.json` | 1 batch, 500 txns | **derived** | zero candidates, zero LLM calls, $0.0000 |
| `benign_lookalikes.json` | 24 | **authored** | triage: must rank below real launderers |
| `complex_queries.json` | 10 | **authored** | context precision ≥ 0.90 |
| `malformed_inputs.json` | 10 (+ files) | **authored** | ingestion degrades, never fabricates |
| `injected_memos.json` | 5 | **authored** | memo text is inert data, not instruction |

## Derived vs authored, and why the distinction is load-bearing

**Derived** means nobody's judgement is in the label. SAML-D ships its suspicious rows already
labelled with a typology, so `labeled_patterns.json` is a *selection* from that, and
`clean_batch.json` is one month with no suspicious rows at all. `uv run python -m eval.build_datasets`
reproduces both exactly, and `--check` fails if the committed files and the corpus have drifted.

**Authored** means somebody decided. Whether a payroll fan-in *should* rank below a real launderer,
or which of four similar-sounding red flags is right for a candidate, is a judgement — so it is
written down with its reasoning rather than computed, and it is reviewable.

Every authored record carries its own `why_benign` / `why_hard` / `why` field. That is not
documentation: it is the label's justification, and a reviewer who disagrees with the reasoning is
disagreeing with the label.

## The corpus these point into is not the one the detectors were tuned on

`labeled_patterns.json` names transactions in `data/processed/eval_ledger/`, built by:

```bash
uv run finguard-ledger --profile eval
```

That is a **different corpus** from `data/processed/ledger/`, deliberately: every number in
`config.yaml` cites a dev-corpus measurement as its evidence, and evaluating on that same data would be
marking my own homework.

In v1 it was less different than this paragraph claimed. Both corpora sliced the same SAML-D months
and took the largest clusters first, so every dev-planted row was also planted here, and 3 of the 75
v1 golden instances were tuning clusters. **v2 partitions SAML-D's clusters by a hash of the anchor
account** (`partition_of` in `pdf_generator.py`); the two corpora now share no planted row.

What else decides a v2 instance — whole structures only, Option 1 for structuring and deposit-send,
deposit-send as a pair — and why two patterns fall short of fifteen is in
[TEST_DESIGN.md](../../docs/TEST_DESIGN.md). Records of a short pattern carry a `supply_limited` field
saying so, and `test_golden_datasets.py` requires it on exactly those records.

The eval corpus is 11 months × 2,400 messages with up to three clusters of each in-scope typology per
month, and it is deterministic: same seed, same SAML-D, same references. That reproducibility is what
makes it safe to commit a dataset that points into a generated corpus, and `--check` is the assertion
that they still agree.

Neither ledger is committed (`data/processed/` is gitignored — it is 60 MB of generated PDFs).

## Labels that are mine, not the data's

These are the ones to argue with:

- **`benign_lookalikes.json` risk bands.** 16 `low` and 8 `medium`; v2 added one per new shape
  (BL-021–024), of which the deposit-send one — a restaurant wiring its takings to an overseas
  importer — is `medium` for BL-007's reason. The `medium` ones are deliberate:
  a cash-intensive restaurant depositing $7,400–$9,100 weekly (BL-007), a refund-and-repurchase round
  trip retaining 97% (BL-012), a court-ordered settlement to 60 claimants (BL-016). Each is
  *explainable* but the explanation is a document somebody has to look at — calling them `low` would
  be asserting the outcome of a check nobody has run. No lookalike expects `high`; if one does, the
  dataset is wrong rather than the system.
- **`complex_queries.json` correct answers.** Nine of ten name one indicator as right and say why the
  near-misses are wrong. CQ-001 vs CQ-002 is the sharpest pair — near-identical vocabulary, different
  correct answers, because one turns on the *amount* being sub-threshold and the other on the *number
  of depositors*. A retriever that treats "structuring" as one topic gets exactly one of them right.
- **CQ-009 is excluded from the precision denominator.** Searching all 479 illustrative chunks for
  circular / round-trip / returns-to-origin language returns **nothing**: the corpus has no
  cycle-specific red flag. So the honest expectation is that no indicator is correct and grounding
  proceeds on the obligations alone (LLD §6 `EMPTY_INDICATOR_RETRIEVAL`). Scoring it would measure the
  corpus, not the retriever. **This is a corpus gap worth closing**, not a scoring convenience.

## What a record means

`labeled_patterns` records carry `detected_when: "a candidate covers >= 0.5 of txn_refs"`. Recall is
"did the system report this instance", and what counts as reporting it has to be stated rather than
left to each runner: requiring every transaction fails an instance because one leg fell outside the
window, and requiring one passes a candidate that merely clipped its edge.

`malformed_inputs` records point at real files under `malformed/`, including one that is genuinely
not UTF-8 (`non_utf8_latin1.txt`) — which is why they are files and not JSON strings.

`injected_memos` records each carry a `control_memo`. The same candidate with a clean memo is the
control, so any difference in outcome is attributable to the injection and to nothing else.

## First measured baseline (Phase 8b)

*v1's measurements, kept for the record. Current numbers are in
[TEST_RESULTS.md](../../docs/TEST_RESULTS.md).*

```
recall (pattern level)            0.8267   target >= 0.90   FAIL
alert volume                     19.1% of records
baseline recall (rules only)      0.9067
baseline alert volume            77.6% of records
context precision (hit@1)         0.60     target >= 0.90   FAIL
context precision (hit@3)         0.80
clean batch candidates            0                         PASS
malformed inputs handled          10/10                     PASS
schema conformance                100%                      PASS
```

Two metrics fail. Both are left failing with a named cause rather than tuned until green — the point
of a first baseline is to be true, and a threshold moved to fit the number it is measuring measures
nothing afterwards.

### Recall: 0.83, and it is one detector

| pattern | ours | rules-only baseline |
|---|---|---|
| cycle | 15/15 | 15/15 |
| fan_in | 14/15 | 15/15 |
| fan_out | 14/15 | 15/15 |
| scatter_gather | 14/15 | 15/15 |
| **structuring** | **5/15** | 8/15 |

Measured, not guessed: the 15 planted structuring clusters have amounts spanning **$1,035–$5,811,
median $2,323**. The detector's bands are `[8000, 10000)` and `[2400, 3000)`, and `min_count` is 3.
So the $10,000 band catches nothing at all — no planted amount reaches $8,000 — and the $2,400 band
catches only the one or two transactions per cluster that happen to land inside it, below the
minimum.

**This is a mismatch of premises, not obviously a bug.** The detector's premise is US law: structuring
under 31 USC §5324 means amounts *chosen* to stay under a reporting threshold, so a cluster spread
across $1,035–$5,811 is not structuring however suspicious it is. SAML-D's `Structuring` and
`Smurfing` labels mean only "split into many small amounts" and model no threshold at all.

What is a real gap is what falls between the detectors: **one account making many modest deposits
over a fortnight is caught by neither** — structuring wants the amounts banded near a threshold, and
fan-in wants four or more *distinct* senders. Three options, none taken yet:

1. Give structuring an aggregate rule — *n* transfers totalling over a threshold within the window,
   regardless of band. Catches these, and will cost precision on ordinary business.
2. Widen `band_fraction`. Cheapest, and the worst: Phase 3 already measured that no band width
   separates structuring from clean traffic (clean median $6,220, 73% under $10,000).
3. Report recall per pattern with this caveat and leave the detector matching the statute.

### Context precision: 0.60 hit@1, 0.80 hit@3

Up from **0.22**, which was a genuine defect: indicators were being searched with *obligation-shaped*
queries. Phase 1 measured that obligation phrasing wins — but that was when obligations were
*discovered by search*, and they now come from a curated map, so the only corpus still searched is the
illustrative one, which is written as descriptions of behaviour rather than duties. `INDICATOR_TEMPLATES`
fixed the register and moved CQ-001 from absent-from-the-top-5 to rank 1.

The two remaining misses:

- **CQ-002** (structuring at the $3,000 recordkeeping threshold) returns CQ-001's $10,000 answer. The
  threshold reaches the query only as an appended clause, and that is not enough to separate two
  chunks whose difference *is* the threshold. Putting the amount inside the template body is the
  obvious fix and is deliberately not done yet: it would be tuning against the one record that
  measures it.
- **CQ-003** (fan-in) ranks "deposits to various accounts that are purportedly unrelated" above
  "multiple accounts used to collect and funnel funds to a small number of beneficiaries". The first
  describes dispersal *across* accounts, the second collection *into* one. I believe the label is
  right and the retriever is wrong; it is the closest call in the set.

### What the first run found in the datasets themselves

Measuring exposed three defects in my own labels, all corrected:

- **Four of ten query records specified attributes no detector emits** — `distinct_senders` on a
  structuring candidate, `outflow_within_days` on a fan-in, `instrument` on a fan-out. They were
  measuring candidate shapes that never occur. `test_every_query_candidate_uses_attributes_a_detector_really_emits`
  now prevents it, checked against a map that a second test holds to the detectors.
- **CQ-005's labelled answer was wrong.** I had labelled a FINRA "rapid movement" flag; the
  retriever's first result — *"A customer deposits funds into several accounts, usually in amounts of
  less than $3,000, which are subsequently consolidated into one account"* — is the definition of
  scatter-gather. Corrected, and the old label kept as a distractor with that history.
- **The set is 6 records, not 10.** `indicator_query` is a function of pattern type and threshold, so
  there are exactly six distinguishable queries; ten records meant four duplicate measurements scored
  under different ids. Eval Design's "~10" is a size guide, and six measured honestly beats ten where
  four are the same query twice.
