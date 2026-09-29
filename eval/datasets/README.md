# Golden datasets

The six corpora of Evaluation Design §3. Sizes are deliberately modest — small enough that every
label can be checked by a person, which is the only thing that makes them *golden*.

| dataset | records | origin | measures |
|---|---|---|---|
| `labeled_patterns.json` | 75 (15 × 5) | **derived** | recall ≥ 0.90, triage, narrative quality |
| `clean_batch.json` | 1 batch, 500 txns | **derived** | zero candidates, zero LLM calls, $0.0000 |
| `benign_lookalikes.json` | 20 | **authored** | triage: must rank below real launderers |
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

That is a **different corpus** from `data/processed/ledger/`, deliberately. Every number in
`config.yaml` — the 14-day window, the 0.2 band fraction, `min_sources: 4` — cites "measured across
the four dev batches" as its evidence. Evaluating on that same data would be marking my own homework.

The eval corpus is 11 months × 1,200 messages with three clusters of each in-scope typology per
month, and it is deterministic: same seed, same SAML-D, same references. That reproducibility is what
makes it safe to commit a dataset that points into a generated corpus, and `--check` is the assertion
that they still agree.

Neither ledger is committed (`data/processed/` is gitignored — it is 60 MB of generated PDFs).

## Labels that are mine, not the data's

These are the ones to argue with:

- **`benign_lookalikes.json` risk bands.** 13 `low` and 7 `medium`. The `medium` ones are deliberate:
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
