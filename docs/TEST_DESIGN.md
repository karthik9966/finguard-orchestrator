# Test design as built

Companion to `FinGuard_Eval_Design.docx`. The three tiers, the six golden corpora, and the two CI
gates. What it scores is in [TEST_RESULTS.md](TEST_RESULTS.md).

---

## The two departures that shape everything else

Eval Design names them and they held:

**Faithfulness is deterministic, not judged.** The critic rejects any finding whose citations are not a
subset of what was retrieved, so the target is 100% *enforced by code* — a proof rather than an
estimate, and cheaper because it needs no extra model call. An LLM judge survives only for narrative
readability, and it is advisory.

**The attack surface is data, not chat.** The model never sees free-form user input, only structured
records plus retrieved rule text. The one realistic injection vector is a transaction's memo field,
which is exactly why the memo reaches the prompt rather than being stripped — a fixture whose memo is
withheld tests nothing.

## The three tiers, and where each lives

| tier | asks | where | cost |
|---|---|---|---|
| 1 · deterministic integration | do the nodes, detectors, retrieval-by-id and the SQLite write behave? does every inter-node object satisfy its model? | `tests/` (489 tests) + `eval.run --tier deterministic` | free |
| 2 · probabilistic ML quality | does the system *report* the finding, cite only retrieved rules, rank launderers above lookalikes, read clearly? | `eval.run --tier live` | ~$0.45 per golden batch |
| 3 · adversarial robustness | malformed input, LLM timeout, clean batch, injected memo, empty RAG | `tests/test_failure_injection.py` (23 tests) | free |

Tier 3 is free and in the unit suite by design. Every row of Eval Design §5 is reachable with a stubbed
model — an empty bundle, a timeout, a malformed file, a clean batch, a complied-with injection — so they
run on every push rather than nightly. The one thing a live model adds is whether the *model* resists an
injection, and that is measured in Tier 2; what Tier 3 asserts is that the system's own defences hold
regardless of what the model does.

## The six golden corpora

Full provenance in [`eval/datasets/README.md`](../eval/datasets/README.md).

| dataset | records | origin | measures |
|---|---|---|---|
| `labeled_patterns` | 75 (15 × 5) | **derived** | recall, triage, narrative quality |
| `clean_batch` | 1 batch, 500 txns | **derived** | zero candidates, zero calls, $0.0000 |
| `benign_lookalikes` | 20 | **authored** | triage: must rank below real launderers |
| `complex_queries` | 6 | **authored** | context precision |
| `malformed_inputs` | 10 + files | **authored** | ingestion degrades, never fabricates |
| `injected_memos` | 5 | **authored** | memo text is inert data |

**Derived** means nobody's judgement is in the label — SAML-D ships its suspicious rows already
labelled, so these are a *selection* and `build_datasets --check` reproduces them exactly. **Authored**
means somebody decided, so every authored record carries its own reasoning field: the reasoning *is* the
label, and a reviewer who disagrees with it is disagreeing with the label.

### The corpus these point at is not the one the detectors were tuned on

`labeled_patterns` names transactions in `data/processed/eval_ledger/` — eleven months, 1,200 messages
each, three clusters of every in-scope typology per month:

```bash
uv run finguard-ledger --profile eval
```

That is deliberately **not** `data/processed/ledger/`. Every number in [CONSTANTS.md](CONSTANTS.md)
cites "measured across the four dev batches" as its evidence; evaluating on that same data would be
marking my own homework. The gap between the two is visible in the results and is the point of having
both.

I got this wrong once and caught it: the first eval build ran with `--append` into the dev corpus, which
put six extra months into `ledger_labels.csv` — and the recall harness groups by log file and iterates
all of them, so the recorded 99% would silently have become a number measured on different data.

### Three changes the golden set required of the generator

- **`clusters_per_typology`** — `select_cases` planted one cluster per typology per month, which caps at
  36 instances across every month SAML-D has. Seventy-five needs three per typology per month. The
  default stays 1, so the dev and large corpora regenerate identically.
- **The answer key names its own instances.** A `Cluster` column, written by the planter. Recall at the
  PRD's level is "did the system report this fan-in", which needs to know which transactions form one
  instance — and a cycle is a walked chain with no anchor account, so reconstructing its membership from
  the labels afterwards is not reliably possible.
- **Disjoint rings.** Cycles were capped at one per month on the theory that a second ring drawn from the
  leftovers is a fragment of the first. Measured, that was wrong: walking the remaining edges gives
  genuine disjoint rings of median 9 transactions, and available cycle instances went 11 → 29.

Sizing is a consequence of `MAX_FLAGGED_SHARE = 0.15` rather than a preference: a cluster averages ~20
transactions, so a 500-message batch holds three before the flagged share stops being credible.

## The rules-only baseline

`eval/baseline.py`, HLD §1.1's comparator, offline and explicitly not a runtime component.

It exists because **recall alone is unfalsifiable** — a detector that flags every transaction scores
1.00 — so the interesting claim is never recall but *recall at a given alert volume*. The baseline is
what a legacy transaction-monitoring engine does: five flat threshold rules, no graph, no window, drawn
from the same statutes the real detectors cite so it is not a strawman.

| rule | |
|---|---|
| R1 | any transfer ≥ $10,000 (CTR filing trigger) |
| R2 | any transfer in [$8,000, $10,000) |
| R3 | any account whose same-day total reaches $10,000 across more than one transfer |
| R4 | any account with ≥ 8 distinct counterparties in the batch — **no window** |
| R5 | any transfer ≥ $3,000 (recordkeeping threshold) |

R5 is included precisely because it is genuinely in the regulations and genuinely useless as an alert: it
fires on most of the batch, which is the thing being demonstrated.

## The two CI gates

**`pr.yml` — every push, free, fast.** The offline guarantee is asserted through the environment
(`HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`) rather than through discipline, after three pre-migration
regressions in which a change quietly made the free suite depend on something external — each invisible
locally because the developer's machine already had the thing. A step **fails the run** if
`LLM_API_KEY`, `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` is present: every test that would reach a model
injects a stub, so a run that somehow made a real call should fail rather than quietly bill, and that is
a stronger guarantee than trusting the stubs. Config and contract validation run first so a broken
`config.yaml` does not bury the run in downstream noise, and a final step reports what skipped — a gate
that silently covers less than it appears to is worse than a smaller gate.

It does **not** build the corpus. The knowledge base needs two third-party downloads (eCFR/FFIEC over
the network, SAML-D from Kaggle), and a gate depending on those fails for reasons unrelated to the change
under review. The suite is built for it: tests needing generated data skip themselves with the command
that would produce it.

**`nightly.yml` — scheduled, paid.** Runs the full suite with the data in place, then both eval tiers,
recording to Langfuse so scores are trended rather than merely pass/fail. Results are uploaded with
`if: always()`, because a failing run's numbers are the ones worth reading.

**How CI gets a built `rule_chunks`: it rebuilds, and caches on the manifest hash plus the embedding
model.** The alternatives and why not — committing the collection puts 74 MB of Chroma SQLite in every
clone for ever, and it is a build artefact of `finguard-store`; caching alone makes a cache miss a
*failure* rather than a slower run, so the first run after any corpus change fails in a way that looks
like a regression; rebuilding unconditionally spends ten minutes of embedding every night on a corpus
that changes when a regulator publishes. The embedding model is part of the cache key because a
collection built at one dimension cannot answer a query embedded at another, and that failure is silent.

Neither workflow has been executed: there is no runner in this environment, and the nightly needs
secrets.

## Coverage

85.6% branch, gated at 85, scoped in `pyproject.toml` to the modules that run **during an audit**. The
omitted files are one-off build and acquisition CLIs — fetch the corpus, chunk it, render the ledgers,
run the retrieval benchmark — each exercised by being run rather than imported, and each of which would
drag the number down while saying nothing about whether an audit is correct. A repository-wide figure
reads 59% and is unactionable. The floor sits just under the measured value on purpose: a gate above
what the suite achieves is turned off within a week.

## The dataset is tested like an instrument

18 tests in `tests/test_golden_datasets.py`, because the failures they catch are all quiet: a curated
indicator pair that stops resolving makes context precision unmeasurable *while still producing a
number*; a labelled reference absent from the batch it names makes recall look worse than it is and looks
exactly like a detector fault; a lookalike expecting `high` makes triage precision meaningless.

One of them exists because of a defect it found: **four of the original ten query records specified
candidate attributes no detector emits** (`distinct_senders` on a structuring candidate,
`outflow_within_days` on a fan-in). They were measuring candidate shapes that never occur.
`test_every_query_candidate_uses_attributes_a_detector_really_emits` now checks every record against a
map of what the detectors actually produce, and a second test holds that map to the detectors.
