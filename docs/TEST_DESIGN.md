# Test design as built — v2

Companion to `FinGuard_Eval_Design_v2.docx`. The three tiers, the six golden corpora, and the two CI
gates. What it scores is in [TEST_RESULTS.md](TEST_RESULTS.md).

**What v2 changed here:** recall is gated **per pattern** across nine (Eval Design v2 §4, "≥ 0.90
each"); a named-pattern recall figure sits beside it; graph-engine correctness is a hard gate of its
own; Tier 3 gains a dense/degenerate-graph case; and the golden corpus is rebuilt on a partition of
SAML-D that makes it genuinely held out — which, it turned out, v1's was not.

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
| 1 · deterministic integration | do the nodes, detectors, the graph engine, retrieval-by-id and the SQLite write behave? does every inter-node object satisfy its model? | `tests/` (539 tests) + `eval.run --tier deterministic` | free |
| 2 · probabilistic ML quality | does the system *report* the finding, cite only retrieved rules, rank launderers above lookalikes, read clearly? | `eval.run --tier live` | ~$0.45 per golden batch |
| 3 · adversarial robustness | malformed input, LLM timeout, clean batch, injected memo, empty RAG, dense/degenerate graph | `tests/test_failure_injection.py` + the bounds tests in `tests/test_graph_engine.py` | free |

**Graph-engine correctness** (Eval Design v2 §4, 100% hard gate) is `tests/test_graph_engine.py`: on
crafted graphs whose structure is known by construction — a ten-feeder, three-collector funnel; a
complete K(2,7); two stacked blocks; a disconnected graph — every query recovers exactly that structure,
and on a near-complete 120-account, 14,280-edge graph every query stays inside its bounds and returns.
It is its own step in `pr.yml` so it is visible as a line rather than buried in the full suite.

Tier 3 is free and in the unit suite by design. Every row of Eval Design §5 is reachable with a stubbed
model — an empty bundle, a timeout, a malformed file, a clean batch, a complied-with injection — so they
run on every push rather than nightly. The one thing a live model adds is whether the *model* resists an
injection, and that is measured in Tier 2; what Tier 3 asserts is that the system's own defences hold
regardless of what the model does.

## The six golden corpora

Full provenance in [`eval/datasets/README.md`](../eval/datasets/README.md).

| dataset | records | origin | measures |
|---|---|---|---|
| `labeled_patterns` | 123 (15 × 7, gather_scatter 12, scatter_gather 6) | **derived** | per-pattern recall, triage, narrative quality |
| `clean_batch` | 1 batch, 500 txns | **derived** | zero candidates, zero calls, $0.0000 |
| `benign_lookalikes` | 24 (one or more per pattern) | **authored** | triage: must rank below real launderers |
| `complex_queries` | 10 (one per distinguishable indicator query) | **authored** | context precision |
| `malformed_inputs` | 10 + files | **authored** | ingestion degrades, never fabricates |
| `injected_memos` | 5 | **authored** | memo text is inert data |

**Derived** means nobody's judgement is in the label — SAML-D ships its suspicious rows already
labelled, so these are a *selection* and `build_datasets --check` reproduces them exactly. **Authored**
means somebody decided, so every authored record carries its own reasoning field: the reasoning *is* the
label, and a reviewer who disagrees with it is disagreeing with the label.

### The corpus these point at is not the one the detectors were tuned on — now by construction

`labeled_patterns` names transactions in `data/processed/eval_ledger/` — eleven months of 2,400
messages, every in-scope typology planted up to three times a month:

```bash
uv run finguard-ledger --profile eval
```

The dev corpus (`data/processed/ledger/`) is seven months of 1,500 messages, two clusters per typology
per month. Both slice the same SAML-D months, and in v1 that was enough to break the separation: both
profiles took the largest clusters first, so **every one of v1's 84 dev-planted rows was also planted in
its eval corpus**, and 3 of the 75 golden instances (1 cycle, 1 scatter-gather, 1 structuring) were
clusters the detectors had been tuned on. Measured by regenerating v1's corpora from commit `a374ac5`.

v2 splits SAML-D's clusters in two once, by a hash of each cluster's anchor account
(`partition_of`): a cluster is tuning data or golden data, never both, whatever months or sizes a
profile asks for. Overlap is **0 rows**, and the 10,000-message timing batch draws from the dev half too.

Three more rules decide what a golden instance is:

- **Whole, inside one batch.** Gather-scatter, scatter-gather, layered and bipartite clusters are
  planted only when SAML-D has them wholly inside one month. Their clusters run 13–23 days; a month
  boundary turned most planted "Gather-Scatter" instances into their scatter side alone — a fan-out
  wearing the wrong label, and a cross-month scheme PRD v2 §2.7 excludes anyway.
- **Option 1** (PRD v2 §5.1). Structuring and deposit-send gold instances are those whose relevant
  amounts mostly sit in [$8,000, $10,000). Only ~21 structuring clusters in all of SAML-D qualify, so
  the eval profile takes every one and the dev profile none.
- **Deposit-send is a pair.** A hub's ~6 deposit-then-send pairs span ~250 days, so a month holds one;
  its minimum instance is 2 transactions rather than the generator's usual 3, and it is exempt from the
  whole-cluster rule because each pair is complete on its own.

**The cost of rigour is two short patterns.** With whole clusters and a clean partition, SAML-D can
supply only **6 scatter-gather and 12 gather-scatter** golden instances (15 and 21 whole-month clusters
exist in the entire dataset). Confirmed with the project owner: take every instance there is, record the
shortfall on each record (`supply_limited`) and put the denominator beside the number. The alternatives —
half-shape instances, or tuning on golden clusters — would each report a better-looking number that
measured something else.

I got this wrong once before and caught it: the first v1 eval build ran with `--append` into the dev
corpus, which put six extra months into `ledger_labels.csv` — and the recall harness iterates every log
file it finds.

### Changes the golden set required of the generator

- **`clusters_per_typology`** (v1) — more than one cluster of a typology per month.
- **The answer key names its own instances** (v1) — a `Cluster` column, written by the planter, because a
  cycle is a walked chain with no anchor account and its membership cannot be reconstructed afterwards.
- **Disjoint rings** (v1) — available cycle instances went 11 → 29.
- **`COMPONENT` shape** (v2) — a structure is planted as its whole connected component. Anchored
  selection planted only the edges touching one account, which is how v1's Scatter-Gather came to be
  planted as its scatter leg alone.
- **`partition`, `threshold_selection`, whole-cluster and `MIN_CLUSTER_OF`** (v2) — above.

Sizing is a consequence of `MAX_FLAGGED_SHARE = 0.15`: eval batches run 3–7% flagged, dev 6–10%.

## Recall, two ways

**Pattern level** (gated ≥ 0.90 overall, and ≥ 0.90 for the weakest pattern): an instance counts if any
candidate covers at least half its transactions. That is what the reviewer sees — the transactions are
in front of them, whatever the finding is called. **Named** (reported, not gated): the covering
candidate must be *of that pattern*. It is the measure of whether the right detector saw it, and the
gap between the two is how much work the reconciler and the fans are doing for the v2 detectors.

## The rules-only baseline

`eval/baseline.py`, HLD §1.1's comparator, offline and explicitly not a runtime component.

It exists because **recall alone is unfalsifiable** — a detector that flags every transaction scores
1.00 — so the interesting claim is never recall but *recall at a given alert volume*. The baseline is
what a legacy transaction-monitoring engine does: five flat threshold rules (unchanged for v2 — the
point of a baseline is that it does not move), no graph, no window, drawn
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

86.8% branch (v1: 85.6%), gated at 85, scoped in `pyproject.toml` to the modules that run **during an audit**. The
omitted files are one-off build and acquisition CLIs — fetch the corpus, chunk it, render the ledgers,
run the retrieval benchmark — each exercised by being run rather than imported, and each of which would
drag the number down while saying nothing about whether an audit is correct. A repository-wide figure
reads 59% and is unactionable. The floor sits just under the measured value on purpose: a gate above
what the suite achieves is turned off within a week.

## The dataset is tested like an instrument

The tests in `tests/test_golden_datasets.py`, because the failures they catch are all quiet: a curated
indicator pair that stops resolving makes context precision unmeasurable *while still producing a
number*; a labelled reference absent from the batch it names makes recall look worse than it is and looks
exactly like a detector fault; a lookalike expecting `high` makes triage precision meaningless.

One of them exists because of a defect it found: **four of the original ten query records specified
candidate attributes no detector emits** (`distinct_senders` on a structuring candidate,
`outflow_within_days` on a fan-in). They were measuring candidate shapes that never occur.
`test_every_query_candidate_uses_attributes_a_detector_really_emits` now checks every record against a
map of what the detectors actually produce, and a second test holds that map to the detectors. It
caught v2's structuring gaining a `side` attribute the day it was added.

**Benign lookalikes are authored but not executed.** Each has a `spec`, and `tests/` checks every one's
schema and reasoning, but no runner builds a candidate from them and grades its risk band — the live
triage metric grades true positives only. A known gap, recorded in TEST_RESULTS rather than papered over.
