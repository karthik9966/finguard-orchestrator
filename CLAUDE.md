# FinGuard Orchestrator — working notes

An agentic AML audit engine. A month of transactions in, a `ComplianceReport` out, in which every
finding carries the clause it rests on. US law only; synthetic data (SAML-D) throughout.

Full docs in `docs/` — start with [docs/DESIGN.md](docs/DESIGN.md). **Don't duplicate them here.**
This file is only for what a newcomer would otherwise get wrong.

## Commands

```bash
uv sync                                    # everything runs through `uv run`; no venv activation
uv run pytest tests/ -q --cov              # 539 tests, ~4.5 min, no API key, no network
uv run pytest tests/test_reasoning.py -q   # one file while iterating
uv run python -m eval.run --tier deterministic       # ~60s, free
uv run python -m eval.run --tier live --batches 1    # real model calls; not yet run on v2 (v1: ~$0.45)
uv run finguard-audit --batch data/processed/ledger/2023-05_private_banking_log.pdf
uv run finguard-audit --ascii              # draw the graph, offline
```

Generated data is **not** committed. If tests skip with "run: uv run finguard-ledger", that is why:

```bash
uv run finguard-ledger --profile dev       # the tuning corpus   → data/processed/ledger/
uv run finguard-ledger --profile eval      # the golden corpus   → data/processed/eval_ledger/
uv run finguard-download && uv run finguard-chunk --rules && uv run finguard-store --rules
```

## Conventions with teeth

Each of these is enforced by a test, so breaking it fails the suite rather than review.

- **Numbers live in `config.yaml`, secrets in the environment.** Never a literal threshold in code.
  And never `NAME = get_config().x.y` at *module* level — that freezes at import and makes the file
  look live while being dead. Read config at the point of use.
- **`src/` must not import from `eval/`.** That boundary is what stops an evaluation fixture becoming
  production behaviour.
- **Tests need no API key.** Anything that would reach a model takes an injectable factory; the PR
  gate *fails* if a key is present. If you find yourself needing one in a test, stub instead.
- **Never write `results.db` into the repo.** `audit_batch` persists by default, so point
  `RESULTS_DB_URL` at a `tmp_path` or pass `store=InMemoryResultsStore()`. A conftest fixture fails
  the suite if a database appears in the working tree.
- **Curated rule references are `(source_id, section_ref)` pairs, never `chunk_id`s.**
  `chunk_id = hash(source_id, section_ref, version)`, so a literal id goes stale on a re-chunk with
  no error at all. Resolve pairs at load time.

## Things that are the way they are for a measured reason

Don't "simplify" these without reading the comment above them:

- **The clean batch costs exactly $0.0000.** Detection finds nothing → the model client is never
  constructed. This is a guarantee with a test, not an optimisation.
- **The faithfulness gate runs *before* the critic model.** Cited ids must be a subset of the
  retrieval bundle. Don't reorder it — there is no score a judge could return that makes a fabricated
  citation acceptable.
- **Errors are per candidate.** One candidate failing must leave its neighbours' findings intact.
  `INGEST_FILE_UNREADABLE` is the single deliberate exception.
- **Two query registers.** `indicator_query` (behaviour-shaped) is what the retriever uses;
  `obligation_query` (duty-shaped) is unused at runtime. Using the duty shape for the indicator
  search cost 38 points of context precision — measured.
- **`detection_confidence` is anti-correlated with planted wires.** Nothing may gate on it.
- **Report generation makes no model call.** The pre-migration version had a model write the filing;
  its risk ratings came back anti-correlated with the truth.
- **`reports.report_json` is immutable; `findings.status` is not.** So reads return a *join*
  (`GET /reports/{id}`) and `…/filed` returns the original. Don't denormalise status into the JSON.

## Two corpora, and never mix them

- `data/processed/ledger/` — **tuning.** Every number in `docs/CONSTANTS.md` cites a dev-corpus
  measurement.
- `data/processed/eval_ledger/` — **held out.** What `eval/` reports against.

They slice the same SAML-D months, so what keeps them apart is not the directory but `partition_of`
in `pdf_generator.py`: every SAML-D cluster belongs to dev or eval by a hash of its anchor account.
v1 lacked it and every dev cluster was also a golden one. Don't give a profile `partition=None`.

`--profile dev` clears the ledger directory, including the 10,000-message timing batch — run
`uv run finguard-ledger --profile large --append` after it, or the large-batch test skips.

Running `finguard-ledger --profile eval --append` would merge the golden months into the tuning
corpus, and the recall harness iterates every log file it finds — so the recorded numbers would
silently become numbers measured on different data. The `eval` profile writes to its own directory
for this reason.

## Layout

```
src/config.py       Settings (env) + Config (config.yaml)
src/models.py       every contract — read this first
src/ingestion/      corpus acquisition, chunking, vector store, batch ingestion
src/detection/      graph engine, nine typology detectors, reconciler, evidence.  no model
src/retrieval/      TierAwareRetriever, cross-encoder
src/graph/          six nodes, the graph, run.py (the orchestrator), prompts.py
src/store/          results store
src/api/ src/ui/    FastAPI; a Streamlit cockpit that is a *client* of it
src/observability/  Langfuse, redaction on the client
eval/               golden datasets, metric runners, the rules-only baseline
```

`src/graph/run.py:audit_batch` is the single entry point — the CLI and the API both go through it.

## Current state

Branch `migrate/v2-graph`: the `_v2.docx` design set is implemented — nine patterns over one money-flow
graph. Two deterministic metrics fail **deliberately**, diagnosed rather than tuned: per-pattern recall
(scatter_gather 5/6, on every whole held-out instance SAML-D has) and context precision (0.33 hit@1,
the four v2 queries). **The live tier has not been run on v2** — its numbers in `docs/TEST_RESULTS.md`
are v1's and marked so. Structuring, v1's open question, is 15/15 via PRD v2's Option 1 plus
beneficiary-side grouping.

When you change behaviour, record the measurement that justified it — in `docs/CONSTANTS.md` for a
number, `docs/CHANGELOG.md` for a decision. A value without evidence beside it is a value nobody can
change safely later.
