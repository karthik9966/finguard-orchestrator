# Low-Level Design

**As-built**, 2026-09-08. Contracts, constants, and the evidence behind each number.
Constants are quoted from source; if a figure here disagrees with the code, the code is right and
this document is stale.

---

## 1 · State

`AgentState` is a `TypedDict` with **no `Annotated` reducers**, so LangGraph merges last-write-wins
per key. **Nothing accumulates automatically.** Every accumulating field does it explicitly:

```python
# audit_node — reads what is held, merges, returns the whole replacement
documents = {d.metadata["chunk_id"]: d for d in state.get("retrieved_context", [])}
return {"queries": queries + new_queries, "retrieved_context": ranked[:MAX_CONTEXT_CLAUSES]}

# critic_node — increments rather than assuming a reducer
return {"loop_count": state.get("loop_count", 0) + 1, ...}
```

A node returning only its *new* clauses would silently discard the first pass on a loop-back — the
refinement would trade one incomplete context for another instead of filling the gap. The same
trap bit `cache_stats`, where a fresh tally on the second pass replaced the first's and a run that
served 7/8 from cache reported `0/1 hits (cold)`.

| field | written by |
|---|---|
| `wires`, `extraction_failures` | parse |
| `candidates` | detect |
| `queries`, `retrieved_context`, `cache_stats` | audit |
| `compliance_draft` | draft |
| `loop_count`, `confidence_score`, `critique`, `reservations` | critic |
| `report`, `is_audit_complete` | generate |
| `audit_id`, `usage`, `auditor_query` | the entry point |

## 2 · Parsing — `src/utils/swift_parser.py`

MT103 is a tag-per-line format where **a field is not a line**: `:50K:` carries account, name,
street and city across four lines. A tag-per-line reader silently loses the ordering customer.

```python
TAG = re.compile(r"^:(\d{2}[A-Z]?):(.*)$")
REQUIRED_TAGS = ("20", "23B", "32A", "50K", "52A", "57A", "59")
```

`read_fields` is a state machine: a `TAG` match opens a field, anything else appends to the open
one, and `{1:` / `-}` close it. That last clause exists because the terminator was being absorbed
as a continuation line — `instruction` read `/INS/CHEQUE 04:22:05 -}`.

**The comma is the decimal separator.** `float("5810,46".replace(",", ""))` is `581046.0` — a 100×
error inside a regulatory filing. The rule that prevents it:

```python
if raw.count(",") != 1:
    raise ValueError(f"amount {raw!r} must carry exactly one decimal comma")
```

The `"." in raw` branch above it is **not** a second correctness guard — every string it rejects is
already caught by the comma count or by `Decimal` refusing to parse. It earns its place only by
naming the format in the refusal, and that text is shown to the fallback model. The docstring says
so, because an earlier version claimed more than it did.

Amounts are `Decimal`, never `float`. Country comes from `bic[4:6]`, since no field carries it.

**Refusal, not repair.** A malformed message raises `MalformedMessage` carrying the reference and
the raw text; `parse_batch(strict=False)` collects it so one bad message does not cost the other
219. Verified: **880/880 wires, every field matching `ledger_labels.csv`, PDF and TXT identical.**

## 3 · Detection — `src/utils/detectors.py`

```python
MIN_CLUSTER_WIRES = 3    # 4 loses Layered_Fan_Out entirely
MIN_PATH_HOPS = 3
MAX_PATH_GAP_DAYS = 7
MAX_PATH_LENGTH = 25
PATH_OVERLAP = 0.6       # above this share, a chain is a retelling of one already reported
MAGNITUDE_MULTIPLE = 20
MIN_CURRENCY_SAMPLE = 5
```

Four primitives — concentration, dispersion, path, magnitude — onto which all 17 SAML-D
suspicious typologies collapse. `find_clusters` runs both directions independently, so one account
can legitimately appear as both. `find_paths` is a DFS that never revisits an account as sender;
without `PATH_OVERLAP` suppression one ring produced 11 near-duplicate chains (branching chains are
not subsets of each other, so plain subset dedup misses them). It found 2 after.

`Candidate` carries anchor, references, dates, currencies, corridors, amounts, coefficient of
variation, distinct counterparties — **and `shape`, never a typology name.**

**Known weakness:** one legitimate £34,121 wire moves a cluster's CV from 0.024 to 1.039 (43×), so
any score keyed on CV over the whole group is blind to a tight subset within it.

## 4 · Retrieval — `audit_node`

```python
RETRIEVE_K = 15
MAX_CONTEXT_CLAUSES = 24
RRF_K = 60
USE_RERANKER = True
REFINEMENT_RESERVE = 5
BASE_TIERS, CROSS_BORDER_TIERS = [1], [1, 2]
```

**Per query:** retrieve 15 → rerank against *that* query → fuse. Reranking each query against its
own text rather than a joined string is measured: joined drops cited clauses from rank 3→7 and
4→12, because a clause answering one of seven questions scores badly against a paragraph
containing all seven.

**Fusion is RRF**, `Σ 1/(60 + rank)`. Distances are measured against each query's own vector and
do not compare across queries — June's seven best hits span 0.3433 to 0.4827.

| merge | rank of the clause June cites |
|---|---|
| raw distance sort *(the original bug)* | 20 / 93 |
| round-robin by per-query rank | 23 |
| min-max normalised distance | 42 |
| **RRF** | **10** |

Both intuitive alternatives are worse than the bug. A clause found by two queries also keeps the
**best** distance it earned, not whichever query reached it first — that alone cost 3 ranks and
affected 7 of 93 clauses.

**`REFINEMENT_RESERVE = 5`** — after seven queries the 24th incumbent holds an RRF score of
0.01562 while a brand-new clause at rank 1 scores `1/61 = 0.01639`. A margin that thin meant only
the refinement's *first* hit could enter: 1 new clause of 15, and a nearly inert loop. The
critic's query, and the auditor's typed one, therefore get seated rather than ranked.

**`MAX_CONTEXT_CLAUSES = 24`** — at 93 clauses the model cited nothing and the critic scored 0.00.
At 4 (the blueprint's figure) a live report loses a clause it grounded a finding on: worst cited
rank is 17 even after reranking.

**Empty retrieval raises.** With 12,273 chunks indexed, zero results means a broken store, not a
finding — and drafting against an empty regulations block yields a SAR that cites nothing while
looking confident.

## 5 · Reranking — `src/graph/rerank.py`

`ms-marco-TinyBERT-L-2-v2`, 3 MB, CPU, ~40 ms per query. Measured on ObliQA's 2,786 labelled
questions:

| | embedding | + FlashRank |
|---|---|---|
| hit@1 | 45.2% | **55.6%** |
| hit@4 | 65.2% | **72.9%** |
| hit@15 | 79.2% | **79.2%** |

The unchanged last row is the mechanism, not a disappointment: **a reranker reorders, it cannot
add.** The 17.2% of questions with no correct clause in the top 15 are untouched by it.

## 6 · Judgement — draft, critic, generate

```python
CONFIDENCE_THRESHOLD = 0.75    # below this, reformulate and loop
MAX_REFINEMENTS = 2            # 17.2% ceiling — a third try usually buys nothing
HIGH_RISK_CONFIDENCE = 0.9     # High means *file a SAR*
GENERATE_MAX_TOKENS = 4096
```

**The critic runs Python first, then the model:**

```python
fabricated = fabricated_citations(draft, state["retrieved_context"])
...
if fabricated:
    # The gate overrides the model. Not a penalty applied to its score -- a veto.
    score = 0.0
```

`CITATION = re.compile(r"\[([^\[\]\n]{3,160})\](?!\()")` — the negative lookahead keeps
`[text](url)` markdown links from being read as claims about the rulebook.

**`generate_node` repairs three fields after the model returns them**, each because the failure was
observed: `flagged_wires` (account numbers arrived where wire references belong), then a fallback
to the candidates the draft named; `source_document_hashes` (empty beside a live citation), matched
by clause text against the draft; and `risk_rating`, capped to Medium below `HIGH_RISK_CONFIDENCE`
because the model rated a clean batch High off a draft the critic had called thin.

**`GENERATE_MAX_TOKENS` plus a Python fallback** exist because a live June run ran to gpt-4o's
16,384-token ceiling emitting JSON that never closed — losing the whole run *after four paid calls
had succeeded*. `fallback_report()` assembles the filing from the approved draft, never asserts
High, and says in the summary that it was a fallback.

## 7 · Cost accounting — `src/graph/cost.py`

Token counts cannot be read off the response: three of four calls use `with_structured_output`,
which returns the parsed object and discards the `AIMessage`. A `BaseCallbackHandler` sees the raw
generation instead, and attributes spend by reading the `node:` tag that §7.2's tracing already
stamps — one notion of "which node was that", not two.

Prices are `Decimal`, matched by **longest prefix**: a response says `gpt-4o-2024-08-06`, and
`gpt-4o-mini-2024-07-18` also starts with `gpt-4o-` — matched naively the cheap model is billed
**17× over**. An unknown model reports tokens with no dollar figure rather than being priced off
the nearest entry.

`@dataclass(eq=False)` — a plain `@dataclass` generates `__eq__`, which nulls `__hash__`, and
LangChain merges callbacks through `set(handlers)`. That crashed a live run mid-flight.

## 8 · Cache — `src/utils/cache.py`

```python
TTL_SECONDS = 86400
THRESHOLD = 0.95
CONNECT_TIMEOUT = 1.0
```

Key is `sha256(backend|k|tiers|query)`. `backend` is in it because minilm and openai vectors are
not comparable; `tiers` because the filter changes the answer.

**Exact first, semantic only on a miss.** The ten templates always take the exact path, so no
embedding is computed for them. Measured, MiniLM scores genuine paraphrases around 0.65 — at 0.95
only near-identical rewordings match, so **the exact path delivers the entire win.**

The **reranked** list is cached, not the raw one, so a hit and a miss cannot disagree about
ordering. Elapsed time is stored *in* the entry: derived from the current run instead, a fully warm
run has nothing left to measure and reports 0.0s saved at the moment it saves most.

**Degradation is the contract.** No reachable Redis ⇒ no caching, silently. The unreachable verdict
is memoised per process — re-learning it burned a full connect timeout per `audit_node` call and
took the suite from 30s to 96s.

## 9 · Interfaces

**CLI** — `finguard-audit` and eight siblings, all `main() -> int`, registered in
`[project.scripts]`. `--ascii` / `--png` / `--mermaid` draw the graph; `existing_log` validates the
batch path at the argparse boundary, so a missing file is one line rather than thirty frames of
LangGraph internals.

**Cockpit** — Streamlit re-executes the whole script on every interaction and a run costs ~$0.10,
so the audit fires only from the button and the result lives in `st.session_state`, keyed by the
batch bytes plus the typed query. `stream_batch()` yields `(node, update)` per node and
`("__final__", state)` last, because LangGraph streams deltas, never the accumulated state.

**API** — `POST /audit` returns 202 with an `audit_id` and runs in the background; the batch is
still parsed *during* the request so a bad upload is a 400 in a second. `GET /health` returns 503
unless the collection is genuinely queryable. The registry is an in-process dict: right for one
instance, wrong for two.

## 10 · Constants index

| constant | value | evidence |
|---|---|---|
| `MIN_CLUSTER_WIRES` | 3 | 4 loses Layered_Fan_Out entirely |
| `MAGNITUDE_MULTIPLE` | 20 | — |
| `PATH_OVERLAP` | 0.6 | 11 near-duplicate chains → 2 |
| `RETRIEVE_K` | 15 | blueprint §9.4 |
| `MAX_CONTEXT_CLAUSES` | 24 | 93 → cited nothing; 4 → drops a rank-17 cited clause |
| `RRF_K` | 60 | original paper |
| `REFINEMENT_RESERVE` | 5 | without it, 1 new clause of 15 entered |
| `CONFIDENCE_THRESHOLD` | 0.75 | §4.2 |
| `HIGH_RISK_CONFIDENCE` | 0.9 | clean batch rated High off a 0.75 draft |
| `MAX_REFINEMENTS` | 2 | 17.2% of questions have no clause in the top 15 |
| `GENERATE_MAX_TOKENS` | 4096 | a live run hit the 16,384 ceiling and died |
| `THRESHOLD` (cache) | 0.95 | blueprint; paraphrases measure ~0.65 |
