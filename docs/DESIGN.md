# DESIGN.md — orientation for anyone (or any agent) picking this up

**FinGuard Orchestrator** turns a batch of SWIFT MT103 payment messages into a
`ComplianceReport` — an AML suspicious-activity finding grounded in real regulatory text.

Read this before changing anything. It is the short version of *why the code looks like it does*,
and most of it is counter-intuitive enough that a reasonable person would "fix" it back.

---

## The one organising principle

> **Anything with a right answer is code. Only judgement is bought.**

Parsing an amount has a right answer. Counting wires into an account has a right answer. Deciding
whether a clause *bears on* a pattern is judgement. So a 220-wire batch costs **three model calls,
not 220** — and the expensive model never sees a raw wire.

Seven nodes, four of them free:

```
parse → detect → [route] → audit → draft → critic → generate
 free    free      free      free     $       $         $
                    │                          │
                    └→ no_findings ($0.00)     └→ back to audit (max 2)
```

The backward edge `critic → audit` is the only reason this is a graph rather than a `for` loop.
It returns to **retrieval**, not to drafting, because a thin finding is usually missing law rather
than bad writing.

---

## Nine decisions that look wrong and are not

Each of these was measured. Changing one without re-measuring will quietly degrade the system.

**1 · Retrieval queries are ten hardcoded templates, not model-generated.**
Rank of the correct clause out of 12,273, same facts:

| | rank |
|---|---|
| raw detector JSON | 11,268 |
| a narrative of events | 315 |
| **an obligation-shaped template** | **5** |

Rulebooks are written as duties (*"a Relevant Person **must**…"*), so a description of events
shares no register with them. Even a good human paraphrase loses: the natural rewording of the
concentration query scores 0.610 cosine and retrieves COBS noise, where the shipped wording scores
0.343 and lands the target clause in the top 8. `src/graph/prompts.py`

**2 · Detectors emit geometry, never a typology name.**
A candidate says `[dispersion]`, never `"structuring"`. Python may observe *"19 wires, CV 1.162"*;
only a retrieved clause may conclude an offence. Otherwise the system invents a label, retrieves
the clause matching its own invention, and cites it as independent authority. `src/utils/detectors.py`

**3 · Four shape primitives, not 17 typology rules.**
All 17 SAML-D suspicious typologies collapse onto concentration / dispersion / path / magnitude.

**4 · 100% recall at 32% precision is the intended trade.**
A missed launderer is a regulatory failure; a false alarm costs an analyst ~5 minutes. Detector
precision work is deliberately deferred.

**5 · Retrieved lists are merged by reciprocal rank fusion, not by distance.**
Distances are measured against each query's own vector and are **not comparable across queries** —
on June the seven queries' best hits span 0.3433 to 0.4827, so a distance sort ranks *how easy the
question was* above *how good the answer is*. RRF moved the clause June cites from rank 20 to 10.
Round-robin (23) and min-max normalisation (42) are both worse than the bug they replace.

**6 · The context holds 24 clauses, not the blueprint's 4.**
At 93 clauses the model cited *nothing* and the critic scored the draft 0.00 — retrieval was fine,
the noise underneath was the problem. But 4 is unsafe: tracking every clause the live reports
actually cited, the worst reached rank 17 even after reranking. 24 is the measured middle.

**7 · The citation veto is Python, and it is a veto, not a penalty.**
Every clause a draft cites is checked against what was retrieved. Absent ⇒ `score = 0.0`.
This is arithmetic, so it runs on every commit rather than being admired once — **and it has
fired in production**: the 2026-09-08 May run fabricated two citations, was vetoed, and shipped
with them recorded as reservations instead of as law. `nodes.critic_node`

**8 · Three report fields are recomputed in Python after the model returns them.**
`flagged_wires` (the model returned *account numbers* where wire references belong),
`source_document_hashes` (returned empty beside a live citation), and `risk_rating` (capped below
`HIGH_RISK_CONFIDENCE`). The model formats prose; it is not trusted with bookkeeping it can get
wrong silently. `nodes.generate_node`

**9 · Reports are never cached; retrieved clauses are.**
A report narrative names real accounts and amounts — June's carries 3 accounts and 11 figures — so
reusing one across batches would put the wrong identifiers into a regulatory filing. Clauses carry
no such risk. Hence semantic matching for clauses, exact-only for findings. `src/utils/cache.py`

---

## Known defects — do not treat these as done

**The risk rating does not reliably separate a clean batch from a dirty one.** Deferred by
decision, asserted as a failing test rather than described in prose
(`tests/eval_suite.py::test_a_clean_batch_is_not_reported_as_a_finding`).

On 2026-09-08 May *did* come back Low — but only because the citation veto forced confidence to
0.00. The correct answer arrived by way of a failure, not by calibration. Do not read that run as
the defect being fixed.

**Contextual Precision is the weak metric** — 0.547 on June. Retrieval ordering, confirmed
independently by the reranker experiment.

**`escalate()` has never run against a live model.** No generated batch contains a malformed
message; it is exercised only by stub.

---

## Where things live

| | |
|---|---|
| `src/utils/swift_parser.py` | MT103 → typed `Wire`. 880/880, exact |
| `src/utils/detectors.py` | four shape primitives → `Candidate` |
| `src/graph/{state,prompts,nodes,graph}.py` | the LangGraph agent |
| `src/graph/{cost,rerank,evalset}.py` | token accounting, cross-encoder, eval capture |
| `src/ingestion/` | corpus → 12,273 chunks in ChromaDB |
| `src/ui/cockpit.py` · `src/api/main.py` | the two front doors |
| `src/utils/cache.py` | §9.3 retrieval cache |

Deeper detail: [HLD.md](HLD.md) for structure, [LLD.md](LLD.md) for contracts and constants,
[TEST_DESIGN.md](TEST_DESIGN.md) for what is verified and how,
[TEST_RESULTS.md](TEST_RESULTS.md) for the latest measured numbers,
[CHANGELOG.md](CHANGELOG.md) for how it got here.

---

## House rules for changing this code

1. **Measure before you tune.** Every constant here came from a number. `MAX_CONTEXT_CLAUSES`,
   `RRF_K`, `MIN_CLUSTER_WIRES`, `HIGH_RISK_CONFIDENCE` all have their evidence in a comment.
2. **The free path must stay free.** `parse`, `detect`, `route_after_detect` and `audit` make no
   model calls. A batch with no candidates must cost exactly $0.00.
3. **`uv run pytest tests/` must need no API key and no network.** 233 tests, ~40s. Anything that
   breaks that has broken the suite's purpose, not just a test.
4. **Never widen what the model is trusted with.** If a field can be derived, derive it.
5. **New guards get a test that would fail without them**, not a comment saying they matter.
