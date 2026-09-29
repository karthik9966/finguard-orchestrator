# FinGuard Orchestrator — as built

The companion to the design set. `AML- PRD.docx`, `FinGuard_HLD.docx`, `FinGuard_LLD.docx` and
`FinGuard_Eval_Design.docx` say what the system should be; these markdown files say what it is, and
name every place the two differ.

| document | as-built companion |
|---|---|
| `AML- PRD.docx` · `FinGuard_HLD.docx` | [HLD.md](HLD.md) |
| `FinGuard_LLD.docx` | [LLD.md](LLD.md) · [CONSTANTS.md](CONSTANTS.md) |
| `FinGuard_Eval_Design.docx` | [TEST_DESIGN.md](TEST_DESIGN.md) · [TEST_RESULTS.md](TEST_RESULTS.md) |
| — | [CHANGELOG.md](CHANGELOG.md) |

---

## What the system does

It audits a month of transactions against US AML regulation and produces a report an examiner could
check. One batch in, one `ComplianceReport` out, with every finding carrying the clause it rests on.

```
batch (MT103 pdf/txt)
  │
  ├─ ingest ─────────── deterministic parse → TransactionRecord; one light-model rescue per
  │                     refused message; anything still unreadable is quarantined, never guessed
  │
  ├─ detect ─────────── five windowed typology detectors, reconciled by precedence.  no model
  │                        structuring · fan_in · fan_out · cycle · scatter_gather
  │                     no candidates → clean report, $0.0000, the model is never constructed
  │
  └─ for each candidate:
        retrieve ────── Tier 1 obligations **by curated id** (never searched)
        │              Tier 2 indicators by `authority: illustrative` search + cross-encoder rerank
        ground ──────── reasoning model → DraftFinding, against redacted context
        critique ────── deterministic faithfulness gate FIRST, then the model judge
        │                 pass → Finding(pending_review)
        │                 thin → refinement hint → back to retrieve  (bounded)
        │                 exhausted → Finding(needs_review), with reasons
        ▼
     report ─────────── templated assembly. no model call.
        │
     store ──────────── immutable report_json + mutable findings.status + append-only reviews
```

The two halves that matter are on opposite sides of the model. **Detection is arithmetic** — it finds
shapes and names none of them suspicious. **Grounding is judgement** — it is shown the shape and the
law, and it may only conclude what the law it was shown supports. The faithfulness gate is what makes
the second half checkable: a finding may cite only clauses that were actually retrieved, enforced by
a subset test in Python before any model is asked its opinion.

## What it is not

Out of scope, per HLD §1.1 and enforced rather than merely stated:

- **No real-time feeds.** The unit of work is a monthly batch.
- **No cross-month memory.** Each batch is audited alone, which is also what bounds every detector's
  window by construction rather than by a check.
- **No non-US law.** `RuleChunk` rejects a non-US jurisdiction *on the model*, so an ADGM clause
  cannot reach a citation even if a metadata filter is later written wrongly. ObliQA's 12,122 ADGM
  chunks live in a separate collection, used only to keep the retrieval benchmark reproducible.
- **No automated filing.** The system produces a report and a human decides.
- **No multi-tenancy.** One deployment audits one institution.

## The four documented deviations from the design set

Recorded here because a deviation nobody wrote down becomes a defect somebody finds.

### 1. Five detectors, not four

The PRD names five in-scope typologies — `structuring`, `fan_in`, `fan_out`, `cycle`,
`scatter_gather` — where earlier drafts described four geometric primitives
(concentration/dispersion/path/magnitude). The five are what shipped, and the difference is not
cosmetic: `pattern_to_obligations` is keyed on typology, so a candidate must carry a named typology
to be grounded at all. The old primitives emitted geometry and left naming to retrieval, which is
why three of the five in-scope detectors had nothing to find when the ledgers were first regenerated.

### 2. Fifteen scenarios per pattern, not ten

Eval Design §3 sizes `Labeled_Patterns` at "~40 (≈10 / pattern)". The golden set holds **75, fifteen
per pattern**, for one reason: with a denominator of 15 a single miss is 6.7% rather than 10%, and
recall is being compared against a 0.90 gate. Eval Design anticipates this — *"they can be scaled up
for a more convincing recall number without changing the design"* — and getting to fifteen per
pattern required three changes to the ledger generator, described in [TEST_DESIGN.md](TEST_DESIGN.md).

### 3. Journey 2 returns 202 + poll, with `?wait=true` as the synchronous variant

HLD §2.2's Journey 2 says the report is *"returned directly in the API response"*. LLD §5.1 step 1
says `POST /audits` → 202 + `job_id`. **The LLD wins**, because an audit runs per candidate and a
held connection is a timeout waiting for a proxy to find it. `?wait=true` gives Journey 2 its one
round trip — the same queue and the same worker, held open for one caller, with a bounded timeout
that degrades to the `job_id` rather than hanging. The status code follows the answer: 200 with a
report, 202 with an id.

### 4. `uv` and a lockfile, not `requirements.txt`

LLD §8 asks for pinned dependencies. The image installs from `uv.lock` with `uv sync --frozen`, which
pins the whole resolved graph *with hashes* rather than a flat list — a stricter answer to the same
requirement. A `requirements.txt` is deliberately **not** checked in, because a file that looks
authoritative while the image installs something else is worse than no file; it is generated on
demand for a scanner that wants one:

```bash
uv export --no-dev --format requirements-txt --no-emit-project > requirements.txt   # 3,174 hashes
```

There is a fifth, smaller one, recorded in [CONSTANTS.md](CONSTANTS.md): `structuring.band_fraction`
is a fraction where LLD §8 names an absolute `band`, because one absolute value cannot serve both the
$10,000 and the $3,000 threshold sensibly.

And one addition rather than a deviation: the results store has a fourth table, `jobs`, which LLD
§3.2 does not list. §3.2 specifies the *results* tables and §5.1 step 1 says only "enqueue background
graph run" — but step 9 then has the client come back for `GET /audits/{job_id}`, and a queue that
lives in a process dict cannot answer that after a restart.

## Reading order

Start here, then:

- [HLD.md](HLD.md) — the zones, the three journeys, what runs in-environment and what leaves it.
- [LLD.md](LLD.md) — the node-by-node build, the error taxonomy, the schemas.
- [CONSTANTS.md](CONSTANTS.md) — every tunable number beside the measurement that chose it.
- [TEST_DESIGN.md](TEST_DESIGN.md) — the three tiers, the six golden corpora, the two CI gates.
- [TEST_RESULTS.md](TEST_RESULTS.md) — what it actually scores, including what fails.
- [CHANGELOG.md](CHANGELOG.md) — what changed and why, with the reversals kept.

The repository's `README.md` is the operational front door: how to run it, what each command does.
