# FinGuard Orchestrator — as built

The companion to the design set. `FinGuard_PRD_v2.docx`, `FinGuard_HLD_v2.docx`,
`FinGuard_LLD_v2.docx` and `FinGuard_Eval_Design_v2.docx` say what the system should be; these markdown
files say what it is, and name every place the two differ. (The v1 documents sit beside them in
`docs/`; v2 superseded them and keeps every v1 boundary.)

| document | as-built companion |
|---|---|
| `FinGuard_PRD_v2.docx` · `FinGuard_HLD_v2.docx` | [HLD.md](HLD.md) |
| `FinGuard_LLD_v2.docx` | [LLD.md](LLD.md) · [CONSTANTS.md](CONSTANTS.md) |
| `FinGuard_Eval_Design_v2.docx` | [TEST_DESIGN.md](TEST_DESIGN.md) · [TEST_RESULTS.md](TEST_RESULTS.md) |
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
  ├─ graph_build ────── one directed money-flow graph per batch (v2).  no model
  │
  ├─ detect ─────────── nine detectors querying that graph, reconciled by precedence.  no model
  │                        structuring · fan_in · fan_out · cycle · scatter_gather
  │                        gather_scatter · deposit_send · layered_fan · bipartite
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
     report ─────────── templated assembly, each finding with its matched subgraph. no model call.
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
- **No KYC, no sanctions screening, no payment holds** (PRD v2 §2). Shape only.
- **Six SAML-D typologies left out on purpose** — Smurfing, Behaviour Change 1 & 2, Over-Invoicing,
  Cash Withdrawal, Single Large — each with the PRD's reason, and each kept in the ledgers as
  unflagged context so precision is measured against genuinely odd activity.

## Deviations from the design set

Recorded here because a deviation nobody wrote down becomes a defect somebody finds. v2's first, since
they are newest; v1's still hold.

### v2 · 1. The deposit-send note describes our parser, not SAML-D

LLD v2 §2.6 says SAML-D's payment type is a flat "cash" with no deposit/withdrawal split, and designs
deposit-send around that. SAML-D has 225,206 `Cash Deposit` and 300,477 `Cash Withdrawal` rows; the
split was being lost in `swift_parser`, which kept the first word of the payment type. Fixed there, and
the detector keys on a real deposit. The LLD's method — payment kind + direction + timing — is kept,
with one addition the data forced: **the amount must match.** Timing alone fires on 66% of clean
depositors in a SAML-D month; a send within 5% of the deposit, on about 4%.

### v2 · 2. Structuring groups by beneficiary as well as by originator

LLD §2.6 says "group by originator". Every one of SAML-D's 224 Structuring clusters hangs off the
*receiving* account — ten parties each paying it once — so originator grouping could never see one —
fan-in was what covered those clusters, under the wrong name.
§5324 reaches whoever causes the splitting, and FFIEC
Appendix F names deposits into one account by several people, so both sides are grouped and the
candidate records which. Held-out structuring: **5/15 → 15/15**.

### v2 · 3. Option 1 — applied as a data policy, with a held-out partition it turned out to need

PRD v2 §5.1: gold instances of the threshold-sensitive patterns are chosen to hug $10,000. As built,
a cluster qualifies when at least half its relevant amounts are in [$8,000, $10,000). Only 21 of 224
Structuring clusters in all of SAML-D do, so the eval corpus takes **every** aligned instance and the
dev corpus takes **none**.

Building that exposed a v1 defect: the "held-out" eval corpus was not held out. Both profiles slice the
same months and take the largest clusters first, so **all 84 planted rows of v1's dev corpus were also
planted in its eval corpus** — 3 of v1's 75 golden instances (1 cycle, 1 scatter-gather, 1 structuring)
were tuning clusters. v2 partitions SAML-D's clusters by a hash of the anchor account: a cluster is
tuning data or golden data, never both. Measured overlap now: **0 rows**.

### v2 · 4. Golden instances must fit in one batch — so two patterns have fewer than 15

SAML-D's structural clusters run 13–23 days, and a month boundary cuts them. Planted as-is, most
"Gather-Scatter" instances were only their scatter side — a fan-out with the wrong label, which PRD v2
§2.7 scopes out anyway. So gather-scatter, scatter-gather, layered and bipartite instances are planted
only when wholly inside a month. SAML-D then cannot supply fifteen held-out whole instances of two
patterns: **scatter_gather has 6, gather_scatter 12** (15 and 21 whole-month clusters exist in the
entire dataset, before the partition halves them). Confirmed with the project owner on 2026-09-29:
take what exists and report the denominator beside the number, rather than plant half-shapes or tune
on golden data. Deposit-send is exempt from whole-cluster — its hubs span ~250 days and each
deposit-then-send pair is a complete instance — and has its own minimum of 2 transactions.

### v2 · 5. `multi_hop_paths` returns levels, not paths

What the layered detector needs is which accounts feed the collectors that feed the root; path
enumeration is combinatorial on a dense graph and nothing consumes it. See [LLD.md](LLD.md) §2.5.

### v2 · 6. One Literal per shape family

`layered_fan` covers Layered_Fan_In/Out and `bipartite` covers plain and stacked — as the LLD v2 Literal
specifies, with direction and stacking in the attributes. Stated because PRD v2 lists them as
"layered fan-in/out" and "bipartite / stacked bipartite" and a reader could expect four values.

### v1 · Fifteen scenarios per pattern, not ten

Eval Design sizes `Labeled_Patterns` at ~10 per pattern; the golden set targets **fifteen** (confirmed
again for v2), because with a denominator of 15 a single miss is 6.7% rather than 10% against a 0.90
gate. v2 deviation 4 is where the data could not meet it.

### v1 · Journey 2 returns 202 + poll, with `?wait=true` as the synchronous variant

HLD §2.2's Journey 2 (repeated in HLD v2) says the report is *"returned directly in the API response"*.
LLD §5.1 step 1 says `POST /audits` → 202 + `job_id`. **The LLD wins**, because an audit runs per
candidate and a held connection is a timeout waiting for a proxy to find it. `?wait=true` gives Journey
2 its one round trip — the same queue and the same worker, held open for one caller, with a bounded
timeout that degrades to the `job_id` rather than hanging. The status code follows the answer: 200 with
a report, 202 with an id.

### v1 · `uv` and a lockfile, not `requirements.txt`

LLD §8 asks for pinned dependencies. The image installs from `uv.lock` with `uv sync --frozen`, which
pins the whole resolved graph *with hashes* rather than a flat list — a stricter answer to the same
requirement. A `requirements.txt` is deliberately **not** checked in, because a file that looks
authoritative while the image installs something else is worse than no file; it is generated on
demand for a scanner that wants one:

```bash
uv export --no-dev --format requirements-txt --no-emit-project > requirements.txt
```

Two smaller ones, recorded in [CONSTANTS.md](CONSTANTS.md): `structuring.band_fraction` is a fraction
where LLD §8 names an absolute `band`, because one absolute value cannot serve both the $10,000 and the
$3,000 threshold; and `gather_scatter`, `layered_fan` and `bipartite` carry their own windows (21, 28,
21 days), because SAML-D's structures run longer than the 14-day window the v1 detectors share.

And one addition rather than a deviation: the results store has a fourth table, `jobs`, which LLD
§3.2 does not list. §3.2 specifies the *results* tables and §5.1 step 1 says only "enqueue background
graph run" — but step 9 then has the client come back for `GET /audits/{job_id}`, and a queue that
lives in a process dict cannot answer that after a restart.

## Reading order

Start here, then:

- [HLD.md](HLD.md) — the zones, the three journeys, what runs in-environment and what leaves it.
- [LLD.md](LLD.md) — the node-by-node build, the error taxonomy, the schemas.
- [CONSTANTS.md](CONSTANTS.md) — every tunable number beside the measurement that chose it.
- [TEST_DESIGN.md](TEST_DESIGN.md) — the three tiers, the six golden corpora, the corpus partition, the CI gates.
- [TEST_RESULTS.md](TEST_RESULTS.md) — what it actually scores, including what fails.
- [CHANGELOG.md](CHANGELOG.md) — what changed and why, with the reversals kept.

The repository's `README.md` is the operational front door: how to run it, what each command does.
