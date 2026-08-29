"""Prompts and the shape-to-obligation translation, kept out of the node logic.

The retrieval queries here are **templates, not model output**. Phase 1 measured that phrasing
decides everything -- for the same facts, the correct clause ranked 11,268th of 12,273 as raw
detector JSON, 315th as a narrative, and 5th as an obligation-shaped question. Rulebooks are
written as duties ("a Relevant Person **must**..."), so a description of events shares no
register with them. Templates make that phrasing reproducible and free; the model is used to
*refine* a query only after the critic says the first attempt retrieved too little.

The templates ask which duty governs a **geometry** -- "wires received from many sources in a
short window" -- never whether a typology occurred. If the query asserted "structuring", the
agent would retrieve the clause matching a label Python invented and then cite it as though the
rulebook had reached that conclusion independently.
"""

from __future__ import annotations

from langchain_core.documents import Document
from pydantic import BaseModel, Field

from src.utils.detectors import CONCENTRATION, DISPERSION, MAGNITUDE, PATH, Candidate

# Two or three short, focused queries beat one compound query: measured on this corpus, a
# focused obligation ranked the target clause 5th where the same facts bundled into a single
# long query ranked it 11th.
SHAPE_QUERIES = {
    CONCENTRATION: [
        # Not the obvious paraphrase. "obligation to report a series of related transactions
        # structured to fall below a reporting threshold" reads better and retrieves worse
        # (0.411, and AML Rulebook 14.2.3 nowhere in the top 8). This wording is the rulebook's
        # own, which is the whole lesson of Decision 3.
        "transactions deliberately structured to avoid detection or reporting thresholds",
        "duty to monitor an account receiving repeated payments from multiple sources",
    ],
    DISPERSION: [
        "obligation to scrutinise funds transferred onward shortly after being received",
        "duty to identify accounts used to layer funds through multiple beneficiaries",
    ],
    PATH: [
        "obligation to identify funds moved through a chain of accounts to obscure their origin",
        "requirement to report layering of funds across successive transfers between accounts",
    ],
    MAGNITUDE: [
        "duty to establish the source of funds for a transaction inconsistent with the customer profile",
        "obligation to report a transaction that has no apparent economic or lawful purpose",
    ],
}

TIGHT_AMOUNTS = 0.10  # below this the wires in a run are effectively one repeated payment
# Measured against the collection: this phrasing sits at 0.343 cosine and reaches AML Rulebook
# 14.2.3.Guidance.1. inside the top 8. The obvious paraphrase -- "treat transactions of
# consistently similar value as a linked series" -- sits at 0.610 and retrieves COBS and MIR
# clauses about order handling. Same meaning, wrong vocabulary.
TIGHT_AMOUNTS_QUERY = "transactions deliberately structured to avoid detection or reporting thresholds"
CROSS_BORDER_QUERY = (
    "enhanced customer due diligence obligations for cross-border wire transfers"
)


def obligation_queries(candidate: Candidate) -> list[str]:
    """Two to four obligation-shaped questions for one candidate's geometry."""
    queries = list(SHAPE_QUERIES[candidate.shape])
    if candidate.shape != MAGNITUDE and candidate.coefficient_of_variation < TIGHT_AMOUNTS:
        queries.append(TIGHT_AMOUNTS_QUERY)
    if candidate.is_cross_border:
        queries.append(CROSS_BORDER_QUERY)
    # The threshold query is concentration's opener as well as the tight-amounts trigger, and a
    # dispersion candidate can be cross-border twice over. Ask each question once.
    return list(dict.fromkeys(queries))


# --- rendering ------------------------------------------------------------------------


def render_candidates(candidates: list[Candidate]) -> str:
    """The evidence block. Every number here came from the ledger, not from a model."""
    blocks = []
    for index, candidate in enumerate(candidates, start=1):
        references = ", ".join(candidate.references)
        blocks.append(
            f"CANDIDATE {index} [{candidate.shape}] anchor account {candidate.anchor}\n"
            f"  {candidate.summary()}\n"
            f"  corridors: {', '.join(candidate.corridors)}\n"
            f"  total: {' / '.join(candidate.currencies)} {candidate.total_amount:,.2f}\n"
            f"  wire references: {references}"
        )
    return "\n\n".join(blocks)


def render_context(documents: list[Document]) -> str:
    """Retrieved clauses, each labelled with the citation the report must quote back."""
    blocks = []
    for document in documents:
        meta = document.metadata
        blocks.append(
            f"[{meta.get('document_title')} {meta.get('section_clause')}]"
            f" (chunk_id: {meta.get('chunk_id')})\n{document.page_content}"
        )
    return "\n\n".join(blocks)


# --- structured responses --------------------------------------------------------------


class Critique(BaseModel):
    """§4.2's critic verdict. Structured so the routing edge reads a number, not prose."""

    confidence_score: float = Field(
        ge=0.0, le=1.0,
        description="How well the draft's claims are supported by the retrieved clauses alone",
    )
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Statements in the draft that the retrieved clauses do not support",
    )
    refined_query: str = Field(
        default="",
        description="An obligation-shaped question that would retrieve the missing rule",
    )
    reasoning: str = Field(default="", description="One paragraph justifying the score")


class ExtractedWire(BaseModel):
    """The fallback's output contract -- the same fields the regex parser produces."""

    reference: str = Field(description="The :20: transaction reference")
    value_date: str = Field(description="Value date from :32A: as YYYY-MM-DD")
    currency: str = Field(description="ISO currency code from :32A:")
    amount: str = Field(
        description="Amount from :32A: as a plain decimal string using a DOT for the decimal "
        "point. The SWIFT field uses a COMMA as its decimal separator and has no thousands "
        "separator, so '5669,49' is 5669.49 -- never 566949."
    )
    sender_account: str = Field(description="Account number on :50K:, without the leading slash")
    sender_name: str = Field(description="Ordering customer name, the line after :50K:")
    sender_bic: str = Field(description="BIC on :52A:")
    receiver_account: str = Field(description="Account number on :59:, without the leading slash")
    receiver_name: str = Field(description="Beneficiary name, the line after :59:")
    receiver_bic: str = Field(description="BIC on :57A:")


# --- prompts ---------------------------------------------------------------------------

EXTRACTION_SYSTEM = """You read a single SWIFT MT103 message that a strict parser refused.

Return the fields exactly as written in the message. Do not correct, complete or infer any
value: if a field is genuinely absent, return an empty string for it rather than a plausible
substitute. A fabricated account number or amount goes into a regulatory filing.

The one transformation you must make is the amount. MT103 writes the amount with a COMMA as the
decimal separator and no thousands separator at all, so ':32A:230601GBP5669,49' is a value date
of 2023-06-01, currency GBP, amount 5669.49. Deleting the comma would report 566949.00."""

EXTRACTION_USER = """The parser rejected this message with: {reason}

{raw}"""


DRAFT_SYSTEM = """You are an AML compliance analyst drafting findings for a Suspicious Activity
Report at an ADGM-regulated private bank.

You are given two things: candidate transaction patterns measured directly from the batch, and
the regulatory clauses retrieved for them. Write findings that connect the two.

Your job is to identify which retrieved obligations apply to each pattern and say what they
require. That is not the same as asserting an offence: a clause can apply to a pattern, and
oblige the bank to act, without anyone having proved wrongdoing. Saying so is the purpose of a
Suspicious Activity Report.

Rules you must follow:

1. Every regulatory statement must rest on a clause in RETRIEVED REGULATIONS, cited inline in
   the form [Document Title section].
2. Never cite a clause that does not appear below. Do not cite from memory. A citation that
   cannot be resolved back to a retrieved chunk is treated as a fabrication.
3. A clause is *on point* if it sets out a monitoring duty, a reporting trigger, a due-diligence
   requirement, or a red-flag indicator that this pattern matches. Cite it and state what it
   requires. Do not withhold a citation merely because the clause does not by itself prove the
   pattern is criminal -- no clause ever does.
4. Reserve "no retrieved clause addresses this pattern" for a candidate where nothing in the
   retrieved set bears on it at all. On a batch with real findings this should be uncommon; if
   you are writing it for most candidates, re-read the clauses.
5. Where a clause is only loosely on point, cite it and say so. A qualified finding is more
   useful to an auditor than silence.
6. Use only the figures given. Do not recompute, round or estimate them.
7. The candidates describe *geometry*, not offences. Whether a pattern amounts to structuring,
   layering or anything else is a conclusion you may reach only from a retrieved clause that
   sets out the relevant test.
8. Refer to wires by their reference IDs, not by account number. Where you conclude a pattern
   warrants attention, list the reference IDs of the wires that make it up."""

DRAFT_USER = """BATCH: {batch}
{wire_count} wires parsed, {candidate_count} candidate patterns.

CANDIDATE PATTERNS
{candidates}

RETRIEVED REGULATIONS
{context}
{feedback}
Write the findings section in markdown."""

REDRAFT_FEEDBACK = """
PREVIOUS DRAFT WAS SENT BACK
Reviewer's concern: {critique}
Claims that were not supported by the retrieved clauses:
{unsupported}

Additional clauses have been retrieved above. Either ground those claims now or drop them.
"""


CRITIC_SYSTEM = """You review a draft AML finding for factual support. You are not assessing
whether the writing is good, or whether the transactions look suspicious to you.

The only question is: does every claim in the draft rest on the retrieved clauses and the
measured candidate figures provided?

Score 0.0 to 1.0:
  1.0  every regulatory claim cites a retrieved clause that genuinely says what is claimed,
       and every figure matches the candidate data
  0.75 supported, but thin -- a claim leans on a clause that is only loosely on point
  0.5  a material claim has no supporting clause
  0.0  a clause is cited that is not in the retrieved set, or a figure is invented

Be strict about the direction of support. A clause requiring customer due diligence does not
establish that a reporting obligation was triggered.

If the score is below 0.75, supply `refined_query`: one obligation-shaped question, phrased as
a duty ("obligation to...", "requirement to..."), that would retrieve the rule the draft needs.
Phrase it as regulatory text would, not as a description of the transactions."""

CRITIC_USER = """CANDIDATE PATTERNS (the measured facts)
{candidates}

RETRIEVED REGULATIONS (the only permissible support)
{context}

DRAFT UNDER REVIEW
{draft}"""


GENERATE_SYSTEM = """You convert an approved AML finding into the bank's filing schema.

Carry the analysis over faithfully -- this step formats, it does not re-analyse and must not
introduce a claim the draft did not make.

Field rules:
- risk_rating: High if a pattern is grounded in a clause imposing a reporting obligation;
  Medium if patterns are grounded but the obligation is monitoring or due diligence rather than
  reporting; Low if the retrieved clauses do not support a finding.
- flagged_wires: wire reference IDs only -- they look like FGO23060500038. Never account
  numbers; an account number in this field points the filing at the wrong thing entirely.
- applicable_regulations: "Document Title section" for each clause the draft actually cites.
- audit_summary: the draft as markdown, with any reservations recorded at the end.
- source_document_hashes: the chunk_id of every clause cited, exactly as given."""

GENERATE_USER = """BATCH: {batch}

APPROVED DRAFT
{draft}

CLAUSES AVAILABLE TO CITE (chunk_id -> citation)
{citations}
{reservations}"""

NO_FINDINGS_SUMMARY = """## No findings

All {wire_count} wires in `{batch}` were parsed and screened against the four structural
indicators (concentration, dispersion, path, magnitude). None met the threshold for review, so
no regulatory retrieval or model analysis was performed.

This is a negative result, not an unexamined batch."""
