"""Verify the §3.2 corpora on disk match what the manifest claims.

These tests read acquired data rather than the network, so they double as a corruption
check: run them after any pull to confirm the ingestion layer has sound inputs.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest
from pypdf import PdfReader

from src.ingestion.download import (
    CFR_DIR,
    CFR_SECTIONS,
    DATA_DIR,
    FFIEC_DOCS,
    MANIFEST_PATH,
    OBLIQA_DOCS,
    RAW_DIR,
    REGULATORY_PDFS,
    SAML_D_CSV,
    US_STATUTES,
    classification_problems,
    classification_table,
    sha256_file,
)
from src.ingestion.obliqa_map import MAP_PATH, load_document_map

pytestmark = pytest.mark.skipif(
    not MANIFEST_PATH.exists(),
    reason="datasets not acquired -- run: uv run python -m src.ingestion.download",
)

SAML_D_COLUMNS = [
    "Time",
    "Date",
    "Sender_account",
    "Receiver_account",
    "Amount",
    "Payment_currency",
    "Received_currency",
    "Sender_bank_location",
    "Receiver_bank_location",
    "Payment_type",
    "Is_laundering",
    "Laundering_type",
]
SAML_D_ROWS = 9_504_852
OBLIQA_DOCUMENTS = 40
OBLIQA_PASSAGES = 13_732


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text())


def test_every_manifest_artifact_is_present_and_unmodified(manifest):
    for relpath, entry in manifest["artifacts"].items():
        path = DATA_DIR / relpath
        assert path.exists(), f"{relpath} is missing"
        assert path.stat().st_size == entry["bytes"], f"{relpath} changed size"
        assert sha256_file(path) == entry["sha256"], f"{relpath} content changed"


def test_manifest_records_provenance_for_every_artifact(manifest):
    for relpath, entry in manifest["artifacts"].items():
        assert entry["url"].startswith("https://"), relpath
        assert entry["licence"], relpath
        assert entry["retrieved"], relpath
    assert "saml_d" in manifest["citations"], "SAML-D is CC BY-NC-SA; the citation is required"


# --- A. transaction ledger -----------------------------------------------------------


def test_saml_d_has_the_expected_shape():
    header = pd.read_csv(SAML_D_CSV, nrows=5)
    assert list(header.columns) == SAML_D_COLUMNS

    rows = sum(len(chunk) for chunk in pd.read_csv(SAML_D_CSV, usecols=["Amount"], chunksize=2_000_000))
    assert rows == SAML_D_ROWS


def test_saml_d_labels_cover_the_documented_typologies():
    types = set()
    laundering = 0
    for chunk in pd.read_csv(
        SAML_D_CSV, usecols=["Is_laundering", "Laundering_type"], chunksize=2_000_000
    ):
        types.update(chunk.Laundering_type.unique())
        laundering += int(chunk.Is_laundering.sum())

    assert len(types) == 28, "SAML-D documents 28 typologies (11 normal / 17 suspicious)"
    assert {"Structuring", "Smurfing", "Deposit-Send", "Cycle"} <= types
    # Heavily imbalanced by design -- the reason the generator selects clusters, not rows.
    assert 0.0005 < laundering / SAML_D_ROWS < 0.005


# --- B. regulatory knowledge base ----------------------------------------------------


def test_obliqa_extracted_forty_documents_without_resource_forks():
    files = list(OBLIQA_DOCS.glob("*.json"))
    assert len(files) == OBLIQA_DOCUMENTS
    assert not list(OBLIQA_DOCS.parent.rglob("__MACOSX"))

    passages = sum(len(json.loads(path.read_text())) for path in files)
    assert passages == OBLIQA_PASSAGES


def test_obliqa_passages_carry_the_fields_the_loader_needs():
    passages = json.loads((OBLIQA_DOCS / "1.json").read_text())
    assert {"ID", "DocumentID", "PassageID", "Passage"} == set(passages[0])
    # ~16% of passages are empty strings or bare headings; §3.4 must filter them.
    usable = [p for p in passages if len((p["Passage"] or "").strip()) >= 40]
    assert 0 < len(usable) < len(passages)


def test_document_map_is_complete_and_injective():
    assert MAP_PATH.exists(), "run: uv run python -m src.ingestion.obliqa_map"
    documents = load_document_map()
    assert len(documents) == OBLIQA_DOCUMENTS

    sources = [entry["source_file"] for entry in documents.values()]
    assert len(set(sources)) == OBLIQA_DOCUMENTS, "two DocumentIDs claimed the same file"
    assert sum(entry["passages"] for entry in documents.values()) == OBLIQA_PASSAGES


def test_document_one_is_the_adgm_aml_rulebook():
    """The single most citation-relevant document in the corpus -- pin it explicitly."""
    entry = load_document_map()[1]
    assert entry["title"] == "AML Rulebook"
    assert entry["source_file"].startswith("AML_")

    passages = json.loads((OBLIQA_DOCS / "1.json").read_text())
    joined = " ".join(p["Passage"] or "" for p in passages)
    assert "money laundering" in joined.lower()


@pytest.mark.parametrize(
    ("number", "marker"),
    [("3310", "written anti-money laundering program"), ("3110", "system to supervise the activities")],
)
def test_finra_rule_text_is_operative_language_not_page_chrome(number, marker):
    text = (DATA_DIR / "raw" / "regulations" / "finra" / f"finra-rule-{number}.txt").read_text()
    assert marker in text
    assert f"finra-rules/{number}" in text, "provenance header is missing"
    for chrome in ("block-plugin-id", "field--name", "<div", "Skip to main content"):
        assert chrome not in text


def test_regulatory_pdfs_yield_extractable_text():
    """A PDF that extracts no text is a scanned image, and it would be embedded as an empty
    chunk -- indexed, retrievable, and useless. The count is derived from the declarations
    rather than written as a literal: it was `== 4` before the migration, and Phase 1a's nine
    FFIEC documents turned a real assertion into a failing arithmetic check."""
    pdfs = sorted((DATA_DIR / "raw" / "regulations").rglob("*.pdf"))
    expected = len(REGULATORY_PDFS) + len(FFIEC_DOCS)
    assert len(pdfs) == expected, f"{len(pdfs)} PDFs on disk, {expected} declared"

    for path in pdfs:
        text = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
        # The FFIEC sections run from one to seventeen pages, so the floor is per-page rather
        # than a flat figure a two-page appendix could never clear.
        floor = 400 * len(PdfReader(path).pages)
        assert len(text) > min(2000, floor), f"{path.name} extracted almost no text"


def test_finra_notice_carries_the_red_flag_guidance():
    path = DATA_DIR / "raw" / "regulations" / "finra" / "regulatory-notice-19-18.pdf"
    text = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages).lower()
    assert "money laundering red flags" in text
    assert "3310" in text


# =============================================================================================
# The US corpus added by the design migration (Phase 1a)
#
# Before the migration this repository had no US statute and no US regulation in it at all --
# the corpus was ADGM law plus US *guidance*. These are the artifacts a finding now rests on, so
# their tests are about identity rather than integrity: the hash check above already proves the
# bytes are unchanged, and what these prove is that the bytes are the right document.
# =============================================================================================

def test_every_tier_one_obligation_is_on_disk():
    """Without these, pattern_to_obligations has nothing to resolve against and grounding is
    impossible for every typology at once."""
    for statute in US_STATUTES:
        path = RAW_DIR / "regulations" / "usc" / f"{statute.source_id}.txt"
        assert path.is_file(), f"missing statute {statute.source_id}"
    for section in CFR_SECTIONS:
        assert (CFR_DIR / f"{section.source_id}.xml").is_file(), f"missing {section.source_id}"


def test_the_structuring_statute_is_the_statute_and_not_an_error_page():
    """govinfo answers a missing granule with its "Page Not Found" page at HTTP 200, so
    raise_for_status() cannot tell a statute from an error page. Asking for the 2025 edition
    returns 44 KB of Drupal markup that a naive fetch would store as the structuring
    prohibition. Only the text distinguishes them."""
    text = (RAW_DIR / "regulations" / "usc" / "31usc5324.txt").read_text()
    assert "structure or assist in structuring" in text
    assert "Structuring transactions to evade reporting requirement" in text
    assert "United States Code, 2024 Edition" in text
    assert "Page Not Found" not in text
    # The provenance header is what makes the artifact reproducible from the file alone.
    assert text.startswith("# 31 U.S.C. § 5324.")
    assert "# Source: https://www.govinfo.gov/" in text


@pytest.mark.parametrize(
    "source_id,marker",
    [(section.source_id, section.marker) for section in CFR_SECTIONS],
)
def test_each_cfr_section_carries_its_operative_language(source_id, marker):
    xml = (CFR_DIR / f"{source_id}.xml").read_text()
    assert marker in xml, f"{source_id} no longer contains {marker!r}"


def test_cfr_sections_are_stored_as_structured_xml():
    """Stored as eCFR XML rather than flattened text on purpose: DIV8/HEAD/P carries the section
    and paragraph structure that the tier-aware chunker needs to emit a real section_ref.
    Flattening at acquisition would discard the structure the chunker exists to use."""
    xml = (CFR_DIR / "31cfr1020.320.xml").read_text()
    assert xml.lstrip().startswith("<?xml")
    assert '<DIV8 N="1020.320" TYPE="SECTION"' in xml
    assert "<HEAD>" in xml and xml.count("<P>") > 5


def test_the_two_monitored_thresholds_each_have_an_obligation_behind_them():
    """config.yaml monitors $10,000 and $3,000. A threshold the system enforces with no rule
    behind it is a finding that cannot cite anything, so each one has to trace to a section."""
    ctr = (CFR_DIR / "31cfr1010.311.xml").read_text()
    assert "more than $10,000" in ctr

    recordkeeping = (CFR_DIR / "31cfr1010.410.xml").read_text()
    assert "$3,000 or more" in recordkeeping


# --- FFIEC ---------------------------------------------------------------------------------
@pytest.mark.parametrize("doc", FFIEC_DOCS, ids=lambda doc: doc.source_id)
def test_each_ffiec_document_is_the_one_we_asked_for(doc):
    """The appendix files are numbered and the numbers are offset from the letters: 05.pdf is
    Appendix D, 06.pdf is Appendix E, and Appendix F -- the red-flag list this system wants --
    is 07.pdf. Nothing on the page says so and the anchors carry no text, so "Appendix F must be
    06.pdf" silently yields a two-page list of international organizations. Both are valid PDFs.
    Only page one distinguishes them."""
    if not doc.dest.is_file():
        pytest.skip(f"{doc.source_id} not acquired (FFIEC blocks some clients)")
    first_page = PdfReader(doc.dest).pages[0].extract_text() or ""
    assert doc.marker in " ".join(first_page.split())


def test_appendix_f_is_the_red_flag_list():
    """The single most valuable Tier-2 artifact: with FINRA 19-18 it is the indicator corpus."""
    path = next(d.dest for d in FFIEC_DOCS if d.source_id == "ffiec-appendix-f")
    if not path.is_file():
        pytest.skip("FFIEC Appendix F not acquired")
    reader = PdfReader(path)
    assert len(reader.pages) >= 8, "Appendix F is a 10-page list; a shorter file is a wrong one"
    text = " ".join(" ".join((page.extract_text() or "").split()) for page in reader.pages)
    assert "Red Flags" in text
    # Phase 1b splits this one bullet per chunk, so the bullets have to actually be there.
    assert text.count("•") > 40, f"only {text.count('•')} bullets found"
    # Headed sections are what give each indicator a section_ref.
    assert "Efforts to Avoid Reporting or Recordkeeping Requirement" in text


# --- classification ------------------------------------------------------------------------
def test_every_artifact_declares_what_it_is(manifest):
    """Checked at acquisition rather than discovered in Phase 1b, because each omission fails
    late and quietly: a missing source_id breaks the (source_id, section_ref) pairs that
    pattern_to_obligations is curated as, and a missing authority silently drops a chunk out of
    the Tier-2 indicator pool -- which reads as a retrieval quality problem, not a metadata bug."""
    assert classification_problems(manifest) == []


def test_the_classification_table_covers_every_declared_source():
    """A source declared above but absent from the table would be fetched and then never
    classified, so it would be acquired and unusable."""
    table = classification_table()
    for statute in US_STATUTES:
        assert f"raw/regulations/usc/{statute.source_id}.txt" in table
    for section in CFR_SECTIONS:
        assert f"raw/regulations/cfr/{section.source_id}.xml" in table
    for doc in FFIEC_DOCS:
        assert f"raw/regulations/ffiec/{doc.source_id}.pdf" in table


def test_nothing_outside_us_jurisdiction_is_citable(manifest):
    """The invariant the whole two-collection split exists to protect. ADGM material stays on
    disk for the retrieval benchmark, and this is what keeps it out of the citable corpus."""
    for relpath, entry in manifest["artifacts"].items():
        if entry.get("role") in ("obligation", "indicator"):
            assert entry.get("jurisdiction") == "US", f"{relpath} is citable but not US"
        if entry.get("source_id", "").startswith("obliqa"):
            assert entry["role"] == "benchmark", f"{relpath} must be benchmark-only"
            assert entry.get("jurisdiction") == "ADGM"


def test_obligations_are_binding_and_indicators_are_reachable(manifest):
    """Tier 1 is fetched by curated id and must be binding law. Tier 2 is fetched by semantic
    search filtered on authority == illustrative, so an indicator recorded as binding is indexed
    and then never retrieved."""
    obligations = [e for e in manifest["artifacts"].values() if e.get("role") == "obligation"]
    assert obligations, "no binding obligations in the manifest"
    assert all(entry["authority"] == "binding" for entry in obligations)
    assert all(entry["tier"] in ("statute", "regulation") for entry in obligations)

    illustrative = [
        entry for entry in manifest["artifacts"].values()
        if entry.get("role") == "indicator" and entry.get("authority") == "illustrative"
    ]
    assert len(illustrative) >= 4, "the Tier-2 indicator pool would be nearly empty"


def test_the_finra_rules_are_recorded_as_binding_on_broker_dealers(manifest):
    """They are binding -- on FINRA members. The audited institution here is a bank, so they are
    never mapped in pattern_to_obligations and can never ground a finding. Recording them as
    illustrative would have been simpler and false; they stay indexed as near-miss distractors,
    which is what makes Context Precision measurable."""
    for number in ("3310", "3110"):
        entry = manifest["artifacts"][f"raw/regulations/finra/finra-rule-{number}.txt"]
        assert entry["authority"] == "binding"
        assert entry["applies_to"] == "broker_dealer"


def test_cfr_sections_record_when_they_were_last_amended(manifest):
    """The effective_date a citation carries. It matters for a reason specific to this system: a
    report defended in an audit years later has to show the rule as it stood, and "when we
    downloaded it" is not that date."""
    for section in CFR_SECTIONS:
        entry = manifest["artifacts"][f"raw/regulations/cfr/{section.source_id}.xml"]
        assert entry.get("effective_date"), f"{section.source_id} has no amendment date"
        assert entry["effective_date"][:2] == "20"
        assert entry["version"].startswith("eCFR ")
