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
    DATA_DIR,
    MANIFEST_PATH,
    OBLIQA_DOCS,
    SAML_D_CSV,
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
    pdfs = sorted((DATA_DIR / "raw" / "regulations").rglob("*.pdf"))
    assert len(pdfs) == 4

    for path in pdfs:
        text = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
        assert len(text) > 2000, f"{path.name} extracted almost no text"


def test_finra_notice_carries_the_red_flag_guidance():
    path = DATA_DIR / "raw" / "regulations" / "finra" / "regulatory-notice-19-18.pdf"
    text = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages).lower()
    assert "money laundering red flags" in text
    assert "3310" in text
