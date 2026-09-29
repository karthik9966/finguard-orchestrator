"""The cockpit page itself, driven through Streamlit's AppTest.

The client's behaviour is covered in `test_ui_client.py`; what is checked here is the *page* --
that it renders, that it says something useful when the service is not there, and that the review
buttons reach the API and move a status.

`FinGuardClient._session` is patched at class level to borrow FastAPI's TestClient, so the page runs
against the real endpoints. Patching the class rather than the module is what makes it work: AppTest
re-executes `cockpit.py` on every run, so anything patched on that module is thrown away.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from src.api import main
from src.models import Candidate, Citation, ComplianceReport, Finding, QuarantinedMessage, RuleChunk
from src.models import ValidationReport
from src.store import SqlResultsStore
from src.ui.client import FinGuardClient

COCKPIT = Path(__file__).resolve().parents[1] / "src" / "ui" / "cockpit.py"
TOKEN = "test-token-not-a-real-secret"

CLAUSE = RuleChunk(
    chunk_id="31cfr1020.320:d000653bf144c90b",
    text="A bank shall file a report of any suspicious transaction relevant to a possible "
         "violation of law or regulation.",
    tier="regulation", authority="binding",
    source_id="31cfr1020.320", section_ref="§ 1020.320(a)",
)


def a_report(run="run-page", *, status="pending_review", quarantined=0) -> ComplianceReport:
    target = Candidate(
        candidate_id="structuring:acct-a:0000000000",
        pattern_type="structuring",
        member_txn_refs=["FGO23060100001", "FGO23060100002"],
        detection_confidence=0.6,
    )
    finding = Finding(
        finding_id=f"f-{run}:{target.candidate_id}",
        candidate=target,
        risk_level="medium",
        narrative="Two transfers just below the $10,000 reporting threshold within eleven days.",
        applicable_regulations=[Citation.from_chunk(CLAUSE)],
        confidence=0.9,
        status=status,
        review_notes=["the evidence was thin"] if status == "needs_review" else [],
    )
    return ComplianceReport(
        report_id=f"rep-{run}", run_id=run, period="2023-06",
        generated_at=datetime(2026, 9, 29, tzinfo=timezone.utc),
        risk_rating="medium", findings=[finding],
        flagged_transactions=list(target.member_txn_refs),
        summary="## Compliance review -- 2023-06\n\n500 transaction(s) screened.",
        source_document_refs=[Citation.from_chunk(CLAUSE)],
        quarantined_count=quarantined,
    )


VALIDATION = ValidationReport(
    batch="2023-06.txt", declared=501, parsed=500, rescued=2,
    quarantined=[QuarantinedMessage(
        ordinal=317, reference="FGO23060100317", reason="missing required tag :32A:",
        raw=":20:FGO23060100317\n:23B:CRED", fallback_attempted=True,
    )],
)


@pytest.fixture
def served(monkeypatch, tmp_path):
    """The page wired to a live app: real endpoints, real store, no server and no network."""
    store = SqlResultsStore(f"sqlite:///{tmp_path / 'results.db'}")
    monkeypatch.setattr(main, "_STORE", store)
    monkeypatch.setattr(main, "counts", lambda: {
        "total": 731, "tier": {"statute": 12, "regulation": 219, "guidance": 500},
        "authority": {"binding": 231, "illustrative": 500},
    })
    monkeypatch.setenv("API_AUTH_TOKEN", TOKEN)
    monkeypatch.setenv("API_BASE_URL", "http://testserver")
    from src.config import reset_caches

    reset_caches()

    with TestClient(main.app) as session:
        @contextmanager
        def borrow(self):
            yield session

        monkeypatch.setattr(FinGuardClient, "_session", borrow)
        yield store
    reset_caches()


def run_page(**session_state) -> AppTest:
    page = AppTest.from_file(str(COCKPIT), default_timeout=60)
    for key, value in session_state.items():
        page.session_state[key] = value
    return page.run()


# --- the first-run experience -------------------------------------------------------------


def test_the_page_says_what_to_do_when_the_api_is_not_running(monkeypatch):
    """The commonest first experience, and a stack trace is a terrible answer to it."""
    monkeypatch.setenv("API_AUTH_TOKEN", TOKEN)
    monkeypatch.setenv("API_BASE_URL", "http://127.0.0.1:9")
    from src.config import reset_caches

    reset_caches()
    page = run_page()
    assert not page.exception
    assert any("not reachable" in error.value for error in page.error)
    assert any("uvicorn" in block.value for block in page.code)
    reset_caches()


def test_the_page_refuses_to_run_without_a_token(monkeypatch):
    monkeypatch.setenv("API_AUTH_TOKEN", "")
    from src.config import reset_caches

    reset_caches()
    page = run_page()
    assert not page.exception
    assert any("API_AUTH_TOKEN" in error.value for error in page.error)
    reset_caches()


# --- §6.1 the ingestion gate --------------------------------------------------------------


def test_the_sidebar_reports_the_corpus_it_would_cite_from(served):
    page = run_page()
    assert not page.exception
    metrics = {metric.label: metric.value for metric in page.metric}
    assert metrics["Chunks"] == "731" and metrics["Binding"] == "231"


def test_with_nothing_selected_the_page_asks_for_a_batch(served):
    page = run_page()
    assert any("Upload an MT103" in info.value for info in page.info)


# --- §6.3 / §6.4 the report and the review loop -------------------------------------------


def test_a_stored_report_renders_with_its_findings_and_citations(served):
    served.save(a_report(), validation=VALIDATION)
    page = run_page(report_id="rep-run-page")
    assert not page.exception

    body = " ".join(block.value for block in page.markdown)
    assert "Compliance review" in body
    assert "just below the $10,000 reporting threshold" in body, "the narrative is on the page"
    assert "§ 1020.320(a)" in body, "the clause it was drafted against is shown with it"


def test_the_review_buttons_offered_are_the_ones_the_store_permits(served):
    """A button that can only fail is worse than no button, so the page mirrors the transitions."""
    served.save(a_report(status="pending_review"))
    labels = {button.label for button in run_page(report_id="rep-run-page").button}
    assert {"Clear", "Escalate"} <= labels
    assert "Approve for filing" not in labels

    served.save(a_report(run="run-esc", status="escalated"))
    labels = {button.label for button in run_page(report_id="rep-run-esc").button}
    assert "Approve for filing" in labels


def test_escalating_from_the_page_moves_the_status(served):
    """§6.4 end to end: the button, the API, the store, and the page rendering the new state."""
    served.save(a_report())
    page = run_page(report_id="rep-run-page", reviewer="analyst@bank")
    next(button for button in page.button if button.label == "Escalate").click().run()

    finding_id = a_report().findings[0].finding_id
    assert served.status_of(finding_id) == "escalated"
    assert [entry.action for entry in served.reviews_for(finding_id)] == ["escalate"]


def test_a_review_without_a_reviewer_id_is_refused_before_it_is_recorded(served):
    """Every review is attributed, permanently. An unattributed one is worse than none."""
    served.save(a_report())
    page = run_page(report_id="rep-run-page", reviewer="")
    next(button for button in page.button if button.label == "Escalate").click().run()

    assert served.status_of(a_report().findings[0].finding_id) == "pending_review"
    assert any("reviewer id" in error.value for error in page.error)


def test_a_finding_the_engine_could_not_ground_says_why_on_the_page(served):
    served.save(a_report(status="needs_review"))
    page = run_page(report_id="rep-run-page")
    assert any("the evidence was thin" in warning.value for warning in page.warning)


# --- §6.5 the ingestion record ------------------------------------------------------------


def test_the_quarantine_panel_names_the_messages_that_were_lost(served):
    """A month whose report is clean because a message failed to parse is not a clean month."""
    served.save(a_report(quarantined=1), validation=VALIDATION)
    page = run_page(report_id="rep-run-page")

    assert any("could not be parsed" in error.value for error in page.error)
    metrics = {metric.label: metric.value for metric in page.metric}
    assert metrics["Parsed"] == "500" and metrics["Quarantined"] == "1"
    assert metrics["Rescued by the fallback"] == "2"


def test_a_report_with_no_ingestion_record_does_not_pretend_to_have_one(served):
    served.save(a_report())
    page = run_page(report_id="rep-run-page")
    assert any("No ingestion record" in caption.value for caption in page.caption)
