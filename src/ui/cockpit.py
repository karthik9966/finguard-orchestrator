"""The auditor cockpit -- LLD §6's five sections, over the API.

Everything the engine does is measurable from the CLI. This page exists because a compliance analyst
is not going to read a terminal, and because the facts that make the output trustworthy are
invisible in a JSON dump: *which clause justified each finding*, *which findings the review could
not stand behind*, and *what the batch did not even contain*.

It talks to the service over HTTP (`src/ui/client.py`) and imports nothing from the graph. Three
reasons, each a thing that would otherwise be found in production: one execution path instead of
two, the API's single worker instead of two analysts' concurrent audits contending for one vector
store, and no way for a Streamlit rerun to bill money.

Run the API first, then::

    uv run streamlit run src/ui/cockpit.py

Both need `API_AUTH_TOKEN`; the page says so plainly if it is missing.
"""

from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from src.config import get_settings
from src.detection import evidence
from src.ui.client import ApiError, FinGuardClient

st.set_page_config(page_title="FinGuard — Auditor Cockpit", page_icon="⚖️", layout="wide")

# Streamlit re-executes this whole file on every widget interaction. The job id is what survives in
# session state -- not the report, which is fetched fresh so a review recorded a moment ago is
# visible immediately. Nothing here can start an audit except the button.
JOB_KEY = "job_id"
REVIEWER_KEY = "reviewer"

# The upload never touches disk in this process. What it used to do -- write into
# `site-packages/.finguard_uploads/` and never clean up -- put run data inside the installed
# environment, survived every restart, and grew without bound. The bytes now go straight to the API,
# and the only temporary file is the API's own, which it deletes when the run ends.
POLL_SECONDS = 1.0

RISK_STYLE = {"high": st.error, "medium": st.warning, "low": st.success, "none": st.success}
STATUS_ICON = {
    "pending_review": "🔍", "needs_review": "⚠️", "cleared": "✅",
    "escalated": "🚩", "approved": "📝",
}
# What an analyst may do from each state, mirroring the store's own transition table. Shown rather
# than enforced here -- the store is the authority and answers 409 -- but a button that can only
# fail is worse than no button.
ACTIONS = {
    "pending_review": [("clear", "Clear"), ("escalate", "Escalate")],
    "needs_review": [("clear", "Clear"), ("escalate", "Escalate")],
    "escalated": [("approve", "Approve for filing"), ("clear", "Clear")],
    "cleared": [],
    "approved": [],
}


@st.cache_resource
def api() -> FinGuardClient:
    return FinGuardClient()


def fail(error: ApiError) -> None:
    """Say what an analyst can actually do about it."""
    if error.status == 0:
        st.error("The FinGuard API is not reachable.")
        st.code("uv run uvicorn src.api.main:app --reload", language="bash")
    elif error.status == 401:
        st.error("The API rejected this token. Set `API_AUTH_TOKEN` to the service's own value.")
    elif error.status == 503:
        st.error(f"The service is not ready: {error.detail}")
    else:
        st.error(f"{error.status}: {error.detail}")


# --- §6.1 the ingestion gate --------------------------------------------------------------

with st.sidebar:
    st.header("Ingestion gate")

    if not get_settings().api_auth_token:
        st.error("`API_AUTH_TOKEN` is not set, so every call will be refused.")
        st.code("openssl rand -hex 32", language="bash")
        st.stop()

    try:
        health = api().health()
    except ApiError as error:
        fail(error)
        st.stop()

    left, right = st.columns(2)
    left.metric("Chunks", f"{health['vectors']:,}")
    right.metric("Binding", f"{health['by_authority'].get('binding', 0):,}")
    st.caption(f"`{health['collection']}` · US BSA/AML corpus · queue {health['queue_depth']}")

    with st.expander("Corpus composition"):
        # §6.1: a citation is only checkable if you know which corpus stands behind it. Tier and
        # authority are different claims -- tier says what kind of document the text came from,
        # authority says whether it *binds* -- so they are shown separately.
        st.dataframe(
            pd.DataFrame([
                {"tier": tier, "chunks": count}
                for tier, count in sorted(health["by_tier"].items())
            ]),
            hide_index=True, width="stretch",
        )

    st.divider()
    st.subheader("Transaction batch")
    upload = st.file_uploader("MT103 batch log", type=["pdf", "txt"], label_visibility="collapsed")
    force = st.checkbox(
        "Re-audit if already seen", value=False,
        help="The same file is normally answered from the existing audit rather than re-run and "
             "re-billed. Tick this to run it again -- worth doing after the corpus is rebuilt.",
    )

    if upload is not None and st.button("Run audit", type="primary", width="stretch"):
        try:
            accepted = api().submit(upload.name, upload.getvalue(), force=force)
        except ApiError as error:
            fail(error)
            st.stop()
        st.session_state[JOB_KEY] = accepted["job_id"]
        if accepted.get("deduplicated"):
            st.info("These exact bytes were already audited — showing that run. Nothing was billed.")

    st.divider()
    st.subheader("Past audits")
    try:
        # Journey 3's entry point: "an officer requests a past report (by month or case)".
        stored = api().reports(limit=25)
    except ApiError:
        stored = []
    if stored:
        chosen = st.selectbox(
            "Open a stored report",
            options=[""] + [f"{row['period']} · {row['report_id']}" for row in stored],
            format_func=lambda label: label or "—",
        )
        if chosen:
            st.session_state[JOB_KEY] = None
            st.session_state["report_id"] = chosen.split(" · ")[1]

    st.caption(f"{health['reports_stored']} report(s) on record")


# --- §6.2 the active audit workspace ------------------------------------------------------

st.title("Active audit workspace")

job_id = st.session_state.get(JOB_KEY)
report_id = st.session_state.get("report_id")

if job_id:
    # Polling, because the API returns a job id rather than holding the connection -- an audit runs
    # per candidate and a held connection is a timeout waiting for a proxy to find it.
    status = st.empty()
    with st.spinner("Auditing…"):
        while True:
            try:
                job = api().audit(job_id)
            except ApiError as error:
                fail(error)
                st.stop()
            if job["status"] != "running":
                break
            status.info(f"**{job['status']}** · {job['batch']} · submitted {job['submitted_at']}")
            time.sleep(POLL_SECONDS)
    status.empty()

    if job["status"] == "failed":
        st.error(f"The audit failed: {job['error']}")
        st.stop()
    report_id = job["report"]["report_id"]
    st.session_state["report_id"] = report_id
    st.session_state[JOB_KEY] = None

if not report_id:
    st.info("Upload an MT103 batch log in the sidebar, or open a stored report.")
    st.caption("Sample batches live in `data/processed/ledger/`.")
    st.stop()

try:
    # Fetched fresh on every rerun, never cached: a review recorded a second ago has to be visible,
    # and this is the *join* endpoint, so it shows where each finding now stands.
    report = api().report(report_id)
except ApiError as error:
    fail(error)
    st.stop()


# --- §6.3 the compliance summary ----------------------------------------------------------

st.divider()
st.subheader("Compliance summary")

RISK_STYLE.get(report.risk_rating, st.info)(
    f"**Risk: {report.risk_rating}** · {report.period} · {len(report.findings)} finding(s) · "
    f"{report.needs_review_count} the engine could not ground"
)
if report.clean:
    st.caption(
        "No qualifying pattern was found, so no obligation was engaged and no model was consulted."
    )
st.caption(f"`{report.report_id}` · filed {report.generated_at:%Y-%m-%d %H:%M} UTC")

if report.findings:
    st.dataframe(
        pd.DataFrame([
            {
                "pattern": f.candidate.pattern_type,
                "risk": f.risk_level,
                "status": f"{STATUS_ICON.get(f.status, '')} {f.status}",
                "confidence": round(f.confidence, 2),
                "transactions": len(f.candidate.member_txn_refs),
                "detection": round(f.candidate.detection_confidence, 2),
                "obligations": len(f.applicable_regulations),
                "indicators": len(f.red_flag_indicators),
            }
            for f in report.findings
        ]),
        hide_index=True, width="stretch",
    )

st.markdown(report.summary)


# --- §6.4 the review loop and the citations drawer ----------------------------------------

if report.findings:
    st.divider()
    st.subheader("Findings, evidence and review")
    st.caption(
        "Every clause below was carried from the retrieval bundle the model was shown — not "
        "re-searched afterwards, so this is the text the finding was actually drafted against. "
        "Reviewing a finding moves its status and appends to its history; the filed report is "
        "never edited."
    )

    reviewer = st.text_input(
        "Your reviewer id", value=st.session_state.get(REVIEWER_KEY, ""),
        placeholder="analyst@bank", help="Recorded against every review action, permanently.",
    )
    st.session_state[REVIEWER_KEY] = reviewer

    for finding in report.findings:
        icon = STATUS_ICON.get(finding.status, "")
        with st.expander(
            f"{icon} {finding.candidate.pattern_type} · {finding.risk_level} risk · "
            f"{finding.status} · {len(finding.candidate.member_txn_refs)} transactions"
        ):
            st.markdown(finding.narrative)

            # PRD v2 §5.3: the matched money-flow structure, so a layered or bipartite finding
            # reads at a glance rather than as a flat list of transactions.
            dot = evidence.to_dot(finding.candidate.subgraph)
            if dot:
                st.markdown("**Money-flow structure**")
                st.caption(evidence.describe(finding.candidate))
                st.graphviz_chart(dot, width="stretch")

            if finding.status == "needs_review" and finding.review_notes:
                st.warning("**Why this needs a human**\n\n" + "\n".join(
                    f"- {note}" for note in finding.review_notes
                ))
            elif finding.review_notes:
                st.caption("History: " + " · ".join(finding.review_notes))

            for label, citations in (
                ("Binding obligations", finding.applicable_regulations),
                ("Red-flag indicators", finding.red_flag_indicators),
            ):
                if not citations:
                    continue
                st.markdown(f"**{label}**")
                for citation in citations:
                    with st.container(border=True):
                        st.markdown(f"`{citation.source_id}` **{citation.section_ref}**")
                        st.write(citation.text_excerpt)
                        st.caption(f"`{citation.chunk_id}`")

            actions = ACTIONS.get(finding.status, [])
            if not actions:
                st.caption(f"This finding is {finding.status} — no further action is available.")
                continue

            note = st.text_input(
                "Note (recorded with the decision)", key=f"note-{finding.finding_id}",
                placeholder="why you are clearing or escalating this",
            )
            columns = st.columns(len(actions))
            for column, (action, label) in zip(columns, actions):
                if not column.button(label, key=f"{action}-{finding.finding_id}",
                                     width="stretch"):
                    continue
                if not reviewer.strip():
                    st.error("Enter your reviewer id first — every review is attributed.")
                    continue
                try:
                    api().review(finding.finding_id, action, reviewer=reviewer.strip(), note=note)
                except ApiError as error:
                    fail(error)
                else:
                    st.rerun()


# --- §6.5 what the batch did not contain, and diagnostics ---------------------------------

st.divider()
st.subheader("Ingestion record")

if report.quarantined_count:
    st.error(
        f"**{report.quarantined_count} message(s) could not be parsed** and were excluded from "
        "screening. This review does not cover them."
    )

try:
    validation = api().validation(report.report_id)
except ApiError as error:
    validation = None
    fail(error)

if validation is None:
    st.caption("No ingestion record was kept for this report.")
else:
    columns = st.columns(4)
    columns[0].metric("Parsed", validation.parsed)
    columns[1].metric("Declared", validation.declared if validation.declared is not None else "—")
    columns[2].metric("Rescued by the fallback", validation.rescued)
    columns[3].metric("Quarantined", len(validation.quarantined))

    if validation.rescued:
        st.caption(
            f"{validation.rescued} message(s) the strict parser refused were read by the "
            "light model and are tagged `llm_fallback`. The fallback runs once per message and "
            "never invents a field."
        )
    if validation.quarantined:
        # The point of the panel: a count is not actionable. An analyst told twelve messages were
        # lost needs to see which twelve to go and fix the source.
        st.markdown("**Messages neither the parser nor the fallback could read**")
        st.dataframe(
            pd.DataFrame([
                {
                    "position": message.ordinal,
                    "reference": message.reference or "—",
                    "reason": message.reason,
                    "fallback tried": message.fallback_attempted,
                }
                for message in validation.quarantined
            ]),
            hide_index=True, width="stretch",
        )
        with st.expander("Raw text of each quarantined message"):
            for message in validation.quarantined:
                st.caption(f"#{message.ordinal} — {message.reason}")
                st.code(message.raw or "(empty)", language="text")
    elif validation.complete:
        st.success("Every message the statement declared came back as a record.")

with st.expander("Diagnostics"):
    st.caption(f"report `{report.report_id}` · run `{report.run_id}` · schema "
               f"{report.schema_version}")
    st.caption(
        f"{len(report.source_document_refs)} distinct clause(s) across "
        f"{len(report.findings)} finding(s)"
    )
    if st.toggle("Show the report exactly as filed (no review applied)"):
        # The immutability claim, made checkable from the UI rather than asserted in a docstring.
        try:
            as_filed = api().filed(report.report_id)
        except ApiError as error:
            fail(error)
        else:
            st.caption(
                "`report_json` is written once and never updated. Review changes where the work "
                "stands, not what the engine concluded."
            )
            st.dataframe(
                pd.DataFrame([
                    {
                        "pattern": f.candidate.pattern_type,
                        "as filed": f.status,
                        "now": next(
                            (g.status for g in report.findings if g.finding_id == f.finding_id),
                            "—",
                        ),
                    }
                    for f in as_filed.findings
                ]),
                hide_index=True, width="stretch",
            )
