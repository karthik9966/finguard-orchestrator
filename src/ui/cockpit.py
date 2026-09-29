"""The auditor cockpit -- LLD §6's sections over the Phase 5 reasoning core.

Everything the engine does is measurable from the CLI. This page exists because a compliance
analyst is not going to read a terminal, and because the facts that make the output trustworthy are
invisible in a JSON dump: *which candidate the money was spent on*, *which exact clause justified
each finding*, and *which findings the review could not stand behind*.

Run with::

    uv run streamlit run src/ui/cockpit.py

The run fires only from the button, never from a rerun -- see `RESULT_KEY`.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from src.graph.graph import tracing_project
from src.graph.run import BatchUnreadable, InMemoryResultsStore, stream_audit
from src.ingestion.store import RULE_COLLECTION, VectorStoreClient
from src.utils.swift_parser import parse_batch

st.set_page_config(page_title="FinGuard — Auditor Cockpit", page_icon="⚖️", layout="wide")

# Streamlit re-executes this whole file on every widget interaction. Holding the finished run in
# session state -- keyed by the batch's own bytes -- is what stops a checkbox from re-billing a
# run. Re-uploading the identical file finds the result already there.
RESULT_KEY = "audit_result"
STORE_KEY = "results_store"
UPLOAD_DIR = Path(st.__file__).parent.parent / ".finguard_uploads"

# The real node names, because narrating the graph that actually executes is the point of a
# reasoning tracker. Retrieval, grounding and critique repeat once per candidate.
NODE_LABELS = {
    "detection": "Screening for structuring / fan-in / fan-out / cycle / scatter-gather",
    "retrieval": "Resolving binding obligations, then searching red-flag indicators",
    "grounding": "Grounding the candidate in the retrieved law (reasoning model)",
    "critic": "Faithfulness gate, then scoring how well the draft is supported",
    "report": "Assembling the filing — no model, every field derived from the findings",
}
RISK_STYLE = {"high": st.error, "medium": st.warning, "low": st.success, "none": st.success}
STATUS_ICON = {"pending_review": "✅", "needs_review": "⚠️"}


@st.cache_data(ttl=60, show_spinner="Reading the vector store...")
def _counts() -> dict:
    return VectorStoreClient(RULE_COLLECTION).counts()


def store() -> InMemoryResultsStore:
    if STORE_KEY not in st.session_state:
        st.session_state[STORE_KEY] = InMemoryResultsStore()
    return st.session_state[STORE_KEY]


# --- §6.1 the ingestion gate --------------------------------------------------------------

with st.sidebar:
    st.header("Ingestion gate")
    try:
        payload = _counts()
    except Exception as error:  # noqa: BLE001 - the empty state is the common case, show it
        st.error(f"Collection {RULE_COLLECTION!r} is not available.")
        st.code("uv run finguard-store --rules", language="bash")
        st.caption(f"{type(error).__name__}: {error}")
        st.stop()

    left, right = st.columns(2)
    left.metric("Chunks", f"{payload['total']:,}")
    right.metric("Binding", f"{payload['authority'].get('binding', 0):,}")
    st.caption(f"`{RULE_COLLECTION}` · US BSA/AML corpus")

    with st.expander("Corpus composition"):
        # §6.1: a citation is only checkable if you know which corpus stands behind it. Tier and
        # authority are shown separately because they are different claims -- tier says what kind
        # of document the text came from, authority says whether it binds.
        st.dataframe(
            pd.DataFrame(
                [{"tier": tier, "chunks": count} for tier, count in sorted(payload["tier"].items())]
            ),
            hide_index=True, use_container_width=True,
        )

    st.divider()
    st.subheader("Transaction batch")
    upload = st.file_uploader("MT103 batch log", type=["pdf", "txt"], label_visibility="collapsed")

    batch_path: Path | None = None
    if upload is not None:
        UPLOAD_DIR.mkdir(exist_ok=True)
        batch_path = UPLOAD_DIR / upload.name
        batch_path.write_bytes(upload.getvalue())
        try:
            # Validated here, not two nodes into a paid run: a file that yields no transactions is
            # rejected in the sidebar for free.
            preview = parse_batch(batch_path, strict=False)
        except Exception as error:  # noqa: BLE001
            st.error(f"Not a readable MT103 batch: {error}")
            st.stop()

        if not preview.wires:
            st.error("No transactions could be parsed from this file.")
            st.stop()

        st.success(f"{preview.parsed} of {preview.declared_messages or preview.parsed} messages")
        st.caption(
            f"{min(w.value_date for w in preview.wires)} to "
            f"{max(w.value_date for w in preview.wires)}"
        )
        if preview.failures:
            st.warning(
                f"{len(preview.failures)} message(s) refused — the light-model fallback will "
                "attempt each once, then quarantine it"
            )

    st.divider()
    project = tracing_project()
    st.caption(f"Tracing: {f'LangSmith `{project}`' if project else 'off'}")


# --- §6.2 the active audit workspace ------------------------------------------------------

st.title("Active audit workspace")

if batch_path is None:
    st.info("Upload an MT103 batch log in the sidebar to begin.")
    st.caption("Sample batches live in `data/processed/ledger/`.")
    st.stop()

fingerprint = hashlib.sha256(upload.getvalue()).hexdigest()[:16]
held = st.session_state.get(RESULT_KEY)

if st.button("Run audit", type="primary"):
    timings: dict[str, float] = {}
    counts: dict[str, int] = {}
    started = time.perf_counter()
    result = None

    with st.container(border=True):
        st.caption("Reasoning graph — retrieval, grounding and review repeat per candidate")
        progress = st.empty()
        try:
            for node, payload in stream_audit(batch_path, store=store(), tags=["COCKPIT"]):
                if node == "__final__":
                    result = payload
                    break
                elapsed = time.perf_counter() - started
                timings[node] = timings.get(node, 0.0) + (elapsed - sum(timings.values()))
                counts[node] = counts.get(node, 0) + 1
                total = len(payload.get("candidates") or [])
                position = min(payload.get("current_index", 0) + 1, max(total, 1))
                progress.success(
                    f"**{node}** ×{counts[node]} — {NODE_LABELS.get(node, node)}"
                    + (f"  ·  candidate {position} of {total}" if total else "")
                )
        except BatchUnreadable as error:
            st.error(str(error))
            st.stop()

    st.session_state[RESULT_KEY] = {
        "fingerprint": fingerprint, "result": result, "timings": timings, "counts": counts,
    }
    held = st.session_state[RESULT_KEY]

if held is None:
    st.info("Press **Run audit** to analyse this batch. A clean batch costs $0.0000.")
    st.stop()

if held["fingerprint"] != fingerprint:
    st.warning(
        "Showing the previous audit — the batch changed. Press **Run audit** to analyse this one."
    )

result = held["result"]
if result is None:
    st.error("The run produced no report.")
    st.stop()
report = result.report


# --- §6.3 the compliance summary ----------------------------------------------------------

st.divider()
st.subheader("Compliance summary")

RISK_STYLE.get(report.risk_rating, st.info)(
    f"**Risk: {report.risk_rating}** · {len(report.findings)} finding(s) from "
    f"{result.candidates} candidate(s) · {report.needs_review_count} needing review"
)
if report.clean:
    st.caption(
        "No qualifying pattern was found, so no obligation was engaged and no model was consulted."
    )
if report.quarantined_count:
    st.warning(
        f"{report.quarantined_count} message(s) could not be parsed and were excluded from "
        "screening. This review does not cover them."
    )

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
        hide_index=True, use_container_width=True,
    )

st.markdown(report.summary)


# --- §6.4 the auditor's citations drawer --------------------------------------------------

st.divider()
st.subheader("Verified citations")
st.caption(
    "Every clause below was carried from the retrieval bundle the model was shown — not "
    "re-searched afterwards, so this is the text the finding was actually drafted against."
)

if not report.source_document_refs:
    st.info("This report cites no clauses.")
else:
    for citation in report.source_document_refs:
        with st.expander(f"{citation.source_id} — {citation.section_ref}"):
            st.write(citation.text_excerpt)
            st.caption(f"`{citation.chunk_id}`")


# --- §6.5 telemetry & diagnostics ---------------------------------------------------------

st.divider()
if st.toggle("Telemetry & diagnostics"):
    st.caption(f"run_id `{result.run_id}` · report_id `{report.report_id}`")
    st.caption(f"Ingestion: {result.validation.summary()}")

    usage = result.usage
    if not usage.nodes:
        # The free path is a result, not an empty table: detection found nothing to audit and no
        # model was ever constructed.
        st.success("**$0.0000** — no model was called. The batch cleared the free path.")
    else:
        total = usage.total_cost
        columns = st.columns(4)
        columns[0].metric("Cost", f"${total:.4f}" if total is not None else "unpriced")
        columns[1].metric("Per candidate",
                          f"${result.cost_per_candidate:.4f}"
                          if result.cost_per_candidate is not None else "—")
        columns[2].metric("Model calls", usage.calls)
        columns[3].metric("Tokens", f"{usage.total_tokens:,}")
        st.dataframe(pd.DataFrame(usage.rows()), hide_index=True, use_container_width=True)

    st.dataframe(
        pd.DataFrame([
            {"node": node, "visits": held["counts"].get(node, 0),
             "seconds": round(seconds, 2), "free": node in {"detection", "retrieval", "report"}}
            for node, seconds in held["timings"].items()
        ]),
        hide_index=True, use_container_width=True,
    )
    st.caption(
        "Detection, retrieval and report assembly cost nothing — the deterministic majority of "
        "the pipeline. Grounding and review are the only paid nodes."
    )

    needs_review = [f for f in report.findings if f.status == "needs_review"]
    if needs_review:
        st.warning(
            "Unresolved after review:\n"
            + "\n".join(
                f"- **{f.candidate.pattern_type}**: " + "; ".join(f.review_notes)
                for f in needs_review
            )
        )
