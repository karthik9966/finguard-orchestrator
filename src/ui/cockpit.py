"""The auditor cockpit -- §6's five sections over the Phase 2 graph.

Everything the engine does is already measurable from the CLI. This page exists because a
compliance analyst is not going to read a terminal, and because the two facts that make the
output trustworthy are invisible in a JSON dump: *which node spent the money*, and *which exact
clause justified each flag*.

Run with::

    uv run streamlit run src/ui/cockpit.py

An audit costs $0.06-$0.18 in gpt-4o calls, so it fires only from the button and never from a
rerun. See `RESULT_KEY` below.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from src.graph.graph import stream_batch, tracing_project
from src.ingestion.store import COLLECTION_NAME, by_id, inventory, stats
from src.utils.swift_parser import parse_batch

st.set_page_config(page_title="FinGuard — Auditor Cockpit", page_icon="⚖️", layout="wide")

# Streamlit re-executes this whole file on every widget interaction. Holding the finished run in
# session state -- keyed by the batch's own bytes -- is what stops a checkbox from re-billing an
# audit. Re-uploading the identical file finds the result already there.
RESULT_KEY = "audit_result"
UPLOAD_DIR = Path(st.__file__).parent.parent / ".finguard_uploads"

# §6.2 lists four steps; the graph runs seven nodes. Narrating the graph that actually executes
# is the point of a reasoning tracker, so these are the real node names.
NODE_LABELS = {
    "parse": "Normalising SWIFT messages (regex; gpt-4o-mini only on refusal)",
    "detect": "Screening for concentration / dispersion / path / magnitude",
    "no_findings": "No candidates — closing the batch without a model call",
    "audit": "Retrieving obligations from ChromaDB",
    "draft": "Drafting findings against the retrieved clauses (gpt-4o)",
    "critic": "Verifying citations and scoring support (gpt-4o)",
    "generate": "Compiling the filing schema (gpt-4o)",
}
RISK_STYLE = {"High": st.error, "Medium": st.warning, "Low": st.success}


@st.cache_data(ttl=60, show_spinner="Reading the vector store...")
def _stats() -> dict:
    return stats()


@st.cache_data(ttl=60, show_spinner=False)
def _inventory() -> list[dict]:
    return inventory()


@st.cache_data(ttl=300, show_spinner=False)
def _clauses(chunk_ids: tuple[str, ...]) -> dict[str, dict]:
    return by_id(list(chunk_ids))


# --- §6.1 the ingestion sidebar ---------------------------------------------------------

with st.sidebar:
    st.header("Ingestion gate")
    try:
        payload = _stats()
        documents = _inventory()
    except Exception as error:  # noqa: BLE001 - the empty state is the common case, show it
        st.error(f"Collection {COLLECTION_NAME!r} is not available.")
        st.code("uv run finguard-store --backend minilm", language="bash")
        st.caption(f"{type(error).__name__}: {error}")
        st.stop()

    left, right = st.columns(2)
    left.metric("Vectors", f"{payload['vectors']:,}")
    right.metric("Documents", f"{len(documents)}")
    st.caption(f"`{payload['collection']}` · {payload['backend']} · built {payload['built']}")

    with st.expander(f"Active rules ({len(documents)})"):
        # §6.1: a citation is only checkable if you know which corpus stands behind it.
        st.dataframe(
            pd.DataFrame(documents)[["document", "chunks", "tier", "corpus"]],
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
            # Validated here, not three nodes into a paid run: a file that yields no wires is
            # rejected in the sidebar for free.
            preview = parse_batch(batch_path, strict=False)
        except Exception as error:  # noqa: BLE001
            st.error(f"Not a readable MT103 batch: {error}")
            st.stop()

        if not preview.wires:
            st.error("No wires could be parsed from this file.")
            st.stop()

        st.success(f"{preview.parsed} of {preview.declared_messages or preview.parsed} messages")
        cross = sum(1 for w in preview.wires if w.is_cross_border)
        st.caption(
            f"{min(w.value_date for w in preview.wires)} to "
            f"{max(w.value_date for w in preview.wires)} · {cross} cross-border"
        )
        if preview.failures:
            st.warning(f"{len(preview.failures)} message(s) refused — the fallback will retry them")

    st.divider()
    project = tracing_project()
    st.caption(f"Tracing: {f'LangSmith `{project}`' if project else 'off'}")


# --- §6.2 the active audit workspace ----------------------------------------------------

st.title("Active audit workspace")

if batch_path is None:
    st.info("Upload an MT103 batch log in the sidebar to begin.")
    st.caption("Sample batches live in `data/processed/ledger/`.")
    st.stop()

auditor_query = st.text_input(
    "Command bar",
    placeholder="Optional: e.g. obligation to report transfers structured below a threshold",
    help=(
        "Asked as an extra retrieval query alongside the built-in obligation templates, never "
        "instead of them — measured, a template ranks the correct clause 5th where a free-text "
        "description of the same facts ranks it 315th."
    ),
)

fingerprint = hashlib.sha256(upload.getvalue()).hexdigest()[:16] + hashlib.sha256(
    auditor_query.encode()
).hexdigest()[:8]
held = st.session_state.get(RESULT_KEY)
run_now = st.button("Run audit", type="primary", use_container_width=False)

if run_now:
    timings: dict[str, float] = {}
    started = time.perf_counter()
    final: dict | None = None

    with st.container(border=True):
        st.caption("Reasoning graph")
        for node, _update in stream_batch(str(batch_path), auditor_query=auditor_query,
                                          tags=["COCKPIT"]):
            if node == "__final__":
                final = _update
                break
            elapsed = time.perf_counter() - started
            timings[node] = elapsed - sum(timings.values())
            st.success(f"**{node}** — {NODE_LABELS.get(node, node)}  ·  {timings[node]:.1f}s")

    st.session_state[RESULT_KEY] = {"fingerprint": fingerprint, "state": final, "timings": timings}
    held = st.session_state[RESULT_KEY]

if held is None:
    st.info("Press **Run audit** to analyse this batch. Roughly $0.06–$0.18 in model calls.")
    st.stop()

if held["fingerprint"] != fingerprint:
    st.warning(
        "Showing the previous audit — the batch or the command bar changed. "
        "Press **Run audit** to analyse the current one."
    )

state = held["state"]
report = state.get("report") if state else None
usage = state.get("usage") if state else None

if report is None:
    st.error("The run produced no report.")
    st.stop()


# --- §6.3 the compliance summary report -------------------------------------------------

st.divider()
st.subheader("Compliance summary")

confidence = state.get("confidence_score", 0.0) or 0.0
loops = state.get("loop_count", 0)
# The rating is shown *with* the critic's confidence deliberately. The rating alone does not yet
# separate a clean batch from a dirty one -- all four sample batches read Medium, and May has no
# laundering at all -- so presenting it as a lone verdict would overstate what it knows.
RISK_STYLE.get(report.risk_rating, st.info)(
    f"**Risk: {report.risk_rating}** · critic confidence {confidence:.2f} "
    f"after {loops} pass(es) · {len(report.flagged_wires)} wire(s) flagged"
)
if confidence < 0.75:
    st.caption(
        "Confidence below the 0.75 threshold: the critic could not fully ground the draft, and "
        "its reservations are recorded at the end of the narrative."
    )

wires = {w.reference: w for w in state.get("wires", [])}
flagged = [wires[r].as_row() for r in report.flagged_wires if r in wires]
if flagged:
    st.dataframe(
        pd.DataFrame(flagged)[
            ["reference", "value_date", "currency", "amount",
             "sender_account", "receiver_account", "corridor", "memo"]
        ],
        hide_index=True, use_container_width=True,
    )
else:
    st.caption("No individual wires were flagged.")

st.markdown(report.audit_summary)


# --- §6.4 the auditor's citations drawer ------------------------------------------------

st.divider()
st.subheader("Verified citations")
st.caption(
    "Every clause below was resolved from the report's `source_document_hashes` back to the "
    "stored chunk it was drafted against — not re-searched, so this is the text the model "
    "actually saw."
)

hashes = tuple(report.source_document_hashes)
if not hashes:
    st.info("This report cites no clauses.")
else:
    resolved = _clauses(hashes)
    for chunk_id in hashes:
        clause = resolved.get(chunk_id)
        if clause is None:
            # Cannot happen while the citation veto holds; said plainly rather than shown blank.
            st.warning(f"`{chunk_id}` could not be resolved in the vector store.")
            continue
        with st.expander(
            f"{clause.get('document_title')} — {clause.get('section_clause')} "
            f"(tier {clause.get('relevance_tier')})"
        ):
            st.write(clause["text"])
            st.caption(f"`{chunk_id}`")


# --- §6.5 telemetry & diagnostics -------------------------------------------------------

st.divider()
if st.toggle("Telemetry & diagnostics"):
    st.caption(f"audit_id `{state.get('audit_id')}`")

    if usage is None or not usage.nodes:
        # §9.1's free path is a result, not an empty table: detect found nothing to audit and the
        # model was never constructed.
        st.success("**$0.0000** — no model was called. The batch cleared the free path.")
    else:
        total = usage.total_cost
        cols = st.columns(3)
        cols[0].metric("Cost", f"${total:.4f}" if total is not None else "unpriced")
        cols[1].metric("Model calls", usage.calls)
        cols[2].metric("Tokens", f"{usage.total_tokens:,}")

        breakdown = pd.DataFrame(usage.rows())
        breakdown["latency_s"] = [
            round(held["timings"].get(row["node"], float("nan")), 2) for _, row in breakdown.iterrows()
        ]
        st.dataframe(breakdown, hide_index=True, use_container_width=True)

    free_nodes = {n: s for n, s in held["timings"].items() if n in {"parse", "detect", "audit"}}
    if free_nodes:
        st.caption(
            "Free nodes: "
            + " · ".join(f"{node} {seconds:.1f}s" for node, seconds in free_nodes.items())
            + " — parsing, detection and retrieval cost nothing."
        )

    st.caption(
        f"Retrieved {len(state.get('retrieved_context', []))} clauses from "
        f"{len(state.get('queries', []))} queries · "
        f"{len(state.get('candidates', []))} candidate pattern(s)"
    )
    if state.get("reservations"):
        st.warning("Unresolved after review:\n" + "\n".join(f"- {r}" for r in state["reservations"]))
