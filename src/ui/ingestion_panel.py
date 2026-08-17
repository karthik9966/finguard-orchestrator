"""Vector store ingestion metrics -- §3.4's third milestone.

Deliberately minimal. The full auditor cockpit is Phase 4 (§6); this page exists to answer the
two questions §6.1 says an auditor must be able to answer before trusting a run:

* Is the knowledge base actually loaded, and with which embedding model?
* **Which rules are active?** -- a citation is only checkable if you know the corpus behind it.

Run with::

    uv run streamlit run src/ui/ingestion_panel.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.ingestion.store import COLLECTION_NAME, inventory, stats

st.set_page_config(page_title="FinGuard — Ingestion", page_icon="📚", layout="wide")
st.title("Vector store ingestion")

try:
    payload = stats()
except Exception as error:  # noqa: BLE001 - the empty state is the common case, show it plainly
    st.error(f"Collection {COLLECTION_NAME!r} is not available.")
    st.code("uv run python -m src.ingestion.store --backend minilm", language="bash")
    st.caption(f"{type(error).__name__}: {error}")
    st.stop()

left, middle, right, far = st.columns(4)
left.metric("Vectors", f"{payload['vectors']:,}")
middle.metric("Embedding model", payload["model"] or "unknown")
right.metric("Documents", f"{len(inventory())}")
far.metric("Undated chunks", f"{payload['undated']:,}")
st.caption(f"Collection `{payload['collection']}` · backend `{payload['backend']}` · built {payload['built']}")

st.divider()

by_corpus, by_tier = st.columns(2)
with by_corpus:
    st.subheader("By corpus")
    st.bar_chart(pd.Series(payload["by_corpus"], name="chunks"))
with by_tier:
    st.subheader("By relevance tier")
    st.bar_chart(pd.Series({f"tier {k}": v for k, v in payload["by_tier"].items()}, name="chunks"))
    st.caption(
        "Tier 1 is the default retrieval scope. Tier 3 is indexed but normally filtered out — "
        "it is the distractor set that makes Context Precision (§8.1) measurable."
    )

st.divider()

st.subheader("Vector store inventory")
st.caption("Which rules are active, so a citation can be checked against a known corpus.")
frame = pd.DataFrame(inventory())
st.dataframe(
    frame,
    use_container_width=True,
    hide_index=True,
    column_config={
        "document": st.column_config.TextColumn("Document", width="large"),
        "corpus": st.column_config.TextColumn("Corpus"),
        "tier": st.column_config.NumberColumn("Tier"),
        "jurisdiction": st.column_config.TextColumn("Jurisdiction"),
        "updated": st.column_config.TextColumn("Last updated"),
        "chunks": st.column_config.NumberColumn("Chunks"),
    },
)
