"""The reasoning graph -- LLD §5.1 steps 4-7 as a LangGraph `StateGraph`.

    START → detection ─(clean)───────────────────────────────→ report → END
                  └─(candidates)→ retrieval → grounding → critic ─┐
                                       ↑                          │
                                       └──── loop (hint) ──────────┤
                                       └──── next candidate ───────┤
                                                                   └→ report → END

The graph exists for the two back edges. A straight line is a `for` loop written expensively; what
a graph gives you is that `critic` can send execution *backwards*. Both back edges matter and they
are different things:

* **loop** -- the same candidate, with a reformulated retrieval question. This is the difference
  between a report that says "no clause covers this" and one that goes and finds the clause.
* **next candidate** -- the same edge in the graph, a different `current_index`. Per-candidate
  isolation is expressed as a cycle rather than as a subgraph per candidate because the state a
  candidate needs is exactly `current_index`, and one cycle keeps the trace linear and readable.

The loop goes back to **retrieval**, not to grounding. A thin finding is usually missing law rather
than bad prose, and `refinement_hint` is written as a retrieval question -- redrafting against the
same context would only produce a better-worded version of the same gap.

Parsing is deliberately absent: LLD §5.1 puts it outside the graph, in `src/graph/run.py`, so a
malformed file is a client error before a run id is ever minted.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

from src.graph.nodes import (
    CriticNode,
    DetectionNode,
    GroundingNode,
    ReportGenerationNode,
    RetrievalNode,
    route_after_critic,
    route_after_detection,
)
from src.models import AgentState

# Explicit, not incidental. The tracing variables do reach the process without this line -- via
# retriever → store → embeddings, which calls load_dotenv for its own reasons -- but that is a
# chain of imports none of which exists for this purpose, and rearranging any of them would turn
# tracing off silently. That is the one failure `tracing_project()` exists to make visible.
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)


def build_graph(
    *,
    detection: Any = None,
    retrieval: Any = None,
    grounding: Any = None,
    critic: Any = None,
    report: Any = None,
):
    """Compile the five-node auditor.

    Every node is injectable. Not for elegance: the suite must exercise the per-candidate loop,
    the faithfulness veto and the clean path without a key or a network, and those are properties
    of the *graph* rather than of any node -- so they have to be testable with stub nodes in place.
    """
    graph = StateGraph(AgentState)

    graph.add_node("detection", detection or DetectionNode())
    graph.add_node("retrieval", retrieval or RetrievalNode())
    graph.add_node("grounding", grounding or GroundingNode())
    graph.add_node("critic", critic or CriticNode())
    graph.add_node("report", report or ReportGenerationNode())

    graph.add_edge(START, "detection")
    # A clean batch never constructs a model or opens the vector store.
    graph.add_conditional_edges(
        "detection", route_after_detection, {"retrieval": "retrieval", "report": "report"}
    )
    graph.add_edge("retrieval", "grounding")
    graph.add_edge("grounding", "critic")
    graph.add_conditional_edges(
        "critic", route_after_critic, {"retrieval": "retrieval", "report": "report"}
    )
    graph.add_edge("report", END)

    return graph.compile()


def recursion_limit(candidate_count: int, max_loops: int) -> int:
    """LangGraph's step budget, sized to the work rather than left at its default of 25.

    Each candidate costs at most `(1 + max_loops)` passes of three nodes, and the default 25 is
    exceeded by the fourth candidate -- which on the 10k batch would be a `GraphRecursionError`
    reported as a system fault when nothing at all had gone wrong.
    """
    per_candidate = 3 * (1 + max(max_loops, 0))
    return max(25, 2 + candidate_count * per_candidate + 2)


def tracing_project() -> str | None:
    """The LangSmith project this run lands in, or None when tracing is off.

    Reported rather than assumed: a trace that silently is not being written is worse than no
    tracing at all, because you go looking for it after the run instead of before.
    """
    enabled = os.environ.get("LANGCHAIN_TRACING_V2", "").strip().lower() in {"true", "1", "yes"}
    if not enabled or not os.environ.get("LANGCHAIN_API_KEY"):
        return None
    return os.environ.get("LANGCHAIN_PROJECT", "default")


def run_config(
    *,
    run_id: str,
    batch_id: str,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    candidate_count: int = 0,
) -> dict[str, Any]:
    """The run-scoped configuration: identity now, per-call detail later.

    Batch-derived numbers do not exist yet -- detection has not run -- so what belongs here is the
    identity of the run, and every span underneath it shares one searchable id.
    `nodes.trace_config` attaches the per-candidate detail.
    """
    from src.config import get_config

    return {
        "run_id": None,  # LangGraph mints its own; ours travels in metadata, where it is greppable
        "tags": ["AML_AUDIT_RUN", *(tags or [])],
        "metadata": {"run_id": run_id, "batch_id": batch_id, **(metadata or {})},
        "recursion_limit": recursion_limit(candidate_count, get_config().reasoning.max_loops),
    }


def new_run_id() -> str:
    return f"run-{uuid4().hex[:12]}"
