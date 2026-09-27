"""Wire the nodes into LangGraph's StateGraph (§4.2) and run a batch.

The graph is here for one edge. PARSE -> DETECT -> AUDIT -> DRAFT -> CRITIC -> GENERATE is a
straight line that a ``for`` loop would express perfectly well; the edge from CRITIC *back to*
AUDIT is the cycle, and a cycle is what a graph gives you that a chain does not. A critic that
can only approve or reject is a filter. One that can reformulate the question and send execution
back to retrieval is the difference between a report that says "no clause covers this" and one
that goes and finds the clause.

Usage::

    uv run python -m src.graph.graph --batch data/processed/ledger/2023-06_private_banking_log.pdf
    uv run python -m src.graph.graph --batch <path> --json report.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from collections.abc import Iterator
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

from src.graph.nodes import (
    audit_node,
    critic_node,
    detect_node,
    draft_node,
    generate_node,
    no_findings_node,
    parse_node,
    route_after_critic,
    route_after_detect,
)
from src.graph.cost import UsageLedger
from src.graph.state import AgentState, initial_state
from src.utils.swift_parser import existing_log


def build_graph():
    """Compile the six-node auditor."""
    graph = StateGraph(AgentState)

    graph.add_node("parse", parse_node)
    graph.add_node("detect", detect_node)
    graph.add_node("no_findings", no_findings_node)
    graph.add_node("audit", audit_node)
    graph.add_node("draft", draft_node)
    graph.add_node("critic", critic_node)
    graph.add_node("generate", generate_node)

    graph.add_edge(START, "parse")
    graph.add_edge("parse", "detect")

    # Nothing to audit costs nothing: the model is never reached.
    graph.add_conditional_edges(
        "detect", route_after_detect, {"audit": "audit", "no_findings": "no_findings"}
    )
    graph.add_edge("no_findings", END)

    graph.add_edge("audit", "draft")
    graph.add_edge("draft", "critic")

    # The cycle. Back to retrieval with a reformulated question, or on to the filing schema.
    graph.add_conditional_edges(
        "critic", route_after_critic, {"audit": "audit", "generate": "generate"}
    )
    graph.add_edge("generate", END)

    return graph.compile()


# Explicit, not incidental. The tracing variables did reach the process before this line -- via
# graph -> nodes -> store -> embeddings, which calls load_dotenv for its own API key -- but that
# is a chain of imports none of which exists for this reason. Rearranging any of them would turn
# tracing off silently, which is the one failure mode `tracing_project()` exists to prevent.
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)


def tracing_project() -> str | None:
    """The LangSmith project this run will land in, or None when tracing is off (§7.1).

    Reported rather than assumed: a trace that silently is not being written is worse than no
    tracing at all, because you go looking for it after the run instead of before.
    """
    enabled = os.environ.get("LANGCHAIN_TRACING_V2", "").strip().lower() in {"true", "1", "yes"}
    if not enabled or not os.environ.get("LANGCHAIN_API_KEY"):
        return None
    return os.environ.get("LANGCHAIN_PROJECT", "default")


def run_config(batch_path: str, *, tags: list[str] | None = None,
               metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """§7.2's scoped run configuration -- the half of it that is knowable before the run.

    Batch-derived numbers (wires, candidates) do not exist yet: PARSE has not run. Those are
    attached per model call by ``nodes.trace_config`` instead. What belongs here is the identity
    of the run, so every span underneath it shares one searchable id.
    """
    return {
        "tags": ["AML_AUDIT_RUN", *(tags or [])],
        "metadata": {
            "audit_id": f"aud-{uuid4().hex[:12]}",
            "batch": Path(batch_path).name,
            **(metadata or {}),
        },
    }


def audit_batch(batch_path: str, *, tags: list[str] | None = None,
                metadata: dict[str, Any] | None = None) -> AgentState:
    config = run_config(batch_path, tags=tags, metadata=metadata)
    # One ledger per run, registered at the top: LangChain propagates run-level callbacks into
    # every nested call, so a node added later is accounted for without registering it anywhere.
    ledger = UsageLedger()
    config["callbacks"] = [ledger]

    state = build_graph().invoke(initial_state(batch_path), config=config) # type: ignore
    state["audit_id"] = config["metadata"]["audit_id"]
    state["usage"] = ledger
    return state # type: ignore


def stream_batch(batch_path: str, *, auditor_query: str = "", tags: list[str] | None = None,
                 metadata: dict[str, Any] | None = None) -> Iterator[tuple[str, dict[str, Any]]]:
    """Run a batch, yielding ``(node_name, update)`` as each node finishes.

    ``audit_batch`` returns only when the whole run is done, which is right for a CLI and wrong
    for §6.2's reasoning tracker -- a cockpit driven by it shows a spinner for forty seconds and
    then everything at once, which tells an auditor nothing about *where* the time and money go.

    The final element is ``("__final__", state)``: LangGraph's update stream yields each node's
    *delta*, never the accumulated state, so a consumer that only wants the report would
    otherwise have to reassemble it. The ledger and audit_id are attached there, exactly as
    ``audit_batch`` attaches them.
    """
    config = run_config(batch_path, tags=tags, metadata=metadata)
    ledger = UsageLedger()
    config["callbacks"] = [ledger]

    state: dict[str, Any] = dict(initial_state(batch_path))
    if auditor_query.strip():
        state["auditor_query"] = auditor_query.strip()

    for step in build_graph().stream(state, config=config, stream_mode="updates"): # type: ignore
        for node, update in step.items():
            state.update(update)
            yield node, update

    state["audit_id"] = config["metadata"]["audit_id"]
    state["usage"] = ledger
    yield "__final__", state


def print_run(state: AgentState) -> None:
    entities = state.get("extracted_entities", {})
    report = state.get("report")

    project = tracing_project()
    print(f"\naudit_id   : {state.get('audit_id', '-')}")
    print(f"tracing    : {f'LangSmith project {project!r}' if project else 'off'}")
    print(f"batch      : {entities.get('batch')}")
    print(f"parsed     : {entities.get('parsed_messages')} of {entities.get('declared_messages')}")
    if state.get("extraction_failures"):
        rescued = sum(1 for f in state["extraction_failures"] if f["rescued_by_model"]) # type: ignore
        print(f"fallback   : {len(state['extraction_failures'])} refused, {rescued} rescued by model") # type: ignore
    print(f"candidates : {len(state.get('candidates', []))} "
          f"covering {entities.get('wires_under_review', 0)} wires")
    print(f"retrieved  : {len(state.get('retrieved_context', []))} clauses "
          f"from {len(state.get('queries', []))} queries")
    print(f"critic     : {state.get('loop_count', 0)} pass(es), "
          f"confidence {state.get('confidence_score', 0):.2f}")
    usage = state.get("usage")
    if usage is not None:
        print(usage.summary())

    if report is None:
        print("\nno report produced")
        return

    print(f"\nrisk       : {report.risk_rating}")
    print(f"flagged    : {len(report.flagged_wires)} wires")
    print(f"regulations: {len(report.applicable_regulations)} clauses")
    for regulation in report.applicable_regulations:
        print(f"    - {regulation}")
    print(f"\n{report.audit_summary}")


def main() -> int:
    assert __doc__ is not None
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--batch", type=existing_log, help="a .pdf or .txt batch log")
    parser.add_argument("--json", type=Path, help="also write the report as JSON")
    parser.add_argument("--tag", action="append", metavar="TAG", default=[],
                        help="extra LangSmith run tag; repeatable")
    parser.add_argument("--mermaid", action="store_true", help="print the graph and exit")
    parser.add_argument("--ascii", action="store_true", help="draw the graph in the terminal")
    parser.add_argument(
        "--png",
        type=Path,
        metavar="FILE",
        help="render the graph to a PNG. Uploads the node names to mermaid.ink to do it",
    )
    args = parser.parse_args()

    if args.mermaid or args.ascii or args.png:
        drawable = build_graph().get_graph()
        if args.mermaid:
            print(drawable.draw_mermaid())
        if args.ascii:
            print(drawable.draw_ascii())
        if args.png:
            # draw_method="api" posts the mermaid source to mermaid.ink and returns the image.
            # It is the only renderer that needs nothing installed; the offline alternatives are
            # `--ascii` here, or piping `--mermaid` into mermaid-cli. Nothing about a batch is
            # sent -- the graph is built before any log is read, so this is node names only.
            args.png.write_bytes(drawable.draw_mermaid_png())
            print(f"graph -> {args.png}")
        return 0
    if args.batch is None:
        parser.error("--batch is required (or use --mermaid/--ascii/--png to draw the graph)")

    state = audit_batch(str(args.batch), tags=args.tag)
    print_run(state)

    if args.json and state.get("report") is not None:
        payload = state["report"].model_dump() # type: ignore
        payload["audit_id"] = state.get("audit_id")
        if state.get("usage") is not None:
            total = state["usage"].total_cost # type: ignore
            payload["usage"] = {
                "calls": state["usage"].calls, # type: ignore
                "total_tokens": state["usage"].total_tokens, # type: ignore
                "total_cost_usd": float(total) if total is not None else None,
                "by_node": state["usage"].rows(), # type: ignore
            }
        args.json.write_text(json.dumps(payload, indent=2))
        print(f"\nreport -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
