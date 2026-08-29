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
from pathlib import Path

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


def audit_batch(batch_path: str) -> AgentState:
    return build_graph().invoke(initial_state(batch_path)) # type: ignore


def print_run(state: AgentState) -> None:
    entities = state.get("extracted_entities", {})
    report = state.get("report")

    print(f"\nbatch      : {entities.get('batch')}")
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

    state = audit_batch(str(args.batch))
    print_run(state)

    if args.json and state.get("report") is not None:
        args.json.write_text(json.dumps(state["report"].model_dump(), indent=2)) # type: ignore
        print(f"\nreport -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
