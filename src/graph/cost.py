"""Per-node token and dollar accounting for a run (§9.1).

The blueprint quotes $0.00 for the deterministic path against $0.12 for the agentic one, but a
quoted figure is not a measured one -- and the figure most worth knowing is not the average. A
batch whose critic passes first time costs three model calls; one that loops costs five, and
until this module existed nothing in the system said which had happened.

Attribution comes from the callback's ``tags``. ``nodes.trace_config`` already stamps every call
with ``node:draft`` and friends for §7.2's tracing, so the same label serves both purposes rather
than introducing a second, parallel notion of "which node was that".

Reading usage off the response is not an option: three of the four calls go through
``with_structured_output``, which returns the parsed Pydantic object and discards the
``AIMessage`` the token counts live on. A callback sees the raw generation first, so it works
with structured and unstructured calls alike without changing what any node returns.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import ChatGeneration, LLMResult

# USD per million tokens, as published. A snapshot: prices move, and when they do this table is
# the single place to correct. A model absent from it is reported in tokens with no dollar figure
# rather than being priced off the nearest-looking entry -- an invented cost is worse than none,
# because it reads as measured.
PRICES: dict[str, tuple[Decimal, Decimal]] = {
    "gpt-4o": (Decimal("2.50"), Decimal("10.00")),
    "gpt-4o-mini": (Decimal("0.150"), Decimal("0.600")),
}

PER_MILLION = Decimal(1_000_000)
NODE_TAG = "node:"


def price_of(model: str) -> tuple[Decimal, Decimal] | None:
    """Input/output price for a model, tolerating the dated suffixes OpenAI returns.

    A response says ``gpt-4o-2024-08-06`` where the request said ``gpt-4o``. Longest prefix wins,
    so ``gpt-4o-mini-2024-07-18`` matches the mini entry rather than the plain one.
    """
    for name in sorted(PRICES, key=len, reverse=True):
        if model == name or model.startswith(f"{name}-"):
            return PRICES[name]
    return None


def cost_of(model: str, input_tokens: int, output_tokens: int) -> Decimal | None:
    prices = price_of(model)
    if prices is None:
        return None
    per_input, per_output = prices
    return (per_input * input_tokens + per_output * output_tokens) / PER_MILLION


@dataclass
class NodeUsage:
    """What one node spent across however many times it ran."""

    node: str
    model: str = ""
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cost(self) -> Decimal | None:
        return cost_of(self.model, self.input_tokens, self.output_tokens)

    def as_row(self) -> dict[str, Any]:
        cost = self.cost
        return {
            "node": self.node,
            "model": self.model,
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": float(cost) if cost is not None else None,
        }


# eq=False keeps the default identity __hash__. A plain @dataclass generates __eq__, which sets
# __hash__ to None, and LangChain merges callback handlers through `set(self.handlers)` when a
# run-level callback meets a per-call config -- exactly what §7.2's trace_config creates. An
# unhashable handler raises there, not here, halfway through a paid run.
@dataclass(eq=False)
class UsageLedger(BaseCallbackHandler):
    """Accumulates token usage per node for one run.

    One instance per run, handed to LangGraph as a run-level callback: LangChain propagates those
    down to every nested call, so a node added later is accounted for without being registered
    anywhere. The lock is not decoration -- LangGraph may run nodes concurrently, and a lost
    update here would silently under-report a bill.
    """

    nodes: dict[str, NodeUsage] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID | None = None,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            generation = response.generations[0][0]
        except IndexError:
            return
        if not isinstance(generation, ChatGeneration):
            return

        message = generation.message
        usage = getattr(message, "usage_metadata", None)
        if not usage:
            return

        node = next(
            (tag[len(NODE_TAG):] for tag in (tags or []) if tag.startswith(NODE_TAG)),
            "untagged",
        )
        model = message.response_metadata.get("model_name", "") if message else ""

        with self._lock:
            entry = self.nodes.setdefault(node, NodeUsage(node=node))
            entry.model = entry.model or model
            entry.calls += 1
            entry.input_tokens += usage.get("input_tokens", 0)
            entry.output_tokens += usage.get("output_tokens", 0)

    # --- reporting ---------------------------------------------------------------------

    @property
    def calls(self) -> int:
        return sum(entry.calls for entry in self.nodes.values())

    @property
    def total_tokens(self) -> int:
        return sum(entry.total_tokens for entry in self.nodes.values())

    @property
    def total_cost(self) -> Decimal | None:
        """None when any node's model is unpriced -- a partial total is a misleading one."""
        costs = [entry.cost for entry in self.nodes.values()]
        if not costs or any(cost is None for cost in costs):
            return None
        return sum(costs, Decimal(0))

    def rows(self) -> list[dict[str, Any]]:
        """Ordered as the graph runs them, so the table reads as the pipeline does."""
        order = ["parse", "extraction_fallback", "detect", "audit", "draft", "critic", "generate"]
        ranked = sorted(
            self.nodes.values(),
            key=lambda entry: (order.index(entry.node) if entry.node in order else len(order),
                               entry.node),
        )
        return [entry.as_row() for entry in ranked]

    def summary(self) -> str:
        if not self.nodes:
            # The free path. §9.1's $0.00 batch is a result worth stating, not an empty table.
            return "cost       : $0.0000 -- no model was called"

        lines = [
            f"cost       : {self.calls} model call(s), {self.total_tokens:,} tokens",
        ]
        for row in self.rows():
            money = f"${row['cost_usd']:.4f}" if row["cost_usd"] is not None else "unpriced"
            lines.append(
                f"    {row['node']:<20} {row['calls']} call(s)  "
                f"{row['input_tokens']:>7,} in  {row['output_tokens']:>6,} out  {money:>10}"
                f"   {row['model']}"
            )
        total = self.total_cost
        lines.append(
            f"    {'TOTAL':<20} "
            + (f"${total:.4f}" if total is not None else "unpriced -- unknown model in the run")
        )
        return "\n".join(lines)
