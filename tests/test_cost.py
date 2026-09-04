"""Per-node token and dollar accounting (§9.1).

No API key and no network: the ledger is a callback, so it can be driven with the same
``LLMResult`` LangChain would hand it.
"""

from __future__ import annotations

from decimal import Decimal

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from src.graph.cost import PRICES, UsageLedger, cost_of, price_of


def result(input_tokens: int, output_tokens: int, model: str = "gpt-4o-2024-08-06") -> LLMResult:
    message = AIMessage(
        content="x",
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
        response_metadata={"model_name": model},
    )
    return LLMResult(generations=[[ChatGeneration(message=message)]])


def feed(ledger: UsageLedger, node: str, *args, **kwargs) -> None:
    ledger.on_llm_end(result(*args, **kwargs), tags=[f"node:{node}", "AML_AUDIT_RUN"])


def test_the_ledger_can_live_in_a_set_of_callback_handlers():
    """LangChain merges run-level callbacks with per-call ones through `set(handlers)`. A plain
    @dataclass would be unhashable there and raise mid-run, after the money was already spent."""
    ledger = UsageLedger()
    assert {ledger, ledger} == {ledger}
    assert ledger in {UsageLedger(), ledger}


# --- pricing ----------------------------------------------------------------------------


def test_a_dated_model_name_is_priced_as_its_family():
    """The request says gpt-4o; the response says gpt-4o-2024-08-06."""
    assert price_of("gpt-4o-2024-08-06") == PRICES["gpt-4o"]


def test_the_longest_prefix_wins_so_mini_is_not_priced_as_full():
    """gpt-4o-mini-2024-07-18 starts with 'gpt-4o-'. Priced as gpt-4o it would be 17x too dear."""
    assert price_of("gpt-4o-mini-2024-07-18") == PRICES["gpt-4o-mini"]
    assert PRICES["gpt-4o-mini"] != PRICES["gpt-4o"]


def test_an_unknown_model_is_not_priced_off_the_nearest_looking_entry():
    """An invented cost is worse than none: it reads as measured."""
    assert price_of("claude-opus-5") is None
    assert cost_of("claude-opus-5", 1000, 1000) is None


def test_cost_is_exact_money_not_a_float():
    # 10,000 in at $2.50/M + 2,000 out at $10.00/M = 0.025 + 0.020
    assert cost_of("gpt-4o", 10_000, 2_000) == Decimal("0.045")


# --- the ledger -------------------------------------------------------------------------


def test_usage_is_attributed_to_the_node_that_spent_it():
    ledger = UsageLedger()
    feed(ledger, "draft", 12_000, 900)
    feed(ledger, "critic", 13_000, 300)

    assert set(ledger.nodes) == {"draft", "critic"}
    assert ledger.nodes["draft"].input_tokens == 12_000
    assert ledger.nodes["critic"].output_tokens == 300


def test_a_node_that_ran_twice_is_summed_not_overwritten():
    """A looping run calls draft and critic twice. Reporting only the last would halve the bill."""
    ledger = UsageLedger()
    feed(ledger, "draft", 12_000, 900)
    feed(ledger, "draft", 14_000, 1_100)

    entry = ledger.nodes["draft"]
    assert entry.calls == 2
    assert (entry.input_tokens, entry.output_tokens) == (26_000, 2_000)


def test_the_fallback_is_costed_at_its_own_models_price():
    """escalate() uses gpt-4o-mini. Charging it at gpt-4o would overstate the rescue 17-fold."""
    ledger = UsageLedger()
    feed(ledger, "extraction_fallback", 500, 200, model="gpt-4o-mini-2024-07-18")
    assert ledger.nodes["extraction_fallback"].cost == cost_of("gpt-4o-mini", 500, 200)


def test_a_call_with_no_usage_reported_is_skipped_not_counted_as_zero():
    ledger = UsageLedger()
    empty = LLMResult(generations=[[ChatGeneration(message=AIMessage(content="x"))]])
    ledger.on_llm_end(empty, tags=["node:draft"])
    assert ledger.nodes == {} and ledger.calls == 0


def test_an_untagged_call_is_still_counted():
    """Better a row labelled 'untagged' than a call that quietly does not appear in the total."""
    ledger = UsageLedger()
    ledger.on_llm_end(result(100, 50), tags=None)
    assert ledger.nodes["untagged"].calls == 1


def test_one_unpriced_model_withholds_the_total_rather_than_understating_it():
    ledger = UsageLedger()
    feed(ledger, "draft", 12_000, 900)
    feed(ledger, "critic", 1_000, 100, model="some-local-model")

    assert ledger.nodes["draft"].cost is not None
    assert ledger.total_cost is None, "a partial total is a misleading one"
    assert "unpriced" in ledger.summary()


def test_rows_read_in_the_order_the_graph_runs_them():
    ledger = UsageLedger()
    for node in ("generate", "draft", "critic"):
        feed(ledger, node, 1_000, 100)
    assert [row["node"] for row in ledger.rows()] == ["draft", "critic", "generate"]


def test_the_free_path_states_its_result_rather_than_printing_an_empty_table():
    """§9.1's $0.00 batch is a finding: the model was never reached."""
    ledger = UsageLedger()
    assert ledger.calls == 0
    assert ledger.total_cost is None
    assert "$0.0000" in ledger.summary() and "no model was called" in ledger.summary()


def test_the_summary_carries_every_node_and_a_total():
    ledger = UsageLedger()
    feed(ledger, "draft", 12_000, 900)
    feed(ledger, "critic", 13_000, 300)
    summary = ledger.summary()

    assert "draft" in summary and "critic" in summary
    expected = cost_of("gpt-4o", 25_000, 1_200)
    assert f"${expected:.4f}" in summary
