"""LLM-as-judge evaluation of the generated reports (§8).

The other 190 tests check things with a right answer -- does ``5669,49`` parse to 5669.49, does a
fabricated citation get vetoed. **None of them can tell you whether a report is any good.** That
is not an oversight: "is this finding well reasoned?" has no assertable answer, so until this file
existed, rewording a prompt was evaluated by reading one report and forming an impression.

Three metrics, each isolating a different failure (§8.1):

* **Faithfulness** -- does every claim rest on the retrieved clauses? Catches hallucination.
* **Answer Relevancy** -- does the report answer what was asked? Catches drift.
* **Contextual Precision** -- did retrieval rank the *useful* clauses above the noise? This is the
  one worth caring about most, because it separates "the model wrote badly" from "the model never
  received the right law". Those need opposite fixes and are indistinguishable from the output.

Not part of the default suite. The judge is a ``gpt-4o`` call per metric per case -- 12 calls for
four batches -- so it is gated behind a marker and runs when asked::

    uv run python -m src.graph.evalset      # capture runs (~$0.13/batch)
    uv run pytest tests/eval_suite.py -m eval -v

``uv run pytest tests/`` stays free and needs no key, exactly as before.
"""

from __future__ import annotations

import json
import os

import pytest

from src.graph.evalset import CASES_PATH

pytestmark = pytest.mark.eval

# Thresholds are the blueprint's for the two metrics it names. Contextual Precision has no
# blueprint figure and 0.70 is a starting line, not a measured one -- the first honest run is what
# sets it. A threshold invented to be comfortably passed measures nothing.
FAITHFULNESS_THRESHOLD = 0.85
RELEVANCY_THRESHOLD = 0.80
PRECISION_THRESHOLD = 0.70

JUDGE_MODEL = os.environ.get("EVAL_MODEL", "gpt-4o")

# DeepEval defaults every metric to async, which fires all four cases at once and drowned the
# first run in `tenacity.RetryError` -- rate limiting, surfacing as an opaque retry exhaustion
# rather than as a 429. Serial is slower (~4 minutes for 16 judgements) and actually finishes.
# The same three cases score 1.00 serially that failed to score at all in parallel.
ASYNC = False

# DeepEval reprints a progress banner on every internal step, tens of thousands of characters per
# run, which buries the reasons -- the one thing an LLM judge is for.
os.environ.setdefault("DEEPEVAL_DISABLE_PROGRESS_BAR", "1")
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")


def load_cases() -> list[dict]:
    if not CASES_PATH.exists():
        pytest.skip(f"{CASES_PATH.name} missing -- run: uv run python -m src.graph.evalset")
    cases = json.loads(CASES_PATH.read_text())
    if not cases:
        pytest.skip("no captured cases")
    return cases


def case_ids(cases: list[dict]) -> list[str]:
    return [case["name"].replace("_private_banking_log.pdf", "") for case in cases]


CASES = json.loads(CASES_PATH.read_text()) if CASES_PATH.exists() else []
PARAMS = pytest.mark.parametrize("case", CASES, ids=case_ids(CASES) if CASES else None)


def as_test_case(case: dict):
    from deepeval.test_case import LLMTestCase

    return LLMTestCase(
        input=case["input"],
        actual_output=case["actual_output"],
        expected_output=case["expected_output"],
        retrieval_context=case["retrieval_context"],
    )


SCORES_PATH = CASES_PATH.parent / "eval_scores.json"


def _diagnose(error: Exception) -> BaseException:
    """Unwrap DeepEval's retry wrapper so the real failure is readable.

    Every judge call goes through tenacity, so an exhausted account surfaces as
    ``tenacity.RetryError[<Future ... state=finished raised ...>]`` with the actual message buried
    three layers down. Three separate runs were spent reading that as rate limiting before the
    real text turned out to be *"You have no credits remaining"*. Once is enough.
    """
    cause: BaseException = error
    for _ in range(5):  # bounded: a cycle here would hang the suite
        nxt = getattr(cause, "__cause__", None)
        if nxt is None and hasattr(cause, "last_attempt"):
            nxt = cause.last_attempt.exception() # type: ignore[attr-defined]
        if nxt is None or nxt is cause:
            break
        cause = nxt

    text = str(cause)
    if "credit_balance_exhausted" in text or "insufficient_quota" in text:
        # Not a failing report -- no report was scored at all. Failing here would record a
        # quality regression that never happened.
        pytest.skip("OpenAI account has no credits -- the judge could not run")
    return cause


def measure(metric, case: dict) -> float:
    """Score one case, record it, and report the judge's reasoning on failure.

    ``assert_test`` would do this in one line, but it raises on the first failing metric and
    prints the score without the reason. A number with no explanation cannot be acted on, and the
    reason is the entire product of an LLM judge.

    Every score is written to ``eval_scores.json``, passing ones included. A suite that only
    surfaces failures cannot answer "did that prompt change help?" -- the run before had no record
    of what it scored, so there is nothing to compare against.
    """
    try:
        metric.measure(as_test_case(case))
    except Exception as error:  # noqa: BLE001 -- unwrapped and re-raised below
        raise _diagnose(error) from error

    recorded = json.loads(SCORES_PATH.read_text()) if SCORES_PATH.exists() else {}
    recorded.setdefault(case["name"], {})[type(metric).__name__] = {
        "score": round(metric.score, 4),
        "threshold": metric.threshold,
        "passed": metric.score >= metric.threshold,
        "reason": metric.reason,
    }
    SCORES_PATH.write_text(json.dumps(recorded, indent=2, sort_keys=True))
    return metric.score


# --- Faithfulness: does the report invent anything? -------------------------------------


@PARAMS
def test_the_report_claims_only_what_the_clauses_support(case):
    from deepeval.metrics import FaithfulnessMetric

    metric = FaithfulnessMetric(threshold=FAITHFULNESS_THRESHOLD, model=JUDGE_MODEL,
                                include_reason=True, async_mode=ASYNC)
    score = measure(metric, case)
    assert score >= FAITHFULNESS_THRESHOLD, (
        f"{case['name']} faithfulness {score:.3f} < {FAITHFULNESS_THRESHOLD}\n{metric.reason}"
    )


# --- Answer Relevancy: does it answer the question asked? -------------------------------


@PARAMS
def test_the_report_addresses_the_audit_that_was_requested(case):
    from deepeval.metrics import AnswerRelevancyMetric

    metric = AnswerRelevancyMetric(threshold=RELEVANCY_THRESHOLD, model=JUDGE_MODEL,
                                   include_reason=True, async_mode=ASYNC)
    score = measure(metric, case)
    assert score >= RELEVANCY_THRESHOLD, (
        f"{case['name']} relevancy {score:.3f} < {RELEVANCY_THRESHOLD}\n{metric.reason}"
    )


# --- Contextual Precision: did retrieval put the useful clauses first? ------------------


@PARAMS
def test_retrieval_ranks_the_useful_clauses_above_the_noise(case):
    """Measured against the *planted* typologies, not against what the report chose to say.

    This is where `ledger_labels.csv` earns its place: without it the suite could only ask whether
    a report is internally consistent, which a confidently wrong report passes.
    """
    from deepeval.metrics import ContextualPrecisionMetric

    metric = ContextualPrecisionMetric(threshold=PRECISION_THRESHOLD, model=JUDGE_MODEL,
                                       include_reason=True, async_mode=ASYNC)
    score = measure(metric, case)
    assert score >= PRECISION_THRESHOLD, (
        f"{case['name']} context precision {score:.3f} < {PRECISION_THRESHOLD}\n{metric.reason}"
    )


# --- what the metrics cannot see --------------------------------------------------------


@PARAMS
def test_a_clean_batch_is_not_reported_as_a_finding(case):
    """The open defect, asserted rather than described.

    Across four live runs every batch came back Medium -- including May, which has zero
    laundering wires in the answer key, and July, which has 23. The rating does not yet separate
    a clean batch from a dirty one. No LLM judge catches this: each report is individually
    plausible, and only the answer key knows better.

    Expected to fail on May until the rating is calibrated. That is the point of writing it down.
    """
    truth = case["ground_truth"]
    rating = case["run"]["risk_rating"]
    if truth["laundering_wires"] == 0:
        assert rating == "Low", (
            f"{case['name']} has no laundering in the answer key but was rated {rating}"
        )
    else:
        assert rating in {"Medium", "High"}, (
            f"{case['name']} has {truth['laundering_wires']} laundering wires but was rated "
            f"{rating}"
        )
