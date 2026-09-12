"""
eval_methods/common.py -- the two helpers every metric file shares.

    build_test_case(...)  our plain arguments  ->  DeepEval's LLMTestCase
    run_metric(metric, test_case)              ->  {metric, score, passed, reason}

Every metric file has the same public function:

    evaluate(question, answer, context, expected_answer=None, expected_behavior=None) -> dict

The framework never imports from app/. It receives strings and lists; where
they came from is not its business. That is what makes it reusable.
"""

import re

from deepeval.test_case import LLMTestCase

# "Sources: ..." to the end of the answer (the citation line the system prompt
# requires). Matches whether it sits on its own line or inline after the text.
SOURCES_PATTERN = re.compile(r"\s*sources?\s*:.*\Z", re.IGNORECASE | re.DOTALL)


def strip_sources_line(answer: str) -> str:
    """Return the answer body without its citation line.

    Faithfulness and relevance judge WHAT the answer says; the citation line is
    a formatting rule judged by instruction_following. Left in, judges call
    the citation "irrelevant meta-information" or read a cited article TITLE as
    a factual claim. One concern per metric -- so each metric sees only its part."""
    body = SOURCES_PATTERN.sub("", answer).strip()
    return body or answer  # never hand the judge an empty string


def build_test_case(
    question: str,
    answer: str,
    context: list[str] | None = None,
    expected_answer: str | None = None,
) -> LLMTestCase:
    """Map our vocabulary onto DeepEval's.

        question         -> input
        answer           -> actual_output      (what the application said)
        context          -> retrieval_context  (what the application saw)
        expected_answer  -> expected_output    (golden reference)
    """
    return LLMTestCase(
        input=question,
        actual_output=answer,
        retrieval_context=context,
        expected_output=expected_answer,
    )


def run_metric(name: str, metric, test_case: LLMTestCase) -> dict:
    """Run one DeepEval metric and flatten its result into a plain dict."""
    metric.measure(test_case, _show_indicator=False)  # no spinner in logs
    return {
        "metric": name,
        "score": round(float(metric.score), 3) if metric.score is not None else None,
        "passed": bool(metric.success),
        "reason": (metric.reason or "").strip(),
    }
