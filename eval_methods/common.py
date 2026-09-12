"""
eval_methods/common.py -- the two helpers every metric file shares.

    build_test_case(...)  our plain arguments  ->  DeepEval's LLMTestCase
    run_metric(metric, test_case)              ->  {metric, score, passed, reason}

Every metric file has the same public function:

    evaluate(question, answer, context, expected_answer=None, expected_behavior=None) -> dict

The framework never imports from app/. It receives strings and lists; where
they came from is not its business. That is what makes it reusable.
"""

from deepeval.test_case import LLMTestCase


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
