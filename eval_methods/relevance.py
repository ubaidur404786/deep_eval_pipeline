"""
eval_methods/relevance.py -- does the answer ACTUALLY ADDRESS THE QUESTION?

    Question:   "Is this an answer to what was asked, without padding?"
    Type:       reference-FREE (needs only question + answer)
    Mechanism:  DeepEval's built-in AnswerRelevancyMetric

How the score is produced (at a high level):
    1. The judge splits the ANSWER into statements.
    2. For each statement it asks: is this relevant to the QUESTION?
    3. score = relevant statements / total statements

So a true, well-grounded answer that also rambles about three unrelated
articles scores LOW here. Relevance does not care whether the statements
are correct (correctness.py) or grounded (faithfulness.py) -- only whether
they belong in a reply to this question.
"""

from deepeval.metrics import AnswerRelevancyMetric

from config.settings import PASS_THRESHOLD
from eval_methods.common import build_test_case, run_metric, strip_sources_line
from eval_methods.judge import get_judge

NAME = "relevance"


def evaluate(
    question: str,
    answer: str,
    context: list[str] | None = None,     # unused: judged on question + answer only
    expected_answer: str | None = None,   # unused
    expected_behavior: str | None = None,  # unused
) -> dict:
    metric = AnswerRelevancyMetric(
        threshold=PASS_THRESHOLD,
        model=get_judge(),
        include_reason=True,
        async_mode=False,
    )
    # judge the body only; the citation line is instruction_following's job
    test_case = build_test_case(question, strip_sources_line(answer))
    return run_metric(NAME, metric, test_case)
