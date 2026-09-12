"""
eval_methods/faithfulness.py -- is the answer SUPPORTED BY THE RETRIEVED CONTEXT?

    Question:   "Did the model make anything up?"
    Type:       reference-FREE (needs no expected answer; needs the context)
    Mechanism:  DeepEval's built-in FaithfulnessMetric

How the score is produced (at a high level):
    1. The judge splits the ANSWER into individual factual claims.
    2. For each claim, the judge checks: is it supported by the CONTEXT,
       contradicted by it, or not mentioned at all?
    3. score = supported claims / total claims

So an answer that is 100% true but says things the context never said
scores LOW here -- and that is correct. Faithfulness is not truth; it is
"did you stick to your sources". Truth is correctness.py's job.

Note the abstention sentence contains no factual claim, so a correct
abstention scores 1.0 by construction.
"""

from deepeval.metrics import FaithfulnessMetric

from config.settings import ABSTENTION_MESSAGE, PASS_THRESHOLD
from eval_methods.common import build_test_case, run_metric, strip_sources_line
from eval_methods.judge import get_judge

NAME = "faithfulness"


def evaluate(
    question: str,
    answer: str,
    context: list[str],
    expected_answer: str | None = None,   # unused: reference-free
    expected_behavior: str | None = None,  # unused
) -> dict:
    # cheap path: the abstention sentence makes no factual claim, so there is
    # nothing that could be unfaithful. Scoring it deterministically also stops
    # a judge from calling "did not use the context" a faithfulness failure.
    if answer.strip() == ABSTENTION_MESSAGE:
        return {
            "metric": NAME,
            "score": 1.0,
            "passed": True,
            "reason": "Abstention contains no claims; nothing to contradict the context (string match, no judge call).",
        }

    metric = FaithfulnessMetric(
        threshold=PASS_THRESHOLD,
        model=get_judge(),
        include_reason=True,
        async_mode=False,
    )
    # judge the body only; the citation line is instruction_following's job
    test_case = build_test_case(question, strip_sources_line(answer), context=context)
    return run_metric(NAME, metric, test_case)
