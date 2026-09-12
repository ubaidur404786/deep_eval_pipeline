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

from config.settings import PASS_THRESHOLD
from eval_methods.common import build_test_case, run_metric
from eval_methods.judge import get_judge

NAME = "faithfulness"


def evaluate(
    question: str,
    answer: str,
    context: list[str],
    expected_answer: str | None = None,   # unused: reference-free
    expected_behavior: str | None = None,  # unused
) -> dict:
    metric = FaithfulnessMetric(
        threshold=PASS_THRESHOLD,
        model=get_judge(),
        include_reason=True,
        async_mode=False,
    )
    test_case = build_test_case(question, answer, context=context)
    return run_metric(NAME, metric, test_case)
