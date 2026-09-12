"""
eval_methods/correctness.py -- is what the answer says TRUE, compared to the golden answer?

    Question:   "Are the facts right?"
    Type:       reference-BASED (needs expected_answer from the golden dataset)
    Mechanism:  GEval -- a custom rubric that the judge follows

What GEval is:
    You write grading instructions in plain English (evaluation_steps) and a
    rubric that maps score bands to outcomes. The judge LLM reads the question,
    the answer and the expected answer, follows your steps, and returns a
    score out of 10 with a written reason. DeepEval scales it to 0-1.

    Because the instructions are ours, we can say precisely what this metric
    must NOT do: penalise brevity, missing points, or style. Those belong to
    completeness (not yet built) and relevance. Keeping the rubric narrow is
    what keeps the metric meaningful.

Limitations to keep in mind:
    - The judge can be wrong; read the reason, not just the number.
    - Scores wobble a little run-to-run even at temperature 0.
    - If the expected answer is wrong, the metric is wrong. Golden data quality
      is the ceiling of this metric.
"""

from deepeval.metrics import GEval
from deepeval.metrics.g_eval import Rubric
from deepeval.test_case import LLMTestCaseParams

from config.settings import PASS_THRESHOLD
from eval_methods.common import build_test_case, run_metric
from eval_methods.judge import get_judge

NAME = "correctness"

EVALUATION_STEPS = [
    "Compare only the factual claims in the actual output against the expected output.",
    "A claim is wrong if it contradicts the expected output or states something the expected output shows to be false.",
    "Judge truth, not coverage: an answer that is shorter than the expected output but says nothing false must still score high.",
    "Do NOT deduct for brevity, omitted points, tone, formatting, or a missing 'Sources:' line.",
    "Extra correct information must never lower the score.",
    "If the actual output makes NO claim that contradicts the expected output, score it 9-10 even if it fails to actually answer the question -- not answering is a relevance problem, not a correctness problem.",
    "If the expected output says the question's premise is false and the actual output goes along with the premise, that is a factual error.",
]

RUBRIC = [
    Rubric(score_range=(9, 10), expected_outcome="Every stated claim is consistent with the expected output. No contradictions. Brevity is fine."),
    Rubric(score_range=(5, 8), expected_outcome="Mostly consistent, but one minor inaccuracy (a slightly wrong number, name, or attribution)."),
    Rubric(score_range=(0, 4), expected_outcome="A clear factual error, an invented detail, or a claim that contradicts the expected output."),
]


def evaluate(
    question: str,
    answer: str,
    context: list[str] | None = None,     # unused: correctness is judged against the reference, not the context
    expected_answer: str | None = None,
    expected_behavior: str | None = None,  # unused
) -> dict:
    if not expected_answer:
        raise ValueError("correctness needs an expected_answer (reference-based metric)")

    metric = GEval(
        name="Correctness",
        evaluation_steps=EVALUATION_STEPS,
        rubric=RUBRIC,
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        threshold=PASS_THRESHOLD,
        model=get_judge(),
        async_mode=False,
    )
    test_case = build_test_case(question, answer, expected_answer=expected_answer)
    return run_metric(NAME, metric, test_case)
