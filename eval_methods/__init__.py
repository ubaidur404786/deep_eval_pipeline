"""
eval_methods -- the reusable evaluation framework.

    from eval_methods import METRICS
    result = METRICS["faithfulness"](question, answer, context)

METRICS maps a metric name (as used in golden_data/test_cases.json) to its
evaluate() function. Adding a metric = one new file + one line here.

Nothing in this package imports from app/. It evaluates strings and lists;
any application that can produce them can be evaluated.
"""

import os

# --- DeepEval environment, set BEFORE deepeval is imported anywhere ----------
# Python runs this __init__ before any `eval_methods.<module>` import, so these
# are guaranteed to be in place when DeepEval reads its settings.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")   # no usage phone-home
# Rate-limit retries are handled in judge.py (DeepEval's Gemini retry policy
# does not load in 4.2.2). Our retry SLEEPS inside the call while waiting for
# the quota window, so DeepEval's own per-attempt timeout (~90 s) must not
# fire in the middle of that wait.
os.environ.setdefault("DEEPEVAL_DISABLE_TIMEOUTS", "1")

from eval_methods import correctness, faithfulness, instruction_following, relevance  # noqa: E402

METRICS = {
    correctness.NAME: correctness.evaluate,
    faithfulness.NAME: faithfulness.evaluate,
    relevance.NAME: relevance.evaluate,
    instruction_following.NAME: instruction_following.evaluate,
}
