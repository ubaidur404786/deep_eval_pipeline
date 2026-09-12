"""
eval_methods/judge.py -- the ONLY file in the framework that names a judge provider.

    from eval_methods.judge import get_judge
    judge = get_judge()          # a DeepEval model object, built once

Every metric passes this object to DeepEval as `model=...`. Which provider and
model it is comes from .env (JUDGE_PROVIDER / JUDGE_MODEL), so switching the
judge -- Gemini free tier today, a local Ollama model tomorrow -- is a one-line
.env change and touches no metric code.

The judge is deliberately SEPARATE from app/llm_client.py: the application
model and the judge model are different roles (see README), even when they
happen to be the same model.
"""

import os
import re
import time
from functools import lru_cache

from config.settings import JUDGE_MODEL, JUDGE_PROVIDER

# Free tiers answer "429 ... Please retry in 52s". DeepEval 4.2.2's own Gemini
# retry policy fails to load (GOOGLE_ERROR_POLICY is None), so a 429 would
# surface immediately as a metric error. We retry here instead, sleeping for
# exactly the delay Google asks for. Daily caps ("PerDay") are NOT retried.
MAX_RATE_LIMIT_RETRIES = 6
RETRY_DELAY_PATTERN = re.compile(r"retry in (\d+(?:\.\d+)?)s")


def with_rate_limit_retry(call):
    """Wrap a model call so that per-minute 429s wait and retry."""
    def wrapped(*args, **kwargs):
        for attempt in range(1, MAX_RATE_LIMIT_RETRIES + 1):
            try:
                return call(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 -- provider SDKs raise many types
                text = str(exc)
                if "429" not in text or "PerDay" in text or attempt == MAX_RATE_LIMIT_RETRIES:
                    raise
                match = RETRY_DELAY_PATTERN.search(text)
                delay = float(match.group(1)) + 1 if match else 20.0
                print(f"  [judge] rate limited, waiting {delay:.0f}s (attempt {attempt}/{MAX_RATE_LIMIT_RETRIES})")
                time.sleep(delay)
    return wrapped


@lru_cache(maxsize=1)
def get_judge():
    """Return the configured DeepEval judge model. Built once per process."""
    if JUDGE_PROVIDER == "gemini":
        from deepeval.models import GeminiModel

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set. Add it to .env (see .env.example).")
        judge = GeminiModel(model=JUDGE_MODEL, api_key=api_key, temperature=0)
        judge.generate = with_rate_limit_retry(judge.generate)  # sync path (async_mode=False)
        return judge

    if JUDGE_PROVIDER == "groq":
        # Groq is OpenAI-compatible; DeepEval's LocalModel is "any OpenAI-
        # compatible endpoint, parse JSON from the reply" -- exactly that.
        from deepeval.models import LocalModel

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set. Add it to .env (see .env.example).")
        return LocalModel(
            model=JUDGE_MODEL,
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            temperature=0,
        )

    if JUDGE_PROVIDER == "ollama":
        from deepeval.models import OllamaModel

        return OllamaModel(
            model=JUDGE_MODEL,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0,
        )

    raise ValueError(f"Unknown JUDGE_PROVIDER '{JUDGE_PROVIDER}'. Use 'gemini', 'groq' or 'ollama'.")


def judge_name() -> str:
    """Human-readable label for result files, e.g. 'gemini/gemini-3.6-flash'."""
    return f"{JUDGE_PROVIDER}/{JUDGE_MODEL}"
