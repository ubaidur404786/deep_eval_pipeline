"""
app/llm_client.py -- the ONLY file in app/ that talks to an LLM provider.

    complete(prompt) -> str

Which provider and model is decided by .env (APP_PROVIDER / APP_MODEL), so
swapping the application model for a comparison run is a one-line .env edit.
Everything else in app/ just calls complete() and never sees a provider name.

Supported providers:
    gemini  -- Google AI Studio free tier (needs GOOGLE_API_KEY)
    groq    -- Groq free tier, OpenAI-compatible API (needs GROQ_API_KEY)
    ollama  -- a local Ollama server (needs nothing but Ollama running)
"""

import os
import time

import requests

from config.settings import APP_MODEL, APP_PROVIDER

# Free tiers rate-limit by requests-per-minute. When we hit the limit we wait
# and try again instead of crashing an evaluation run half-way through.
MAX_ATTEMPTS = 4
BACKOFF_SECONDS = 15


# ---------------------------------------------------------------------------
# provider implementations -- each is a plain function: prompt in, text out
# ---------------------------------------------------------------------------
def _complete_gemini(prompt: str, model: str) -> str:
    from google import genai
    from google.genai import types

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set. Add it to .env (see .env.example).")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        # temperature=0: as deterministic as the model allows. We deliberately
        # do NOT touch thinking_config -- some Gemini 3.x models reject it.
        config=types.GenerateContentConfig(temperature=0),
    )
    return (response.text or "").strip()


GROQ_BASE_URL = "https://api.groq.com/openai/v1"


def _complete_groq(prompt: str, model: str) -> str:
    # Groq exposes the OpenAI chat-completions API, so the openai SDK works
    # unchanged: only base_url, api_key and the model slug differ.
    from openai import OpenAI

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set. Add it to .env (see .env.example).")

    client = OpenAI(base_url=GROQ_BASE_URL, api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return (response.choices[0].message.content or "").strip()


def _complete_ollama(prompt: str, model: str) -> str:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    response = requests.post(
        f"{base_url}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0}},
        timeout=300,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


PROVIDERS = {
    "gemini": _complete_gemini,
    "groq": _complete_groq,
    "ollama": _complete_ollama,
}


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------
def complete(prompt: str, provider: str = APP_PROVIDER, model: str = APP_MODEL) -> str:
    """Send a prompt to the configured provider and return the answer text.

    Retries on rate-limit / transient errors with a fixed back-off.
    """
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider '{provider}'. Choose from: {list(PROVIDERS)}")

    call = PROVIDERS[provider]
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return call(prompt, model)
        except Exception as exc:  # noqa: BLE001 -- provider SDKs raise many types
            if attempt == MAX_ATTEMPTS or not _is_retryable(exc):
                raise
            print(f"  [llm_client] {type(exc).__name__}: retrying in {BACKOFF_SECONDS}s "
                  f"(attempt {attempt}/{MAX_ATTEMPTS})")
            time.sleep(BACKOFF_SECONDS)


def _is_retryable(exc: Exception) -> bool:
    """Rate limits (429) and server hiccups (5xx) are worth retrying; a bad key
    or a bad model name is not."""
    text = str(exc)
    if "PerDay" in text:      # a DAILY cap: waiting 15 s will not help, fail fast
        return False
    return any(code in text for code in ("429", "RESOURCE_EXHAUSTED", "503", "500", "timed out"))
