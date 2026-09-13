# Archived runs

Kept for honesty and history, but not part of the main comparison. The demo's
"Compare runs" tab only lists `results/run_*.json` at the top level — the three
runs there share the same judge (`gemini-3.1-flash-lite`), prompt and corpus,
so any pair of them is comparable.

| File | Why it is here |
|---|---|
| `run_gemini_gemini-3.1-flash-lite_20260912_124547.json` | The project's first complete, clean run. Judged by `gemini-3.5-flash-lite`, so it cannot be compared with the top-level runs (different judge). |
| `run_gemini_gemini-3.1-flash-lite_20260912_123158.json` | Same configuration as the run above, but with 16 metric errors from rate limits (before the judge retry was fixed). Together with the clean run it is a two-sample noise-floor measurement. |
| `run_groq_openai_gpt-oss-120b_20260912_161708.json` | First gpt-oss-120b run, made before the citation-line fix and partly re-scored. Superseded by the fresh 2026-09-13 run. Kept because in this run the model went along with the `case_16` false premise ("the alignment lead resigned in protest") — which did **not** reproduce the next day. |
| `run_gemini_gemini-3.6-flash_20260912_121847.json` | Partial (11/26) with the stronger `gemini-3.6-flash`, stopped by its 20-requests-per-day cap. |
| `run_gemini_gemini-3.6-flash_20260912_121456.json` | The first 5-case smoke test of the pipeline. |
