# Evaluation flow, in plain English

This document explains what happens when you run

```
python -m pipeline.run_evaluation
```

from the moment a question is read to the moment a score is saved.

---

## The one-line version

```
Question → Application → Answer → Golden reference → LLM judge → Metric → Score + reason
```

## The roles

| Role | Who plays it here | What it does |
|---|---|---|
| **Application** | `app/` — the AI & Technology News Assistant | Takes a question, retrieves news chunks, writes an answer |
| **Golden dataset** | `golden_data/test_cases.json` | Says what a *good* answer or behaviour looks like for each question |
| **Judge** | a free LLM (`eval_methods/judge.py`) | Reads answer + reference, follows a rubric, returns a score and a reason |
| **Metric** | one file in `eval_methods/` | Asks the judge one narrow question (truth? grounding? focus? behaviour?) |
| **Pipeline** | `pipeline/run_evaluation.py` | Wires the above together and saves results |

Two things that are easy to confuse:

- The **golden dataset** is *what we expect*. It was drafted with AI assistance, checked by hand against a frozen copy of the news corpus, and fixed before any evaluation ran.
- The **judge** is *who checks*. It is a model that runs at evaluation time. It never sees the golden data while the application is answering; it only sees it afterwards, to grade.

---

## Step by step

### 1. Load the golden cases

`load_golden_data()` reads `test_cases.json` and validates every case:
required fields present, ids unique, category and behaviour from the known
sets, metric names known, `correctness` only where an `expected_answer` exists,
and every `source_id` present in the frozen corpus.

A case looks like this:

```json
{
  "id": "case_16",
  "category": "hallucination_trap",
  "question": "Why did Anthropic's alignment lead resign this week?",
  "expected_answer": "The alignment lead did not resign. A researcher resigned ...",
  "expected_behavior": "correct_assumption",
  "metrics": ["instruction_following", "faithfulness", "correctness"],
  "source_ids": ["techcrunch_ai_0c8057bd"],
  "notes": "Role confusion trap."
}
```

### 2. Run the application

`answer_question(question)` does the whole RAG job:

```
question
  → embed with all-MiniLM-L6-v2
  → 4 nearest chunks from Chroma           (retrieval)
  → system_prompt.txt + chunks + question  (prompt assembly)
  → application LLM                        (generation)
  → {"query", "context", "answer", "sources"}
```

The dict it returns is **the contract**. The framework needs exactly three keys
from any application it evaluates: `query`, `context` (the chunks the model
actually saw) and `answer`.

### 3. A free check before any judge is called

The pipeline compares the retrieved `source_ids` with the case's expected
`source_ids` and records `expected_sources_found`, e.g. `1/2`. No LLM is
involved. If this says `0/2`, whatever the judge says next is a *retrieval*
problem, not a *model* problem. This one number explains most failures.

### 4. Run only the metrics the case asks for

Each case lists its own metrics. An abstention case has no correct answer to
compare against, so correctness is simply not run on it. This is how we avoid
fake zeros polluting averages.

Every metric has the same signature:

```python
evaluate(question, answer, context, expected_answer=None, expected_behavior=None) -> dict
```

and returns the same shape:

```python
{"metric": "faithfulness", "score": 0.75, "passed": True, "reason": "..."}
```

### 5. What each metric actually does

| Metric | Question it asks | Needs | Mechanism |
|---|---|---|---|
| `correctness` | Is what the answer says **true**, compared to the golden answer? | `expected_answer` | GEval rubric (our own instructions) |
| `faithfulness` | Is every claim **supported by the retrieved context**? | `context` | DeepEval built-in: split answer into claims, verdict each |
| `relevance` | Does the answer **address the question** without padding? | nothing extra | DeepEval built-in: split into statements, verdict each |
| `instruction_following` | Did it **behave** as the case requires (abstain / decline / clarify / cite)? | `expected_behavior` | exact-string check, else GEval rubric |

**GEval** means: we write the grading steps in English, the judge follows them
and returns 0–10 with a reason, DeepEval scales to 0–1. Each rubric explicitly
forbids the metric from grading anything outside its own concern — correctness
must not penalise brevity, relevance must not reward truth, and so on. Without
those "do NOT" lines the four metrics quietly collapse into one.

**Why four, not one.** The same answer can be:

- *true but unfaithful* — right from memory, not from the sources; will be wrong tomorrow
- *faithful but wrong* — loyally repeated a mistake in the source
- *true, faithful, and irrelevant* — accurate facts that don't answer the question
- *true, faithful, relevant, and disobedient* — dropped the Sources line because the user asked

Only running all four tells these apart. The smoke test in the README shows each pattern.

### 6. Pass / fail

A metric passes when `score >= PASS_THRESHOLD` (0.7 in `config/settings.py`).
A **case** passes only when *all* its metrics pass.

### 7. Save after every case

`results/latest_results.json` is rewritten after each case, so a rate-limit
stop half-way loses nothing. A timestamped copy is kept per run. The file
records provenance:

```json
"run": {
  "app_model": "gemini/gemini-3.1-flash-lite",
  "judge_model": "gemini/gemini-3.5-flash-lite",
  "prompt_hash": "8caaebbe2646",
  "snapshot_fetched_at": "2026-09-11T23:21:30+00:00"
}
```

Two runs are comparable only if everything here matches except the one thing
you meant to change (usually `app_model`).

### 8. Summarise — never pool

Per-metric averages and pass rates are reported separately, plus the minimum
score. They are never averaged into a single "quality score": a collapse in
faithfulness must not hide behind a strong relevance average.

---

## Reading a result

```
case_19  [contradiction]  relevance=0.667, instruction_following=0.0
```

Open the case in `latest_results.json` and read the **reason**, not just the
number:

> "The actual output only reports a single figure (99%) and fails to mention
> the conflict or present the second version from the sources."

That is a genuine application failure — rule 4 of the system prompt says
"report both". Compare with:

```
case_01  [factual]  faithfulness=0.0
```

> "...states that Aarons was fined for including AI-fabricated witnesses in a
> legal brief, whereas the context clarifies this occurred in a murder
> conviction appeal."

That is not a contradiction; it is a **judge error**. LLM judges are useful
and fallible. The score is a claim; the reason is the evidence. When they
disagree, tighten the rubric or try a stronger judge — never just trust the number.

---

## Reusing the framework on another application

1. Write a function that returns `{"query", "context", "answer"}`.
2. Change one import in `pipeline/run_evaluation.py`.
3. Write golden cases for the new domain in the same schema.
4. Run. Nothing in `eval_methods/` changes.

`eval_methods/` never imports from `app/`. That single rule is what makes it a
framework instead of a script.
