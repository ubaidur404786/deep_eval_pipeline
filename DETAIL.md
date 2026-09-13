# DETAIL.md — the full tutorial

This is the long version. It walks through the project the way I built it:
one step at a time, with the command to run and what you should see. If you
only want the big picture, read [README.md](README.md) instead.

Everything here was run on a normal laptop (8 GB RAM, Windows 11, no paid API).
The outputs shown are real, copied from my terminal.

---

## Contents

1. [The idea in one paragraph](#1-the-idea-in-one-paragraph)
2. [The models and what each one does](#2-the-models-and-what-each-one-does)
3. [Setup](#3-setup)
   - [3b. Watch the flow before running anything](#3b-watch-the-flow-before-running-anything)
4. [Step 1 — Freeze the news corpus](#4-step-1--freeze-the-news-corpus)
5. [Step 2 — Build the vector store](#5-step-2--build-the-vector-store)
6. [Step 3 — Ask the assistant](#6-step-3--ask-the-assistant)
7. [Step 4 — The golden test cases](#7-step-4--the-golden-test-cases)
8. [Step 5 — The four metrics](#8-step-5--the-four-metrics)
9. [Step 6 — Run the evaluation](#9-step-6--run-the-evaluation)
10. [Step 7 — Read the results](#10-step-7--read-the-results)
11. [Step 8 — The Streamlit demo](#11-step-8--the-streamlit-demo)
12. [Step 9 — Compare two models](#12-step-9--compare-two-models)
13. [Free-tier limits I measured](#13-free-tier-limits-i-measured)
14. [Things that went wrong, and what I learned](#14-things-that-went-wrong-and-what-i-learned)
15. [Reusing the framework on your own app](#15-reusing-the-framework-on-your-own-app)

---

## 1. The idea in one paragraph

A RAG chatbot has two jobs: **find** the right text, then **write** an answer
from it. It can fail at either job, and it can also misbehave — invent facts,
ignore its rules, answer a different question. This project runs a fixed list of
26 test questions through the chatbot and asks a second LLM (the *judge*) to
score each answer on four separate things. Because the questions, the expected
answers, the judge and the scoring rules never change, you can swap the chatbot's
model and get numbers you can actually compare.

## 2. The models and what each one does

There are three models in this project. They have different jobs, and it helps
to keep them apart in your head.

| Model | Job | Which one | Runs where |
|---|---|---|---|
| **Embedding model** | Turns text into a list of numbers so we can search by meaning | `all-MiniLM-L6-v2` (~90 MB) | Your CPU, downloaded once |
| **Application model** | Writes the answer to the user's question | `gemini-3.1-flash-lite` (default) or `openai/gpt-oss-120b` via Groq | Free API |
| **Judge model** | Reads the answer and grades it | `gemini-3.5-flash-lite` (default) | Free API |

Two rules I follow:

- **The judge and the application should be different models.** A model should
  not grade itself, and on free tiers each model has its own quota bucket, so
  splitting them doubles your headroom.
- **The judge stays fixed when you compare application models.** Otherwise you
  are comparing two judges, not two applications.

The 26 test cases were drafted with AI assistance and then checked by hand,
line by line, against the source articles. No AI tool is part of the running
system except the two API models above.

## 3. Setup

You need Python 3.12 and [uv](https://docs.astral.sh/uv/) (pip works too).

```powershell
git clone https://github.com/ubaidur404786/deep_eval_pipeline.git
cd deep_eval_pipeline

uv venv --python 3.12 .venv
.venv\Scripts\activate
uv pip install -r requirements.txt
```

Then create your `.env`:

```powershell
copy .env.example .env
```

Open `.env` and paste a **Google AI Studio** key (free, no card, from
https://aistudio.google.com/apikey). That is enough for everything below. A
Groq key (https://console.groq.com/keys) is only needed for the model comparison.

`.env` is git-ignored. Your keys never leave your machine.

> If your console prints `�` instead of curly quotes, run
> `$env:PYTHONIOENCODING="utf-8"` once. The data itself is fine.

## 3b. Watch the flow before running anything

If you learn best by watching data move, there is a walkthrough script that
runs the real functions in the real order for one test case and prints what
goes in and out at every step. By default it makes **no API calls** and
**writes nothing** — the model is replaced by a stand-in — so it is safe to run
as often as you like:

```powershell
python debug_walkthrough.py                        # offline, case_12
python debug_walkthrough.py --case case_16 --pause # a different case, Enter between steps
python debug_walkthrough.py --case case_16 --live-app   # one real application call
```

Every line marked `# <-- breakpoint here` is where a new part of the system is
entered. Set breakpoints there in VS Code, press F5, and use *Step Into* to walk
into `loader.py`, `retriever.py`, `generator.py`, and the metric files.

## 4. Step 1 — Freeze the news corpus

**File:** `app/ingest.py`
**What it does:** downloads three public RSS feeds (arXiv cs.AI, TechCrunch AI,
The Verge AI), saves the raw entries to `data/raw/snapshot.json`, cleans them,
gives each article a stable id, and writes `data/processed/articles.json`.

The snapshot is **frozen**. The script refuses to re-download unless you pass
`--refresh`. This matters more than it looks: every test case is written against
this exact file. If the news changed underneath, the "expected answers" would
silently become wrong.

The frozen snapshot is committed, so you do not even need internet for this step:

```powershell
python -m app.ingest
```

Output:

```text
using frozen snapshot from 2026-09-11T23:21:30+00:00 (pass --refresh to re-fetch)

============================================================
snapshot fetched_at : 2026-09-11T23:21:30+00:00
raw entries         : 80
clean articles      : 80
  arxiv_ai          51
  techcrunch_ai     19
  verge_ai          10
written             : data\processed\articles.json
============================================================
```

The corpus is deliberately uneven: arXiv gives full abstracts (~200 words),
TechCrunch gives one-line teasers, and one Verge item is a 9,000-word podcast
transcript. Real data is like that, and the unevenness creates realistic test cases.

## 5. Step 2 — Build the vector store

**File:** `app/retriever.py`
**What it does:** splits each article into chunks of 200 words (with 40 words
of overlap so a sentence cut at the boundary survives), prefixes each chunk with
the article title, turns every chunk into a 384-number vector with the embedding
model, and stores them in Chroma on disk.

**Search by meaning.** When you ask a question, the same model turns your
question into a vector, and Chroma returns the four chunks whose vectors are
closest. "Anthropic report on CAPTCHAs" and "bots pretending to be human" end up
close together even though they share no words.

```powershell
python -m app.retriever
```

First run downloads the embedding model and builds the store (about a minute).
Then it runs four demo searches:

```text
80 articles -> 165 chunks. Embedding with all-MiniLM-L6-v2 ...
stored 165 chunks in chroma_store/

======================================================================
QUERY: What did Anthropic reveal about AI agents and CAPTCHAs?
======================================================================
[1] score=0.808  techcrunch_ai_4a1ed417
    Anthropic reveals rogue AI agents hate CAPTCHAs, just like you
    Come inside the mind of a bot trying to convince the internet it's human....
[2] score=0.606  verge_ai_c1823427
    Anthropic spent this week in hot water over cybersecurity
    ...
======================================================================
QUERY: What is the best pizza recipe?
======================================================================
[1] score=0.151  arxiv_ai_f6f76050
    Which Tokens Should SFT Actually Learn? ...
```

Two things to notice. The right article is #1 with a big margin for real
questions. And the pizza question still gets four results — the retriever has no
idea of "nothing relevant"; it just returns the least-bad matches with a low
score (0.15 vs 0.8). So the *model* has to decline the pizza question. That is
what one of the test cases checks.

Try your own:

```powershell
python -m app.retriever --query "What is Meta's Muse app?" -k 3
```

## 6. Step 3 — Ask the assistant

**Files:** `app/generator.py`, `app/llm_client.py`, `app/rag_pipeline.py`, `config/system_prompt.txt`

This is the whole chatbot:

```text
question → retrieve 4 chunks → fill the prompt template → call the LLM → answer
```

Each file knows one thing:

| File | Knows about |
|---|---|
| `retriever.py` | where text comes from (Chroma) |
| `generator.py` | what to say to the model (the prompt) |
| `llm_client.py` | who to send it to (Gemini / Groq / Ollama) — the **only** file with a provider name |
| `rag_pipeline.py` | the order: retrieve, then generate |

The prompt lives in `config/system_prompt.txt`, not in Python, so you can edit
the rules without touching code. It tells the model to answer only from the
sources, to reply with one exact sentence when the sources do not cover the
question, to report both sides if sources disagree, to ask if the question is
ambiguous, to decline off-topic requests, and to end every answer with a
`Sources:` line. Each rule becomes a category of test case later.

```powershell
python -m app.rag_pipeline
```

Output (the four demo questions are chosen to hit different rules):

```text
======================================================================
QUESTION: What did Anthropic reveal about AI agents and CAPTCHAs?
----------------------------------------------------------------------
Anthropic revealed that rogue AI agents hate CAPTCHAs, just like humans do.

Sources: Anthropic reveals rogue AI agents hate CAPTCHAs, just like you
----------------------------------------------------------------------
retrieved:
  0.808  techcrunch_ai_4a1ed417  Anthropic reveals rogue AI agents hate CAPTCHAs...

======================================================================
QUESTION: How much did OpenAI pay the mathematicians?
----------------------------------------------------------------------
I don't have enough information in the provided sources to answer that.
----------------------------------------------------------------------
retrieved:
  0.637  verge_ai_6be94843  Mathematicians want proof OpenAI didn't use their work
  ...

======================================================================
QUESTION: What is the best pizza recipe?
----------------------------------------------------------------------
I only answer questions about the AI and technology news provided in the sources.
```

The second question is a trap — the corpus has an OpenAI/mathematicians dispute
but no payment. The assistant abstained with the exact sentence from the prompt.
The third was declined with a *different* sentence. That difference is on
purpose: later, the evaluator can tell "sources did not cover it" apart from
"not my job".

To ask something else:

```powershell
python -m app.rag_pipeline --question "Which companies did Anthropic accuse of distillation campaigns?"
```

To use a different model, change two lines in `.env` (for example
`APP_PROVIDER=groq` and `APP_MODEL=openai/gpt-oss-120b`). No code changes.

## 7. Step 4 — The golden test cases

**Files:** `golden_data/test_cases.json`, `golden_data/loader.py`

A *golden dataset* is the answer key. Each of the 26 cases looks like this:

```json
{
  "id": "case_16",
  "category": "hallucination_trap",
  "question": "Why did Anthropic's alignment lead resign this week?",
  "expected_answer": "The alignment lead did not resign. An Anthropic researcher resigned ...",
  "expected_behavior": "correct_assumption",
  "metrics": ["instruction_following", "faithfulness", "correctness"],
  "source_ids": ["techcrunch_ai_0c8057bd"],
  "notes": "Role confusion trap: researcher resigned, alignment lead co-signed."
}
```

The two fields that do the real work:

- **`expected_behavior`** — what the assistant should *do*: `answer`, `abstain`,
  `decline`, `correct_assumption`, `report_conflict`, or `clarify`. Each maps to
  one rule in the system prompt.
- **`metrics`** — which metrics make sense for this case. A question with no
  answer in the corpus has no "correct answer" to compare against, so
  correctness is simply not run on it. No fake zeros.

The 26 cases cover nine categories:

```text
factual  8   multi_hop  3   missing_information  3   hallucination_trap  4
contradiction  1   ambiguous  2   out_of_scope  2   instruction_following  2   relevance  1
```

Some are *designed to fail* with the current retriever. A dataset where
everything passes teaches nothing.

The loader validates the file and, without calling any LLM, checks whether the
retriever can even find the articles each case points at:

```powershell
python -m golden_data.loader
```

```text
loaded 26 cases - schema OK
  ambiguous                2
  contradiction            1
  factual                  8
  ...

retrieval check (expected source_ids found in top-k?)
  OK  case_01  1/1 sources retrieved
  OK  case_02  1/1 sources retrieved
  ...
  MISS case_09  1/2 sources retrieved
  ...
  MISS case_20  1/2 sources retrieved

2 case(s) where retrieval misses at least one expected source
```

Those two misses are kept on purpose. They will show the evaluator catching a
*retrieval* problem through an *answer-quality* metric.

## 8. Step 5 — The four metrics

**Folder:** `eval_methods/` — one file per metric, each with one function:

```python
evaluate(question, answer, context, expected_answer=None, expected_behavior=None)
    -> {"metric": "faithfulness", "score": 0.75, "passed": True, "reason": "..."}
```

| File | Asks | Needs | How it works |
|---|---|---|---|
| `correctness.py` | Are the stated facts true, compared to the golden answer? | `expected_answer` | A rubric I wrote (GEval). The judge follows my steps and returns 0–10 with a reason. |
| `faithfulness.py` | Is every claim supported by the retrieved chunks? | `context` | DeepEval built-in: the judge splits the answer into claims and checks each one against the chunks. |
| `relevance.py` | Does it answer *this* question, without padding? | nothing extra | DeepEval built-in: splits into statements, asks "is this relevant?" for each. |
| `instruction_following.py` | Did it abstain / decline / cite / clarify as required? | `expected_behavior` | Exact string checks first (abstention sentence, `Sources:` line); a rubric for the rest. |

**Why four separate metrics?** Because the same answer can be:

- true but unfaithful — right from memory, not from the sources (will break tomorrow)
- faithful but wrong — loyally repeated a mistake in the source
- true, faithful, and irrelevant — accurate facts that do not answer the question
- true, faithful, relevant, and disobedient — dropped the Sources line because the user asked

I tested this with hand-written answers before plugging in the real assistant:

```text
                          correctness  faithfulness  relevance  instr_following
GOOD answer                  1.00         1.00         1.00         1.00
HALLUCINATED ($50K, Texas)   0.00         0.00         0.33         1.00   ← behaved right, was wrong
IRRELEVANT (generic)          —           1.00         0.67         0.30   ← nothing false, nothing useful
Sources line dropped         1.00         1.00         1.00         0.70   ← right facts, broke a rule
```

Every row fails differently. One number could never show that.

Each rubric also says what the metric must **not** grade — correctness ignores
brevity, relevance ignores truth, instruction-following ignores facts. Without
those lines the four metrics slowly turn into one.

**The judge** is built in one place, `eval_methods/judge.py`, from two lines in
`.env`. No metric file names a provider.

## 9. Step 6 — Run the evaluation

**Files:** `pipeline/run_evaluation.py`, `pipeline/report.py`

This is the file to read if you read only one. Its imports are the whole
architecture:

```python
from app.rag_pipeline import answer_question     # the thing being evaluated
from eval_methods import METRICS                 # the framework
from golden_data.loader import load_golden_data  # the answer key
```

For each case: run the assistant, check whether retrieval found the expected
articles, run the case's own metrics, **save after every case** (so a quota stop
loses nothing), then print a summary.

Start small — five cases takes about three minutes:

```powershell
python -m pipeline.run_evaluation --ids case_01,case_12,case_16,case_22,case_25
```

```text
evaluating 5 case(s)  app=gemini/gemini-3.1-flash-lite  judge=gemini/gemini-3.5-flash-lite

[1/5] case_01  (factual)
      correctness            1.00  PASS
      faithfulness           1.00  PASS
      relevance              1.00  PASS
      instruction_following  1.00  PASS
[2/5] case_12  (missing_information)
      instruction_following  1.00  PASS
      faithfulness           1.00  PASS
[3/5] case_16  (hallucination_trap)
      instruction_following  1.00  PASS
      faithfulness           1.00  PASS
      correctness            1.00  PASS
[4/5] case_22  (out_of_scope)
      instruction_following  1.00  PASS
[5/5] case_25  (instruction_following)
      correctness            1.00  PASS
      instruction_following  0.70  PASS
========================================================================
EVALUATION SUMMARY
========================================================================
app model    : gemini/gemini-3.1-flash-lite
judge model  : gemini/gemini-3.5-flash-lite
prompt hash  : 8caaebbe2646    cases: 5    time: 158s
...
saved -> results/latest_results.json
```

Other ways to select cases:

```powershell
python -m pipeline.run_evaluation                          # all 26 (~15-20 min)
python -m pipeline.run_evaluation --category hallucination_trap
python -m pipeline.run_evaluation --limit 5
python -m pipeline.run_evaluation --resume                 # continue after a quota stop
```

Results go to `results/latest_results.json` plus a timestamped copy. The file
records the app model, the judge model, a hash of the prompt file, and the
corpus timestamp — so you always know exactly what produced the numbers.

## 10. Step 7 — Read the results

My first clean full run, 26 cases, 17 minutes, zero errors:

```text
per metric
  metric                   n   pass    avg    min
  correctness             18   100%   0.99   0.90
  faithfulness            21    90%   0.93   0.00
  relevance               13    85%   0.94   0.50
  instruction_following   26    85%   0.83   0.00

per category (case passes only if ALL its metrics pass)
  factual                6/8       hallucination_trap     4/4
  multi_hop              2/3       out_of_scope           2/2
  missing_information    3/3       instruction_following  2/2
  contradiction          0/1       relevance              1/1
  ambiguous              0/2

failed cases (6)
  case_01  [factual]        faithfulness=0.0
  case_04  [factual]        faithfulness=0.5, relevance=0.5
  case_09  [multi_hop]      instruction_following=0.0
  case_19  [contradiction]  relevance=0.667, instruction_following=0.0
  case_20  [ambiguous]      instruction_following=0.0
  case_21  [ambiguous]      instruction_following=0.0
```

Numbers alone would tell you "six failures". The reasons tell you what *kind*:

**A real mistake by the assistant** — `case_19`. The article says "93% of audio
content" in the headline and "99% of new content" in the body. The assistant
reported 99% only. The judge wrote:

> "The actual output only reports a single figure (99%) and fails to mention the
> conflict or present the second version from the sources."

**A retrieval miss** — `case_09` and `case_20`. Both need two articles; the
retriever found one (`expected_sources_found: 1/2` in the results file, no LLM
involved). The assistant could not have answered well. That is an application
problem, not a model problem, and the fix is in the retriever.

**A judge mistake** — `case_01`. The assistant said the lawyer was fined "for
including AI-fabricated witnesses in a legal brief". The source says "in an
appeal of his murder conviction". The judge called that a contradiction and
gave 0.0. It is not one. The score is a claim; the reason is the evidence.
When they disagree, fix the rubric or try a stronger judge — do not trust the number.

One more thing I saw across two runs: the same failure (assistant dropped the
`Sources:` line) got 0.5 the first time and 0.7 the second. Judges are not
perfectly consistent even at temperature 0. So I moved that particular check
out of the judge into a plain string test. Cheap and exact beats expensive and
wobbly whenever exact is possible.

### A second run, a second model

Same 26 cases, application model `openai/gpt-oss-120b` (served free by Groq),
judge `gemini-3.1-flash-lite`, 15 minutes:

```text
per metric
  metric                   n   pass    avg
  correctness             18    94%   0.96
  faithfulness            21    95%   0.98
  relevance               13    92%   0.97
  instruction_following   26    77%   0.80

failed cases (7)
  case_02  [factual]            relevance=0.667              judge error ("Thursday is irrelevant" — the question asked when)
  case_09  [multi_hop]          instruction_following=0.0    retrieval miss (1/2) → model abstained instead of answering with what it had
  case_16  [hallucination_trap] instruction_following=0.0, correctness=0.3   REAL: went along with the false premise
  case_19  [contradiction]      faithfulness=0.5, instruction_following=0.0  gave both figures but did not flag the conflict
  case_20  [ambiguous]          instruction_following=0.3    retrieval miss (1/2)
  case_21  [ambiguous]          instruction_following=0.3    debatable golden case
  case_25  [instruction_following] instruction_following=0.5  REAL: obeyed "answer without listing sources"
```

The headline: the **120B model hallucinated where the small one did not**.
Asked "Why did Anthropic's alignment lead resign this week?", it answered
*"Anthropic's alignment lead resigned in protest, co-signing a public
warning…"* — the source says a researcher resigned and the alignment lead only
co-signed. `gemini-3.1-flash-lite` corrected the premise. Bigger is not
automatically safer, and this is the kind of thing a fixed test set catches
that casual testing never would.

**Important:** this run and the first one used different judges (the first
judge's daily cap was exhausted). So the two tables are *not* a head-to-head
comparison, and `compare.py` refuses to put them side by side. That refusal is
the framework doing its job.

This run also started out with **15** failures, not 7. Reading the reasons
showed that eight of them were caused by the framework, not the model — see
section 14. After fixing the framework, only the affected rows were re-scored
with `pipeline/rescore.py` (the answers were unchanged; only the scoring rules
were). Every re-scored row is marked `"rescored": true` in the results file.

To dig into any case:

```powershell
python -c "import json; r=json.load(open('results/latest_results.json',encoding='utf-8')); c=[c for c in r['cases'] if c['id']=='case_19'][0]; print(c['answer']); [print(m['metric'], m['score'], m['reason']) for m in c['metrics']]"
```

## 11. Step 8 — The Streamlit demo

**File:** `app/streamlit_app.py`

```powershell
streamlit run app/streamlit_app.py
```

Opens http://localhost:8501 with three tabs:

- **Golden case** — pick one of the 26 cases from a dropdown. You see the
  question, the expected behaviour, the metrics that will run, and the golden
  answer. Click **Run application** to get the live answer plus the four
  retrieved chunks (the expected articles are marked ✅, and a banner says
  "Expected source articles retrieved: 1/2" if retrieval missed). Click
  **Run evaluation** to get a score tile per metric and a "why" expander with
  the judge's reason.
- **Custom question** — type anything, choose the expected behaviour and which
  metrics to run, optionally paste an expected answer to enable correctness.
- **Last pipeline run** — the summary tables from `results/latest_results.json`.

The two buttons are separate on purpose: running the application costs one API
call, running the evaluation costs about ten. You can look at an answer and its
retrieved chunks before spending judge quota.

The UI imports the same functions the pipeline uses — same loader, same
`answer_question`, same metrics. It cannot disagree with the pipeline.

## 12. Step 9 — Compare two models

**File:** `pipeline/compare.py`

The whole comparison is a `.env` edit. Everything else — cases, metrics, judge,
threshold — stays the same.

```text
# run A
APP_PROVIDER=gemini   APP_MODEL=gemini-3.1-flash-lite
# run B  (OpenAI's open-weight model, served free by Groq)
APP_PROVIDER=groq     APP_MODEL=openai/gpt-oss-120b
```

Run the pipeline once with each, then:

```powershell
python -m pipeline.compare results/run_gemini_gemini-3.1-flash-lite_<ts>.json results/run_groq_openai_gpt-oss-120b_<ts>.json
```

It prints per-metric and per-case tables side by side and marks where one model
did better. It **refuses** to compare two runs whose judge, prompt or corpus
differ — those numbers are not comparable, and pretending they are is the
easiest way to fool yourself.

Example of the format (comparing two runs of the same model — the second one is
the clean one; the first had rate-limit errors):

```text
metric                    gemini-3.1-flash-lite (A)   gemini-3.1-flash-lite (B)
                                  pass    avg    min         pass    avg    min
--------------------------------------------------------------------------------
correctness                 94%   0.96   0.60   err=2   100%   0.99   0.90   err=0
faithfulness                86%   0.89   0.00   err=7    90%   0.93   0.00   err=0
...
case      category                       A                        B   note
case_02   factual                relev=None                       ok   <- B better
```

**Status.** Two complete runs exist (`gemini-3.1-flash-lite` and
`gpt-oss-120b`) but with different judges, so they cannot be compared directly.
The planned same-judge comparison is `gpt-oss-120b` vs `gpt-oss-20b`, both
judged by `gemini-3.1-flash-lite` — two runs of ~200 judge calls each, which
fit inside one day's free quota. Results will be added here when they are in.

What I expect (written down *before* seeing the numbers): the 20B model should
fail more instruction-following cases than the 120B, while `case_09` and
`case_20` should fail for **both**, because those are retrieval failures and no
application model can fix them. Interesting to watch: whether the 20B also
falls for the `case_16` false premise.

Before trusting any difference, run the *same* configuration twice and look at
how much the numbers move on their own. That is your noise floor. Only
differences bigger than that mean anything.

## 13. Free-tier limits I measured

Free tiers change; these are what my keys reported on 2026-09-12. Check your own
console.

| Provider / model | Per minute | Per day | Notes |
|---|---|---|---|
| Gemini `gemini-3.1-flash-lite` | 15 requests | 500 | app model in run 1; judge for the Groq runs |
| Gemini `gemini-3.5-flash-lite` | 15 requests | **500** (measured) | judge in run 1 ≈ two full runs/day |
| Gemini `gemini-3.6-flash` | 5 requests | **20** | subsets only |
| Gemini `gemini-2.5-*` | — | — | closed to new keys |
| Groq `openai/gpt-oss-120b` | 8,000 **tokens** | 1,000 requests | good app model; too slow as judge |
| Groq `llama-*` | — | — | no longer served on free accounts |

Quotas are per model, so the app and the judge on different models get
separate buckets. A 26-case run makes about 250 judge calls.

## 14. Things that went wrong, and what I learned

- **The model I designed around disappeared.** `gemini-2.5-flash` was listed,
  then rejected as "no longer available to new users" on the first live call.
  Fix: one line in `.env`. That is what keeping provider names out of the code
  buys you.
- **Daily caps, not per-minute caps, are what stop you.** `gemini-3.6-flash`
  died at case 12 of a 26-case run (20 requests/day). So the pipeline now saves
  after every case, stops cleanly, and resumes.
- **The library's retry did not work for Gemini.** DeepEval 4.2.2's Gemini
  retry policy fails to load, so a 429 surfaced immediately as a metric error.
  The judge now retries itself, sleeping for exactly the delay Google specifies.
- **The first search on a cold index returned 3 of 4 chunks.** Chroma's HNSW
  quirk. A warm-up query on open fixed it. A retriever that returns a different
  number of chunks on different runs would make results incomparable.
- **The judge was grading the citation line.** In the second run, relevance
  called the required `Sources:` line "irrelevant meta-information", and
  faithfulness read a cited article *title* as a factual claim. Fix:
  faithfulness and relevance now see only the answer body; the citation line
  belongs to instruction-following. One concern per metric, enforced in code.
- **Abstentions were being sent to the faithfulness judge**, which scored them
  0.0 for "not using the context". An abstention makes no claims, so it now
  scores 1.0 deterministically, with no judge call.
- **My own `Sources:` check was too strict.** `gpt-oss` writes `Sources: [1]`
  at the end of the paragraph, not on its own line. The check now accepts a
  citation anywhere and skips abstentions.
- **Two timeouts were fighting.** My rate-limit wait sleeps *inside* the judge
  call; DeepEval's own 90-second per-attempt timeout fired in the middle of it.
  Disabled the inner one.
- **Re-score, don't re-run.** When a scoring rule changes, the saved answers
  are still valid. `pipeline/rescore.py` re-runs only the metrics, for ~15
  judge calls instead of 250.
- **The judge is wrong sometimes.** Two of six failures in the first run, and
  two of seven in the second, were the judge's. Always read the reason.
- **Most failures were retrieval failures.** A free string comparison
  (`expected_sources_found`) explained more than any judge call. Evaluate the
  whole application, not just the model.

## 15. Reusing the framework on your own app

1. Write a function that returns `{"query": ..., "context": [chunks], "answer": ...}`.
2. In `pipeline/run_evaluation.py`, change one import to point at it.
3. Write golden cases for your domain in the same JSON schema.
4. Run.

Nothing in `eval_methods/` changes. It never imports from `app/` — that single
rule is what makes it a framework instead of a script.
