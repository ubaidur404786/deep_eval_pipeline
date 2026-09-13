"""
app/streamlit_app.py -- interactive demo of the evaluation framework.

Four tabs:

    1. Golden case      pick one of the 26 test cases, run the app, run the
                        evaluation, compare with the expected behaviour
    2. Custom question  ask anything, choose the metrics, evaluate
    3. Last run         summary tables from results/latest_results.json
    4. Compare runs     any two saved runs side by side (same judge/prompt/corpus)

The sidebar lets a viewer switch the APPLICATION model live. The JUDGE stays
fixed from .env on purpose: comparisons are only meaningful with one judge.

Run from the PROJECT ROOT:
    streamlit run app/streamlit_app.py
"""

import json
import os
import sys
from pathlib import Path

# Streamlit puts app/ on sys.path, not the project root. Add the root so that
# `from config.settings import ...` resolves exactly as it does everywhere else.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

# On Streamlit Community Cloud there is no .env; keys live in st.secrets.
# Copy top-level secrets into the environment so the rest of the code (which
# uses os.getenv) works unchanged. Locally, .env is loaded by config.settings.
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, str) and _k not in os.environ:
            os.environ[_k] = _v
except Exception:  # noqa: BLE001 -- no secrets file locally; that is fine
    pass

from app.rag_pipeline import answer_question
from app.retriever import build_store
from app.ui_theme import failure_map_html, hero, inject_css
from config.settings import ABSTENTION_MESSAGE, APP_MODEL, APP_PROVIDER, RESULTS_DIR
from eval_methods import METRICS
from eval_methods.instruction_following import BEHAVIOR_DESCRIPTIONS
from eval_methods.judge import judge_name
from golden_data.loader import load_golden_data
from pipeline.compare import check_comparable

st.set_page_config(page_title="RAG Evaluation Framework", page_icon="🧪", layout="wide")

# Models a viewer can pick for the APPLICATION. Only providers with a key in
# .env are offered. Limits measured 2026-09-12 on free tiers.
APP_MODEL_OPTIONS = {
    "gemini/gemini-3.1-flash-lite": ("gemini", "gemini-3.1-flash-lite", "15 requests a minute, about 500 a day"),
    "gemini/gemini-3.5-flash-lite": ("gemini", "gemini-3.5-flash-lite", "15 requests a minute, about 500 a day"),
    "gemini/gemini-3.6-flash":      ("gemini", "gemini-3.6-flash",      "5 requests a minute and only 20 a day"),
    "groq/openai/gpt-oss-120b":     ("groq",   "openai/gpt-oss-120b",   "1,000 requests a day, 8K tokens a minute"),
    "groq/openai/gpt-oss-20b":      ("groq",   "openai/gpt-oss-20b",    "1,000 requests a day, 8K tokens a minute"),
    "groq/qwen/qwen3.8-27b":        ("groq",   "qwen/qwen3.8-27b",      "1,000 requests a day, 8K tokens a minute"),
}
KEY_FOR_PROVIDER = {"gemini": "GOOGLE_API_KEY", "groq": "GROQ_API_KEY"}


# ---------------------------------------------------------------------------
# cached loaders (run once per server process, not on every click)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading vector store…")
def ready_store() -> bool:
    build_store()
    return True


@st.cache_data
def golden_cases() -> list[dict]:
    return load_golden_data()


def load_results(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def result_files() -> list[tuple[str, Path]]:
    """Complete runs only (all cases scored), newest first, with a readable label."""
    out = []
    for path in sorted(RESULTS_DIR.glob("run_*.json"), reverse=True):
        r = load_results(path)
        if len(r["cases"]) < r["run"]["n_cases"]:
            continue  # partial run -- not comparable
        stamp = r["run"]["timestamp"][:16].replace("T", " ")
        out.append((f"{r['run']['app_model']}  ({stamp}, judge {r['run']['judge_model'].split('/')[-1]})", path))
    return out


# ---------------------------------------------------------------------------
# rendering helpers
# ---------------------------------------------------------------------------
def render_answer(result: dict, expected_ids: list[str]) -> None:
    st.markdown("#### Answer")
    # a statement, not a bare expression: Streamlit "magic" would otherwise
    # render the returned element object as a help table
    if result["answer"].strip() == ABSTENTION_MESSAGE:
        st.info(result["answer"])
    else:
        st.success(result["answer"])

    retrieved = {s["source_id"] for s in result["sources"]}
    if expected_ids:
        found = sum(1 for s in expected_ids if s in retrieved)
        msg = f"Expected source articles retrieved: **{found}/{len(expected_ids)}**"
        if found == len(expected_ids):
            st.markdown(f"✅ {msg}")
        else:
            st.markdown(f"⚠️ {msg} — whatever the judge says next is partly a *retrieval* problem")

    with st.expander(f"Retrieved context — {len(result['sources'])} chunks the model actually saw"):
        for i, (src, chunk) in enumerate(zip(result["sources"], result["context"]), 1):
            mark = " ✅ expected" if src["source_id"] in expected_ids else ""
            st.markdown(f"**[{i}] {src['title']}**  ·  similarity {src['score']:.3f}  ·  `{src['source_id']}`{mark}")
            st.text(chunk.split("\n\n", 1)[-1][:700])
            if i < len(result["sources"]):
                st.divider()


def render_scores(rows: list[dict]) -> None:
    st.markdown("#### Evaluation")
    cols = st.columns(max(len(rows), 1))
    for col, r in zip(cols, rows):
        score = "ERR" if r["score"] is None else f"{r['score']:.2f}"
        col.metric(label=r["metric"].replace("_", " "), value=score,
                   delta="PASS" if r["passed"] else "FAIL",
                   delta_color="normal" if r["passed"] else "inverse")
    for r in rows:
        icon = "✅" if r["passed"] else "❌"
        with st.expander(f"{icon} {r['metric']} — why the judge scored it this way"):
            st.write(r["reason"])


def run_metrics(names: list[str], question: str, result: dict,
                expected_answer: str | None, expected_behavior: str) -> list[dict]:
    rows = []
    progress = st.progress(0.0, text="Judging…")
    for i, name in enumerate(names, 1):
        progress.progress((i - 1) / len(names), text=f"Judging: {name}  ({i}/{len(names)})")
        try:
            rows.append(METRICS[name](
                question=question, answer=result["answer"], context=result["context"],
                expected_answer=expected_answer, expected_behavior=expected_behavior,
            ))
        except Exception as exc:  # noqa: BLE001
            rows.append({"metric": name, "score": None, "passed": False, "reason": f"ERROR: {exc}"})
    progress.empty()
    return rows


def summary_tables(res: dict) -> None:
    run = res["run"]
    st.markdown(
        f"**App:** `{run['app_model']}` · **Judge:** `{run['judge_model']}` · "
        f"**Prompt hash:** `{run['prompt_hash']}` · **Cases:** {len(res['cases'])}/{run['n_cases']} · "
        f"**Time:** {run['elapsed_s']:.0f}s"
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Per metric**")
        st.dataframe([{"metric": k, **v} for k, v in res["summary"]["per_metric"].items()],
                     hide_index=True, width="stretch")
    with c2:
        st.markdown("**Per category** (a case passes only if all its metrics pass)")
        st.dataframe([{"category": k, **v} for k, v in res["summary"]["per_category"].items()],
                     hide_index=True, width="stretch")
    st.markdown("**Per case**")
    rows = []
    for c in res["cases"]:
        row = {"id": c["id"], "category": c["category"], "sources": c["expected_sources_found"] or "-"}
        for m in c["metrics"]:
            row[m["metric"]] = "ERR" if m["score"] is None else m["score"]
        rows.append(row)
    st.dataframe(rows, hide_index=True, width="stretch")


# ---------------------------------------------------------------------------
# sidebar: application model (live), judge (fixed)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Models")
    available = {k: v for k, v in APP_MODEL_OPTIONS.items() if os.getenv(KEY_FOR_PROVIDER[v[0]])}
    default_key = f"{APP_PROVIDER}/{APP_MODEL}"
    keys = list(available)
    choice = st.selectbox(
        "Application model",
        keys,
        index=keys.index(default_key) if default_key in keys else 0,
    )
    app_provider, app_model, limits = available[choice]
    st.caption(f"Free tier: {limits}. Switch freely: same question, same judge, different model.")

    st.markdown(f"**Judge:** `{judge_name()}`")
    st.caption("Fixed from .env. Comparisons only mean something with one judge.")

    st.divider()
    st.caption(
        "*Run application* makes one API call. *Run evaluation* makes 5 to 12 judge calls; "
        "the free tier allows about 500 a day."
    )
    if st.button("Clear results", width="stretch"):
        for k in list(st.session_state):
            del st.session_state[k]
        st.rerun()


# ---------------------------------------------------------------------------
# page
# ---------------------------------------------------------------------------
inject_css()

hero(
    "Does the chatbot actually answer well?",
    "A reusable evaluation framework for RAG applications, shown on an AI & technology news "
    "assistant: 26 hand-written test cases, four metrics that each check one thing, one fixed "
    "judge model, three application models. Every score comes with the judge's written reason.",
    'Built on free APIs only. <a href="https://github.com/ubaidur404786/deep_eval_pipeline" target="_blank">Code and full write-up on GitHub</a>.',
)

# --- the failure map: one row per complete run, one cell per case ---
_files = result_files()
_runs = []
_judges = set()
for _label, _path in _files:
    _r = load_results(_path)
    _judges.add(_r["run"]["judge_model"])
    _runs.append((_r["run"]["app_model"].split("/", 1)[-1], _r))
_runs.sort(key=lambda t: t[0])  # a stable, readable order: gemini…, openai/gpt-oss-120b, openai/gpt-oss-20b
_case_ids = [c["id"] for c in golden_cases()]
if _runs:
    st.markdown("#### Three models, the same 26 questions, the same judge" if len(_runs) == 3
                else f"#### {len(_runs)} complete run{'s' if len(_runs) != 1 else ''}, the same 26 questions")
    st.markdown(failure_map_html(_runs, _case_ids), unsafe_allow_html=True)
    if len(_judges) > 1:
        st.caption("Rows were judged by different models and are not directly comparable.")
    st.markdown(
        "Every model fails the same three cases — two where the retriever found one of two needed "
        "articles, and one where the user said *don't cite sources* and every model obeyed. Those are "
        "problems in the application, not in the models; the framework separates the two."
        if len(_runs) == 3 else ""
    )

with st.expander("What each tab does"):
    st.markdown(
        """
- **Golden case** — pick one of 26 hand-written test cases. *Run application* asks the news assistant
  (one API call) and shows the answer plus the four text chunks it was given. *Run evaluation* asks a
  second model, the **judge**, to score the answer on the metrics that apply to that case.
- **Custom question** — the same, for any question you type.
- **Last pipeline run** — the summary of the most recent full 26-case run.
- **Compare runs** — two complete runs side by side. Only runs with the same judge, prompt and corpus
  can be compared; the tab refuses otherwise.

Every score comes with the judge's written reason. Read it — the judge is sometimes wrong, and the reason is how you find out.
        """
    )
ready_store()

tab_golden, tab_custom, tab_last, tab_compare = st.tabs(
    ["Golden case", "Custom question", "Last pipeline run", "Compare runs"]
)

# ---- TAB 1: golden case -----------------------------------------------------
with tab_golden:
    cases = golden_cases()
    labels = [f"{i}. {c['question'][:80]}  ({c['category'].replace('_', ' ')})" for i, c in enumerate(cases, 1)]
    _wanted = st.query_params.get("case")
    _default = next((i for i, c in enumerate(cases) if c["id"] == _wanted), 0)
    idx = st.selectbox("Select a test case", range(len(cases)), index=_default, format_func=lambda i: labels[i])
    case = cases[idx]

    left, right = st.columns([3, 2])
    with left:
        st.markdown(f"**Question**  \n{case['question']}")
        if case["expected_answer"]:
            with st.expander("Expected answer (golden)"):
                st.write(case["expected_answer"])
    with right:
        st.markdown(f"**Expected behaviour:** `{case['expected_behavior']}`")
        st.caption(BEHAVIOR_DESCRIPTIONS[case["expected_behavior"]])
        st.markdown(f"**Metrics for this case:** {', '.join(case['metrics'])}")
        if case["notes"]:
            st.caption(case["notes"])

    b1, b2 = st.columns([1, 1])
    run_key = (case["id"], choice)
    if b1.button("Run application", key="run_golden", width="stretch"):
        with st.spinner(f"Retrieving and generating with {choice}…"):
            st.session_state["golden_result"] = (run_key, answer_question(
                case["question"], provider=app_provider, model=app_model))
            st.session_state.pop("golden_scores", None)

    stored = st.session_state.get("golden_result")
    if stored and stored[0] == run_key:
        result = stored[1]
        render_answer(result, case["source_ids"])
        if b2.button("Run evaluation", key="eval_golden", width="stretch"):
            st.session_state["golden_scores"] = run_metrics(
                case["metrics"], case["question"], result,
                case["expected_answer"], case["expected_behavior"],
            )
        if "golden_scores" in st.session_state:
            render_scores(st.session_state["golden_scores"])
    elif stored:
        st.caption("Model or case changed — run the application again.")

# ---- TAB 2: custom question -------------------------------------------------
with tab_custom:
    question = st.text_input("Your question",
                             placeholder="What did Anthropic reveal about AI agents and CAPTCHAs?")
    c1, c2 = st.columns(2)
    with c1:
        behavior = st.selectbox("Expected behaviour", list(BEHAVIOR_DESCRIPTIONS), index=0)
        metric_names = st.multiselect("Metrics to run", list(METRICS),
                                      default=["faithfulness", "relevance", "instruction_following"])
    with c2:
        expected = st.text_area("Expected answer (optional — enables correctness)", height=120)
    if "correctness" in metric_names and not expected.strip():
        st.warning("correctness needs an expected answer — it will be skipped.")
        metric_names = [m for m in metric_names if m != "correctness"]

    b1, b2 = st.columns([1, 1])
    run_key = (question, choice)
    if b1.button("Run application", key="run_custom", width="stretch", disabled=not question.strip()):
        with st.spinner(f"Retrieving and generating with {choice}…"):
            st.session_state["custom_result"] = (run_key, answer_question(
                question, provider=app_provider, model=app_model))
            st.session_state.pop("custom_scores", None)

    stored = st.session_state.get("custom_result")
    if stored and stored[0] == run_key:
        result = stored[1]
        render_answer(result, [])
        if b2.button("Run evaluation", key="eval_custom", width="stretch", disabled=not metric_names):
            st.session_state["custom_scores"] = run_metrics(
                metric_names, question, result, expected.strip() or None, behavior,
            )
        if "custom_scores" in st.session_state:
            render_scores(st.session_state["custom_scores"])

# ---- TAB 3: last pipeline run ------------------------------------------------
with tab_last:
    latest = RESULTS_DIR / "latest_results.json"
    if not latest.exists():
        st.info("No results yet. Run `python -m pipeline.run_evaluation` first.")
    else:
        summary_tables(load_results(latest))

# ---- TAB 4: compare two runs ---------------------------------------------------
with tab_compare:
    files = result_files()
    if len(files) < 2:
        st.info("Need at least two **complete** runs in results/ to compare. "
                "Partial runs (stopped by a quota) are not listed.")
    else:
        labels = [f[0] for f in files]
        c1, c2 = st.columns(2)
        fa = c1.selectbox("Run A", labels, index=min(1, len(labels) - 1))
        fb = c2.selectbox("Run B", labels, index=0)
        a = load_results(dict(files)[fa])
        b = load_results(dict(files)[fb])
        try:
            check_comparable(a, b)
        except SystemExit as exc:
            st.error(str(exc))
        else:
            la, lb = a["run"]["app_model"], b["run"]["app_model"]
            st.markdown(f"### `{la}`  vs  `{lb}`")
            st.caption(f"Same judge (`{a['run']['judge_model']}`), same prompt (`{a['run']['prompt_hash']}`), "
                       f"same corpus, same 26 cases. Only the application model differs.")

            st.markdown("**Per metric** — pass rate and average score (0–1). Δ is B minus A.")
            rows = []
            for name in sorted(set(a["summary"]["per_metric"]) | set(b["summary"]["per_metric"])):
                sa = a["summary"]["per_metric"].get(name, {})
                sb = b["summary"]["per_metric"].get(name, {})
                da = sa.get("avg_score"); db = sb.get("avg_score")
                delta = round(db - da, 3) if da is not None and db is not None else None
                better = "" if delta is None or abs(delta) < 0.02 else ("B ▲" if delta > 0 else "A ▲")
                rows.append({"metric": name,
                             "A pass": sa.get("pass_rate"), "A avg": da,
                             "B pass": sb.get("pass_rate"), "B avg": db,
                             "Δ avg": delta, "better": better})
            st.dataframe(rows, hide_index=True, width="stretch")
            st.caption("Differences under ~0.02 are within judge noise (measured by running one config twice).")

            st.markdown("**Per case** — only the cases where the two runs disagree")
            ca = {c["id"]: c for c in a["cases"]}
            cb = {c["id"]: c for c in b["cases"]}
            diff = []
            for cid in sorted(set(ca) | set(cb)):
                fa_ = [f"{m['metric']}={m['score']}" for m in ca[cid]["metrics"] if not m["passed"]] if cid in ca else ["(not run)"]
                fb_ = [f"{m['metric']}={m['score']}" for m in cb[cid]["metrics"] if not m["passed"]] if cid in cb else ["(not run)"]
                if fa_ != fb_:
                    src = (ca.get(cid) or cb.get(cid)).get("expected_sources_found") or "-"
                    diff.append({"case": cid, "category": (ca.get(cid) or cb.get(cid))["category"],
                                 "retrieval": src,
                                 "A": "ok" if not fa_ else ", ".join(fa_),
                                 "B": "ok" if not fb_ else ", ".join(fb_)})
            if diff:
                st.dataframe(diff, hide_index=True, width="stretch")
                st.caption("`retrieval` = expected source articles found in the top-4 chunks. "
                           "A case that fails for BOTH models with retrieval < full is a retrieval problem, not a model problem.")
            else:
                st.success("The two runs agree on every case.")
