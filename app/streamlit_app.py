"""
app/streamlit_app.py -- interactive demo of the evaluation framework.

Two things a visitor can do:

    1. Pick a GOLDEN CASE from the dropdown (the same test_cases.json the
       pipeline uses), run the application, run the evaluation, and compare
       the live result with the expected behaviour.
    2. Type a CUSTOM QUESTION, choose which metrics to run, and evaluate it.

A third tab shows the summary of the last pipeline run (results/latest_results.json).

Run from the PROJECT ROOT:
    streamlit run app/streamlit_app.py
"""

import json
import sys
from pathlib import Path

# Streamlit puts app/ on sys.path, not the project root. Add the root so that
# `from config.settings import ...` resolves exactly as it does everywhere else.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from app.rag_pipeline import answer_question
from app.retriever import build_store
from config.settings import ABSTENTION_MESSAGE, APP_MODEL, APP_PROVIDER, RESULTS_DIR
from eval_methods import METRICS
from eval_methods.instruction_following import BEHAVIOR_DESCRIPTIONS
from eval_methods.judge import judge_name
from golden_data.loader import load_golden_data

st.set_page_config(page_title="RAG Eval Demo", page_icon="🧪", layout="wide")


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


def latest_results() -> dict | None:
    path = RESULTS_DIR / "latest_results.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# rendering helpers
# ---------------------------------------------------------------------------
def render_answer(result: dict, expected_ids: list[str]) -> None:
    st.subheader("Answer")
    st.markdown(result["answer"])
    if result["answer"].strip() == ABSTENTION_MESSAGE:
        st.info("The application abstained (exact abstention sentence).")

    retrieved = {s["source_id"] for s in result["sources"]}
    with st.expander(f"Retrieved context — {len(result['sources'])} chunks", expanded=False):
        for i, (src, chunk) in enumerate(zip(result["sources"], result["context"]), 1):
            mark = " ✅ expected" if src["source_id"] in expected_ids else ""
            st.markdown(f"**[{i}] {src['title']}**  ·  score {src['score']:.3f}  ·  `{src['source_id']}`{mark}")
            st.text(chunk.split("\n\n", 1)[-1][:700])
            st.divider()
    if expected_ids:
        found = sum(1 for s in expected_ids if s in retrieved)
        (st.success if found == len(expected_ids) else st.warning)(
            f"Expected source articles retrieved: {found}/{len(expected_ids)}"
        )


def render_scores(rows: list[dict]) -> None:
    st.subheader("Evaluation")
    cols = st.columns(len(rows)) if rows else []
    for col, r in zip(cols, rows):
        score = "ERR" if r["score"] is None else f"{r['score']:.2f}"
        col.metric(label=r["metric"], value=score, delta="PASS" if r["passed"] else "FAIL",
                   delta_color="normal" if r["passed"] else "inverse")
    for r in rows:
        with st.expander(f"{'✅' if r['passed'] else '❌'} {r['metric']} — why"):
            st.write(r["reason"])


def run_metrics(names: list[str], question: str, result: dict,
                expected_answer: str | None, expected_behavior: str) -> list[dict]:
    rows = []
    progress = st.progress(0.0, text="Judging…")
    for i, name in enumerate(names, 1):
        progress.progress(i / len(names), text=f"Judging: {name}")
        try:
            rows.append(METRICS[name](
                question=question, answer=result["answer"], context=result["context"],
                expected_answer=expected_answer, expected_behavior=expected_behavior,
            ))
        except Exception as exc:  # noqa: BLE001
            rows.append({"metric": name, "score": None, "passed": False, "reason": f"ERROR: {exc}"})
    progress.empty()
    return rows


# ---------------------------------------------------------------------------
# page
# ---------------------------------------------------------------------------
st.title("🧪 RAG Evaluation Framework — demo")
st.caption(
    "A reusable LLM/RAG evaluation framework, demonstrated on an AI & Technology News Assistant. "
    f"App model: `{APP_PROVIDER}/{APP_MODEL}` · Judge: `{judge_name()}`"
)
ready_store()

tab_golden, tab_custom, tab_summary = st.tabs(["Golden case", "Custom question", "Last pipeline run"])

# ---- TAB 1: golden case -----------------------------------------------------
with tab_golden:
    cases = golden_cases()
    labels = [f"{c['id']} · {c['category']} · {c['question'][:70]}" for c in cases]
    choice = st.selectbox("Select a test case", range(len(cases)), format_func=lambda i: labels[i])
    case = cases[choice]

    left, right = st.columns([3, 2])
    with left:
        st.markdown(f"**Question**  \n{case['question']}")
    with right:
        st.markdown(f"**Expected behaviour:** `{case['expected_behavior']}`")
        st.caption(BEHAVIOR_DESCRIPTIONS[case["expected_behavior"]])
        st.markdown(f"**Metrics:** {', '.join(case['metrics'])}")
        if case["notes"]:
            st.caption(f"Notes: {case['notes']}")
    if case["expected_answer"]:
        with st.expander("Expected answer (golden)"):
            st.write(case["expected_answer"])

    if st.button("▶ Run application", key="run_golden"):
        with st.spinner("Retrieving and generating…"):
            st.session_state["golden_result"] = (case["id"], answer_question(case["question"]))
            st.session_state.pop("golden_scores", None)

    stored = st.session_state.get("golden_result")
    if stored and stored[0] == case["id"]:
        result = stored[1]
        render_answer(result, case["source_ids"])

        if st.button("⚖ Run evaluation", key="eval_golden"):
            st.session_state["golden_scores"] = run_metrics(
                case["metrics"], case["question"], result,
                case["expected_answer"], case["expected_behavior"],
            )
        if "golden_scores" in st.session_state:
            render_scores(st.session_state["golden_scores"])

# ---- TAB 2: custom question -------------------------------------------------
with tab_custom:
    question = st.text_input("Your question", placeholder="What did Anthropic reveal about AI agents and CAPTCHAs?")
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

    if st.button("▶ Run application", key="run_custom", disabled=not question.strip()):
        with st.spinner("Retrieving and generating…"):
            st.session_state["custom_result"] = (question, answer_question(question))
            st.session_state.pop("custom_scores", None)

    stored = st.session_state.get("custom_result")
    if stored and stored[0] == question:
        result = stored[1]
        render_answer(result, [])
        if st.button("⚖ Run evaluation", key="eval_custom", disabled=not metric_names):
            st.session_state["custom_scores"] = run_metrics(
                metric_names, question, result, expected.strip() or None, behavior,
            )
        if "custom_scores" in st.session_state:
            render_scores(st.session_state["custom_scores"])

# ---- TAB 3: last pipeline run ------------------------------------------------
with tab_summary:
    res = latest_results()
    if not res:
        st.info("No results yet. Run `python -m pipeline.run_evaluation` first.")
    else:
        run = res["run"]
        st.markdown(
            f"**App:** `{run['app_model']}` · **Judge:** `{run['judge_model']}` · "
            f"**Prompt hash:** `{run['prompt_hash']}` · **Cases:** {len(res['cases'])}/{run['n_cases']} · "
            f"**Time:** {run['elapsed_s']:.0f}s"
        )
        st.markdown("**Per metric**")
        st.dataframe(
            [{"metric": k, **v} for k, v in res["summary"]["per_metric"].items()],
            hide_index=True, width="stretch",
        )
        st.markdown("**Per category** (a case passes only if all its metrics pass)")
        st.dataframe(
            [{"category": k, **v} for k, v in res["summary"]["per_category"].items()],
            hide_index=True, width="stretch",
        )
        st.markdown("**Per case**")
        rows = []
        for c in res["cases"]:
            row = {"id": c["id"], "category": c["category"], "sources": c["expected_sources_found"] or "-"}
            for m in c["metrics"]:
                row[m["metric"]] = "ERR" if m["score"] is None else m["score"]
            rows.append(row)
        st.dataframe(rows, hide_index=True, width="stretch")
