"""
pipeline/run_evaluation.py -- THE evaluation flow, top to bottom.

This is the only file that knows about BOTH halves of the project:

    app/            (the thing being evaluated)   -> answer_question()
    eval_methods/   (the reusable framework)      -> METRICS[name]()

To evaluate a different application, change the one import below and make
sure its function returns {"query", "context", "answer"}. Nothing in
eval_methods/ changes.

Flow:
    load settings + system prompt (hash it for provenance)
      -> load golden cases
      -> for each case:
           run the application            -> answer + context
           run the case's own metrics     -> score + reason each
           save after every case          (a quota stop loses nothing)
      -> aggregate, save, print

Run from the project root:
    python -m pipeline.run_evaluation                       # all cases
    python -m pipeline.run_evaluation --ids case_01,case_12
    python -m pipeline.run_evaluation --category hallucination_trap
    python -m pipeline.run_evaluation --limit 5
    python -m pipeline.run_evaluation --resume              # finish an interrupted run

Free tiers have DAILY caps. If the application call fails after its retries,
the run stops cleanly with everything so far saved; --resume picks up where it
left off -- but only if app model, judge model, prompt and corpus all match,
so two different configurations can never be mixed into one results file.
"""

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone

# --- the application under test (swap this import to evaluate something else)
from app.rag_pipeline import answer_question
from app.retriever import build_store

# --- the framework
from eval_methods import METRICS
from eval_methods.judge import judge_name

# --- shared config + data
from config.settings import APP_MODEL, APP_PROVIDER, RAW_SNAPSHOT_FILE, SYSTEM_PROMPT_FILE
from golden_data.loader import load_golden_data
from pipeline.report import LATEST_FILE, print_summary, save_results, summarize


class ApplicationError(RuntimeError):
    """The application itself failed (not a metric) -- usually a daily quota."""


class JudgeQuotaError(RuntimeError):
    """The judge hit a DAILY cap. No metric can be scored until it resets."""


# ---------------------------------------------------------------------------
# 1. PROVENANCE  -- record exactly what produced these numbers
# ---------------------------------------------------------------------------
def prompt_hash() -> str:
    """First 12 hex chars of sha256(system_prompt.txt). A silently edited
    prompt shows up as a different hash in the results file."""
    return hashlib.sha256(SYSTEM_PROMPT_FILE.read_bytes()).hexdigest()[:12]


def snapshot_stamp() -> str:
    with open(RAW_SNAPSHOT_FILE, encoding="utf-8") as f:
        return json.load(f)["fetched_at"]


# ---------------------------------------------------------------------------
# 2. ONE CASE  -- run the app, then every metric the case asks for
# ---------------------------------------------------------------------------
def evaluate_case(case: dict) -> dict:
    started = time.perf_counter()

    # 2a. the application answers
    try:
        result = answer_question(case["question"])
    except Exception as exc:  # noqa: BLE001
        raise ApplicationError(f"{type(exc).__name__}: {str(exc)[:300]}") from exc

    # 2b. did retrieval even find the articles the golden case points at?
    retrieved_ids = {s["source_id"] for s in result["sources"]}
    expected_ids = case["source_ids"]
    sources_found = sum(1 for s in expected_ids if s in retrieved_ids)

    # 2c. each metric the case lists -- and only those
    metric_rows = []
    for name in case["metrics"]:
        try:
            row = METRICS[name](
                question=case["question"],
                answer=result["answer"],
                context=result["context"],
                expected_answer=case["expected_answer"],
                expected_behavior=case["expected_behavior"],
            )
        except Exception as exc:  # noqa: BLE001 -- keep the run alive, record the failure
            if "PerDay" in str(exc):
                raise JudgeQuotaError(str(exc)[:200]) from exc
            row = {"metric": name, "score": None, "passed": False, "reason": f"ERROR: {exc}"}
        metric_rows.append(row)
        print(f"      {name:<22} {_fmt(row)}")

    return {
        "id": case["id"],
        "category": case["category"],
        "question": case["question"],
        "expected_behavior": case["expected_behavior"],
        "expected_answer": case["expected_answer"],
        "answer": result["answer"],
        "sources": result["sources"],
        "context": result["context"],
        "expected_sources_found": f"{sources_found}/{len(expected_ids)}" if expected_ids else None,
        "metrics": metric_rows,
        "elapsed_s": round(time.perf_counter() - started, 1),
    }


def _fmt(row: dict) -> str:
    if row["score"] is None:
        return "ERR   " + row["reason"][:60]
    return f"{row['score']:.2f}  {'PASS' if row['passed'] else 'FAIL'}"


# ---------------------------------------------------------------------------
# 3. THE RUN
# ---------------------------------------------------------------------------
def select_cases(cases: list[dict], ids: str | None, category: str | None, limit: int | None) -> list[dict]:
    if ids:
        wanted = set(ids.split(","))
        cases = [c for c in cases if c["id"] in wanted]
    if category:
        cases = [c for c in cases if c["category"] == category]
    if limit:
        cases = cases[:limit]
    return cases


def new_results(n_cases: int) -> dict:
    return {
        "run": {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "app_model": f"{APP_PROVIDER}/{APP_MODEL}",
            "judge_model": judge_name(),
            "prompt_hash": prompt_hash(),
            "snapshot_fetched_at": snapshot_stamp(),
            "n_cases": n_cases,
            "elapsed_s": 0.0,
        },
        "cases": [],
        "summary": {},
    }


def load_resumable(n_cases: int) -> dict | None:
    """Return the previous results if they were produced by the SAME
    configuration (app, judge, prompt, corpus) and are incomplete."""
    if not LATEST_FILE.exists():
        return None
    with open(LATEST_FILE, encoding="utf-8") as f:
        previous = json.load(f)
    fresh = new_results(n_cases)["run"]
    same_config = all(previous["run"].get(k) == fresh[k]
                      for k in ("app_model", "judge_model", "prompt_hash", "snapshot_fetched_at"))
    if not same_config:
        print("cannot resume: configuration differs from the previous run -- starting fresh")
        return None
    if len(previous["cases"]) >= previous["run"]["n_cases"]:
        print("previous run is already complete -- starting fresh")
        return None
    return previous


def run(ids: str | None = None, category: str | None = None,
        limit: int | None = None, resume: bool = False) -> dict:
    t_start = time.perf_counter()

    cases = select_cases(load_golden_data(), ids, category, limit)
    build_store()  # no-op if the vector store exists

    results = (load_resumable(len(cases)) if resume else None) or new_results(len(cases))
    # a case counts as done only if every metric produced a score; cases with
    # errored metrics are dropped and re-run
    results["cases"] = [c for c in results["cases"]
                        if all(m["score"] is not None for m in c["metrics"])]
    done_ids = {c["id"] for c in results["cases"]}
    if done_ids:
        print(f"resuming: {len(done_ids)} case(s) already done")
    already_elapsed = results["run"]["elapsed_s"]

    print(f"evaluating {len(cases)} case(s)  app={results['run']['app_model']}  "
          f"judge={results['run']['judge_model']}\n")

    for i, case in enumerate(cases, 1):
        if case["id"] in done_ids:
            continue
        print(f"[{i}/{len(cases)}] {case['id']}  ({case['category']})", flush=True)
        try:
            results["cases"].append(evaluate_case(case))
        except (ApplicationError, JudgeQuotaError) as exc:
            who = "APPLICATION" if isinstance(exc, ApplicationError) else "JUDGE (daily quota)"
            print(f"\n{who} FAILED on {case['id']}: {exc}")
            print(f"{len(results['cases'])}/{len(cases)} cases saved. "
                  f"Wait for the quota to reset, then re-run with --resume.")
            break
        finally:
            # save after EVERY case: a rate-limit stop half-way loses nothing
            results["run"]["elapsed_s"] = round(already_elapsed + time.perf_counter() - t_start, 1)
            results["summary"] = summarize(results["cases"])
            save_results(results)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the golden cases through the app and score them.")
    parser.add_argument("--ids", help="comma-separated case ids, e.g. case_01,case_12")
    parser.add_argument("--category", help="only cases in this category")
    parser.add_argument("--limit", type=int, help="only the first N selected cases")
    parser.add_argument("--resume", action="store_true",
                        help="continue the previous incomplete run (same config only)")
    args = parser.parse_args()

    results = run(ids=args.ids, category=args.category, limit=args.limit, resume=args.resume)
    print_summary(results)
    print("\nsaved -> results/latest_results.json")


if __name__ == "__main__":
    main()
