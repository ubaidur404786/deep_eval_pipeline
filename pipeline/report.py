"""
pipeline/report.py -- aggregate, save, and print evaluation results.

    summarize(case_results)   -> {"per_metric": {...}, "per_category": {...}}
    save_results(results)     -> results/latest_results.json (+ a timestamped copy)
    print_summary(results)    -> readable table on the terminal

Per-metric numbers are NEVER pooled into one "quality score": a collapse in
faithfulness must not hide behind a strong relevance average.
"""

import json
from datetime import datetime

from config.settings import RESULTS_DIR

LATEST_FILE = RESULTS_DIR / "latest_results.json"


def summarize(case_results: list[dict]) -> dict:
    """Roll per-case metric rows up into per-metric and per-category stats."""
    per_metric: dict[str, dict] = {}
    per_category: dict[str, dict] = {}

    for case in case_results:
        cat = per_category.setdefault(case["category"], {"cases": 0, "all_passed": 0})
        cat["cases"] += 1
        case_ok = True

        for m in case["metrics"]:
            bucket = per_metric.setdefault(m["metric"], {"n": 0, "passed": 0, "scores": [], "errors": 0})
            bucket["n"] += 1
            if m["score"] is None:            # metric crashed (quota, parse error, ...)
                bucket["errors"] += 1
                case_ok = False
                continue
            bucket["scores"].append(m["score"])
            if m["passed"]:
                bucket["passed"] += 1
            else:
                case_ok = False

        if case_ok:
            cat["all_passed"] += 1

    for name, b in per_metric.items():
        scored = len(b["scores"])
        per_metric[name] = {
            "n": b["n"],
            "errors": b["errors"],
            "pass_rate": round(b["passed"] / scored, 3) if scored else None,
            "avg_score": round(sum(b["scores"]) / scored, 3) if scored else None,
            "min_score": min(b["scores"]) if scored else None,
        }
    for name, c in per_category.items():
        per_category[name] = {
            "cases": c["cases"],
            "all_metrics_passed": c["all_passed"],
            "pass_rate": round(c["all_passed"] / c["cases"], 3),
        }
    return {"per_metric": per_metric, "per_category": per_category}


def save_results(results: dict) -> None:
    """Write latest_results.json and a timestamped copy for history."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # Use the run's own start time, so repeated saves during one run
    # (after every case) keep overwriting the SAME history file.
    started = datetime.fromisoformat(results["run"]["timestamp"])
    stamp = started.strftime("%Y%m%d_%H%M%S")
    model_slug = results["run"]["app_model"].replace("/", "_")
    for path in (LATEST_FILE, RESULTS_DIR / f"run_{model_slug}_{stamp}.json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


def print_summary(results: dict) -> None:
    run = results["run"]
    summary = results["summary"]

    print("\n" + "=" * 72)
    print("EVALUATION SUMMARY")
    print("=" * 72)
    print(f"app model    : {run['app_model']}")
    print(f"judge model  : {run['judge_model']}")
    print(f"prompt hash  : {run['prompt_hash']}    cases: {run['n_cases']}    "
          f"time: {run['elapsed_s']:.0f}s")

    print("\nper metric")
    print(f"  {'metric':<22} {'n':>3} {'pass':>6} {'avg':>6} {'min':>6} {'err':>4}")
    for name, s in summary["per_metric"].items():
        pr = f"{s['pass_rate']:.0%}" if s["pass_rate"] is not None else "-"
        avg = f"{s['avg_score']:.2f}" if s["avg_score"] is not None else "-"
        mn = f"{s['min_score']:.2f}" if s["min_score"] is not None else "-"
        print(f"  {name:<22} {s['n']:>3} {pr:>6} {avg:>6} {mn:>6} {s['errors']:>4}")

    print("\nper category (case passes only if ALL its metrics pass)")
    for name, c in summary["per_category"].items():
        print(f"  {name:<22} {c['all_metrics_passed']}/{c['cases']}  ({c['pass_rate']:.0%})")

    failed = [c for c in results["cases"] if any(not m["passed"] for m in c["metrics"])]
    if failed:
        print(f"\nfailed cases ({len(failed)})")
        for c in failed:
            bad = ", ".join(f"{m['metric']}={m['score']}" for m in c["metrics"] if not m["passed"])
            print(f"  {c['id']}  [{c['category']}]  {bad}")
    print("=" * 72)
