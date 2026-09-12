"""
pipeline/rescore.py -- re-run METRICS on answers already saved in a results file.

    python -m pipeline.rescore results/run_X.json
    python -m pipeline.rescore results/run_X.json --ids case_04,case_09 --metrics instruction_following

Why this exists: when a scoring rule changes (a rubric is tightened, a cheap
string check is fixed) the application's answers are still valid -- only the
scores need redoing. Re-scoring costs judge calls but NO application calls,
and keeps the answers identical, which is exactly what you want when the
thing under test is the metric, not the model.

The file is updated in place. Every re-scored metric row gets "rescored": true,
and the run metadata records what was re-scored and when, so the provenance
stays honest. latest_results.json is refreshed if it is the same run.
"""

import argparse
import json
from datetime import datetime, timezone

from eval_methods import METRICS
from golden_data.loader import load_golden_data
from pipeline.report import LATEST_FILE, summarize


def rescore(path: str, ids: set[str] | None, metric_names: set[str] | None) -> dict:
    with open(path, encoding="utf-8") as f:
        results = json.load(f)
    golden = {c["id"]: c for c in load_golden_data()}

    touched = []
    for case in results["cases"]:
        if ids and case["id"] not in ids:
            continue
        g = golden[case["id"]]
        for i, row in enumerate(case["metrics"]):
            name = row["metric"]
            if metric_names and name not in metric_names:
                continue
            old = row["score"]
            new = METRICS[name](
                question=case["question"], answer=case["answer"], context=case["context"],
                expected_answer=g["expected_answer"], expected_behavior=g["expected_behavior"],
            )
            new["rescored"] = True
            case["metrics"][i] = new
            touched.append(f"{case['id']}.{name}")
            print(f"  {case['id']:<8} {name:<22} {old} -> {new['score']}  {'PASS' if new['passed'] else 'FAIL'}")

    results["summary"] = summarize(results["cases"])
    results["run"].setdefault("rescored", []).append({
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": touched,
    })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # keep latest_results.json in step if it is this same run
    if LATEST_FILE.exists():
        with open(LATEST_FILE, encoding="utf-8") as f:
            latest = json.load(f)
        if latest["run"]["timestamp"] == results["run"]["timestamp"]:
            with open(LATEST_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Re-score saved answers with the current metrics.")
    ap.add_argument("results_file")
    ap.add_argument("--ids", help="comma-separated case ids (default: all)")
    ap.add_argument("--metrics", help="comma-separated metric names (default: all)")
    args = ap.parse_args()

    results = rescore(
        args.results_file,
        set(args.ids.split(",")) if args.ids else None,
        set(args.metrics.split(",")) if args.metrics else None,
    )
    from pipeline.report import print_summary
    print_summary(results)


if __name__ == "__main__":
    main()
