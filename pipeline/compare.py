"""
pipeline/compare.py -- put two evaluation runs side by side.

    python -m pipeline.compare results/run_A.json results/run_B.json

Prints per-metric averages / pass rates for both runs, then a per-case table
showing where the two runs disagree. Refuses to compare runs that differ in
judge, prompt or corpus -- those numbers are not comparable, and pretending
otherwise is the most common way to fool yourself with evaluations.

This is deliberately NOT a regression gate (no tolerances, no verdict). It is
a reading aid. Run the same configuration twice first to learn your noise
floor; only differences larger than that mean anything.
"""

import argparse
import json


MUST_MATCH = ("judge_model", "prompt_hash", "snapshot_fetched_at")


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def check_comparable(a: dict, b: dict) -> None:
    for key in MUST_MATCH:
        if a["run"].get(key) != b["run"].get(key):
            raise SystemExit(
                f"NOT COMPARABLE: '{key}' differs ({a['run'].get(key)} vs {b['run'].get(key)}). "
                "Only the app model should differ between compared runs."
            )


def label(run: dict) -> str:
    return run["run"]["app_model"]


def per_metric_table(a: dict, b: dict) -> None:
    la, lb = label(a), label(b)
    print(f"\n{'metric':<22} {'':>2} {la:>26} {lb:>26}")
    print(f"{'':<22} {'':>2} {'pass    avg    min':>26} {'pass    avg    min':>26}")
    print("-" * 80)
    names = sorted(set(a["summary"]["per_metric"]) | set(b["summary"]["per_metric"]))
    for name in names:
        row = f"{name:<22}   "
        for run in (a, b):
            s = run["summary"]["per_metric"].get(name)
            if s and s["avg_score"] is not None:
                row += f"{s['pass_rate']:>6.0%} {s['avg_score']:>6.2f} {s['min_score']:>6.2f}   err={s['errors']:<2}"
            else:
                row += f"{'-':>26}"
        print(row)


def per_case_table(a: dict, b: dict) -> None:
    la, lb = label(a), label(b)
    cases_a = {c["id"]: c for c in a["cases"]}
    cases_b = {c["id"]: c for c in b["cases"]}
    print(f"\n{'case':<9} {'category':<22} {la[-22:]:>24} {lb[-22:]:>24}   note")
    print("-" * 100)
    for cid in sorted(set(cases_a) | set(cases_b)):
        ca, cb = cases_a.get(cid), cases_b.get(cid)
        fa = _failed(ca)
        fb = _failed(cb)
        note = ""
        if ca and cb:
            if fa and not fb:
                note = f"<- {lb} better"
            elif fb and not fa:
                note = f"<- {la} better"
            elif fa and fb and fa != fb:
                note = "both fail, differently"
        cat = (ca or cb)["category"]
        print(f"{cid:<9} {cat:<22} {_short(fa):>24} {_short(fb):>24}   {note}")


def _failed(case: dict | None) -> list[str]:
    if case is None:
        return ["(not run)"]
    return [f"{m['metric'][:5]}={m['score']}" for m in case["metrics"] if not m["passed"]]


def _short(failed: list[str]) -> str:
    return "ok" if not failed else ", ".join(failed)[:24]


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare two evaluation runs side by side.")
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    args = ap.parse_args()

    a, b = load(args.run_a), load(args.run_b)
    check_comparable(a, b)

    print("=" * 80)
    print(f"A: {label(a)}   ({len(a['cases'])} cases, {a['run']['elapsed_s']:.0f}s)")
    print(f"B: {label(b)}   ({len(b['cases'])} cases, {b['run']['elapsed_s']:.0f}s)")
    print(f"judge {a['run']['judge_model']}   prompt {a['run']['prompt_hash']}   corpus {a['run']['snapshot_fetched_at']}")
    print("=" * 80)
    per_metric_table(a, b)
    per_case_table(a, b)
    print()


if __name__ == "__main__":
    main()
