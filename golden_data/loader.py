"""
golden_data/loader.py -- read and validate the golden dataset.

    from golden_data.loader import load_golden_data
    cases = load_golden_data()          # list[dict], validated

Both the evaluation pipeline and the Streamlit app call this, so the two can
never disagree about what a test case looks like.

Run it directly to validate the file and check that the retriever can find
each case's source articles (a cheap, LLM-free sanity check):

    python -m golden_data.loader
"""

import json

from config.settings import ARTICLES_FILE, GOLDEN_DATA_DIR

GOLDEN_FILE = GOLDEN_DATA_DIR / "test_cases.json"

# --- the schema, in one place -------------------------------------------------
REQUIRED_FIELDS = {
    "id", "category", "question", "expected_answer",
    "expected_behavior", "metrics", "source_ids", "notes",
}

CATEGORIES = {
    "factual", "multi_hop", "missing_information", "hallucination_trap",
    "contradiction", "ambiguous", "out_of_scope", "instruction_following", "relevance",
}

# What the application is expected to DO. Checked by the instruction-following metric.
BEHAVIORS = {
    "answer",              # give the information
    "abstain",             # reply with the exact ABSTENTION_MESSAGE
    "decline",             # out-of-scope / injected instruction: politely refuse
    "correct_assumption",  # the question's premise is false: say so (or abstain)
    "report_conflict",     # sources disagree: present both
    "clarify",             # question matches several items: ask, or keep them separate
}

METRICS = {"correctness", "faithfulness", "relevance", "instruction_following"}


def load_golden_data(path=GOLDEN_FILE) -> list[dict]:
    """Read test_cases.json, validate every case, return the list."""
    with open(path, encoding="utf-8") as f:
        cases = json.load(f)
    validate(cases)
    return cases


def validate(cases: list[dict]) -> None:
    """Raise ValueError on the first schema problem found."""
    seen_ids = set()
    known_sources = _known_source_ids()

    for case in cases:
        cid = case.get("id", "<no id>")

        missing = REQUIRED_FIELDS - case.keys()
        if missing:
            raise ValueError(f"{cid}: missing fields {sorted(missing)}")
        if cid in seen_ids:
            raise ValueError(f"{cid}: duplicate id")
        seen_ids.add(cid)

        if case["category"] not in CATEGORIES:
            raise ValueError(f"{cid}: unknown category '{case['category']}'")
        if case["expected_behavior"] not in BEHAVIORS:
            raise ValueError(f"{cid}: unknown expected_behavior '{case['expected_behavior']}'")

        bad_metrics = set(case["metrics"]) - METRICS
        if bad_metrics:
            raise ValueError(f"{cid}: unknown metrics {sorted(bad_metrics)}")
        if "correctness" in case["metrics"] and not case["expected_answer"]:
            raise ValueError(f"{cid}: correctness needs an expected_answer")

        unknown_sources = set(case["source_ids"]) - known_sources
        if unknown_sources:
            raise ValueError(f"{cid}: source_ids not in articles.json: {sorted(unknown_sources)}")


def _known_source_ids() -> set[str]:
    with open(ARTICLES_FILE, encoding="utf-8") as f:
        return {article["source_id"] for article in json.load(f)}


# ---------------------------------------------------------------------------
# Standalone: validate + retrieval sanity check
# ---------------------------------------------------------------------------
def main() -> None:
    cases = load_golden_data()
    print(f"loaded {len(cases)} cases - schema OK")

    per_category = {}
    for c in cases:
        per_category[c["category"]] = per_category.get(c["category"], 0) + 1
    for cat, n in sorted(per_category.items()):
        print(f"  {cat:<24} {n}")

    # Does the retriever actually surface the articles each case points at?
    # This is NOT an LLM evaluation - just "can the answer even reach the model?"
    from app.retriever import retrieve

    print("\nretrieval check (expected source_ids found in top-k?)")
    misses = 0
    for c in cases:
        if not c["source_ids"]:
            continue
        found = {h["source_id"] for h in retrieve(c["question"])}
        hit = [s for s in c["source_ids"] if s in found]
        status = "OK " if len(hit) == len(c["source_ids"]) else "MISS"
        if status == "MISS":
            misses += 1
        print(f"  {status} {c['id']}  {len(hit)}/{len(c['source_ids'])} sources retrieved")
    print(f"\n{misses} case(s) where retrieval misses at least one expected source")


if __name__ == "__main__":
    main()
