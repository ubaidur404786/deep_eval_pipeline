"""
debug_walkthrough.py -- watch one test case travel through the whole system.

Runs the REAL functions in the REAL order, printing what goes in and what
comes out at every step. Safe by default:

    * no LLM call      (the application model is replaced by a stand-in)
    * no judge call    (metrics that need the judge are replaced by a stand-in;
                        the deterministic "cheap paths" run for real)
    * nothing written  (results/ is never touched)

    python debug_walkthrough.py                        # fully offline
    python debug_walkthrough.py --case case_16         # a different case
    python debug_walkthrough.py --pause                # press Enter between steps
    python debug_walkthrough.py --live-app             # 1 real application call (uses APP quota)
    python debug_walkthrough.py --live-app --live-judge  # + real judge calls (uses JUDGE quota)

To step through in a debugger instead, put a breakpoint on any line marked
"# <-- breakpoint here" and run this file under VS Code / PyCharm / pdb:

    python -m pdb debug_walkthrough.py
"""

import argparse
import hashlib
import json
import textwrap

# ---------------------------------------------------------------------------
# tiny helpers for readable output
# ---------------------------------------------------------------------------
STEP = 0


def banner(title: str) -> None:
    global STEP
    STEP += 1
    print("\n" + "=" * 78)
    print(f"STEP {STEP}  {title}")
    print("=" * 78)


def show(label: str, value, width: int = 600) -> None:
    text = value if isinstance(value, str) else json.dumps(value, indent=2, ensure_ascii=False)
    if len(text) > width:
        text = text[:width] + f"  … [{len(text) - width} more chars]"
    print(f"\n{label}:")
    print(textwrap.indent(text, "    "))


def pause(enabled: bool) -> None:
    if enabled:
        input("\n[Enter] to continue …")


# ---------------------------------------------------------------------------
# the walkthrough
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="case_12", help="golden case id (default: case_12, an abstention)")
    ap.add_argument("--pause", action="store_true", help="wait for Enter between steps")
    ap.add_argument("--live-app", action="store_true", help="make ONE real application call")
    ap.add_argument("--live-judge", action="store_true", help="make real judge calls for this case")
    args = ap.parse_args()

    # ======================================================================
    banner("Configuration  (config/settings.py)")
    # ======================================================================
    from config import settings                                   # <-- breakpoint here

    show("where things live", {
        "PROJECT_ROOT": str(settings.PROJECT_ROOT),
        "RAW_SNAPSHOT_FILE": str(settings.RAW_SNAPSHOT_FILE.relative_to(settings.PROJECT_ROOT)),
        "ARTICLES_FILE": str(settings.ARTICLES_FILE.relative_to(settings.PROJECT_ROOT)),
        "CHROMA_DIR": str(settings.CHROMA_DIR.relative_to(settings.PROJECT_ROOT)),
        "SYSTEM_PROMPT_FILE": str(settings.SYSTEM_PROMPT_FILE.relative_to(settings.PROJECT_ROOT)),
    })
    show("models (from .env)", {
        "APP": f"{settings.APP_PROVIDER}/{settings.APP_MODEL}",
        "JUDGE": f"{settings.JUDGE_PROVIDER}/{settings.JUDGE_MODEL}",
        "PASS_THRESHOLD": settings.PASS_THRESHOLD,
        "TOP_K": settings.TOP_K,
    })
    prompt_text = settings.load_system_prompt()
    show("system prompt (first 400 chars)", prompt_text, 400)
    show("prompt hash (sha256, 12 chars) — recorded with every result",
         hashlib.sha256(settings.SYSTEM_PROMPT_FILE.read_bytes()).hexdigest()[:12])
    show("ABSTENTION_MESSAGE (exact string the cheap checks look for)", settings.ABSTENTION_MESSAGE)
    pause(args.pause)

    # ======================================================================
    banner("Frozen corpus  (data/raw/snapshot.json -> data/processed/articles.json)")
    # ======================================================================
    with open(settings.RAW_SNAPSHOT_FILE, encoding="utf-8") as f:
        snapshot = json.load(f)
    with open(settings.ARTICLES_FILE, encoding="utf-8") as f:
        articles = json.load(f)
    show("snapshot metadata", {"fetched_at": snapshot["fetched_at"], "feeds": list(snapshot["feeds"]),
                               "raw entries": len(snapshot["entries"])})
    show("one processed article (what the retriever indexes)", {**articles[0], "text": articles[0]["text"][:200] + " …"})
    pause(args.pause)

    # ======================================================================
    banner("Golden dataset  (golden_data/loader.py)")
    # ======================================================================
    from golden_data.loader import load_golden_data                # <-- breakpoint here

    cases = load_golden_data()          # reads + VALIDATES test_cases.json
    case = next(c for c in cases if c["id"] == args.case)
    show(f"loaded {len(cases)} cases; this walkthrough uses", case)
    print("\n    -> 'metrics' decides WHICH metrics run for this case; 'expected_behavior' is what")
    print("       instruction_following checks; 'source_ids' lets the pipeline check retrieval for free.")
    pause(args.pause)

    # ======================================================================
    banner("Retrieval  (app/retriever.py)  — local, deterministic, no API")
    # ======================================================================
    from app.retriever import build_store, retrieve, get_collection   # <-- breakpoint here

    build_store()                        # no-op when chroma_store/ exists
    show("vector store", {"chunks indexed": get_collection().count(), "top_k": settings.TOP_K})
    hits = retrieve(case["question"])    # embed the question -> nearest chunks
    show("retrieve(question) -> hits (score = cosine similarity)",
         [{k: (v[:90] + "…" if k == "text" else v) for k, v in h.items()} for h in hits], 1400)

    retrieved_ids = {h["source_id"] for h in hits}
    found = [s for s in case["source_ids"] if s in retrieved_ids]
    show("expected_sources_found (the pipeline's free retrieval check)",
         f"{len(found)}/{len(case['source_ids'])}" if case["source_ids"] else "n/a (case has no expected sources)")
    pause(args.pause)

    # ======================================================================
    banner("Prompt assembly  (app/generator.py)  — still no API")
    # ======================================================================
    from app.generator import build_prompt                          # <-- breakpoint here

    context = [h["text"] for h in hits]
    prompt = build_prompt(case["question"], context)
    show("the exact string sent to the application model (last 900 chars)", prompt[-900:], 900)
    show("prompt size", {"characters": len(prompt), "chunks inside": len(context)})
    pause(args.pause)

    # ======================================================================
    banner("Generation  (app/llm_client.py -> provider)")
    # ======================================================================
    from app import generator, llm_client                           # <-- breakpoint here

    if args.live_app:
        print(f"\n    LIVE: calling {settings.APP_PROVIDER}/{settings.APP_MODEL} once …")
        answer = generator.generate(case["question"], context)
    else:
        print("\n    OFFLINE: the provider call is replaced by a stand-in (no quota used).")
        print(f"    (llm_client.PROVIDERS has: {list(llm_client.PROVIDERS)} — complete() picks one by APP_PROVIDER)")
        answer = settings.ABSTENTION_MESSAGE if case["expected_behavior"] == "abstain" else (
            "STAND-IN ANSWER: the sources say X and Y.\n\nSources: " + hits[0]["title"])
    show("answer", answer)

    result = {"query": case["question"], "context": context, "answer": answer,
              "sources": [{"source_id": h["source_id"], "title": h["title"], "score": h["score"]} for h in hits]}
    show("THE CONTRACT the framework receives ({query, context, answer} + sources for display)",
         {**result, "context": [c[:60] + "…" for c in context]}, 900)
    print("\n    -> Any application that returns this dict can be evaluated. Nothing below imports app/.")
    pause(args.pause)

    # ======================================================================
    banner("Evaluation  (eval_methods/)  — one function per metric, same signature")
    # ======================================================================
    from eval_methods import METRICS                                 # <-- breakpoint here
    from eval_methods.instruction_following import has_sources_line
    from eval_methods.common import strip_sources_line

    show("METRICS registry (name -> evaluate function)", {k: f"{v.__module__}.{v.__name__}" for k, v in METRICS.items()})
    show("helpers the cheap paths use", {
        "answer is the exact abstention?": answer.strip() == settings.ABSTENTION_MESSAGE,
        "has_sources_line(answer)": has_sources_line(answer),
        "strip_sources_line(answer) -> what faithfulness/relevance judge": strip_sources_line(answer)[:120],
    })

    rows = []
    for name in case["metrics"]:
        fn = METRICS[name]
        cheap = (name == "faithfulness" and answer.strip() == settings.ABSTENTION_MESSAGE) or \
                (name == "instruction_following" and (
                    (case["expected_behavior"] == "abstain" and answer.strip() == settings.ABSTENTION_MESSAGE) or
                    (case["expected_behavior"] == "answer" and answer.strip() != settings.ABSTENTION_MESSAGE
                     and not has_sources_line(answer))))
        if cheap or args.live_judge:
            tag = "REAL (deterministic cheap path, no judge)" if cheap else "REAL (live judge call)"
            row = fn(question=case["question"], answer=answer, context=context,
                     expected_answer=case["expected_answer"], expected_behavior=case["expected_behavior"])
        else:
            tag = "STAND-IN (would call the judge; skipped to save quota)"
            row = {"metric": name, "score": 0.88, "passed": True, "reason": "stand-in reason — run with --live-judge to see the judge's real reasoning"}
        rows.append(row)
        show(f"{name}  [{tag}]", row)
    pause(args.pause)

    # ======================================================================
    banner("Aggregation  (pipeline/report.py)  — never pooled across metrics")
    # ======================================================================
    from pipeline.report import summarize                            # <-- breakpoint here

    case_result = {"id": case["id"], "category": case["category"], "question": case["question"],
                   "answer": answer, "expected_sources_found": f"{len(found)}/{len(case['source_ids'])}" if case["source_ids"] else None,
                   "metrics": rows}
    show("summarize([one case]) -> per_metric / per_category", summarize([case_result]))
    print("\n    In the real pipeline this is recomputed and SAVED after every case")
    print("    (results/latest_results.json + a timestamped copy). This walkthrough writes nothing.")

    print("\n" + "=" * 78)
    print("END  question -> retrieve -> prompt -> LLM -> {query,context,answer} -> metrics -> summary")
    print("=" * 78)


if __name__ == "__main__":
    main()
