"""
app/rag_pipeline.py -- the whole application in one function.

    question -> retrieve() -> generate() -> answer

    from app.rag_pipeline import answer_question
    result = answer_question("What did Anthropic reveal about CAPTCHAs?")

answer_question() returns THE CONTRACT that the evaluation framework depends on:

    {
        "query":   the question that was asked,
        "context": [chunk text, ...]        # what the generator actually saw
        "answer":  the generated answer,
        "sources": [{source_id, title, url, score}, ...]   # for display / tracing
    }

Any other application that returns a dict with query / context / answer can be
evaluated by the same framework -- that is the reuse point of this project.

Run from the project root:
    python -m app.rag_pipeline
    python -m app.rag_pipeline --question "..."
"""

import argparse

from app.generator import generate
from app.retriever import build_store, retrieve
from config.settings import TOP_K


def answer_question(question: str, k: int = TOP_K,
                    provider: str | None = None, model: str | None = None) -> dict:
    """Retrieve context, generate an answer, return everything the evaluator needs.

    provider/model are optional overrides of .env (used by the Streamlit demo)."""
    # 1. RETRIEVE
    hits = retrieve(question, k=k)
    context = [hit["text"] for hit in hits]

    # 2. GENERATE
    answer = generate(question, context, provider=provider, model=model)

    # 3. RETURN THE CONTRACT
    return {
        "query": question,
        "context": context,
        "answer": answer,
        "sources": [
            {"source_id": h["source_id"], "title": h["title"], "url": h["url"], "score": h["score"]}
            for h in hits
        ],
    }


DEMO_QUESTIONS = [
    "What did Anthropic reveal about AI agents and CAPTCHAs?",           # factual
    "Which companies did Anthropic accuse of distillation campaigns?",   # factual, list
    "How much did OpenAI pay the mathematicians?",                      # hallucination trap
    "What is the best pizza recipe?",                                   # out of scope
]


def print_result(result: dict) -> None:
    print("\n" + "=" * 70)
    print(f"QUESTION: {result['query']}")
    print("-" * 70)
    print(result["answer"])
    print("-" * 70)
    print("retrieved:")
    for s in result["sources"]:
        print(f"  {s['score']:.3f}  {s['source_id']}  {s['title'][:60]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the news assistant a question.")
    parser.add_argument("--question", help="a single question (default: demo set)")
    args = parser.parse_args()

    build_store()  # no-op if already built
    for question in [args.question] if args.question else DEMO_QUESTIONS:
        print_result(answer_question(question))


if __name__ == "__main__":
    main()
