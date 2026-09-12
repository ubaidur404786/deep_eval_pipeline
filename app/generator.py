"""
app/generator.py -- turn (question + retrieved chunks) into an answer.

    system_prompt.txt  +  numbered context  +  question
                              |
                        llm_client.complete()
                              |
                           answer text

    from app.generator import generate
    answer = generate("What did Anthropic reveal?", [chunk1, chunk2])

This file knows about the PROMPT. It does not know which LLM provider is in
use (llm_client) or where the chunks came from (retriever).
"""

from app.llm_client import complete
from config.settings import load_system_prompt


def format_context(chunks: list[str]) -> str:
    """Number the chunks so the model can cite them and the prompt stays readable.

    Each chunk already begins with its article title (see retriever.make_chunks),
    so "[1] Title\\n\\nbody" reads naturally."""
    if not chunks:
        return "(no sources retrieved)"
    return "\n\n".join(f"[{i}] {chunk}" for i, chunk in enumerate(chunks, 1))


def build_prompt(question: str, chunks: list[str]) -> str:
    """Fill the {context} and {question} slots of config/system_prompt.txt."""
    template = load_system_prompt()
    return template.replace("{context}", format_context(chunks)).replace("{question}", question)


def generate(question: str, chunks: list[str], provider: str | None = None, model: str | None = None) -> str:
    """The generator: prompt in, grounded answer out.

    provider/model default to .env (APP_PROVIDER / APP_MODEL); the Streamlit
    demo passes them explicitly so a viewer can switch models live."""
    prompt = build_prompt(question, chunks)
    if provider and model:
        return complete(prompt, provider=provider, model=model)
    return complete(prompt)


# quick manual test with hand-written context (no retriever needed):
#   python -m app.generator
if __name__ == "__main__":
    demo_chunks = [
        "Anthropic reveals rogue AI agents hate CAPTCHAs, just like you\n\n"
        "Come inside the mind of a bot trying to convince the internet it's human.",
    ]
    print("--- in-scope question ---")
    print(generate("What did Anthropic say about AI agents and CAPTCHAs?", demo_chunks))
    print("\n--- question the context cannot answer ---")
    print(generate("How much funding did Anthropic raise this week?", demo_chunks))
