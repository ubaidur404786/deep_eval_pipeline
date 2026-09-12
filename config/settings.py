"""
config/settings.py -- every tunable value in ONE place.

Rule: Python files elsewhere never hard-code a path, a model name, or a feed
URL. They import it from here. Secrets (API keys) come from .env and are read
with os.getenv(); everything else is a plain constant you can edit.

    from config.settings import RAW_SNAPSHOT_FILE, load_system_prompt
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # reads .env into os.environ (no-op if .env does not exist)


# ---------------------------------------------------------------------------
# 1. PATHS  (all relative to the project root, so scripts work from anywhere)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
GOLDEN_DATA_DIR = PROJECT_ROOT / "golden_data"
RESULTS_DIR = PROJECT_ROOT / "results"

# The FROZEN corpus. ingest.py only overwrites it when you pass --refresh.
# The golden dataset is written against this exact file, so refreshing it
# means "I am about to re-author my golden cases".
RAW_SNAPSHOT_FILE = DATA_RAW_DIR / "snapshot.json"
ARTICLES_FILE = DATA_PROCESSED_DIR / "articles.json"

SYSTEM_PROMPT_FILE = PROJECT_ROOT / "config" / "system_prompt.txt"


# ---------------------------------------------------------------------------
# 2. DATA SOURCES  (free, public, key-less RSS feeds -- verified 2026-09-10)
# ---------------------------------------------------------------------------
FEEDS = {
    # name  : url
    "arxiv_ai": "http://export.arxiv.org/rss/cs.AI",
    "techcrunch_ai": "https://techcrunch.com/category/artificial-intelligence/feed/",
    "verge_ai": "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml",
}

# Cap per feed so the corpus stays small enough to read by hand (we need to
# read it to write golden cases). arXiv publishes 100-400 papers a day.
MAX_ITEMS_PER_FEED = 60

# Be a polite client: identify ourselves. arXiv asks for this.
HTTP_USER_AGENT = "deep-eval-pipeline/0.1 (learning project; RSS reader)"
HTTP_TIMEOUT_SECONDS = 30


# ---------------------------------------------------------------------------
# 3. RETRIEVAL  (free, local, CPU-friendly)
# ---------------------------------------------------------------------------
# Where Chroma keeps the vector store on disk. Regenerable: delete it and run
# `python -m app.retriever --rebuild`.
CHROMA_DIR = PROJECT_ROOT / "chroma_store"
CHROMA_COLLECTION = "news_articles"

# Sentence-transformers model. ~90 MB, downloads once, runs fine on CPU.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Chunking, in WORDS (simpler to reason about than characters or tokens).
# Most articles (~200 words) become exactly one chunk; only the long ones split.
CHUNK_WORDS = 200
CHUNK_OVERLAP_WORDS = 40

# How many chunks the retriever hands to the generator.
TOP_K = 4


# ---------------------------------------------------------------------------
# 4. MODELS  (values live in .env so they can change without a code edit)
# ---------------------------------------------------------------------------
APP_PROVIDER = os.getenv("APP_PROVIDER", "gemini")
APP_MODEL = os.getenv("APP_MODEL", "gemini-3.1-flash-lite")

JUDGE_PROVIDER = os.getenv("JUDGE_PROVIDER", "gemini")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemini-3.5-flash-lite")


# ---------------------------------------------------------------------------
# 5. EVALUATION
# ---------------------------------------------------------------------------
# A metric "passes" when its 0-1 score is >= this. One threshold for every
# metric keeps results comparable; per-metric thresholds are a later refinement.
PASS_THRESHOLD = 0.7


# ---------------------------------------------------------------------------
# 6. SYSTEM PROMPT  (kept as a text file so it can be edited without Python)
# ---------------------------------------------------------------------------
def load_system_prompt() -> str:
    """Read config/system_prompt.txt and return it as a string."""
    return SYSTEM_PROMPT_FILE.read_text(encoding="utf-8").strip()


# The exact sentence the assistant must say when the sources don't cover the
# question. Kept as a constant so "did it abstain?" is a cheap string check,
# not an LLM call. The system prompt must use this sentence verbatim.
ABSTENTION_MESSAGE = "I don't have enough information in the provided sources to answer that."
