"""
app/retriever.py -- turn articles into searchable vectors, then search them.

    articles.json  ->  chunk  ->  embed (MiniLM)  ->  Chroma (on disk)
                                                          |
    "what did Anthropic say about CAPTCHAs?"  ->  embed  ->  nearest chunks

Two public functions:

    build_store(rebuild=False)   index the corpus (skipped if already built)
    retrieve(query, k=TOP_K)     -> list of {text, title, source_id, url, score}

Run from the project root:

    python -m app.retriever                       # build if needed, run demo queries
    python -m app.retriever --query "..."         # search for one question
    python -m app.retriever --rebuild             # wipe and re-index
"""

import argparse
import json
from functools import lru_cache

# Chroma needs sqlite >= 3.35. Some hosts (Streamlit Community Cloud) ship an
# older system sqlite; pysqlite3-binary provides a modern one on Linux only.
try:
    __import__("pysqlite3")
    import sys as _sys
    _sys.modules["sqlite3"] = _sys.modules.pop("pysqlite3")
except ImportError:
    pass

import chromadb
from sentence_transformers import SentenceTransformer

from config.settings import (
    ARTICLES_FILE,
    CHROMA_COLLECTION,
    CHROMA_DIR,
    CHUNK_OVERLAP_WORDS,
    CHUNK_WORDS,
    EMBEDDING_MODEL,
    TOP_K,
)


# ---------------------------------------------------------------------------
# 1. CHUNK  (one article -> one or more overlapping pieces)
# ---------------------------------------------------------------------------
def chunk_words(text: str, size: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP_WORDS) -> list[str]:
    """Split text into pieces of `size` words, each sharing `overlap` words
    with the previous piece so a sentence cut at the boundary survives in one
    of them. A text shorter than `size` comes back as a single chunk."""
    words = text.split()
    if len(words) <= size:
        return [text]

    chunks = []
    step = size - overlap
    for start in range(0, len(words), step):
        piece = words[start : start + size]
        chunks.append(" ".join(piece))
        if start + size >= len(words):
            break
    return chunks


def make_chunks(articles: list[dict]) -> list[dict]:
    """Every chunk carries the article's title (so the embedding knows the
    topic) plus metadata that points back to the source article."""
    chunks = []
    for article in articles:
        pieces = chunk_words(article["text"])
        for i, piece in enumerate(pieces):
            chunks.append(
                {
                    "id": f"{article['source_id']}_c{i}",
                    "text": f"{article['title']}\n\n{piece}",
                    "metadata": {
                        "source_id": article["source_id"],
                        "source": article["source"],
                        "title": article["title"],
                        "url": article["url"],
                        "chunk_index": i,
                    },
                }
            )
    return chunks


# ---------------------------------------------------------------------------
# 2. EMBED + STORE  (loaded once per process thanks to lru_cache)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    """The model that turns text into a vector. Downloads on first use."""
    return SentenceTransformer(EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_collection():
    """Open (or create) the on-disk Chroma collection."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={
            "hnsw:space": "cosine",  # similarity = 1 - cosine distance
            # How widely the index searches per query. The default (10) can
            # return FEWER than n_results on small collections, which would
            # make retrieval non-deterministic. 100 is cheap at our size.
            "hnsw:search_ef": 100,
        },
    )
    # The FIRST query against a freshly opened HNSW index sometimes returns
    # fewer than n_results (observed: 3 of 4). Warm it up with a throwaway
    # query so no real query is ever the cold one.
    if collection.count() > 0:
        collection.query(query_embeddings=[[0.0] * 384], n_results=1)
    return collection


def build_store(rebuild: bool = False) -> None:
    """Chunk + embed + store the corpus. No-op if the store is already populated."""
    collection = get_collection()

    if rebuild and collection.count() > 0:
        chromadb.PersistentClient(path=str(CHROMA_DIR)).delete_collection(CHROMA_COLLECTION)
        get_collection.cache_clear()
        collection = get_collection()

    if collection.count() > 0:
        print(f"store already built ({collection.count()} chunks). Use --rebuild to re-index.")
        return

    with open(ARTICLES_FILE, encoding="utf-8") as f:
        articles = json.load(f)
    chunks = make_chunks(articles)
    print(f"{len(articles)} articles -> {len(chunks)} chunks. Embedding with {EMBEDDING_MODEL} ...")

    embeddings = get_embedder().encode(
        [c["text"] for c in chunks], show_progress_bar=True, normalize_embeddings=True
    )

    collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
        embeddings=embeddings.tolist(),
    )
    print(f"stored {collection.count()} chunks in {CHROMA_DIR.name}/")


# ---------------------------------------------------------------------------
# 3. SEARCH
# ---------------------------------------------------------------------------
def retrieve(query: str, k: int = TOP_K) -> list[dict]:
    """Embed the query and return the k most similar chunks, best first."""
    query_vector = get_embedder().encode([query], normalize_embeddings=True)[0]

    result = get_collection().query(
        query_embeddings=[query_vector.tolist()],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    # Chroma's HNSW index very occasionally returns fewer than k hits (seen
    # once on a cold index even with search_ef=100). A second query is always
    # complete, and a stable k is essential for comparable evaluation runs.
    if len(result["ids"][0]) < min(k, get_collection().count()):
        result = get_collection().query(
            query_embeddings=[query_vector.tolist()],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

    hits = []
    for text, meta, distance in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        hits.append(
            {
                "text": text,
                "title": meta["title"],
                "source_id": meta["source_id"],
                "source": meta["source"],
                "url": meta["url"],
                "score": round(1 - distance, 3),  # cosine similarity, higher = closer
            }
        )
    return hits


# ---------------------------------------------------------------------------
# 4. ENTRYPOINT
# ---------------------------------------------------------------------------
DEMO_QUERIES = [
    "What did Anthropic reveal about AI agents and CAPTCHAs?",
    "Which companies are involved in distillation of frontier models?",
    "How are AI scientist workflows being evaluated?",
    "What is the best pizza recipe?",  # out of scope -- watch the scores
]


def print_hits(query: str, hits: list[dict]) -> None:
    print("\n" + "=" * 70)
    print(f"QUERY: {query}")
    print("=" * 70)
    for i, h in enumerate(hits, 1):
        print(f"[{i}] score={h['score']:.3f}  {h['source_id']}")
        print(f"    {h['title']}")
        body = h["text"].split("\n\n", 1)[-1]
        print(f"    {body[:160]}...")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and query the vector store.")
    parser.add_argument("--rebuild", action="store_true", help="wipe and re-index")
    parser.add_argument("--query", help="run a single query instead of the demo set")
    parser.add_argument("-k", type=int, default=TOP_K, help="number of chunks to return")
    args = parser.parse_args()

    build_store(rebuild=args.rebuild)

    for query in [args.query] if args.query else DEMO_QUERIES:
        print_hits(query, retrieve(query, k=args.k))


if __name__ == "__main__":
    main()
