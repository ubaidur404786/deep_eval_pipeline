"""
app/ingest.py -- turn public RSS feeds into a FROZEN, readable corpus.

Two steps, two output files:

  1. FETCH   feeds -> data/raw/snapshot.json
             The feed entries exactly as received (title, link, date, summary).
             Written ONCE. Re-fetching requires --refresh, on purpose: the
             golden dataset is authored against this file, so silently
             replacing it would make every expected answer wrong.

  2. PROCESS snapshot.json -> data/processed/articles.json
             Cleaned text + a stable source_id per article. This is what the
             retriever indexes and what golden cases point at via source_ids.

Run from the project root:

    python -m app.ingest             # process the existing snapshot (fetch if none exists)
    python -m app.ingest --refresh   # re-fetch the feeds and overwrite the snapshot
"""

import argparse
import hashlib
import html
import json
import re
from datetime import datetime, timezone

import feedparser
import requests

from config.settings import (
    ARTICLES_FILE,
    FEEDS,
    HTTP_TIMEOUT_SECONDS,
    HTTP_USER_AGENT,
    MAX_ITEMS_PER_FEED,
    RAW_SNAPSHOT_FILE,
)


# ---------------------------------------------------------------------------
# 1. FETCH  (network -> raw snapshot)
# ---------------------------------------------------------------------------
def entry_body(entry) -> str:
    """Longest text a feed gives us: full <content> if present, else <summary>."""
    content = entry.get("content")
    if content:
        return content[0].get("value", "")
    return entry.get("summary", "")


def fetch_feed(name: str, url: str) -> list[dict]:
    """Download one RSS feed and return its entries as plain dicts."""
    response = requests.get(
        url,
        headers={"User-Agent": HTTP_USER_AGENT},
        timeout=HTTP_TIMEOUT_SECONDS,
    )
    response.raise_for_status()  # turn HTTP 4xx/5xx into an exception

    parsed = feedparser.parse(response.content)

    entries = []
    for entry in parsed.entries:
        # arXiv re-announces updated papers as "replace"; keep only brand-new ones
        if entry.get("arxiv_announce_type", "new") != "new":
            continue

        entries.append(
            {
                "feed": name,
                "title": entry.get("title", "").strip(),
                "link": entry.get("link", ""),
                "published": entry.get("published", ""),
                "summary": entry_body(entry),
            }
        )
        if len(entries) >= MAX_ITEMS_PER_FEED:
            break

    return entries


def fetch_all_feeds() -> dict:
    """Fetch every feed in settings.FEEDS. Returns the snapshot structure."""
    entries = []
    for name, url in FEEDS.items():
        print(f"fetching {name} ... ", end="", flush=True)
        feed_entries = fetch_feed(name, url)
        print(f"{len(feed_entries)} entries")
        entries.extend(feed_entries)

    return {
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "feeds": FEEDS,
        "entries": entries,
    }


def save_json(data, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 2. PROCESS  (raw snapshot -> clean articles)
# ---------------------------------------------------------------------------
# arXiv summaries start with "arXiv:2609.09203v1 Announce Type: new \nAbstract: "
ARXIV_PREFIX = re.compile(r"^arXiv:\S+\s+Announce Type:\s*\w+\s*Abstract:\s*", re.IGNORECASE)
HTML_TAG = re.compile(r"<[^>]+>")


def clean_text(raw: str) -> str:
    """Strip HTML tags, decode entities, drop arXiv boilerplate, tidy whitespace."""
    text = HTML_TAG.sub(" ", raw)
    text = html.unescape(text)
    text = ARXIV_PREFIX.sub("", text)
    text = re.sub(r"\s+", " ", text)  # collapse newlines / double spaces
    return text.strip()


def make_source_id(feed: str, link: str) -> str:
    """Stable, short id derived from the link, e.g. 'arxiv_ai_3f9c2a1b'.

    A hash of the link (rather than a running number) means the id does not
    change if the order of entries changes between fetches.
    """
    digest = hashlib.sha1(link.encode("utf-8")).hexdigest()[:8]
    return f"{feed}_{digest}"


def build_articles(snapshot: dict) -> list[dict]:
    """Convert raw snapshot entries into clean, id-stamped articles."""
    articles = []
    for entry in snapshot["entries"]:
        text = clean_text(entry["summary"])
        if not text:
            continue  # nothing to index

        articles.append(
            {
                "source_id": make_source_id(entry["feed"], entry["link"]),
                "source": entry["feed"],
                "title": clean_text(entry["title"]),
                "url": entry["link"],
                "published": entry["published"],
                "text": text,
            }
        )
    return articles


# ---------------------------------------------------------------------------
# 3. ENTRYPOINT
# ---------------------------------------------------------------------------
def print_summary(snapshot: dict, articles: list[dict]) -> None:
    print("\n" + "=" * 60)
    print(f"snapshot fetched_at : {snapshot['fetched_at']}")
    print(f"raw entries         : {len(snapshot['entries'])}")
    print(f"clean articles      : {len(articles)}")
    per_feed = {}
    for a in articles:
        per_feed[a["source"]] = per_feed.get(a["source"], 0) + 1
    for feed, n in per_feed.items():
        print(f"  {feed:<15} {n:>4}")
    print(f"written             : {ARTICLES_FILE.relative_to(ARTICLES_FILE.parents[2])}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and freeze the news corpus.")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="re-download the feeds and OVERWRITE data/raw/snapshot.json",
    )
    args = parser.parse_args()

    if args.refresh or not RAW_SNAPSHOT_FILE.exists():
        snapshot = fetch_all_feeds()
        save_json(snapshot, RAW_SNAPSHOT_FILE)
        print(f"saved raw snapshot -> {RAW_SNAPSHOT_FILE.name}")
    else:
        snapshot = load_json(RAW_SNAPSHOT_FILE)
        print(f"using frozen snapshot from {snapshot['fetched_at']} (pass --refresh to re-fetch)")

    articles = build_articles(snapshot)
    save_json(articles, ARTICLES_FILE)
    print_summary(snapshot, articles)


if __name__ == "__main__":
    main()
