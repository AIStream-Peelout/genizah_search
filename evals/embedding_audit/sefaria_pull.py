"""Pull Sefaria data for embedder training (concept bridge + canonical texts). Resumable, polite.

Stages
------
``topics``  — every topic's graph links and source refs (``/api/v2/topics/<slug>?with_links=1&with_refs=1``),
              one JSON per topic. The links give the concept-implication graph (lulav *participates-in* Sukkot,
              *member-of* the four species…); the refs carry Sefaria's curated per-source titles/prompts, which
              are natural-language descriptions of passages (query-like text paired with a source).
``texts``   — Hebrew + English text for a list of refs (``/api/v3/texts/<ref>``), one JSON per ref; the ref
              list comes from ``--refs-file`` (one ref per line), e.g. the curated topic refs or KTIV frames.

Everything is cached under ``<AUDIT_ROOT>/sefaria/``; existing files are skipped. Requests are throttled to
``--rate`` per second with retries on 429/5xx.
"""

import argparse
import json
import re
import time
import urllib.parse
from pathlib import Path

import requests

from embed_utils import AUDIT_ROOT

OUT = AUDIT_ROOT / "sefaria"
BASE = "https://www.sefaria.org"
SESSION = requests.Session()
SESSION.headers["User-Agent"] = "cairogenizah.ai research (embedding training data; polite crawler)"


def safe_name(text: str) -> str:
    """Filesystem-safe file stem.

    :param text: Slug or ref.
    :returns: Safe stem.
    :rtype: str
    """
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)[:180]


def fetch(url: str, rate: float, retries: int = 5) -> dict:
    """GET JSON with throttling and retries.

    :param url: URL.
    :param rate: Max requests per second.
    :param retries: Retry count.
    :returns: Parsed JSON ({} on persistent 404).
    :rtype: dict
    """
    for attempt in range(retries):
        time.sleep(1.0 / rate)
        try:
            r = SESSION.get(url, timeout=60)
        except requests.RequestException:
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code == 404:
            return {}
        time.sleep(10 * (attempt + 1))  # 429 / 5xx back-off
    raise RuntimeError(f"giving up on {url}")


def pull_topics(rate: float) -> None:
    """Stage 1: all topics with links + refs.

    :param rate: Requests per second.
    """
    tdir = OUT / "topics"
    tdir.mkdir(parents=True, exist_ok=True)
    index_path = OUT / "topics_index.json"
    if not index_path.exists():
        index_path.write_text(json.dumps(fetch(f"{BASE}/api/topics?limit=0", rate)))
    topics = json.loads(index_path.read_text())
    done = 0
    for t in topics:
        slug = t["slug"]
        path = tdir / f"{safe_name(slug)}.json"
        if path.exists():
            continue
        data = fetch(f"{BASE}/api/v2/topics/{urllib.parse.quote(slug)}?with_links=1&with_refs=1", rate)
        path.write_text(json.dumps(data, ensure_ascii=False))
        done += 1
        if done % 200 == 0:
            print(f"topics fetched {done} (of {len(topics)})", flush=True)
    print("topics done", flush=True)


def pull_texts(refs_file: Path, rate: float) -> None:
    """Stage 2: Hebrew + English text for refs.

    :param refs_file: One Sefaria ref per line.
    :param rate: Requests per second.
    """
    xdir = OUT / "texts"
    xdir.mkdir(parents=True, exist_ok=True)
    refs = [r.strip() for r in refs_file.read_text().splitlines() if r.strip()]
    done = 0
    for ref in refs:
        path = xdir / f"{safe_name(ref)}.json"
        if path.exists():
            continue
        url = f"{BASE}/api/v3/texts/{urllib.parse.quote(ref)}?version=hebrew&version=english&return_format=text_only"
        path.write_text(json.dumps(fetch(url, rate), ensure_ascii=False))
        done += 1
        if done % 500 == 0:
            print(f"texts fetched {done} (of {len(refs)})", flush=True)
    print("texts done", flush=True)


def main() -> None:
    """CLI."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=["topics", "texts"])
    parser.add_argument("--refs-file", type=Path)
    parser.add_argument("--rate", type=float, default=3.0)
    args = parser.parse_args()
    if args.stage == "topics":
        pull_topics(args.rate)
    else:
        pull_texts(args.refs_file, args.rate)


if __name__ == "__main__":
    main()
