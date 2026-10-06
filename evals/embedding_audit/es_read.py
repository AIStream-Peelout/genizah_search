"""Read-only access to the production Elasticsearch on the MBP (search/count/scroll only).

Credentials come from the sibling repo's ``.env`` (``ELASTIC_USER`` / ``ELASTIC_PASSWORD``), the same
ones its indexers use. This module never issues writes: only ``_search``, ``_count``, ``_mapping`` and
scroll continuation/clear.
"""

import os
import re
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import requests

ES_URL = os.environ.get("AUDIT_ES_URL", "http://isaacs-MacBook-Pro.local:9200")
HDA_ENV = Path.home() / "Documents/GitHub/historical-document-analysis/.env"


def _auth() -> tuple:
    """Read (user, password) from the HDA .env without printing them.

    :returns: Basic-auth tuple.
    :rtype: tuple
    """
    vals = {}
    for line in HDA_ENV.read_text().splitlines():
        m = re.match(r"\s*(ELASTIC_USER|ELASTIC_PASSWORD)\s*=\s*(.+?)\s*$", line)
        if m:
            vals[m.group(1)] = m.group(2).strip("'\"")
    return vals["ELASTIC_USER"], vals["ELASTIC_PASSWORD"]


SESSION = requests.Session()
SESSION.auth = _auth()


def search(index: str, body: dict, params: Optional[dict] = None) -> dict:
    """POST a search body (read-only).

    :param index: Index name.
    :param body: Query DSL.
    :param params: URL params (e.g. scroll).
    :returns: Parsed response.
    :rtype: dict
    """
    r = SESSION.post(f"{ES_URL}/{index}/_search", json=body, params=params or {}, timeout=120)
    r.raise_for_status()
    return r.json()


def scan(index: str, query: dict, source: List[str], size: int = 500) -> Iterator[dict]:
    """Scroll through all hits of a query (read-only).

    :param index: Index name.
    :param query: Query clause.
    :param source: ``_source`` fields to return.
    :param size: Page size.
    :yields: Hit dicts.
    """
    resp = search(index, {"query": query, "_source": source, "size": size, "sort": ["_doc"]}, {"scroll": "5m"})
    sid = resp.get("_scroll_id")
    try:
        while resp["hits"]["hits"]:
            yield from resp["hits"]["hits"]
            r = SESSION.post(f"{ES_URL}/_search/scroll", json={"scroll": "5m", "scroll_id": sid}, timeout=120)
            r.raise_for_status()
            resp = r.json()
            sid = resp.get("_scroll_id", sid)
    finally:
        if sid:
            SESSION.delete(f"{ES_URL}/_search/scroll", json={"scroll_id": sid}, timeout=30)


def get_docs(index: str, ids: List[str], source: List[str]) -> Dict[str, dict]:
    """Fetch documents by id via an ``ids`` query (read-only).

    :param index: Index name.
    :param ids: Document ids.
    :param source: Fields to return.
    :returns: Mapping id -> _source.
    :rtype: Dict[str, dict]
    """
    out = {}
    for start in range(0, len(ids), 500):
        chunk = ids[start:start + 500]
        resp = search(index, {"query": {"ids": {"values": chunk}}, "_source": source, "size": len(chunk)})
        out.update({h["_id"]: h["_source"] for h in resp["hits"]["hits"]})
    return out
