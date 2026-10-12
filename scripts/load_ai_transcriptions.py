#!/usr/bin/env python3
"""Validate and load offline AI reads (pipeline sidecars) into the side index.

Input is JSONL, one record per line: the envelope naming the document and
image plus the pipeline's ``ai_read`` sidecar verbatim::

    {"doc_id": "...", "source_index": "genizah_merged_v5", "image_index": 0,
     "image_url": "https://...", "image_width": 2800, "image_height": 1785,
     "image_sha256": "...",
     "ai_read": {"vlm_model": "...", "vlm_revision": "...", "htr_model": "...",
                 "rule_version": "lines-v1-20260908", "decoded_at": "...",
                 "parsed": true, "n_lines": 16, "n_agreed": 3,
                 "lines": [{"index": 0, "text": "...", "bbox": [..], "agreement": 0.86,
                            "status": "agreed", "htr_text": "...", "htr_fragments": [[..]]}]}}

``status`` and ``index`` are filled in when missing and validated against the
served rule when present, so the pipeline and the site can never disagree on
what "agreed" means.  ``surfaced`` (the visitor-visibility flag),
``text_all`` / ``text_agreed`` (the joined text the search endpoint matches)
and ``read_key`` are derived here; ``published`` is the maintainer's switch.

The default target is the searchable v2 index (``SEARCH_INDEX``,
``genizah_ai_transcriptions_v2``), whose strict mapping differs from v1's;
``--index`` overrides it.  The backend keeps serving v1 until
``AI_TRANSCRIPTIONS_INDEX`` is switched (docs/ai_transcription_search.md).

The default is a dry run that validates every record and prints a summary.
Nothing is written to Elasticsearch without ``--apply``.

Usage (from repo root, using the backend venv)::

    PYTHONPATH=. .venv/bin/python scripts/load_ai_transcriptions.py \\
        --input data/ai_transcriptions/pilot.jsonl            # validate only
    PYTHONPATH=. .venv/bin/python scripts/load_ai_transcriptions.py \\
        --input data/ai_transcriptions/pilot.jsonl --apply    # create index + upsert
    PYTHONPATH=. .venv/bin/python scripts/load_ai_transcriptions.py \\
        --input pilot.jsonl --apply --es-url http://localhost:9200  # local ES only

Writing to the default cluster (``ELASTICSEARCH_HOST``, the prod cluster) is
a production data change; run ``--apply`` deliberately.  ``--es-url`` points
the load at another cluster, e.g. the local experimentation container.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

import dotenv
from elasticsearch import Elasticsearch, helpers

from src.backend.ai_transcriptions import (
    INDEX_MAPPING,
    RULE_VERSION,
    SEARCH_INDEX,
    AiTranscriptionRecord,
    build_lines,
    surfacing_thresholds,
)


def es_client(url: str | None = None, user: str | None = None, password: str | None = None) -> Elasticsearch:
    """Elasticsearch client using the same environment as the backend.

    :param url: Cluster URL override (default: ``https://$ELASTICSEARCH_HOST:$ELASTICSEARCH_PORT``).
    :param user: Basic-auth user override (default: ``ELASTICSEARCH_USER``).
    :param password: Basic-auth password override (default: ``ELASTICSEARCH_PASSWORD``).
    :return: Connected client.
    :rtype: Elasticsearch
    """
    dotenv.load_dotenv("src/backend/.env")
    dotenv.load_dotenv(".env")
    host = os.getenv("ELASTICSEARCH_HOST", "elastic.cairogenizah.ai")
    port = os.getenv("ELASTICSEARCH_PORT", "443")
    scheme = os.getenv("ELASTICSEARCH_SCHEME", "https")
    # The cluster is shared with prod search, the pipeline's writes and the
    # collection-hierarchy scan, so bursts of contention make the default ~10s
    # request timeout fire mid-bulk. A generous timeout plus transport-level
    # retries keeps this idempotent upsert from failing whole batches.
    return Elasticsearch(
        [url or f"{scheme}://{host}:{port}"],
        basic_auth=(
            user or os.getenv("ELASTICSEARCH_USER", "cairo_user"),
            password or os.getenv("ELASTICSEARCH_PASSWORD"),
        ),
        verify_certs=False,
        request_timeout=60,
        retry_on_timeout=True,
        max_retries=3,
    )


def read_records(path: Path) -> Iterator[Tuple[int, Dict]]:
    """Yield ``(line_number, raw_record)`` from a JSONL file.

    :param path: Input file.
    :return: Iterator over numbered raw records; blank lines are skipped.
    :rtype: Iterator[Tuple[int, Dict]]
    """
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if line.strip():
                yield number, json.loads(line)


def to_record(raw: Dict) -> AiTranscriptionRecord:
    """Validate one raw record, deriving line status and the surfacing flag.

    :param raw: Envelope + ``ai_read`` sidecar from the pipeline.
    :return: Validated record.
    :rtype: AiTranscriptionRecord
    """
    payload = dict(raw)
    sidecar = dict(raw["ai_read"])
    sidecar["lines"] = build_lines(sidecar.get("lines", []))
    payload["ai_read"] = sidecar
    payload.pop("text", None)
    payload.pop("surfaced", None)
    return AiTranscriptionRecord(**payload)


def bulk_actions(records: Iterable[AiTranscriptionRecord], index: str) -> Iterator[Dict]:
    """Bulk upsert actions for the given records.

    :param records: Validated records.
    :param index: Target index.
    :return: Iterator of ``index`` actions keyed by the deterministic id.
    :rtype: Iterator[Dict]
    """
    for record in records:
        yield {
            "_op_type": "index",
            "_index": index,
            "_id": record.es_id(),
            "_source": json.loads(record.model_dump_json()),
        }


def ensure_index(es: Elasticsearch, index: str) -> bool:
    """Create the side index with the canonical mapping if it is missing.

    :param es: Client.
    :param index: Index name.
    :return: ``True`` if the index was created, ``False`` if it already existed.
    :rtype: bool
    """
    if es.indices.exists(index=index):
        return False
    es.indices.create(index=index, **INDEX_MAPPING)
    return True


def main(argv: List[str] | None = None) -> int:
    """CLI entry point.

    :param argv: Arguments (defaults to ``sys.argv[1:]``).
    :return: Process exit code.
    :rtype: int
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path, help="JSONL of envelope + ai_read records")
    parser.add_argument("--index", default=None, help=f"Target index (default: {SEARCH_INDEX})")
    parser.add_argument("--apply", action="store_true", help="Actually write to Elasticsearch")
    parser.add_argument(
        "--es-url", default=None,
        help="Cluster URL override, e.g. http://localhost:9200 for the local experimentation ES "
             "(default: the backend's ELASTICSEARCH_HOST, i.e. the prod cluster)",
    )
    parser.add_argument("--es-user", default=None, help="Basic-auth user for --es-url (default: ELASTICSEARCH_USER)")
    parser.add_argument("--es-password", default=None, help="Basic-auth password for --es-url (default: ELASTICSEARCH_PASSWORD)")
    parser.add_argument("--unpublished", action="store_true", help="Load with published=false (hidden from the site)")
    args = parser.parse_args(argv)

    index = args.index or SEARCH_INDEX
    records: List[AiTranscriptionRecord] = []
    failures = 0
    for number, raw in read_records(args.input):
        try:
            record = to_record(raw)
        except (KeyError, ValueError) as exc:  # pydantic ValidationError is a ValueError
            failures += 1
            print(f"line {number}: INVALID: {exc}", file=sys.stderr)
            continue
        if args.unpublished:
            record.published = False
        records.append(record)

    n_lines = sum(r.ai_read.n_lines for r in records)
    n_agreed = sum(r.ai_read.n_agreed for r in records)
    n_surfaced = sum(1 for r in records if r.surfaced)
    n_text_agreed = sum(1 for r in records if r.text_agreed)
    n_searchable = sum(1 for r in records if r.surfaced and r.published and r.text_agreed)
    print(f"index: {index}")
    print(f"rule: {RULE_VERSION}; surfacing thresholds: {surfacing_thresholds()}")
    print(f"records: {len(records)} valid, {failures} invalid; {n_surfaced} pass the surfacing rule")
    print(f"lines: {n_lines} total, {n_agreed} agreed")
    print(f"text_agreed non-empty: {n_text_agreed} records ({n_searchable} surfaced+published, i.e. searchable by default)")

    if failures:
        return 1
    if not args.apply:
        print("dry run: nothing written (pass --apply to load)")
        return 0

    es = es_client(args.es_url, args.es_user, args.es_password)
    created = ensure_index(es, index)
    print(f"index {'created' if created else 'exists'}: {index}")
    # request_timeout here overrides the client default for the (potentially
    # large) bulk call specifically; the load is idempotent, so a rare failure
    # is simply retried on the watcher's next pass.
    ok, errors = helpers.bulk(
        es, bulk_actions(records, index), stats_only=False, raise_on_error=False,
        request_timeout=120,
    )
    print(f"indexed {ok} records, {len(errors)} errors")
    for error in errors[:10]:
        print(json.dumps(error), file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
