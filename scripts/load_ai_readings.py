#!/usr/bin/env python3
"""Create the readings index and load readings sidecars (advanced Readers view).

The Studio's half-hourly loop is the normal writer of the readings index
(``genizah_ai_readings_v1``, mapping :data:`~src.backend.ai_readings.READINGS_INDEX_MAPPING`).
This script exists to create the index with the site's mapping and to load a
handful of sidecars by hand (for example the acceptance pages).  Each input is
one ``readings-v1`` sidecar JSON file, or a directory of them; the stored
document is :func:`~src.backend.ai_readings.readings_document` under the ``_id``
``<doc_id>__<image_index>__<vlm_model>``, so a later Studio push of the same page
overwrites it.

The default is a dry run that validates every file and prints a summary.
Nothing is written to Elasticsearch without ``--apply``.

Sidecars are only written into an index that has the site's mapping.  A
missing index is refused unless ``--create-index`` is given (a bulk write
would make Elasticsearch auto-create it with a dynamic mapping in which
``doc_id`` is analysed text, so every page lookup misses), and an existing
index whose ``doc_id`` is not a keyword is refused outright.  Both exit 2.

Usage (from repo root, using the backend venv)::

    PYTHONPATH=. .venv/bin/python scripts/load_ai_readings.py --create-index            # dry run
    PYTHONPATH=. .venv/bin/python scripts/load_ai_readings.py --create-index --apply    # empty index
    PYTHONPATH=. .venv/bin/python scripts/load_ai_readings.py \\
        --input data/ai_readings/samples --create-index --apply                        # load files

Writing to the default cluster (``ELASTICSEARCH_HOST``, the prod cluster) is
a production data change; run ``--apply`` deliberately.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from elasticsearch import Elasticsearch, helpers

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.load_ai_transcriptions import es_client  # noqa: E402
from src.backend.ai_readings import (  # noqa: E402
    DEFAULT_READINGS_INDEX,
    READINGS_INDEX_MAPPING,
    readings_document,
    readings_document_id,
    validate_sidecar,
)


def sidecar_paths(inputs: List[Path]) -> Iterator[Path]:
    """Every sidecar file named directly or found in a named directory.

    :param inputs: Files and/or directories.
    :return: ``*.json`` paths in sorted order per directory.
    :rtype: Iterator[Path]
    """
    for path in inputs:
        if path.is_dir():
            yield from sorted(path.glob("*.json"))
        else:
            yield path


def read_sidecars(paths: Iterator[Path]) -> Tuple[List[Dict], List[str]]:
    """Parse and validate sidecar files.

    :param paths: Sidecar files.
    :return: ``(valid sidecars, problems)``; a file with any problem is skipped.
    :rtype: Tuple[List[Dict], List[str]]
    """
    valid: List[Dict] = []
    problems: List[str] = []
    for path in paths:
        try:
            sidecar = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            problems.append(f"{path}: {exc}")
            continue
        issues = validate_sidecar(sidecar)
        if issues:
            problems.append(f"{path}: " + "; ".join(issues))
            continue
        valid.append(sidecar)
    return valid, problems


def ensure_index(es: Elasticsearch, index: str) -> bool:
    """Create the readings index with the site's mapping if it is missing.

    :param es: Client.
    :param index: Index name.
    :return: ``True`` if the index was created, ``False`` if it already existed.
    :rtype: bool
    """
    if es.indices.exists(index=index):
        return False
    es.indices.create(index=index, **READINGS_INDEX_MAPPING)
    return True


def mapping_problem(es: Elasticsearch, index: str) -> Optional[str]:
    """Why an existing index cannot take sidecars, if it cannot.

    An index that a bulk write auto-created has a dynamic mapping in which
    ``doc_id`` is analysed text, so the site's ``terms`` lookups never match.

    :param es: Client.
    :param index: Index (or alias) name.
    :return: A message when some backing index does not map ``doc_id`` as a
        keyword, else ``None``.
    :rtype: Optional[str]
    """
    response = es.indices.get_mapping(index=index)
    for name in response:
        properties = (response[name].get("mappings") or {}).get("properties") or {}
        doc_id_type = (properties.get("doc_id") or {}).get("type")
        if doc_id_type != "keyword":
            return (
                f"index {name} maps doc_id as {doc_id_type or 'nothing'}, not keyword: it was not created "
                "with the site's mapping (probably auto-created by an earlier write); recreate it with "
                "--create-index before loading"
            )
    return None


def main(argv: List[str] | None = None) -> int:
    """CLI entry point.

    :param argv: Arguments (defaults to ``sys.argv[1:]``).
    :return: Process exit code: 0 on success, 1 when files were skipped or
        writes failed, 2 when the index is missing or has the wrong mapping.
    :rtype: int
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", nargs="*", type=Path, default=[], help="Sidecar JSON files or directories")
    parser.add_argument("--index", default=DEFAULT_READINGS_INDEX, help=f"Target index (default: {DEFAULT_READINGS_INDEX})")
    parser.add_argument("--create-index", action="store_true", help="Create the index if it is missing")
    parser.add_argument("--apply", action="store_true", help="Actually write to Elasticsearch")
    parser.add_argument("--es-url", default=None, help="Cluster URL override (default: the backend's cluster)")
    args = parser.parse_args(argv)

    sidecars, problems = read_sidecars(sidecar_paths(args.input))
    for problem in problems:
        print(f"SKIP {problem}")
    for sidecar in sidecars:
        document = readings_document(sidecar)
        print(
            f"OK   {readings_document_id(sidecar)}  readers={document['n_readers']} "
            f"policy={document['policy_action']} final={document['final_source']} "
            f"lines={len(sidecar['final']['lines'])}"
        )
    print(f"{len(sidecars)} valid, {len(problems)} skipped; index {args.index}")
    if not args.apply:
        print("Dry run: nothing written (pass --apply).")
        return 1 if problems else 0

    if not args.create_index and not sidecars:
        print("Nothing to write.")
        return 1 if problems else 0
    es = es_client(args.es_url)
    created = ensure_index(es, args.index) if args.create_index else False
    exists = created or bool(es.indices.exists(index=args.index))
    print(f"Index {args.index}: {'created' if created else 'exists' if exists else 'missing'}")
    if not sidecars:
        return 1 if problems else 0
    # Never let a bulk write auto-create the index (or write into one that was).
    if not exists:
        print(f"ERROR index {args.index} does not exist; pass --create-index to create it with the site's mapping")
        return 2
    problem = None if created else mapping_problem(es, args.index)
    if problem:
        print(f"ERROR {problem}")
        return 2
    actions = (
        {"_op_type": "index", "_index": args.index, "_id": readings_document_id(s), "_source": readings_document(s)}
        for s in sidecars
    )
    written, errors = helpers.bulk(es, actions, raise_on_error=False, refresh="wait_for")
    print(f"Wrote {written} sidecar(s); {len(errors)} error(s)")
    for error in errors:
        print(f"ERROR {error}")
    if errors:
        return 1
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
