"""OUTLINE (not run, not tested against any server) of the restore for the logical KG export.

Restores ``manifest.json`` + ``nodes.jsonl.gz`` + ``rels.jsonl.gz`` (written by the export,
see kg_export_prototype.py for the format) into an EMPTY Neo4j 2026.04 Community database,
without APOC. It is meant for a throwaway container/volume (e.g. a ``neo4j:2026.04.0``
container on 127.0.0.1:7691 with its own volume), never the production volume; replacing
production would be: restore into a scratch volume, verify, then swap volumes with the
user's approval.

Design:
1. Refuse unless the target has 0 nodes and 0 relationships.
2. Create the uniqueness constraints and non-lookup indexes from the manifest's DDL first
   (a fresh DB already has its own two LOOKUP indexes, so LOOKUP DDL is skipped). The
   constraints double as an integrity check: a duplicate key aborts the restore.
3. Nodes: group by label combination (labels cannot be parameters without APOC; each label
   is validated against ``^[A-Za-z_][A-Za-z0-9_]*$`` and backtick-quoted), then batched
   ``UNWIND $rows AS row CREATE (n:`L`) SET n = row.props RETURN row.eid AS old, elementId(n) AS new``.
   The old->new elementId map is kept in Python (77k entries, a few MB).
4. Relationships: group by type (same validation), translate endpoints through the id map
   (abort if any endpoint is missing, i.e. the export was inconsistent), then batched
   ``UNWIND $rows AS row MATCH (a) WHERE elementId(a) = row.s MATCH (b) WHERE elementId(b) = row.e
   CREATE (a)-[r:`T`]->(b) SET r = row.props``. On the production server EXPLAIN shows two
   NodeByElementIdSeek operators for this pattern, so each row is two O(1) seeks.
5. Verify: total and per-label / per-type counts equal the manifest; per-label property-key
   counts match; spot-check N random nodes by their unique key (Scholar.name,
   Fragment.canonical_shelfmark, ...) for an identical property map.

All property values in the current graph are STRING, LIST<STRING>, BOOLEAN, INTEGER or FLOAT
(property_type_survey.json), so plain JSON round-trips them losslessly (json keeps int vs float).
New elementIds differ from the exported ones; anything that stored old elementIds (e.g. a
change log) must key on business keys instead.
"""

import gzip
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List

from neo4j import GraphDatabase

NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
BATCH = 5000


def q(name: str) -> str:
    """Validate and backtick-quote a label or relationship type.

    :param name: Label or type from the manifest/export.
    :returns: The quoted identifier.
    :rtype: str
    """
    if not NAME_RE.match(name):
        raise ValueError(f"unsafe identifier: {name!r}")
    return f"`{name}`"


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    """Yield records from a gzip JSONL file.

    :param path: File path.
    :returns: Iterator of decoded records.
    :rtype: Iterator[Dict[str, Any]]
    """
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for raw in fh:
            yield json.loads(raw)


def batches(rows: List[Dict[str, Any]], size: int = BATCH) -> Iterator[List[Dict[str, Any]]]:
    """Split a list into fixed-size chunks.

    :param rows: Rows to split.
    :param size: Chunk size.
    :returns: Iterator of chunks.
    :rtype: Iterator[List[Dict[str, Any]]]
    """
    for i in range(0, len(rows), size):
        yield rows[i : i + size]


def restore(export_dir: Path, uri: str, auth: tuple, database: str = "neo4j") -> Dict[str, Any]:
    """Restore a logical export into an empty database and return verification counts.

    :param export_dir: Directory holding manifest.json, nodes.jsonl.gz, rels.jsonl.gz.
    :param uri: Bolt URI of the EMPTY scratch target.
    :param auth: (user, password) of the scratch target.
    :param database: Target database name.
    :returns: Restored counts and the manifest's expected counts.
    :rtype: Dict[str, Any]
    """
    manifest = json.loads((export_dir / "manifest.json").read_text())
    driver = GraphDatabase.driver(uri, auth=auth)
    with driver.session(database=database) as s:
        n = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        r = s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        if n or r:
            raise SystemExit(f"target not empty ({n} nodes, {r} rels); refusing")

        for ddl in manifest["constraints"] + [d for d in manifest["indexes"] if "LOOKUP" not in d]:
            s.run(ddl.replace("CREATE CONSTRAINT", "CREATE CONSTRAINT IF NOT EXISTS", 1)
                  if ddl.startswith("CREATE CONSTRAINT") else ddl.replace(" INDEX ", " INDEX IF NOT EXISTS ", 1))
        s.run("CALL db.awaitIndexes(300)")

        by_labels: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
        for rec in read_jsonl(export_dir / "nodes.jsonl.gz"):
            by_labels[tuple(rec["labels"])].append({"eid": rec["eid"], "props": rec["props"]})
        id_map: Dict[str, str] = {}
        for labels, rows in by_labels.items():
            label_expr = ":" + ":".join(q(l) for l in labels) if labels else ""
            cypher = f"UNWIND $rows AS row CREATE (n{label_expr}) SET n = row.props RETURN row.eid AS old, elementId(n) AS new"
            for chunk in batches(rows):
                for rec in s.execute_write(lambda tx: list(tx.run(cypher, rows=chunk))):
                    id_map[rec["old"]] = rec["new"]

        by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for rec in read_jsonl(export_dir / "rels.jsonl.gz"):
            if rec["start"] not in id_map or rec["end"] not in id_map:
                raise SystemExit(f"dangling relationship {rec['eid']}; export inconsistent")
            by_type[rec["type"]].append({"s": id_map[rec["start"]], "e": id_map[rec["end"]], "props": rec["props"]})
        for rtype, rows in by_type.items():
            cypher = (
                "UNWIND $rows AS row MATCH (a) WHERE elementId(a) = row.s MATCH (b) WHERE elementId(b) = row.e "
                f"CREATE (a)-[r:{q(rtype)}]->(b) SET r = row.props"
            )
            for chunk in batches(rows):
                s.execute_write(lambda tx: tx.run(cypher, rows=chunk).consume())

        got = {
            "nodes": s.run("MATCH (n) RETURN count(n) AS c").single()["c"],
            "rels": s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"],
        }
    driver.close()
    return {"restored": got, "expected": manifest.get("counts_after") or manifest.get("counts_before")}
