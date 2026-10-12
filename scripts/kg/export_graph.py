"""No-downtime logical export of the knowledge graph (read-only).

Writes every node and relationship, with labels, types and properties, through the
Python driver in read transactions, so the database keeps serving. Output directory::

    manifest.json     counts before/after, constraints, indexes, file sha256s
    nodes.jsonl.gz    {"eid", "labels", "props"} per node
    rels.jsonl.gz     {"eid", "type", "start", "end", "props"} per relationship

The export is read-committed, not a snapshot: pause any importer writing to the graph
while it runs. The manifest's before/after counts show whether anything changed.
It does not carry users or roles. Restore outline:
``docs/kg_consolidation/census_2026-10-12/kg_restore_outline.py``.

Usage (from the repo root)::

    python scripts/kg/export_graph.py --out backups/neo4j/<UTC stamp>

Credentials come from ``NEO4J_AUTH`` (``user/password``) in the environment or the
repo-root ``.env``; they are never printed.
"""

import argparse
import datetime as dt
import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

from neo4j import GraphDatabase, READ_ACCESS

REPO_ROOT = Path(__file__).resolve().parents[2]
NODE_QUERY = "MATCH (n) RETURN elementId(n) AS eid, labels(n) AS labels, properties(n) AS props"
REL_QUERY = (
    "MATCH (a)-[r]->(b) RETURN elementId(r) AS eid, type(r) AS type, "
    "elementId(a) AS start, elementId(b) AS end, properties(r) AS props"
)


def neo4j_auth() -> Tuple[str, str]:
    """Return the (user, password) pair from NEO4J_AUTH without printing it.

    :returns: Basic-auth credentials.
    :rtype: Tuple[str, str]
    :raises SystemExit: When NEO4J_AUTH is set nowhere.
    """
    value = os.environ.get("NEO4J_AUTH", "")
    if not value:
        env_file = REPO_ROOT / ".env"
        for line in env_file.read_text().splitlines() if env_file.exists() else []:
            if line.startswith("NEO4J_AUTH="):
                value = line.split("=", 1)[1].strip().strip('"').strip("'")
    if "/" not in value:
        raise SystemExit("NEO4J_AUTH (user/password) not found in the environment or .env")
    user, _, password = value.partition("/")
    return user, password


def counts(session: Any) -> Dict[str, int]:
    """Count nodes and relationships.

    :param session: A read session.
    :returns: ``{"nodes": n, "relationships": m}``.
    :rtype: Dict[str, int]
    """
    nodes = session.execute_read(lambda tx: tx.run("MATCH (n) RETURN count(n) AS c").single()["c"])
    rels = session.execute_read(lambda tx: tx.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"])
    return {"nodes": nodes, "relationships": rels}


def write_jsonl(session: Any, query: str, path: Path) -> Tuple[int, str]:
    """Stream one query's rows into a gzipped JSONL file.

    :param session: A read session.
    :param query: Cypher returning one row per record to export.
    :param path: Output file.
    :returns: Rows written and the file's sha256.
    :rtype: Tuple[int, str]
    """

    def dump(tx: Any) -> int:
        """Write every row inside the read transaction.

        :param tx: Read transaction.
        :returns: Rows written.
        :rtype: int
        """
        written = 0
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            for record in tx.run(query):
                handle.write(json.dumps(record.data(), ensure_ascii=False, default=str) + "\n")
                written += 1
        return written

    rows = session.execute_read(dump)
    return rows, hashlib.sha256(path.read_bytes()).hexdigest()


def schema(session: Any) -> Dict[str, Iterable[Dict[str, Any]]]:
    """Read constraints and indexes so a restore can recreate them.

    :param session: A read session.
    :returns: ``{"constraints": [...], "indexes": [...]}``.
    :rtype: Dict[str, Iterable[Dict[str, Any]]]
    """
    constraints = session.execute_read(lambda tx: [r.data() for r in tx.run("SHOW CONSTRAINTS YIELD *")])
    indexes = session.execute_read(lambda tx: [r.data() for r in tx.run("SHOW INDEXES YIELD *")])
    return {"constraints": constraints, "indexes": indexes}


def main() -> None:
    """Export the graph to ``--out``.

    :returns: None
    :rtype: None
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--uri", default=os.environ.get("NEO4J_EXPORT_URI", "bolt://localhost:7681"))
    parser.add_argument("--database", default="neo4j")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=False)
    driver = GraphDatabase.driver(args.uri, auth=neo4j_auth())
    with driver.session(database=args.database, default_access_mode=READ_ACCESS) as session:
        started = dt.datetime.now(dt.timezone.utc)
        before = counts(session)
        node_rows, node_sha = write_jsonl(session, NODE_QUERY, args.out / "nodes.jsonl.gz")
        rel_rows, rel_sha = write_jsonl(session, REL_QUERY, args.out / "rels.jsonl.gz")
        after = counts(session)
        manifest = {
            "started_utc": started.isoformat(),
            "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "uri": args.uri,
            "database": args.database,
            "counts_before": before,
            "counts_after": after,
            "rows_written": {"nodes": node_rows, "relationships": rel_rows},
            "consistent": before == after == {"nodes": node_rows, "relationships": rel_rows},
            "sha256": {"nodes.jsonl.gz": node_sha, "rels.jsonl.gz": rel_sha},
            **schema(session),
        }
    driver.close()
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1, default=str))
    print(json.dumps({key: manifest[key] for key in ("counts_before", "counts_after", "rows_written", "consistent")}))


if __name__ == "__main__":
    main()
