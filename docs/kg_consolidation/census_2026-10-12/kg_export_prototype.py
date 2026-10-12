"""SAMPLE-ONLY prototype of a no-downtime logical export of the production KG.

Reads through the read-only helper (every query runs in ``execute_read``) and writes a
small sample (default 500 nodes + 500 relationships) in the proposed backup format:

* ``manifest.json``  - export metadata, per-label / per-type counts, constraint and index DDL
* ``nodes.jsonl.gz`` - one JSON object per node: ``{"eid", "labels", "props"}``
* ``rels.jsonl.gz``  - one JSON object per relationship: ``{"eid", "type", "start", "end", "props"}``

It also times reads of different sizes to separate per-query overhead from per-row cost,
and measures average serialized bytes per record on a stratified random sample (kept in
memory only) to estimate the size of a full export. It never exports the full graph.
"""

import gzip
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

SCRATCH = "/private/tmp/claude-501/-Users-isaac1-Documents-GitHub-terraform-gcp-search/3dfa2281-deee-4175-b3bf-5b75eb25fcc6/scratchpad"
sys.path.insert(0, SCRATCH)
from neo4j_ro import run  # noqa: E402

OUT = Path(SCRATCH) / "kg_census" / "backup" / "sample_export"

NODE_Q = "MATCH (n) RETURN elementId(n) AS eid, labels(n) AS labels, properties(n) AS props LIMIT $limit"
REL_Q = (
    "MATCH (a)-[r]->(b) RETURN elementId(r) AS eid, type(r) AS type, elementId(a) AS start, "
    "elementId(b) AS end, properties(r) AS props LIMIT $limit"
)


def line(record: Dict[str, Any]) -> bytes:
    """Serialize one export record as a compact JSON line.

    :param record: Node or relationship record.
    :returns: UTF-8 bytes ending in a newline.
    :rtype: bytes
    """
    return (json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n").encode("utf-8")


def timed(query: str, limit: int, repeats: int = 3) -> Tuple[float, List[Dict[str, Any]]]:
    """Run a bounded read query several times and return the median wall time.

    :param query: Cypher with a ``$limit`` parameter.
    :param limit: Row limit.
    :param repeats: Number of runs.
    :returns: (median seconds, rows of the last run).
    :rtype: Tuple[float, List[Dict[str, Any]]]
    """
    times, rows = [], []
    for _ in range(repeats):
        t0 = time.perf_counter()
        rows = run(query, {"limit": limit})
        times.append(time.perf_counter() - t0)
    return statistics.median(times), rows


def stratified_bytes() -> Dict[str, Any]:
    """Estimate full-export size from a random sample per label and per relationship type.

    :returns: Per-stratum sample size, mean bytes and estimated totals (uncompressed and gzip).
    :rtype: Dict[str, Any]
    """
    labels = run("CALL db.labels() YIELD label CALL (label) { MATCH (n) WHERE label IN labels(n) RETURN count(n) AS c } RETURN label, c")
    types = run("CALL db.relationshipTypes() YIELD relationshipType AS t CALL (t) { MATCH ()-[r]->() WHERE type(r) = t RETURN count(r) AS c } RETURN t, c")
    strata, node_blob, rel_blob = [], bytearray(), bytearray()
    est_nodes = est_rels = 0.0
    for row in labels:
        sample = run(
            f"MATCH (n:`{row['label']}`) WITH n, rand() AS x ORDER BY x LIMIT 300 "
            "RETURN elementId(n) AS eid, labels(n) AS labels, properties(n) AS props"
        )
        sizes = [len(line(r)) for r in sample]
        for r in sample:
            node_blob += line(r)
        mean = statistics.mean(sizes)
        est_nodes += mean * row["c"]
        strata.append({"kind": "node", "name": row["label"], "count": row["c"], "sampled": len(sample), "mean_bytes": round(mean, 1)})
    for row in types:
        sample = run(
            f"MATCH (a)-[r:`{row['t']}`]->(b) WITH a, r, b, rand() AS x ORDER BY x LIMIT 200 "
            "RETURN elementId(r) AS eid, type(r) AS type, elementId(a) AS start, elementId(b) AS end, properties(r) AS props"
        )
        sizes = [len(line(r)) for r in sample]
        for r in sample:
            rel_blob += line(r)
        mean = statistics.mean(sizes)
        est_rels += mean * row["c"]
        strata.append({"kind": "rel", "name": row["t"], "count": row["c"], "sampled": len(sample), "mean_bytes": round(mean, 1)})
    node_ratio = len(gzip.compress(bytes(node_blob), 6)) / len(node_blob)
    rel_ratio = len(gzip.compress(bytes(rel_blob), 6)) / len(rel_blob)
    return {
        "strata": strata,
        "est_nodes_jsonl_bytes": round(est_nodes),
        "est_rels_jsonl_bytes": round(est_rels),
        "gzip6_ratio_nodes": round(node_ratio, 3),
        "gzip6_ratio_rels": round(rel_ratio, 3),
        "est_nodes_gz_bytes": round(est_nodes * node_ratio),
        "est_rels_gz_bytes": round(est_rels * rel_ratio),
    }


def main() -> None:
    """Write the sample export, time reads, and save a measurements file.

    :returns: None
    :rtype: None
    """
    OUT.mkdir(parents=True, exist_ok=True)
    m: Dict[str, Any] = {"exported_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "sample_only": True}
    m["neo4j"] = run("CALL dbms.components() YIELD name, versions, edition RETURN name, versions, edition")
    m["counts_before"] = {
        "nodes": run("MATCH (n) RETURN count(n) AS c")[0]["c"],
        "rels": run("MATCH ()-[r]->() RETURN count(r) AS c")[0]["c"],
    }
    m["constraints"] = [r["createStatement"] for r in run("SHOW CONSTRAINTS YIELD createStatement RETURN createStatement")]
    m["indexes"] = [
        r["createStatement"]
        for r in run("SHOW INDEXES YIELD createStatement, owningConstraint WHERE owningConstraint IS NULL RETURN createStatement")
    ]

    timings: Dict[str, Any] = {}
    for kind, query in (("nodes", NODE_Q), ("rels", REL_Q)):
        for limit in (500, 5000):
            secs, rows = timed(query, limit)
            timings[f"{kind}_{limit}"] = {"median_s": round(secs, 4), "rows": len(rows)}
            if limit == 500:
                t0 = time.perf_counter()
                with gzip.open(OUT / f"{kind}.jsonl.gz", "wb", compresslevel=6) as fh:
                    for r in rows:
                        fh.write(line(r))
                timings[f"{kind}_500_write_s"] = round(time.perf_counter() - t0, 4)
                timings[f"{kind}_500_gz_bytes"] = (OUT / f"{kind}.jsonl.gz").stat().st_size
                timings[f"{kind}_500_raw_bytes"] = sum(len(line(r)) for r in rows)
    for kind in ("nodes", "rels"):
        small, big = timings[f"{kind}_500"]["median_s"], timings[f"{kind}_5000"]["median_s"]
        per_row = (big - small) / 4500
        timings[f"{kind}_per_row_ms"] = round(per_row * 1000, 4)
        timings[f"{kind}_fixed_overhead_s"] = round(small - 500 * per_row, 4)
    m["timings"] = timings
    m["size_estimate"] = stratified_bytes()
    m["counts_after"] = {
        "nodes": run("MATCH (n) RETURN count(n) AS c")[0]["c"],
        "rels": run("MATCH ()-[r]->() RETURN count(r) AS c")[0]["c"],
    }
    (OUT / "manifest.json").write_text(json.dumps(m, ensure_ascii=False, indent=1, default=str))
    print(json.dumps({k: v for k, v in m.items() if k not in ("constraints", "indexes", "size_estimate")}, indent=1, default=str))
    se = dict(m["size_estimate"])
    se.pop("strata")
    print(json.dumps(se, indent=1))


if __name__ == "__main__":
    main()
