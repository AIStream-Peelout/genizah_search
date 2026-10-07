#!/usr/bin/env python3
"""Move the Elasticsearch indexes from the Mac Studio to the MBP.

Two transports, both leaving the source cluster's data untouched:

* **snapshot / restore** (default path, via the NAS): ``--snapshot`` on the Studio
  writes every non-system index into the repository at
  ``/usr/share/elasticsearch/backup`` (the ``./backups/elasticsearch`` bind mount,
  needs ``path.repo`` active); ``--restore-snapshot latest`` on the MBP restores it
  from the same bind mount after the directory was copied over.
* **reindex-from-remote** (LAN fallback): ``--mirror`` pulls index by index from
  ``--source`` into ``--dest``; the destination must whitelist the source.

Standard library only, so it runs on a fresh macOS python3 with no venv.
Credentials are read from the repo's env files and never printed:

* source (Studio cluster): ``ELASTICSEARCH_USER`` / ``ELASTICSEARCH_PASSWORD`` in
  ``src/backend/.env`` (the backend's superuser; also valid on ``localhost:9200``)
* destination (MBP cluster): ``elastic`` / ``LOCAL_ELASTIC_PASSWORD`` in the root
  ``.env`` (the bootstrap password set by docker-compose.mbp.yml)

Examples::

    es_migrate.py --snapshot                                   # Studio
    es_migrate.py --wait --restore-snapshot latest --setup-users --kibana-token   # MBP
    es_migrate.py --source http://192.168.8.120:9200 --mirror --verify           # LAN fallback
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parents[2]
REPO_NAME = "migration"
REPO_PATH = "/usr/share/elasticsearch/backup"
NON_SYSTEM = "*,-.*"

# Serving indexes (SHARED_RUNTIME.md, docs/ai_transcription_search.md) go first
# in the reindex fallback so the site can be checked while the rest streams.
PRIORITY_INDEXES = [
    "genizah_merged_v5",
    "bibliography_text_only_0.7",
    "genizah_ai_transcriptions_v1",
    "genizah_ai_transcriptions_v2",
    "genizah_missing_fragments_v1",
]

# Settings that belong to the source cluster, not to the index definition.
NON_PORTABLE_SETTINGS = ("uuid", "version", "creation_date", "provided_name", "history", "routing")


def read_env(path: Path) -> Dict[str, str]:
    """Parse a dotenv file into a dict without exporting anything.

    :param path: Path to a ``KEY=value`` file; ``#`` lines are ignored.
    :return: Mapping of keys to raw values (surrounding quotes stripped).
    """
    values: Dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip("'\"")
    return values


class EsClient:
    """Minimal Elasticsearch HTTP client over urllib with basic auth."""

    def __init__(self, base_url: str, user: str, password: str) -> None:
        """
        :param base_url: Cluster URL including scheme, e.g. ``http://localhost:9200``.
        :param user: Basic-auth user name.
        :param password: Basic-auth password.
        """
        self.base_url = base_url.rstrip("/")
        self.user = user
        self.password = password
        token = base64.b64encode(f"{user}:{password}".encode()).decode()
        self._auth = f"Basic {token}"

    def request(
        self,
        method: str,
        path: str,
        body: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, str]] = None,
        timeout: int = 120,
    ) -> Any:
        """Send one request and decode the JSON response.

        :param method: HTTP verb.
        :param path: Path starting with ``/``.
        :param body: JSON body, if any.
        :param params: Query-string parameters.
        :param timeout: Socket timeout in seconds.
        :return: Decoded JSON response (``{}`` for an empty body).
        :raises urllib.error.HTTPError: On any non-2xx status.
        """
        url = self.base_url + path
        if params:
            url += "?" + urllib.parse.urlencode(params)
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Authorization", self._auth)
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
        return json.loads(raw) if raw else {}

    def exists(self, path: str) -> bool:
        """Return True when ``HEAD path`` succeeds.

        :param path: Path starting with ``/``.
        """
        try:
            self.request("HEAD", path)
            return True
        except urllib.error.HTTPError as err:
            if err.code == 404:
                return False
            raise

    def doc_count(self, index: str) -> int:
        """Refresh the index and return its document count.

        :param index: Index name.
        """
        self.request("POST", f"/{index}/_refresh")
        return int(self.request("GET", f"/{index}/_count")["count"])

    def user_indexes(self) -> List[str]:
        """Return every non-system index name, sorted."""
        rows = self.request("GET", "/_cat/indices", params={"h": "index", "format": "json"})
        return sorted(row["index"] for row in rows if not row["index"].startswith("."))


def wait_for_cluster(client: EsClient, timeout: int = 300) -> None:
    """Block until the cluster answers ``/_cluster/health`` with yellow or green.

    :param client: Cluster client.
    :param timeout: Seconds to wait before giving up.
    :raises SystemExit: When the cluster is not reachable in time.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            health = client.request("GET", "/_cluster/health", params={"wait_for_status": "yellow", "timeout": "10s"})
            print(f"cluster {health['cluster_name']} is {health['status']}")
            return
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(5)
    sys.exit(f"Elasticsearch at {client.base_url} did not come up within {timeout}s")


# --- snapshot / restore -------------------------------------------------------------

def ensure_repository(client: EsClient) -> None:
    """Register the filesystem snapshot repository, checking ``path.repo`` first.

    :param client: Cluster client.
    :raises SystemExit: When the node has no ``path.repo`` (container must be recreated).
    """
    settings = client.request("GET", "/_nodes/settings", params={"filter_path": "nodes.*.settings.path.repo"})
    if not settings:
        sys.exit(
            "path.repo is not active on this node. docker-compose.yml sets it, but the\n"
            "elasticsearch container must be recreated for it to apply:\n"
            "    docker compose up -d elasticsearch      # ~1 min search outage; needs approval on the Studio"
        )
    client.request(
        "PUT",
        f"/_snapshot/{REPO_NAME}",
        {"type": "fs", "settings": {"location": REPO_PATH, "compress": True}},
    )
    print(f"repository '{REPO_NAME}' -> {REPO_PATH}")


def create_snapshot(client: EsClient) -> str:
    """Snapshot every non-system index and wait for completion.

    :param client: Cluster client (source).
    :return: The snapshot name.
    """
    ensure_repository(client)
    name = time.strftime("migration-%Y%m%dt%H%M%S")
    client.request(
        "PUT",
        f"/_snapshot/{REPO_NAME}/{name}",
        {"indices": NON_SYSTEM, "include_global_state": False, "partial": False},
        params={"wait_for_completion": "false"},
    )
    print(f"snapshot {name} started ({len(client.user_indexes())} indexes)")
    while True:
        status = client.request("GET", f"/_snapshot/{REPO_NAME}/{name}/_status")["snapshots"][0]
        shards = status["shards_stats"]
        print(f"  {status['state']}: {shards['done']}/{shards['total']} shards", end="\r", flush=True)
        if status["state"] in ("SUCCESS", "FAILED", "PARTIAL"):
            print()
            break
        time.sleep(10)
    if status["state"] != "SUCCESS":
        sys.exit(f"snapshot ended in state {status['state']}: {json.dumps(status.get('failures', []))[:500]}")
    info = client.request("GET", f"/_snapshot/{REPO_NAME}/{name}")["snapshots"][0]
    print(f"snapshot {name} SUCCESS: {len(info['indices'])} indexes")
    return name


def restore_snapshot(client: EsClient, name: str) -> None:
    """Restore a snapshot's non-system indexes onto a single-node cluster.

    Indexes that already exist on the destination are left alone, so a
    re-run after an interruption only restores what is missing.

    :param client: Cluster client (destination).
    :param name: Snapshot name, or ``latest`` for the newest successful one.
    """
    ensure_repository(client)
    snapshots = client.request("GET", f"/_snapshot/{REPO_NAME}/_all")["snapshots"]
    good = [s for s in snapshots if s["state"] == "SUCCESS"]
    if not good:
        sys.exit(f"no successful snapshot in repository '{REPO_NAME}'")
    snap = good[-1] if name == "latest" else next((s for s in good if s["snapshot"] == name), None)
    if snap is None:
        sys.exit(f"snapshot {name} not found; have: {[s['snapshot'] for s in good]}")

    existing = set(client.user_indexes())
    todo = [i for i in snap["indices"] if not i.startswith(".") and i not in existing]
    skipped = [i for i in snap["indices"] if i in existing]
    if skipped:
        print(f"already present, skipping: {', '.join(sorted(skipped))}")
    if not todo:
        print("nothing to restore")
        return
    print(f"restoring {len(todo)} indexes from {snap['snapshot']}: {', '.join(todo)}")
    client.request(
        "POST",
        f"/_snapshot/{REPO_NAME}/{snap['snapshot']}/_restore",
        {
            "indices": ",".join(todo),
            "include_global_state": False,
            "index_settings": {"index.number_of_replicas": 0},
        },
        params={"wait_for_completion": "false"},
    )
    while True:
        done = recovered_indexes(client, todo)
        print(f"  {len(done)}/{len(todo)} indexes recovered", end="\r", flush=True)
        if len(done) == len(todo):
            print()
            break
        time.sleep(10)
    for index in todo:
        print(f"  {index:40} {client.doc_count(index):>10} docs")


def recovered_indexes(client: EsClient, indexes: List[str]) -> List[str]:
    """Return the subset of ``indexes`` whose shards are all assigned and started.

    Uses per-index cluster health rather than ``_recovery``: a restored index can
    sit with unassigned primaries (queued, not yet recovering) and answer 503 to
    ``_count`` even though nothing shows as actively recovering.

    :param client: Cluster client.
    :param indexes: Index names to check.
    :return: Names that are yellow/green with no initializing or unassigned shards.
    """
    try:
        health = client.request("GET", f"/_cluster/health/{','.join(indexes)}", params={"level": "indices", "timeout": "10s"})
    except urllib.error.HTTPError as err:
        if err.code in (404, 408):
            return []
        raise
    return [
        name
        for name, info in health.get("indices", {}).items()
        if info["status"] in ("yellow", "green") and info["initializing_shards"] == 0 and info["unassigned_shards"] == 0
    ]


# --- users / kibana -------------------------------------------------------------------

def ensure_user(client: EsClient, name: str, password: str, roles: List[str]) -> None:
    """Create or update a native-realm user (idempotent).

    :param client: Cluster client with security privileges.
    :param name: User name.
    :param password: Password to set.
    :param roles: Role names to grant.
    """
    client.request("PUT", f"/_security/user/{name}", {"password": password, "roles": roles})
    print(f"user {name} ready with roles {roles}")


def update_env_var(path: Path, key: str, value: str) -> None:
    """Set ``key=value`` in a dotenv file, replacing an existing line or appending.

    :param path: Dotenv file.
    :param key: Variable name.
    :param value: New value.
    """
    text = path.read_text()
    pattern = re.compile(rf"^{re.escape(key)}=.*$", re.MULTILINE)
    replacement = f"{key}={value}"
    text = pattern.sub(replacement, text) if pattern.search(text) else text.rstrip("\n") + f"\n{replacement}\n"
    path.write_text(text)


def create_kibana_token(client: EsClient, env_path: Path) -> None:
    """Mint a Kibana service-account token and store it in the root .env.

    :param client: Destination cluster client.
    :param env_path: Root ``.env`` whose ``ELASTICSEARCH_SERVICEACCOUNTTOKEN`` line is replaced.
    """
    name = "mbp"
    try:
        client.request("DELETE", f"/_security/service/elastic/kibana/credential/token/{name}")
    except urllib.error.HTTPError as err:
        if err.code != 404:
            raise
    token = client.request("POST", f"/_security/service/elastic/kibana/credential/token/{name}")["token"]["value"]
    update_env_var(env_path, "ELASTICSEARCH_SERVICEACCOUNTTOKEN", token)
    print(f"kibana service token '{name}' written to {env_path}")


# --- reindex-from-remote fallback -------------------------------------------------------

def discover_indexes(source: EsClient, exclude: List[str]) -> List[str]:
    """List every non-system index on the source, serving indexes first.

    :param source: Source cluster client.
    :param exclude: Index names to leave out.
    :return: Ordered index names (``PRIORITY_INDEXES`` first, the rest alphabetical).
    """
    names = set(source.user_indexes()) - set(exclude)
    ordered = [name for name in PRIORITY_INDEXES if name in names]
    return ordered + sorted(names - set(ordered))


def portable_definition(source: EsClient, index: str) -> Dict[str, Any]:
    """Fetch an index's settings + mappings with cluster-specific keys removed.

    :param source: Source cluster client.
    :param index: Index name.
    :return: Body suitable for ``PUT /<index>`` on the destination.
    """
    raw = source.request("GET", f"/{index}")[index]
    index_settings = {k: v for k, v in raw["settings"]["index"].items() if k not in NON_PORTABLE_SETTINGS}
    index_settings["number_of_replicas"] = "0"
    return {"settings": {"index": index_settings}, "mappings": raw["mappings"]}


def reindex_from_remote(source: EsClient, dest: EsClient, index: str, batch_size: int) -> None:
    """Create ``index`` on the destination if needed and pull its documents from the source.

    :param source: Source cluster client (its credentials are handed to the destination's reindex task).
    :param dest: Destination cluster client.
    :param index: Index name.
    :param batch_size: Documents per scroll batch (keep small; merged docs carry dense vectors).
    """
    expected = source.doc_count(index)
    if dest.exists(f"/{index}"):
        have = dest.doc_count(index)
        if have >= expected:
            print(f"{index}: destination already has {have}/{expected} docs, skipping")
            return
        print(f"{index}: destination has {have}/{expected} docs, deleting and re-pulling")
        dest.request("DELETE", f"/{index}")

    dest.request("PUT", f"/{index}", portable_definition(source, index))
    dest.request("PUT", f"/{index}/_settings", {"index": {"refresh_interval": "-1"}})
    task = dest.request(
        "POST",
        "/_reindex",
        {
            "source": {
                "remote": {
                    "host": source.base_url,
                    "username": source.user,
                    "password": source.password,
                    "socket_timeout": "2m",
                    "connect_timeout": "30s",
                },
                "index": index,
                "size": batch_size,
            },
            "dest": {"index": index},
        },
        params={"wait_for_completion": "false"},
    )["task"]
    print(f"{index}: reindex task {task}, {expected} docs to pull")
    while True:
        status = dest.request("GET", f"/_tasks/{task}")
        progress = status["task"]["status"]
        print(f"  {index}: {progress.get('created', 0)}/{progress.get('total', expected)}", end="\r", flush=True)
        if status.get("completed", False):
            print()
            failures = status.get("response", {}).get("failures", [])
            if failures:
                sys.exit(f"{index}: reindex reported failures: {json.dumps(failures[:3], indent=2)}")
            break
        time.sleep(10)
    dest.request("PUT", f"/{index}/_settings", {"index": {"refresh_interval": None}})
    print(f"{index}: {dest.doc_count(index)} docs on destination")


def verify(source: EsClient, dest: EsClient, indexes: List[str]) -> bool:
    """Compare document counts per index and print a table.

    :param source: Source cluster client.
    :param dest: Destination cluster client.
    :param indexes: Index names to compare.
    :return: True when every index matches.
    """
    ok = True
    print(f"{'index':40} {'source':>10} {'dest':>10}")
    for index in indexes:
        src = source.doc_count(index)
        dst = dest.doc_count(index) if dest.exists(f"/{index}") else -1
        flag = "" if src == dst else "   <-- MISMATCH"
        ok = ok and src == dst
        print(f"{index:40} {src:>10} {dst:>10}{flag}")
    return ok


# --- CLI -----------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="http://localhost:9200", help="Studio cluster URL (backend creds)")
    parser.add_argument("--dest", default="http://localhost:9200", help="MBP cluster URL (elastic bootstrap creds)")
    parser.add_argument("--snapshot", action="store_true", help="[Studio] snapshot every non-system index into the repo")
    parser.add_argument("--wait", action="store_true", help="[MBP] wait for the destination cluster to come up")
    parser.add_argument("--restore-snapshot", metavar="NAME", help="[MBP] restore NAME or 'latest' from the repo")
    parser.add_argument("--setup-users", action="store_true", help="[MBP] create the backend's ES user")
    parser.add_argument("--kibana-token", action="store_true", help="[MBP] mint a Kibana service token into .env")
    parser.add_argument("--mirror", action="store_true", help="[fallback] reindex-from-remote source -> dest")
    parser.add_argument("--verify", action="store_true", help="compare doc counts source vs dest")
    parser.add_argument("--indexes", nargs="+", help="indexes for --mirror/--verify (default: all non-system)")
    parser.add_argument("--exclude", nargs="+", default=[], help="index names to skip when discovering")
    parser.add_argument("--batch-size", type=int, default=200, help="reindex scroll batch size")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point.

    :param argv: Arguments (defaults to ``sys.argv[1:]``).
    :return: Process exit code.
    """
    args = build_parser().parse_args(argv)
    backend_env = read_env(REPO / "src" / "backend" / ".env")
    root_env = read_env(REPO / ".env")
    source = EsClient(args.source, backend_env["ELASTICSEARCH_USER"], backend_env["ELASTICSEARCH_PASSWORD"])
    dest = EsClient(args.dest, "elastic", root_env["LOCAL_ELASTIC_PASSWORD"])

    if args.snapshot:
        create_snapshot(source)
    if args.wait:
        wait_for_cluster(dest)
    if args.restore_snapshot:
        restore_snapshot(dest, args.restore_snapshot)
    if args.setup_users:
        ensure_user(dest, backend_env["ELASTICSEARCH_USER"], backend_env["ELASTICSEARCH_PASSWORD"], ["superuser"])
    if args.kibana_token:
        create_kibana_token(dest, REPO / ".env")
    if args.mirror or args.verify:
        indexes = args.indexes or discover_indexes(source, args.exclude)
        if args.mirror:
            print(f"mirroring {len(indexes)} indexes: {', '.join(indexes)}")
            for index in indexes:
                reindex_from_remote(source, dest, index, args.batch_size)
        if args.verify and not verify(source, dest, indexes):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
