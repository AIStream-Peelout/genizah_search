#!/usr/bin/env python3
"""Mirror the serving Elasticsearch indexes from the Mac Studio onto the MBP.

Uses reindex-from-remote, which needs no downtime on the source and does not
touch its data. The destination cluster must whitelist the source
(``reindex.remote.whitelist`` in docker-compose.mbp.yml).

Standard library only, so it runs on a fresh macOS python3 with no venv.

Credentials are read from the repo's env files and never printed:

* source: ``ELASTICSEARCH_USER`` / ``ELASTICSEARCH_PASSWORD`` in ``src/backend/.env``
  (the backend's superuser, which also works on the Studio's local container)
* destination: ``elastic`` / ``LOCAL_ELASTIC_PASSWORD`` in the root ``.env``
  (the bootstrap password set by docker-compose.mbp.yml)

Examples::

    mirror_es_indexes.py --source http://192.168.8.120:9200 --dest http://localhost:9200 --wait --setup-users
    mirror_es_indexes.py --source ... --dest ... --mirror --verify          # every index
    mirror_es_indexes.py --source ... --dest ... --mirror --indexes genizah_merged_v5
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

# By default EVERY non-system index on the source is mirrored. These serving
# indexes (see SHARED_RUNTIME.md, docs/ai_transcription_search.md) go first so
# the site can be verified while the older versions are still streaming.
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
    ) -> Dict[str, Any]:
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


def discover_indexes(source: EsClient, exclude: List[str]) -> List[str]:
    """List every non-system index on the source, serving indexes first.

    :param source: Source cluster client.
    :param exclude: Index names to leave out.
    :return: Ordered index names (``PRIORITY_INDEXES`` first, the rest alphabetical).
    """
    rows = source.request("GET", "/_cat/indices", params={"h": "index", "format": "json"})
    names = {row["index"] for row in rows if not row["index"].startswith(".")} - set(exclude)
    ordered = [name for name in PRIORITY_INDEXES if name in names]
    return ordered + sorted(names - set(ordered))


def wait_for_cluster(client: EsClient, timeout: int = 300) -> None:
    """Block until the cluster answers ``/_cluster/health`` with yellow or green.

    :param client: Destination cluster client.
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


def ensure_user(client: EsClient, name: str, password: str, roles: List[str]) -> None:
    """Create or update a native-realm user (idempotent).

    :param client: Cluster client with security privileges.
    :param name: User name.
    :param password: Password to set.
    :param roles: Role names to grant.
    """
    client.request("PUT", f"/_security/user/{name}", {"password": password, "roles": roles})
    print(f"user {name} ready with roles {roles}")


def portable_definition(source: EsClient, index: str) -> Dict[str, Any]:
    """Fetch an index's settings + mappings with cluster-specific keys removed.

    :param source: Source cluster client.
    :param index: Index name.
    :return: Body suitable for ``PUT /<index>`` on the destination.
    """
    raw = source.request("GET", f"/{index}")[index]
    index_settings = {k: v for k, v in raw["settings"]["index"].items() if k not in NON_PORTABLE_SETTINGS}
    index_settings["number_of_replicas"] = "0"  # single node on the MBP
    return {"settings": {"index": index_settings}, "mappings": raw["mappings"]}


def reindex_from_remote(source: EsClient, dest: EsClient, index: str, batch_size: int) -> None:
    """Create ``index`` on the destination if needed and pull its documents from the source.

    Skips indexes that already hold the full document count so the script can be
    re-run after an interruption.

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
        done = status.get("completed", False)
        progress = status["task"]["status"]
        print(f"  {index}: {progress.get('created', 0)}/{progress.get('total', expected)}", end="\r", flush=True)
        if done:
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


def create_kibana_token(dest: EsClient, env_path: Path) -> None:
    """Mint a Kibana service-account token on the destination and store it in the root .env.

    :param dest: Destination cluster client.
    :param env_path: Root ``.env`` whose ``ELASTICSEARCH_SERVICEACCOUNTTOKEN`` line is replaced.
    """
    name = "mbp"
    try:
        dest.request("DELETE", f"/_security/service/elastic/kibana/credential/token/{name}")
    except urllib.error.HTTPError as err:
        if err.code != 404:
            raise
    token = dest.request("POST", f"/_security/service/elastic/kibana/credential/token/{name}")["token"]["value"]
    update_env_var(env_path, "ELASTICSEARCH_SERVICEACCOUNTTOKEN", token)
    print(f"kibana service token '{name}' written to {env_path}")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="source cluster URL, e.g. http://192.168.8.120:9200")
    parser.add_argument("--dest", required=True, help="destination cluster URL, e.g. http://localhost:9200")
    parser.add_argument("--indexes", nargs="+", help="indexes to mirror/verify (default: every non-system index on the source)")
    parser.add_argument("--exclude", nargs="+", default=[], help="index names to skip when discovering")
    parser.add_argument("--batch-size", type=int, default=200, help="reindex scroll batch size")
    parser.add_argument("--wait", action="store_true", help="wait for the destination cluster to come up")
    parser.add_argument("--setup-users", action="store_true", help="create the backend's ES user on the destination")
    parser.add_argument("--mirror", action="store_true", help="reindex the indexes from source to destination")
    parser.add_argument("--verify", action="store_true", help="compare doc counts")
    parser.add_argument("--kibana-token", action="store_true", help="mint a Kibana service token and store it in .env")
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

    if args.wait:
        wait_for_cluster(dest)
    if args.setup_users:
        ensure_user(dest, backend_env["ELASTICSEARCH_USER"], backend_env["ELASTICSEARCH_PASSWORD"], ["superuser"])
    indexes = args.indexes or discover_indexes(source, args.exclude)
    if args.mirror:
        print(f"mirroring {len(indexes)} indexes: {', '.join(indexes)}")
        for index in indexes:
            reindex_from_remote(source, dest, index, args.batch_size)
    if args.verify and not verify(source, dest, indexes):
        return 1
    if args.kibana_token:
        create_kibana_token(dest, REPO / ".env")
    return 0


if __name__ == "__main__":
    sys.exit(main())
