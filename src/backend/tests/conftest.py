"""Shared setup of the backend tests.

``search_service``, ``search_bibliography`` and ``missing_fragments`` build
Elasticsearch clients at import time with ``ELASTICSEARCH_PASSWORD``.  That
value comes from ``src/backend/.env``, which exists only in the main checkout
(not in git worktrees or CI).  The file is loaded here first, exactly as those
modules would load it.

Without a password, a placeholder lets the modules import, and the cluster
address is pinned to a closed local port (``127.0.0.1:9``).  Most tests use
fakes, but some suites (``test_fetch_primary_sources``, ``test_agentic_rag``,
``test_fetch_secondary_sources``, ``test_primary_source_fetching``) build real
clients; with the placeholder they fail fast on this machine instead of sending
requests to the production cluster.
"""

import os
from pathlib import Path

import dotenv

PLACEHOLDER_PASSWORD = "unused-placeholder"
# A port nothing listens on: connections are refused at once, nothing leaves the host.
CLOSED_LOCAL_HOST = "127.0.0.1"
CLOSED_LOCAL_PORT = "9"

dotenv.load_dotenv(Path(__file__).resolve().parents[1] / ".env")
if "ELASTICSEARCH_PASSWORD" not in os.environ:
    os.environ["ELASTICSEARCH_PASSWORD"] = PLACEHOLDER_PASSWORD
    # Overrides a host exported in the shell too: the placeholder is never sent anywhere.
    os.environ["ELASTICSEARCH_HOST"] = CLOSED_LOCAL_HOST
    os.environ["ELASTICSEARCH_PORT"] = CLOSED_LOCAL_PORT
